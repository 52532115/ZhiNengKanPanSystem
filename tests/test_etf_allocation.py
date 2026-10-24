import math
import random
import statistics
import unittest

import etf_allocation as E


def series(values):
    return [float(v) for v in values]


class SignalTest(unittest.TestCase):
    def test_sma_handles_missing_values(self):
        values = [1.0, 2.0, None, 4.0, 5.0]
        out = E.sma(values, 2)
        self.assertIsNone(out[0])
        self.assertAlmostEqual(out[1], 1.5)
        self.assertIsNone(out[2])          # 窗口含缺值 → 不给值
        self.assertIsNone(out[3])          # 窗口 [None, 4] 只有一个有效值，仍不给
        self.assertAlmostEqual(out[4], 4.5)

    def test_momentum_skips_recent_month(self):
        closes = [10.0 * (1.01 ** i) for i in range(300)]
        value = E.momentum(closes, 299)
        expected = closes[299 - 21] / closes[299 - 252] - 1
        self.assertAlmostEqual(value, expected, places=12)
        self.assertIsNone(E.momentum(closes, 100))   # 预热不足

    def test_drawdown_from_high(self):
        closes = [10.0] * 30 + [8.0]
        self.assertAlmostEqual(E.drawdown_from_high(closes, 30, 252), -0.2, places=9)


class EligibilityTest(unittest.TestCase):
    def _setup(self, uptrend=True, crash=False):
        if uptrend:
            closes = [10.0 * (1.002 ** i) for i in range(400)]
        else:
            closes = [10.0 * (0.998 ** i) for i in range(400)]
        if crash:
            closes = [10.0] * 260 + [10.0 * (0.97 ** i) for i in range(1, 141)]
        config = E.Config()
        fast = E.sma(closes, config.ma_fast)
        slow = E.sma(closes, config.ma_slow)
        return closes, config, fast, slow

    def test_uptrend_enters(self):
        closes, config, fast, slow = self._setup()
        ok, info = E.eligibility(closes, len(closes) - 1, config=config,
                                 ma_fast=fast, ma_slow=slow, holding=False)
        self.assertTrue(ok)
        self.assertGreater(info["mom"], 0)

    def test_downtrend_blocked(self):
        closes, config, fast, slow = self._setup(uptrend=False)
        ok, info = E.eligibility(closes, len(closes) - 1, config=config,
                                 ma_fast=fast, ma_slow=slow, holding=False)
        self.assertFalse(ok)

    def test_stop_triggers_before_trend_breaks(self):
        closes, config, fast, slow = self._setup(crash=True)
        ok, info = E.eligibility(closes, len(closes) - 1, config=config,
                                 ma_fast=fast, ma_slow=slow, holding=True)
        self.assertFalse(ok)
        self.assertEqual(info["reason"], "stop")
        self.assertLess(info["drawdown"], config.stop_drawdown)

    def test_holding_ignores_fast_ma(self):
        """持有只看 MA120：找到"跌破 MA60 但仍在 MA120 上方"的点，验证两套规则分叉。"""
        closes = [8.0 * (1.0015 ** i) for i in range(320)]
        level = closes[-1]
        for _ in range(12):
            level *= 0.992
            closes.append(level)
        config = E.Config()
        fast = E.sma(closes, config.ma_fast)
        slow = E.sma(closes, config.ma_slow)
        found = None
        for index in range(260, len(closes)):
            if fast[index] is None or slow[index] is None:
                continue
            if closes[index] < fast[index] and closes[index] > slow[index]:
                found = index
                break
        self.assertIsNotNone(found, "构造的序列里没有出现『破 MA60、仍在 MA120 上方』的样本")
        ok_hold, info = E.eligibility(closes, found, config=config, ma_fast=fast,
                                      ma_slow=slow, holding=True)
        ok_entry, _ = E.eligibility(closes, found, config=config, ma_fast=fast,
                                    ma_slow=slow, holding=False)
        self.assertFalse(info["above_fast"])
        self.assertTrue(info["above_slow"])
        self.assertTrue(ok_hold)       # 持有只看慢线
        self.assertFalse(ok_entry)     # 入场还要求站上快线


class WeightTest(unittest.TestCase):
    def _cov(self, size=4, seed=5):
        rng = random.Random(seed)
        matrix = [[0.0] * size for _ in range(size)]
        for i in range(size):
            for j in range(i, size):
                value = 0.02 + 0.05 * rng.random() if i == j else 0.01
                matrix[i][j] = matrix[j][i] = value
        return matrix

    def test_erc_equalises_risk_contribution(self):
        cov = self._cov()
        weights = E.erc_weights(cov)
        self.assertAlmostEqual(sum(weights), 1.0, places=9)
        marginal = [sum(cov[i][j] * weights[j] for j in range(len(weights))) for i in range(len(weights))]
        contribution = [weights[i] * marginal[i] for i in range(len(weights))]
        self.assertLess(max(contribution) / min(contribution), 1.02)
        self.assertTrue(all(w > 0 for w in weights))

    def test_inverse_vol_favours_low_vol(self):
        cov = [[0.04, 0.0, 0.0], [0.0, 0.09, 0.0], [0.0, 0.0, 0.16]]
        weights = E.inverse_vol_weights(cov)
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])
        self.assertAlmostEqual(sum(weights), 1.0, places=12)

    def test_portfolio_vol_matches_formula(self):
        cov = [[0.04, 0.01], [0.01, 0.09]]
        weights = [0.5, 0.5]
        expected = math.sqrt(0.25 * 0.04 + 0.25 * 0.09 + 2 * 0.25 * 0.01)
        self.assertAlmostEqual(E.portfolio_vol(weights, cov), expected, places=12)

    def test_limits_cap_single_and_bucket(self):
        weights = {"510300": 0.5, "510500": 0.3, "511010": 0.2}
        config = E.Config()
        capped = E.apply_limits(weights, config=config, max_weight=0.3)
        self.assertLessEqual(max(capped.values()), 0.3 + 1e-9)
        a_share = sum(v for s, v in capped.items() if E.UNIVERSE[s]["bucket"] == "a_share")
        self.assertLessEqual(a_share, 0.50 + 1e-9)


class PeriodTest(unittest.TestCase):
    def test_window_metrics_computes_return_and_drawdown(self):
        values = [1.0] + [1.1] * 10 + [0.99] * 10 + [1.2] * 4
        nav = [{"date": f"2020-01-{i + 1:02d}", "nav": value} for i, value in enumerate(values)]
        out = E.window_metrics(nav, "2020-01-01", "2020-01-31")
        self.assertAlmostEqual(out["return"], 20.0, places=6)
        self.assertAlmostEqual(out["max_drawdown"], -10.0, places=6)

    def test_short_period_labels(self):
        self.assertEqual(E._short_period("2018 熊市"), "2018")
        self.assertEqual(E._short_period("2019–2021 结构牛"), "19–21")


class ConfigTest(unittest.TestCase):
    def test_bucket_caps_reference_known_buckets(self):
        buckets = {info["bucket"] for info in E.UNIVERSE.values()}
        for bucket, cap in E.Config().bucket_caps:
            self.assertIn(bucket, buckets)
            self.assertTrue(0 < cap <= 1)

    def test_universe_has_twelve_legs_with_alternatives(self):
        self.assertEqual(len(E.UNIVERSE), 12)
        for symbol, info in E.UNIVERSE.items():
            self.assertTrue(info.get("note"), symbol)


if __name__ == "__main__":
    unittest.main()
