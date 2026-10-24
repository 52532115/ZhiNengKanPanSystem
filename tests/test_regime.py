import math
import random
import unittest

import regime


def make_index(closes, start="2020-01-01"):
    """把收盘价序列包成日线行（日期用序号，便于断言）。"""
    rows = []
    for index, close in enumerate(closes):
        rows.append({
            "date": f"{start[:4]}-{index // 250 + 1:02d}-{index % 28 + 1:02d}-{index:04d}",
            "close": float(close),
        })
    return rows


def trending(count=600, start=10.0, drift=0.001, noise=0.002, seed=3):
    rng = random.Random(seed)
    closes = [start]
    for _ in range(count - 1):
        closes.append(closes[-1] * (1 + drift + rng.gauss(0, noise)))
    return closes


class BasicTest(unittest.TestCase):
    def test_realized_vol_is_causal_and_needs_history(self):
        closes = trending(80)
        self.assertIsNone(regime.realized_vol(closes, 5, 20))
        value = regime.realized_vol(closes, 40, 20)
        self.assertIsNotNone(value)
        self.assertGreater(value, 0)
        # 改动未来数据不能改变历史波动
        mutated = list(closes)
        mutated[-1] *= 1.5
        self.assertAlmostEqual(regime.realized_vol(mutated, 40, 20), value, places=12)

    def test_percentile_rank_handles_ties(self):
        series = [1.0, 2.0, 2.0, 3.0]
        self.assertAlmostEqual(regime.percentile_rank(series, 2.0), 0.5)
        self.assertAlmostEqual(regime.percentile_rank(series, 3.0), 0.875)
        self.assertEqual(regime.percentile_rank([], 1.0), 0.5)

    def test_config_rejects_bad_weights(self):
        with self.assertRaises(ValueError):
            regime.RegimeConfig(trend_weight=0.5, vol_weight=0.5, breadth_weight=0.5)
        with self.assertRaises(ValueError):
            regime.RegimeConfig(risk_off_exposure=0.0)
        with self.assertRaises(ValueError):
            regime.RegimeConfig(confirm_days=0)


class ClassifyTest(unittest.TestCase):
    def test_warmup_has_no_state(self):
        closes = trending(300)
        points = regime.classify(make_index(closes), {})
        early = points[:regime.MIN_HISTORY]
        self.assertTrue(all(point.state == "" for point in early))
        self.assertTrue(all(point.exposure == 1.0 for point in early))
        self.assertTrue(any(point.state for point in points[regime.MIN_HISTORY + 30:]))

    def test_downtrend_flips_to_risk_off(self):
        up = trending(320, drift=0.002, noise=0.001)
        down = [up[-1] * (1 - 0.004) ** i for i in range(1, 200)]
        points = regime.classify(make_index(up + down), {})
        last = points[-1]
        self.assertEqual(last.state, "risk_off")
        self.assertAlmostEqual(last.exposure, 0.5)

    def test_uptrend_stays_risk_on(self):
        points = regime.classify(make_index(trending(400, drift=0.003, noise=0.0008)), {})
        states = {point.state for point in points[regime.MIN_HISTORY + 40:]}
        self.assertEqual(states, {"risk_on"})

    def test_hysteresis_increases_divergence_days(self):
        # 确认期越长，"原始判断已翻面、确认状态还没翻"的天数就越多——这是滞后的直接度量
        closes = trending(700, drift=0.0, noise=0.006, seed=5)
        points_fast = regime.classify(make_index(closes), {}, regime.RegimeConfig(confirm_days=1))
        points_slow = regime.classify(make_index(closes), {}, regime.RegimeConfig(confirm_days=10))
        lag_fast = sum(1 for p in points_fast if p.state and p.raw_state != p.state)
        lag_slow = sum(1 for p in points_slow if p.state and p.raw_state != p.state)
        self.assertGreater(lag_slow, lag_fast)
        self.assertGreaterEqual(lag_slow, 5)

    def test_state_never_flips_without_confirmation_window(self):
        closes = trending(400, drift=0.0, noise=0.004, seed=11)
        points = regime.classify(make_index(closes), {}, regime.RegimeConfig(confirm_days=5))
        for index, point in enumerate(points):
            if point.flipped:
                # 切换当天 raw 必须与 state 一致
                self.assertEqual(point.raw_state, point.state)
                # 之前 4 天 raw 必须一直是新状态
                window = points[max(0, index - 4): index]
                self.assertTrue(all(item.raw_state == point.state for item in window))

    def test_exposure_for_open_shifts_one_day(self):
        closes = trending(320, drift=0.002, noise=0.001) + [
            (trending(320, drift=0.002, noise=0.001)[-1]) * (1 - 0.004) ** i for i in range(1, 120)
        ]
        points = regime.classify(make_index(closes), {})
        mapping = regime.exposure_for_open(points)
        for index in range(1, len(points)):
            self.assertEqual(mapping[points[index].date], points[index - 1].exposure)

    def test_stop_leg_fires_faster_than_slow_leg(self):
        up = trending(320, drift=0.002, noise=0.001)
        crash = [up[-1] * (1 - 0.02) ** i for i in range(1, 30)]
        closes = up + crash
        index_rows = make_index(closes)
        plain = regime.classify(index_rows, {})
        guarded = regime.classify(index_rows, {}, regime.RegimeConfig(stop_drawdown=0.10))
        first_plain = next((p.date for p in plain if p.state == "risk_off"), None)
        first_guarded = next((p.date for p in guarded if p.state == "risk_off"), None)
        self.assertIsNotNone(first_guarded)
        if first_plain is not None:
            self.assertLess(first_guarded, first_plain)
        self.assertTrue(any(point.stop for point in guarded))

    def test_stop_leg_does_not_block_reentry(self):
        closes = trending(320, drift=0.002, noise=0.001)
        slump = [closes[-1] * (1 - 0.02) ** i for i in range(1, 25)]
        recovery = [slump[-1] * (1 + 0.01) ** i for i in range(1, 120)]
        points = regime.classify(
            make_index(closes + slump + recovery), {}, regime.RegimeConfig(stop_drawdown=0.10)
        )
        self.assertEqual(points[-1].state, "risk_on")


class SwitchLogTest(unittest.TestCase):
    def test_lag_days_matches_gap(self):
        closes = trending(320, drift=0.002, noise=0.001) + [
            trending(320, drift=0.002, noise=0.001)[-1] * (1 - 0.004) ** i for i in range(1, 120)
        ]
        points = regime.classify(make_index(closes), {})
        events = regime.switch_log(points)
        self.assertTrue(events)
        dates = [point.date for point in points]
        for event in events:
            gap = dates.index(event["confirm_date"]) - dates.index(event["raw_date"])
            self.assertEqual(event["lag_days"], gap)
            self.assertGreaterEqual(event["lag_days"], 0)

    def test_no_events_on_smooth_uptrend(self):
        points = regime.classify(make_index(trending(400, drift=0.003, noise=0.0008)), {})
        self.assertEqual(regime.switch_log(points), [])


class TiltTest(unittest.TestCase):
    def test_risk_on_returns_same_weights(self):
        weights = {"mom12_1": 0.010, "ivol": 0.047, "rev5": 0.020, "turnover20": 0.10}
        self.assertEqual(regime.tilt_weights(weights, state="risk_on"), weights)

    def test_risk_off_damps_momentum_and_preserves_l1(self):
        weights = {"mom12_1": 0.010, "ivol": 0.047, "rev5": 0.020, "turnover20": 0.10}
        tilted = regime.tilt_weights(weights, state="risk_off")
        self.assertLess(tilted["mom12_1"], weights["mom12_1"])
        self.assertGreater(tilted["ivol"], weights["ivol"])
        self.assertAlmostEqual(
            sum(abs(v) for v in tilted.values()),
            sum(abs(v) for v in weights.values()),
            places=12,
        )


class ForwardStatsTest(unittest.TestCase):
    def test_separates_forward_vol(self):
        # 构造：前段平稳（risk_on）、后段剧烈震荡（risk_off 应在这里）
        calm = [10.0 * (1 + 0.001 * i) for i in range(400)]
        wild = []
        level = calm[-1]
        for index in range(200):
            level *= 1 + (0.03 if index % 2 == 0 else -0.028)
            wild.append(level)
        closes = calm + wild
        points = regime.classify(make_index(closes), {})
        stats = regime.forward_stats(points, closes, horizon=10)
        if stats.get("risk_off", {}).get("n") and stats.get("risk_on", {}).get("n"):
            self.assertGreater(stats["risk_off"]["vol"], stats["risk_on"]["vol"])


class YearTest(unittest.TestCase):
    def test_state_by_year_counts(self):
        points = regime.classify(make_index(trending(400, drift=0.003, noise=0.0008)), {})
        years = regime.state_by_year(points)
        self.assertTrue(years)
        total = sum(bucket["days"] for bucket in years.values())
        self.assertEqual(total, sum(1 for point in points if point.state))
        for bucket in years.values():
            self.assertLessEqual(bucket["risk_off"], bucket["days"])
            self.assertTrue(0.0 <= bucket["risk_off_share"] <= 100.0)


if __name__ == "__main__":
    unittest.main()
