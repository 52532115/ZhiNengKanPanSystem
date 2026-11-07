"""因子库的契约测试：因果性、分位与正交化语义、市值反推恒等式。"""

import math
import random
import unittest

import factors as fa


def make_history(count=320, drift=0.0004, wave=0.01, seed=0, symbol="600000"):
    random.seed(seed)
    rows = []
    price = 10.0
    for index in range(count):
        price = max(1.0, price * (1 + drift + wave * math.sin(index / 7) + random.gauss(0, 0.004)))
        volume = 1_000_000 * (1 + 0.3 * math.sin(index / 11))
        rows.append({
            "date": f"{2015 + index // 250}-{index % 12 + 1:02d}-{index % 28 + 1:02d}",
            "open": price * 1.002, "high": price * 1.01, "low": price * 0.99,
            "close": price, "volume": volume, "amount": volume * price,
            "turnover": 1.5, "pct_change": 0.0,
        })
    return {"symbol": symbol, "name": "测试", "asset_type": "stock", "rows": rows}


class SpecTests(unittest.TestCase):
    def test_every_factor_declares_family_prior_and_logic(self):
        for key, spec in fa.FACTOR_SPECS.items():
            self.assertIn(spec["family"], fa.FAMILY_SPECS, key)
            self.assertIn(spec["prior"], (1, -1), key)
            self.assertTrue(spec["formula"], key)
            self.assertTrue(spec["note"], key)
        covered = {spec["family"] for spec in fa.FACTOR_SPECS.values()}
        self.assertEqual(covered, set(fa.FAMILY_SPECS))

    def test_existing_factors_are_not_duplicated_as_new_ones(self):
        for key in fa.EXISTING_FACTORS:
            self.assertNotIn(key, fa.FACTOR_SPECS)


class RankTests(unittest.TestCase):
    def test_rank_pct_bounds_and_ties(self):
        values = [3.0, 1.0, 2.0, 2.0]
        ranks = fa.rank_pct(values)
        # 中位秩口径：100×(平均名次+0.5)/n，取值落在 (0, 100]，并列取同一值
        self.assertEqual(ranks[1], 12.5)          # 最小
        self.assertEqual(ranks[0], 87.5)          # 最大
        self.assertEqual(ranks[2], ranks[3])      # 并列同分位
        self.assertTrue(all(0 < value <= 100 for value in ranks))

    def test_rank_pct_handles_constant_input(self):
        ranks = fa.rank_pct([5.0, 5.0, 5.0])
        self.assertEqual(len(set(ranks)), 1)


class OrthogonalizeTests(unittest.TestCase):
    def test_residual_is_uncorrelated_with_controls(self):
        random.seed(7)
        n = 200
        control = [random.gauss(0, 1) for _ in range(n)]
        second = [random.gauss(0, 1) for _ in range(n)]
        target = [3.0 * a - 2.0 * b + random.gauss(0, 0.3) for a, b in zip(control, second)]
        residual = fa.orthogonalize(target, [control, second])
        for column in (control, second):
            mean_r = sum(residual) / n
            mean_c = sum(column) / n
            cov = sum((r - mean_r) * (c - mean_c) for r, c in zip(residual, column))
            var_r = sum((r - mean_r) ** 2 for r in residual) ** 0.5
            var_c = sum((c - mean_c) ** 2 for c in column) ** 0.5
            self.assertLess(abs(cov) / (var_r * var_c), 1e-6)

    def test_orthogonalize_without_controls_just_demeans(self):
        residual = fa.orthogonalize([1.0, 2.0, 3.0], [])
        self.assertAlmostEqual(sum(residual), 0.0, places=9)


class RegressionTests(unittest.TestCase):
    def test_rolling_regression_recovers_slope(self):
        random.seed(3)
        x = [random.gauss(0, 0.01) for _ in range(300)]
        y = [1.5 * value + random.gauss(0, 0.002) for value in x]
        slope, r2, resid = fa._rolling_regression(y, x, 120)
        self.assertAlmostEqual(slope[-1], 1.5, delta=0.15)
        self.assertGreater(r2[-1], 0.7)
        self.assertGreater(resid[-1], 0)


class ComputeTests(unittest.TestCase):
    def test_series_are_causal(self):
        history = make_history(320)
        market = {row["date"]: 0.0 for row in history["rows"]}
        full = fa.compute(history, market)
        prefix = {**history, "rows": history["rows"][:-30]}
        short = fa.compute(prefix, market)
        for key in fa.FACTOR_SPECS:
            self.assertEqual(
                [v for v in full[key][:-30]],
                [v for v in short[key]],
                f"{key} 用到了未来数据",
            )

    def test_factor_ranges(self):
        history = make_history(400)
        market = {row["date"]: 0.001 * math.sin(i / 5) for i, row in enumerate(history["rows"])}
        values = fa.compute(history, market)
        clv = [v for v in values["clv20"] if v is not None]
        self.assertTrue(all(0 <= v <= 1 for v in clv))
        stack = [v for v in values["ma_stack"] if v is not None]
        self.assertTrue(all(0 <= v <= 3 for v in stack))
        high52 = [v for v in values["high52"] if v is not None]
        self.assertTrue(all(0 < v for v in high52))

    def test_float_cap_matches_amount_over_turnover(self):
        history = make_history(10)
        caps = fa.float_cap(history)
        row = history["rows"][0]
        expected = row["amount"] / (row["turnover"] / 100.0) / 1e8
        self.assertAlmostEqual(caps[0], expected, places=6)

    def test_float_cap_is_none_when_suspended(self):
        history = make_history(3)
        history["rows"][1]["turnover"] = 0.0
        self.assertIsNone(fa.float_cap(history)[1])

    def test_existing_factors_replicate_reference_formulas(self):
        history = make_history(300, drift=0.001, wave=0.002)
        bench = {"ret20": {row["date"]: 0.5 for row in history["rows"]},
                 "ret1": {row["date"]: 0.0 for row in history["rows"]}}
        series = fa.existing(history, bench)
        self.assertEqual(set(series), set(fa.EXISTING_FACTORS))
        volumes = [row["volume"] for row in history["rows"]]
        ratio = series["vol_ratio"][-1]
        self.assertAlmostEqual(ratio, volumes[-1] / (sum(volumes[-20:]) / 20), places=6)


if __name__ == "__main__":
    unittest.main()
