import math
import random
import statistics
import unittest

import scoring


class TransformTest(unittest.TestCase):
    def test_tanh_is_bounded_monotone_and_odd(self):
        values = [scoring.tanh_score(z) for z in (-100, -2, -1, 0, 1, 2, 100)]
        self.assertTrue(all(-1.0 <= v <= 1.0 for v in values))
        self.assertLess(abs(values[0] + 1.0), 1e-9)   # 饱和到 −1 但不越界
        self.assertLess(abs(values[-1] - 1.0), 1e-9)
        self.assertAlmostEqual(values[3], 0.0, places=12)
        self.assertEqual(values, sorted(values))
        self.assertAlmostEqual(values[1], -values[5], places=12)
        self.assertAlmostEqual(values[3], 0.0, places=12)
        self.assertEqual(values, sorted(values))
        self.assertAlmostEqual(values[1], -values[5], places=12)

    def test_tanh_matches_definition(self):
        self.assertAlmostEqual(scoring.tanh_score(1.25), math.tanh(1.0), places=12)

    def test_piecewise_linear_hits_knots_and_clamps(self):
        knots = [0.0, 1.0, 2.0]
        scores = [-1.0, 0.0, 0.8]
        self.assertEqual(scoring.piecewise_score(-3, knots, scores), -1.0)
        self.assertEqual(scoring.piecewise_score(1.0, knots, scores), 0.0)
        self.assertAlmostEqual(scoring.piecewise_score(1.5, knots, scores), 0.4, places=12)
        self.assertEqual(scoring.piecewise_score(9, knots, scores), 0.8)

    def test_piecewise_rejects_non_monotone_scores(self):
        with self.assertRaises(ValueError):
            scoring.piecewise_score(0.5, [0, 1], [0.5, -0.5])
        with self.assertRaises(ValueError):
            scoring.piecewise_score(0.5, [1, 0], [0.5, 1.0])

    def test_factor_score_applies_prior_and_handles_none(self):
        self.assertAlmostEqual(scoring.factor_score(2.5, scale=1.0), scoring.tanh_score(2.5))
        self.assertAlmostEqual(scoring.factor_score(2.5, scale=1.0, prior=-1), -scoring.tanh_score(2.5))
        self.assertIsNone(scoring.factor_score(None))
        self.assertIsNone(scoring.factor_score(float("nan")))

    def test_robust_center_scale_ignores_single_outlier(self):
        values = [1.0] * 40 + [1000.0]
        center, scale = scoring.robust_center_scale(values)
        self.assertAlmostEqual(center, 1.0, places=6)
        self.assertLessEqual(scale, 1.0)

    def test_robust_z_is_winsorized(self):
        values = [0.0, 1.0, 2.0, 3.0, 1000.0]
        z = scoring.robust_z(values)
        self.assertLessEqual(max(z), scoring.CLIP_MAD + 1e-9)
        self.assertGreaterEqual(min(z), -scoring.CLIP_MAD - 1e-9)

    def test_quantile_squash_is_monotone_in_value(self):
        sample = [1.0, 2.0, 3.0, 4.0]
        values = [scoring.quantile_squash(v, sample) for v in (0, 1.5, 2.5, 5)]
        self.assertEqual(values, sorted(values))
        self.assertLess(values[0], 0)
        self.assertGreater(values[-1], 0)

    def test_quantile_squash_series_is_causal(self):
        series = [float(i % 17) for i in range(300)]
        out = scoring.quantile_squash_series(series, window=120, min_obs=100)
        self.assertTrue(all(v is None for v in out[:100]))
        # 改掉最后一点不应影响它之前的输出
        mutated = list(series)
        mutated[-1] = 999.0
        expected = scoring.quantile_squash_series(mutated, window=120, min_obs=100)
        self.assertEqual(out[:-1], expected[:-1])

    def test_expanding_scores_respects_prior(self):
        series = [float(i % 13) for i in range(200)]
        up = scoring.expanding_scores(series, min_obs=50, prior=1)
        down = scoring.expanding_scores(series, min_obs=50, prior=-1)
        for a, b in zip(up[60:], down[60:]):
            if a is not None and b is not None:
                self.assertAlmostEqual(a, -b, places=12)


class StandardizeTest(unittest.TestCase):
    def test_rank01_matches_positions_with_ties(self):
        ranks = scoring.rank01([3.0, 1.0, 2.0, 2.0])
        self.assertAlmostEqual(ranks[0], 0.875)
        self.assertAlmostEqual(ranks[1], 0.125)
        self.assertAlmostEqual(ranks[2], 0.5)   # 并列取平均名次 (1.5+0.5)/4
        self.assertAlmostEqual(ranks[2], ranks[3])

    def test_industry_demean_removes_group_means(self):
        values = [1.0, 2.0, 3.0, 10.0, 11.0, 12.0]
        groups = ["A", "A", "A", "B", "B", "B"]
        out = scoring.industry_demean(values, groups)
        for group in ("A", "B"):
            members = [o for o, g in zip(out, groups) if g == group]
            self.assertAlmostEqual(statistics.fmean(members), 0.0, places=12)

    def test_small_industry_falls_back_to_full_mean(self):
        values = [0.0, 10.0, 20.0, 100.0]
        groups = ["A", "A", "A", "Z"]  # Z 只有 1 只
        out = scoring.industry_demean(values, groups, min_group=3)
        self.assertAlmostEqual(out[3], 100.0 - statistics.fmean(values), places=12)

    def test_neutralize_removes_cap_exposure(self):
        caps = [math.log(v) for v in (10, 20, 30, 40, 50, 60, 70, 80)]
        values = [0.1 * c + ((i % 3) - 1) * 0.2 for i, c in enumerate(caps)]
        out = scoring.neutralize(values, log_cap=caps)
        mean_x = statistics.fmean(caps)
        mean_y = statistics.fmean(out)
        covariance = sum((c - mean_x) * (y - mean_y) for c, y in zip(caps, out))
        variance = sum((c - mean_x) ** 2 for c in caps)
        self.assertAlmostEqual(covariance / variance, 0.0, places=8)

    def test_standardize_outputs_are_bounded(self):
        random.seed(3)
        values = [random.gauss(0, 1) for _ in range(60)]
        caps = [float(i) for i in range(60)]
        for mode in ("z", "rank"):
            out = scoring.standardize(values, log_cap=caps, mode=mode)
            self.assertEqual(len(out), len(values))
            self.assertTrue(all(-1.0 <= v <= 1.0 for v in out))
            self.assertLess(abs(statistics.fmean(out)), 0.35)

    def test_standardize_rank_mode_is_scale_invariant(self):
        random.seed(4)
        values = [random.gauss(0, 1) for _ in range(40)]
        scaled = [v * 1000 + 7 for v in values]
        a = scoring.standardize(values, mode="rank")
        b = scoring.standardize(scaled, mode="rank")
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, places=12)


class WeightTest(unittest.TestCase):
    def _design(self, signal=0.05, n_days=200, n_stocks=30, noise=1.0, seed=5):
        rng = random.Random(seed)
        rows_by_date = {}
        for day in range(n_days):
            date = f"2020-{day // 28 + 1:02d}-{day % 28 + 1:02d}"
            rows = []
            for i in range(n_stocks):
                factor = rng.gauss(0, 1)
                record = {"date": date, "symbol": f"S{i:03d}", "label": rng.random()}
                record["s_a"] = factor
                record["s_b"] = rng.gauss(0, 1)
                record["label"] = 0.5 + signal * factor + rng.gauss(0, noise) * 0.1
                rows.append(record)
            rows_by_date[date] = rows
        return rows_by_date

    def test_ridge_shrinks_towards_prior(self):
        rows = self._design()
        design = scoring.build_design(rows, ["a", "b"], prefix="s_")
        weak = scoring.fit_ridge(
            design.blocks, prior_mean=[0.02, 0.02], prior_tau=0.001, features=["a", "b"]
        )
        loose = scoring.fit_ridge(
            design.blocks, prior_mean=[0.02, 0.02], prior_tau=10.0, features=["a", "b"]
        )
        self.assertGreater(abs(loose["coefficients"]["a"]), abs(weak["coefficients"]["a"]))
        # 强收缩时系数应贴住先验，而不是贴住最小二乘解
        self.assertLess(abs(weak["coefficients"]["a"] - 0.02), 0.05)

    def test_bootstrap_sign_rate_high_for_real_signal(self):
        rows = self._design(signal=0.30, noise=0.5)
        design = scoring.build_design(rows, ["a", "b"], prefix="s_")
        boot = scoring.bootstrap_fit(
            design, prior_mean=[0.02, 0.02], prior_tau=0.1, iters=40, block_days=5, seed=1
        )
        self.assertGreater(boot["sign_rate"]["a"], 0.9)
        self.assertLess(boot["sign_rate"]["b"], boot["sign_rate"]["a"])

    def test_stability_weights_zero_out_unstable_factor(self):
        fit = {"coefficients": {"good": 0.10, "noise": 0.08}}
        boot = {"sign_rate": {"good": 0.98, "noise": 0.55}}
        result = scoring.stability_weights(
            fit, boot, prior_sign={"good": 1, "noise": 1}, min_sign_rate=0.8
        )
        self.assertAlmostEqual(result["weights"]["good"], 0.10 * (2 * 0.98 - 1), places=6)
        self.assertNotIn("noise", result["weights"])
        self.assertEqual(len(result["dropped"]), 1)
        self.assertAlmostEqual(sum(result["share"].values()), 1.0, places=9)

    def test_stability_weights_rejects_prior_conflict(self):
        fit = {"coefficients": {"x": -0.05}}
        boot = {"sign_rate": {"x": 1.0}}
        result = scoring.stability_weights(fit, boot, prior_sign={"x": 1})
        self.assertEqual(result["weights"], {})


class CalibrationTest(unittest.TestCase):
    def test_pava_is_monotone_and_fits_without_weights(self):
        values = [3.0, 1.0, 2.0, 5.0, 4.0]
        fitted = scoring.pava(values)
        self.assertEqual(fitted, sorted(fitted))
        self.assertAlmostEqual(statistics.fmean(fitted), statistics.fmean(values), places=9)

    def test_pava_pools_inversions(self):
        fitted = scoring.pava([1.0, 2.0, 0.0])
        self.assertAlmostEqual(fitted[0], 1.0, places=9)  # (1+0)/2 与 2 合并后……
        self.assertTrue(all(fitted[i] <= fitted[i + 1] for i in range(2)))
        self.assertAlmostEqual(statistics.fmean(fitted), 1.0, places=9)

    def test_isotonic_recovers_monotone_mapping(self):
        rng = random.Random(9)
        scores = [rng.uniform(-1, 1) for _ in range(4000)]
        targets = [0.5 + 0.3 * s + rng.gauss(0, 0.05) for s in scores]
        calibrator = scoring.IsotonicCalibrator(bins=20).fit(scores, targets)
        self.assertTrue(calibrator.summary()["monotone"])
        self.assertLess(calibrator.predict(-1.0), calibrator.predict(1.0))
        self.assertAlmostEqual(calibrator.predict(0.0), 0.5, places=1)

    def test_calibration_metrics_perfect_diagonal(self):
        curve = [{"n": 100, "pred": i / 10, "real": i / 10} for i in range(1, 10)]
        metrics = scoring.calibration_metrics(curve)
        self.assertAlmostEqual(metrics["ece"], 0.0, places=9)
        self.assertAlmostEqual(metrics["slope"], 1.0, places=6)

    def test_calibration_metrics_detects_compression(self):
        curve = [{"n": 100, "pred": i / 10, "real": 0.5 + (i / 10 - 0.5) * 0.4} for i in range(11)]
        metrics = scoring.calibration_metrics(curve)
        self.assertAlmostEqual(metrics["slope"], 0.4, places=6)
        self.assertGreater(metrics["ece"], 0.05)

    def test_shrink_estimated_without_dates_is_one(self):
        rng = random.Random(21)
        scores = [rng.uniform(-1, 1) for _ in range(500)]
        targets = [0.5 + 0.1 * s for s in scores]
        cal = scoring.IsotonicCalibrator(bins=10).fit(scores, targets)
        self.assertEqual(cal.shrink, 1.0)
        self.assertFalse(cal.shrink_estimated)

    def test_cv_shrink_collapses_on_pure_noise(self):
        # 单个种子下 λ 的抽样误差很大（预测跨度越小、λ 越难估），所以这里看 6 个
        # 种子的均值：纯噪声下应当贴着 0，而不是某个具体抽样值。
        estimates = []
        for seed in range(6):
            rng = random.Random(seed)
            scores, targets, dates = [], [], []
            for day in range(600):
                for _ in range(20):
                    scores.append(rng.gauss(0, 1))
                    targets.append(rng.random())
                    dates.append(f"d{day:04d}")
            cal = scoring.IsotonicCalibrator(bins=20).fit(
                scores, targets, dates=dates, folds=4, purge=20
            )
            self.assertTrue(cal.shrink_estimated)
            estimates.append(cal.shrink)
        self.assertLess(statistics.fmean(estimates), 0.25)
        spread = max(cal.values) - min(cal.values)
        self.assertLess(spread, 0.10)

    def test_cv_shrink_keeps_real_signal(self):
        rng = random.Random(23)
        scores, targets, dates = [], [], []
        for day in range(600):
            for _ in range(20):
                value = rng.gauss(0, 1)
                scores.append(value)
                targets.append(min(0.99, max(0.01, 0.5 + 0.12 * value + rng.gauss(0, 0.02))))
                dates.append(f"d{day:04d}")
        cal = scoring.IsotonicCalibrator(bins=20).fit(
            scores, targets, dates=dates, folds=4, purge=20
        )
        self.assertGreater(cal.shrink, 0.7)
        self.assertLess(cal.predict(-2.0), cal.predict(2.0))

    def test_quantile_fit_recovers_slope(self):
        rng = random.Random(12)
        xs = [rng.uniform(-1, 1) for _ in range(800)]
        ys = [0.5 + 0.25 * x + rng.gauss(0, 0.1) for x in xs]
        result = scoring.quantile_fit(xs, ys, 0.5, iters=40)
        self.assertAlmostEqual(result["slope"], 0.25, delta=0.06)
        self.assertAlmostEqual(result["intercept"], 0.5, delta=0.03)


class CompositeTest(unittest.TestCase):
    def test_composite_renormalizes_missing_factors(self):
        scores = {"a": 1.0, "b": -0.5, "c": None}
        weights = {"a": 0.3, "b": 0.3, "c": 0.4}
        self.assertAlmostEqual(scoring.composite_score(scores, weights), 0.25, places=12)

    def test_composite_returns_none_when_empty(self):
        self.assertIsNone(scoring.composite_score({}, {"a": 1.0}))
        self.assertIsNone(scoring.composite_score({"a": None}, {"a": 1.0}))


class ValidationTest(unittest.TestCase):
    def test_daily_ic_detects_perfect_ranking(self):
        scores, forward = {}, {}
        for day in range(30):
            date = f"2021-01-{day + 1:02d}"
            scores[date] = {f"S{i}": float(i) for i in range(25)}
            forward[date] = {f"S{i}": float(i) for i in range(25)}
        series = scoring.daily_ic(scores, forward)
        self.assertEqual(len(series), 30)
        self.assertAlmostEqual(statistics.fmean(series), 1.0, places=6)

    def test_quintile_table_is_monotone_for_sorted_scores(self):
        scores, forward = {}, {}
        for day in range(40):
            date = f"2021-02-{day + 1:02d}"
            scores[date] = {f"S{i}": float(i) for i in range(40)}
            forward[date] = {f"S{i}": float(i) for i in range(40)}
        table = scoring.quintile_table(scores, forward)
        means = [row["mean"] for row in table]
        self.assertEqual(means, sorted(means))
        mono = scoring.quintile_monotonicity(table)
        self.assertEqual(mono["spearman"], 1.0)
        self.assertEqual(mono["steps"], "4/4")

    def test_turnover_series_matches_hand_count(self):
        targets = [["a", "b"], ["a", "b"], ["a", "c"], ["a", "c"]]
        result = scoring.turnover_series(targets, step=5)
        self.assertAlmostEqual(result["mean"], 0.5 / 3, places=4)
        self.assertAlmostEqual(result["annual"], 0.5 / 3 * 252 / 5 * 100, delta=0.5)

    def test_top_fraction_is_deterministic_on_ties(self):
        picks = scoring.top_fraction({"b": 1.0, "a": 1.0, "c": 0.5}, 0.34)
        self.assertEqual(picks, ["a"])

    def test_buffer_select_keeps_held_name_inside_band(self):
        held = ["b"]
        scores = {f"s{i:02d}": float(100 - i) for i in range(20)}
        scores["b"] = 95.0  # 第 6 名 / 21 只 ≈ 24%，落在 20%–35% 的缓冲带内
        picked = scoring.buffer_select(scores, held, entry_pct=0.2, exit_pct=0.35, target_pct=0.2)
        self.assertIn("b", picked)
        self.assertEqual(len(picked), 4)

    def test_buffer_select_drops_name_outside_exit(self):
        scores = {f"s{i:02d}": float(100 - i) for i in range(20)}
        scores["b"] = 1.0
        picked = scoring.buffer_select(scores, ["b"], entry_pct=0.2, exit_pct=0.35, target_pct=0.2)
        self.assertNotIn("b", picked)


if __name__ == "__main__":
    unittest.main()
