import math
import random
import statistics
import unittest

import wfv2


def fake_rows_and_bars(days=120, stocks=30, seed=3):
    rng = random.Random(seed)
    bars = {}
    for i in range(stocks):
        symbol = f"S{i:02d}"
        price = 10.0
        series = {}
        for day in range(days):
            price *= 1 + rng.gauss(0, 0.01)
            series[f"d{day:04d}"] = {"open": price, "close": price, "high": price, "low": price}
        bars[symbol] = series
    dates = [f"d{day:04d}" for day in range(days)]
    rows_by_date = {
        date: [
            {"date": date, "symbol": f"S{i:02d}",
             **{f"s_{key}": rng.gauss(0, 1) for key in wfv2.ALL_FEATURES}}
            for i in range(stocks)
        ]
        for date in dates
    }
    index = {date: 100 + day * 0.05 for date, day in zip(dates, range(days))}
    return rows_by_date, bars, index, dates


class LabelTest(unittest.TestCase):
    def test_labels_are_percentiles_and_drop_tail(self):
        rows, bars, index, dates = fake_rows_and_bars()
        labels = wfv2.forward_labels(rows, bars, index, dates, horizons=(5,))
        values = [entry["label"] for entry in labels[5].values() if "label" in entry]
        self.assertTrue(values)
        self.assertTrue(all(0 < value < 1 for value in values))
        # 最后 5 天没有完整未来窗口，不应有标签
        for offset in range(5):
            date = dates[-1 - offset]
            self.assertFalse(any(key[0] == date for key in labels[5]))

    def test_uniqueness_weight_is_inverse_horizon(self):
        weights = wfv2.uniqueness_weights(["a", "b"], 20)
        self.assertAlmostEqual(weights["a"], 0.05)


class StatsTest(unittest.TestCase):
    def test_subset_stats_matches_direct_build(self):
        rows, _, _, dates = fake_rows_and_bars(days=60)
        labels = wfv2.forward_labels(rows, *fake_rows_and_bars(days=60)[1:], horizons=(5,))[5]
        stats = wfv2.build_stats(rows, wfv2.ALL_FEATURES, labels)
        subset = wfv2.subset_stats(stats, ("rev5", "ivol"))
        direct = wfv2.build_stats(rows, ("rev5", "ivol"), labels)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(subset.xtx[i][j], direct.xtx[i][j], places=6)
            self.assertAlmostEqual(subset.xty[i], direct.xty[i], places=6)

    def test_constant_weights_do_not_change_fit(self):
        rows, bars, index, dates = fake_rows_and_bars(days=80)
        labels = wfv2.forward_labels(rows, bars, index, dates, horizons=(5,))[5]
        plain = wfv2.build_stats(rows, wfv2.ALL_FEATURES, labels)
        weighted = wfv2.build_stats(rows, wfv2.ALL_FEATURES, labels,
                                    weight_fn=lambda date: 1 / 5)
        prior = [0.25 / len(wfv2.ALL_FEATURES)] * len(wfv2.ALL_FEATURES)
        # 权重整体缩放只改变有效岭强度：取极大的 τ 让岭项可忽略，两者应一致
        a = wfv2.fit_ridge(plain.xtx, plain.xty, yyw=plain.yyw, count=plain.n_effective,
                           prior_mean=prior, tau=1e4)
        b = wfv2.fit_ridge(weighted.xtx, weighted.xty, yyw=weighted.yyw,
                           count=weighted.n_effective, prior_mean=prior, tau=1e4)
        for left, right in zip(a, b):
            self.assertAlmostEqual(left, right, places=3)


class RidgeTest(unittest.TestCase):
    def _stats(self):
        rows, bars, index, dates = fake_rows_and_bars(days=90, seed=7)
        labels = wfv2.forward_labels(rows, bars, index, dates, horizons=(5,))[5]
        return rows, wfv2.build_stats(rows, wfv2.ALL_FEATURES, labels)

    def test_strong_prior_pulls_coefficients_to_prior(self):
        rows, stats = self._stats()
        prior = [0.02] * len(wfv2.ALL_FEATURES)
        tight = wfv2.fit_ridge(stats.xtx, stats.xty, yyw=stats.yyw,
                              count=stats.n_effective, prior_mean=prior, tau=1e-3)
        loose = wfv2.fit_ridge(stats.xtx, stats.xty, yyw=stats.yyw,
                              count=stats.n_effective, prior_mean=prior, tau=1e3)
        self.assertLess(max(abs(value - 0.02) for value in tight), 0.02)
        self.assertGreater(sum(abs(value) for value in loose),
                           sum(abs(value - 0.02) for value in tight))

    def test_gamma_pulls_towards_previous(self):
        rows, stats = self._stats()
        prior = [0.0] * len(wfv2.ALL_FEATURES)
        previous = [0.05] * len(wfv2.ALL_FEATURES)
        plain = wfv2.fit_ridge(stats.xtx, stats.xty, yyw=stats.yyw,
                              count=stats.n_effective, prior_mean=prior, tau=1e3)
        smoothed = wfv2.fit_ridge(stats.xtx, stats.xty, yyw=stats.yyw,
                                  count=stats.n_effective, prior_mean=prior, tau=1e3,
                                  previous=previous, gamma=10.0)
        distance_plain = sum(abs(a - b) for a, b in zip(plain, previous))
        distance_smoothed = sum(abs(a - b) for a, b in zip(smoothed, previous))
        self.assertLess(distance_smoothed, distance_plain)


class RankTest(unittest.TestCase):
    def test_rank_average_is_mean_and_handles_missing(self):
        left = {("d1", "a"): 0.1, ("d1", "b"): 0.9}
        right = {("d1", "a"): 0.3}
        out = wfv2.rank_average([left, right])
        self.assertAlmostEqual(out[("d1", "a")], (0.1 + 0.3) / 2)
        self.assertAlmostEqual(out[("d1", "b")], 0.9)   # 不补 0.5

    def test_tiebreak_breaks_plateaus(self):
        left = {("d1", "a"): 0.5, ("d1", "b"): 0.5, ("d1", "c"): 0.5}
        tiebreak = {("d1", "a"): 3.0, ("d1", "b"): 1.0, ("d1", "c"): 2.0}
        out = wfv2.rank_average([left], tiebreak=tiebreak)
        self.assertGreater(out[("d1", "a")], out[("d1", "c")])
        self.assertGreater(out[("d1", "c")], out[("d1", "b")])


class PurgeTest(unittest.TestCase):
    def test_purge_excludes_window_around_test(self):
        dates = [f"d{i:04d}" for i in range(100)]
        train = dates[:40] + dates[60:]
        kept = wfv2.purge_dates(train, dates, "d0040", "d0059", horizon=20, embargo=5)
        self.assertNotIn("d0039", kept)    # 距测试块 1 天，落在 purge 区
        self.assertNotIn("d0015", kept)    # 距测试块 25 天，仍被 20+5 排除
        self.assertIn("d0014", kept)
        self.assertIn("d0065", kept)


class PeriodReturnTest(unittest.TestCase):
    def test_cost_lowers_period_returns(self):
        random.seed(1)
        dates = [f"d{i:04d}" for i in range(60)]
        scores = {(date, f"S{i:02d}"): random.random() for date in dates for i in range(20)}
        labels = {(date, f"S{i:02d}"): {"excess": 1.0} for date in dates for i in range(20)}
        cheap = wfv2.period_returns(scores, labels, dates, cost_buy=0.0, cost_sell=0.0)
        dear = wfv2.period_returns(scores, labels, dates, cost_buy=0.01, cost_sell=0.01)
        self.assertGreater(cheap["mean"], dear["mean"])


if __name__ == "__main__":
    unittest.main()
