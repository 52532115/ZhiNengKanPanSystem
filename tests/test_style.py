"""风格基准的测试：市值代理、动量、相关性聚类、Fama-MacBeth 归因。"""

from __future__ import annotations

import math
import unittest

import style as st


def history(symbol: str, rows: list[dict], name: str | None = None) -> dict:
    return {"symbol": symbol, "name": name or symbol, "asset_type": "stock", "rows": rows}


def row(date: str, close: float, amount: float = 1e8, turnover: float = 2.0) -> dict:
    return {
        "date": date,
        "open": close,
        "close": close,
        "high": close,
        "low": close,
        "volume": 1000.0,
        "amount": amount,
        "turnover": turnover,
        "pct_change": 0.0,
        "change": 0.0,
    }


class SizeProxyTest(unittest.TestCase):
    def test_size_is_amount_over_turnover(self) -> None:
        series = st.size_series(history("600000", [row("d1", 10.0, amount=1e8, turnover=2.0)]))
        self.assertAlmostEqual(series["d1"], 1e8 / 0.02)

    def test_missing_turnover_falls_back_to_amount(self) -> None:
        series = st.size_series(history("600000", [row("d1", 10.0, amount=5e7, turnover=0.0)]))
        self.assertGreater(series["d1"], 0.0)


class MomentumTest(unittest.TestCase):
    def _series(self, step: float) -> list[dict]:
        return [row(f"d{i:04d}", 10.0 * (1 + step) ** i) for i in range(300)]

    def test_rising_series_has_positive_momentum(self) -> None:
        series = st.momentum_series(history("600000", self._series(0.001)))
        self.assertTrue(series)
        self.assertGreater(max(series.values()), 0.0)

    def test_lookback_is_respected(self) -> None:
        series = st.momentum_series(history("600000", self._series(0.001)), lookback=252)
        self.assertNotIn("d0100", series)
        self.assertIn("d0252", series)

    def test_skip_window_excludes_recent_month(self) -> None:
        rows = self._series(0.001)
        series = st.momentum_series(history("600000", rows), lookback=252, skip=21)
        expected = (rows[252 - 21]["close"] / rows[0]["close"] - 1) * 100
        self.assertAlmostEqual(series["d0252"], expected, places=6)


class ClusterTest(unittest.TestCase):
    def _histories(self) -> list[dict]:
        histories = []
        for group, phase in ((0, 0.0), (1, math.pi)):
            for index in range(4):
                rows = []
                for day in range(260):
                    value = 10.0 * (1 + 0.02 * math.sin(day * 0.9 + phase) + 0.001 * index)
                    rows.append(row(f"2024-{day:04d}", value))
                histories.append(history(f"{group}{index:05d}", rows, f"组{group}-{index}"))
        return histories

    def test_clusters_separate_unrelated_groups(self) -> None:
        clubs = st.correlation_clusters(self._histories(), clusters=2, min_overlap=200)
        first = clubs["mapping"]["000000"]
        for index in range(4):
            self.assertEqual(clubs["mapping"][f"{0}{index:05d}"], first)
        self.assertNotEqual(clubs["mapping"]["100000"], first)
        self.assertEqual(len(clubs["table"]), 2)

    def test_clustering_is_deterministic(self) -> None:
        histories = self._histories()
        first = st.correlation_clusters(histories, clusters=2, min_overlap=200)["mapping"]
        second = st.correlation_clusters(list(reversed(histories)), clusters=2, min_overlap=200)["mapping"]
        self.assertEqual(first, second)

    def test_style_panel_exposes_size_and_momentum(self) -> None:
        histories = self._histories()
        clubs = st.correlation_clusters(histories, clusters=2, min_overlap=200)
        panel = st.style_panel(histories, clubs)
        self.assertTrue(panel)
        sample = next(iter(panel))
        member = next(iter(panel[sample].values()))
        self.assertIn("size", member)
        self.assertIn("cluster", member)


class AttributionTest(unittest.TestCase):
    def _period(self, *, style_only: bool = True) -> dict:
        returns, exposures, benchmark = {}, {}, []
        for index in range(200):
            symbol = f"{index:06d}"
            size = math.log(1e9 * (1 + index))
            benchmark.append(symbol)
            exposures[symbol] = {"size": size, "mom": 5.0 + index, "cluster": 0.0}
            if style_only:
                returns[symbol] = 0.01 * (size - math.log(1e9 * 100.5))
            else:
                # 与市值无关的确定性噪声，才会被归因成残差 alpha
                noise = math.sin(index * 12.9898) * 43758.5453
                returns[symbol] = (noise - math.floor(noise)) * 0.04 - 0.02
        portfolio = sorted(benchmark)[::7] if not style_only else sorted(benchmark)[-20:]
        return {"start": "d1", "end": "d2", "returns": returns,
                "exposures": exposures, "portfolio": portfolio, "benchmark": benchmark}

    def test_pure_style_return_is_attributed_to_size(self) -> None:
        result = st.attribute_period(self._period(style_only=True))
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["contrib_size"], result["excess"], delta=0.05)
        self.assertAlmostEqual(result["alpha"], 0.0, delta=0.05)

    def test_pure_idiosyncratic_return_shows_up_as_alpha(self) -> None:
        result = st.attribute_period(self._period(style_only=False))
        self.assertIsNotNone(result)
        self.assertGreater(abs(result["alpha"]), 3 * abs(result["contrib_size"]))

    def test_thin_cross_section_is_skipped(self) -> None:
        period = self._period()
        period["returns"] = dict(list(period["returns"].items())[:5])
        self.assertIsNone(st.attribute_period(period))

    def test_aggregate_reports_t_stat_and_share(self) -> None:
        periods = []
        for scale in (1.0, 0.8, 1.3, 0.9, 1.1, 1.6):
            period = self._period(style_only=True)
            period["returns"] = {k: v * scale for k, v in period["returns"].items()}
            periods.append(period)
        summary = st.attribute(periods)["summary"]
        self.assertEqual(summary["n_periods"], 6)
        self.assertIn("alpha", summary)
        self.assertIsNotNone(summary["contrib_size"]["t"])
        self.assertLess(abs(summary["alpha_share"]), 5.0)

    def test_attribute_handles_empty_input(self) -> None:
        self.assertEqual(st.attribute([])["summary"], {})


if __name__ == "__main__":
    unittest.main()
