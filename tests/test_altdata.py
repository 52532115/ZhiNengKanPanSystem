import math
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import evalkit
import fields as fd
from research import fundamentals as fz
from research.datasets import base as dsbase
from research.datasets.csv_drop import CsvDrop
from research.pit_join import AsOfBook, audit, derive, vintages


def record(symbol, field, value, period, available, dimension="fundamental"):
    return {"symbol": symbol, "dimension": dimension, "field": field, "value": value,
            "period": period, "available": available, "source": "test"}


class ValidateTest(unittest.TestCase):
    def test_rejects_available_before_period(self):
        bad = [record("600519", "roe", 0.2, "2023-12-31", "2023-12-01")]
        with self.assertRaises(dsbase.SourceError):
            dsbase.validate(bad)
        self.assertTrue(dsbase.validate(bad, strict=False))

    def test_drops_none_and_normalises_symbol(self):
        rows = [
            record("519", "roe", 0.2, "2023-12-31", "2024-04-03"),
            record("600519", "roe", None, "2023-12-31", "2024-04-03"),
            {"symbol": "600519", "dimension": "fundamental", "field": "roe", "value": float("nan"),
             "period": "2023-12-31", "available": "2024-04-03", "source": "test"},
        ]
        out = dsbase.compact(rows)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["symbol"], "000519")

    def test_forward_looking_event_is_allowed(self):
        # 业绩预告：2023-01-20 公告，报告期 2022-12-31 之后才结束 → 负滞后是正常的
        rows = [record("600519", "forecast_direction", 1.0, "2022-12-31", "2023-01-20", "event")]
        self.assertEqual(dsbase.validate(rows, strict=False), [])

    def test_duplicate_records_are_reported(self):
        rows = [record("600519", "roe", 0.2, "2023-12-31", "2024-04-03")] * 2
        problems = dsbase.validate(rows, strict=False)
        self.assertTrue(any("重复" in item for item in problems))


class AsOfTest(unittest.TestCase):
    def setUp(self):
        self.book = AsOfBook([
            record("A", "roe", 0.10, "2022-12-31", "2023-04-20"),
            record("A", "roe", 0.12, "2023-12-31", "2024-04-20"),
            record("A", "roe", 0.11, "2023-12-31", "2025-04-20"),   # 重述：向下修正
        ])

    def test_never_sees_future_announcement(self):
        self.assertIsNone(self.book.value("A", "roe", "2023-04-19"))
        self.assertAlmostEqual(self.book.value("A", "roe", "2023-04-20"), 0.10)

    def test_uses_latest_revision_available_then(self):
        # 2024-05 看到的是 0.12；2025-05 之后才看到修正后的 0.11
        self.assertAlmostEqual(self.book.value("A", "roe", "2024-05-01"), 0.12)
        self.assertAlmostEqual(self.book.value("A", "roe", "2025-05-01"), 0.11)

    def test_max_stale_blanks_old_values(self):
        self.assertIsNone(self.book.value("A", "roe", "2027-01-01", max_stale=400))

    def test_window_sum_and_count(self):
        book = AsOfBook([
            record("A", "div", 0.5, "2023-01-10", "2023-01-10", "event"),   # 窗口外
            record("A", "div", 0.7, "2024-01-15", "2024-01-15", "event"),
            record("A", "div", 0.9, "2024-09-01", "2024-09-01", "event"),
        ])
        self.assertAlmostEqual(book.window_sum("A", "div", "2024-06-01", 365), 0.7)
        self.assertAlmostEqual(book.window_sum("A", "div", "2024-12-31", 365), 1.6)
        self.assertEqual(book.window_count("A", "div", "2024-12-31", 365), 2)


class DeriveTest(unittest.TestCase):
    def test_ttm_uses_year_ago_annual_and_quarter(self):
        rows = [
            # 2022 年报：累计净利 100
            record("A", "net_profit", 100.0, "2022-12-31", "2023-04-01"),
            record("A", "revenue", 1000.0, "2022-12-31", "2023-04-01"),
            # 2023Q1：累计 30
            record("A", "net_profit", 30.0, "2023-03-31", "2023-04-28"),
            record("A", "revenue", 300.0, "2023-03-31", "2023-04-28"),
            # 2023 年报：累计 90（TTM 必须用它，不能用 2022 年报）
            record("A", "net_profit", 90.0, "2023-12-31", "2024-04-20"),
            record("A", "revenue", 900.0, "2023-12-31", "2024-04-20"),
            # 2024Q1：累计 40 → TTM = 40 + 100 − 30 = 110
            record("A", "net_profit", 40.0, "2024-03-31", "2024-04-28"),
            record("A", "revenue", 360.0, "2024-03-31", "2024-04-28"),
            record("A", "roe", 0.10, "2024-03-31", "2024-04-28"),
            record("A", "total_share", 10.0, "2024-03-31", "2024-04-28"),
        ]
        derived = fz.derive_fundamentals(rows)
        latest = [r for r in derived if r["period"] == "2024-03-31"]
        values = {r["field"]: r["value"] for r in latest}
        self.assertAlmostEqual(values["np_ttm"], 40 + 90 - 30)      # 100
        self.assertAlmostEqual(values["rev_ttm"], 360 + 900 - 300)  # 960
        # 净资产 = 40 / 0.10 = 400，BPS = 400/10 = 40
        self.assertAlmostEqual(values["equity"], 400.0)
        self.assertAlmostEqual(values["bps"], 40.0)
        # 同比：上年同期 TTM 需要 2022 年报与 2022Q1，后者缺失 → 不给值
        self.assertNotIn("np_yoy", values)

    def test_loss_making_has_no_bp(self):
        rows = [
            record("A", "net_profit", -20.0, "2023-12-31", "2024-04-01"),
            record("A", "roe", -0.05, "2023-12-31", "2024-04-01"),
            record("A", "total_share", 10.0, "2023-12-31", "2024-04-01"),
        ]
        derived = fz.derive_fundamentals(rows)
        fields = {r["field"] for r in derived}
        self.assertNotIn("bps", fields)
        self.assertNotIn("equity", fields)

    def test_derive_runs_per_vintage(self):
        rows = [record("A", "x", 1.0, "2023-12-31", "2024-04-01")]
        out = derive(rows, lambda s, p, a, frame: {"double": frame["x"] * 2},
                     dimension="fundamental")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["value"], 2.0)
        frames = vintages(rows)
        self.assertIn(("A", "2023-12-31", "2024-04-01"), frames)


class AuditTest(unittest.TestCase):
    def test_reports_lag_and_revisions(self):
        rows = [
            record("A", "roe", 0.1, "2023-03-31", "2023-04-30"),
            record("A", "roe", 0.11, "2023-03-31", "2023-05-20"),
            record("A", "eps", 1.0, "2023-06-30", "2023-08-31"),
        ]
        out = audit(rows)
        self.assertEqual(out["violations"], 0)
        self.assertEqual(out["revised_series"], 1)
        self.assertEqual(out["lag_days"]["fundamental"]["median"], 50)   # 30 / 50 / 62 的中位


class PanelTest(unittest.TestCase):
    """构造一个"价值因子真的能预测"的合成市场，检查管线方向正确。"""

    def _market(self, symbols=40, days=140, seed=11):
        rng = random.Random(seed)
        prices, records, industry = {}, [], {}
        for i in range(symbols):
            symbol = f"S{i:03d}"
            quality = rng.gauss(0, 1)
            industry[symbol] = "C" if i % 2 == 0 else "I"
            level, bars = 10.0, []
            drift = 0.0006 * quality
            for day in range(days):
                level *= 1 + drift + rng.gauss(0, 0.004)
                bars.append({"date": f"2023-{1 + day // 28:02d}-{1 + day % 28:02d}",
                             "open": level, "close": level, "high": level, "low": level})
            for day in range(0, days, 5):
                prices.setdefault(symbol, {"symbol": symbol, "rows": []})
            prices[symbol] = {"symbol": symbol, "rows": bars}
            # 每个季度一条"高 EP 且未来收益更高"的记录
            for pivot in ("2023-01-30", "2023-04-30"):
                records.append(record(symbol, "eps_ttm", 1.0 + 0.2 * quality,
                                      "2022-12-31", pivot))
                records.append(record(symbol, "close_raw", 10.0, "2022-12-31", "2022-12-31", "market"))
                for bar in bars:
                    records.append(record(symbol, "close_raw", bar["close"], bar["date"], bar["date"], "market"))
        return prices, records, industry

    def test_panel_is_ordered_and_has_forward_returns(self):
        prices, records, industry = self._market()
        rows = fz.build_panel(records=records, prices=prices, industry=industry, benchmark={})
        self.assertTrue(rows)
        self.assertEqual(rows, sorted(rows, key=lambda r: (r["date"], r["symbol"])))
        self.assertTrue(all(row["fwd"] is not None for row in rows))

    def test_period_alignment_leaks_the_future(self):
        prices, records, industry = self._market()
        book_period = fz.build_panel(records=records, prices=prices, industry=industry,
                                     benchmark={}, alignment="period")
        # 报告期对齐：2022-12-31 就"看到"了 2023-04-30 才公告的 EPS
        early = [row for row in book_period if row["date"] <= "2023-01-05"]
        self.assertTrue(early)
        self.assertTrue(all(row.get("ep") is not None for row in early))

        book_available = fz.build_panel(records=records, prices=prices, industry=industry, benchmark={})
        early2 = [row for row in book_available if row["date"] <= "2023-01-05"]
        self.assertTrue(all(row.get("ep") is None for row in early2))


class NeutralizeTest(unittest.TestCase):
    def _rows(self, signal_strength, seed=5):
        rng = random.Random(seed)
        rows = []
        for day in range(60):
            date = f"2024-{1 + day // 28:02d}-{1 + day % 28:02d}"
            for i in range(60):
                industry = "C" if i < 30 else "I"
                size = rng.gauss(10, 0.4)
                value = rng.gauss(0, 1)
                fwd = signal_strength * value + rng.gauss(0, 1)
                rows.append({"date": date, "symbol": f"S{i:03d}", "industry": industry,
                             "fwd": fwd, "log_cap": size, "beta120": rng.gauss(1, 0.2),
                             "base_score": rng.gauss(0, 1), "probe": value})
        return rows

    def test_neutralization_keeps_real_signal(self):
        rows = self._rows(0.8)
        fz.neutralize(rows, ["probe"])
        series = fz.daily_ic(rows, "neu_probe")
        self.assertGreater(evalkit.spearman([r["probe"] for r in rows if r["probe"] is not None],
                                            [r["fwd"] for r in rows if r["probe"] is not None]), 0.3)
        self.assertGreater(len(series), 50)

    def test_incremental_strips_overlap_with_baseline(self):
        rows = self._rows(0.0)
        # 因子 = 基线 + 噪声：对基线正交化后应该几乎什么都不剩
        for row in rows:
            row["copy"] = row["base_score"] + 0.01 * row["probe"]
        fz.neutralize(rows, ["copy"])
        inc = fz.daily_ic(rows, "inc_copy")
        neu = fz.daily_ic(rows, "neu_copy")
        self.assertLess(abs(evalkit.ic_ci(inc, horizon=20, iters=200)["mean"]), 0.10)
        self.assertGreater(len(neu), 0)


class CsvDropTest(unittest.TestCase):
    def test_reads_template_rows(self):
        with TemporaryDirectory() as tmp:
            cache = Path(tmp)
            (cache / "csv").mkdir()
            (cache / "csv" / "manual.csv").write_text(
                "symbol,field,period,available,value\n"
                "600519,analyst_revision,2024-03-31,2024-04-08,0.021\n"
                "999999,analyst_revision,2024-03-31,2024-04-08,0.5\n",
                encoding="utf-8",
            )
            source = CsvDrop()
            ctx = dsbase.FetchContext(symbols=["600519"], start="2024-01-01", end="2024-12-31",
                                      cache=cache, log=lambda *_: None)
            rows = source.fetch(ctx)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["field"], "analyst_revision")
            self.assertEqual(rows[0]["value"], 0.021)


class RegistryTest(unittest.TestCase):
    def test_catalog_has_five_dimensions_and_probe_reasons(self):
        from research.datasets import catalog
        cat = catalog()
        self.assertIn("baostock_fundamental", cat)
        dimensions = {d for entry in cat.values() for d in entry["dimensions"]}
        self.assertTrue({"fundamental", "event", "flow", "market"} <= dimensions)
        for entry in cat.values():
            self.assertIn("available", entry)
            if not entry["available"]:
                self.assertTrue(entry["reason"])


if __name__ == "__main__":
    unittest.main()
