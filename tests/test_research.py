"""研究层测试：版本化缓存、清洗、PIT 池子与抽样、池层对比。

全部用合成数据，不碰网络也不依赖 baostock，因此可以在生产环境直接跑。
"""

from __future__ import annotations

import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from research import clean, pit
from research.manifest import Dataset
from research import pool_validate


def raw_row(date: str, close: float, *, preclose: float | None = None, volume: float = 1e6,
            amount: float | None = None, turn: float | None = 1.0, is_st: str = "0",
            status: str = "1", high: float | None = None, low: float | None = None,
            open_price: float | None = None) -> dict:
    return {
        "date": date,
        "open": str(open_price if open_price is not None else close),
        "high": str(high if high is not None else close * 1.01),
        "low": str(low if low is not None else close * 0.99),
        "close": str(close),
        "preclose": str(preclose if preclose is not None else close),
        "volume": str(volume),
        "amount": str(amount) if amount is not None else str(volume * close),
        "turn": "" if turn is None else str(turn),
        "tradestatus": status,
        "isST": is_st,
    }


class DatasetTest(unittest.TestCase):
    def test_write_read_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Dataset.create(version="v1", root=Path(tmp))
            dataset.write_table("calendar", ["2024-01-02"], source="test", rows=1)
            count = dataset.write_bars("600000", [("2024-01-02", "1", "1", "1", "10", "9", "100",
                                                  "1000", "1", "1", "0")])
            dataset.finalize_bars()
            self.assertEqual(count, 1)
            self.assertEqual(dataset.read_table("calendar"), ["2024-01-02"])
            self.assertEqual(dataset.read_bars("600000")[0]["close"], "10")
            self.assertTrue(dataset.verify()["ok"])

    def test_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Dataset.create(version="v1", root=Path(tmp))
            dataset.write_table("calendar", ["2024-01-02"], source="test")
            (dataset.root / "calendar.json").write_text('["2024-01-03"]', encoding="utf-8")
            result = dataset.verify()
            self.assertFalse(result["ok"])
            self.assertIn("calendar", result["changed"])

    def test_latest_version_picks_last(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            Dataset.create(version="pit-20260101T0000", root=Path(tmp))
            Dataset.create(version="pit-20260201T0000", root=Path(tmp))
            from research.manifest import latest_version
            self.assertEqual(latest_version(root=Path(tmp)), "pit-20260201T0000")


class NormalizeTest(unittest.TestCase):
    def _rows(self) -> list[dict]:
        return [
            raw_row("2024-01-02", 10.0, preclose=8.0),               # IPO 首日：+25%
            raw_row("2024-01-03", 11.0, preclose=10.0),
            raw_row("2024-01-04", 11.5, preclose=11.0),
            raw_row("2024-01-05", 11.5, preclose=11.5, status="0", volume=0.0),   # 停牌
            raw_row("2024-01-08", 12.65, preclose=11.5, high=12.65, low=12.65),   # 一字涨停
        ]

    def test_first_rows_are_excluded_from_returns(self) -> None:
        bars = clean.normalize(self._rows(), "600000")
        self.assertFalse(bars[0]["ret_ok"])
        self.assertFalse(bars[1]["ret_ok"])          # IPO_SKIP_DAYS = 2
        self.assertTrue(bars[2]["ret_ok"])
        self.assertAlmostEqual(bars[2]["ret"], 11.5 / 11.0 - 1)

    def test_suspension_and_limit_flags(self) -> None:
        bars = clean.normalize(self._rows(), "600000")
        self.assertTrue(bars[3]["suspended"])
        self.assertTrue(bars[4]["at_limit_up"])
        self.assertTrue(bars[4]["one_word"])
        self.assertFalse(bars[4]["suspended"])

    def test_amount_fallback_and_volume_adjustment(self) -> None:
        rows = [raw_row("2024-01-02", 10.0, volume=2e6, amount=0.0)]
        bars = clean.normalize(rows, "600000")
        self.assertTrue(bars[0]["amount_estimated"])
        self.assertAlmostEqual(bars[0]["amount"], 2e6 * 10.0)
        self.assertAlmostEqual(bars[0]["volume_adj"], 2e6)

    def test_st_uses_five_percent_limit(self) -> None:
        rows = [raw_row("2024-01-02", 10.0),
                raw_row("2024-01-03", 10.5, preclose=10.0, high=10.5, low=10.5, is_st="1")]
        bars = clean.normalize(rows, "600000")
        self.assertTrue(bars[1]["at_limit_up"])

    def test_star_board_uses_twenty_percent(self) -> None:
        rows = [raw_row("2024-01-02", 10.0),
                raw_row("2024-01-03", 12.0, preclose=10.0, high=12.0, low=12.0)]
        bars = clean.normalize(rows, "688001")
        self.assertTrue(bars[1]["at_limit_up"])

    def test_absurd_return_is_flagged(self) -> None:
        rows = [raw_row("2024-01-02", 10.0), raw_row("2024-01-03", 10.1),
                raw_row("2024-01-04", 10.1), raw_row("2024-01-05", 30.0, preclose=10.1)]
        bars = clean.normalize(rows, "600000")
        self.assertFalse(bars[3]["ret_ok"])

    def test_pre_listed_stocks_keep_long_listing_age(self) -> None:
        rows = [raw_row("2017-07-03", 10.0), raw_row("2017-07-04", 10.1)]
        fresh = clean.normalize(rows, "600000")
        old = clean.normalize(rows, "600000", dataset_start="2017-07-03")
        self.assertEqual(fresh[0]["listing_days"], 0)
        self.assertGreater(old[0]["listing_days"], 9000)

    def test_free_float_mcap_from_turnover(self) -> None:
        bar = {"amount": 1e8, "turnover": 2.0}
        self.assertAlmostEqual(clean.free_float_mcap(bar), 5e9)
        self.assertIsNone(clean.free_float_mcap({"amount": 1e8, "turnover": None}))

    def test_quality_report_counts_issues(self) -> None:
        bars = {"600000": clean.normalize(self._rows(), "600000")}
        report = clean.quality_report(bars)
        self.assertEqual(report["symbols"], 1)
        self.assertGreater(report["one_word_days"], 0)


class MembershipTest(unittest.TestCase):
    def setUp(self) -> None:
        self.book = pit.MembershipBook(
            {
                "2020-06-30": {"000300": ["600001", "600002"], "000905": ["000001"]},
                "2020-12-31": {"000300": ["600001", "600003"], "000905": ["000001", "000002"]},
            }
        )

    def test_snapshot_never_looks_ahead(self) -> None:
        self.assertEqual(self.book.snapshot_for("2020-07-15"), "2020-06-30")
        self.assertEqual(self.book.snapshot_for("2021-03-01"), "2020-12-31")
        self.assertIsNone(self.book.snapshot_for("2020-01-01"))

    def test_members_are_point_in_time(self) -> None:
        self.assertEqual(self.book.members("2020-07-15", "000300"), ["600001", "600002"])
        self.assertEqual(self.book.members("2021-01-05", "000300"), ["600001", "600003"])

    def test_union_collects_all_history(self) -> None:
        self.assertEqual(self.book.union("000300"), {"600001", "600002", "600003"})

    def test_survivors_bias_counts_removed_names(self) -> None:
        result = pit.survivors_bias(self.book, "000300", alive_end={"600001", "600003"})
        self.assertEqual(result["turnover_names"], 1)
        self.assertEqual(result["delisted_among_removed"], 1)
        self.assertAlmostEqual(result["turnover_share"], 50.0)


class PoolTest(unittest.TestCase):
    def _snapshot(self) -> dict[str, dict]:
        snapshot = {}
        for index in range(30):
            code = f"{600000 + index}"
            snapshot[code] = {
                "close": 10.0,
                "mcap": 1e9 * (1 + index),
                "is_st": index == 3,
                "listing_days": 100 if index == 5 else 800,
                "at_limit_up": False,
                "at_limit_down": False,
                "turnover": 1.5,
                "amount": 5e7,
                "board": "主板",
            }
        return snapshot

    def _industry(self) -> dict[str, dict]:
        mapping = {}
        for index in range(30):
            code = f"{600000 + index}"
            mapping[code] = {
                "name": f"公司{index}",
                "industry": "C39计算机通信电子" if index % 3 == 0 else "J66货币金融服务",
            }
        return mapping

    def test_filters_st_new_and_illiquid(self) -> None:
        pool = pit.build_pool("2024-01-02", snapshot=self._snapshot(), members=None,
                              industry=self._industry())
        codes = {item["symbol"] for item in pool}
        self.assertNotIn("600003", codes)       # ST
        self.assertNotIn("600005", codes)       # 上市不足 250 日
        self.assertEqual(len(pool), 28)

    def test_stratified_sample_is_deterministic_and_balanced(self) -> None:
        pool = pit.build_pool("2024-01-02", snapshot=self._snapshot(), members=None,
                              industry=self._industry())
        first = pit.stratified_sample(pool, target=20, seed=7)
        second = pit.stratified_sample(pool, target=20, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)
        industries = {item["symbol"]: item["industry"] for item in pool}
        picked = {industries[code] for code in first}
        self.assertEqual(len(picked), 2)         # 两个行业都要有代表

    def test_coverage_reports_concentration(self) -> None:
        pool = pit.build_pool("2024-01-02", snapshot=self._snapshot(), members=None,
                              industry=self._industry())
        result = pit.coverage(pool, self._industry())
        self.assertEqual(result["industries"], 2)
        self.assertGreater(result["top3_share"], 50)
        self.assertTrue(0 < result["hhi"] <= 1)

    def test_breadth_shrinks_with_correlation(self) -> None:
        base = [0.01 * math.sin(i * 0.3) for i in range(60)]
        independent = {f"s{i}": [base[j] * (1 if j % (i + 2) else -1) for j in range(60)]
                       for i in range(10)}
        result = pit.breadth_metrics(list(independent), independent)
        self.assertAlmostEqual(result["effective_holdings"], 10.0)
        self.assertLessEqual(result["grinold_breadth"], 10.0)

    def test_breadth_handles_empty(self) -> None:
        self.assertEqual(pit.breadth_metrics([], {})["effective_holdings"], 0)


class ComparisonHelpersTest(unittest.TestCase):
    def test_diff_pct_uses_legacy_as_denominator(self) -> None:
        self.assertEqual(pool_validate.diff_pct(0.013, 0.010), 30.0)
        self.assertIsNone(pool_validate.diff_pct(0.01, 0.0))
        self.assertIsNone(pool_validate.diff_pct(None, 0.1))

    def test_criteria_flags_short_history_and_thin_breadth(self) -> None:
        coverage = {"hs300": {"top3_share_mean": 45.0}}
        breadth = {"hs300": {"effective_holdings": 6.0}}
        comparisons = {"A": {"ic_diff_pct": 10.0, "nav_diff_pct": 5.0, "same_sign": True}}
        criteria = pool_validate._criteria(500, coverage,
                                           breadthe=breadth, comparisons=comparisons)
        status = {item["name"]: item["pass"] for item in criteria}
        self.assertFalse(status["有效宽度 > 10"])
        self.assertFalse(status["历史 ≥ 8 年"])
        self.assertTrue(status["行业不集中（前三行业 ≤ 50%）"])

    def test_to_history_maps_adjusted_volume(self) -> None:
        rows = clean.normalize(
            [raw_row("2024-01-02", 10.0), raw_row("2024-01-03", 10.5)],
            "600000", "测试",
        )
        history = pool_validate.to_history("600000", rows, "测试")
        self.assertEqual(history["rows"][0]["close"], 10.0)
        self.assertAlmostEqual(history["rows"][0]["volume"], rows[0]["amount"] / 10.0)
        self.assertEqual(history["data_source"], "baostock-hfq")


class PipelineIntegrationTest(unittest.TestCase):
    """小规模端到端：合成数据集 → 池子 → 对比，确认各层接口对得上。"""

    def _write_dataset(self, root: Path) -> str:
        dataset = Dataset.create(version="pit-test", root=root)
        dates = []
        day = 1
        while len(dates) < 520:
            month = 1 + (len(dates) // 22) % 12
            dates.append(f"20{18 + (len(dates) // 264)}-{month:02d}-{(len(dates) % 22) + 1:02d}")
        dataset.write_table("calendar", dates, source="test")
        codes_all = [f"{600000 + index}" for index in range(40)]
        membership = {
            "2018-01-05": {"000300": codes_all[:30], "000905": codes_all[10:40]},
            "2020-01-05": {"000300": codes_all[5:35], "000905": codes_all[:20]},
        }
        dataset.write_table("membership", membership, source="test")
        dataset.write_table("listed", {}, source="test")
        industry = {}
        random.seed(3)
        codes = [f"{600000 + index}" for index in range(40)]
        for index, code in enumerate(codes):
            industry[code] = {"name": f"公司{index}", "industry": "C39计算机通信电子"}
            price = 10.0 + index
            rows = []
            for position, date in enumerate(dates):
                drift = 0.0008 * math.sin(position / 13 + index) + 0.0002 * random.gauss(0, 1)
                previous = price
                price = max(1.0, price * (1 + drift))
                rows.append((date, f"{price:.3f}", f"{price * 1.01:.3f}", f"{price * 0.99:.3f}",
                             f"{price:.3f}", f"{previous:.3f}", "1000000", f"{1000000 * price:.2f}",
                             "1.2", "1", "0"))
            dataset.write_bars(code, rows)
        dataset.write_table("industry", industry, source="test")
        dataset.finalize_bars()
        benchmark = {
            "symbol": "000300",
            "name": "沪深300",
            "rows": [
                {"date": date, "open": 4000 + i, "high": 4010 + i, "low": 3990 + i,
                 "close": 4000 + i, "preclose": 3999 + i, "volume": 1e9, "amount": 1e11,
                 "turnover": None}
                for i, date in enumerate(dates)
            ],
        }
        (dataset.root / "benchmark.json").write_text(json.dumps(benchmark), encoding="utf-8")
        return dataset.version

    def test_end_to_end_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            version = self._write_dataset(Path(tmp))
            report = pool_validate.run(version, target=30, iters=200,
                                       layers=("hs300",), root=Path(tmp), log=lambda *_: None)
            self.assertIn("criteria", report)
            # 5 条：宽度、行业、历史、IC 一致性、净值一致性（IC 与净值必须分开判）
            self.assertEqual(len(report["criteria"]), 5)
            self.assertIn("A_current_vs_pit", report["comparisons"])
            self.assertTrue(report["eval"]["days"] > 100)
            markdown = pool_validate.render_markdown(report)
            self.assertIn("幸存者偏差", markdown)
            self.assertIn("通过标准", markdown)


if __name__ == "__main__":
    unittest.main()
