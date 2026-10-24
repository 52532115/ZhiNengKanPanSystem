"""PIT 池子的回归测试：成分不偷看未来、字段齐全、横截面按日过滤。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import protocol
from research import fetch_bars, universe
from research.manifest import Dataset


def _bar(day: str, close: float, *, amount: float = 1e8, turn: float = 2.0,
         is_st: bool = False, status: str = "1") -> tuple[str, ...]:
    return (
        day, f"{close}", f"{close}", f"{close}", f"{close}", f"{close}",
        f"{amount / close:.0f}", f"{amount}", f"{turn}", status, "1" if is_st else "0",
    )


def _build_dataset(root: Path) -> str:
    """两只票、三个交易日、两期成分快照的迷你数据集。"""
    version = "pit-test"
    dataset = Dataset.create(version=version, params={"test": True}, root=root)
    days = ["2020-01-02", "2020-01-03", "2020-01-06"]
    dataset.write_table("calendar", days, source="test", rows=len(days))
    dataset.write_table(
        "membership",
        {
            "2020-01-02": {"000300": ["000001"], "000905": ["000002"]},
            "2020-01-31": {"000300": ["000002"], "000905": []},
        },
        source="test",
        rows=2,
    )
    dataset.write_table(
        "industry",
        {
            "000001": {"name": "甲", "industry": "C39"},
            "000002": {"name": "乙", "industry": "J66"},
        },
        source="test",
        rows=2,
    )
    dataset.write_bars("000001", [_bar(day, 10.0 + index) for index, day in enumerate(days)])
    dataset.write_bars("000002", [
        _bar(days[0], 20.0),
        _bar(days[1], 20.0, is_st=True),          # 第二天起 ST
        _bar(days[2], 20.0, status="0"),          # 第三天停牌
    ])
    dataset.finalize_bars()
    return version


class PitUniverseTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.version = _build_dataset(self.root)
        self.pool = universe.PitUniverse(self.version, root=self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_members_never_look_ahead(self) -> None:
        # 2020-01-03 只能用 01-02 那一期快照，不能看到 01-31 的名单
        self.assertEqual(self.pool.members("2020-01-03", kind="hs300"), {"000001"})
        self.assertEqual(self.pool.members("2020-02-03", kind="hs300"), {"000002"})

    def test_members_union_covers_both_indexes(self) -> None:
        # 并集口径：当日 000001 在沪深300、000002 在中证500
        self.assertEqual(self.pool.members("2020-01-03"), {"000001", "000002"})
        self.assertEqual(self.pool.members("2020-01-03", kind="zz500"), {"000002"})

    def test_members_exclude_codes_without_bars(self) -> None:
        # 名单里有、但没有日线的标的（如已退市且未抓取）不进池
        self.assertNotIn("000003", self.pool.members("2020-01-03"))

    def test_tradable_filters_st_and_suspension(self) -> None:
        self.assertEqual(self.pool.tradable("2020-01-02"), ["000001", "000002"])
        self.assertEqual(self.pool.tradable("2020-01-03"), ["000001"])       # 000002 已 ST
        self.assertEqual(self.pool.tradable("2020-01-06"), ["000001"])       # 000002 停牌

    def test_fields_carry_industry_and_mcap(self) -> None:
        table = self.pool.fields("2020-01-02")
        self.assertEqual(table["000001"]["industry"], "C39")
        self.assertAlmostEqual(table["000001"]["mcap"], 1e8 / 0.02, places=0)
        self.assertIn("at_limit_up", table["000001"])

    def test_histories_are_backtest_ready(self) -> None:
        histories = self.pool.histories(min_rows=1)
        self.assertEqual([item["symbol"] for item in histories], ["000001", "000002"])
        self.assertEqual(histories[0]["rows"][0]["date"], "2020-01-02")
        self.assertGreater(histories[0]["rows"][0]["volume"], 0)

    def test_filter_by_members_drops_non_members(self) -> None:
        records = [
            {"date": "2020-01-02", "symbol": "000001"},
            {"date": "2020-01-02", "symbol": "000002"},
            {"date": "2020-02-03", "symbol": "000001"},
        ]
        kept = universe.filter_by_members(records, self.pool.members_fn(kind="hs300"))
        # 02-03 用 01-31 那一期名单，成分已换成 000002，000001 被剔除
        self.assertEqual([(r["date"], r["symbol"]) for r in kept], [("2020-01-02", "000001")])

    def test_field_audit_reports_coverage(self) -> None:
        audit = universe.field_audit(self.pool, sample_days=3)
        self.assertEqual(audit["samples"], 3)          # 三个交易日都晚于 2018
        self.assertGreaterEqual(audit["industry_known_pct"], 50)

    def test_write_cache_round_trip(self) -> None:
        out = self.root / "cache" / "stocks_pit"
        info = self.pool.write_cache(out, min_rows=1)
        self.assertEqual(info["symbols"], 2)
        members = universe.members_loader(Path(info["members"]))
        self.assertEqual(members("2020-01-03"), {"000001", "000002"})
        self.assertEqual(members("2019-12-31"), set())      # 快照之前为空


class FetchOrderTest(unittest.TestCase):
    def test_universe_alternates_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = Dataset.create(version="order-test", params={}, root=root)
            dataset.write_table(
                "membership",
                {
                    "2020-01-02": {"000300": ["000001", "000003"], "000905": ["000002"]},
                    "2020-01-31": {"000300": ["000001"], "000905": ["000002", "000004"]},
                },
                source="test",
                rows=2,
            )
            codes = fetch_bars.universe_codes(dataset)
            # 两只指数交替取值，且不重复
            self.assertEqual(codes[:3], ["000001", "000002", "000003"])
            self.assertEqual(len(codes), len(set(codes)))


class EqualWeightMembersTest(unittest.TestCase):
    def test_members_restrict_benchmark(self) -> None:
        bars = {
            "A": {day: {"open": 10.0, "close": 10.0, "high": 10.0, "low": 10.0,
                        "volume": 1e6, "amount": 1e7, "turnover": 2.0}
                  for day in ("2020-01-02", "2020-01-03")},
            "B": {day: {"open": 10.0, "close": 10.0, "high": 10.0, "low": 10.0,
                        "volume": 1e6, "amount": 1e7, "turnover": 2.0}
                  for day in ("2020-01-02", "2020-01-03")},
        }
        calendar = ["2020-01-02", "2020-01-03"]
        full = protocol.equal_weight_nav(calendar, bars, rebalance=1)
        restricted = protocol.equal_weight_nav(calendar, bars, rebalance=1,
                                               members=lambda day: {"A"})
        self.assertTrue(full["nav"])
        # 只允许 A 时仓位集中在 A 上：没有 B 的持仓
        held = {symbol for item in restricted["holdings"] for symbol in item.get("symbols", [])}
        self.assertEqual(held, {"A"})


if __name__ == "__main__":
    unittest.main()
