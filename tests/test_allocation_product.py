"""ETF 配置产品的验收：确定性、无前视、约束、过期数据、纸面跟踪口径。"""

from __future__ import annotations

import copy
import datetime
import json
import math
import tempfile
import unittest
import warnings
from pathlib import Path

import etf_allocation as E

START = "2019-01-02"


def trading_days(count: int, start: str = START) -> list[str]:
    days = []
    day = datetime.date.fromisoformat(start)
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day.isoformat())
        day += datetime.timedelta(days=1)
    return days


def make_history(symbol: str, days: list[str], *, drift: float, wave: float = 0.004,
                 start: str = START) -> dict:
    rows = []
    price = 1.0
    for index, day in enumerate(days):
        price = max(0.1, price * (1 + drift + wave * math.sin(index / 9)))
        rows.append({
            "date": day,
            "open": price * 0.998,
            "high": price * 1.004,
            "low": price * 0.996,
            "close": price,
            "volume": 1_000_000,
            "amount": price * 1_000_000,
            "turnover": 1.2,
            "pct_change": drift * 100,
            "change": price * drift,
        })
    return {"symbol": symbol, "name": symbol, "asset_type": "etf", "data_source": "test", "rows": rows}


def make_panel(count: int = 420, *, truncate: str | None = None) -> dict:
    days = trading_days(count)
    panel = {
        "510300": make_history("510300", days, drift=0.0009),
        "510500": make_history("510500", days, drift=0.0007),
        "511010": make_history("511010", days, drift=0.0002, wave=0.0006),
        "518880": make_history("518880", days, drift=0.0006, wave=0.003),
        "513100": make_history("513100", days, drift=0.0011),
        E.CASH: make_history(E.CASH, days, drift=0.00008, wave=0.00002),
    }
    if truncate:
        keep = int(count * 0.6)
        panel[truncate] = {**panel[truncate], "rows": panel[truncate]["rows"][:keep]}
    return panel


class SnapshotTest(unittest.TestCase):
    def test_snapshot_is_deterministic(self):
        panel = make_panel()
        first = E.compute_snapshot(copy.deepcopy(panel))
        second = E.compute_snapshot(copy.deepcopy(panel))
        self.assertEqual(first["weights"], second["weights"])
        self.assertEqual(first["weights_hash"], second["weights_hash"])
        self.assertEqual(first["data_version"], second["data_version"])
        self.assertTrue(first["data_version"].startswith("etf-"))

    def test_constraints_hold(self):
        snapshot = E.compute_snapshot(make_panel())
        weights = snapshot["weights"]
        self.assertLessEqual(sum(weights.values()), 1 + 1e-9)
        legs = {key: value for key, value in weights.items() if key != E.CASH}
        self.assertTrue(legs, "合成面板应当有腿入选")
        self.assertLessEqual(max(legs.values()), 0.30 + 1e-9)
        for bucket, cap in E.Config().bucket_caps:
            total = sum(value for key, value in legs.items()
                        if E.UNIVERSE.get(key, {}).get("bucket") == bucket)
            self.assertLessEqual(total, cap + 1e-9, f"{bucket} 超上限")
        self.assertGreaterEqual(snapshot["cash"]["weight"], 0.0)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)
        self.assertTrue(snapshot["constraints"]["ok"], snapshot["constraints"]["violations"])

    def test_cash_leg_earns_yield(self):
        snapshot = E.compute_snapshot(make_panel())
        self.assertGreater(snapshot["cash"]["weight"], 0.0)
        self.assertIsNotNone(snapshot["cash"]["yield_60d_annual"])
        self.assertGreater(snapshot["cash"]["yield_60d_annual"], 0.0)

    def test_signal_date_and_fill_date_are_separate(self):
        panel = make_panel()
        days = [row["date"] for row in panel["510300"]["rows"]]
        middle = days[-40]
        backdated = E.compute_snapshot(panel, as_of=middle)
        self.assertEqual(backdated["signal_date"], middle)
        self.assertNotEqual(backdated["fill_date"], backdated["signal_date"])
        self.assertGreater(backdated["fill_date"], backdated["signal_date"])
        self.assertEqual(backdated["fill_status"], "effective")
        # 最新一根收盘：T+1 还没发生，绝不能假装已成交
        live = E.compute_snapshot(panel)
        self.assertIsNone(live["fill_date"])
        self.assertEqual(live["fill_status"], "pending_next_open")

    def test_holdings_carry_reasons(self):
        snapshot = E.compute_snapshot(make_panel())
        for leg in snapshot["holdings"]:
            self.assertIn("momentum_12_1", leg["reason"])
            self.assertIsNotNone(leg["reason"]["above_ma120"])
            self.assertIn(leg["reason"]["rule"], {"enter", "hold"})

    def test_entry_uses_next_trading_day(self):
        """信号日之后的下一个交易日才是成交日；不是自然日 +1。"""
        panel = make_panel()
        days = [row["date"] for row in panel["510300"]["rows"]]
        snapshot = E.compute_snapshot(panel, as_of=days[-40])
        position = days.index(snapshot["signal_date"])
        self.assertEqual(snapshot["fill_date"], days[position + 1])


class StaleTest(unittest.TestCase):
    def test_stale_symbol_is_excluded_but_calendar_kept(self):
        full = make_panel()
        broken = make_panel(truncate="513100")
        full_calendar = E._calendar(full, start=START)
        broken_calendar = E._calendar(broken, start=START)
        # 关键：一条腿清盘不能把整段日历截短
        self.assertEqual(len(full_calendar), len(broken_calendar))
        stale = E.stale_symbols(broken)
        self.assertIn("513100", stale)
        self.assertNotIn("510300", stale)
        snapshot = E.compute_snapshot(broken)
        self.assertNotIn("513100", snapshot["weights"])
        self.assertNotIn("513100", snapshot["eligible"])
        self.assertIn("513100", snapshot["stale"])
        self.assertTrue(any("过期" in note for note in E._risk_notes(snapshot, None)))


class LatestJsonTest(unittest.TestCase):
    def test_latest_contract(self):
        panel = make_panel()
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = E.compute_snapshot(panel)
            path = E.write_latest(snapshot, outdir=Path(tmp), backtest={"metrics": {}}, paper={})
            self.assertTrue(path.is_file())
            payload = json.loads(path.read_text(encoding="utf-8"))
            required = ("data_version", "rule_version", "as_of", "signal_date", "fill_date",
                        "fill_status", "holdings", "cash", "exposure", "weights_sum",
                        "constraints", "last_rebalance", "backtest", "disclaimer")
            for key in required:
                self.assertIn(key, payload)
            self.assertEqual(payload["disclaimer"], "研究结果，不是投资建议。")
            self.assertLessEqual(payload["exposure"] + payload["cash"]["weight"], 1 + 1e-6)
            self.assertTrue(payload["holdings"])
            self.assertTrue(payload["rules"]["execution"].startswith("T 日收盘"))
            # 观察期闸门：纸面不足 3 期时不允许对外宣称可执行
            self.assertEqual(payload["observation"]["status"], "observing")
            self.assertLess(payload["observation"]["periods"], 3)

    def test_changes_fall_back_to_last_rebalance(self):
        panel = make_panel()
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = E.compute_snapshot(panel)
            path = E.write_latest(snapshot, outdir=Path(tmp))
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("changes_basis", payload)
            self.assertIn("上次调仓", payload["changes_basis"])


class PaperTest(unittest.TestCase):
    def test_record_is_idempotent(self):
        panel = make_panel()
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "paper.jsonl"
            snapshot = E.compute_snapshot(panel)
            first = E.record_paper(snapshot, path=ledger)
            second = E.record_paper(snapshot, path=ledger)
            self.assertTrue(first["recorded"])
            self.assertFalse(second["recorded"])
            self.assertEqual(second["reason"], "duplicate")
            self.assertEqual(len(E.read_paper(ledger)), 1)

    def test_record_respects_rebalance_cadence(self):
        panel = make_panel()
        days = [row["date"] for row in panel["510300"]["rows"]]
        calendar = E._calendar(panel, start=START)
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "paper.jsonl"
            E.record_paper(E.compute_snapshot(panel, as_of=days[-40]), path=ledger)
            # 次日就再记一次：不是调仓日，应被挡住（否则纸面变成日频，成本口径与回测不可比）
            skipped = E.record_paper(E.compute_snapshot(panel, as_of=days[-20]), path=ledger,
                                     calendar=calendar)
            self.assertFalse(skipped["recorded"])
            self.assertIn("not_due", skipped["reason"])
            forced = E.record_paper(E.compute_snapshot(panel, as_of=days[-20]), path=ledger,
                                    calendar=calendar, force=True)
            self.assertTrue(forced["recorded"])

    def test_turnover_and_cost_match_backtest_formula(self):
        panel = make_panel()
        days = [row["date"] for row in panel["510300"]["rows"]]
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "paper.jsonl"
            early = E.compute_snapshot(panel, as_of=days[-60])
            later = E.compute_snapshot(panel, as_of=days[-10])
            E.record_paper(early, path=ledger)
            E.record_paper(later, path=ledger)
            records = E.read_paper(ledger)
            previous, current = records[0]["weights"], records[1]["weights"]
            buy = sum(max(0.0, current.get(s, 0.0) - previous.get(s, 0.0)) for s in set(previous) | set(current))
            sell = sum(max(0.0, previous.get(s, 0.0) - current.get(s, 0.0)) for s in set(previous) | set(current))
            self.assertAlmostEqual(records[1]["one_way_turnover"], 0.5 * (buy + sell), places=6)
            self.assertAlmostEqual(records[1]["cost_bps"],
                                   (buy * E.BUY_COST + sell * E.SELL_COST) * 10000, places=6)
            # 与回测同一组费率：单边 7.5bp
            self.assertAlmostEqual(E.BUY_COST * 10000, 7.5, places=6)
            self.assertAlmostEqual(E.SELL_COST * 10000, 7.5, places=6)

    def test_paper_summary_settles_on_open_prices(self):
        panel = make_panel()
        days = [row["date"] for row in panel["510300"]["rows"]]
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "paper.jsonl"
            E.record_paper(E.compute_snapshot(panel, as_of=days[-80]), path=ledger)
            E.record_paper(E.compute_snapshot(panel, as_of=days[-40]), path=ledger)
            summary = E.paper_summary(panel, path=ledger, start=START)
            self.assertTrue(summary["available"])
            self.assertEqual(summary["n_periods"], 1)
            period = summary["periods"][0]
            position = days.index(period["signal_date"])
            self.assertEqual(period["entry_date"], days[position + 1])
            self.assertIn("net_return", period)
            self.assertIn("cost_consistent", summary)


class AppPayloadTest(unittest.TestCase):
    def test_app_reads_latest_json_only(self):
        import app as web

        panel = make_panel()
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = E.compute_snapshot(panel)
            path = E.write_latest(snapshot, outdir=Path(tmp))
            original, stamp = web.ALLOCATION_PATH, web._ALLOCATION_STAMP
            try:
                web.ALLOCATION_PATH = path
                web._ALLOCATION_STAMP = 0.0
                payload = web.allocation_payload()
                self.assertTrue(payload["available"])
                self.assertEqual(payload["disclaimer"], E.DISCLAIMER)
                self.assertEqual(payload["signal_date"], snapshot["signal_date"])
                # 生产层只读产物：既不能 import research，也不能 import 配置引擎
                import ast

                tree = ast.parse(Path(web.__file__).read_text(encoding="utf-8"))
                modules = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        modules.update(alias.name.split(".")[0] for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                        modules.add(node.module.split(".")[0])
                self.assertNotIn("research", modules)
                self.assertNotIn("etf_allocation", modules)
                self.assertNotIn("crossasset", modules)
            finally:
                web.ALLOCATION_PATH = original
                web._ALLOCATION_STAMP = stamp

    def test_missing_file_is_reported_not_faked(self):
        import app as web

        original, stamp = web.ALLOCATION_PATH, web._ALLOCATION_STAMP
        try:
            web.ALLOCATION_PATH = Path("does-not-exist.json")
            web._ALLOCATION_STAMP = 0.0
            payload = web.allocation_payload()
            self.assertFalse(payload["available"])
            self.assertIn("etf_allocation", payload["hint"])
        finally:
            web.ALLOCATION_PATH = original
            web._ALLOCATION_STAMP = stamp


class DeprecationTest(unittest.TestCase):
    def test_crossasset_delegates_and_warns(self):
        import crossasset as ca

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stats = ca.nav_metrics([{"date": f"2020-01-{i:02d}", "nav": 1 + i * 0.01}
                                    for i in range(1, 21)])
        self.assertTrue(any(item.category is DeprecationWarning for item in caught))
        self.assertGreater(stats["sharpe"], 0)
        self.assertIs(ca.UNIVERSE, E.UNIVERSE)


if __name__ == "__main__":
    unittest.main()
