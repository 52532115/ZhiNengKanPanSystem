"""T+1 执行层的测试：成交时点、涨跌停/停牌过滤、T+1 卖出、成本与换手。"""

from __future__ import annotations

import unittest

import execution as ex


def bar(date: str, open_price: float, close: float, high: float | None = None,
        low: float | None = None, volume: float = 1000.0, turnover: float = 1.0) -> dict:
    return {
        "date": date,
        "open": open_price,
        "close": close,
        "high": high if high is not None else max(open_price, close),
        "low": low if low is not None else min(open_price, close),
        "volume": volume,
        "amount": volume * close,
        "turnover": turnover,
        "pct_change": 0.0,
        "change": 0.0,
    }


def history(symbol: str, rows: list[dict], name: str = "测试") -> dict:
    return {"symbol": symbol, "name": name, "asset_type": "stock", "rows": rows}


class LimitRuleTest(unittest.TestCase):
    def test_limit_ratio_by_board(self) -> None:
        self.assertEqual(ex.limit_ratio("600519"), 0.10)
        self.assertEqual(ex.limit_ratio("000001"), 0.10)
        self.assertEqual(ex.limit_ratio("300750"), 0.20)
        self.assertEqual(ex.limit_ratio("688111"), 0.20)
        self.assertEqual(ex.limit_ratio("830799"), 0.30)
        self.assertEqual(ex.limit_ratio("920001"), 0.30)

    def test_st_gets_five_percent(self) -> None:
        self.assertEqual(ex.limit_ratio("600519", "*ST 某某"), 0.05)
        self.assertEqual(ex.limit_ratio("600519", "ST 某某"), 0.05)

    def test_limit_prices_round_to_cent(self) -> None:
        self.assertEqual(ex.limit_prices(10.0, 0.10), (11.0, 9.0))
        up, down = ex.limit_prices(10.05, 0.10)
        self.assertAlmostEqual(up, 11.06, places=2)     # 11.055 → 11.06
        self.assertAlmostEqual(down, 9.05, places=2)    # 9.045 → 9.05

    def test_limit_prices_guard_invalid_previous_close(self) -> None:
        self.assertEqual(ex.limit_prices(0.0, 0.1), (0.0, 0.0))


class TradabilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            bar("2024-01-02", 10.0, 10.0),
            bar("2024-01-03", 11.0, 11.0, high=11.0, low=11.0),      # 一字涨停
            bar("2024-01-04", 11.0, 11.0, volume=0.0),               # 停牌
            bar("2024-01-05", 10.0, 10.2),
        ]
        self.flags = ex.tradability([history("600000", self.rows)])

    def test_limit_up_open_is_flagged(self) -> None:
        self.assertEqual(self.flags["600000"]["2024-01-03"], "limit_up")

    def test_suspended_day_is_flagged(self) -> None:
        self.assertEqual(self.flags["600000"]["2024-01-04"], "suspended")

    def test_normal_day_is_clean(self) -> None:
        self.assertEqual(self.flags["600000"]["2024-01-02"], "")
        self.assertEqual(self.flags["600000"]["2024-01-05"], "")

    def test_limit_down_detected(self) -> None:
        rows = [bar("2024-01-02", 10.0, 10.0), bar("2024-01-03", 9.0, 9.0, high=9.0, low=9.0)]
        flags = ex.tradability([history("600000", rows)])
        self.assertEqual(flags["600000"]["2024-01-03"], "limit_down")

    def test_execution_summary_counts_blocked_days(self) -> None:
        summary = ex.execution_summary(self.flags)
        self.assertEqual(summary["blocked_symbol_days"].get("limit_up"), 1)
        self.assertEqual(summary["blocked_symbol_days"].get("suspended"), 1)


class SimulateT1Test(unittest.TestCase):
    def _bars(self) -> dict:
        rows = [
            bar("d1", 10.0, 10.0),
            bar("d2", 10.0, 10.5),
            bar("d3", 11.0, 11.5),
            bar("d4", 12.0, 12.5),
            bar("d5", 13.0, 13.0),
        ]
        return ex.build_bars([history("600000", rows)])

    def test_signal_is_filled_at_next_open(self) -> None:
        bars = self._bars()
        result = ex.simulate_t1(
            ["d1", "d2", "d3", "d4", "d5"], bars,
            lambda date, available: ["600000"],
            rebalance_every=10, cost_buy=0.0, cost_sell=0.0, max_turnover=1.0,
        )
        self.assertEqual(result["rebalances"][0]["signal_date"], "d1")
        self.assertEqual(result["rebalances"][0]["date"], "d2")
        # 首日没有仓位，净值不动；次日按开盘价建仓，收盘价估值
        self.assertAlmostEqual(result["nav"][0]["nav"], 1.0, places=6)
        self.assertAlmostEqual(result["nav"][1]["nav"], 10.5 / 10.0, places=6)
        self.assertAlmostEqual(result["nav"][2]["nav"], 11.5 / 10.0, places=6)

    def test_limit_up_open_blocks_the_buy(self) -> None:
        rows = [
            bar("d1", 10.0, 10.0),
            bar("d2", 11.0, 11.0, high=11.0, low=11.0),   # 一字板，开盘买不进
            bar("d3", 11.0, 12.0),
        ]
        bars = ex.build_bars([history("600000", rows)])
        flags = ex.tradability([history("600000", rows)])
        result = ex.simulate_t1(
            ["d1", "d2", "d3"], bars,
            lambda date, available: ["600000"],
            flags=flags, rebalance_every=10, cost_buy=0.0, cost_sell=0.0, max_turnover=1.0,
        )
        self.assertEqual(result["blocked"]["buy_limit_up"], 1)
        self.assertAlmostEqual(result["final_nav"], 1.0, places=6)
        self.assertEqual(result["holdings"][0]["symbols"], [])

    def test_limit_down_open_blocks_the_sell(self) -> None:
        rows = [
            bar("d1", 10.0, 10.0),
            bar("d2", 10.0, 10.0),
            bar("d3", 10.0, 10.0),
            bar("d4", 9.0, 9.0, high=9.0, low=9.0),       # 一字跌停，卖不出
            bar("d5", 9.0, 9.0),
        ]
        bars = ex.build_bars([history("600000", rows)])
        flags = ex.tradability([history("600000", rows)])
        picks = iter([["600000"], ["600000"], [], []])

        def pick(date: str, available: list[str]) -> list[str]:
            return next(picks, [])

        result = ex.simulate_t1(
            ["d1", "d2", "d3", "d4", "d5"], bars, pick,
            flags=flags, rebalance_every=1, cost_buy=0.0, cost_sell=0.0, max_turnover=1.0,
        )
        self.assertGreaterEqual(result["blocked"]["sell_limit_down"], 1)
        # 卖不出去说明仓位还在，净值跟着跌停价一起跌
        self.assertAlmostEqual(result["final_nav"], 0.9, places=6)

    def test_cost_is_charged_on_both_sides(self) -> None:
        bars = self._bars()

        def pick(date: str, available: list[str]) -> list[str]:
            return ["600000"] if date < "d3" else []

        free = ex.simulate_t1(["d1", "d2", "d3", "d4", "d5"], bars, pick,
                              rebalance_every=1, cost_buy=0.0, cost_sell=0.0, max_turnover=1.0)
        costly = ex.simulate_t1(["d1", "d2", "d3", "d4", "d5"], bars, pick,
                                rebalance_every=1, cost_buy=0.001, cost_sell=0.002, max_turnover=1.0)
        self.assertAlmostEqual(free["total_cost"], 0.0, places=12)
        self.assertGreater(costly["total_cost"], 0.0)
        self.assertLess(costly["final_nav"], free["final_nav"])

    def test_turnover_cap_keeps_part_of_the_old_position(self) -> None:
        rows = [bar(f"d{i}", 10.0, 10.0) for i in range(1, 6)]
        bars = ex.build_bars([
            history("600000", rows),
            history("600001", [bar(f"d{i}", 10.0, 10.0) for i in range(1, 6)]),
        ])
        picks = iter([["600000"], ["600001"], ["600001"]])

        def pick(date: str, available: list[str]) -> list[str]:
            return next(picks, ["600001"])

        result = ex.simulate_t1(
            ["d1", "d2", "d3", "d4", "d5"], bars, pick,
            rebalance_every=1, cost_buy=0.0, cost_sell=0.0, max_turnover=0.1,
        )
        rotated = [item for item in result["holdings"] if item["date"] == "d3"]
        self.assertTrue(rotated)
        self.assertIn("600000", rotated[0]["symbols"])   # 只换了 10%，老仓还在
        self.assertTrue(result["rebalances"][1]["capped"])

    def test_exposure_leaves_cash(self) -> None:
        bars = self._bars()
        result = ex.simulate_t1(
            ["d1", "d2", "d3", "d4", "d5"], bars,
            lambda date, available: ["600000"],
            rebalance_every=10, cost_buy=0.0, cost_sell=0.0, max_turnover=1.0,
            exposure=lambda date: 0.5,
        )
        self.assertAlmostEqual(result["rebalances"][0]["gross"], 0.5, places=6)
        # 半仓：涨 5% 时净值只涨 2.5%
        self.assertAlmostEqual(result["nav"][1]["nav"], 1.025, places=6)
        self.assertGreater(result["cash_ratio_mean"], 0.4)

    def test_results_are_reproducible(self) -> None:
        bars = self._bars()
        first = ex.simulate_t1(["d1", "d2", "d3", "d4", "d5"], bars,
                              lambda date, available: ["600000"], rebalance_every=2)
        second = ex.simulate_t1(["d1", "d2", "d3", "d4", "d5"], bars,
                               lambda date, available: ["600000"], rebalance_every=2)
        self.assertEqual([p["nav"] for p in first["nav"]], [p["nav"] for p in second["nav"]])


class BoardMixTest(unittest.TestCase):
    def test_board_label(self) -> None:
        self.assertIn("双创", ex._board_of("300750"))
        self.assertIn("北交所", ex._board_of("830799"))
        self.assertIn("主板", ex._board_of("600519"))
        self.assertEqual(ex._board_of("600519", "ST 某某"), "ST")


if __name__ == "__main__":
    unittest.main()
