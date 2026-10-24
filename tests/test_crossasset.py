"""跨资产组合的不变量：权重口径、现金腿、过期数据剔除、指标自洽。"""

import datetime
import math
import unittest

import crossasset as ca


def make_history(symbol, *, count=420, drift=0.0004, wave=0.006, trend=0.0, start="2019-01-02"):
    rows = []
    price = 1.0
    day = datetime.date.fromisoformat(start)
    for index in range(count):
        price = max(0.05, price * (1 + drift + trend * math.sin(index / 40) + wave * math.sin(index / 3)))
        rows.append({
            "date": day.isoformat(),
            "open": price * 0.999, "high": price * 1.004, "low": price * 0.996,
            "close": price, "volume": 1_000_000, "amount": 1_000_000,
            "pct_change": 0.0, "turnover": 1.0,
        })
        day += datetime.timedelta(days=1)
    return {"symbol": symbol, "name": symbol, "asset_type": "etf", "data_source": "test", "rows": rows}


def make_panel():
    panel = {
        "510300": make_history("510300", drift=0.0005),
        "510500": make_history("510500", drift=0.0004),
        "511010": make_history("511010", drift=0.0001, wave=0.0008),
        "518880": make_history("518880", drift=0.0003, wave=0.004),
        "510170": make_history("510170", drift=0.0002, wave=0.005),
        "513100": make_history("513100", drift=0.0006),
        "159920": make_history("159920", drift=0.0002, wave=0.007),
        "511880": make_history("511880", drift=0.00008, wave=0.0001),
    }
    return panel


class PanelTests(unittest.TestCase):
    def test_stale_symbols_are_excluded(self):
        panel = make_panel()
        panel["510170"] = {**panel["510170"], "rows": panel["510170"]["rows"][:120]}
        stale = ca.stale_symbols(panel)
        self.assertIn("510170", stale)
        self.assertNotIn("510300", stale)

    def test_target_weights_leave_room_for_cash(self):
        panel = make_panel()
        calendar = [row["date"] for row in panel["510300"]["rows"]][300:]
        target = ca.build_target(panel, calendar)
        weights = target(calendar[-1], list(panel))
        self.assertTrue(weights)
        self.assertLessEqual(sum(weights.values()), 1.000001)
        self.assertIn(ca.CASH, weights)          # 余额进货币 ETF，不假设 0 收益
        for symbol in weights:
            self.assertIn(symbol, list(ca.UNIVERSE) + [ca.CASH])

    def test_throttle_and_overlay_scale_exposure(self):
        panel = make_panel()
        calendar = [row["date"] for row in panel["510300"]["rows"]][300:]
        base = ca.build_target(panel, calendar)(calendar[-1], list(panel))
        halved = ca.build_target(
            panel, calendar, throttle_by_date={calendar[-1]: 0.5}
        )(calendar[-1], list(panel))
        invested = lambda w: sum(v for k, v in w.items() if k != ca.CASH)
        self.assertAlmostEqual(invested(halved), invested(base) * 0.5, places=6)

    def test_metrics_are_self_consistent(self):
        nav = [{"date": f"2020-01-{i:02d}", "nav": 1.0 + i * 0.01} for i in range(1, 21)]
        stats = ca.nav_metrics(nav)
        self.assertAlmostEqual(stats["total_return"], (1.20 / 1.01 - 1) * 100, places=1)
        self.assertEqual(stats["max_drawdown"], 0.0)
        self.assertGreater(stats["sharpe"], 0)


class BacktestTests(unittest.TestCase):
    def test_run_returns_full_calendar(self):
        panel = make_panel()
        result = ca.run(panel, rebalance=21)
        self.assertGreater(len(result["nav"]), 300)
        self.assertEqual(result["nav"][0]["date"], panel["510300"]["rows"][0]["date"])
        self.assertIn("sharpe", result["metrics"])
        self.assertIn("excluded", result)
        self.assertGreaterEqual(result["throttle_days"], 0)

    def test_equal_weight_uses_active_universe(self):
        panel = make_panel()
        result = ca.equal_weight(panel, rebalance=21)
        self.assertGreater(len(result["nav"]), 300)
        self.assertGreater(result["metrics"]["cagr"], -100)


if __name__ == "__main__":
    unittest.main()
