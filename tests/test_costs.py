"""成本口径的单一来源守卫。

这些断言看起来琐碎，但它们守的是一件具体的事：**改一次费率只需要改一个文件**。
上一版费率散在四个地方（`xsec` 的 bp 分量、`score_study` 与 `etf_allocation` 的字面量、
`execution.simulate_t1` 的默认参数），外加 `factor_study` 把 `(12.6 + 7.6)` 写进公式。
只要有一处漏改，同一份报告里就会出现两个成本口径，而"净 Sharpe"差 0.03 时没人查得出来。
"""

from __future__ import annotations

import ast
import inspect
import unittest
from pathlib import Path

import costs

ROOT = Path(__file__).resolve().parent.parent

# 这些模块历史上各自持有成本常量；现在必须是同一个来源。
CONST_HOLDERS = ("xsec.py", "score_study.py", "etf_allocation.py", "crossasset.py")
LITERALS = {0.00076, 0.00126, 0.00075}


class StockCostTest(unittest.TestCase):
    def test_stock_buy_is_7_6bp(self) -> None:
        self.assertAlmostEqual(costs.STOCK_COST.buy_bps, 7.6, places=6)
        self.assertEqual(costs.STOCK_COST.buy, 0.00076)

    def test_stock_sell_is_12_6bp_and_includes_stamp_duty(self) -> None:
        # 卖出比买入多的正好是印花税 5bp——这是 A 股个股与 ETF 的关键差异，
        # 用差值断言而不是写死 12.6，能同时锁住"差额从哪来"。
        self.assertAlmostEqual(costs.STOCK_COST.sell_bps, 12.6, places=6)
        self.assertAlmostEqual(
            costs.STOCK_COST.sell_bps - costs.STOCK_COST.buy_bps, costs.STAMP_DUTY_BPS, places=6
        )

    def test_round_trip_is_sum_of_both_sides(self) -> None:
        self.assertAlmostEqual(costs.STOCK_COST.round_trip_bps, 20.2, places=6)

    def test_label_matches_report_wording(self) -> None:
        self.assertEqual(costs.STOCK_COST.label, "7.6/12.6bp")


class EtfCostTest(unittest.TestCase):
    def test_etf_is_symmetric_7_5bp(self) -> None:
        self.assertAlmostEqual(costs.ETF_COST.buy_bps, 7.5, places=6)
        self.assertAlmostEqual(costs.ETF_COST.sell_bps, 7.5, places=6)
        self.assertEqual(costs.ETF_COST.buy, costs.ETF_COST.sell)

    def test_etf_has_no_stamp_duty(self) -> None:
        # ETF 免印花税，所以双边相同；这条如果被改成 12.6，就说明有人把个股口径套到了 ETF 上。
        self.assertEqual(costs.ETF_COST.components["stamp_duty"], 0.0)
        self.assertLess(costs.ETF_COST.sell_bps, costs.STOCK_COST.sell_bps)

    def test_for_asset_routes_etf_and_index_to_etf_cost(self) -> None:
        self.assertIs(costs.for_asset("etf"), costs.ETF_COST)
        self.assertIs(costs.for_asset("index"), costs.ETF_COST)
        self.assertIs(costs.for_asset("stock"), costs.STOCK_COST)
        self.assertIs(costs.for_asset(None), costs.STOCK_COST)   # 未知 → 保守取个股


class SingleSourceTest(unittest.TestCase):
    """跨模块断言：引用必须指向同一个对象，而不是"数值恰好相等"。"""

    def test_holders_reference_the_same_numbers(self) -> None:
        import crossasset, etf_allocation, score_study, xsec

        self.assertIs(xsec.BUY_COST, costs.STOCK_COST.buy)
        self.assertIs(xsec.SELL_COST, costs.STOCK_COST.sell)
        self.assertIs(score_study.BUY_COST, costs.STOCK_COST.buy)
        self.assertIs(score_study.SELL_COST, costs.STOCK_COST.sell)
        self.assertIs(etf_allocation.BUY_COST, costs.ETF_COST.buy)
        self.assertIs(etf_allocation.SELL_COST, costs.ETF_COST.sell)
        self.assertIs(crossasset.BUY_COST, costs.ETF_COST.buy)

    def test_execution_default_matches_stock_cost(self) -> None:
        import execution

        self.assertIs(execution.DEFAULT_COST, costs.STOCK_COST)
        params = inspect.signature(execution.simulate_t1).parameters
        self.assertEqual(params["cost_buy"].default, costs.STOCK_COST.buy)
        self.assertEqual(params["cost_sell"].default, costs.STOCK_COST.sell)

    def test_factor_study_uses_round_trip_not_literal(self) -> None:
        import factor_study

        # 旧实现是 ``turnover * (12.6 + 7.6) / 100``；换成常量后数值必须一致。
        source = inspect.getsource(factor_study.spread_after_cost)
        self.assertIn("round_trip_bps", source)
        self.assertNotIn("12.6", source)
        # 逐值对照：换手 1.0 时成本 = 20.2bp = 0.202（百分数）
        self.assertAlmostEqual(costs.STOCK_COST.round_trip_bps / 100.0, 0.202, places=9)

    def test_no_cost_literal_in_production_sources(self) -> None:
        """生产/研究模块里不许再出现成本字面量。

        用 AST 扫数值常量而不是 grep 文本：注释里提到 "0.00076" 是说明，不是副本，
        而 grep 会把说明也判成违规，最后只能被迫删掉有价值的注释。
        """
        offenders = []
        for path in sorted(ROOT.glob("*.py")):
            if path.name == "costs.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, float):
                    if any(abs(node.value - literal) < 1e-12 for literal in LITERALS):
                        offenders.append(f"{path.name}:{node.lineno} = {node.value}")
        self.assertEqual(offenders, [], "成本字面量应改为 costs.* 引用：" + "; ".join(offenders))

    def test_report_prose_uses_label(self) -> None:
        """报告文案里的「7.6/12.6bp」应由 costs.label 生成，不再手写。"""
        for name in ("regime_study.py", "wfv2_study.py", "score_study.py"):
            source = (ROOT / name).read_text(encoding="utf-8")
            self.assertIn("costs.STOCK_COST", source, name)
            self.assertNotIn("7.6/12.6bp", source, name)


class CostBehaviourTest(unittest.TestCase):
    def test_scaled_doubles_both_sides(self) -> None:
        doubled = costs.STOCK_COST.scaled(2)
        self.assertAlmostEqual(doubled.buy_bps, 15.2, places=6)
        self.assertAlmostEqual(doubled.sell_bps, 25.2, places=6)
        self.assertIn("压力测试", doubled.note)

    def test_side_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            costs.STOCK_COST.side("short")

    def test_describe_is_report_ready(self) -> None:
        row = costs.describe(costs.ETF_COST)
        self.assertEqual(row["buy_bps"], 7.5)
        self.assertEqual(row["sell_bps"], 7.5)
        self.assertEqual(row["components"]["stamp_duty"], 0.0)


if __name__ == "__main__":
    unittest.main()
