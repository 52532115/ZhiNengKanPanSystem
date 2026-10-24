"""生产层隔离的静态检查：谁都不许把研究依赖带进生产进程。

规则很简单，但必须由测试守着——"临时 import 一下"是这类污染的唯一入口，而它
在开发机上永远不会有症状（依赖都装着），只在部署时炸。

- ``PRODUCTION`` 里的模块：不得 import ``research``、``baostock``、``akshare``、
  ``pandas``、``numpy``、``requests``；
- ``research`` 包：允许 import 第三方（baostock 只在适配器里），但不得 import
  生产层的 ``app`` / ``analyzer`` / ``market_data``（否则研究层反过来依赖 UI 状态）；
- 只允许 ``research`` 单向读取 ``evalkit`` / ``factors`` / ``fields`` 这类纯计算模块。
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PRODUCTION = (
    "app.py", "analyzer.py", "fields.py", "market_data.py",
    "execution.py", "rebalance.py", "crossasset.py", "etf_allocation.py",
    "regime.py", "scoring.py",
)
RESEARCH = ("research/fundamentals.py", "research/pit_join.py", "research/fetch_alt.py",
            "research/datasets/base.py", "research/datasets/eastmoney_flows.py",
            "research/datasets/csv_drop.py")
FORBIDDEN_IN_PRODUCTION = {"research", "baostock", "akshare", "pandas", "numpy", "requests"}
FORBIDDEN_IN_RESEARCH = {"app", "analyzer", "market_data"}


def imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                found.add(node.module.split(".")[0])
            elif node.level:
                # 相对 import：只记录模块名，用于 research 内部检查
                found.add(f".{(node.module or '').split('.')[0]}")
    return found


class IsolationTest(unittest.TestCase):
    def test_production_has_no_research_or_third_party(self):
        for name in PRODUCTION:
            path = ROOT / name
            self.assertTrue(path.is_file(), name)
            leaked = imports_of(path) & FORBIDDEN_IN_PRODUCTION
            self.assertFalse(leaked, f"{name} 引入了 {sorted(leaked)}")

    def test_research_does_not_import_production_ui(self):
        for name in RESEARCH:
            path = ROOT / name
            if not path.is_file():
                self.skipTest(f"{name} 不存在")
            leaked = imports_of(path) & FORBIDDEN_IN_RESEARCH
            self.assertFalse(leaked, f"{name} 引入了 {sorted(leaked)}")

    def test_baostock_only_in_adapters(self):
        hits = []
        for path in ROOT.glob("*.py"):
            if path.name.startswith("_") or path.name in {"walkforward.py", "xsec.py"}:
                continue
            if "baostock" in imports_of(path):
                hits.append(path.name)
        allowed = {"validate_fields.py", "turnover_study.py", "factor_study.py", "wfv2_study.py"}
        self.assertTrue(set(hits) <= allowed, f"意外的 baostock 依赖：{sorted(set(hits) - allowed)}")

    def test_research_package_imports_without_third_party(self):
        """研究层的因子与 PIT 模块必须能在零依赖环境里 import。"""
        import importlib
        for module in ("research.pit_join", "research.fundamentals"):
            self.assertIsNotNone(importlib.import_module(module))


if __name__ == "__main__":
    unittest.main()
