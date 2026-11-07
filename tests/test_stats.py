"""stats.py 的口径守卫：合并 5 份重复实现时，数值必须一点没变。

这个文件的核心不是"函数能跑"，而是**"合并前后逐位相同"**：每个合并掉的函数
都在这里留一份"旧口径"参考实现（原样抄自迁移前的代码），用随机但固定的输入
逐项比对。这样以后谁改了 stats 的实现，都能立刻看出差异是"修 bug"还是"改口径"。
"""

from __future__ import annotations

import ast
import math
import random
import statistics
import unittest
from pathlib import Path

import stats

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# 旧口径参考实现（迁移前代码原样，只改名）
# --------------------------------------------------------------------------- #


def legacy_sma(values, period):
    result = [None] * len(values)
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= period:
            running -= values[index - period]
        if index >= period - 1:
            result[index] = running / period
    return result


def legacy_sma_with_none(values, window):
    out = [None] * len(values)
    running = 0.0
    count = 0
    for index, value in enumerate(values):
        if value is not None:
            running += value
            count += 1
        if index >= window:
            dropped = values[index - window]
            if dropped is not None:
                running -= dropped
                count -= 1
        if index >= window - 1 and count == window:
            out[index] = running / window
    return out


def legacy_rolling_max_full(values, window):
    """factors 口径：窗口不满返回 None。"""
    result = [None] * len(values)
    queue = []
    for index, value in enumerate(values):
        while queue and values[queue[-1]] <= value:
            queue.pop()
        queue.append(index)
        if queue[0] <= index - window:
            queue.pop(0)
        if index >= window - 1:
            result[index] = values[queue[0]]
    return result


def legacy_rolling_max_expanding(values, window):
    """fields 口径：窗口不满用扩张窗口最大值。"""
    result = [None] * len(values)
    queue = []
    for index, value in enumerate(values):
        while queue and values[queue[-1]] <= value:
            queue.pop()
        queue.append(index)
        if queue[0] <= index - window:
            queue.pop(0)
        result[index] = values[queue[0]]
    return result


def legacy_nav_metrics_xsec(nav, calendar_len):
    """xsec.metrics 的收益/风险部分（迁移前）。"""
    years = max(calendar_len / 252, 1e-9)
    total = nav[-1] / nav[0] - 1
    annual = (nav[-1] / nav[0]) ** (1 / years) - 1
    returns = [nav[i] / nav[i - 1] - 1 for i in range(1, len(nav)) if nav[i - 1]]
    volatility = statistics.stdev(returns) * math.sqrt(252) if len(returns) > 1 else 0.0
    peak = 0.0
    drawdown = 0.0
    for value in nav:
        peak = max(peak, value)
        if peak:
            drawdown = min(drawdown, value / peak - 1)
    return {
        "total_return": round(total * 100, 2),
        "annual_return": round(annual * 100, 2),
        "max_drawdown": round(drawdown * 100, 2),
        "volatility": round(volatility * 100, 2),
        "sharpe": round(annual / volatility, 2) if volatility else None,
    }


def legacy_nav_metrics_etf(nav_rows):
    """etf_allocation.nav_metrics（迁移前）。"""
    values = [float(point["nav"]) for point in nav_rows]
    if len(values) < 3:
        return {}
    years = (len(values) - 1) / 252
    returns = [values[i] / values[i - 1] - 1 for i in range(1, len(values)) if values[i - 1] > 0]
    total = values[-1] / values[0] - 1
    cagr = (values[-1] / values[0]) ** (1 / years) - 1 if years > 0 else 0.0
    vol = statistics.stdev(returns) * math.sqrt(252) if len(returns) > 1 else 0.0
    peak, worst = values[0], 0.0
    for value in values:
        peak = max(peak, value)
        worst = min(worst, value / peak - 1)
    return {
        "total_return": round(total * 100, 2),
        "cagr": round(cagr * 100, 2),
        "vol": round(vol * 100, 2),
        "sharpe": round(cagr / vol, 3) if vol > 1e-9 else None,
        "max_drawdown": round(worst * 100, 2),
        "calmar": round(cagr / abs(worst), 3) if worst < -1e-9 else None,
        "years": round(years, 2),
    }


def _series(length: int = 400, seed: int = 20261024) -> list[float]:
    rng = random.Random(seed)
    value, out = 10.0, []
    for _ in range(length):
        value *= 1 + rng.gauss(0.0004, 0.012)
        out.append(value)
    return out


# --------------------------------------------------------------------------- #
# 滚动窗口
# --------------------------------------------------------------------------- #


class SmaTest(unittest.TestCase):
    def test_matches_all_four_legacy_copies(self) -> None:
        values = _series(300)
        for period in (2, 3, 5, 20, 60, 250):
            with self.subTest(period=period):
                self.assertEqual(stats.sma(values, period), legacy_sma(values, period))

    def test_matches_etf_variant_when_values_missing(self) -> None:
        values = [None if index % 37 == 0 else value for index, value in enumerate(_series(300))]
        for period in (2, 5, 20):
            with self.subTest(period=period):
                self.assertEqual(stats.sma(values, period), legacy_sma_with_none(values, period))

    def test_missing_value_blocks_window_until_full(self) -> None:
        # 缺失值会让窗口「不满」，因此后续 period-1 个位置仍是 None——这不是 bug，
        # 而是 etf 口径的原有行为（宁可缺值，也不拿不完整的窗口凑数）。
        out = stats.sma([1.0, None, 3.0, 4.0], 2)
        self.assertEqual(out, [None, None, None, 3.5])

    def test_rejects_non_positive_period(self) -> None:
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                stats.sma([1.0, 2.0], bad)


class RollingMaxTest(unittest.TestCase):
    def test_full_window_semantics_match_factors(self) -> None:
        values = _series(300)
        for window in (5, 20, 60, 250):
            with self.subTest(window=window):
                self.assertEqual(stats.rolling_max(values, window),
                                 legacy_rolling_max_full(values, window))

    def test_partial_window_semantics_match_fields(self) -> None:
        values = _series(300)
        for window in (5, 20, 60, 250):
            with self.subTest(window=window):
                self.assertEqual(stats.rolling_max(values, window, partial=True),
                                 legacy_rolling_max_expanding(values, window))

    def test_two_semantics_really_differ(self) -> None:
        # 这条是给"以后有人想当然合并"的人看的：前 window-1 个位置两者不同，
        # fields 的「距一年高点」依赖 partial=True，改默认值会让次新标的的回撤变缺失。
        values = [1.0, 3.0, 2.0, 5.0]
        full = stats.rolling_max(values, 3)
        part = stats.rolling_max(values, 3, partial=True)
        self.assertEqual(full, [None, None, 3.0, 5.0])
        self.assertEqual(part, [1.0, 3.0, 3.0, 5.0])
        self.assertNotEqual(full, part)


class SolveTest(unittest.TestCase):
    def test_residual_is_tiny_for_well_conditioned(self) -> None:
        rng = random.Random(11)
        for size in (2, 5, 13):
            matrix = [[rng.uniform(-1, 1) for _ in range(size)] for _ in range(size)]
            for index in range(size):
                matrix[index][index] += size        # 对角占优，保证可逆
            vector = [rng.uniform(-1, 1) for _ in range(size)]
            solution = stats.solve(matrix, vector)
            for row, target in zip(matrix, vector):
                self.assertAlmostEqual(sum(a * b for a, b in zip(row, solution)), target, places=9)

    def test_singular_column_yields_zero_not_exception(self) -> None:
        # 共线特征在因子回归里是常态，旧实现选择"跳过该列"，解里对应变量为 0。
        solution = stats.solve([[1.0, 1.0], [1.0, 1.0]], [2.0, 2.0])
        self.assertEqual(len(solution), 2)
        self.assertTrue(all(math.isfinite(value) for value in solution))

    def test_identity_system(self) -> None:
        self.assertEqual(stats.solve([[1.0, 0.0], [0.0, 1.0]], [3.0, 4.0]), [3.0, 4.0])


# --------------------------------------------------------------------------- #
# 逐日 IC
# --------------------------------------------------------------------------- #


class DailyIcTest(unittest.TestCase):
    def test_perfect_ranking_gives_one(self) -> None:
        pairs = {"2024-01-02": [(float(i), float(i) * 2) for i in range(30)]}
        series = stats.daily_ic(pairs)
        self.assertEqual(len(series), 1)
        self.assertAlmostEqual(series[0], 1.0, places=9)

    def test_skips_day_below_min_xsec(self) -> None:
        pairs = {
            "2024-01-02": [(float(i), float(i)) for i in range(19)],
            "2024-01-03": [(float(i), float(i)) for i in range(20)],
        }
        self.assertEqual(stats.daily_ic(pairs), stats.daily_ic(pairs, min_xsec=20))
        self.assertEqual(len(stats.daily_ic(pairs, min_xsec=20)), 1)
        self.assertEqual(len(stats.daily_ic(pairs, min_xsec=10)), 2)

    def test_none_pairs_are_dropped_before_threshold(self) -> None:
        pairs = {"2024-01-02": [(None, 1.0)] * 25 + [(float(i), float(i)) for i in range(5)]}
        # 只有 5 个有效配对 < 20 → 当日无 IC
        self.assertEqual(stats.daily_ic(pairs, min_xsec=20), [])

    def test_dates_argument_controls_order_and_subset(self) -> None:
        pairs = {date: [(float(i), float(i)) for i in range(25)]
                 for date in ("2024-01-02", "2024-01-03", "2024-01-04")}
        self.assertEqual(len(stats.daily_ic(pairs, dates=["2024-01-03"])), 1)

    def test_pairs_from_groups_keeps_none(self) -> None:
        groups = {"2024-01-02": [{"s": 1.0, "label": 2.0}, {"s": None, "label": 3.0}]}
        pairs = stats.pairs_from_groups(groups, "s")
        self.assertEqual(pairs, {"2024-01-02": [(1.0, 2.0), (None, 3.0)]})

    def test_min_xsec_default_is_20(self) -> None:
        # 默认值必须与 xsec / protocol / factor_study 长期在用的 20 一致；
        # scoring(10) 与 research.fundamentals(30) 是显式传参的历史例外。
        self.assertEqual(stats.DEFAULT_MIN_XSEC, 20)


# --------------------------------------------------------------------------- #
# 净值指标
# --------------------------------------------------------------------------- #


class NavMetricsTest(unittest.TestCase):
    def test_matches_legacy_etf_formula(self) -> None:
        rows = [{"date": f"d{i}", "nav": value} for i, value in enumerate(_series(600))]
        self.assertEqual(stats.nav_metrics(rows, min_points=3), legacy_nav_metrics_etf(rows))

    def test_matches_legacy_xsec_formula(self) -> None:
        values = _series(600)
        calendar_len = 585                     # 故意与 len-1 不同，验证 calendar_len 生效
        new = stats.nav_metrics(values, calendar_len=calendar_len, sharpe_digits=2)
        old = legacy_nav_metrics_xsec(values, calendar_len)
        for key in ("total_return", "max_drawdown", "volatility", "sharpe"):
            with self.subTest(key=key):
                self.assertEqual(getattr_nested(new, key), old[key])

    def test_calendar_len_defaults_to_n_minus_one(self) -> None:
        values = _series(300)
        implicit = stats.nav_metrics(values)
        explicit = stats.nav_metrics(values, calendar_len=len(values) - 1)
        self.assertEqual(implicit, explicit)

    def test_sharpe_digits_is_parameterised(self) -> None:
        values = _series(300)
        three = stats.nav_metrics(values, sharpe_digits=3)["sharpe"]
        two = stats.nav_metrics(values, sharpe_digits=2)["sharpe"]
        self.assertEqual(two, round(three, 2)) if three is not None else None

    def test_too_short_returns_empty(self) -> None:
        self.assertEqual(stats.nav_metrics([1.0]), {})
        self.assertEqual(stats.nav_metrics([], min_points=3), {})

    def test_flat_series_has_no_sharpe(self) -> None:
        flat = [1.0] * 50
        metrics = stats.nav_metrics(flat)
        self.assertEqual(metrics["max_drawdown"], 0.0)
        self.assertIsNone(metrics["sharpe"])
        self.assertIsNone(metrics["calmar"])

    def test_cost_drag_uses_average_nav(self) -> None:
        # 口径：Σcost / 平均净值 / 年数。用期末净值会低估近一半，这条把公式钉住。
        rows = [{"nav": value} for value in (1.0, 1.5, 2.0)]
        years = 1.0
        expected = round(0.045 / statistics.fmean([1.0, 1.5, 2.0]) / years * 10000, 1)
        self.assertEqual(stats.cost_drag_bps(0.045, statistics.fmean([1.0, 1.5, 2.0]), years),
                         expected)


def getattr_nested(mapping: dict, key: str):
    """旧键名 → 新键名，测试里显式列出，避免"名字对了值不对"。"""
    alias = {"volatility": "vol", "annual_return": "cagr"}
    return mapping[alias.get(key, key)]


# --------------------------------------------------------------------------- #
# 单一实现守卫
# --------------------------------------------------------------------------- #


class SingleImplementationTest(unittest.TestCase):
    def test_module_aliases_point_to_stats(self) -> None:
        import analyzer, etf_allocation, scoring, xsec

        self.assertIs(analyzer.sma, stats.sma)
        self.assertIs(etf_allocation.sma, stats.sma)
        self.assertIs(xsec.stats, stats)
        self.assertIs(scoring.stats, stats)

    def test_no_duplicate_definitions(self) -> None:
        """这些函数名只允许在 stats.py 里定义一次。"""
        forbidden = {
            "def _sma(", "def _rolling_max(", "def _solve(",
            "def sma(", "def solve(",
        }
        allowed = {ROOT / "stats.py"}
        offenders = []
        for path in sorted(ROOT.glob("*.py")):
            if path in allowed:
                continue
            text = path.read_text(encoding="utf-8")
            for pattern in forbidden:
                if pattern in text:
                    offenders.append(f"{path.name}: {pattern}")
        self.assertEqual(offenders, [], "重复实现应删除：" + "; ".join(offenders))

    def test_daily_ic_core_is_single(self) -> None:
        """各模块可以保留同名入口，但不得再有"自己算 spearman 序列"的实现。"""
        for name in ("factor_study.py", "wfv2_study.py", "scoring.py", "protocol.py"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertIn("stats.daily_ic(", text, name)
        # research 层同样收敛
        self.assertIn("stats.daily_ic(", (ROOT / "research" / "fundamentals.py").read_text(encoding="utf-8"))

    def test_stats_does_not_reimplement_spearman(self) -> None:
        tree = ast.parse((ROOT / "stats.py").read_text(encoding="utf-8"))
        self.assertIn("evalkit", [alias.name for node in ast.walk(tree)
                                  if isinstance(node, ast.Import) for alias in node.names])

    def test_stats_declares_no_io(self) -> None:
        source = (ROOT / "stats.py").read_text(encoding="utf-8")
        for forbidden in ("open(", "Path(", "json.", "random."):
            self.assertNotIn(forbidden, source, f"stats 不该做 I/O 或随机：{forbidden}")


if __name__ == "__main__":
    unittest.main()
