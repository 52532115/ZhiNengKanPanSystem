"""确定性数值内核：滚动窗口、线性方程、逐日 IC、净值指标。

边界（很重要，越界就会长回重复实现）：

- **只做确定性计算**：给定输入就得到同一个数，不含随机数、不含 p 值、不含多重检验。
  带推断的部分（块自助区间、BH-FDR、通货紧缩夏普、仅含岭项的 OLS）留在 ``evalkit``。
  ``daily_ic`` 里算秩相关必须调 ``evalkit.spearman``，不许出现第二份秩相关实现。
- **不做 I/O**：不读缓存、不读数据集、不写报告。数据装配属于各 study。
- 不 import 任何 study、不 import ``research``。

为什么需要这个模块：``sma`` 曾有 4 份（``analyzer.sma`` / ``factors._sma`` /
``fields._sma`` / ``etf_allocation.sma``）、``rolling_max`` 与 ``solve`` 各 2 份、
``daily_ic`` 5 份、``nav_metrics`` 2 份。其中 ``rolling_max`` 两份**语义不同**
（一份前 ``window-1`` 个返回 ``None``，一份返回扩张窗口最大值），``daily_ic``
的最小横截面门槛是 10/20/30 三档——同一指标在不同报告里根本不可比。
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable, Mapping, Sequence

import evalkit

# 当日横截面最少多少只才构成一个有效样本。统一默认值 = 长期在用的 20；
# 需要偏离时必须显式传参并在报告里说明（``scoring`` 用 10、``research.fundamentals``
# 用 30 的历史口径暂时保留，属于待收敛项而不是本模块的默认真理）。
DEFAULT_MIN_XSEC = 20


# --------------------------------------------------------------------------- #
# 滚动窗口
# --------------------------------------------------------------------------- #


def sma(values: Sequence[float | None], period: int) -> list[float | None]:
    """简单移动平均。允许 ``None``（视为缺失，不计入分母）。

    四个副本合并成一个：``analyzer.sma`` / ``factors._sma`` / ``fields._sma`` 逐字节相同，
    ``etf_allocation.sma`` 多一层 ``None`` 处理。无缺失值时四者输出完全一致；
    有缺失值时只有这一版不炸。
    """
    if period <= 0:
        raise ValueError(f"period 必须为正，收到 {period}")
    out: list[float | None] = [None] * len(values)
    running = 0.0
    count = 0
    for index, value in enumerate(values):
        if value is not None:
            running += value
            count += 1
        if index >= period:
            dropped = values[index - period]
            if dropped is not None:
                running -= dropped
                count -= 1
        if index >= period - 1 and count == period:
            out[index] = running / period
    return out


def rolling_max(
    values: Sequence[float],
    window: int,
    *,
    partial: bool = False,
) -> list[float | None]:
    """因果滚动最大值（含当日），单调队列 O(n)。

    ``partial`` 决定窗口不满时怎么算——这正是原来两份实现的差异，必须显式选：

    - ``partial=False``（默认，``factors`` 语义）：前 ``window-1`` 个返回 ``None``；
    - ``partial=True``（``fields`` 语义）：返回**扩张窗口**最大值，
      即"上市不足一年时用已有历史算高点"。``fields`` 的「距一年高点」依赖这个行为，
      若改成 ``False``，次新标的的回撤会从"有值"变成"缺失"。
    """
    if window <= 0:
        raise ValueError(f"window 必须为正，收到 {window}")
    out: list[float | None] = [None] * len(values)
    queue: list[int] = []
    for index, value in enumerate(values):
        while queue and values[queue[-1]] <= value:
            queue.pop()
        queue.append(index)
        if queue[0] <= index - window:
            queue.pop(0)
        if partial or index >= window - 1:
            out[index] = values[queue[0]]
    return out


# --------------------------------------------------------------------------- #
# 线性方程
# --------------------------------------------------------------------------- #


def solve(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    """高斯-约当消元（部分主元）。

    奇异列（主元 < ``1e-14``）直接跳过，返回该列对应变量为 0 的解——比抛异常更适合
    "特征共线"这种常态。阈值取两份旧实现里**更严的那个**（``scoring`` 的 1e-14），
    并保留除零保护（``xsec`` 那版没有）。
    """
    size = len(vector)
    aug = [list(matrix[row])[:] + [vector[row]] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(aug[r][column]))
        if abs(aug[pivot][column]) < 1e-14:
            continue
        aug[column], aug[pivot] = aug[pivot], aug[column]
        divisor = aug[column][column]
        if abs(divisor) < 1e-16:
            continue
        for c in range(column, size + 1):
            aug[column][c] /= divisor
        for row in range(size):
            if row == column:
                continue
            factor = aug[row][column]
            if not factor:
                continue
            for c in range(column, size + 1):
                aug[row][c] -= factor * aug[column][c]
    return [aug[i][size] for i in range(size)]


# --------------------------------------------------------------------------- #
# 逐日横截面 IC
# --------------------------------------------------------------------------- #


def pairs_from_groups(
    groups: Mapping[str, Iterable[Mapping[str, Any]]],
    score_key: str,
    label_key: str = "label",
) -> dict[str, list[tuple[Any, Any]]]:
    """``{date: [row, ...]}`` → ``{date: [(score, label), ...]}``，缺值保留为 ``None``。

    保留 ``None`` 而不是就地过滤，是为了让 ``daily_ic`` 成为唯一决定"哪些样本算数"
    的地方——否则每个 study 各自过滤一次，门槛就又分叉了。
    """
    return {
        date: [(row.get(score_key), row.get(label_key)) for row in rows]
        for date, rows in groups.items()
    }


def daily_ic(
    pairs_by_date: Mapping[str, Iterable[tuple[Any, Any]]],
    *,
    min_xsec: int = DEFAULT_MIN_XSEC,
    dates: Sequence[str] | None = None,
) -> list[float]:
    """逐日横截面 Spearman IC 序列（缺值跳过，样本不足当日跳过）。

    ``pairs_by_date`` 是 ``{date: [(score, label), ...]}``。之所以用"配对"而不是
    "两个 dict"，是因为历史上五种调用形态里有三种是 dict 对 dict、两种是行列表，
    配对数形态能把五种都表达出来，且不会出现"两个 dict 键不齐"的隐式丢样本。
    """
    series: list[float] = []
    for date in (dates if dates is not None else sorted(pairs_by_date)):
        pairs = [
            (score, label)
            for score, label in pairs_by_date.get(date) or []
            if score is not None and label is not None
        ]
        if len(pairs) < min_xsec:
            continue
        value = evalkit.spearman([p[0] for p in pairs], [p[1] for p in pairs])
        if value is not None:
            series.append(value)
    return series


# --------------------------------------------------------------------------- #
# 净值指标
# --------------------------------------------------------------------------- #


def nav_metrics(
    nav: Sequence[Any],
    *,
    calendar_len: int | None = None,
    sharpe_digits: int = 3,
    min_points: int = 2,
) -> dict[str, Any]:
    """净值曲线 → 收益 / 风险指标（比率与收益率**一律以百分数返回**）。

    ``nav`` 既接受 ``[{"date": ..., "nav": 1.0}, ...]`` 也接受 ``[1.0, 1.01, ...]``：
    两种形态在历史调用里各占一半，统一成一种就要在调用点加适配层，适配层本身
    又会分叉（一处忘了取 ``["nav"]`` 就得到 TypeError）。

    两个历史口径在这里合并，参数化而不是二选一：

    - **年数**：``etf_allocation`` 用 ``(n-1)/252``，``xsec`` 用 ``calendar_len/252``。
      缺省按 ``(n-1)/252``；``xsec`` 显式传 ``calendar_len``（它的 nav 可能短于日历）。
    - **Sharpe 小数位**：``etf`` 给 3 位、``xsec`` 给 2 位。默认 3，``xsec`` 传 2
      以保持既有 JSON 不变（历史报告口径，收敛后再统一）。

    ``calendar_len`` 缺省时取 ``len(nav) - 1``——这正是 ``etf`` 原口径。
    """
    values = [float(point["nav"] if isinstance(point, Mapping) else point) for point in nav]
    if len(values) < min_points:
        return {}
    span = (len(values) - 1) if calendar_len is None else calendar_len
    years = max(span / 252, 1e-9)
    total = values[-1] / values[0] - 1
    cagr = (values[-1] / values[0]) ** (1 / years) - 1
    returns = [values[i] / values[i - 1] - 1 for i in range(1, len(values)) if values[i - 1] > 0]
    vol = statistics.stdev(returns) * math.sqrt(252) if len(returns) > 1 else 0.0
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - 1)
    return {
        "total_return": round(total * 100, 2),
        "cagr": round(cagr * 100, 2),
        "vol": round(vol * 100, 2),
        "sharpe": round(cagr / vol, sharpe_digits) if vol > 1e-9 else None,
        "max_drawdown": round(worst * 100, 2),
        "calmar": round(cagr / abs(worst), 3) if worst < -1e-9 else None,
        "years": round(years, 2),
    }


def annualized_turnover(total_turnover: float, years: float) -> float:
    """累计换手 → 年化换手（百分数）。口径统一处，避免各处再写 ``/years*100``。"""
    return round(total_turnover / max(years, 1e-9) * 100, 1)


def cost_drag_bps(total_cost: float, average_nav: float, years: float) -> float:
    """成本拖累：``Σcost / 平均净值 / 年数``，单位 bp/年。

    注意不要用 ``总成本 / 期末净值``——净值增长会把它低估近一半。
    """
    return round(total_cost / max(average_nav, 1e-9) / max(years, 1e-9) * 10000, 1)


__all__ = [
    "DEFAULT_MIN_XSEC",
    "sma", "rolling_max", "solve",
    "pairs_from_groups", "daily_ic",
    "nav_metrics", "annualized_turnover", "cost_drag_bps",
]
