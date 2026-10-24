"""验证与建模 v2：5 折 purged walk-forward + 样本权重 + 收缩/稳定选择 + 集成。

要解决的问题
------------
单次 60/40 切分有两个结构性缺陷，`protocol.py` 已经修掉了一部分（5 折 purged
WF + 末 2 年 lockbox），但这个文件针对的是剩下的四个：

1. **有效样本被高估**。17,760 条训练样本、20 日重叠标签，独立块只有 ~888 个。
   t 统计必须用 block bootstrap（块长 = 标签周期），否则标准误会低到 1/√20。
2. **特征筛选用训练集 IC 最大值**。这是选择偏差的直接来源：候选越多，最大值越
   乐观。这里改成"先验族 + 稳定性选择"——只保留在 bootstrap 里符号稳定的因子，
   且不按 IC 排序挑最好的表达。
3. **没有收缩**。12 个因子在 55 只股票的横截面上高度共线，无正则 OLS 会把噪声
   放大成权重。这里用带等权先验的贝叶斯岭，并对系数做跨折平滑（§5 的换手惩罚）。
4. **没有集成**。单一 horizon + 单一因子子集的 IC 序列方差很大；对不同 horizon、
   不同因子族、不同训练窗口做**秩平均**，是几乎唯一能稳定降低方差的免费午餐。

口径与约定
----------
- 折边界固定用 h=20 的 purged k-fold（5 折，purge 20 + embargo 5），保证所有
  horizon、所有模型在**同一批测试块**上评估，横向可比。某个成员用更长 horizon 时，
  在折内再做一次 `h−20` 天的额外 purge（更保守）。
- 标签一律是"当日横截面分位"（0–1），所以不同 horizon 的标签在同一尺度上，可以
  按 1/h 加权合并——这也是"样本权重"唯一真正起作用的地方（见 ``uniqueness_weights``）。
- 超参只在**训练块内部的 purged 验证块**上选，测试块只在最后评估一次。
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import evalkit
import protocol
import scoring
import xsec

HORIZONS = (5, 10, 20, 60)
BASE_HORIZON = 20
K_FOLDS = 5
EMBARGO = 5
REBALANCE = 5
TOP_FRAC = 0.20

# 因子族：集成成员按族切分，族内先验方向已在打分时乘掉
# 与 score_study.BATTERY 完全一致：键名必须能对上记录里的 ``s_<key>``，
# 多一个不存在的因子就会让整批样本被判定为缺值（曾因多写了一个 gap5 导致全空）。
FAMILIES: dict[str, tuple[str, ...]] = {
    "reversal": ("rev5",),
    "momentum": ("mom12_1", "macd_hist", "rs20", "high52"),
    "risk": ("ivol", "downside20", "mdd60"),
    "liquidity": ("turnover20", "amihud"),
    "pattern": ("clv20", "ma_stack"),
}

ALL_FEATURES: tuple[str, ...] = tuple(
    key for family in ("reversal", "momentum", "risk", "liquidity", "pattern")
    for key in FAMILIES[family]
)


def features(drop: Sequence[str] = ()) -> tuple[str, ...]:
    """电池的因子子集。``drop`` 用于剔除"方向与先验相反"的因子。

    剔除的判据必须是**跨折一致的符号冲突**（而不是某一折算出来是负的）：
    单折系数受当年 regime 影响，用单折反号当剔除理由等于在拟合噪声。
    """
    removed = set(drop)
    return tuple(key for key in ALL_FEATURES if key not in removed)


def families(drop: Sequence[str] = ()) -> dict[str, tuple[str, ...]]:
    """按 ``drop`` 过滤后的族；空族直接丢弃（否则会训练出一个空模型）。"""
    removed = set(drop)
    out: dict[str, tuple[str, ...]] = {}
    for family, keys in FAMILIES.items():
        kept = tuple(key for key in keys if key not in removed)
        if kept:
            out[family] = kept
    return out

# 超参网格（只在训练块内部选）
TAU_GRID = (0.02, 0.05, 0.15)
GAMMA_GRID = (0.0, 0.5, 2.0, 8.0)


# --------------------------------------------------------------------------- #
# 标签：多 horizon 的前向超额收益分位
# --------------------------------------------------------------------------- #


def forward_labels(
    rows_by_date: dict[str, list[dict[str, Any]]],
    bars: dict[str, dict[str, dict[str, float]]],
    index_closes: dict[str, float],
    dates: Sequence[str],
    *,
    horizons: Sequence[int] = HORIZONS,
) -> dict[int, dict[tuple[str, str], dict[str, float]]]:
    """为每个 horizon 生成 ``(date, symbol) -> {fwd, excess, label}``。

    超额收益用基准的同日收益，横截面分位在**当日有未来收益的标的**之间计算，
    避免最后 h 天因为没有标签而被单独分位。
    """
    position = {date: i for i, date in enumerate(dates)}
    result: dict[int, dict[tuple[str, str], dict[str, float]]] = {}
    for horizon in horizons:
        table: dict[tuple[str, str], dict[str, float]] = {}
        by_date: dict[str, list[tuple[str, float]]] = {}
        for date, rows in rows_by_date.items():
            start = position.get(date)
            if start is None or start + horizon >= len(dates):
                continue
            end_date = dates[start + horizon]
            bench = index_closes.get(end_date, 0.0) / index_closes.get(date, 1.0) - 1
            for record in rows:
                series = bars.get(record["symbol"])
                if not series or date not in series or end_date not in series:
                    continue
                entry = series[date]["close"]
                exit_price = series[end_date]["close"]
                if not entry or not exit_price:
                    continue
                forward = (exit_price / entry - 1) * 100
                table[(date, record["symbol"])] = {
                    "fwd": forward,
                    "excess": forward - bench * 100,
                }
                by_date.setdefault(date, []).append((record["symbol"], forward - bench * 100))
        for date, items in by_date.items():
            if len(items) < 20:
                continue
            ranks = scoring.rank01([value for _, value in items])
            for (symbol, _), rank in zip(items, ranks):
                table[(date, symbol)]["label"] = rank
        result[horizon] = table
    return result


def uniqueness_weights(dates: Sequence[str], horizon: int) -> dict[str, float]:
    """标签唯一性权重：每个样本的窗口平均"倒数重叠数"。

    逐日采样时每一天都被 h 个窗口覆盖，所以每个样本的权重恒为 1/h——**在单
    horizon 的拟合里它是一个常数因子，不改变 OLS 解**（只改变有效岭强度）。
    它的作用有两个真实场景：
    1. 不同 horizon 合并拟合时，让 60 日样本不要按"1 条 = 1 条"压倒 5 日样本；
    2. 非等间隔采样（stride）时，稀有窗口权重自然变大。

    换句话说："按 1/重叠数 加权"能修正的是**合并**与**采样**，修正不了单模型
    的点估计——后者只能靠 block bootstrap 的区间和"每折只看一次"来解决。
    """
    return {date: 1.0 / horizon for date in dates}


# --------------------------------------------------------------------------- #
# 加权充分统计量与贝叶斯岭
# --------------------------------------------------------------------------- #


@dataclass
class Stats:
    features: list[str]
    xtx: list[list[float]]
    xty: list[float]
    yyw: float
    n_effective: float
    blocks: list[tuple[list[list[float]], list[float], float, float]] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    scales: list[float] = field(default_factory=list)


def build_stats(
    rows_by_date: dict[str, list[dict[str, Any]]],
    features: Sequence[str],
    labels: dict[tuple[str, str], dict[str, float]],
    *,
    prefix: str = "s_",
    weight_fn: Callable[[str], float] | None = None,
) -> Stats:
    """按日累加加权充分统计量，同时保留每日块以便做块自助。

    权重乘在 ``x`` 和 ``y`` 上时等价于对残差平方加权（WLS）；用 ``sqrt(w)``
    缩放可以保持 ``XtWX`` 的形式不变。
    """
    grid = list(features)
    columns = len(grid)
    # 池化标准差用于把各列缩放到单位方差（否则 τ 无法跨因子比较）
    total = [0.0] * columns
    squares = [0.0] * columns
    count = 0
    for rows in rows_by_date.values():
        for record in rows:
            values = [record.get(prefix + key) for key in grid]
            if any(value is None for value in values):
                continue
            for i, value in enumerate(values):
                total[i] += value
                squares[i] += value * value
            count += 1
    scales: list[float] = []
    for i in range(columns):
        mean = total[i] / max(count, 1)
        variance = max(squares[i] / max(count, 1) - mean * mean, 1e-12)
        scales.append(math.sqrt(variance) or 1.0)

    stats = Stats(features=grid, xtx=[[0.0] * columns for _ in range(columns)],
                  xty=[0.0] * columns, yyw=0.0, n_effective=0.0, scales=scales)
    for date in sorted(rows_by_date):
        block_x = [[0.0] * columns for _ in range(columns)]
        block_y = [0.0] * columns
        block_yy = 0.0
        weight_sum = 0.0
        for record in rows_by_date[date]:
            label = labels.get((date, record["symbol"]), {}).get("label")
            if label is None:
                continue
            values = [record.get(prefix + key) for key in grid]
            if any(value is None for value in values):
                continue
            weight = math.sqrt(weight_fn(date)) if weight_fn else 1.0
            x = [float(value) / scales[i] * weight for i, value in enumerate(values)]
            y = (float(label) - 0.5) * weight
            weight_sum += weight * weight
            block_yy += y * y
            for i in range(columns):
                block_y[i] += x[i] * y
                for j in range(i, columns):
                    block_x[i][j] += x[i] * x[j]
        if weight_sum <= 0:
            continue
        for i in range(columns):
            for j in range(i):
                block_x[i][j] = block_x[j][i]
            for j in range(columns):
                stats.xtx[i][j] += block_x[i][j]
            stats.xty[i] += block_y[i]
        stats.yyw += block_yy
        stats.n_effective += weight_sum
        stats.blocks.append((block_x, block_y, block_yy, weight_sum))
        stats.dates.append(date)
    return stats


def fit_ridge(
    xtx: Sequence[Sequence[float]],
    xty: Sequence[float],
    *,
    yyw: float,
    count: float,
    prior_mean: Sequence[float],
    tau: float,
    previous: Sequence[float] | None = None,
    gamma: float = 0.0,
) -> list[float]:
    """``β = (XᵀWX/σ² + I/τ² + γI)⁻¹ (XᵀWy/σ² + μ/τ² + γβ_prev)``。

    ``γ`` 是**换手惩罚**：把系数拉向上一次调仓的系数。它比在目标函数里加
    ``|Δrank|`` 好实现得多（后者不可导），效果等价——限制权重的自相关就是
    限制排名的抖动，而排名抖动才是换手的来源。
    """
    columns = len(xty)
    plain = scoring._solve([list(row) for row in xtx], list(xty))
    residual = max(yyw - sum(plain[i] * xty[i] for i in range(columns)), 0.0)
    noise = math.sqrt(max(residual / max(count - columns - 1, 1), 1e-8))
    precision = 1.0 / (noise * noise)
    prior_precision = 1.0 / max(tau, 1e-6) ** 2
    matrix = [
        [xtx[i][j] * precision + ((prior_precision + gamma) if i == j else 0.0) for j in range(columns)]
        for i in range(columns)
    ]
    vector = [
        xty[i] * precision + prior_mean[i] * prior_precision
        + (gamma * previous[i] if previous is not None else 0.0)
        for i in range(columns)
    ]
    return scoring._solve(matrix, vector)


def score_block(
    rows_by_date: dict[str, list[dict[str, Any]]],
    features: Sequence[str],
    beta: Sequence[float],
    scales: Sequence[float],
    *,
    prefix: str = "s_",
) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for date, rows in rows_by_date.items():
        for record in rows:
            values = [record.get(prefix + key) for key in features]
            if any(value is None for value in values):
                continue
            out[(date, record["symbol"])] = sum(
                beta[i] * float(values[i]) / scales[i] for i in range(len(features))
            )
    return out


def rank_normalize(scores: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    """把分数换成当日横截面分位（0–1），用于跨成员秩平均。

    秩平均必须先分位化：不同 horizon 的分数尺度完全不同（60 日模型的系数
    天然更小），直接平均等于让 5 日模型独占权重。
    """
    by_date: dict[str, list[tuple[tuple[str, str], float]]] = {}
    for key, value in scores.items():
        by_date.setdefault(key[0], []).append((key, value))
    out: dict[tuple[str, str], float] = {}
    for date, items in by_date.items():
        items.sort(key=lambda item: item[1])
        count = len(items)
        for position, (key, _) in enumerate(items):
            out[key] = (position + 0.5) / count
    return out


def rank_average(
    members: Sequence[dict[tuple[str, str], float]],
    *,
    tiebreak: dict[tuple[str, str], float] | None = None,
) -> dict[tuple[str, str], float]:
    """成员秩平均。缺失成员按"已有成员的均值"处理，不补 0.5。

    补 0.5 会把"没有这个成员"和"成员给出中位分位"混为一谈，等价于给缺失成员
    一个中性票——这会系统性把分数拉向中间，正好抵消集成想要的分散化。
    """
    totals: dict[tuple[str, str], float] = {}
    counts: dict[tuple[str, str], int] = {}
    for member in members:
        for key, value in member.items():
            totals[key] = totals.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
    averaged = {key: totals[key] / counts[key] for key in totals}
    if tiebreak is None:
        return averaged
    # 秩平均会制造大量并列（9 个成员 × 55 只股票的名次和只有有限个取值），
    # 并列会让 top-20% 的边界随机抖动，直接表现为换手虚高。用主模型的分数
    # 做极小的 tie-break：只在并列时起作用，不改变非并列样本的排序。
    base = rank_normalize({key: value for key, value in tiebreak.items() if key in averaged})
    return {key: averaged[key] + 1e-6 * (base.get(key, 0.5) - 0.5) for key in averaged}


# --------------------------------------------------------------------------- #
# 折内超参选择（只用训练块内部的 purged 验证块）
# --------------------------------------------------------------------------- #


def period_returns(
    scores: dict[tuple[str, str], float],
    labels: dict[tuple[str, str], dict[str, float]],
    dates: Sequence[str],
    *,
    horizon: int = 5,
    top_fraction: float = TOP_FRAC,
    cost_buy: float = xsec.BUY_COST,
    cost_sell: float = xsec.SELL_COST,
    step: int = REBALANCE,
) -> dict[str, Any]:
    """按调仓周期的成本后收益序列（用于内层选参的目标函数）。

    这里是**周期级**近似：持有 top-20% 一个周期，收益取该周期的平均超额，
    成本按换手折算成周期收益率。它不需要 T+1 撮合，所以可以在内层网格上反复
    调用；最终报告的净净值仍然走 ``execution.simulate_t1``。
    """
    ordered = sorted(scores)
    by_date: dict[str, list[tuple[str, float]]] = {}
    for date, symbol in ordered:
        by_date.setdefault(date, []).append((symbol, scores[(date, symbol)]))
    returns: list[float] = []
    held: set[str] = set()
    for index, date in enumerate(dates):
        if index % step:
            continue
        row = by_date.get(date)
        if not row or len(row) < 10:
            continue
        row.sort(key=lambda item: -item[1])
        count = max(1, round(len(row) * top_fraction))
        target = {symbol for symbol, _ in row[:count]}
        turnover = len(held - target) / len(held) if held else 1.0
        cost = turnover * (cost_sell if target else 0) + (turnover * cost_buy)
        forward = [
            labels[(date, symbol)]["excess"]
            for symbol in target
            if (date, symbol) in labels and "excess" in labels[(date, symbol)]
        ]
        if not forward:
            held = target
            continue
        returns.append(statistics.fmean(forward) / 100 - cost)
        held = target
    if len(returns) < 6:
        return {"n": len(returns), "sharpe": None, "mean": None}
    mean = statistics.fmean(returns)
    deviation = statistics.pstdev(returns)
    periods_per_year = 252 / step
    return {
        "n": len(returns),
        "mean": mean,
        "sharpe": mean / deviation * math.sqrt(periods_per_year) if deviation > 1e-12 else None,
    }


def inner_split(dates: Sequence[str], *, horizon: int, blocks: int = 3) -> tuple[list[str], list[str]]:
    """训练块内部再切一刀：前 ``blocks-1`` 块训练、最后一块验证（带 purge）。"""
    bounds = [round(len(dates) * i / blocks) for i in range(blocks + 1)]
    train = list(dates[: max(0, bounds[-2] - horizon)])
    valid = list(dates[bounds[-2]:])
    return train, valid


def select_hyperparams(
    rows_by_date: dict[str, list[dict[str, Any]]],
    labels: dict[tuple[str, str], dict[str, float]],
    train_dates: Sequence[str],
    features: Sequence[str],
    *,
    horizon: int,
    prior_mean: Sequence[float],
    previous: Sequence[float] | None = None,
    objective: str = "sharpe",
) -> dict[str, Any]:
    """在训练块内部选 (τ, γ)。

    ``objective="sharpe"``：内层验证块的周期级净 Sharpe（默认，对应"目标函数
    从最小化分位误差改为最大化成本后收益"）。
    ``objective="mse"``：内层验证块的加权 MSE（复现旧口径，用于对照）。

    无论哪种，验证块都在训练块内部，测试折一次都没被碰过。
    """
    inner_train, inner_valid = inner_split(train_dates, horizon=horizon)
    if len(inner_valid) < horizon * 3 or len(inner_train) < horizon * 6:
        return {"tau": TAU_GRID[1], "gamma": 0.0, "objective": objective, "inner": None, "grid": []}
    inner_rows_train = {date: rows_by_date[date] for date in inner_train if date in rows_by_date}
    inner_rows_valid = {date: rows_by_date[date] for date in inner_valid if date in rows_by_date}
    if not inner_rows_train or not inner_rows_valid:
        return {"tau": TAU_GRID[1], "gamma": 0.0, "objective": objective, "inner": None, "grid": []}

    stats = build_stats(inner_rows_train, features, labels)
    grid: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for tau in TAU_GRID:
        for gamma in GAMMA_GRID:
            beta = fit_ridge(
                stats.xtx, stats.xty, yyw=stats.yyw, count=stats.n_effective,
                prior_mean=prior_mean, tau=tau, previous=previous, gamma=gamma,
            )
            scores = score_block(inner_rows_valid, features, beta, stats.scales)
            if objective == "mse":
                errors = []
                for key, value in scores.items():
                    label = labels.get(key, {}).get("label")
                    if label is not None:
                        errors.append((value - (label - 0.5)) ** 2)
                metric = -statistics.fmean(errors) if errors else None
                info = {"mse": statistics.fmean(errors) if errors else None}
            else:
                result = period_returns(scores, labels, inner_valid)
                metric = result["sharpe"]
                info = {"sharpe": result["sharpe"], "periods": result["n"]}
            grid.append({"tau": tau, "gamma": gamma, "metric": metric, **info})
            if metric is not None and (best is None or metric > best["metric"]):
                best = {"tau": tau, "gamma": gamma, "metric": metric, **info}
    if best is None:
        best = {"tau": TAU_GRID[1], "gamma": 0.0, "metric": None}
    return {"tau": best["tau"], "gamma": best["gamma"], "objective": objective,
            "inner": {k: best.get(k) for k in ("metric", "sharpe", "mse", "periods")},
            "grid": grid}


# --------------------------------------------------------------------------- #
# 折内训练与集成
# --------------------------------------------------------------------------- #


def subset_stats(stats: Stats, features: Sequence[str]) -> Stats:
    """从 12 列充分统计量里切出任意子集。

    族的子矩阵就是全矩阵的子块——这一步让"5 个族成员"的代价从 5 次 O(n·k²)
    降到 5 次索引，是本文件里最划算的一个优化（族成员和全模型共用同一批样本
    与标签，本来就该共用统计量）。
    """
    index = [stats.features.index(key) for key in features]
    columns = len(index)
    xtx = [[stats.xtx[i][j] for j in index] for i in index]
    xty = [stats.xty[i] for i in index]
    blocks = []
    for block_x, block_y, block_yy, weight in stats.blocks:
        blocks.append((
            [[block_x[i][j] for j in index] for i in index],
            [block_y[i] for i in index],
            block_yy,
            weight,
        ))
    return Stats(
        features=list(features), xtx=xtx, xty=xty, yyw=stats.yyw,
        n_effective=stats.n_effective, blocks=blocks, dates=list(stats.dates),
        scales=[stats.scales[i] for i in index],
    )


def fit_from_stats(
    stats: Stats,
    *,
    tau: float,
    gamma: float = 0.0,
    previous: Sequence[float] | None = None,
    prior_spread: float = 0.25,
) -> dict[str, Any]:
    prior_mean = [prior_spread / len(stats.features)] * len(stats.features)
    beta = fit_ridge(
        stats.xtx, stats.xty, yyw=stats.yyw, count=stats.n_effective,
        prior_mean=prior_mean, tau=tau, previous=previous, gamma=gamma,
    )
    return {
        "features": list(stats.features),
        "coefficients": {key: beta[i] for i, key in enumerate(stats.features)},
        "vector": beta,
        "scales": stats.scales,
        "n_effective": stats.n_effective,
        "n_days": len(stats.dates),
    }


def train_members(
    stats_by_horizon: dict[int, Stats],
    *,
    tau: float,
    gamma: float,
    previous: Sequence[float] | None = None,
    family_map: dict[str, tuple[str, ...]] | None = None,
) -> list[dict[str, Any]]:
    """集成成员：4 个 horizon（全因子）+ 5 个因子族（h=20）。

    为什么是这两组而不是"所有组合"：horizon 是方差的来源（标签噪声），族是偏差
    的来源（选错表达）。两者各取一组，成员数 9，秩平均的方差约降到 1/9，而每
    个成员只需要一次闭式解——集成在这里几乎是免费的。
    """
    members: list[dict[str, Any]] = []
    for horizon in sorted(stats_by_horizon):
        stats = stats_by_horizon[horizon]
        model = fit_from_stats(
            stats, tau=tau, gamma=gamma,
            previous=previous if horizon == BASE_HORIZON else None,
        )
        members.append({"kind": f"h{horizon}", "horizon": horizon, "model": model})
    base = stats_by_horizon.get(BASE_HORIZON)
    if base is not None:
        for family, keys in (family_map or FAMILIES).items():
            sub = subset_stats(base, keys)
            model = fit_from_stats(
                sub, tau=tau, gamma=gamma,
                previous=previous if family == "momentum" else None,
            )
            members.append({"kind": f"family:{family}", "horizon": BASE_HORIZON, "model": model})
    return members


def member_scores(member: dict[str, Any], rows_by_date: dict[str, list[dict[str, Any]]]) -> dict[tuple[str, str], float]:
    model = member["model"]
    return rank_normalize(
        score_block(rows_by_date, model["features"], model["vector"], model["scales"])
    )


def purge_dates(
    train_dates: Sequence[str],
    all_dates: Sequence[str],
    test_start: str,
    test_end: str,
    *,
    horizon: int,
    embargo: int = EMBARGO,
) -> list[str]:
    """给更长 horizon 的成员做额外 purge（折边界按 h=20 定，成员按自己的 h 再退）。"""
    position = {date: i for i, date in enumerate(all_dates)}
    left = position.get(test_start, 0)
    right = position.get(test_end, len(all_dates) - 1)
    keep = []
    for date in train_dates:
        index = position.get(date)
        if index is None:
            continue
        if index < left - horizon - embargo or index > right + embargo:
            keep.append(date)
    return keep
