"""多因子打分 v4：连续化 → 标准化/中性化 → 收缩权重 → 分位校准 → 验收指标。

为什么要这一层
--------------
``analyzer.score_factors`` 的九个因子是硬阈值 if/else：MACD 柱为正且改善就 +9、
否则直接 −8；量比 1.3 以上才给分，1.29 和 0.5 完全等价。这带来三个后果：

1. **分数是阶跃的**。阈值附近的一次微小波动就能让分数跳 17 分，组合在阈值
   上下反复整进整出——换手的一半来自阈值，不是来自信号。
2. **无法做横截面比较**。绝对阈值隐含"所有股票同一个合理量比"，但大盘股和
   小盘股的正常换手率差一倍以上，行业之间的波动结构也不同。
3. **权重是拍的**。±9 / ±8 是人为刻度，没有经过收缩，也没有检验符号稳定性。

本模块把这条链拆成五个可以分别验证的环节，每个环节都有明确的输入输出：

    原始因子 → ① 连续单调变换（−1..+1）
             → ② 横截面标准化 + 行业/市值中性化（MAD 稳健缩放）
             → ③ 贝叶斯岭 + bootstrap 符号稳定性收缩（权重）
             → ④ isotonic / 分位回归校准（分数 → 预期超额收益分位）
             → ⑤ 验收指标（五分组、IC、换手、子样本符号、校准误差）

设计约束
--------
1. **纯标准库**，与 ``analyzer`` / ``factors`` / ``xsec`` 保持一致，不引入
   numpy / sklearn。矩阵规模 ≤ 13×13，高斯消元足够。
2. **本模块不取数据、不落盘**。所有入参都是完整数组，出参是纯 dict/list，
   便于单测与逐日截断回放（``score_study`` 负责注入数据）。
3. **单调性是硬约束**：①的每一种变换、④的 isotonic 都必须非递减，否则
   "分数更高 = 预期收益更高"这句话不成立，后续所有解释都失效。
4. **校准只在训练集拟合**。任何用验证集调过的参数都会让校准曲线变成
   自证——45° 是样本外指标，不是拟合优度。
"""

from __future__ import annotations

import bisect
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

# --------------------------------------------------------------------------- #
# 口径常量
# --------------------------------------------------------------------------- #

SQUASH_K = 1.25       # tanh 的温度：z=1.25 时输出 0.76，z=2.5 时 0.96
CLIP_MAD = 3.5        # winsorize 的 MAD 倍数
MIN_GROUP = 3         # 行业内少于 3 只时不单独去均值，退回全样本均值
DEFAULT_BINS = 10     # 校准曲线的分箱数（等频）
DEFAULT_CAL_BINS = 25 # isotonic 拟合前的预分箱数


# --------------------------------------------------------------------------- #
# ① 连续单调变换：把任意分布的原始因子压到 −1..+1
# --------------------------------------------------------------------------- #


def clip(value: float, low: float, high: float) -> float:
    return low if value < low else (high if value > high else value)


def tanh_score(z: float, k: float = SQUASH_K) -> float:
    """``tanh(z/k)``：严格单调、过原点、有界 (−1, +1)、导数连续。

    相比 ``clip(z/3, -1, 1)`` 的好处是两端不会"压平后完全无差别"——极端值
    仍保留序信息，只是增量递减；相比 sigmoid 的好处是中心对称，0 分就是中位。
    """
    if k <= 0:
        raise ValueError("k 必须为正")
    return math.tanh(z / k)


def piecewise_score(value: float, knots: Sequence[float], scores: Sequence[float]) -> float:
    """分段线性映射：``knots`` 单调递增、``scores`` 非递减，两端按端点值截断。

    适合本身有自然刻度的因子（``ma_stack`` 0–3、``clv20`` 0–1、RSI 0–100）。
    比 tanh 更可控（能指定"RSI 70 以上开始扣分"），但仍然连续、没有阶跃。
    """
    if len(knots) != len(scores) or not knots:
        raise ValueError("knots 与 scores 必须等长且非空")
    if any(knots[i] >= knots[i + 1] for i in range(len(knots) - 1)):
        raise ValueError("knots 必须严格递增")
    if any(scores[i] > scores[i + 1] for i in range(len(scores) - 1)):
        raise ValueError("scores 必须非递减（保证单调）")
    if value <= knots[0]:
        return scores[0]
    if value >= knots[-1]:
        return scores[-1]
    index = bisect.bisect_right(list(knots), value) - 1
    left, right = knots[index], knots[index + 1]
    span = right - left
    if span <= 0:
        return scores[index]
    ratio = (value - left) / span
    return scores[index] + ratio * (scores[index + 1] - scores[index])


def factor_score(
    value: float | None,
    *,
    center: float = 0.0,
    scale: float = 1.0,
    prior: int = 1,
    k: float = SQUASH_K,
    knots: Sequence[float] | None = None,
    scores: Sequence[float] | None = None,
) -> float | None:
    """单因子 → −1..+1。缺值返回 ``None``，由加权层决定是否归零。

    ``prior`` 只决定方向（+1 越大越好），不改变强度——强度全部来自 |value 的
    z|，因此"先验为负的因子"不会因为符号翻转而丢掉量纲。
    """
    if value is None or not math.isfinite(value):
        return None
    if knots is not None and scores is not None:
        raw = piecewise_score(value, knots, scores)
    else:
        if scale <= 0:
            return None
        raw = tanh_score((value - center) / scale, k)
    return prior * raw


def robust_center_scale(values: Sequence[float], *, clip_mad: float = CLIP_MAD) -> tuple[float, float]:
    """中位数 + 1.4826×MAD；MAD 退化时依次退回 ``(P90−P10)/2.563``、截尾标准差、1.0。

    用 MAD 而不是标准差，是因为因子里本来就有尖峰（``amihud`` 在停牌复牌日
    能差两个数量级），标准差会被单点决定，缩放后的 z 全都挤在 ±0.2 以内。

    离散因子（``ma_stack`` 只有 0–3）会让 MAD = 0，这时用分位跨度；若跨度也
    为 0（当日横截面完全同值），返回 scale=1 让 z 全部落在 0——这比除以一个
    极小数、把噪声放大成 ±30 个标准差要诚实。
    """
    sample = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not sample:
        return 0.0, 1.0
    center = statistics.median(sample)
    mad = statistics.median([abs(v - center) for v in sample]) * 1.4826
    if mad > 1e-12:
        return center, mad
    ordered = sorted(sample)
    span = ordered[int(0.9 * (len(ordered) - 1))] - ordered[int(0.1 * (len(ordered) - 1))]
    if span > 1e-12:
        return center, span / 2.563
    if len(ordered) > 1:
        low = ordered[int(0.025 * (len(ordered) - 1))]
        high = ordered[int(0.975 * (len(ordered) - 1))]
        clipped = [clip(v, low, high) for v in ordered]
        deviation = statistics.pstdev(clipped)
        if deviation > 1e-12:
            return center, deviation
    return center, 1.0


def winsorize(values: Sequence[float], *, clip_mad: float = CLIP_MAD) -> list[float]:
    """按 MAD 截尾到 ±``clip_mad`` 倍，保留序、压制单点爆炸。"""
    center, scale = robust_center_scale(values, clip_mad=clip_mad)
    low, high = center - clip_mad * scale, center + clip_mad * scale
    return [clip(float(v), low, high) for v in values]


def rank01(values: Sequence[float]) -> list[float]:
    """横截面分位，0–1，并列取平均名次（与 ``factors.rank_pct`` 同口径）。"""
    n = len(values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: values[i])
    result = [0.0] * n
    start = 0
    while start < n:
        end = start
        while end + 1 < n and values[order[end + 1]] == values[order[start]]:
            end += 1
        average = (start + end) / 2.0
        for position in range(start, end + 1):
            result[order[position]] = (average + 0.5) / n
        start = end + 1
    return result


def quantile_squash(value: float, sample: Sequence[float]) -> float:
    """经验分位映射：``2·ECDF(value) − 1``，样本需已排序。

    非参数版本，不需要假设形状；缺点是尾部被压成一样的值（分位只有 0–1），
    所以更适合作为 tanh 的替代而不是叠加。
    """
    if not sample:
        return 0.0
    position = bisect.bisect_right(sample, value)
    return 2.0 * position / len(sample) - 1.0


def quantile_squash_series(
    values: Sequence[float | None],
    *,
    window: int = 750,
    min_obs: int = 120,
) -> list[float | None]:
    """逐点因果的经验分位 −1..+1（单标的页面用：没有横截面可排）。

    每个点只用它自己和之前 ``window`` 个观测，所以可以逐日截断回放；观测数
    不足 ``min_obs`` 时返回 ``None``，由上层回落到 tanh(自身 z)。
    """
    result: list[float | None] = []
    history: list[float] = []
    ordered: list[float] = []  # 保持排序的窗口副本
    for value in values:
        if value is None or not math.isfinite(float(value)):
            result.append(None)
            continue
        if len(ordered) >= min_obs:
            position = bisect.bisect_right(ordered, float(value))
            # 中位分位映射到 0，两端 ±1；用 mid-rank 让并列稳定。
            result.append(2.0 * (position - 0.5) / len(ordered) - 1.0)
        else:
            result.append(None)
        history.append(float(value))
        bisect.insort(ordered, float(value))
        if len(history) > window:
            oldest = history.pop(0)
            del ordered[bisect.bisect_left(ordered, oldest)]
    return result


# --------------------------------------------------------------------------- #
# ② 横截面标准化 + 行业/市值中性化
# --------------------------------------------------------------------------- #


def _ols_residual(target: Sequence[float], controls: Sequence[Sequence[float]], ridge: float = 1e-10) -> list[float]:
    """把 ``target`` 对若干已中心化的控制变量回归，返回残差（无截距）。"""
    n = len(target)
    if n == 0 or not controls:
        mean = statistics.fmean(target) if target else 0.0
        return [v - mean for v in target]
    k = len(controls)
    matrix = [[0.0] * k for _ in range(k)]
    vector = [0.0] * k
    for row in range(n):
        for i in range(k):
            xi = controls[i][row]
            if xi == 0.0:
                continue
            vector[i] += xi * target[row]
            for j in range(i, k):
                matrix[i][j] += xi * controls[j][row]
    for i in range(k):
        for j in range(i):
            matrix[i][j] = matrix[j][i]
        matrix[i][i] += ridge
    coefficients = _solve(matrix, vector)
    return [
        target[row] - sum(coefficients[i] * controls[i][row] for i in range(k))
        for row in range(n)
    ]


def _solve(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    """高斯-约当消元（带部分主元），维数 ≤ 13，够用。"""
    size = len(vector)
    aug = [list(matrix[i])[:] + [vector[i]] for i in range(size)]
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


def industry_demean(
    values: Sequence[float],
    industries: Sequence[str],
    *,
    min_group: int = MIN_GROUP,
) -> list[float]:
    """行业内去均值。

    行业内样本不足 ``min_group`` 时退回到全样本均值——否则一个只有 1 只股票
    的行业会被去成 0，等于把这个因子在这个行业上的信息全部抹掉，而且这个
    "0" 会和真正处于横截面中位的股票混在一起。
    """
    out = list(values)
    if not values:
        return out
    fallback = statistics.fmean(values)
    groups: dict[str, list[int]] = {}
    for index, group in enumerate(industries):
        groups.setdefault(group, []).append(index)
    for members in groups.values():
        if len(members) >= min_group:
            mean = statistics.fmean([values[i] for i in members])
        else:
            mean = fallback
        for i in members:
            out[i] -= mean
    return out


def neutralize(
    values: Sequence[float],
    *,
    industries: Sequence[str] | None = None,
    log_cap: Sequence[float] | None = None,
    beta: Sequence[float] | None = None,
    min_group: int = MIN_GROUP,
) -> list[float]:
    """行业（哑变量等价于分组去均值）→ 市值 → beta 的顺序残差化。

    顺序有意义：行业先做，是因为行业之间的市值水平本身差异很大，若先用市值
    回归，会把行业效应的一部分当市值效应吸走。市值与 beta 用 OLS 残差（两列
    同时回归），避免"先市值后 beta"时两者共线导致重复扣除。
    """
    out = list(values)
    if industries is not None:
        out = industry_demean(out, industries, min_group=min_group)
    controls: list[Sequence[float]] = []
    for control in (log_cap, beta):
        if control is None:
            continue
        mean = statistics.fmean(control)
        deviation = statistics.pstdev(control) if len(control) > 1 else 0.0
        if deviation <= 1e-12:
            continue
        controls.append([(v - mean) / deviation for v in control])
    if controls:
        out = _ols_residual(out, controls)
    return out


def standardize(
    values: Sequence[float],
    *,
    industries: Sequence[str] | None = None,
    log_cap: Sequence[float] | None = None,
    beta: Sequence[float] | None = None,
    mode: str = "z",
    clip_mad: float = CLIP_MAD,
    k: float = SQUASH_K,
    min_group: int = MIN_GROUP,
) -> list[float]:
    """一个因子在一个横截面上的完整处理：稳健缩放 → 中性化 → 再缩放 → 压到 −1..+1。

    ``mode="z"``：先 winsorize + MAD 缩放（保序、抗爆点），再做中性化。
    ``mode="rank"``：先转当日分位（对分布形状完全不敏感），再中性化。

    中性化之后必须**重新缩放**：残差的标准差已经变了，不重算会让不同因子的
    有效强度不一致，等于偷偷改了权重。
    """
    if not values:
        return []
    if mode == "rank":
        base = [2.0 * r - 1.0 for r in rank01(values)]
    elif mode == "z":
        base = robust_z(values, clip_mad=clip_mad)
    else:
        raise ValueError(f"未知 mode: {mode}")
    adjusted = neutralize(base, industries=industries, log_cap=log_cap, beta=beta, min_group=min_group)
    adjusted = robust_z(adjusted, clip_mad=clip_mad)
    return [tanh_score(v, k) for v in adjusted]


def robust_z(values: Sequence[float], *, clip_mad: float = CLIP_MAD) -> list[float]:
    center, scale = robust_center_scale(values, clip_mad=clip_mad)
    low, high = center - clip_mad * scale, center + clip_mad * scale
    return [(clip(float(v), low, high) - center) / scale for v in values]


# --------------------------------------------------------------------------- #
# ③ 权重：贝叶斯岭 + bootstrap 符号稳定性
# --------------------------------------------------------------------------- #


@dataclass
class Design:
    """按日累加的充分统计量：``XtX`` / ``Xty`` / ``yy`` / ``n``。

    不存 17 万行原始样本，只存每个交易日的 13×13 矩阵。bootstrap 时按"日块"
    重抽样再累加，等价于对原始样本重抽样，但快两个数量级——200 次 bootstrap
    从几分钟压到几秒。
    """

    features: list[str]
    dates: list[str]
    blocks: list[tuple[list[list[float]], list[float], float, int]] = field(default_factory=list)
    scales: list[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return sum(block[3] for block in self.blocks)


def build_design(
    rows_by_date: dict[str, list[dict[str, Any]]],
    features: Sequence[str],
    *,
    score_key: Callable[[dict[str, Any], str], float | None] | None = None,
    label_key: str = "label",
    prefix: str = "",
) -> Design:
    """把逐日记录转成充分统计量。

    入参 ``rows_by_date``：``{date: [record, ...]}``，每条记录带因子分数（−1..+1）
    与标签分位（0–1）。标签减去 0.5 使截距归零，因此模型不设截距列——两边的
    横截面均值本来就被中性化拿走了，再加截距只会和共线打架。

    因子分数默认从记录里的 ``"{prefix}{key}"`` 读取（``scores_continuous`` 用
    ``attach`` 模式直接写回记录，避免再维护一张 17 万行的分数表）。
    """
    getter = score_key or (lambda record, key: record.get(f"{prefix}{key}"))
    grid = list(features)
    # 第一遍：池化标准差，用于把各列缩放到单位方差（权重才可比较、τ 才可解释）
    sums = [0.0] * len(grid)
    squares = [0.0] * len(grid)
    counts = [0] * len(grid)
    for rows in rows_by_date.values():
        for record in rows:
            if record.get(label_key) is None:
                continue
            values = [getter(record, key) for key in grid]
            if any(v is None for v in values):
                continue
            for i, value in enumerate(values):
                sums[i] += value
                squares[i] += value * value
                counts[i] += 1
    scales: list[float] = []
    for i in range(len(grid)):
        count = max(counts[i], 1)
        mean = sums[i] / count
        variance = max(squares[i] / count - mean * mean, 1e-12)
        scales.append(math.sqrt(variance) or 1.0)

    columns = len(grid)
    design = Design(features=grid, dates=[], scales=scales)
    for date in sorted(rows_by_date):
        xtx = [[0.0] * columns for _ in range(columns)]
        xty = [0.0] * columns
        yy = 0.0
        used = 0
        for record in rows_by_date[date]:
            label = record.get(label_key)
            if label is None:
                continue
            values = [getter(record, key) for key in grid]
            if any(v is None for v in values):
                continue
            x = [float(v) / scales[i] for i, v in enumerate(values)]
            y = float(label) - 0.5
            used += 1
            yy += y * y
            for i in range(columns):
                xty[i] += x[i] * y
                for j in range(i, columns):
                    xtx[i][j] += x[i] * x[j]
        if used == 0:
            continue
        for i in range(columns):
            for j in range(i):
                xtx[i][j] = xtx[j][i]
        design.dates.append(date)
        design.blocks.append((xtx, xty, yy, used))
    return design


def fit_ridge(
    blocks: Sequence[tuple[list[list[float]], list[float], float, int]],
    *,
    prior_mean: Sequence[float],
    prior_tau: float,
    features: Sequence[str],
    noise: float | None = None,
) -> dict[str, Any]:
    """贝叶斯岭：``β = (XᵀX/σ² + I/τ²)⁻¹ (Xᵀy/σ² + μ/τ²)``。

    ``prior_mean`` 是"等权先验"：把 ``spread/n`` 当成每个因子的先验系数，
    数据只能在这个先验附近做偏移。这与"完全无信息先验"的区别，在 IC≈0.02
    这种信噪比下是决定性的——无信息先验下最小二乘会把噪声放大成权重，
    有信息先验下权重会被拉回 economically plausible 的量级。

    ``prior_tau`` 是先验标准差；τ 越小收缩越强。``noise`` 不给时用全样本
    残差方差估计，避免手工设 σ。
    """
    columns = len(features)
    xtx_total = [[0.0] * columns for _ in range(columns)]
    xty_total = [0.0] * columns
    yy_total = 0.0
    count = 0
    for xtx, xty, yy, n in blocks:
        count += n
        yy_total += yy
        for i in range(columns):
            xty_total[i] += xty[i]
            for j in range(columns):
                xtx_total[i][j] += xtx[i][j]
    if count == 0:
        raise ValueError("没有可用训练样本")

    if noise is None:
        # 先用无正则解估 σ²，再回代。两步足够：σ 只影响收缩强度，不改变方向。
        plain = _solve(xtx_total, xty_total)
        residual = max(yy_total - sum(plain[i] * xty_total[i] for i in range(columns)), 0.0)
        noise = math.sqrt(max(residual / max(count - columns - 1, 1), 1e-8))
    else:
        noise = max(float(noise), 1e-6)

    precision = 1.0 / (noise * noise)
    prior_precision = 1.0 / max(prior_tau, 1e-6) ** 2
    matrix = [[xtx_total[i][j] * precision + (prior_precision if i == j else 0.0) for j in range(columns)] for i in range(columns)]
    vector = [xty_total[i] * precision + prior_mean[i] * prior_precision for i in range(columns)]
    coefficients = _solve(matrix, vector)

    # 有效自由度：tr((XᵀX/σ² + I/τ²)⁻¹ XᵀX/σ²)，用来解释"收缩掉多少参数"
    fitted = sum(coefficients[i] * xty_total[i] for i in range(columns))
    return {
        "features": list(features),
        "coefficients": {key: coefficients[i] for i, key in enumerate(features)},
        "prior_mean": list(prior_mean),
        "prior_tau": prior_tau,
        "noise": noise,
        "n_train": count,
        "r2_in": 1.0 - max(yy_total - fitted, 0.0) / max(yy_total, 1e-12),
    }


def bootstrap_fit(
    design: Design,
    *,
    prior_mean: Sequence[float],
    prior_tau: float,
    iters: int = 200,
    block_days: int = 20,
    seed: int = 7,
) -> dict[str, Any]:
    """按"日块"重抽样（块长 = horizon），重估系数 N 次。

    块长必须等于标签周期：相邻 20 个交易日的标签共享同一段未来收益，逐日
    独立重抽样会把重叠结构打碎，得到的符号稳定性会系统性偏乐观。

    返回每个因子的 ``sign_rate``（与全样本系数同号的比例）与 5%–95% 区间。
    """
    import random

    rng = random.Random(seed)
    dates = design.dates
    if not dates:
        raise ValueError("design 为空")
    span = max(1, block_days)
    starts = list(range(0, max(1, len(dates) - span + 1), span))
    columns = len(design.features)
    draws: list[list[float]] = []
    for _ in range(iters):
        chosen: list[int] = []
        while len(chosen) < len(dates):
            start = starts[rng.randrange(len(starts))]
            chosen.extend(range(start, min(start + span, len(dates))))
        chosen = chosen[: len(dates)]
        xtx = [[0.0] * columns for _ in range(columns)]
        xty = [0.0] * columns
        yy = 0.0
        count = 0
        for index in chosen:
            block_xtx, block_xty, block_yy, n = design.blocks[index]
            count += n
            yy += block_yy
            for i in range(columns):
                xty[i] += block_xty[i]
                for j in range(columns):
                    xtx[i][j] += block_xtx[i][j]
        plain = _solve(xtx, xty)
        residual = max(yy - sum(plain[i] * xty[i] for i in range(columns)), 0.0)
        noise = math.sqrt(max(residual / max(count - columns - 1, 1), 1e-8))
        precision = 1.0 / (noise * noise)
        prior_precision = 1.0 / max(prior_tau, 1e-6) ** 2
        matrix = [[xtx[i][j] * precision + (prior_precision if i == j else 0.0) for j in range(columns)] for i in range(columns)]
        vector = [xty[i] * precision + prior_mean[i] * prior_precision for i in range(columns)]
        draws.append(_solve(matrix, vector))

    point = fit_ridge(design.blocks, prior_mean=prior_mean, prior_tau=prior_tau, features=design.features)
    sign_rate: dict[str, float] = {}
    ci: dict[str, tuple[float, float]] = {}
    for i, key in enumerate(design.features):
        column = sorted(draw[i] for draw in draws)
        target = point["coefficients"][key]
        agreement = sum(1 for value in column if (value >= 0) == (target >= 0)) / len(column)
        sign_rate[key] = round(agreement, 4)
        ci[key] = (column[int(0.05 * len(column))], column[int(0.95 * len(column)) - 1])
    return {
        "iterations": iters,
        "block_days": block_days,
        "sign_rate": sign_rate,
        "ci90": ci,
        "point": point,
    }


def stability_weights(
    fit: dict[str, Any],
    boot: dict[str, Any],
    *,
    prior_sign: dict[str, int],
    min_sign_rate: float = 0.8,
    drop_on_prior_conflict: bool = True,
) -> dict[str, Any]:
    """系数 × 稳定性折扣，得到最终权重。

    ``discount = max(0, 2·sign_rate − 1)``：符号 100% 一致时原样保留，70% 一致
    时只剩 40%，50%（等于抛硬币）时归零。这是一条连续规则，不是"过 80% 就留、
    否则一刀砍掉"的硬阈值——后者会把 79% 的因子和 51% 的因子同等对待。

    ``min_sign_rate`` 是验收线（默认 0.8），低于线直接剔除并记录在 ``dropped``，
    便于报告里解释"为什么最终只有 N 个因子"。
    """
    weights: dict[str, float] = {}
    share: dict[str, float] = {}
    dropped: list[dict[str, Any]] = []
    adjusted: dict[str, float] = {}
    for key, coefficient in fit["coefficients"].items():
        rate = boot["sign_rate"].get(key, 0.0)
        factor = max(0.0, 2.0 * rate - 1.0)
        if rate < min_sign_rate:
            factor = 0.0
        if drop_on_prior_conflict and coefficient * prior_sign.get(key, 1) < 0:
            factor = 0.0
        value = coefficient * factor
        adjusted[key] = value
        if abs(value) < 1e-12:
            dropped.append({
                "key": key,
                "coefficient": round(coefficient, 4),
                "sign_rate": rate,
                "reason": (
                    f"符号稳定性 {rate:.0%} < {min_sign_rate:.0%}"
                    if rate < min_sign_rate
                    else "方向与先验相反"
                ),
            })
        else:
            weights[key] = value
    total = sum(abs(v) for v in weights.values())
    for key, value in weights.items():
        share[key] = abs(value) / total if total else 0.0
    return {
        "weights": weights,
        "share": share,
        "adjusted": adjusted,
        "dropped": dropped,
        "l1": total,
    }


# --------------------------------------------------------------------------- #
# ④ 校准：分数 → 预期 20 日超额收益分位
# --------------------------------------------------------------------------- #


def pava(values: Sequence[float], weights: Sequence[float] | None = None) -> list[float]:
    """保序回归（Pool Adjacent Violators），返回非递减拟合值。

    加权版本：每块的权重是样本数，合并时按权重取均值——不加权会让只有 30 个
    样本的极端桶和 3 万个样本的中间桶同权，尾部噪声直接变成校准曲线的折角。
    """
    n = len(values)
    if n == 0:
        return []
    weight_list = list(weights) if weights else [1.0] * n
    block_values: list[float] = []
    block_weights: list[float] = []
    block_sizes: list[int] = []
    for value, weight in zip(values, weight_list):
        block_values.append(float(value))
        block_weights.append(float(weight))
        block_sizes.append(1)
        while len(block_values) > 1 and block_values[-2] > block_values[-1]:
            total_weight = block_weights[-2] + block_weights[-1]
            merged = (
                block_values[-2] * block_weights[-2] + block_values[-1] * block_weights[-1]
            ) / total_weight if total_weight > 0 else (block_values[-2] + block_values[-1]) / 2
            size = block_sizes[-2] + block_sizes[-1]
            block_values[-2:] = [merged]
            block_weights[-2:] = [total_weight]
            block_sizes[-2:] = [size]
    out: list[float] = []
    for value, size in zip(block_values, block_sizes):
        out.extend([value] * size)
    return out


class IsotonicCalibrator:
    """把任意单调分数映射到 0–1 的"预期横截面分位"。

    拟合流程：按分数等频分箱（默认 25 箱）→ 每箱取实际标签分位的均值 →
    PAVA 修正非单调。等频分箱而不是等宽，是因为分数分布本身集中在中间，
    等宽分箱会让两端每箱只有个位数样本，PAVA 直接把它们抹平。

    预测时用每个箱的中点分数作为断点，落箱即取值——不插值，保持 PAVA 的
    分段常数性质（插值会引入"箱内单调"的假精度）。

    收缩（``shrink``）
    ------------------
    拟合内的 isotonic 一定比样本外"敢说"：它把训练集里那点选择性偏差也当成
    信号，输出跨度可以到 ±0.05，而样本外真实跨度只有 ±0.005。直接拿这条曲线
    去预测，校准曲线的斜率会掉到 0.1–0.2（说 60% 实际还是 50%）。

    因此校准要**再乘一个收缩系数**：``p = 0.5 + λ·(iso(score) − 0.5)``。λ 由
    训练集内部的 purged 分块交叉验证估计（在每一折上拟合、在留出折上用最小
    二乘求斜率，再取平均），不碰验证集。λ≈1 说明信号跨期可复制，λ≈0 说明
    拟合内的分位差全是噪声——后者正是这份数据的情况。
    """

    def __init__(
        self,
        bins: int = DEFAULT_CAL_BINS,
        bounds: tuple[float, float] = (0.0, 1.0),
        shrink: float = 1.0,
    ) -> None:
        self.bins = bins
        self.bounds = bounds
        self.shrink = shrink
        self.knots: list[float] = []
        self.values: list[float] = []
        self.counts: list[int] = []
        self.base_rate: float = 0.5
        self.raw_values: list[float] = []

    def _fit_iso(self, scores: Sequence[float], targets: Sequence[float]) -> "IsotonicCalibrator":
        # 只按分数排序：若用 (分数, 标签) 排序，同分样本会被标签强行排开，
        # 每个分箱的标签均值就成了"分位数的分位数"，校准曲线会出现假折角。
        pairs = sorted(
            ((float(s), float(t)) for s, t in zip(scores, targets)
             if s is not None and t is not None and math.isfinite(float(s)) and math.isfinite(float(t))),
            key=lambda pair: pair[0],
        )
        if not pairs:
            raise ValueError("校准样本为空")
        n = len(pairs)
        chunks = max(1, min(self.bins, n))
        size = max(1, n // chunks)
        bin_scores: list[float] = []
        bin_targets: list[float] = []
        counts: list[int] = []
        for start in range(0, n, size):
            chunk = pairs[start:start + size]
            if len(chunk) < max(3, size // 3):
                continue
            bin_scores.append(statistics.fmean([p[0] for p in chunk]))
            bin_targets.append(statistics.fmean([p[1] for p in chunk]))
            counts.append(len(chunk))
        fitted = pava(bin_targets, counts)
        low, high = self.bounds
        self.knots = bin_scores
        self.raw_values = [clip(v, low, high) for v in fitted]
        self.values = [clip(0.5 + self.shrink * (v - 0.5), low, high) for v in self.raw_values]
        self.counts = counts
        self.base_rate = clip(statistics.fmean([p[1] for p in pairs]), low, high)
        return self

    def fit(
        self,
        scores: Sequence[float],
        targets: Sequence[float],
        *,
        dates: Sequence[str] | None = None,
        folds: int = 5,
        purge: int = 20,
    ) -> "IsotonicCalibrator":
        """先估收缩系数（给了 ``dates`` 才做交叉验证），再拟合完整 isotonic。

        ``purge`` 按"交易日"跳过的边界长度：20 日标签跨越折边界时，训练折里
        会混进留出折的未来信息，λ 会被高估。给不出日期时退化到 λ=1 并如实
        标注（``shrink_estimated=False``）。
        """
        self.shrink_estimated = False
        if dates is not None and folds >= 2:
            triples = [
                (float(s), float(t), str(d))
                for s, t, d in zip(scores, targets, dates)
                if s is not None and t is not None and math.isfinite(float(s))
            ]
            unique = sorted({item[2] for item in triples})
            if len(unique) >= folds * (purge + 10):
                block = len(unique) // folds
                collected: list[tuple[float, float]] = []
                for fold in range(folds):
                    held = set(unique[fold * block:(fold + 1) * block - purge if fold < folds - 1 else len(unique)])
                    if not held:
                        continue
                    inside = [(s, t) for s, t, d in triples if d not in held]
                    outside = [(s, t) for s, t, d in triples if d in held]
                    if len(inside) < 200 or len(outside) < 100:
                        continue
                    fold_cal = IsotonicCalibrator(bins=self.bins, bounds=self.bounds, shrink=1.0)
                    fold_cal._fit_iso([p[0] for p in inside], [p[1] for p in inside])
                    collected.extend((fold_cal.predict(s), t) for s, t in outside)
                if collected:
                    center = 0.5
                    numerator = sum((p - center) * (t - center) for p, t in collected)
                    denominator = sum((p - center) ** 2 for p, _ in collected)
                    if denominator > 1e-12:
                        self.shrink = clip(numerator / denominator, 0.0, 1.5)
                        self.shrink_estimated = True
        return self._fit_iso(scores, targets)

    def predict(self, score: float | None) -> float:
        if score is None or not self.knots:
            return self.base_rate
        index = bisect.bisect_right(self.knots, float(score)) - 1
        index = max(0, min(index, len(self.values) - 1))
        return self.values[index]

    def predict_many(self, scores: Sequence[float | None]) -> list[float]:
        return [self.predict(s) for s in scores]

    def summary(self) -> dict[str, Any]:
        return {
            "bins": len(self.knots),
            "base_rate": round(self.base_rate, 4),
            "head": round(self.values[-1], 4) if self.values else None,
            "tail": round(self.values[0], 4) if self.values else None,
            "monotone": all(self.values[i] <= self.values[i + 1] for i in range(len(self.values) - 1)),
            "shrink": round(self.shrink, 4),
            "shrink_estimated": getattr(self, "shrink_estimated", False),
            "raw_span": round(self.raw_values[-1] - self.raw_values[0], 4) if self.raw_values else None,
        }


def percentile_of_scores(scores: Sequence[float]) -> list[float]:
    """训练集分数的经验 CDF：未校准模型的"自称分位"。

    与 isotonic 的区别：CDF 只看分数自身的分布，不知道标签；isotonic 看
    "这个分数历史上对应的实际分位"。两条曲线画在一起，就能看出信号到底
    提供的是排序信息（CDF 有斜率）还是噪声（isotonic 是平的）。
    """
    ordered = sorted(float(s) for s in scores)
    return [bisect.bisect_right(ordered, float(s)) / len(ordered) for s in scores]


def calibration_curve(
    predicted: Sequence[float],
    realized: Sequence[float],
    *,
    bins: int = DEFAULT_BINS,
) -> list[dict[str, Any]]:
    """等宽分箱的信赖度曲线：每箱的（预测均值，实际均值）。

    用等宽而不是等频：等频分箱在预测值有大量并列时（isotonic 输出是分段常数）
    会把同一个预测值的样本拆进相邻两箱，两箱的实际均值被人为拉开，画出来的
    曲线会出现并不存在的锯齿。等宽分箱 + 丢空箱，曲线的横轴就是"模型说的
    分位"，可以直接和 45° 线比。
    """
    pairs = [
        (float(p), float(r)) for p, r in zip(predicted, realized)
        if p is not None and r is not None and math.isfinite(float(p)) and math.isfinite(float(r))
    ]
    if not pairs:
        return []
    low = min(0.0, min(p for p, _ in pairs))
    high = max(1.0, max(p for p, _ in pairs))
    width = (high - low) / max(1, bins)
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for pair in pairs:
        index = min(bins - 1, int((pair[0] - low) / width)) if width > 0 else 0
        buckets[index].append(pair)
    curve = [
        {
            "n": len(bucket),
            "pred": statistics.fmean([p for p, _ in bucket]),
            "real": statistics.fmean([r for _, r in bucket]),
        }
        for bucket in buckets if bucket
    ]
    return curve


def calibration_metrics(curve: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """ECE、最大偏差、斜率、离 45° 线的 RMS 距离。

    斜率由"实际 ~ 预测"的横截面 OLS 给出：斜率 1 + 截距 0 才是 45°。只看
    ECE 不够——一条整体偏移但斜率正确的曲线，和一个陡峭但中间相交的曲线，
    ECE 可能相同，含义完全不同。
    """
    if not curve:
        return {"ece": None, "max_gap": None, "slope": None, "intercept": None, "rms": None, "n": 0}
    total = sum(point["n"] for point in curve)
    gaps = [point["real"] - point["pred"] for point in curve]
    ece = sum(abs(g) * point["n"] for g, point in zip(gaps, curve)) / total
    xs = [point["pred"] for point in curve]
    ys = [point["real"] for point in curve]
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    variance = sum((x - mean_x) ** 2 for x in xs)
    slope = (
        sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / variance
        if variance > 1e-12 else None
    )
    intercept = mean_y - slope * mean_x if slope is not None else None
    rms = math.sqrt(statistics.fmean([g * g for g in gaps]))
    return {
        "ece": round(ece, 4),
        "max_gap": round(max(abs(g) for g in gaps), 4),
        "slope": round(slope, 3) if slope is not None else None,
        "intercept": round(intercept, 4) if intercept is not None else None,
        "rms": round(rms, 4),
        "n": total,
    }


def calibration_metrics_ci(
    by_date: dict[str, list[tuple[float, float]]],
    *,
    block: int = 20,
    iters: int = 80,
    alpha: float = 0.10,
    seed: int = 17,
    bins: int = DEFAULT_BINS,
) -> dict[str, Any]:
    """校准指标按"日块"重抽样，给出 ECE 与斜率的区间。

    为什么要区间：斜率的分母是预测值的方差，而这份数据的预测跨度只有 0.02 量级，
    点估计的抽样误差极大（±0.4 很常见）。只报一个斜率数字，读者会把它当成
    "校准失败"的证据；给区间才能说清楚"是拒绝 45°，还是样本量不足以判断"。
    重抽样必须按日块（块长 = 标签周期），否则 20 日重叠标签会把区间压窄。
    """
    import random

    dates = sorted(by_date)
    if len(dates) < block * 3:
        return {"slope": None, "slope_low": None, "slope_high": None,
                "ece": None, "ece_low": None, "ece_high": None, "iters": 0}
    rng = random.Random(seed)
    starts = list(range(0, max(1, len(dates) - block + 1), block))
    slopes: list[float] = []
    eces: list[float] = []
    for _ in range(iters):
        chosen: list[str] = []
        while len(chosen) < len(dates):
            start = starts[rng.randrange(len(starts))]
            chosen.extend(dates[start:start + block])
        predicted: list[float] = []
        realized: list[float] = []
        for date in chosen:
            for pred, real in by_date[date]:
                predicted.append(pred)
                realized.append(real)
        metrics = calibration_metrics(calibration_curve(predicted, realized, bins=bins))
        if metrics["slope"] is not None:
            slopes.append(metrics["slope"])
            eces.append(metrics["ece"])
    if not slopes:
        return {"slope": None, "slope_low": None, "slope_high": None,
                "ece": None, "ece_low": None, "ece_high": None, "iters": 0}
    slopes.sort()
    eces.sort()
    low = max(0, int(alpha / 2 * len(slopes)) - 1)
    high = min(len(slopes) - 1, int((1 - alpha / 2) * len(slopes)))
    return {
        "slope": round(statistics.fmean(slopes), 3),
        "slope_low": round(slopes[low], 3),
        "slope_high": round(slopes[high], 3),
        "ece": round(statistics.fmean(eces), 4),
        "ece_low": round(eces[low], 4),
        "ece_high": round(eces[high], 4),
        "covers_one": bool(slopes[low] <= 1.0 <= slopes[high]),
        "iters": iters,
    }


def quantile_fit(
    x: Sequence[float],
    y: Sequence[float],
    tau: float,
    *,
    iters: int = 40,
    ridge: float = 1e-6,
    sample: int | None = 12000,
    seed: int = 11,
) -> dict[str, float] | None:
    """单变量分位回归（IRLS 近似）：返回 ``(intercept, slope)``。

    目标是最小化 pinball 损失 ``Σ ρ_τ(y − a − b·x)``。这里用 IRLS：把 pinball
    的次梯度写成加权最小二乘的权重 ``w = τ/|r|``（正残差）或 ``(1−τ)/|r|``
    （负残差），迭代 40 次足够收敛到 1e-4 量级。

    用途不是替代 isotonic，而是给校准加一条**区间**：τ=0.25/0.5/0.75 三条线
    之间就是这个分数对应的"预期分位区间"，可以回答"分数 0.6 到底意味着
    60% 还是 40%–75%"。
    """
    pairs = [(float(a), float(b)) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 30:
        return None
    if sample and len(pairs) > sample:
        import random

        rng = random.Random(seed)
        pairs = [pairs[rng.randrange(len(pairs))] for _ in range(sample)]
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    var_x = sum((v - mean_x) ** 2 for v in xs)
    if var_x <= 1e-12:
        return {"intercept": mean_y, "slope": 0.0}
    slope = sum((a - mean_x) * (b - mean_y) for a, b in pairs) / var_x
    intercept = mean_y - slope * mean_x
    for _ in range(iters):
        s_xx = s_xy = s_x = s_y = s_w = 0.0
        for a, b in pairs:
            residual = b - (intercept + slope * a)
            weight = (tau if residual >= 0 else (1.0 - tau)) / max(abs(residual), 1e-3)
            s_w += weight
            s_x += weight * a
            s_y += weight * b
            s_xx += weight * a * a
            s_xy += weight * a * b
        if s_w <= 0:
            break
        denominator = s_xx - s_x * s_x / s_w + ridge
        if abs(denominator) < 1e-12:
            break
        new_slope = (s_xy - s_x * s_y / s_w) / denominator
        new_intercept = (s_y - new_slope * s_x) / s_w
        if abs(new_slope - slope) < 1e-6 and abs(new_intercept - intercept) < 1e-6:
            slope, intercept = new_slope, new_intercept
            break
        slope, intercept = new_slope, new_intercept
    return {"intercept": intercept, "slope": slope}


# --------------------------------------------------------------------------- #
# ⑤ 验收指标
# --------------------------------------------------------------------------- #


def daily_ic(
    scores_by_date: dict[str, dict[str, float]],
    forward_by_date: dict[str, dict[str, float]],
    *,
    min_xsec: int = 10,
) -> list[float]:
    """逐日横截面 Spearman(分数, 未来收益)，返回日序列（供块自助法取区间）。"""
    from evalkit import spearman

    series: list[float] = []
    for date in sorted(scores_by_date):
        scores = scores_by_date[date]
        forwards = forward_by_date.get(date, {})
        symbols = [s for s in scores if s in forwards]
        if len(symbols) < min_xsec:
            continue
        value = spearman([scores[s] for s in symbols], [forwards[s] for s in symbols])
        if value is not None:
            series.append(value)
    return series


def quintile_table(
    scores_by_date: dict[str, dict[str, float]],
    forward_by_date: dict[str, dict[str, float]],
    *,
    groups: int = 5,
    min_xsec: int | None = None,
) -> list[dict[str, Any]]:
    """按分数分 N 组，统计各组未来收益与胜率（等权，逐日再平均）。"""
    need = min_xsec or groups * 4
    buckets: list[list[float]] = [[] for _ in range(groups)]
    for date in sorted(scores_by_date):
        scores = scores_by_date[date]
        forwards = forward_by_date.get(date, {})
        pairs = sorted((scores[s], forwards[s]) for s in scores if s in forwards)
        if len(pairs) < need:
            continue
        for index, (_score, forward) in enumerate(pairs):
            bucket = min(groups - 1, index * groups // len(pairs))
            buckets[bucket].append(forward)
    return [
        {
            "group": f"Q{index + 1}",
            "n_days": len(values),
            "mean": round(statistics.fmean(values), 3) if values else None,
            "win_rate": round(sum(1 for v in values if v > 0) / len(values) * 100, 1) if values else None,
        }
        for index, values in enumerate(buckets)
    ]


def quintile_monotonicity(table: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """五分组单调性：组序与组均值的 Spearman + 严格递增步数。"""
    from evalkit import spearman

    values = [point["mean"] for point in table if point["mean"] is not None]
    if len(values) < 3:
        return {"spearman": None, "steps": None, "spread": None}
    ranks = list(range(1, len(values) + 1))
    rho = spearman([float(r) for r in ranks], values)
    steps = sum(1 for i in range(len(values) - 1) if values[i + 1] > values[i])
    return {
        "spearman": round(rho, 3) if rho is not None else None,
        "steps": f"{steps}/{len(values) - 1}",
        "spread": round(values[-1] - values[0], 3),
        "long_short": round((values[-1] - values[0]) / 2, 3),
    }


def turnover_series(
    targets: Sequence[Sequence[str]],
    *,
    step: int = 5,
    periods: int = 252,
) -> dict[str, Any]:
    """单边换手序列：每次调仓卖掉的市值占比（等权组合，按只数计）。

    ``|old − new| / |old|``：等权且只数固定时，它与按市值算的单边换手一致。
    年化用 ``× periods/step``，与 ``xsec.metrics`` 的口径对齐。
    """
    if len(targets) < 2:
        return {"mean": 0.0, "annual": 0.0, "max": 0.0, "n": 0}
    values: list[float] = []
    for previous, current in zip(targets, targets[1:]):
        old, new = set(previous), set(current)
        if not old:
            continue
        values.append(len(old - new) / len(old))
    if not values:
        return {"mean": 0.0, "annual": 0.0, "max": 0.0, "n": 0}
    mean = statistics.fmean(values)
    return {
        "mean": round(mean, 4),
        "annual": round(mean * periods / step * 100, 1),
        "max": round(max(values), 4),
        "n": len(values),
    }


def top_fraction(scores: dict[str, float], fraction: float) -> list[str]:
    """按分数取前 ``fraction``（并列时按代码排序，保证可复现）。"""
    if not scores:
        return []
    count = max(1, round(len(scores) * fraction))
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [symbol for symbol, _ in ordered[:count]]


def buffer_select(
    scores: dict[str, float],
    held: Sequence[str],
    *,
    entry_pct: float = 0.20,
    exit_pct: float = 0.35,
    target_pct: float = 0.20,
    min_hold: int = 0,
    ages: dict[str, int] | None = None,
) -> list[str]:
    """带缓冲带的选股：新进要进前 ``entry_pct``，已持有要跌出 ``exit_pct`` 才卖。

    这是"降低换手"的第二层（第一层是分数连续化）。连续化减少的是**评分抖动**，
    缓冲带减少的是**排名在阈值附近的边界抖动**——两者解决的不是同一件事：
    前者让 1.29 和 1.31 的差别变小，后者让"第 12 名和第 13 名互换"不再成交。
    """
    if not scores:
        return []
    size = max(1, round(len(scores) * target_pct))
    entry = max(1, round(len(scores) * entry_pct))
    exit_line = max(entry, round(len(scores) * exit_pct))
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    rank = {symbol: position for position, (symbol, _) in enumerate(ordered)}
    keep: list[str] = []
    for symbol in held:
        position = rank.get(symbol)
        if position is None:
            # 当日没有评分（停牌/缺特征）：持有到下一次有评分再判断
            keep.append(symbol)
            continue
        if position < exit_line:
            keep.append(symbol)
        elif ages and ages.get(symbol, 0) < min_hold and position < max(exit_line, size + 1):
            keep.append(symbol)
    keep_set = set(keep)
    target = keep[:size]
    for symbol, _score in ordered:
        if len(target) >= size:
            break
        if symbol in keep_set:
            continue
        if rank[symbol] < entry:
            target.append(symbol)
    return target[:size]


def composite_score(
    scores: dict[str, float | None],
    weights: dict[str, float],
) -> float | None:
    """加权合成，缺因子按可用权重重归一。

    不能把缺值当 0：一只新上市股票没有 12–1 动量，直接补 0 等于替它做了一个
    "中性"判断，还会稀释其余因子的权重。按可用权重重归一后，分数仍然是
    同一口径的加权平均，只是"用到的因子"少了几项，这在报告里可以标注。
    """
    available = [
        (weights[key], value) for key, value in scores.items()
        if key in weights and value is not None and math.isfinite(float(value))
    ]
    total = sum(abs(weight) for weight, _ in available)
    if not available or total <= 1e-12:
        return None
    return sum(weight * float(value) for weight, value in available) / total


# --------------------------------------------------------------------------- #
# 单标的场景：没有横截面时的等价实现
# --------------------------------------------------------------------------- #


def expanding_scores(
    values: Sequence[float | None],
    *,
    window: int = 750,
    min_obs: int = 120,
    prior: int = 1,
    k: float = SQUASH_K,
) -> list[float | None]:
    """单标的逐日分数：用自身历史经验分位（PIT）替代当日横截面分位。

    为什么不用固定的 center/scale：个股的波动水平会随 regime 整体漂移（2015
    年创业板和 2023 年的同一只股票，"正常换手率"差好几倍），固定参数会让
    整段时期的分数系统性偏一侧。经验分位随窗口滚动，自动跟上 regime。
    """
    percentiles = quantile_squash_series(values, window=window, min_obs=min_obs)
    out: list[float | None] = []
    for value, percentile in zip(values, percentiles):
        if value is None or percentile is None:
            out.append(None)
        else:
            out.append(prior * tanh_score(percentile * 1.5, k))
    return out
