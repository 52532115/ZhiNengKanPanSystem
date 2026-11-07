"""统计口径层：所有"信号到底有没有用"的判断都只从这里出数。

为什么单独一层
--------------
以前的报告直接比较 IC 与 0、净值与基准，等于把点估计当结论。这里把四件事
固定成可复用函数，任何新方向都必须走同一套口径：

1. **区间**：``block_bootstrap_mean`` / ``ic_ci``。IC 是逐日序列，而 H 日标签
   让相邻日的 IC 天然重叠，普通 t 检验会高估有效样本。这里用块长 = H 的
   moving block bootstrap，块内保留重叠结构，块间近似独立。
2. **功效**：``required_independent_blocks`` / ``detectable_ic``。先回答"这个
   样本量能看见多大的 IC"，再谈"看到的 IC 是不是真的"。
3. **多重检验**：``bh_fdr`` 控制已试次数的假发现率，``deflated_sharpe`` 用
   试验次数抬高"看起来显著"的门槛。
4. **记账**：``TrialLedger`` 把每次试过的因子表达 / horizon / 换手上限 /
   universe 都落盘，``n_trials`` 直接进上面两个校正公式。

只依赖标准库。
"""

from __future__ import annotations

import json
import math
import random
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_ALPHA = 0.05
DEFAULT_ITERS = 2000
DEFAULT_SEED = 20261021
EULER_GAMMA = 0.5772156649015329
LEDGER_PATH = Path(__file__).resolve().parent / ".cache" / "eval" / "trials.jsonl"

# 试验台账的主键：同一组配置重复跑只更新数值，不重复计数。
LEDGER_KEY_FIELDS = (
    "study",
    "target",
    "feature",
    "horizon",
    "universe",
    "rebalance",
    "top",
    "turnover",
    "execution",
    "split",
)


# --------------------------------------------------------------------------- #
# 分布函数（标准库没有，自己实现，避免引入 scipy）
# --------------------------------------------------------------------------- #


def norm_cdf(value: float) -> float:
    return 0.5 * math.erfc(-value / math.sqrt(2.0))


def norm_ppf(probability: float) -> float:
    """标准正态分位数（Acklam 近似，绝对误差 < 1.2e-9）。"""
    if not 0.0 < probability < 1.0:
        raise ValueError("probability 必须在 (0, 1) 内")
    a = (
        -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
        1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
        6.680131188771972e01, -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
        -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
        3.754408661907416e00,
    )
    low, high = 0.02425, 1.0 - 0.02425
    if probability < low:
        q = math.sqrt(-2.0 * math.log(probability))
        return (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if probability > high:
        q = math.sqrt(-2.0 * math.log(1.0 - probability))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = probability - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


# --------------------------------------------------------------------------- #
# 秩相关
# --------------------------------------------------------------------------- #


def rank_average(values: Sequence[float]) -> list[float]:
    """升序平均秩（1 起），并列取平均。"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and values[order[end + 1]] == values[order[index]]:
            end += 1
        average = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            ranks[order[position]] = average
        index = end + 1
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    rx, ry = rank_average(left), rank_average(right)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx <= 0 or dy <= 0:
        return None
    return num / (dx * dy)


# --------------------------------------------------------------------------- #
# 块自助法区间
# --------------------------------------------------------------------------- #


def block_bootstrap_mean(
    series: Sequence[float | None],
    *,
    block: int | None = None,
    iters: int = DEFAULT_ITERS,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """均值序列的 moving block bootstrap 区间与双侧 p 值。

    块长默认取 ``n**(1/3)``，但验证 IC 时必须显式传 ``block=horizon``：
    H 日标签让相邻 H 个交易日的 IC 共享同一段未来收益，重叠结构必须整块保留。
    p 值用**重心化**的 bootstrap 检验（把重采样分布平移到 0，看观察均值有多远），
    避免偏态分布下尾部比例给出"强烈支持原假设"的荒谬数。显著性结论统一以
    区间是否覆盖 0 为准，p 只用于 BH-FDR 这类需要连续 scale 的校正。
    """
    values = [float(v) for v in series if v is not None and math.isfinite(float(v))]
    n = len(values)
    if n < 4:
        return {
            "mean": statistics.fmean(values) if values else None,
            "ci_low": None, "ci_high": None, "se": None, "p": None,
            "n": n, "block": None, "n_blocks": None, "iters": 0,
        }
    if block is None:
        block = max(2, round(n ** (1.0 / 3.0)))
    block = int(max(2, min(block, n)))
    rng = random.Random(seed)
    starts_max = n - block
    means: list[float] = []
    for _ in range(iters):
        sample: list[float] = []
        while len(sample) < n:
            start = rng.randint(0, starts_max)
            sample.extend(values[start:start + block])
        means.append(statistics.fmean(sample[:n]))
    means.sort()
    lo = means[max(0, int(alpha / 2.0 * iters) - 1)]
    hi = means[min(iters - 1, int((1.0 - alpha / 2.0) * iters))]
    center = statistics.fmean(values)
    spread = statistics.pstdev(means)
    if spread <= 1e-15:
        # 退化序列（重采样后每块都相同）：自助法对抽样误差一无所知，
        # 这时给 p=0 会假装"极显著"，必须显式返回 None。
        return {
            "mean": center, "ci_low": center, "ci_high": center, "se": 0.0,
            "p": None, "n": n, "block": block,
            "n_blocks": round(n / block, 1), "iters": iters,
            "covers_zero": True, "significant": False,
        }
    # 重心化检验：把重采样分布平移到 0，再看观察到的均值有多"远"。
    # 直接用尾部比例在偏态分布上会给出 0.998 这种"强烈支持原假设"的荒谬值。
    if center == 0.0:
        p_value = 1.0
    else:
        extreme = sum(1 for m in means if abs(m - center) >= abs(center))
        p_value = (extreme + 1) / (iters + 1)
    return {
        "mean": center,
        "ci_low": lo,
        "ci_high": hi,
        "se": spread,
        "p": p_value,
        "n": n,
        "block": block,
        "n_blocks": round(n / block, 1),
        "iters": iters,
        "covers_zero": bool(lo <= 0.0 <= hi),
        "significant": bool(lo > 0.0 or hi < 0.0),
    }


def ic_ci(
    daily_ics: Sequence[float | None],
    *,
    horizon: int,
    iters: int = DEFAULT_ITERS,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """逐日横截面 IC 序列 → 带块自助区间的均值 IC。"""
    stat = block_bootstrap_mean(daily_ics, block=horizon, iters=iters, alpha=alpha, seed=seed)
    clean = [v for v in daily_ics if v is not None]
    stat["positive_rate"] = (
        round(sum(1 for v in clean if v > 0) / len(clean) * 100, 1) if clean else None
    )
    # 无区间时不允许给"有效 / 无效"的结论，统一用一个显式字段表达
    if stat["ci_low"] is None:
        stat["significant"] = None
        stat["covers_zero"] = None
    return stat


# --------------------------------------------------------------------------- #
# 功效：先问样本量能看见多大的 IC
# --------------------------------------------------------------------------- #


def required_independent_blocks(
    ic: float,
    sigma: float,
    *,
    alpha: float = DEFAULT_ALPHA,
    power: float = 0.8,
) -> float | None:
    """在给定 IC 与 IC 抽样标准差下，需要多少个独立块才能有 ``power`` 把握检出。

    ``n = ((z_{1-alpha/2} + z_{power}) * sigma / ic) ** 2``

    ``sigma`` 是"一块 IC 的抽样标准差"。重叠标签下块均值的标准差不会因块内
    多日平均而下降，因此直接取逐日 IC 的实际标准差即可（这是我们自己的
    保守口径，不假装块内独立）。
    """
    if not ic or sigma <= 0:
        return None
    z_alpha = norm_ppf(1.0 - alpha / 2.0)
    z_power = norm_ppf(power)
    return ((z_alpha + z_power) * sigma / abs(ic)) ** 2


def detectable_ic(
    n_blocks: float,
    sigma: float,
    *,
    alpha: float = DEFAULT_ALPHA,
    power: float = 0.8,
) -> float | None:
    """反过来：给定独立块数，能检出的最小 IC（同一公式解出 ic）。"""
    if n_blocks <= 0 or sigma <= 0:
        return None
    z_alpha = norm_ppf(1.0 - alpha / 2.0)
    z_power = norm_ppf(power)
    return (z_alpha + z_power) * sigma / math.sqrt(n_blocks)


def power_table(
    sigma: float,
    *,
    ics: Iterable[float] = (0.02, 0.03, 0.05, 0.10),
    horizons: Iterable[int] = (5, 20, 60),
    alpha: float = DEFAULT_ALPHA,
    power: float = 0.8,
) -> list[dict[str, Any]]:
    rows = []
    for target_ic in ics:
        blocks = required_independent_blocks(target_ic, sigma, alpha=alpha, power=power)
        rows.append(
            {
                "ic": target_ic,
                "sigma": sigma,
                "n_blocks": None if blocks is None else round(blocks, 1),
                "n_days": {
                    f"h{horizon}": None if blocks is None else round(blocks * horizon)
                    for horizon in horizons
                },
                "n_years": {
                    f"h{horizon}": None if blocks is None else round(blocks * horizon / 252, 1)
                    for horizon in horizons
                },
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# 多重检验
# --------------------------------------------------------------------------- #


def bh_fdr(pvalues: Sequence[float], *, q: float = 0.05) -> dict[str, Any]:
    """Benjamini–Hochberg：在 m 次试验里控制假发现率 q。"""
    m = len(pvalues)
    if m == 0:
        return {"reject": [], "qvalues": [], "n_reject": 0, "threshold": None, "m": 0}
    order = sorted(range(m), key=lambda i: pvalues[i])
    k_max = 0
    for rank, index in enumerate(order, 1):
        if pvalues[index] <= q * rank / m:
            k_max = rank
    qvalues = [1.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        index = order[rank - 1]
        running = min(running, pvalues[index] * m / rank)
        qvalues[index] = running
    reject = [False] * m
    for rank, index in enumerate(order, 1):
        if rank <= k_max:
            reject[index] = True
    return {
        "reject": reject,
        "qvalues": qvalues,
        "n_reject": k_max,
        "threshold": (q * k_max / m) if k_max else None,
        "m": m,
        "q": q,
    }


def _sharpe_moments(returns: Sequence[float]) -> dict[str, float]:
    clean = [float(r) for r in returns if r is not None]
    n = len(clean)
    if n < 3:
        return {"sr": 0.0, "n": n, "skew": 0.0, "kurt": 3.0}
    mean = statistics.fmean(clean)
    sd = statistics.pstdev(clean)
    sr = mean / sd if sd else 0.0
    if sd:
        standardized = [(r - mean) / sd for r in clean]
        skew = statistics.fmean([s ** 3 for s in standardized])
        kurt = statistics.fmean([s ** 4 for s in standardized])
    else:
        skew, kurt = 0.0, 3.0
    return {"sr": sr, "n": n, "skew": skew, "kurt": kurt}


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """N 次独立试验下最大夏普的期望（Bailey & López de Prado, 2014）。"""
    if n_trials <= 1 or sr_variance <= 0:
        return 0.0
    return math.sqrt(sr_variance) * (
        (1.0 - EULER_GAMMA) * norm_ppf(1.0 - 1.0 / n_trials)
        + EULER_GAMMA * norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    )


def deflated_sharpe(
    returns: Sequence[float],
    *,
    n_trials: int,
    trial_sr_variance: float | None = None,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """通货紧缩夏普：把"试了 N 次"折算进显著性门槛。

    返回单期夏普 ``sr``、门槛 ``sr0``（N 次试验的期望最大值）、
    ``dsr``（真实夏普 > 0 的概率）与年化夏普。
    """
    moments = _sharpe_moments(returns)
    n, sr, skew, kurt = moments["n"], moments["sr"], moments["skew"], moments["kurt"]
    if n < 3:
        return {"sr": None, "sr0": None, "dsr": None, "annual": None, "n": n, "n_trials": n_trials}
    variance = trial_sr_variance if trial_sr_variance is not None else (
        (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2) / max(n - 1, 1)
    )
    sr0 = expected_max_sharpe(max(variance, 0.0), max(n_trials, 1))
    denominator = math.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2))
    dsr = norm_cdf((sr - sr0) * math.sqrt(max(n - 1, 1)) / denominator)
    return {
        "sr": round(sr, 4),
        "sr0": round(sr0, 4),
        "dsr": round(dsr, 4),
        "annual": round(sr * math.sqrt(periods_per_year), 3),
        "n": n,
        "n_trials": n_trials,
        "skew": round(skew, 3),
        "kurt": round(kurt, 3),
    }


# --------------------------------------------------------------------------- #
# 因子重要性（线性模型版的 MDI）
# --------------------------------------------------------------------------- #


def ols_fit(
    rows: Sequence[Sequence[float]],
    targets: Sequence[float],
    *,
    ridge: float = 1e-8,
) -> dict[str, Any]:
    """带极小岭项的正规方程 OLS。返回系数、截距与 R²。

    系数第 0 位是截距，其余与 ``rows`` 的列顺序一一对应。岭项只为防止共线
    导致矩阵不可逆，量级 1e-8 对结果没有实际影响。
    """
    if not rows or not targets:
        return {"coefficients": [], "intercept": None, "r2": 0.0, "n": 0}
    columns = len(rows[0]) + 1
    xtx = [[0.0] * columns for _ in range(columns)]
    xty = [0.0] * columns
    for vector, y in zip(rows, targets):
        full = [1.0] + list(vector)
        for i in range(columns):
            xty[i] += full[i] * y
            for j in range(columns):
                xtx[i][j] += full[i] * full[j]
    for i in range(1, columns):
        xtx[i][i] += ridge
    augmented = [row[:] + [xty[i]] for i, row in enumerate(xtx)]
    size = columns
    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(augmented[r][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            continue
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        for c in range(column, size + 1):
            augmented[column][c] /= divisor
        for r in range(size):
            if r == column:
                continue
            factor = augmented[r][column]
            if not factor:
                continue
            for c in range(column, size + 1):
                augmented[r][c] -= factor * augmented[column][c]
    coefficients = [augmented[i][size] for i in range(size)]
    mean = statistics.fmean(targets)
    total = sum((y - mean) ** 2 for y in targets)
    residual = 0.0
    for vector, y in zip(rows, targets):
        prediction = coefficients[0] + sum(c * v for c, v in zip(coefficients[1:], vector))
        residual += (y - prediction) ** 2
    return {
        "coefficients": coefficients[1:],
        "intercept": coefficients[0],
        "r2": max(0.0, 1.0 - residual / total) if total > 0 else 0.0,
        "n": len(targets),
    }


def _ols_r2(rows: Sequence[Sequence[float]], targets: Sequence[float], *, ridge: float = 1e-8) -> float:
    return ols_fit(rows, targets, ridge=ridge)["r2"]


def mdi_r2(
    rows: Sequence[dict[str, Any]],
    features: Sequence[str],
    *,
    label: str = "label",
    seed: int = DEFAULT_SEED,
    permutations: int = 5,
) -> list[dict[str, Any]]:
    """线性模型的"平均降低不纯度"替代品：剔除单个因子的 R² 损失 + 置换重要性。

    树模型用的是不纯度下降，线性分位模型没有不纯度，最接近的同源量是
    "去掉这个因子，R² 掉多少"。置换重要性再加一层稳健性：把该因子分位在
    当日横截面内打乱，看 R² 掉多少，避免只用一次拟合的偶然性。
    """
    usable = [
        record for record in rows
        if record.get(label) is not None
        and all(record.get(f"rank_{key}") is not None for key in features)
    ]
    if len(usable) < 50 or not features:
        return []
    targets = [float(record[label]) for record in usable]
    design = [[float(record[f"rank_{key}"]) for key in features] for record in usable]
    full = _ols_r2(design, targets)
    rng = random.Random(seed)
    results = []
    for position, key in enumerate(features):
        reduced_design = [
            [value for j, value in enumerate(vector) if j != position] for vector in design
        ]
        drop = full - _ols_r2(reduced_design, targets) if len(features) > 1 else full
        shuffled_scores = []
        for _ in range(permutations):
            column = [vector[position] for vector in design]
            shuffled = column[:]
            rng.shuffle(shuffled)
            permuted = [vector[:] for vector in design]
            for index, value in enumerate(shuffled):
                permuted[index][position] = value
            shuffled_scores.append(full - _ols_r2(permuted, targets))
        results.append(
            {
                "feature": key,
                "full_r2": round(full, 5),
                "drop_r2": round(drop, 5),
                "drop_share": round(drop / full * 100, 1) if full else None,
                "permutation_drop": round(statistics.fmean(shuffled_scores), 5),
                "permutation_share": (
                    round(statistics.fmean(shuffled_scores) / full * 100, 1) if full else None
                ),
            }
        )
    return results


# --------------------------------------------------------------------------- #
# 试验台账
# --------------------------------------------------------------------------- #


class TrialLedger:
    """把试过的每一个配置写进 JSONL，供多重检验校正使用。

    这不是日志，是统计口径的一部分：``n_trials`` 直接决定 BH-FDR 的阈值和
    deflated Sharpe 的门槛。只记录**真正看过的**配置，重复跑同一配置只更新
    数值（``runs`` 计数），不会把同一件事算成多次试验。
    """

    def __init__(self, path: Path | str = LEDGER_PATH) -> None:
        self.path = Path(path)

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        rows = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def record(self, **fields: Any) -> dict[str, Any]:
        key = {name: fields.get(name) for name in LEDGER_KEY_FIELDS}
        entries = self.entries()
        previous = None
        for entry in entries:
            if all(entry.get(name) == key[name] for name in LEDGER_KEY_FIELDS):
                previous = entry
                break
        entry = {
            **key,
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "metric": fields.get("metric"),
            "value": fields.get("value"),
            "p": fields.get("p"),
            "ci_low": fields.get("ci_low"),
            "ci_high": fields.get("ci_high"),
            "n_blocks": fields.get("n_blocks"),
            "note": fields.get("note"),
            "runs": (previous or {}).get("runs", 0) + 1,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if previous is not None:
            entries = [entry if all(e.get(n) == key[n] for n in LEDGER_KEY_FIELDS) else e for e in entries]
        else:
            entries.append(entry)
        with self.path.open("w", encoding="utf-8") as handle:
            for item in entries:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        return entry

    def summary(self, *, q: float = 0.05) -> dict[str, Any]:
        entries = self.entries()
        tested = [e for e in entries if e.get("p") is not None]
        pvalues = [float(e["p"]) for e in tested]
        correction = bh_fdr(pvalues, q=q)
        for entry, rejected, qvalue in zip(tested, correction["reject"], correction["qvalues"]):
            entry["bh_reject"] = rejected
            entry["qvalue"] = round(qvalue, 4)
        by_universe: dict[str, int] = {}
        for entry in entries:
            key = str(entry.get("universe"))
            by_universe[key] = by_universe.get(key, 0) + 1
        by_metric: dict[str, int] = {}
        for entry in entries:
            by_metric[str(entry.get("metric"))] = by_metric.get(str(entry.get("metric")), 0) + 1
        return {
            "n_trials": len(entries),
            "n_tested": len(tested),
            "n_bh_reject": correction["n_reject"],
            "n_raw_p05": sum(1 for p in pvalues if p < 0.05),
            "q": q,
            "by_metric": by_metric,
            "by_universe": by_universe,
            # 收益类试验（净值 / 超额 / 夏普）都要进口径校正，不只是显式叫 sharpe 的
            "n_sharpe_trials": sum(
                count for metric, count in by_metric.items() if metric in ("sharpe", "excess", "return")
            ),
            "entries": entries,
        }
