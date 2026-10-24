"""风格基准与归因：把"选股 alpha"和"仓位其实压在某个风格上"分开。

没有行业分类数据，所以用**收益相关性聚类**做行业代理：同一行业受同一批
宏观/政策冲击驱动，日收益相关性显著高于跨行业。市值也拿不到，用
``成交额 / 换手率`` 反推流通市值（A 股日线同时给这两个字段，换手率是百分数），
量级误差不影响横截面排序。动量用标准 12-1。

归因走 Fama-MacBeth：每个调仓期做一次横截面回归

    r_i = a + b_size·size_z_i + b_mom·mom_z_i + Σ_c g_c·1[cluster_i=c] + e_i

得到该期的风格收益 λ；组合相对等权的超额可以精确拆成

    超额 = Σ_k λ_k·(组合暴露_k − 基准暴露_k) + 残差 alpha

残差才是"选股"部分。若残差接近 0 而风格项很大，说明所谓 alpha 只是
小盘 / 行业押注，换个风格基准就会消失。
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable, Sequence

from evalkit import ols_fit

DEFAULT_CLUSTERS = 8
MIN_OVERLAP = 150        # 两两相关系数至少需要的共同交易日
SIZE_FALLBACK_DAYS = 20


def _zscore(values: Sequence[float]) -> list[float]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(clean) < 2:
        return [0.0 for _ in values]
    mean = statistics.fmean(clean)
    sd = statistics.pstdev(clean)
    if sd <= 0:
        return [0.0 for _ in values]
    return [0.0 if v is None else (float(v) - mean) / sd for v in values]


# --------------------------------------------------------------------------- #
# 三个风格变量
# --------------------------------------------------------------------------- #


def size_series(history: dict[str, Any]) -> dict[str, float]:
    """流通市值代理：成交额 / (换手率/100)。换手率缺失时退回 20 日均成交额。"""
    rows = history["rows"]
    amounts = [float(row.get("amount") or 0.0) for row in rows]
    result: dict[str, float] = {}
    for index, row in enumerate(rows):
        turnover = float(row.get("turnover") or 0.0)
        amount = float(row.get("amount") or 0.0)
        if turnover > 0 and amount > 0:
            result[row["date"]] = amount / (turnover / 100.0)
            continue
        window = [value for value in amounts[max(0, index - SIZE_FALLBACK_DAYS + 1): index + 1] if value > 0]
        if window:
            result[row["date"]] = statistics.fmean(window) * 100.0
    return result


def momentum_series(history: dict[str, Any], *, lookback: int = 252, skip: int = 21) -> dict[str, float]:
    """12-1 动量：``close[t-skip] / close[t-lookback] - 1``，跳过最近一个月。"""
    rows = history["rows"]
    closes = [float(row["close"]) for row in rows]
    result: dict[str, float] = {}
    for index in range(lookback, len(rows)):
        base = closes[index - lookback]
        if base > 0:
            result[rows[index]["date"]] = (closes[index - skip] / base - 1.0) * 100.0
    return result


def _return_series(history: dict[str, Any]) -> dict[str, float]:
    rows = history["rows"]
    result: dict[str, float] = {}
    for index in range(1, len(rows)):
        previous = float(rows[index - 1]["close"])
        if previous > 0:
            result[rows[index]["date"]] = float(rows[index]["close"]) / previous - 1.0
    return result


def correlation_clusters(
    histories: Sequence[dict[str, Any]],
    *,
    clusters: int = DEFAULT_CLUSTERS,
    min_overlap: int = MIN_OVERLAP,
) -> dict[str, Any]:
    """按日收益相关性做层次聚类，作为行业代理。

    距离 = ``1 - corr``，average linkage，聚到 ``clusters`` 个簇。全过程确定性
    （标的按代码排序、并列按序号打破），同一份数据每次得到同一分组。
    """
    symbols = sorted(history["symbol"] for history in histories)
    returns = {
        history["symbol"]: _return_series(history)
        for history in histories
        if history["symbol"] in symbols
    }
    names = {history["symbol"]: history.get("name") or history["symbol"] for history in histories}
    counts = {symbol: len(series) for symbol, series in returns.items()}
    base = min(counts.values()) if counts else 0
    target = max(min_overlap, int(base * 0.6))
    date_counts: dict[str, int] = {}
    for series in returns.values():
        for date in series:
            date_counts[date] = date_counts.get(date, 0) + 1
    shared_dates = sorted(date for date, count in date_counts.items() if count >= max(2, len(symbols) * 0.8))
    if len(shared_dates) < min_overlap:
        shared_dates = sorted(date_counts)

    def corr(left: str, right: str) -> float:
        common = [d for d in shared_dates if d in returns[left] and d in returns[right]]
        if len(common) < target:
            return 0.0
        xs = [returns[left][d] for d in common]
        ys = [returns[right][d] for d in common]
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
        dy = math.sqrt(sum((b - my) ** 2 for b in ys))
        return num / (dx * dy) if dx > 0 and dy > 0 else 0.0

    matrix = {(a, b): corr(a, b) for i, a in enumerate(symbols) for b in symbols[i + 1:]}
    groups: list[list[str]] = [[symbol] for symbol in symbols]
    while len(groups) > max(1, clusters):
        best: tuple[float, int, int] | None = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                pairs = [
                    (1.0 - matrix.get((a, b), matrix.get((b, a), 0.0)))
                    for a in groups[i] for b in groups[j]
                ]
                distance = statistics.fmean(pairs) if pairs else 0.0
                if best is None or distance < best[0] - 1e-12:
                    best = (distance, i, j)
        if best is None:
            break
        _, i, j = best
        groups[i] = groups[i] + groups[j]
        groups.pop(j)

    amounts = {
        history["symbol"]: statistics.fmean(
            [float(row.get("amount") or 0.0) for row in history["rows"][-60:]]
        )
        for history in histories
    }
    mapping: dict[str, int] = {}
    table = []
    for index, group in enumerate(sorted(groups, key=lambda g: -max(amounts.get(s, 0.0) for s in g))):
        for symbol in group:
            mapping[symbol] = index
        representative = max(group, key=lambda s: (amounts.get(s, 0.0), s))
        table.append(
            {
                "cluster": index,
                "label": f"{names.get(representative, representative)} 等",
                "representative": representative,
                "members": sorted(group),
                "size": len(group),
            }
        )
    return {"mapping": mapping, "table": table, "names": names, "correlation": matrix}


def style_panel(
    histories: Sequence[dict[str, Any]],
    clusters: dict[str, Any],
) -> dict[str, dict[str, dict[str, float]]]:
    """``{date: {symbol: {"size", "mom", "cluster"}}}``，值为原始量（未标准化）。"""
    sizes = {history["symbol"]: size_series(history) for history in histories}
    momentum = {history["symbol"]: momentum_series(history) for history in histories}
    mapping = clusters["mapping"]
    panel: dict[str, dict[str, dict[str, float]]] = {}
    for history in histories:
        symbol = history["symbol"]
        for row in history["rows"]:
            date = row["date"]
            size = sizes[symbol].get(date)
            mom = momentum[symbol].get(date)
            if size is None or size <= 0:
                continue
            panel.setdefault(date, {})[symbol] = {
                "size": math.log(size),
                "mom": mom if mom is not None else float("nan"),
                "cluster": float(mapping.get(symbol, -1)),
            }
    return panel


# --------------------------------------------------------------------------- #
# Fama-MacBeth 归因
# --------------------------------------------------------------------------- #


def attribute_period(
    period: dict[str, Any],
    *,
    min_xsec: int = 20,
) -> dict[str, Any] | None:
    """把一个调仓期拆成风格贡献 + 残差 alpha。"""
    returns = period["returns"]
    exposures = period["exposures"]
    symbols = sorted(s for s in returns if s in exposures and returns[s] is not None)
    if len(symbols) < min_xsec:
        return None
    y = [float(returns[s]) for s in symbols]
    size_z = _zscore([exposures[s]["size"] for s in symbols])
    mom_raw = [exposures[s]["mom"] for s in symbols]
    if any(math.isnan(value) for value in mom_raw):
        # 动量缺失的样本不参与该期归因，避免用 0 假装"中性"
        keep = [i for i, value in enumerate(mom_raw) if not math.isnan(value)]
        if len(keep) < min_xsec:
            return None
        symbols = [symbols[i] for i in keep]
        y = [y[i] for i in keep]
        size_z = [size_z[i] for i in keep]
        mom_raw = [mom_raw[i] for i in keep]
    mom_z = _zscore(mom_raw)
    cluster_ids = sorted({int(exposures[s]["cluster"]) for s in symbols})
    reference = cluster_ids[0]
    dummies = [[1.0 if int(exposures[s]["cluster"]) == c else 0.0 for c in cluster_ids[1:]] for s in symbols]
    design = [[size_z[i], mom_z[i]] + dummies[i] for i in range(len(symbols))]
    fit = ols_fit(design, y, ridge=1e-6)
    coefficients = fit["coefficients"]
    b_size, b_mom = coefficients[0], coefficients[1]
    gammas = coefficients[2:]

    def mean_exposure(group: Sequence[str]) -> tuple[float, float, list[float]]:
        present = [s for s in group if s in set(symbols)]
        if not present:
            return (0.0, 0.0, [0.0] * len(cluster_ids[1:]))
        index = {s: i for i, s in enumerate(symbols)}
        size_avg = statistics.fmean(size_z[index[s]] for s in present)
        mom_avg = statistics.fmean(mom_z[index[s]] for s in present)
        cluster_avg = [
            sum(1.0 for s in present if int(exposures[s]["cluster"]) == c) / len(present)
            for c in cluster_ids[1:]
        ]
        return (size_avg, mom_avg, cluster_avg)

    p_size, p_mom, p_cluster = mean_exposure(period["portfolio"])
    b_size_exp, b_mom_exp, b_cluster = mean_exposure(period["benchmark"])
    actual_excess = (
        statistics.fmean([float(returns[s]) for s in period["portfolio"] if s in returns])
        - statistics.fmean([float(returns[s]) for s in period["benchmark"] if s in returns])
    ) * 100
    contrib_size = b_size * (p_size - b_size_exp) * 100
    contrib_mom = b_mom * (p_mom - b_mom_exp) * 100
    contrib_industry = sum(g * (p - b) for g, p, b in zip(gammas, p_cluster, b_cluster)) * 100
    style_total = contrib_size + contrib_mom + contrib_industry
    return {
        "start": period["start"],
        "end": period["end"],
        "n_xsec": len(symbols),
        "r2": round(fit["r2"], 3),
        "excess": round(actual_excess, 3),
        "contrib_size": round(contrib_size, 3),
        "contrib_momentum": round(contrib_mom, 3),
        "contrib_industry": round(contrib_industry, 3),
        "contrib_style": round(style_total, 3),
        "alpha": round(actual_excess - style_total, 3),
        "exposure_size_diff": round(p_size - b_size_exp, 3),
        "exposure_mom_diff": round(p_mom - b_mom_exp, 3),
        "exposure_cluster_diff": round(max((abs(p - b) for p, b in zip(p_cluster, b_cluster)), default=0.0), 3),
        "lambda_size": round(b_size * 100, 3),
        "lambda_momentum": round(b_mom * 100, 3),
        "dominant_industry": cluster_ids[1:][
            max(range(len(gammas)), key=lambda i: abs(gammas[i] * (p_cluster[i] - b_cluster[i])))
        ] if gammas else None,
    }


def _t_stat(values: Sequence[float]) -> float | None:
    clean = [v for v in values if v is not None]
    if len(clean) < 3:
        return None
    sd = statistics.stdev(clean)
    if sd <= 0:
        return None
    return statistics.fmean(clean) / (sd / math.sqrt(len(clean)))


def attribute(
    periods: Sequence[dict[str, Any]],
    *,
    min_xsec: int = 20,
) -> dict[str, Any]:
    """逐期归因并汇总。区间不重叠时，跨期 t 值可以直接按 iid 算。"""
    rows = [row for period in periods if (row := attribute_period(period, min_xsec=min_xsec))]
    if not rows:
        return {"periods": [], "summary": {}}
    keys = ("excess", "contrib_size", "contrib_momentum", "contrib_industry", "contrib_style", "alpha")
    summary = {}
    for key in keys:
        values = [row[key] for row in rows]
        summary[key] = {
            "mean": round(statistics.fmean(values), 3),
            "t": round(_t_stat(values), 2) if _t_stat(values) is not None else None,
            "positive_rate": round(sum(1 for v in values if v > 0) / len(values) * 100, 1),
            "sum": round(sum(values), 3),
        }
    summary["n_periods"] = len(rows)
    summary["alpha_share"] = (
        round(summary["alpha"]["sum"] / summary["excess"]["sum"] * 100, 1)
        if summary["excess"]["sum"] else None
    )
    summary["exposure_size"] = round(statistics.fmean([r["exposure_size_diff"] for r in rows]), 3)
    summary["exposure_momentum"] = round(statistics.fmean([r["exposure_mom_diff"] for r in rows]), 3)
    summary["mean_r2"] = round(statistics.fmean([r["r2"] for r in rows]), 3)
    return {"periods": rows, "summary": summary}
