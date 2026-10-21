"""Offline walk-forward validation for the rule-based research score.

跑法（离线，不改动线上链路）：

    python walkforward.py
    python walkforward.py --symbols 600519,300750,510300 --days 600 --refresh

设计要点：

1. 每个评估点 t 只用 ``rows[:t+1]`` 这一个前缀调用 ``analyze()``，未来数据在
   物理上不存在于入参里，因此不会泄露（见 tests/test_walkforward.py 的截断测试）。
2. 收益、胜率、窗口内最大回撤全部在评分之后单独计算，属于事后标签，不回流。
3. 复用 ``market_data.MarketDataClient``，不引入新数据源；行情落盘到
   ``.cache/walkforward``，断网时可直接用缓存复跑。
4. 分桶直接对齐 ``analyze()`` 里既有的五档研判阈值，验证"分数含义"本身而非
   另造一套标签。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from analyzer import FACTOR_META, analyze, max_drawdown, score_factors
from market_data import MarketDataClient, MarketDataError

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / ".cache" / "walkforward"
REPORT_DIR = ROOT / "reports"

HORIZONS = (5, 20, 60)

# 与 analyze() 的五档 view 阈值一一对应，避免"评分一套阈值、验证另一套"。
BUCKETS: tuple[tuple[str, int, int, str, str], ...] = (
    ("lt30", 0, 29, "高风险", "negative"),
    ("30_44", 30, 44, "谨慎", "negative"),
    ("45_59", 45, 59, "中性观察", "neutral"),
    ("60_74", 60, 74, "谨慎偏多", "positive"),
    ("ge75", 75, 100, "偏积极", "positive"),
)

# MA120 与 60 日波动率窗口所需的最小预热长度。
WARMUP = 130

DEFAULT_SYMBOLS = ("600519", "300750", "601318", "000858", "510300", "588000", "159915", "518880")


# --------------------------------------------------------------------------- #
# 行情读取（复用现有数据层，只加文件缓存）
# --------------------------------------------------------------------------- #


def _cache_path(symbol: str, days: int, *, index: bool = False) -> Path:
    return CACHE_DIR / f"{symbol}{'_index' if index else ''}_{days}.json"


def load_history(
    client: MarketDataClient,
    symbol: str,
    days: int,
    *,
    refresh: bool = False,
    offline: bool = False,
    index: bool = False,
) -> dict[str, Any]:
    path = _cache_path(symbol, days, index=index)
    if path.is_file() and not refresh:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    if offline:
        raise MarketDataError(f"{symbol} 没有本地缓存（{path}），无法离线运行")
    history = client.history(symbol, days, index=index)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, ensure_ascii=False)
    return history


def load_benchmark(
    days: int,
    *,
    symbol: str = "000300",
    refresh: bool = False,
    offline: bool = False,
) -> dict[str, Any] | None:
    """加载超额收益基准（默认沪深300）。

    基准只是增强口径，取不到时返回 None，让 walk-forward 继续按绝对收益跑完，
    而不是让整份报告失败。
    """
    try:
        return load_history(
            MarketDataClient(),
            symbol,
            days,
            refresh=refresh,
            offline=offline,
            index=True,
        )
    except (MarketDataError, ValueError):
        return None


def load_all(
    symbols: Iterable[str],
    days: int,
    *,
    refresh: bool = False,
    offline: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    client = MarketDataClient()
    histories: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for symbol in symbols:
        try:
            history = load_history(client, symbol, days, refresh=refresh, offline=offline)
        except (MarketDataError, ValueError) as exc:
            failures.append({"symbol": symbol, "error": str(exc)})
            continue
        if len(history["rows"]) < WARMUP + max(HORIZONS) + 1:
            failures.append(
                {
                    "symbol": symbol,
                    "error": f"历史长度 {len(history['rows'])} 不足，至少需要 {WARMUP + max(HORIZONS) + 1} 个交易日",
                }
            )
            continue
        histories.append(history)
    return histories, failures


# --------------------------------------------------------------------------- #
# 单因子拆解：直接复用 analyzer.score_factors，不再维护影子实现
# --------------------------------------------------------------------------- #
#
# 早期这里独立复刻过一遍九类规则，并在每个评估点断言与 analyze() 一致。
# 现在评分只有 analyzer.score_factors 一个事实来源，复刻逻辑已删除；
# 保留 score_breakdown 作为报告层的适配器（返回 key -> delta）。

FACTOR_LABELS: dict[str, tuple[str, str]] = FACTOR_META


def score_breakdown(history: dict[str, Any]) -> tuple[int, dict[str, int]]:
    score, factors = score_factors(history)
    return score, {factor["key"]: factor["delta"] for factor in factors}



# --------------------------------------------------------------------------- #
# Walk-forward 采样
# --------------------------------------------------------------------------- #


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[int(position)]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def walk_forward(
    history: dict[str, Any],
    *,
    benchmark: dict[str, Any] | None = None,
    horizons: tuple[int, ...] = HORIZONS,
    stride: int = 1,
    warmup: int = WARMUP,
) -> list[dict[str, Any]]:
    """逐个历史交易日打分，并附上事后的未来收益标签。

    第 t 个评估点的入参只有 ``rows[:t+1]``；未来收益由完整序列在外部拼接，
    不会进入 analyze()。传入 ``benchmark`` 时，额外按同一批日期计算基准收益
    与超额收益，用于剥离 beta 和普涨。
    """
    rows = history["rows"]
    closes = [row["close"] for row in rows]
    bench_closes = {
        row["date"]: row["close"]
        for row in (benchmark or {}).get("rows", [])
        if row.get("close")
    }
    span = max(horizons)
    meta = {
        "symbol": history["symbol"],
        "name": history["name"],
        "asset_type": history.get("asset_type", "stock"),
        "data_source": history.get("data_source", "primary"),
    }

    samples: list[dict[str, Any]] = []
    for index in range(warmup - 1, len(rows) - span):
        prefix = {**meta, "rows": rows[: index + 1]}
        result = analyze(prefix)
        parts = {factor["key"]: factor["delta"] for factor in result["factors"]}
        if (index - (warmup - 1)) % stride:
            continue

        entry = closes[index]
        entry_date = rows[index]["date"]
        forward: dict[str, dict[str, float | None]] = {}
        for horizon in horizons:
            exit_price = closes[index + horizon]
            exit_date = rows[index + horizon]["date"]
            stock_return = (exit_price / entry - 1) * 100
            benchmark_return = None
            if bench_closes:
                bench_entry = bench_closes.get(entry_date)
                bench_exit = bench_closes.get(exit_date)
                if bench_entry and bench_exit:
                    benchmark_return = (bench_exit / bench_entry - 1) * 100
            window = closes[index : index + horizon + 1]
            forward[str(horizon)] = {
                "return": stock_return,
                "mdd": max_drawdown(window, horizon + 1),
                "benchmark": benchmark_return,
                "excess": stock_return - benchmark_return if benchmark_return is not None else None,
            }
        samples.append(
            {
                "symbol": meta["symbol"],
                "name": meta["name"],
                "date": entry_date,
                "score": result["score"],
                "view": result["view"],
                "parts": parts,
                "forward": forward,
            }
        )
    return samples


# --------------------------------------------------------------------------- #
# 统计
# --------------------------------------------------------------------------- #


def bucket_key(score: int) -> str:
    for key, low, high, _label, _tone in BUCKETS:
        if low <= score <= high:
            return key
    return "ge75" if score > 100 else "lt30"


def _describe(
    returns: list[float],
    mdds: list[float],
    horizon: int,
    excess: list[float | None] | None = None,
) -> dict[str, Any]:
    clean_excess = [value for value in (excess or []) if value is not None]
    if not returns:
        return {
            "n": 0,
            "independent_windows": 0,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
            "worst": None,
            "best": None,
            "win_rate": None,
            "mdd_mean": None,
            "mdd_worst": None,
            "excess_n": 0,
            "excess_mean": None,
            "excess_median": None,
            "excess_win_rate": None,
        }
    return {
        "n": len(returns),
        # 重叠窗口会让有效样本远小于名义样本，粗略折算成互不重叠的窗口数。
        "independent_windows": max(1, round(len(returns) / max(horizon, 1))),
        "mean": round(statistics.fmean(returns), 2),
        "median": round(statistics.median(returns), 2),
        "p25": round(_percentile(returns, 0.25), 2),
        "p75": round(_percentile(returns, 0.75), 2),
        "worst": round(min(returns), 2),
        "best": round(max(returns), 2),
        "win_rate": round(sum(1 for value in returns if value > 0) / len(returns) * 100, 1),
        "mdd_mean": round(statistics.fmean(mdds), 2) if mdds else None,
        "mdd_worst": round(min(mdds), 2) if mdds else None,
        "excess_n": len(clean_excess),
        "excess_mean": round(statistics.fmean(clean_excess), 2) if clean_excess else None,
        "excess_median": round(statistics.median(clean_excess), 2) if clean_excess else None,
        "excess_win_rate": (
            round(sum(1 for value in clean_excess if value > 0) / len(clean_excess) * 100, 1)
            if clean_excess
            else None
        ),
    }


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2 + 1
        for cursor in range(position, end + 1):
            ranks[order[cursor]] = average
        position = end + 1
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3:
        return None
    mean_l = statistics.fmean(left)
    mean_r = statistics.fmean(right)
    cov = sum((a - mean_l) * (b - mean_r) for a, b in zip(left, right))
    var_l = sum((a - mean_l) ** 2 for a in left)
    var_r = sum((b - mean_r) ** 2 for b in right)
    if not var_l or not var_r:
        return None
    return cov / math.sqrt(var_l * var_r)


def rank_ic(samples: list[dict[str, Any]], horizon: int, metric: str = "return") -> dict[str, Any]:
    """评分与未来收益（或超额收益）的 Spearman 秩相关。

    分数单调有效时 IC 应显著为正。相邻评估点的收益窗口高度重叠，名义样本数会
    大幅高估自由度，因此 t 值用 "有效样本" = 名义样本 / 窗口长度 折算。这样
    得到的显著性更接近用 ``--stride`` 抽样复核时的结论。
    """
    pairs = [
        (float(sample["score"]), sample["forward"][str(horizon)].get(metric))
        for sample in samples
    ]
    pairs = [(score, value) for score, value in pairs if value is not None]
    if len(pairs) < 10:
        return {"ic": None, "n": len(pairs), "n_effective": 0, "t": None, "significant": False}
    scores = _rank([pair[0] for pair in pairs])
    returns = _rank([float(pair[1]) for pair in pairs])
    ic = _pearson(scores, returns)
    n_effective = max(3, round(len(pairs) / max(horizon, 1)))
    if ic is None:
        return {"ic": None, "n": len(pairs), "n_effective": n_effective, "t": None, "significant": False}
    denominator = max(1e-9, 1 - ic * ic)
    t_stat = ic * math.sqrt((n_effective - 2) / denominator) if n_effective > 2 else None
    return {
        "ic": round(ic, 3),
        "n": len(pairs),
        "n_effective": n_effective,
        "t": round(t_stat, 2) if t_stat is not None else None,
        "significant": t_stat is not None and abs(t_stat) >= 1.96,
    }


def summarize(samples: list[dict[str, Any]], horizons: tuple[int, ...]) -> dict[str, Any]:
    bucket_rows: list[dict[str, Any]] = []
    for key, low, high, label, tone in BUCKETS:
        members = [sample for sample in samples if bucket_key(sample["score"]) == key]
        row: dict[str, Any] = {
            "key": key,
            "range": f"{low}–{high}",
            "label": label,
            "tone": tone,
            "n": len(members),
            "share": round(len(members) / len(samples) * 100, 1) if samples else 0.0,
            "score_mean": round(statistics.fmean([m["score"] for m in members]), 1) if members else None,
            "horizons": {},
        }
        for horizon in horizons:
            returns = [m["forward"][str(horizon)]["return"] for m in members]
            mdds = [m["forward"][str(horizon)]["mdd"] for m in members]
            excess = [m["forward"][str(horizon)].get("excess") for m in members]
            row["horizons"][str(horizon)] = _describe(returns, mdds, horizon, excess)
        bucket_rows.append(row)

    ic: dict[str, Any] = {str(horizon): rank_ic(samples, horizon) for horizon in horizons}

    spread: dict[str, Any] = {}
    for horizon in horizons:
        means = [
            row["horizons"][str(horizon)]["mean"]
            for row in bucket_rows
            if row["horizons"][str(horizon)]["n"]
        ]
        pairs = len(means) - 1
        monotone_pairs = sum(1 for a, b in zip(means, means[1:]) if b > a)
        top = bucket_rows[-1]["horizons"][str(horizon)]["mean"]
        bottom = bucket_rows[0]["horizons"][str(horizon)]["mean"]
        spread[str(horizon)] = {
            "top_minus_bottom": round(top - bottom, 2) if top is not None and bottom is not None else None,
            "monotone_pairs": f"{monotone_pairs}/{pairs}" if pairs > 0 else "0/0",
        }

    return {"buckets": bucket_rows, "ic": ic, "spread": spread}


def excess_summary(
    samples: list[dict[str, Any]],
    horizons: tuple[int, ...],
    benchmark: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """分桶的超额收益（个股收益 − 同期基准收益），用于剥离 beta / 普涨。

    绝对收益在单边行情里会把"所有分桶都涨"误读成"评分有效"，超额收益才是
    评分自身排序能力的口径。基准缺失时 available=False，报告只展示绝对收益。
    """
    bucket_rows: list[dict[str, Any]] = []
    for key, low, high, label, tone in BUCKETS:
        members = [sample for sample in samples if bucket_key(sample["score"]) == key]
        row: dict[str, Any] = {
            "key": key,
            "range": f"{low}–{high}",
            "label": label,
            "tone": tone,
            "n": len(members),
            "share": round(len(members) / len(samples) * 100, 1) if samples else 0.0,
            "horizons": {},
        }
        for horizon in horizons:
            values = [
                member["forward"][str(horizon)].get("excess")
                for member in members
                if member["forward"][str(horizon)].get("excess") is not None
            ]
            row["horizons"][str(horizon)] = {
                "n": len(values),
                "mean": round(statistics.fmean(values), 2) if values else None,
                "median": round(statistics.median(values), 2) if values else None,
                "win_rate": (
                    round(sum(1 for value in values if value > 0) / len(values) * 100, 1)
                    if values
                    else None
                ),
            }
        bucket_rows.append(row)

    ic = {str(horizon): rank_ic(samples, horizon, "excess") for horizon in horizons}
    spread: dict[str, Any] = {}
    for horizon in horizons:
        means = [
            row["horizons"][str(horizon)]["mean"]
            for row in bucket_rows
            if row["horizons"][str(horizon)]["n"]
        ]
        top = bucket_rows[-1]["horizons"][str(horizon)]["mean"]
        bottom = bucket_rows[0]["horizons"][str(horizon)]["mean"]
        spread[str(horizon)] = {
            "top_minus_bottom": round(top - bottom, 2) if top is not None and bottom is not None else None,
            "monotone_pairs": (
                f"{sum(1 for a, b in zip(means, means[1:]) if b > a)}/{len(means) - 1}"
                if len(means) > 1
                else "0/0"
            ),
        }

    available = benchmark is not None and any(
        row["horizons"][str(horizon)]["n"] for row in bucket_rows for horizon in horizons
    )
    return {
        "available": available,
        "benchmark": (
            {"symbol": benchmark["symbol"], "name": benchmark["name"]} if benchmark else None
        ),
        "buckets": bucket_rows,
        "ic": ic,
        "spread": spread,
    }


def factor_table(
    samples: list[dict[str, Any]],
    horizon: int = 20,
    metric: str = "return",
) -> list[dict[str, Any]]:
    """单因子边际贡献：因子触发时，未来收益比未触发好多少。

    edge = mean(收益 | 该因子加分) − mean(收益 | 该因子扣分)。
    edge ≤ 0 说明这个条件在历史上没有区分度，甚至方向相反，是权重需要
    重新审视的直接证据。

    有两类因子天然是单边的（如回撤只扣分、不加分），此时退化为
    "未触发 vs 扣分" 的对比，并在 basis 里注明口径，避免用 0 样本算差值。
    """
    rows: list[dict[str, Any]] = []
    for key, (label, weight) in FACTOR_LABELS.items():
        positive: list[float] = []
        negative: list[float] = []
        neutral: list[float] = []
        for sample in samples:
            value = sample["forward"][str(horizon)].get(metric)
            if value is None:
                continue
            delta = sample["parts"].get(key, 0)
            if delta > 0:
                positive.append(value)
            elif delta < 0:
                negative.append(value)
            else:
                neutral.append(value)

        edge = None
        basis = "—"
        if len(positive) >= 20 and len(negative) >= 20:
            edge = statistics.fmean(positive) - statistics.fmean(negative)
            basis = "加分 vs 扣分"
        elif len(negative) >= 20 and len(neutral) >= 20:
            # 单边扣分因子：未被扣分时是否真的更好。
            edge = statistics.fmean(neutral) - statistics.fmean(negative)
            basis = "未触发 vs 扣分"
        elif len(positive) >= 20 and len(neutral) >= 20:
            edge = statistics.fmean(positive) - statistics.fmean(neutral)
            basis = "加分 vs 未触发"

        if edge is None:
            verdict = "样本不足"
        elif edge >= 1.0:
            verdict = "方向有效"
        elif edge > 0:
            verdict = "弱有效"
        elif edge <= -1.0:
            verdict = "方向相反"
        else:
            verdict = "接近噪声"

        rows.append(
            {
                "key": key,
                "label": label,
                "weight": weight,
                "n_positive": len(positive),
                "n_negative": len(negative),
                "n_neutral": len(neutral),
                "return_positive": round(statistics.fmean(positive), 2) if positive else None,
                "return_negative": round(statistics.fmean(negative), 2) if negative else None,
                "return_neutral": round(statistics.fmean(neutral), 2) if neutral else None,
                "edge": round(edge, 2) if edge is not None else None,
                "basis": basis,
                "hit_positive": round(sum(1 for v in positive if v > 0) / len(positive) * 100, 1) if positive else None,
                "hit_negative": round(sum(1 for v in negative if v > 0) / len(negative) * 100, 1) if negative else None,
                "verdict": verdict,
            }
        )
    rows.sort(key=lambda row: -abs(row["edge"]) if row["edge"] is not None else 0)
    return rows


def quantile_groups(
    samples: list[dict[str, Any]],
    horizons: tuple[int, ...] = HORIZONS,
    groups: int = 5,
) -> list[dict[str, Any]]:
    """按评分排序五等分，绕开固定阈值造成的样本倾斜，再查一次单调性。"""
    ordered = sorted(samples, key=lambda sample: (sample["score"], sample["date"]))
    count = len(ordered)
    if count < groups * 10:
        return []
    result: list[dict[str, Any]] = []
    for group in range(groups):
        start = round(group * count / groups)
        end = round((group + 1) * count / groups)
        members = ordered[start:end]
        if not members:
            continue
        entry: dict[str, Any] = {
            "group": f"Q{group + 1}",
            "score_low": min(member["score"] for member in members),
            "score_high": max(member["score"] for member in members),
            "n": len(members),
            "horizons": {},
        }
        for horizon in horizons:
            returns = [member["forward"][str(horizon)]["return"] for member in members]
            entry["horizons"][str(horizon)] = {
                "mean": round(statistics.fmean(returns), 2),
                "win_rate": round(sum(1 for value in returns if value > 0) / len(returns) * 100, 1),
            }
        result.append(entry)
    return result


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #


def per_symbol_stats(
    samples: list[dict[str, Any]],
    histories: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """分标的核对：合并样本的相关性可能只反映品种间差异，而非标的内的择时能力。"""
    rows: list[dict[str, Any]] = []
    for history in histories:
        symbol = history["symbol"]
        members = [sample for sample in samples if sample["symbol"] == symbol]
        if not members:
            continue
        stats = summarize(members, (20, 60))
        top = stats["buckets"][-1]
        bottom = stats["buckets"][0]

        def spread(horizon: str) -> float | None:
            high = top["horizons"][horizon]["mean"]
            low = bottom["horizons"][horizon]["mean"]
            if high is None or low is None:
                return None
            return round(high - low, 2)

        ic20 = rank_ic(members, 20)["ic"]
        ic60 = rank_ic(members, 60)["ic"]
        rows.append(
            {
                "symbol": symbol,
                "name": history["name"],
                "asset_type": history.get("asset_type", "stock"),
                "n": len(members),
                "score_mean": round(statistics.fmean([m["score"] for m in members]), 1),
                "ic20": ic20,
                "ic60": ic60,
                "spread20": spread("20"),
                "spread60": spread("60"),
                "win_top20": top["horizons"]["20"]["win_rate"],
                "n_top": top["n"],
                "verdict": "同向" if (ic20 or 0) > 0 else "反向",
            }
        )
    return rows


def consistency_check(rows: list[dict[str, Any]], pooled_ic20: float | None) -> dict[str, Any]:
    positive20 = sum(1 for row in rows if (row["ic20"] or 0) > 0)
    positive60 = sum(1 for row in rows if (row["ic60"] or 0) > 0)
    total = len(rows)
    divergent = False
    if total and pooled_ic20 is not None:
        divergent = (pooled_ic20 > 0 and positive20 * 2 < total) or (
            pooled_ic20 < 0 and positive20 * 2 > total
        )
    return {
        "symbols": total,
        "positive_ic20": positive20,
        "positive_ic60": positive60,
        "divergent": divergent,
    }


def build_report(
    histories: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    *,
    days: int,
    stride: int,
    failures: list[dict[str, str]],
    benchmark: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dates = sorted({sample["date"] for sample in samples})
    summary = summarize(samples, HORIZONS)
    symbol_rows = per_symbol_stats(samples, histories)
    factors = factor_table(samples, 20)
    excess_factors = {row["key"]: row for row in factor_table(samples, 20, "excess")}
    for row in factors:
        peer = excess_factors.get(row["key"], {})
        row["excess_edge"] = peer.get("edge")
        row["excess_basis"] = peer.get("basis", "—")
        row["excess_verdict"] = peer.get("verdict", "样本不足")
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "params": {
            "symbols": [history["symbol"] for history in histories],
            "days": days,
            "stride": stride,
            "warmup": WARMUP,
            "horizons": list(HORIZONS),
            "score_base": 50,
            "benchmark": benchmark["symbol"] if benchmark else None,
        },
        "coverage": {
            "symbols": len(histories),
            "samples": len(samples),
            "trading_days": len(dates),
            "start_date": dates[0] if dates else None,
            "end_date": dates[-1] if dates else None,
            "failures": failures,
        },
        "buckets": summary["buckets"],
        "ic": summary["ic"],
        "spread": summary["spread"],
        "excess": excess_summary(samples, HORIZONS, benchmark),
        "quantiles": quantile_groups(samples),
        "per_symbol_detail": symbol_rows,
        "consistency": consistency_check(symbol_rows, summary["ic"]["20"]["ic"]),
        "factors": factors,
        "per_symbol": [
            {
                "symbol": history["symbol"],
                "name": history["name"],
                "asset_type": history.get("asset_type", "stock"),
                "rows": len(history["rows"]),
                "first": history["rows"][0]["date"],
                "last": history["rows"][-1]["date"],
                "samples": sum(1 for sample in samples if sample["symbol"] == history["symbol"]),
            }
            for history in histories
        ],
    }


def _fmt(value: float | None, suffix: str = "", digits: int = 2, signed: bool = True) -> str:
    if value is None:
        return "—"
    template = f"{{:+.{digits}f}}" if signed else f"{{:.{digits}f}}"
    return template.format(value) + suffix


def _cell(value: float | None, suffix: str = "", invert: bool = False) -> str:
    """收益类数值：正为涨（红，沿用项目 .up 约定），负为跌（绿）。"""
    if value is None:
        return '<td class="num">—</td>'
    tone = "wf-up" if value > 0 else "wf-down" if value < 0 else ""
    if invert:
        tone = ""
    return f'<td class="num {tone}">{_fmt(value, suffix)}</td>'


def render_markdown(report: dict[str, Any]) -> str:
    coverage = report["coverage"]
    params = report["params"]
    lines: list[str] = [
        "# 评分有效性验证（walk-forward）",
        "",
        f"生成时间：{report['generated_at']}　样本：{coverage['symbols']} 只标的、"
        f"{coverage['trading_days']} 个交易日、{coverage['samples']} 个评估点　"
        f"区间：{coverage['start_date']} → {coverage['end_date']}",
        "",
        f"规则：每个交易日只用截至当日的数据调用 `analyze()` 打分（预热 {params['warmup']} 根 K 线，"
        f"步长 {params['stride']}），收益标签在其后单独计算。",
        "",
        "## 评分分桶 × 未来收益",
        "",
        "| 评分分桶 | 研判 | 样本 | 5 日均值 | 5 日胜率 | 5 日回撤 | 20 日均值 | 20 日胜率 | 20 日回撤 | 60 日均值 | 60 日胜率 | 60 日回撤 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["buckets"]:
        cells = [f"| {row['range']} | {row['label']} | {row['n']} "]
        for horizon in ("5", "20", "60"):
            stat = row["horizons"][horizon]
            cells.append(
                f"| {_fmt(stat['mean'], '%')} | {_fmt(stat['win_rate'], '%', 1, signed=False)} "
                f"| {_fmt(stat['mdd_mean'], '%')} "
            )
        lines.append("".join(cells) + "|")

    lines += ["", "## 单调性与排序能力", ""]
    for horizon in ("5", "20", "60"):
        ic = report["ic"][horizon]
        spread = report["spread"][horizon]
        lines.append(
            f"- 未来 {horizon} 日：Spearman IC = {_fmt(ic['ic'], '', 3)}"
            f"（有效样本 ≈ {ic['n_effective']}，t = {ic['t']}，{'显著' if ic['significant'] else '不显著'}），"
            f"最高分桶 − 最低分桶 = {_fmt(spread['top_minus_bottom'], '%')}，"
            f"相邻分桶单调递增 {spread['monotone_pairs']}。"
        )

    if report.get("quantiles"):
        lines += ["", "## 分位数交叉验证（五等分）", ""]
        for horizon in ("5", "20", "60"):
            series = [_fmt(item["horizons"][horizon]["mean"], "%") for item in report["quantiles"]]
            low = report["quantiles"][0]["horizons"][horizon]["mean"]
            high = report["quantiles"][-1]["horizons"][horizon]["mean"]
            lines.append(
                f"- 未来 {horizon} 日：Q1 → Q5 = {' → '.join(series)}，首尾差 {_fmt(high - low, '%')}。"
            )

    excess = report.get("excess") or {}
    if excess.get("available"):
        benchmark = excess["benchmark"]
        lines += [
            "",
            f"## 超额收益验证（相对 {benchmark['name']}）",
            "",
            "| 评分分桶 | 样本 | 5 日超额 | 5 日胜率 | 20 日超额 | 20 日胜率 | 60 日超额 | 60 日胜率 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in excess["buckets"]:
            cells = [f"| {row['range']} | {row['n']} "]
            for horizon in ("5", "20", "60"):
                stat = row["horizons"][horizon]
                cells.append(
                    f"| {_fmt(stat['mean'], '%')} | {_fmt(stat['win_rate'], '%', 1, signed=False)} "
                )
            lines.append("".join(cells) + "|")

        lines.append("")
        for horizon in ("5", "20", "60"):
            ic = excess["ic"][horizon]
            spread = excess["spread"][horizon]
            lines.append(
                f"- 未来 {horizon} 日超额：Spearman IC = {_fmt(ic['ic'], '', 3)}"
                f"（t = {ic['t']}，{'显著' if ic['significant'] else '不显著'}），"
                f"最高桶 − 最低桶 = {_fmt(spread['top_minus_bottom'], '%')}，"
                f"单调 {spread['monotone_pairs']}。"
            )

    if report.get("per_symbol_detail"):
        consistency = report["consistency"]
        note = (
            "合并口径与分标的结论方向相反：正相关主要来自品种间差异（不同标的的评分中枢不同），"
            "不能直接当作标的内部的择时能力。"
            if consistency["divergent"]
            else "合并口径与分标的结论方向基本一致。"
        )
        lines += [
            "",
            "## 分标的复核",
            "",
            f"{consistency['positive_ic20']}/{consistency['symbols']} 只标的的未来 20 日 IC 为正"
            f"（60 日 {consistency['positive_ic60']}/{consistency['symbols']}）。{note}",
            "",
            "| 标的 | 评分均值 | 样本 | IC(20 日) | IC(60 日) | 首尾差(20 日) | 首尾差(60 日) | 最高桶 20 日胜率 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in report["per_symbol_detail"]:
            lines.append(
                f"| {row['name']}（{row['symbol']}） | {row['score_mean']} | {row['n']} | "
                f"{_fmt(row['ic20'], '', 3)} | {_fmt(row['ic60'], '', 3)} | "
                f"{_fmt(row['spread20'], '%')} | {_fmt(row['spread60'], '%')} | "
                f"{_fmt(row['win_top20'], '%', 1, signed=False)}（n={row['n_top']}） |"
            )

    lines += [
        "",
        "## 单因子边际贡献（未来 20 日）",
        "",
        "| 因子 | 当前权重 | 加分样本 | 加分收益 | 扣分样本 | 扣分收益 | edge | 超额 edge | 口径 | 结论 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in report["factors"]:
        lines.append(
            f"| {row['label']} | {row['weight']} | {row['n_positive']} | "
            f"{_fmt(row['return_positive'], '%')} | {row['n_negative']} | "
            f"{_fmt(row['return_negative'], '%')} | {_fmt(row['edge'], '%')} | "
            f"{_fmt(row.get('excess_edge'), '%')} | "
            f"{row['basis']} | {row['verdict']} |"
        )

    if coverage["failures"]:
        lines += ["", "## 未纳入的标的", ""]
        for item in coverage["failures"]:
            lines.append(f"- `{item['symbol']}`：{item['error']}")

    lines += [
        "",
        "> 说明：绝对收益未剔除同期指数涨跌；超额收益为个股收益减去同期基准收益，"
        "更接近评分自身的排序能力。两者均以评估日收盘价为入场价，"
        "未计入次日开盘的执行偏差；相邻评估点的收益窗口高度重叠，"
        "IC 的 t 值按「名义样本 ÷ 窗口长度」折算出有效样本后再计算，胜率与均值只作方向性比较，"
        "不作为收益预测。",
        "",
    ]
    return "\n".join(lines)


def render_html(report: dict[str, Any]) -> str:
    """可直接粘贴进 index.html 的 methodology 区块（自带作用域样式）。"""
    coverage = report["coverage"]
    params = report["params"]
    rows: list[str] = []
    for row in report["buckets"]:
        cells = [
            f'<td class="wf-bucket"><strong>{row["range"]}</strong><span>{row["label"]}</span></td>',
            f'<td class="num">{row["n"]}</td>',
        ]
        for horizon in ("5", "20", "60"):
            stat = row["horizons"][horizon]
            cells.append(_cell(stat["mean"], "%"))
            cells.append(f'<td class="num">{_fmt(stat["win_rate"], "%", 1, signed=False)}</td>')
            cells.append(_cell(stat["mdd_mean"], "%"))
        rows.append("<tr>" + "".join(cells) + "</tr>")

    ic_items = []
    for horizon in ("5", "20", "60"):
        ic = report["ic"][horizon]
        spread = report["spread"][horizon]
        ic_items.append(
            f'<li><b>未来 {horizon} 日：</b>IC {_fmt(ic["ic"], "", 3)}'
            f'（有效样本 ≈ {ic["n_effective"]}，{"显著" if ic["significant"] else "不显著"}），'
            f'最高桶 − 最低桶 {_fmt(spread["top_minus_bottom"], "%")}，'
            f'单调 {spread["monotone_pairs"]}</li>'
        )

    factor_rows = []
    for row in report["factors"]:
        basis = "" if row["basis"] in ("加分 vs 扣分", "—") else f'<span class="wf-mono">{row["basis"]}</span>'
        factor_rows.append(
            "<tr>"
            f'<td>{row["label"]}</td>'
            f'<td class="num wf-mono">{row["weight"]}</td>'
            f'<td class="num">{row["n_positive"]}</td>'
            f'{_cell(row["return_positive"], "%")}'
            f'<td class="num">{row["n_negative"]}</td>'
            f'{_cell(row["return_negative"], "%")}'
            f'{_cell(row["edge"], "%")}'
            f'{_cell(row.get("excess_edge"), "%")}'
            f'<td class="wf-verdict">{row["verdict"]}{basis}</td>'
            "</tr>"
        )

    quintile_item = ""
    if report.get("quantiles"):
        gaps = []
        for horizon in ("5", "20", "60"):
            low = report["quantiles"][0]["horizons"][horizon]["mean"]
            high = report["quantiles"][-1]["horizons"][horizon]["mean"]
            gaps.append(f"未来 {horizon} 日 {_fmt(high - low, '%')}")
        quintile_item = (
            f'<li><b>五等分交叉验证：</b>Q1（最低分）至 Q5（最高分）的收益差为 {"，".join(gaps)}</li>'
        )

    excess = report.get("excess") or {}
    excess_html = ""
    if excess.get("available"):
        benchmark = excess["benchmark"]
        ex_rows = []
        for row in excess["buckets"]:
            cells = [
                f'<td class="wf-bucket"><strong>{row["range"]}</strong><span>{row["label"]}</span></td>',
                f'<td class="num">{row["n"]}</td>',
            ]
            for horizon in ("5", "20", "60"):
                stat = row["horizons"][horizon]
                cells.append(_cell(stat["mean"], "%"))
                cells.append(f'<td class="num">{_fmt(stat["win_rate"], "%", 1, signed=False)}</td>')
            ex_rows.append("<tr>" + "".join(cells) + "</tr>")
        ex_notes = []
        for horizon in ("5", "20", "60"):
            ic = excess["ic"][horizon]
            spread = excess["spread"][horizon]
            ex_notes.append(
                f'<li><b>未来 {horizon} 日超额：</b>IC {_fmt(ic["ic"], "", 3)}'
                f'（t={ic["t"]}，{"显著" if ic["significant"] else "不显著"}），'
                f'最高桶 − 最低桶 {_fmt(spread["top_minus_bottom"], "%")}，'
                f'单调 {spread["monotone_pairs"]}</li>'
            )
        excess_html = f"""<div class="wf-factor">
  <h3>超额收益（相对 {benchmark['name']}）</h3>
  <p class="wf-sub">个股收益减去同期基准收益，用于剥离 beta 和普涨；绝对收益分桶全部为正也可能只是行情好。</p>
  <div class="wf-scroll">
    <table>
      <thead>
        <tr>
          <th rowspan="2" scope="col">评分分桶</th>
          <th rowspan="2" scope="col">样本</th>
          <th colspan="2" scope="colgroup">未来 5 日</th>
          <th colspan="2" scope="colgroup">未来 20 日</th>
          <th colspan="2" scope="colgroup">未来 60 日</th>
        </tr>
        <tr>
          <th scope="col">超额</th><th scope="col">胜率</th>
          <th scope="col">超额</th><th scope="col">胜率</th>
          <th scope="col">超额</th><th scope="col">胜率</th>
        </tr>
      </thead>
      <tbody>
        {''.join(ex_rows)}
      </tbody>
    </table>
  </div>
  <ul class="wf-notes">{''.join(ex_notes)}</ul>
</div>"""

    symbol_rows = []
    for row in report.get("per_symbol_detail", []):
        symbol_rows.append(
            "<tr>"
            f'<td>{row["name"]}<span class="wf-code">{row["symbol"]}</span></td>'
            f'<td class="num">{row["score_mean"]}</td>'
            f'{_cell(row["ic20"], "", invert=True)}'
            f'{_cell(row["ic60"], "", invert=True)}'
            f'{_cell(row["spread20"], "%")}'
            f'{_cell(row["spread60"], "%")}'
            f'<td class="num">{_fmt(row["win_top20"], "%", 1, signed=False)}'
            f'<span class="wf-code">n={row["n_top"]}</span></td>'
            "</tr>"
        )

    consistency = report.get("consistency") or {}
    warn_html = ""
    if consistency.get("divergent"):
        warn_html = (
            f'<p class="wf-warn">注意：合并样本的排序能力为正，但仅 '
            f'{consistency["positive_ic20"]}/{consistency["symbols"]} 只标的自身为正相关。'
            f"这说明合并口径的优势主要来自品种间差异，不宜当作单只标的内的择时能力。</p>"
        )

    return f"""<section class="wf" aria-labelledby="wfTitle">
  <style>
    .wf {{ margin: 0; padding: 0; color: var(--ink, #16221d); }}
    .wf h3 {{ margin: 0 0 6px; font-size: 17px; letter-spacing: .01em; }}
    .wf .wf-sub {{ margin: 0 0 14px; color: var(--muted, #6e7b74); font-size: 12.5px; line-height: 1.7; }}
    .wf .wf-scroll {{ overflow-x: auto; border: 1px solid var(--line, #d8ddd6); border-radius: 8px; background: var(--panel, rgba(255,255,252,.84)); }}
    .wf table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; white-space: nowrap; }}
    .wf caption {{ caption-side: top; text-align: left; padding: 10px 12px 6px; color: var(--muted, #6e7b74); font-size: 11.5px; white-space: normal; line-height: 1.6; }}
    .wf thead th {{ padding: 8px 10px; color: var(--muted, #6e7b74); font-weight: 600; font-size: 11px; letter-spacing: .04em; border-bottom: 1px solid var(--line, #d8ddd6); background: rgba(23,107,77,.04); }}
    .wf thead tr:first-child th {{ border-bottom: 1px solid var(--line, #d8ddd6); }}
    .wf tbody td {{ padding: 8px 10px; border-bottom: 1px solid rgba(216,221,214,.7); }}
    .wf tbody tr:last-child td {{ border-bottom: 0; }}
    .wf tbody tr:hover td {{ background: rgba(23,107,77,.045); }}
    .wf .num {{ text-align: right; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-variant-numeric: tabular-nums; }}
    .wf .wf-bucket {{ display: flex; flex-direction: column; gap: 1px; }}
    .wf .wf-bucket span {{ color: var(--muted, #6e7b74); font-size: 11px; }}
    .wf .wf-up {{ color: var(--red, #b34938); }}
    .wf .wf-down {{ color: var(--green, #176b4d); }}
    .wf .wf-mono {{ color: var(--muted, #6e7b74); }}
    .wf .wf-verdict {{ color: var(--ink, #16221d); }}
    .wf .wf-verdict span {{ display: block; margin-top: 1px; font-size: 10.5px; }}
    .wf .wf-code {{ display: block; margin-top: 1px; color: var(--muted, #6e7b74); font: 700 10px/1.4 ui-monospace, Menlo, monospace; letter-spacing: .04em; }}
    .wf .wf-sym td:first-child {{ white-space: normal; min-width: 132px; max-width: 190px; }}
    .wf .wf-warn {{ margin: 14px 0 0; padding: 9px 12px; border-left: 3px solid var(--amber, #aa7627); border-radius: 0 6px 6px 0; background: rgba(170,118,39,.09); color: #7a5418; font-size: 12px; line-height: 1.75; }}
    .wf .wf-notes {{ margin: 14px 0 0; padding-left: 18px; color: var(--muted, #6e7b74); font-size: 12px; line-height: 1.85; }}
    .wf .wf-note {{ margin: 12px 0 0; color: var(--muted, #6e7b74); font-size: 11.5px; line-height: 1.8; }}
    .wf .wf-factor {{ margin-top: 18px; }}
  </style>
  <h3 id="wfTitle">评分有效性验证</h3>
  <p class="wf-sub">每个交易日仅用截至当日的数据重新打分，再回看未来 5 / 20 / 60 日表现。
    {coverage['symbols']} 只标的 · {coverage['trading_days']} 个交易日 · {coverage['samples']} 个评估点 ·
    {coverage['start_date']} → {coverage['end_date']}</p>
  <div class="wf-scroll">
    <table>
      <caption>分桶口径：{' / '.join(f"{row['range']}={row['label']}" for row in report['buckets'])}</caption>
      <thead>
        <tr>
          <th rowspan="2" scope="col">评分分桶</th>
          <th rowspan="2" scope="col">样本</th>
          <th colspan="3" scope="colgroup">未来 5 日</th>
          <th colspan="3" scope="colgroup">未来 20 日</th>
          <th colspan="3" scope="colgroup">未来 60 日</th>
        </tr>
        <tr>
          <th scope="col">均值</th><th scope="col">胜率</th><th scope="col">回撤</th>
          <th scope="col">均值</th><th scope="col">胜率</th><th scope="col">回撤</th>
          <th scope="col">均值</th><th scope="col">胜率</th><th scope="col">回撤</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </div>
  <ul class="wf-notes">
    {''.join(ic_items)}
    {quintile_item}
  </ul>
  {warn_html}
  {excess_html}
  <div class="wf-factor">
    <h3>分标的复核</h3>
    <p class="wf-sub">同一套评分放在不同品种上可能指向相反方向，这里逐只核对排序能力。</p>
    <div class="wf-scroll">
      <table class="wf-sym">
        <thead>
          <tr>
            <th scope="col">标的</th><th scope="col">评分均值</th>
            <th scope="col">IC（20 日）</th><th scope="col">IC（60 日）</th>
            <th scope="col">首尾差（20 日）</th><th scope="col">首尾差（60 日）</th>
            <th scope="col">最高桶 20 日胜率</th>
          </tr>
        </thead>
        <tbody>
          {''.join(symbol_rows)}
        </tbody>
      </table>
    </div>
  </div>
  <div class="wf-factor">
    <h3>单因子边际贡献（未来 20 日）</h3>
    <p class="wf-sub">加分时的平均收益 − 扣分时的平均收益。edge 越大，说明该条件越能区分后续走势。</p>
    <div class="wf-scroll">
      <table>
        <thead>
          <tr>
            <th scope="col">因子</th><th scope="col">当前权重</th>
            <th scope="col">加分样本</th><th scope="col">加分收益</th>
            <th scope="col">扣分样本</th><th scope="col">扣分收益</th>
            <th scope="col">edge</th><th scope="col">超额 edge</th><th scope="col">结论</th>
          </tr>
        </thead>
        <tbody>
          {''.join(factor_rows)}
        </tbody>
      </table>
    </div>
  </div>
  <p class="wf-note">回撤为该周期窗口内收盘价的最大回撤均值；胜率为未来收益大于 0 的比例。
    绝对收益未剔除同期指数涨跌，超额收益为个股收益减去同期基准收益，以评估日收盘价为入场价；
    相邻评估点窗口重叠，IC 的 t 值按有效样本折算，指标只用于方向性比较。
    统计区间 {coverage['start_date']} 至 {coverage['end_date']}，生成于 {report['generated_at']}，预热 {params['warmup']} 根 K 线。</p>
</section>
"""


def render_preview(report: dict[str, Any]) -> str:
    """独立预览页：把片段套上项目同色板，便于粘贴前先看效果。"""
    fragment = render_html(report)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>评分有效性表 · 预览</title>
<style>
  :root {{
    --ink: #16221d; --muted: #6e7b74; --paper: #f4f2eb; --panel: rgba(255,255,252,.84);
    --line: #d8ddd6; --green: #176b4d; --green-soft: #dceade; --red: #b34938;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--paper); color: var(--ink);
    font: 15px/1.65 -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; }}
  .wrap {{ max-width: 1040px; margin: 0 auto; padding: 36px 24px 56px; }}
  .mock-panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
    padding: 22px 22px 18px; box-shadow: 0 18px 50px rgba(31,48,40,.08); }}
  .mock-label {{ color: var(--green); font: 700 11px/1.4 ui-monospace, Menlo, monospace;
    letter-spacing: .18em; text-transform: uppercase; margin-bottom: 4px; }}
  .mock-title {{ margin: 0 0 18px; font-size: 20px; font-weight: 650; }}
</style>
</head>
<body>
  <div class="wrap">
    <div class="mock-panel">
      <div class="mock-label">METHODOLOGY</div>
      <h2 class="mock-title">评分方法与有效性</h2>
      {fragment}
    </div>
  </div>
</body>
</html>
"""


def write_reports(report: dict[str, Any], outdir: Path = REPORT_DIR) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": outdir / "walkforward.json",
        "markdown": outdir / "walkforward.md",
        "html": outdir / "walkforward_table.html",
        "preview": outdir / "walkforward_preview.html",
    }
    paths["json"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["markdown"].write_text(render_markdown(report), encoding="utf-8")
    paths["html"].write_text(render_html(report), encoding="utf-8")
    paths["preview"].write_text(render_preview(report), encoding="utf-8")
    return paths


def run(
    symbols: Iterable[str] = DEFAULT_SYMBOLS,
    *,
    days: int = 600,
    stride: int = 1,
    refresh: bool = False,
    offline: bool = False,
    benchmark_symbol: str = "000300",
) -> dict[str, Any]:
    histories, failures = load_all(symbols, days, refresh=refresh, offline=offline)
    if not histories:
        raise MarketDataError("没有任何标的可用，请检查网络或先联网跑一次生成缓存")
    benchmark = load_benchmark(
        days,
        symbol=benchmark_symbol,
        refresh=refresh,
        offline=offline,
    )
    if benchmark is None:
        print("提示：未取得基准行情，超额收益口径留空，仅输出绝对收益。", file=sys.stderr)
    samples: list[dict[str, Any]] = []
    for history in histories:
        samples.extend(walk_forward(history, benchmark=benchmark, stride=stride))
    samples.sort(key=lambda sample: sample["date"])
    return build_report(
        histories,
        samples,
        days=days,
        stride=stride,
        failures=failures,
        benchmark=benchmark,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="评分有效性 walk-forward 验证（离线）")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="逗号分隔的 6 位代码")
    parser.add_argument("--days", type=int, default=600, help="每只标的取多少根日线（80–600）")
    parser.add_argument("--stride", type=int, default=1, help="评估步长，>1 可降低重叠样本")
    parser.add_argument("--refresh", action="store_true", help="忽略缓存重新拉取行情")
    parser.add_argument("--offline", action="store_true", help="只用本地缓存，不访问网络")
    parser.add_argument("--benchmark", default="000300", help="超额收益基准代码，默认沪深300")
    parser.add_argument("--outdir", default=str(REPORT_DIR), help="报告输出目录")
    args = parser.parse_args(argv)

    symbols = [item.strip() for item in args.symbols.split(",") if item.strip()]
    try:
        report = run(
            symbols,
            days=max(WARMUP + max(HORIZONS) + 1, min(args.days, 600)),
            stride=max(1, args.stride),
            refresh=args.refresh,
            offline=args.offline,
            benchmark_symbol=args.benchmark,
        )
    except MarketDataError as exc:
        print(f"验证失败：{exc}", file=sys.stderr)
        return 2

    paths = write_reports(report, Path(args.outdir))
    coverage = report["coverage"]
    print(
        f"评估点 {coverage['samples']} 个，{coverage['symbols']} 只标的，"
        f"{coverage['start_date']} → {coverage['end_date']}"
    )
    for horizon in ("5", "20", "60"):
        ic = report["ic"][horizon]
        print(f"  未来 {horizon} 日 IC = {ic['ic']}（t={ic['t']}）")
    excess = report.get("excess") or {}
    if excess.get("available"):
        print(f"  超额收益基准：{excess['benchmark']['name']}")
        for horizon in ("5", "20", "60"):
            ic = excess["ic"][horizon]
            print(f"  未来 {horizon} 日超额 IC = {ic['ic']}（t={ic['t']}）")
    for key, path in paths.items():
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
