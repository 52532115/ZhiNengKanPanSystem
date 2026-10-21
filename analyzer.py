"""Deterministic, explainable analysis for A-shares and listed funds."""

from __future__ import annotations

import math
import statistics
from typing import Any


def sma(values: list[float], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= period:
            running -= values[index - period]
        if index >= period - 1:
            result[index] = running / period
    return result


def ema(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    alpha = 2 / (period + 1)
    result = [values[0]]
    for value in values[1:]:
        result.append(alpha * value + (1 - alpha) * result[-1])
    return result


def rsi(values: list[float], period: int = 14) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    if len(values) <= period:
        return result
    gains = [max(values[i] - values[i - 1], 0) for i in range(1, len(values))]
    losses = [max(values[i - 1] - values[i], 0) for i in range(1, len(values))]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    result[period] = 100 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    for i in range(period + 1, len(values)):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        result[i] = 100 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    return result


def pct_return(values: list[float], days: int) -> float | None:
    if len(values) <= days or values[-days - 1] == 0:
        return None
    return (values[-1] / values[-days - 1] - 1) * 100


def annualized_volatility(values: list[float], days: int = 60) -> float:
    sample = values[-(days + 1) :]
    returns = [sample[i] / sample[i - 1] - 1 for i in range(1, len(sample)) if sample[i - 1]]
    return statistics.stdev(returns) * math.sqrt(252) * 100 if len(returns) > 1 else 0.0


def max_drawdown(values: list[float], days: int = 120) -> float:
    peak = 0.0
    worst = 0.0
    for value in values[-days:]:
        peak = max(peak, value)
        if peak:
            worst = min(worst, value / peak - 1)
    return worst * 100


def _round(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(value, digits)


# 九类评分规则的唯一事实来源：标签、权重、打分逻辑都只在这里维护。
# 权重顺序与 score_factors 的分支一一对应：主加分 / 主扣分 / 次扣分。
BASE_SCORE = 50

FACTOR_LABELS: dict[str, str] = {
    "trend": "价格 vs MA20",
    "align": "MA20 vs MA60",
    "long": "价格 vs MA120",
    "rs": "相对强弱（20 日）",
    "macd": "MACD 柱与斜率",
    "rsi": "RSI 区间",
    "ret20": "近 20 日涨幅",
    "volume": "量比 + 当日涨跌",
    "volatility": "年化波动率",
    "drawdown": "近 120 日回撤",
}

# v1：最初的拍脑袋权重，保留用于复现和对照。
WEIGHTS_V1: dict[str, tuple[int, ...]] = {
    "trend": (9, -9),
    "align": (9, -7),
    "long": (6, -6),
    "macd": (8, -7),
    "rsi": (7, -6, -4),
    "ret20": (5, -6, -3),
    "volume": (4, -5),
    "volatility": (4, -8),
    "drawdown": (-6,),
}

# v2：按超额收益单因子 edge 重估——下调 RSI / 均线对齐等噪声项，
# 上调波动率、MACD 与 MA120 等方向更稳定的项；总振幅同时收窄。
WEIGHTS_V2: dict[str, tuple[int, ...]] = {
    "trend": (8, -8),
    "align": (3, -3),
    "long": (5, -5),
    "macd": (9, -8),
    "rsi": (3, -3, -2),
    "ret20": (3, -4, -2),
    "volume": (3, -3),
    "volatility": (5, -9),
    "drawdown": (-7,),
}

# v3：在 v2 基础上加入相对强弱因子（20 日超额收益 ±3% 触发，±6 分）。
# 基准行情通过 analyze(benchmark=...) 注入；缺失时该因子记 0 分。
WEIGHTS_V3: dict[str, tuple[int, ...]] = {
    **WEIGHTS_V2,
    "rs": (6, -6),
}

WEIGHT_PROFILES: dict[str, dict[str, tuple[int, ...]]] = {
    "v1": WEIGHTS_V1,
    "v2": WEIGHTS_V2,
    "v3": WEIGHTS_V3,
}
DEFAULT_PROFILE = "v3"
DEFAULT_WEIGHTS = WEIGHT_PROFILES[DEFAULT_PROFILE]


def weight_text(values: tuple[int, ...]) -> str:
    """把权重元组渲染成前端/报告里的 ``+9 / −9`` 文案。"""
    return " / ".join(f"{value:+d}".replace("-", "−") for value in values)


# 兼容旧调用：默认档案的 (标签, 权重文案)。
FACTOR_META: dict[str, tuple[str, str]] = {
    key: (label, weight_text(DEFAULT_WEIGHTS[key])) for key, label in FACTOR_LABELS.items()
}


def _relative_excess(
    rows: list[dict[str, Any]],
    benchmark: dict[str, Any] | None,
    days: int = 20,
) -> tuple[float, str] | None:
    """个股相对基准的 N 日超额收益；重合交易日不足时返回 None。"""
    if not benchmark:
        return None
    close_by_date = benchmark.get("close_by_date")
    if close_by_date is None:
        close_by_date = {
            row["date"]: row["close"]
            for row in benchmark.get("rows", [])
            if row.get("close")
        }
    pairs = [
        (row["date"], row["close"], close_by_date[row["date"]])
        for row in rows
        if row.get("close") and row["date"] in close_by_date
    ]
    if len(pairs) <= days:
        return None
    start = pairs[-days - 1]
    stock_return = (pairs[-1][1] / start[1] - 1) * 100
    bench_return = (pairs[-1][2] / start[2] - 1) * 100
    name = benchmark.get("name") or benchmark.get("symbol") or "基准"
    return stock_return - bench_return, name


def score_factors(
    history: dict[str, Any],
    weights: dict[str, tuple[int, ...]] | None = None,
    benchmark: dict[str, Any] | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    """把评分拆成单因子贡献，返回 ``(总分, 因子列表)``。

    总分 = 50 + Σdelta，与 ``analyze()`` 输出的 score 完全一致；
    ``reason`` 非空的因子会进入前端“判断依据”，空值只参与打分。
    ``weights`` 可切换权重档案，walkforward 用它比较不同版本的排序能力。
    """
    profile = weights or DEFAULT_WEIGHTS
    rows = history["rows"]
    closes = [row["close"] for row in rows]
    volumes = [row["volume"] for row in rows]
    ma20 = sma(closes, 20)
    ma60 = sma(closes, 60)
    ma120 = sma(closes, 120)
    rsi14 = rsi(closes)
    dif = [fast - slow for fast, slow in zip(ema(closes, 12), ema(closes, 26))]
    dea = ema(dif, 9)
    histogram = [(d - e) * 2 for d, e in zip(dif, dea)]

    close = closes[-1]
    current_rsi = rsi14[-1] or 50.0
    ret20 = pct_return(closes, 20) or 0.0
    volatility = annualized_volatility(closes)
    drawdown = max_drawdown(closes)
    volume_ma20 = sum(volumes[-20:]) / min(20, len(volumes))
    volume_ratio = volumes[-1] / volume_ma20 if volume_ma20 else 1.0

    factors: list[dict[str, Any]] = []

    def record(key: str, delta: int, tone: str, reason: str | None = None) -> None:
        factors.append(
            {
                "key": key,
                "label": FACTOR_LABELS[key],
                "weight": weight_text(profile[key]),
                "delta": delta,
                "tone": tone,
                "reason": reason,
            }
        )

    if ma20[-1] and close > ma20[-1]:
        record("trend", profile["trend"][0], "positive", "收盘价位于 20 日均线上方，短期趋势占优")
    else:
        record("trend", profile["trend"][1], "negative", "收盘价位于 20 日均线下方，短期趋势偏弱")

    if ma20[-1] and ma60[-1] and ma20[-1] > ma60[-1]:
        record("align", profile["align"][0], "positive", "20 日均线高于 60 日均线，中期结构偏多")
    else:
        record("align", profile["align"][1], "negative", "20 日均线未站上 60 日均线，中期结构仍需修复")

    if ma120[-1] is None:
        record("long", 0, "neutral")
    elif close > ma120[-1]:
        record("long", profile["long"][0], "positive")
    else:
        record("long", profile["long"][1], "negative")

    if "rs" in profile:
        snapshot = _relative_excess(rows, benchmark)
        if snapshot is None:
            record("rs", 0, "neutral")
        else:
            excess, benchmark_name = snapshot
            if excess >= 3:
                record("rs", profile["rs"][0], "positive", f"近 20 日跑赢{benchmark_name} {excess:+.1f}%")
            elif excess <= -3:
                record("rs", profile["rs"][1], "negative", f"近 20 日跑输{benchmark_name} {abs(excess):.1f}%")
            else:
                record("rs", 0, "neutral")

    if histogram[-1] > 0 and dif[-1] > dif[-2]:
        record("macd", profile["macd"][0], "positive", "MACD 动能柱为正且继续改善")
    elif histogram[-1] < 0:
        record("macd", profile["macd"][1], "negative", "MACD 动能柱为负，反弹持续性需观察")
    else:
        record("macd", 0, "neutral")

    if 48 <= current_rsi <= 68:
        record("rsi", profile["rsi"][0], "positive", f"RSI 为 {current_rsi:.1f}，动能健康且未明显过热")
    elif current_rsi > 75:
        record("rsi", profile["rsi"][1], "warning", f"RSI 为 {current_rsi:.1f}，短线存在过热和回撤风险")
    elif current_rsi < 35:
        record("rsi", profile["rsi"][2], "warning", f"RSI 为 {current_rsi:.1f}，处于弱势/超卖区间")
    else:
        record("rsi", 0, "neutral")

    if 0 < ret20 < 15:
        record("ret20", profile["ret20"][0], "positive")
    elif ret20 <= -10:
        record("ret20", profile["ret20"][1], "negative")
    elif ret20 > 25:
        record("ret20", profile["ret20"][2], "warning", "近 20 日涨幅较大，不宜在加速段追高")
    else:
        record("ret20", 0, "neutral")

    if volume_ratio >= 1.3 and rows[-1]["pct_change"] > 0:
        record("volume", profile["volume"][0], "positive", f"上涨伴随放量，量比约 {volume_ratio:.2f}")
    elif volume_ratio >= 1.5 and rows[-1]["pct_change"] < 0:
        record("volume", profile["volume"][1], "negative", "下跌伴随明显放量，抛压风险上升")
    else:
        record("volume", 0, "neutral")

    if volatility > 55:
        record("volatility", profile["volatility"][1], "warning", f"年化波动约 {volatility:.1f}%，风险较高")
    elif volatility < 28:
        record("volatility", profile["volatility"][0], "positive")
    else:
        record("volatility", 0, "neutral")

    if drawdown < -25:
        record("drawdown", profile["drawdown"][0], "warning", f"近 120 日最大回撤约 {drawdown:.1f}%")
    else:
        record("drawdown", 0, "neutral")

    total = BASE_SCORE + sum(factor["delta"] for factor in factors)
    return max(0, min(100, round(total))), factors


def analyze(
    history: dict[str, Any],
    quote: dict[str, Any] | None = None,
    weights: dict[str, tuple[int, ...]] | None = None,
    benchmark: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = history["rows"]
    asset_type = history.get("asset_type", "stock")
    asset_label = {"stock": "A 股", "etf": "ETF", "fund": "场内基金", "index": "指数"}.get(asset_type, "证券")
    is_fund = asset_type in ("etf", "fund")
    price_digits = 3 if is_fund else 2
    closes = [row["close"] for row in rows]
    volumes = [row["volume"] for row in rows]
    dates = [row["date"] for row in rows]
    ma5 = sma(closes, 5)
    ma20 = sma(closes, 20)
    ma60 = sma(closes, 60)
    ma120 = sma(closes, 120)
    rsi14 = rsi(closes)
    macd_fast = ema(closes, 12)
    macd_slow = ema(closes, 26)
    dif = [fast - slow for fast, slow in zip(macd_fast, macd_slow)]
    dea = ema(dif, 9)
    histogram = [(d - e) * 2 for d, e in zip(dif, dea)]

    close = closes[-1]
    current_rsi = rsi14[-1] or 50.0
    ret20 = pct_return(closes, 20) or 0.0
    ret60 = pct_return(closes, 60) or 0.0
    volatility = annualized_volatility(closes)
    drawdown = max_drawdown(closes)
    volume_ma20 = sum(volumes[-20:]) / min(20, len(volumes))
    volume_ratio = volumes[-1] / volume_ma20 if volume_ma20 else 1.0
    volume_ma5_series = sma(volumes, 5)
    volume_ma10_series = sma(volumes, 10)

    score, factors = score_factors(history, weights, benchmark)
    reasons = [
        {"type": factor["tone"], "text": factor["reason"]}
        for factor in factors
        if factor["reason"]
    ]
    if score >= 75:
        view, tone = "偏积极", "positive"
        summary = "趋势与动能形成共振，但更适合等待回踩确认后分批关注。"
        position = "研究仓可从计划资金的 5%–10% 试探，避免一次性重仓。"
    elif score >= 60:
        view, tone = "谨慎偏多", "positive"
        summary = "多项指标正在改善，仍需价格与成交量继续确认。"
        position = "以观察或小仓跟踪为主，突破关键位后再评估是否增加暴露。"
    elif score >= 45:
        view, tone = "中性观察", "neutral"
        summary = "多空信号分化，当前胜率优势不清晰。"
        position = "不追涨，等待趋势、动量和量能出现一致方向。"
    elif score >= 30:
        view, tone = "谨慎", "negative"
        summary = "弱势信号较多，应优先控制回撤而非博弈反弹。"
        position = "已有仓位可复核止损纪律；无仓位宜等待结构修复。"
    else:
        view, tone = "高风险", "negative"
        summary = "趋势和风险指标均不理想，暂不具备较好的风险收益特征。"
        position = "避免新增暴露，先等待价格重回关键均线并缩小波动。"

    if is_fund:
        if score >= 75:
            position = "可作为组合配置候选分批观察，并结合底层指数估值与行业集中度控制仓位。"
        elif score >= 60:
            position = "以小仓跟踪为主，确认突破后再评估；主题型 ETF 应设置更低的仓位上限。"
        elif score >= 45:
            position = "暂不追涨，等待趋势与量能共振，并与同类 ETF 比较费率、规模和流动性。"
        else:
            position = "弱势阶段优先控制组合回撤，等待净值趋势重新站上关键均线。"

    support_candidates = [value for value in (ma20[-1], ma60[-1], min(closes[-20:])) if value]
    resistance_candidates = [max(closes[-20:]), max(row["high"] for row in rows[-60:])]
    support = min(support_candidates, key=lambda x: abs(close - x))
    resistance_above = [x for x in resistance_candidates if x >= close]
    resistance = min(resistance_above) if resistance_above else max(resistance_candidates)
    stop_reference = min(support * 0.97, close * 0.92)

    chart_start = max(0, len(rows) - 180)
    chart = []
    for i in range(chart_start, len(rows)):
        chart.append(
            {
                "date": dates[i],
                "open": rows[i]["open"],
                "high": rows[i]["high"],
                "low": rows[i]["low"],
                "close": closes[i],
                "pct_change": rows[i]["pct_change"],
                "ma20": _round(ma20[i], price_digits),
                "ma60": _round(ma60[i], price_digits),
                "dif": _round(dif[i], 4),
                "dea": _round(dea[i], 4),
                "macd": _round(histogram[i], 4),
                "rsi14": _round(rsi14[i], 1),
                "volume": volumes[i],
                "volume_ma5": _round(volume_ma5_series[i], 0),
                "volume_ma10": _round(volume_ma10_series[i], 0),
            }
        )

    live_price = quote.get("price") if quote else None
    display_price = live_price or close
    return {
        "symbol": history["symbol"],
        "name": (quote or {}).get("name") or history["name"],
        "asset_type": asset_type,
        "asset_label": asset_label,
        "data_source": history.get("data_source", "primary"),
        "price_precision": price_digits,
        "as_of": dates[-1],
        "price": display_price,
        "day_change": (quote or {}).get("pct_change") if quote else rows[-1]["pct_change"],
        "score": score,
        "view": view,
        "tone": tone,
        "summary": summary,
        "position_guidance": position,
        "levels": {
            "support": _round(support, price_digits),
            "resistance": _round(resistance, price_digits),
            "risk_reference": _round(stop_reference, price_digits),
        },
        "metrics": {
            "return_20d": _round(ret20),
            "return_60d": _round(ret60),
            "rsi14": _round(current_rsi, 1),
            "volatility": _round(volatility, 1),
            "max_drawdown": _round(drawdown, 1),
            "volume_ratio": _round(volume_ratio, 2),
            "ma20": _round(ma20[-1], price_digits),
            "ma60": _round(ma60[-1], price_digits),
            "pe_ttm": None if is_fund else (quote or {}).get("pe_ttm"),
            "pb": None if is_fund else (quote or {}).get("pb"),
        },
        "reasons": reasons[:7],
        "factors": factors,
        "chart": chart,
        "methodology": (
            "ETF 评分基于日线趋势、MACD、RSI、量价、波动率与最大回撤；不等同于底层指数估值。"
            if is_fund
            else "基于日线趋势、MACD、RSI、量价、波动率与最大回撤的规则评分，不使用未来数据。"
        ),
    }


def relative_strength(
    stock: dict[str, Any],
    benchmark: dict[str, Any],
    window: int = 120,
) -> dict[str, Any]:
    """Normalize a stock and a benchmark to 100 and derive an RS ratio curve.

    RS = 个股归一化净值 / 基准归一化净值 × 100。基准之上说明跑赢，反之跑输；
    当超额收益很小而同区间两者都上涨时，涨幅更可能来自普涨而非个股强势。
    """
    bench_close = {row["date"]: row["close"] for row in benchmark["rows"] if row["close"]}
    pairs = [
        (row["date"], row["close"], bench_close[row["date"]])
        for row in stock["rows"]
        if row["date"] in bench_close and row["close"]
    ]
    if len(pairs) < 25:
        raise ValueError("个股与基准的重合交易日不足，无法计算相对强弱")

    span = max(20, min(window, len(pairs) - 1))
    pairs = pairs[-(span + 1) :]
    stock_base, bench_base = pairs[0][1], pairs[0][2]

    series: list[dict[str, Any]] = []
    rs_values: list[float] = []
    for date, stock_close, bench_close_value in pairs:
        stock_norm = stock_close / stock_base * 100
        bench_norm = bench_close_value / bench_base * 100
        rs_value = stock_norm / bench_norm * 100
        rs_values.append(rs_value)
        series.append(
            {
                "date": date,
                "stock": round(stock_norm, 2),
                "benchmark": round(bench_norm, 2),
                "rs": round(rs_value, 2),
            }
        )

    stock_return = (pairs[-1][1] / stock_base - 1) * 100
    bench_return = (pairs[-1][2] / bench_base - 1) * 100
    excess = stock_return - bench_return
    rs_last = rs_values[-1]
    rs_ma20 = sma(rs_values, 20)[-1]
    rs_change_20d = rs_last - rs_values[-min(21, len(rs_values))]

    if excess >= 5 and (rs_ma20 is None or rs_last >= rs_ma20):
        verdict, tone = "真强", "positive"
        note = (
            "个股涨幅明显跑赢基准，且 RS 曲线仍在上行，超额收益来自自身强度而非普涨。"
            if stock_return >= 0
            else "两者同处调整，但个股跌幅明显小于基准，相对强度来自抗跌能力而非跟涨。"
        )
    elif excess <= -5:
        verdict, tone = "跑输基准", "negative"
        note = (
            "个股涨幅落后基准，即使绝对收益为正，也应优先审视其相对弱势的原因。"
            if stock_return >= 0
            else "个股跌幅大于基准，调整阶段的防御性不足，应优先控制风险。"
        )
    elif abs(excess) <= 2:
        verdict, tone = "同步", "neutral"
        note = (
            "个股与基准涨幅几乎一致，当前上涨更可能由普涨带动，持续性需等 RS 曲线走强再确认。"
            if stock_return >= 0
            else "个股与基准同步回落，暂未看到独立的抗跌能力。"
        )
    elif excess > 0:
        verdict, tone = "略强", "positive"
        note = (
            "个股小幅跑赢基准，强度尚不显著，可继续跟踪 RS 能否站上 20 日均值。"
            if stock_return >= 0
            else "个股跌幅略小于基准，暂视为弱市中的相对抗跌，尚不构成独立强势。"
        )
    else:
        verdict, tone = "略弱", "negative"
        note = (
            "个股小幅跑输基准，属于普涨中的跟随者，不宜按强势股对待。"
            if stock_return >= 0
            else "个股跌幅略大于基准，弱市中缺少相对优势。"
        )

    if rs_ma20 is None:
        rs_trend = "flat"
    elif rs_last > rs_ma20:
        rs_trend = "up"
    elif rs_last < rs_ma20:
        rs_trend = "down"
    else:
        rs_trend = "flat"

    return {
        "symbol": stock["symbol"],
        "name": stock["name"],
        "asset_type": stock.get("asset_type", "stock"),
        "benchmark_symbol": benchmark["symbol"],
        "benchmark_name": benchmark["name"],
        "window": len(pairs) - 1,
        "start_date": pairs[0][0],
        "as_of": pairs[-1][0],
        "stock_return": round(stock_return, 2),
        "benchmark_return": round(bench_return, 2),
        "excess_return": round(excess, 2),
        "rs": round(rs_last, 2),
        "rs_ma20": _round(rs_ma20, 2),
        "rs_change_20d": round(rs_change_20d, 2),
        "rs_trend": rs_trend,
        "rs_high": round(max(rs_values), 2),
        "rs_low": round(min(rs_values), 2),
        "verdict": verdict,
        "tone": tone,
        "note": note,
        "series": series,
    }


def market_regime(index_analyses: list[dict[str, Any]]) -> dict[str, Any]:
    average = round(sum(item["score"] for item in index_analyses) / len(index_analyses))
    positive = sum(item["price"] > (item["metrics"]["ma20"] or item["price"]) for item in index_analyses)
    if average >= 65 and positive >= 2:
        regime, tone, guidance = "偏强", "positive", "市场风险偏好较好，可适度提高关注度，但仍应分散配置。"
    elif average < 45 or positive == 0:
        regime, tone, guidance = "偏弱", "negative", "市场环境不利，建议降低仓位上限并提高信号确认要求。"
    else:
        regime, tone, guidance = "震荡", "neutral", "指数信号分化，控制节奏并优先选择相对强势标的。"
    return {"score": average, "regime": regime, "tone": tone, "guidance": guidance, "indexes": index_analyses}
