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


def analyze(history: dict[str, Any], quote: dict[str, Any] | None = None) -> dict[str, Any]:
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

    score = 50
    reasons: list[dict[str, str]] = []

    if ma20[-1] and close > ma20[-1]:
        score += 9
        reasons.append({"type": "positive", "text": "收盘价位于 20 日均线上方，短期趋势占优"})
    else:
        score -= 9
        reasons.append({"type": "negative", "text": "收盘价位于 20 日均线下方，短期趋势偏弱"})

    if ma20[-1] and ma60[-1] and ma20[-1] > ma60[-1]:
        score += 9
        reasons.append({"type": "positive", "text": "20 日均线高于 60 日均线，中期结构偏多"})
    else:
        score -= 7
        reasons.append({"type": "negative", "text": "20 日均线未站上 60 日均线，中期结构仍需修复"})

    if ma120[-1] is not None:
        if close > ma120[-1]:
            score += 6
        else:
            score -= 6

    if histogram[-1] > 0 and dif[-1] > dif[-2]:
        score += 8
        reasons.append({"type": "positive", "text": "MACD 动能柱为正且继续改善"})
    elif histogram[-1] < 0:
        score -= 7
        reasons.append({"type": "negative", "text": "MACD 动能柱为负，反弹持续性需观察"})

    if 48 <= current_rsi <= 68:
        score += 7
        reasons.append({"type": "positive", "text": f"RSI 为 {current_rsi:.1f}，动能健康且未明显过热"})
    elif current_rsi > 75:
        score -= 6
        reasons.append({"type": "warning", "text": f"RSI 为 {current_rsi:.1f}，短线存在过热和回撤风险"})
    elif current_rsi < 35:
        score -= 4
        reasons.append({"type": "warning", "text": f"RSI 为 {current_rsi:.1f}，处于弱势/超卖区间"})

    if 0 < ret20 < 15:
        score += 5
    elif ret20 <= -10:
        score -= 6
    elif ret20 > 25:
        score -= 3
        reasons.append({"type": "warning", "text": "近 20 日涨幅较大，不宜在加速段追高"})

    if volume_ratio >= 1.3 and rows[-1]["pct_change"] > 0:
        score += 4
        reasons.append({"type": "positive", "text": f"上涨伴随放量，量比约 {volume_ratio:.2f}"})
    elif volume_ratio >= 1.5 and rows[-1]["pct_change"] < 0:
        score -= 5
        reasons.append({"type": "negative", "text": "下跌伴随明显放量，抛压风险上升"})

    if volatility > 55:
        score -= 8
        reasons.append({"type": "warning", "text": f"年化波动约 {volatility:.1f}%，风险较高"})
    elif volatility < 28:
        score += 4
    if drawdown < -25:
        score -= 6
        reasons.append({"type": "warning", "text": f"近 120 日最大回撤约 {drawdown:.1f}%"})

    score = max(0, min(100, round(score)))
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
                "close": closes[i],
                "pct_change": rows[i]["pct_change"],
                "ma20": _round(ma20[i], price_digits),
                "ma60": _round(ma60[i], price_digits),
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
        "chart": chart,
        "methodology": (
            "ETF 评分基于日线趋势、MACD、RSI、量价、波动率与最大回撤；不等同于底层指数估值。"
            if is_fund
            else "基于日线趋势、MACD、RSI、量价、波动率与最大回撤的规则评分，不使用未来数据。"
        ),
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
