"""三字段口径：趋势状态（只做描述）、横截面分位（只做选股）、市场 regime（只做择时）。

为什么拆
--------
原来的单一 0–100 分同时承担三件事：描述"现在什么状态"、给选股排序、暗示仓位。
于是所有跨口径推断都长得像结论——"评分 75+ 所以看多"其实是把描述当择时用。
这个模块把三件事拆成三个字段，每个字段带**自己的参照系**和**自己的验证口径**：

=================  ==========  ==========================  ==================
字段               用途        参照系                       通过与否看什么
=================  ==========  ==========================  ==================
descriptor         描述       同类资产的历史分数分布       风险标签的秩相关
selection          选股       当日同池的横截面             样本外 IC 区间下限
regime             择时       基准自身近 3 年的标准化分数   组合回撤是否改善
=================  ==========  ==========================  ==================

三条硬约束
----------
1. **没有综合分**。本模块不提供任何把三个字段加权求和的东西。
2. **分位而不是绝对分**。阈值只有百分位切点（20/40/60/80），分数本身不再参与
   判断；同一分数在个股和 ETF 上的含义由各自校准表决定。
3. **跨口径调用直接报错**。``assert_use("descriptor", "timing")`` 抛 ValueError，
   前端文案也由 ``usage_note()`` 生成，避免再写出"75 分所以看多"。
"""

from __future__ import annotations

import bisect
import json
import math
import statistics
from pathlib import Path
from typing import Any, Sequence


# --------------------------------------------------------------------------- #
# 字段定义：用途是硬编码的，不是注释
# --------------------------------------------------------------------------- #

DESCRIPTOR = "descriptor"
SELECTION = "selection"
REGIME = "regime"

FIELD_SPECS: dict[str, dict[str, Any]] = {
    DESCRIPTOR: {
        "name": "趋势状态",
        "use": "description",
        "question": "这个标的现在处于什么趋势 / 风险状态？",
        "reference": "同类资产（个股 / ETF）的历史分数分布；指数无同类校准，只看相对自身",
        "validation": "对风险标签（未来 20 日波动、回撤、下行波动、VaR）的秩相关",
        "not_for": ("selection", "timing"),
        "note": "只描述状态，不含方向判断，也不能据此决定仓位。",
    },
    SELECTION: {
        "name": "横截面分位",
        "use": "selection",
        "question": "在同一时点、同一池子里，它排第几？",
        "reference": "当日同池横截面",
        "validation": "样本外 IC 区间下限是否大于 0",
        "not_for": ("description", "timing"),
        "note": "只在池内有意义；池子换了排名就换了，不能跨池比较，也不能用来择时。",
    },
    REGIME: {
        "name": "市场状态",
        "use": "timing",
        "question": "现在该给多少总暴露？",
        "reference": "基准自身近 3 年标准化分数",
        "validation": "切换后的组合回撤是否下降、成本是否被覆盖",
        "not_for": ("description", "selection"),
        "note": "只决定总暴露上限，不决定买哪只，也不描述单只标的的好坏。",
    },
}

_PURPOSE_LABEL = {"description": "描述", "selection": "选股", "timing": "择时"}


def assert_use(field: str, purpose: str) -> None:
    """跨口径调用直接失败，而不是靠注释提醒。"""
    if field not in FIELD_SPECS:
        raise ValueError(f"未知字段：{field}")
    spec = FIELD_SPECS[field]
    if purpose != spec["use"]:
        raise ValueError(
            f"{spec['name']}分的用途是「{_PURPOSE_LABEL[spec['use']]}」，"
            f"不能用于「{_PURPOSE_LABEL.get(purpose, purpose)}」；请改用对应字段。"
        )


def usage_note(field: str) -> str:
    spec = FIELD_SPECS[field]
    return f"{spec['name']}（{_PURPOSE_LABEL[spec['use']]}口径）：{spec['note']}"


# --------------------------------------------------------------------------- #
# 校准表：分数 → 百分位，按资产类别分开
# --------------------------------------------------------------------------- #

# 百分位切点固定为五档；分数切点由校准表给出（不同资产类别不一样）。
BAND_CUTS = (20, 40, 60, 80)
BAND_LABELS = (
    (80, "强势", "strong", "同类别里最健康的一档"),
    (60, "偏强", "firm", "好于同类别多数时间"),
    (40, "中性", "neutral", "同类别中位附近"),
    (20, "偏弱", "soft", "弱于同类别多数时间"),
    (0, "弱势", "weak", "同类别里最弱的一档"),
)

CALIBRATION_PATH = Path(__file__).resolve().parent / "reports" / "fields" / "calibration.json"

# 由 validate_fields.py 在 2016-12 ~ 2026-10 的面板上重新生成；
# 结构：{"stock": {"score_knots": [...], "band_cuts": [...], "n": ..., "source": ...}, "etf": {...}}
# score_knots 是分数分布的 0/5/…/100 百分位（21 个点），用于把分数映射成同类分位。
DEFAULT_CALIBRATION: dict[str, dict[str, Any]] = {
    "stock": {
        "score_knots": [
            8.0, 26.0, 32.0, 36.0, 39.0, 42.0, 45.0, 47.0, 50.0, 52.0, 55.0,
            58.0, 61.0, 64.0, 67.0, 70.0, 73.0, 77.0, 81.0, 86.0, 95.0,
        ],
        "band_cuts": list(BAND_CUTS),
        "n": 0,
        "source": "内置占位（未校准）",
    },
    "etf": {
        "score_knots": [
            10.0, 28.0, 34.0, 38.0, 41.0, 44.0, 47.0, 49.0, 52.0, 54.0, 57.0,
            60.0, 63.0, 66.0, 69.0, 72.0, 75.0, 79.0, 83.0, 87.0, 94.0,
        ],
        "band_cuts": list(BAND_CUTS),
        "n": 0,
        "source": "内置占位（未校准）",
    },
    # 指数单独一类、而且**故意留空**：样本只有沪深300 / 中证500 / 中证1000 三条序列，
    # 构不成"同类分布"。空 knots 就是"未校准"的机器可读标记，describe() 据此关闭同类分位，
    # 不再拿 ETF 的校准表冒充指数（那会得出"指数处于 ETF 同类 12 分位"这种没意义的结论）。
    "index": {
        "score_knots": [],
        "band_cuts": list(BAND_CUTS),
        "n": 0,
        "source": "指数不做同类校准（样本仅 3 条序列）",
    },
}

ASSET_CLASS_LABEL = {"stock": "个股", "etf": "ETF", "fund": "ETF", "index": "指数"}


def asset_class_of(asset_type: str | None) -> str:
    """资产类别 → 校准表键。指数**不并入 ETF**：两者的分数分布不同（指数波动更低、
    趋势更平滑），共用一张表会让"同类分位"名不副实。"""
    if asset_type == "index":
        return "index"
    return "etf" if asset_type in ("etf", "fund") else "stock"


def is_calibrated(asset_class: str, calibration: dict | None = None) -> bool:
    """该类是否真有校准表：knots 少于 2 个视为未校准，不产出同类分位。"""
    table = calibration or DEFAULT_CALIBRATION
    knots = (table.get(asset_class) or {}).get("score_knots") or []
    return len(knots) >= 2


def load_calibration(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """优先读重新校准的结果；缺失或损坏时退回内置表。"""
    target = path or CALIBRATION_PATH
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_CALIBRATION
    table: dict[str, dict[str, Any]] = {}
    for key, fallback in DEFAULT_CALIBRATION.items():
        block = payload.get(key) if isinstance(payload, dict) else None
        knots = (block or {}).get("score_knots") or []
        if len(knots) == 21 and all(isinstance(x, (int, float)) for x in knots):
            table[key] = {
                "score_knots": [float(x) for x in knots],
                "band_cuts": list((block or {}).get("band_cuts") or BAND_CUTS),
                "n": (block or {}).get("n", 0),
                "source": (block or {}).get("source", "reports/fields/calibration.json"),
            }
        else:
            table[key] = fallback
    return table


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("空样本无法取分位")
    if probability <= 0:
        return float(sorted_values[0])
    if probability >= 1:
        return float(sorted_values[-1])
    position = probability * (len(sorted_values) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return float(sorted_values[low] * (1 - weight) + sorted_values[high] * weight)


def score_knots(values: Sequence[float], knots: int = 21) -> list[float]:
    """把样本分布压成 ``knots`` 个分位点（0/5/…/100）。"""
    ordered = sorted(float(value) for value in values)
    return [round(_quantile(ordered, i / (knots - 1)), 2) for i in range(knots)]


def calibrate(samples: dict[str, Sequence[float]], *, source: str = "") -> dict[str, dict[str, Any]]:
    """按资产类别生成校准表；样本不足的类别保留内置表。"""
    table: dict[str, dict[str, Any]] = {}
    for key, fallback in DEFAULT_CALIBRATION.items():
        values = [float(v) for v in (samples.get(key) or []) if v is not None]
        if len(values) < 500:
            if not (fallback.get("score_knots") or []):
                # 本来就不做同类校准的类别（指数）：别把它说成"样本不足沿用内置表"，
                # 那会让人以为"多喂点样本就能有分位"。
                table[key] = dict(fallback)
            else:
                table[key] = dict(fallback, source=f"样本不足（{len(values)}），沿用内置表")
            continue
        table[key] = {
            "score_knots": score_knots(values),
            "band_cuts": list(BAND_CUTS),
            "n": len(values),
            "source": source or "面板校准",
        }
    return table


def percentile_of_score(score: float, asset_class: str, calibration: dict | None = None) -> float | None:
    """分数 → 同类资产的百分位（0–100，越大越好）。线性插值到校准节点。

    没有校准表（``knots`` 为空）时返回 ``None``：宁可说"没有同类分位"，也不该拿别的
    类别的表插值出一个看起来很像真的数字。
    """
    table = calibration or DEFAULT_CALIBRATION
    knots = (table.get(asset_class) or {}).get("score_knots") or []
    if len(knots) < 2:
        return None
    if score <= knots[0]:
        return 0.0
    if score >= knots[-1]:
        return 100.0
    step = 100.0 / (len(knots) - 1)
    for index in range(len(knots) - 1):
        left, right = knots[index], knots[index + 1]
        if left <= score <= right:
            span = right - left
            inner = 0.0 if span <= 0 else (score - left) / span
            return round(step * (index + inner), 1)
    return 50.0


def band_of(percentile: float, band_cuts: Sequence[int] = BAND_CUTS) -> dict[str, Any]:
    """百分位 → 五档标签。切点是**分位**，阈值一律取自 ``band_cuts``（降序），
    档位名称按位置取 ``BAND_LABELS``。

    旧实现把 ``band_cuts`` 排完序就丢掉、继续用写死的 80/60/40/20，于是自定义切点
    永远不生效——看着有参数，其实是装饰品。
    """
    cuts = sorted((int(c) for c in band_cuts), reverse=True)
    for index, (_, label, tone, meaning) in enumerate(BAND_LABELS):
        threshold = cuts[index] if index < len(cuts) else 0
        if percentile >= threshold:
            return {"key": tone, "label": label, "meaning": meaning, "cut": threshold}
    return {"key": "weak", "label": "弱势", "meaning": BAND_LABELS[-1][3], "cut": 0}


# --------------------------------------------------------------------------- #
# 描述分：趋势 + 风险，时间序列口径
# --------------------------------------------------------------------------- #

# 风险主导（0.75）+ 趋势位置（0.25）。权重只在这里维护；没有"综合分"出口。
#
# v1 曾是「趋势 0.60 / 风险 0.40」，并把 12–1 动量算进去。归因显示动量分量与
# 未来 20 日波动是**正相关**（ρ≈+0.29，趋势越强未来波动越大），把风险分量的
# 负相关抵消掉，合成后 ρ≈−0.02——一个声称描述风险的分数，实际不含风险信息。
# v2 因此移除动量（它属于收益信号，归选股/择时），并把风险分量提到 0.75。
# 这是一次看过后验结果的设计修正，报告里按"第二次尝试"如实标注。
DESCRIPTOR_WEIGHTS: dict[str, float] = {
    "vol": 0.40,        # 20 日波动率在自身一年分布中的位置（低=高分）
    "downside": 0.20,   # 20 日下行波动在自身一年分布中的位置（低=高分）
    "drawdown": 0.15,   # 距 252 日高点的回撤在自身一年分布中的位置（浅=高分）
    "trend": 0.125,     # 收盘 vs MA20
    "align": 0.075,     # MA20 vs MA60
    "long": 0.05,       # 收盘 vs MA120
}

DESCRIPTOR_LABELS: dict[str, str] = {
    "trend": "价格 vs MA20",
    "align": "MA20 vs MA60",
    "long": "价格 vs MA120",
    "vol": "波动率位置",
    "downside": "下行波动位置",
    "drawdown": "距一年高点",
}

MIN_BARS = 260          # 描述分最少需要的交易日（波动/回撤分位要 80 个有效值）
VOL_WINDOW = 20         # 波动率窗口
DD_WINDOW = 252         # 回撤参考窗口
SELF_WINDOW = 60        # "相对自身"分位的回看窗口


def _logistic(value: float, scale: float = 1.0) -> float:
    """把 z 值压到 0–100；scale 越小越敏感。"""
    x = max(-30.0, min(30.0, value / scale))
    return 100.0 / (1.0 + math.exp(-x))


def _sma(values: Sequence[float], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= period:
            running -= values[index - period]
        if index >= period - 1:
            result[index] = running / period
    return result


def _rolling_max(values: Sequence[float], window: int) -> list[float | None]:
    """因果滚动最大值（含当日），单调队列 O(n)。"""
    result: list[float | None] = [None] * len(values)
    queue: list[int] = []
    for index, value in enumerate(values):
        while queue and values[queue[-1]] <= value:
            queue.pop()
        queue.append(index)
        if queue[0] <= index - window:
            queue.pop(0)
        result[index] = values[queue[0]]
    return result


def _daily_vol(closes: Sequence[float], window: int) -> list[float | None]:
    returns = [None] + [
        (closes[i] / closes[i - 1] - 1.0) if closes[i - 1] else None
        for i in range(1, len(closes))
    ]
    result: list[float | None] = [None] * len(closes)
    for index in range(window, len(closes)):
        sample = [r for r in returns[index - window + 1: index + 1] if r is not None]
        if len(sample) >= max(6, window // 2):
            result[index] = statistics.stdev(sample) if len(sample) > 1 else 0.0
    return result


def _downside_series(closes: Sequence[float], window: int) -> list[float | None]:
    """滚动下行波动（只取负收益的均方根），与风险标签的口径一致。"""
    result: list[float | None] = [None] * len(closes)
    for index in range(window, len(closes)):
        sample = [
            closes[i] / closes[i - 1] - 1.0
            for i in range(index - window + 1, index + 1)
            if closes[i - 1] > 0
        ]
        negative = [value for value in sample if value < 0]
        if len(sample) >= window // 2:
            result[index] = math.sqrt(statistics.fmean([v * v for v in negative])) if negative else 0.0
    return result


def _trailing_percentile(
    series: Sequence[float | None],
    window: int,
    *,
    min_count: int = 60,
    strict: bool = False,
) -> list[float | None]:
    """每个点在自己过去 ``window`` 个有效值中的百分位（0–100，因果）。

    ``strict=True`` 时只统计"严格小于"当前值的比例，适合回撤这类"越接近 0 越好"、
    且大量取值为 0 的序列——否则新高日会把分位算成 0。
    """
    result: list[float | None] = [None] * len(series)
    buffer: list[float] = []
    pending: list[float] = []
    for index, value in enumerate(series):
        if value is not None:
            bisect.insort(buffer, value)
            pending.append(value)
            if len(pending) > window:
                old = pending.pop(0)
                del buffer[bisect.bisect_left(buffer, old)]
            if len(buffer) >= min_count:
                below = (
                    bisect.bisect_left(buffer, value)
                    if strict
                    else bisect.bisect_right(buffer, value)
                )
                result[index] = 100.0 * below / len(buffer)
    return result


def sanitize(history: dict[str, Any]) -> dict[str, Any]:
    """去掉非正价格并截断到最后一个非正价之后。

    后复权（baostock ``adjustflag=1``）对高分红股在早年可能给出负价格，这是
    加法复权的已知产物，不是真实行情。与其在每处下判断，不如在入口清洗一次。
    """
    rows = history.get("rows") or []
    last_bad = -1
    for index, row in enumerate(rows):
        if float(row["close"]) <= 0 or float(row["open"]) <= 0:
            last_bad = index
    if last_bad < 0:
        return history
    cleaned = rows[last_bad + 1:]
    return {**history, "rows": cleaned}


def descriptor_series(history: dict[str, Any], *, asset_class: str | None = None) -> list[dict[str, Any] | None]:
    """逐日描述分（因果）。返回与 ``history["rows"]`` 等长的列表。

    每一项含 ``score`` 与六个分量的 0–100 子分。分数只是同类百分位的原料，
    不直接对外表达观点；对外一律给百分位 + 档位。
    """
    rows = history["rows"]
    closes = [float(row["close"]) for row in rows]
    n = len(closes)
    if n < MIN_BARS:
        return [None] * n

    ma20 = _sma(closes, 20)
    ma60 = _sma(closes, 60)
    ma120 = _sma(closes, 120)
    vol = _daily_vol(closes, VOL_WINDOW)
    vol_rank = _trailing_percentile(vol, DD_WINDOW, min_count=80)
    rolling_high = _rolling_max(closes, DD_WINDOW)
    depth = [
        (close / high - 1.0) if high else None
        for close, high in zip(closes, rolling_high)
    ]
    depth_rank = _trailing_percentile(depth, DD_WINDOW, min_count=40, strict=True)
    downside = _downside_series(closes, VOL_WINDOW)
    downside_rank = _trailing_percentile(downside, DD_WINDOW, min_count=80)

    result: list[dict[str, Any] | None] = [None] * n
    for index in range(n):
        if index < 130:
            continue
        close = closes[index]
        sigma = vol[index] or 0.0
        if close <= 0 or sigma <= 0 or not ma20[index] or not ma60[index]:
            continue

        components: dict[str, float] = {}
        components["trend"] = _logistic(math.log(close / ma20[index]) / (sigma * math.sqrt(20)), 1.45)
        components["align"] = _logistic(math.log(ma20[index] / ma60[index]) / (sigma * math.sqrt(40)), 1.45)
        if ma120[index]:
            components["long"] = _logistic(math.log(close / ma120[index]) / (sigma * math.sqrt(120)), 1.45)
        if vol_rank[index] is not None:
            components["vol"] = 100.0 - vol_rank[index]
        if downside_rank[index] is not None:
            components["downside"] = 100.0 - downside_rank[index]
        if depth_rank[index] is not None:
            components["drawdown"] = depth_rank[index]

        weights = {key: DESCRIPTOR_WEIGHTS[key] for key in components}
        total_weight = sum(weights.values())
        if total_weight < 0.5:
            continue
        score = sum(components[key] * weights[key] for key in components) / total_weight
        result[index] = {
            "score": round(score, 2),
            "coverage": round(total_weight / sum(DESCRIPTOR_WEIGHTS.values()), 3),
            "components": {key: round(value, 1) for key, value in components.items()},
        }
    return result


def describe(
    history: dict[str, Any],
    *,
    asset_type: str | None = None,
    pool_percentiles: dict[str, float] | None = None,
    calibration: dict | None = None,
    self_window: int = SELF_WINDOW,
) -> dict[str, Any]:
    """描述字段：同类分位（主）+ 相对自身分位（辅）+ 档位 + 分量拆解。

    主分位的参照系是"同类资产的历史分布"，所以个股与 ETF 各用各的校准表；
    传入 ``pool_percentiles``（当日同池横截面）时并列给出池内分位，两者不混用。
    """
    rows = history["rows"]
    series = descriptor_series(history, asset_class=asset_type)
    index = len(rows) - 1
    while index >= 0 and series[index] is None:
        index -= 1
    if index < 0:
        return {
            "field": DESCRIPTOR,
            "use": "description",
            "available": False,
            "reason": f"历史行情不足 {MIN_BARS} 个交易日，无法给出状态描述",
            "usage": usage_note(DESCRIPTOR),
        }

    point = series[index]
    asset_class = asset_class_of(asset_type or history.get("asset_type"))
    table = calibration or DEFAULT_CALIBRATION

    score = point["score"]
    entry = table.get(asset_class) or {}
    class_percentile = percentile_of_score(score, asset_class, table)

    window_scores = [item["score"] for item in series[max(0, index - self_window + 1): index + 1] if item]
    self_percentile = _rank_of(score, window_scores)

    if class_percentile is None:
        # 未校准（指数）：档位改用"相对自身"分位，并在返回里标明依据，
        # 免得读者把"自身分位档"误当成"同类分位档"。
        band = band_of(self_percentile, entry.get("band_cuts", BAND_CUTS))
        band["basis"] = "self"
    else:
        band = band_of(class_percentile, entry.get("band_cuts", BAND_CUTS))
        band["basis"] = "asset_class"

    pool_percentile = None
    if pool_percentiles:
        pool_percentile = pool_percentiles.get(history["symbol"])

    components = [
        {
            "key": key,
            "label": DESCRIPTOR_LABELS[key],
            "score": point["components"].get(key),
            "weight": DESCRIPTOR_WEIGHTS[key],
        }
        for key in DESCRIPTOR_WEIGHTS
        if key in point["components"]
    ]
    strongest = max(components, key=lambda item: item["score"] or 0)
    weakest = min(components, key=lambda item: item["score"] or 0)

    return {
        "field": DESCRIPTOR,
        "use": "description",
        "available": True,
        "as_of": rows[index]["date"],
        "asset_class": asset_class,
        "asset_class_label": ASSET_CLASS_LABEL.get(asset_class, asset_class),
        "score": score,
        "coverage": point["coverage"],
        "percentile": {
            "value": class_percentile,
            "mode": "asset_class",
            "available": class_percentile is not None,
            "label": (f"同类（{ASSET_CLASS_LABEL.get(asset_class, asset_class)}）历史分位"
                      if class_percentile is not None
                      else f"{ASSET_CLASS_LABEL.get(asset_class, asset_class)}不做同类分位"),
            "reason": ("" if class_percentile is not None else
                       "指数只有三条序列，构不成同类分布；请看「相对自身」分位"),
            "n": entry.get("n", 0),
            "calibration": entry.get("source", ""),
        },
        "self_percentile": {
            "value": self_percentile,
            "mode": "self",
            "window": len(window_scores),
            "label": f"近 {self_window} 日分位",
        },
        "pool_percentile": None if pool_percentile is None else {
            "value": pool_percentile,
            "mode": "pool",
            "label": "当日同池分位",
            "n": len(pool_percentiles or {}),
        },
        "band": band,
        "components": components,
        "headline": (
            (f"趋势状态处于同类{band['label']}档（{class_percentile:.0f} 分位），"
             f"近 {len(window_scores)} 个交易日里位于 {self_percentile:.0f} 分位；"
             if class_percentile is not None else
             f"趋势状态相对自身近 {len(window_scores)} 个交易日为{band['label']}档"
             f"（{self_percentile:.0f} 分位，指数无同类校准）；")
            + f"相对最强的是{strongest['label']}，最弱的是{weakest['label']}。"
        ),
        "usage": usage_note(DESCRIPTOR),
    }


def _rank_of(value: float, sample: Sequence[float]) -> float:
    if not sample:
        return 50.0
    below = sum(1 for item in sample if item <= value)
    return round(100.0 * below / len(sample), 1)


# --------------------------------------------------------------------------- #
# 选股分：当日同池分位
# --------------------------------------------------------------------------- #


def pool_percentiles(scores: dict[str, float]) -> dict[str, float]:
    """把当日池内分数转成百分位。池子是横截面的全部，缺一只就少一只。"""
    values = [(symbol, score) for symbol, score in scores.items() if score is not None]
    if len(values) < 2:
        return {}
    ordered = sorted(values, key=lambda item: item[1])
    result: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index
        while end + 1 < len(ordered) and ordered[end + 1][1] == ordered[index][1]:
            end += 1
        average_rank = (index + end) / 2
        percentile = 100.0 * (average_rank + 0.5) / len(ordered)
        for position in range(index, end + 1):
            result[ordered[position][0]] = round(percentile, 1)
        index = end + 1
    return result


def selection_field(
    symbol: str,
    percentiles: dict[str, float] | None,
    *,
    pool_size: int = 0,
    as_of: str = "",
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """选股字段：只回答"池内排第几"，并把证据状态一并带上。"""
    value = (percentiles or {}).get(symbol)
    payload: dict[str, Any] = {
        "field": SELECTION,
        "use": "selection",
        "available": value is not None,
        "as_of": as_of,
        "percentile": value,
        "pool_size": pool_size or len(percentiles or {}),
        "label": "当日同池分位",
        "usage": usage_note(SELECTION),
    }
    if value is None:
        payload["reason"] = "池内标的不足 2 只，横截面分位无意义"
    payload["evidence"] = evidence or {
        "status": "untested",
        "label": "未验证",
        "detail": "该字段尚未通过样本外检验，不应据此断言超额收益",
    }
    return payload


# --------------------------------------------------------------------------- #
# 择时分：市场 regime
# --------------------------------------------------------------------------- #

REGIME_STATES: dict[str, dict[str, Any]] = {
    "risk_on": {"label": "风险偏好上行", "tone": "positive", "exposure": 1.0,
                "guidance": "总暴露上限 100%，仍需分散。"},
    "neutral": {"label": "中性", "tone": "neutral", "exposure": 0.6,
                "guidance": "总暴露上限 60%，等信号同向再提高。"},
    "risk_off": {"label": "防守", "tone": "negative", "exposure": 0.3,
                 "guidance": "总暴露上限 30%，优先控制回撤。"},
}

REGIME_Z_CUT = 0.5      # 以自身近三年标准差为单位，不是绝对分数
REGIME_Z_WINDOW = 756


def _zscore_rank(
    series: Sequence[float | None],
    window: int = REGIME_Z_WINDOW,
    *,
    tail: int | None = None,
) -> list[float | None]:
    """每个点相对自己过去 ``window`` 个有效值的 z 值（因果）。

    ``tail`` 只保留最后若干个结果，供前端按需计算（验证脚本用全量）。
    """
    result: list[float | None] = [None] * len(series)
    buffer: list[float] = []
    pending: list[float] = []
    record_from = 0 if not tail else max(0, len(series) - tail)
    for index, value in enumerate(series):
        if value is None:
            continue
        bisect.insort(buffer, value)
        pending.append(value)
        if len(pending) > window:
            old = pending.pop(0)
            del buffer[bisect.bisect_left(buffer, old)]
        if len(buffer) < 120 or index < record_from:
            continue
        mean = statistics.fmean(buffer)
        deviation = statistics.pstdev(buffer)
        result[index] = 0.0 if deviation <= 1e-12 else (value - mean) / deviation
    return result


def _raw_regime_score(closes: Sequence[float], breadth: Sequence[float | None] | None) -> list[float | None]:
    """regime 的原始分（未标准化）：长期趋势、趋势斜率、波动率位置、市场宽度。"""
    n = len(closes)
    ma60 = _sma(closes, 60)
    ma200 = _sma(closes, 200)
    vol = _daily_vol(closes, VOL_WINDOW)
    vol_rank = _trailing_percentile(vol, REGIME_Z_WINDOW, min_count=250)
    result: list[float | None] = [None] * n
    for index in range(n):
        sigma = vol[index] or 0.0
        if index < 210 or sigma <= 0 or not ma60[index] or not ma200[index]:
            continue
        trend = math.log(closes[index] / ma200[index]) / (sigma * math.sqrt(200))
        slope = math.log(ma60[index] / ma60[index - 20]) / (sigma * math.sqrt(20))
        parts = {
            "trend": 0.35 * max(-3.0, min(3.0, trend)),
            "slope": 0.20 * max(-3.0, min(3.0, slope)),
            "vol": 0.25 * (1.5 - 3.0 * ((vol_rank[index] or 50.0) / 100.0)),
        }
        if breadth is not None and index < len(breadth) and breadth[index] is not None:
            parts["breadth"] = 0.20 * max(-1.0, min(1.0, (breadth[index] - 0.5) * 2))
        result[index] = sum(parts.values()) / sum(
            [0.35, 0.20, 0.25] + ([0.20] if "breadth" in parts else [])
        )
    return result


def regime_series(
    benchmark: dict[str, Any],
    *,
    breadth_by_date: dict[str, float] | None = None,
    tail: int | None = None,
) -> list[dict[str, Any] | None]:
    """逐日 regime：z 值 → 三档状态 + 目标暴露。切点以自身标准差为单位。"""
    rows = benchmark["rows"]
    closes = [float(row["close"]) for row in rows]
    dates = [row["date"] for row in rows]
    breadth = None
    if breadth_by_date:
        breadth = [breadth_by_date.get(date) for date in dates]
    raw = _raw_regime_score(closes, breadth)
    zscores = _zscore_rank(raw, tail=tail)
    result: list[dict[str, Any] | None] = [None] * len(rows)
    for index, z in enumerate(zscores):
        if z is None:
            continue
        if z >= REGIME_Z_CUT:
            state = "risk_on"
        elif z <= -REGIME_Z_CUT:
            state = "risk_off"
        else:
            state = "neutral"
        result[index] = {
            "date": dates[index],
            "z": round(z, 3),
            "state": state,
            "exposure": REGIME_STATES[state]["exposure"],
        }
    return result


def regime(
    benchmark: dict[str, Any],
    *,
    breadth_by_date: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """择时字段：只输出状态与总暴露上限，不给方向、不给个股意见。"""
    series = regime_series(benchmark, breadth_by_date=breadth_by_date, tail=90)
    index = len(series) - 1
    while index >= 0 and series[index] is None:
        index -= 1
    if index < 0:
        return {
            "field": REGIME,
            "use": "timing",
            "available": False,
            "reason": "基准历史不足 500 个交易日（regime 需要一年波动分位 + 三年标准化），无法判断市场状态",
            "usage": usage_note(REGIME),
        }
    point = series[index]
    spec = REGIME_STATES[point["state"]]
    history_states = [item["state"] for item in series[max(0, index - 60): index + 1] if item]
    share = 100.0 * history_states.count(point["state"]) / max(1, len(history_states))
    return {
        "field": REGIME,
        "use": "timing",
        "available": True,
        "as_of": point["date"],
        "state": point["state"],
        "label": spec["label"],
        "tone": spec["tone"],
        "z": point["z"],
        "cut": REGIME_Z_CUT,
        "exposure": spec["exposure"],
        "guidance": spec["guidance"],
        "history": {
            "window": len(history_states),
            "share": round(share, 1),
            "switches": sum(
                1 for a, b in zip(history_states, history_states[1:]) if a != b
            ),
        },
        "benchmark": benchmark.get("name") or benchmark.get("symbol"),
        "usage": usage_note(REGIME),
        "extra": extra or {},
    }
