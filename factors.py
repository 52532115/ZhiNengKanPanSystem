"""纯 OHLCV 因子库：五族，全部因果、只用标准库。

设计约束
--------
1. **只用日线的 open/high/low/close/volume/amount/turnover**，不引入财务、公告、
   行业指数等外部数据；行业分类只在横截面中性化时用到（`industry.json` 快照）。
2. **所有序列因果**：第 i 个值只依赖 ≤ i 的数据，便于逐日截断回放。
3. **符号先验写死**（`prior`）：+1 表示"值越大越看多"，−1 表示"值越大越看空"。
   先验在跑数据之前就定好，事后不许因为 IC 反号就翻转——反号本身是结论。

市值口径
--------
没有逐日总股本时，用 ``amount / (turnover/100)`` 反推**自由流通市值**（turnover 是
成交量/流通股本，amount 是成交额）。这条恒等式也说明一件事：``成交额/流通市值``
与换手率是同一个量，所以本库只保留换手率，不再单独造一个"成交额/市值"因子来
重复计价——用户原清单里的这一项在这里被合并掉了。
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Sequence

# --------------------------------------------------------------------------- #
# 因子定义：族、标签、先验符号、公式、经济逻辑
# --------------------------------------------------------------------------- #

FAMILY_SPECS: dict[str, dict[str, Any]] = {
    "reversal": {
        "label": "反转",
        "rationale": "A 股散户占比高、短期过度反应明显，5 日与隔夜跳空后的均值回复最强，"
                     "但换手极高，必须与成本一起看。",
        "prior": "短期收益与跳空为负向",
    },
    "momentum": {
        "label": "动量",
        "rationale": "中期趋势延续（12–1 跳过最近一月以避开短期反转），趋势的平滑程度"
                     "（斜率 × R²）比单纯的涨幅更能区分「真趋势」与「震荡」。",
        "prior": "中期动量为正向，加速度为正向",
    },
    "risk": {
        "label": "风险",
        "rationale": "低风险异象：高特质波动、高 beta、深回撤、尖峰在 A 股长期跑输；"
                     "偏度为正（彩票型）反而被高估。",
        "prior": "波动 / beta / 回撤 / 峰度为负向，偏度为正向",
    },
    "liquidity": {
        "label": "流动性",
        "rationale": "Amihud 非流动性溢价；换手率过高往往是情绪与拥挤；"
                     "放量上涨后的量价背离预示动能衰减。",
        "prior": "非流动性为正向，换手率为负向，量价背离为负向",
    },
    "pattern": {
        "label": "价格形态",
        "rationale": "收盘在日内区间的位置、均线多头排列反映买盘持续性；"
                     "ATR 与布林带位置刻画波动与超买。",
        "prior": "收盘位置 / 多头排列为正向，ATR 与布林带上轨为负向",
    },
}

FACTOR_SPECS: dict[str, dict[str, Any]] = {
    # 反转
    "rev5": {"family": "reversal", "label": "5 日反转", "prior": +1,
             "formula": "−(close_t / close_{t−5} − 1)",
             "note": "短期过度反应；换手最高的表达之一"},
    "rev1": {"family": "reversal", "label": "1 日反转", "prior": +1,
             "formula": "−(close_t / close_{t−1} − 1)",
             "note": "日频反转，几乎全部收益被交易成本吃掉，只作对照"},
    "gap5": {"family": "reversal", "label": "隔夜跳空（5 日）", "prior": +1,
             "formula": "−mean_{5}(open_t / close_{t−1} − 1)",
             "note": "跳空回补；同时是流动性/情绪的代理"},
    # 动量
    "mom12_1": {"family": "momentum", "label": "12–1 动量", "prior": +1,
                "formula": "close_{t−21} / close_{t−252} − 1",
                "note": "跳过最近一月，经典 TSMOM 表达"},
    "mom6_1": {"family": "momentum", "label": "6–1 动量", "prior": +1,
               "formula": "close_{t−21} / close_{t−126} − 1",
               "note": "更短的中期动量，A 股上通常弱于 12–1"},
    "high52": {"family": "momentum", "label": "52 周高点距离", "prior": +1,
               "formula": "close_t / max(high_{t−251..t})",
               "note": "接近一年高点 = 无套牢盘；取值 (0,1]"},
    "slope_r2": {"family": "momentum", "label": "趋势斜率 × R²", "prior": +1,
                 "formula": "60 日 log 价格对时间回归的斜率 × R²",
                 "note": "用拟合优度奖励「平滑」的趋势，抑制震荡中的假动量"},
    "accel": {"family": "momentum", "label": "动量加速度", "prior": +1,
              "formula": "(close_t/close_{t−21} − 1) − (close_{t−21}/close_{t−42} − 1)",
              "note": "二阶动量，区分「加速」与「减速」"},
    # 风险
    "ivol": {"family": "risk", "label": "特质波动率", "prior": -1,
             "formula": "60 日对市场收益回归的残差标准差 × √252",
             "note": "剥离 beta 后的个股风险；低特质波动异象"},
    "beta120": {"family": "risk", "label": "beta（120 日）", "prior": -1,
                "formula": "120 日个股收益对市场收益回归斜率",
                "note": "低 beta 异象；同时作为中性化控制变量"},
    "downside20": {"family": "risk", "label": "下行波动", "prior": -1,
                   "formula": "√mean(min(r,0)²) over 20 日 × √252",
                   "note": "只罚下跌的波动，比总波动更贴近持有体验"},
    "mdd60": {"family": "risk", "label": "60 日最大回撤", "prior": -1,
              "formula": "−min(close/滚动峰 − 1) over 60 日（取正值）",
              "note": "回撤深度；与下行波动相关但不相同"},
    "skew60": {"family": "risk", "label": "偏度（60 日）", "prior": +1,
               "formula": "60 日日收益的三阶矩偏度",
               "note": "正偏=彩票型，A 股常被高估；先验取正、由数据裁决"},
    "kurt60": {"family": "risk", "label": "峰度（60 日）", "prior": -1,
               "formula": "60 日日收益的超额峰度",
               "note": "尖峰=尾部风险，先验为负"},
    # 流动性
    "amihud": {"family": "liquidity", "label": "Amihud 非流动性", "prior": +1,
               "formula": "mean(abs(r) / 成交额亿元) over 20 日",
               "note": "单位成交额推动的价格幅度；非流动性溢价"},
    "turnover20": {"family": "liquidity", "label": "换手率（20 日）", "prior": -1,
                   "formula": "mean(turnover) over 20 日（%）",
                   "note": "拥挤度/情绪代理；等价于「成交额/流通市值」，不再重复造因子"},
    "pv_div": {"family": "liquidity", "label": "量价背离", "prior": -1,
               "formula": "corr(Δlog volume, r) over 20 日",
               "note": "正相关=放量上涨，先验认为过热；负相关=缩量下跌"},
    # 形态
    "clv20": {"family": "pattern", "label": "收盘位置（20 日）", "prior": +1,
              "formula": "mean((close−low)/(high−low)) over 20 日",
              "note": "收盘靠近日内高点=买盘占优"},
    "atr20": {"family": "pattern", "label": "ATR（20 日）", "prior": -1,
              "formula": "mean(TR/close) over 20 日（%）",
              "note": "日内波幅，与收盘价波动互补"},
    "boll_pos": {"family": "pattern", "label": "布林带位置", "prior": -1,
                 "formula": "(close − MA20) / (2 × σ20)",
                 "note": ">1 表示突破上轨，先验认为短期过热"},
    "ma_stack": {"family": "pattern", "label": "均线多头排列", "prior": +1,
                 "formula": "(MA5>MA10)+(MA10>MA20)+(MA20>MA60)，取值 0–3",
                 "note": "趋势结构的离散计数，抗噪"},
}

# 现有三族（xsec.py 的入模因子），只用于正交化与对照，不重复计入新因子。
EXISTING_FACTORS: dict[str, str] = {
    "macd_hist": "MACD 柱 / 收盘价",
    "vol_ratio": "量比（成交量 / 20 日均量）",
    "rs20": "相对沪深300 的 20 日超额收益",
}

CONTROLS: tuple[str, ...] = ("log_cap", "beta120")

MAX_WINDOW = 252


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #


def _cumsum(values: Sequence[float]) -> list[float]:
    out = [0.0]
    running = 0.0
    for value in values:
        running += value
        out.append(running)
    return out


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
    result: list[float | None] = [None] * len(values)
    queue: list[int] = []
    for index, value in enumerate(values):
        while queue and values[queue[-1]] <= value:
            queue.pop()
        queue.append(index)
        if queue[0] <= index - window:
            queue.pop(0)
        if index >= window - 1:
            result[index] = values[queue[0]]
    return result


def _returns(closes: Sequence[float]) -> list[float | None]:
    return [None] + [
        (closes[i] / closes[i - 1] - 1.0) if closes[i - 1] > 0 else None
        for i in range(1, len(closes))
    ]


def _rolling_regression(
    y: Sequence[float | None],
    x: Sequence[float | None],
    window: int,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """滚动 OLS ``y = a + b·x``：返回 (斜率, R², 残差标准差)。

    用前缀和做 O(1) 更新；任一侧缺值的窗口整段跳过（不做插补，避免制造信息）。
    """
    n = len(y)
    py, px = _cumsum([0.0 if v is None else v for v in y]), _cumsum([0.0 if v is None else v for v in x])
    pyy, pxx = _cumsum([0.0 if v is None else v * v for v in y]), _cumsum([0.0 if v is None else v * v for v in x])
    pxy = _cumsum([
        0.0 if (a is None or b is None) else a * b for a, b in zip(y, x)
    ])
    bad = _cumsum([1.0 if (a is None or b is None) else 0.0 for a, b in zip(y, x)])

    slope: list[float | None] = [None] * n
    r2: list[float | None] = [None] * n
    resid: list[float | None] = [None] * n
    for index in range(window - 1, n):
        start = index - window + 1
        if bad[index + 1] - bad[start] > 0:
            continue
        count = float(window)
        sy = py[index + 1] - py[start]
        sx = px[index + 1] - px[start]
        syy = pyy[index + 1] - pyy[start]
        sxx = pxx[index + 1] - pxx[start]
        sxy = pxy[index + 1] - pxy[start]
        den = sxx - sx * sx / count
        if abs(den) < 1e-12:
            continue
        b = (sxy - sx * sy / count) / den
        a = (sy - b * sx) / count
        var_y = max(0.0, (syy - sy * sy / count) / count)
        var_e = max(0.0, (syy - a * sy - b * sxy) / count)
        slope[index] = b
        r2[index] = 0.0 if var_y <= 1e-16 else max(0.0, 1.0 - var_e / var_y)
        resid[index] = math.sqrt(var_e)
    return slope, r2, resid


def _rolling_skew_kurt(values: Sequence[float | None], window: int) -> tuple[list[float | None], list[float | None]]:
    skew: list[float | None] = [None] * len(values)
    kurt: list[float | None] = [None] * len(values)
    for index in range(window - 1, len(values)):
        sample = [v for v in values[index - window + 1: index + 1] if v is not None]
        if len(sample) < window - 2:
            continue
        mean = statistics.fmean(sample)
        deviation = statistics.pstdev(sample)
        if deviation <= 1e-12:
            continue
        skew[index] = statistics.fmean([((v - mean) / deviation) ** 3 for v in sample])
        kurt[index] = statistics.fmean([((v - mean) / deviation) ** 4 for v in sample]) - 3.0
    return skew, kurt


def _rolling_corr(left: Sequence[float | None], right: Sequence[float | None], window: int) -> list[float | None]:
    result: list[float | None] = [None] * len(left)
    for index in range(window - 1, len(left)):
        pairs = [
            (a, b)
            for a, b in zip(left[index - window + 1: index + 1], right[index - window + 1: index + 1])
            if a is not None and b is not None
        ]
        if len(pairs) < window - 2:
            continue
        xs = [a for a, _ in pairs]
        ys = [b for _, b in pairs]
        mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
        sd_x, sd_y = statistics.pstdev(xs), statistics.pstdev(ys)
        if sd_x <= 1e-12 or sd_y <= 1e-12:
            continue
        cov = statistics.fmean([(x - mean_x) * (y - mean_y) for x, y in pairs])
        result[index] = cov / (sd_x * sd_y)
    return result


def _mean_window(values: Sequence[float | None], window: int) -> list[float | None]:
    prefix = _cumsum([0.0 if v is None else v for v in values])
    count = _cumsum([0.0 if v is None else 1.0 for v in values])
    result: list[float | None] = [None] * len(values)
    for index in range(window - 1, len(values)):
        start = index - window + 1
        size = count[index + 1] - count[start]
        if size >= window * 0.8:
            result[index] = (prefix[index + 1] - prefix[start]) / size
    return result


# --------------------------------------------------------------------------- #
# 单标的因子序列
# --------------------------------------------------------------------------- #


def float_cap(history: dict[str, Any]) -> list[float | None]:
    """自由流通市值（亿元）：``成交额 / (换手率/100)``。

    换手率为 0（停牌）或缺失时置空，由下游做前值填充。这条恒等式同时说明
    "成交额/流通市值"与换手率是同一个量，因子表里不再重复。
    """
    result: list[float | None] = []
    for row in history["rows"]:
        amount = float(row.get("amount") or 0.0)
        turnover = float(row.get("turnover") or 0.0)
        if amount > 0 and turnover > 0:
            result.append(amount / (turnover / 100.0) / 1e8)
        else:
            result.append(None)
    return result


def compute(history: dict[str, Any], market_returns: dict[str, float]) -> dict[str, list[float | None]]:
    """把一只标的的日线转成所有因子的因果序列。"""
    rows = history["rows"]
    count = len(rows)
    closes = [float(row["close"]) for row in rows]
    opens = [float(row["open"]) for row in rows]
    highs = [float(row["high"]) for row in rows]
    lows = [float(row["low"]) for row in rows]
    volumes = [float(row["volume"]) for row in rows]
    amounts = [float(row.get("amount") or 0.0) for row in rows]
    turnovers = [float(row.get("turnover") or 0.0) for row in rows]
    dates = [row["date"] for row in rows]
    returns = _returns(closes)
    market = [market_returns.get(date) for date in dates]

    factors: dict[str, list[float | None]] = {
        key: [None] * count for key in FACTOR_SPECS
    }

    # ---- 反转 ----
    for index in range(5, count):
        if closes[index - 5] > 0:
            factors["rev5"][index] = -(closes[index] / closes[index - 5] - 1.0)
    for index in range(1, count):
        if closes[index - 1] > 0:
            factors["rev1"][index] = -(closes[index] / closes[index - 1] - 1.0)
    gaps = [None] + [
        (opens[i] / closes[i - 1] - 1.0) if closes[i - 1] > 0 else None
        for i in range(1, count)
    ]
    gap_mean = _mean_window(gaps, 5)
    for index, value in enumerate(gap_mean):
        if value is not None:
            factors["gap5"][index] = -value

    # ---- 动量 ----
    for index in range(252, count):
        if closes[index - 252] > 0:
            factors["mom12_1"][index] = closes[index - 21] / closes[index - 252] - 1.0
    for index in range(126, count):
        if closes[index - 126] > 0:
            factors["mom6_1"][index] = closes[index - 21] / closes[index - 126] - 1.0
    high_max = _rolling_max(highs, 252)
    for index, peak in enumerate(high_max):
        if peak:
            factors["high52"][index] = closes[index] / peak
    log_close = [math.log(c) if c > 0 else None for c in closes]
    time_index = [float(i) for i in range(count)]
    slope, r2, _ = _rolling_regression(log_close, time_index, 60)
    for index in range(count):
        if slope[index] is not None and r2[index] is not None:
            factors["slope_r2"][index] = slope[index] * 100 * r2[index]
    for index in range(42, count):
        if closes[index - 42] > 0 and closes[index - 21] > 0:
            recent = closes[index] / closes[index - 21] - 1.0
            prior = closes[index - 21] / closes[index - 42] - 1.0
            factors["accel"][index] = recent - prior

    # ---- 风险 ----
    # 60 日回归只取残差（特质波动）；beta 用 120 日，慢一点更稳。
    _, _, resid = _rolling_regression(returns, market, 60)
    for index in range(count):
        if resid[index] is not None:
            factors["ivol"][index] = resid[index] * math.sqrt(252) * 100
    beta_long, _, _ = _rolling_regression(returns, market, 120)
    for index in range(count):
        if beta_long[index] is not None:
            factors["beta120"][index] = beta_long[index]
    for index in range(20, count):
        sample = [r for r in returns[index - 19: index + 1] if r is not None]
        negative = [r for r in sample if r < 0]
        if len(sample) >= 16:
            factors["downside20"][index] = (
                math.sqrt(statistics.fmean([r * r for r in negative])) * math.sqrt(252) * 100
                if negative else 0.0
            )
    for index in range(60, count):
        window = closes[index - 59: index + 1]
        peak = window[0]
        worst = 0.0
        for price in window:
            peak = max(peak, price)
            if peak > 0:
                worst = min(worst, price / peak - 1.0)
        factors["mdd60"][index] = -worst * 100
    skew, kurt = _rolling_skew_kurt(returns, 60)
    for index in range(count):
        if skew[index] is not None:
            factors["skew60"][index] = skew[index]
        if kurt[index] is not None:
            factors["kurt60"][index] = kurt[index]

    # ---- 流动性 ----
    amihud_input = [
        (abs(r) / (amount / 1e8)) if (r is not None and amount > 0) else None
        for r, amount in zip(returns, amounts)
    ]
    amihud = _mean_window(amihud_input, 20)
    for index, value in enumerate(amihud):
        if value is not None:
            factors["amihud"][index] = value
    turnover_series = [t if t > 0 else None for t in turnovers]
    turnover_mean = _mean_window(turnover_series, 20)
    for index, value in enumerate(turnover_mean):
        if value is not None:
            factors["turnover20"][index] = value
    log_volume = [math.log(v) if v > 0 else None for v in volumes]
    volume_change = [None] + [
        (log_volume[i] - log_volume[i - 1])
        if (log_volume[i] is not None and log_volume[i - 1] is not None) else None
        for i in range(1, count)
    ]
    pv = _rolling_corr(volume_change, returns, 20)
    for index, value in enumerate(pv):
        if value is not None:
            factors["pv_div"][index] = value

    # ---- 形态 ----
    clv = [
        ((c - low) / (high - low)) if high > low else 0.5
        for c, high, low in zip(closes, highs, lows)
    ]
    clv_mean = _mean_window(clv, 20)
    for index, value in enumerate(clv_mean):
        if value is not None:
            factors["clv20"][index] = value
    true_range: list[float | None] = [None]
    for index in range(1, count):
        previous = closes[index - 1]
        true_range.append(max(
            highs[index] - lows[index],
            abs(highs[index] - previous),
            abs(lows[index] - previous),
        ))
    tr_ratio = [
        (tr / c * 100) if (tr is not None and c > 0) else None
        for tr, c in zip(true_range, closes)
    ]
    atr = _mean_window(tr_ratio, 20)
    for index, value in enumerate(atr):
        if value is not None:
            factors["atr20"][index] = value
    ma20 = _sma(closes, 20)
    for index in range(20, count):
        window = closes[index - 19: index + 1]
        if ma20[index] and len(window) >= 20:
            deviation = statistics.pstdev(window)
            if deviation > 1e-9:
                factors["boll_pos"][index] = (closes[index] - ma20[index]) / (2 * deviation)
    ma5, ma10, ma60 = _sma(closes, 5), _sma(closes, 10), _sma(closes, 60)
    for index in range(60, count):
        if None in (ma5[index], ma10[index], ma20[index], ma60[index]):
            continue
        stack = (
            int(ma5[index] > ma10[index])
            + int(ma10[index] > ma20[index])
            + int(ma20[index] > ma60[index])
        )
        factors["ma_stack"][index] = float(stack)

    return factors


def existing(history: dict[str, Any], benchmark_returns: dict[str, dict[str, float]]) -> dict[str, list[float | None]]:
    """现有三族因子的复刻（公式与 ``xsec._symbol_series`` 一致），用于正交化对照。"""
    rows = history["rows"]
    closes = [float(row["close"]) for row in rows]
    volumes = [float(row["volume"]) for row in rows]
    count = len(rows)
    alpha = 2 / 13
    ema12 = [closes[0]] if closes else []
    for value in closes[1:]:
        ema12.append(alpha * value + (1 - alpha) * ema12[-1])
    alpha26 = 2 / 27
    ema26 = [closes[0]] if closes else []
    for value in closes[1:]:
        ema26.append(alpha26 * value + (1 - alpha26) * ema26[-1])
    dif = [fast - slow for fast, slow in zip(ema12, ema26)]
    alpha9 = 2 / 10
    dea = [dif[0]] if dif else []
    for value in dif[1:]:
        dea.append(alpha9 * value + (1 - alpha9) * dea[-1])
    hist = [(d - e) * 2 for d, e in zip(dif, dea)]
    volume_ma = _sma(volumes, 20)

    series: dict[str, list[float | None]] = {key: [None] * count for key in EXISTING_FACTORS}
    for index in range(count):
        close = closes[index]
        if close:
            series["macd_hist"][index] = hist[index] / close * 100
        if volume_ma[index]:
            series["vol_ratio"][index] = volumes[index] / volume_ma[index]
        if index >= 20 and closes[index - 20] > 0:
            ret20 = (closes[index] / closes[index - 20] - 1.0) * 100
            bench = benchmark_returns.get("ret20", {}).get(rows[index]["date"])
            if bench is not None:
                series["rs20"][index] = ret20 - bench
    return series


def benchmark_returns(benchmark: dict[str, Any]) -> dict[str, dict[str, float]]:
    rows = benchmark["rows"]
    closes = [float(row["close"]) for row in rows]
    result: dict[str, dict[str, float]] = {"ret1": {}, "ret20": {}}
    for index, row in enumerate(rows):
        if index >= 1 and closes[index - 1] > 0:
            result["ret1"][row["date"]] = closes[index] / closes[index - 1] - 1.0
        if index >= 20 and closes[index - 20] > 0:
            result["ret20"][row["date"]] = (closes[index] / closes[index - 20] - 1.0) * 100
    return result


# --------------------------------------------------------------------------- #
# 横截面工具
# --------------------------------------------------------------------------- #


def rank_pct(values: Sequence[float]) -> list[float]:
    """横截面百分位（0–100，并列取平均名次）。"""
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
            result[order[position]] = 100.0 * (average + 0.5) / n
        start = end + 1
    return result


def orthogonalize(target: Sequence[float], controls: Sequence[Sequence[float]]) -> list[float]:
    """把 ``target`` 对若干控制变量做横截面 OLS，返回残差。

    控制变量先各自中心化，再用正规方程求解（维数 ≤ 4，直接高斯消元）。
    """
    n = len(target)
    if n == 0:
        return []
    mean_y = statistics.fmean(target)
    design = [[value - statistics.fmean(column) for value in column] for column in controls]
    size = len(design)
    if size == 0:
        return [value - mean_y for value in target]
    matrix = [[0.0] * size for _ in range(size)]
    vector = [0.0] * size
    for row in range(n):
        for i in range(size):
            vector[i] += design[i][row] * (target[row] - mean_y)
            for j in range(i, size):
                matrix[i][j] += design[i][row] * design[j][row]
    for i in range(size):
        for j in range(i):
            matrix[i][j] = matrix[j][i]
    # 高斯消元（带极小岭项，避免共线时炸掉）
    for i in range(size):
        matrix[i][i] += 1e-10
    aug = [matrix[i][:] + [vector[i]] for i in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(aug[r][column]))
        if abs(aug[pivot][column]) < 1e-14:
            continue
        aug[column], aug[pivot] = aug[pivot], aug[column]
        divisor = aug[column][column]
        for j in range(column, size + 1):
            aug[column][j] /= divisor
        for row in range(size):
            if row == column:
                continue
            factor = aug[row][column]
            if abs(factor) < 1e-16:
                continue
            for j in range(column, size + 1):
                aug[row][j] -= factor * aug[column][j]
    coefficients = [aug[i][size] for i in range(size)]
    return [
        (target[row] - mean_y) - sum(coefficients[i] * design[i][row] for i in range(size))
        for row in range(n)
    ]
