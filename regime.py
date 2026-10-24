"""Ex-ante regime 分类、暴露缩放与因子倾斜（纯规则，不拟合参数）。

为什么是规则而不是模型
----------------------
样本只有 10 年、标签周期 20 日，有效独立块约 120 个。在这个样本量上训练一个
regime 分类器（哪怕只有三个输入），等价于给"哪段是高波动"这件事本身过拟合；
而 regime 标签一旦过拟合，后面所有"分状态评估"都会自我循环。所以这里只做三件
事：

1. **分类器是三个先验规则的多数票**，阈值写死（MA200、波动分位 80%、广度 40%），
   没有任何参数在数据上选过。三个输入都只用 T 日及以前的收盘数据。
2. **暴露只做二值缩放**（100% / 50%），不切换因子权重。一个参数、一个自由度。
3. **切换慢**：分数 5 日平滑 + 连续 5 日确认的 hysteresis，把"日频抖动"过滤掉。

三段结构
--------
    index_rows + breadth ──► classify() ──► [RegimePoint] ──► exposure_for_open()
                                                  │                    │
                                                  └──► switch_log() ◄──┘  （滞后成本）

``RegimePoint.state`` 是"确认后"的状态（hysteresis 生效），``raw_state`` 是当日
未确认的原始判断。两者之差就是切换滞后的来源，验证时必须分开统计。
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

# --------------------------------------------------------------------------- #
# 口径常量：全部写死，不参与调参
# --------------------------------------------------------------------------- #

MA_WINDOW = 200        # 长期趋势线
VOL_FAST = 20          # 短期已实现波动
VOL_SLOW = 60          # 中期已实现波动
VOL_LOOKBACK = 504     # 波动分位的历史窗口（约 2 年）
BREADTH_WINDOW = 20    # 广度：收盘价在 MA20 上方的个股占比（再取 20 日均）
VOL_HIGH_PCT = 0.80    # 波动分位高于 80% 视为高波动
BREADTH_LOW = 0.40     # 广度低于 40% 视为走弱
SCORE_SMOOTH = 5       # 慢信号：票数取 5 日均
CONFIRM_DAYS = 5       # hysteresis：连续 5 日反向才切换
RISK_OFF_EXPOSURE = 0.50
MIN_HISTORY = 260      # 前 260 个交易日不出信号（MA200 + 波动窗口预热）


@dataclass(frozen=True)
class RegimeConfig:
    ma_window: int = MA_WINDOW
    vol_fast: int = VOL_FAST
    vol_slow: int = VOL_SLOW
    vol_lookback: int = VOL_LOOKBACK
    vol_high_pct: float = VOL_HIGH_PCT
    breadth_low: float = BREADTH_LOW
    breadth_smooth: int = BREADTH_WINDOW
    score_smooth: int = SCORE_SMOOTH
    confirm_days: int = CONFIRM_DAYS
    stop_drawdown: float | None = None   # 熔断腿：距 60 日高点的回撤阈值（None=关闭）
    stop_window: int = 60
    risk_off_exposure: float = RISK_OFF_EXPOSURE
    min_history: int = MIN_HISTORY
    # 趋势是主腿：200 日均线是三个输入里唯一直接刻画"趋势状态"的，波动与广度
    # 是"风险状态"的确认项。0.50 意味着跌破年线单独就能触发 risk-off（经典规则），
    # 波动+广度合计 0.50 也能触发，广度单独 0.20 不足以触发。
    trend_weight: float = 0.50
    vol_weight: float = 0.30
    breadth_weight: float = 0.20

    def __post_init__(self) -> None:
        total = self.trend_weight + self.vol_weight + self.breadth_weight
        if abs(total - 1.0) > 1e-9:
            raise ValueError("三个权重之和必须为 1")
        if not 0 < self.risk_off_exposure <= 1:
            raise ValueError("risk_off_exposure 必须在 (0, 1]")
        if self.confirm_days < 1 or self.score_smooth < 1:
            raise ValueError("confirm_days / score_smooth 必须 ≥ 1")


@dataclass
class RegimePoint:
    date: str
    trend_off: bool
    vol_fast_pct: float | None
    vol_slow_pct: float | None
    breadth: float | None
    votes: float            # 当日"看空票数"，0–1
    slow_votes: float       # 5 日平滑后的票数
    raw_state: str          # risk_on / risk_off（未确认）
    state: str              # 确认后的状态
    exposure: float
    flipped: bool = False   # 当日是否发生了确认切换
    raw_flipped: bool = False
    stop: bool = False      # 当日是否由熔断腿强制 risk_off

    @property
    def risk_off(self) -> bool:
        return self.state == "risk_off"


# --------------------------------------------------------------------------- #
# 基础指标（全部因果）
# --------------------------------------------------------------------------- #


def realized_vol(closes: Sequence[float], index: int, window: int) -> float | None:
    """``index`` 往前 ``window`` 个交易日的年化已实现波动（%），只用 ≤ index 的数据。"""
    if index < window:
        return None
    sample = closes[index - window: index + 1]
    returns = [
        math.log(sample[i] / sample[i - 1])
        for i in range(1, len(sample))
        if sample[i - 1] > 0 and sample[i] > 0
    ]
    if len(returns) < window - 2:
        return None
    return statistics.pstdev(returns) * math.sqrt(252) * 100


def percentile_rank(series: Sequence[float], value: float) -> float:
    """``value`` 在 ``series``（含当日）里的分位，0–1，并列取中位。"""
    if not series:
        return 0.5
    below = sum(1 for item in series if item < value)
    equal = sum(1 for item in series if item == value)
    return (below + 0.5 * equal) / len(series)


# --------------------------------------------------------------------------- #
# 分类器
# --------------------------------------------------------------------------- #


def classify(
    index_rows: Sequence[dict[str, Any]],
    breadth_by_date: dict[str, float],
    config: RegimeConfig | None = None,
) -> list[RegimePoint]:
    """逐日出 regime 状态。

    输入
    ----
    ``index_rows``：基准日线（含 ``date`` / ``close``），必须按日期升序。
    ``breadth_by_date``：``date -> 0–1``，当日收盘价在自身 MA20 上方的个股占比。

    输出
    ----
    ``[RegimePoint]``，长度与 ``index_rows`` 相同；预热期内 ``state`` 为空字符串
    （下游按"无信号 = 满仓"处理，而不是当成 risk_on）。

    三张票的权重是 0.50 / 0.30 / 0.20，阈值 0.5：跌破年线**单独**即可触发
    （这是最经典也最稳的一条 regime 规则），波动+广度同时看空也触发（0.50），
    广度单独（0.20）不足以触发。

    早期版本用 0.40 / 0.35 / 0.25（等价"三选二"），回测发现一个致命盲区：
    **从高位开始的缓跌**里 MA200 还没破、波动也没进前 20%，三票凑不齐两票，
    于是全程 risk_on——2026 年 1–3 月（−13.8%）就是这样被完整漏掉的。
    """
    cfg = config or RegimeConfig()
    closes = [float(row["close"]) for row in index_rows]
    dates = [row["date"] for row in index_rows]
    vol_fast_series: list[float | None] = [
        realized_vol(closes, i, cfg.vol_fast) for i in range(len(closes))
    ]
    vol_slow_series: list[float | None] = [
        realized_vol(closes, i, cfg.vol_slow) for i in range(len(closes))
    ]

    points: list[RegimePoint] = []
    vote_history: list[float] = []
    breadth_history: list[float] = []
    state = ""
    candidate = ""
    streak = 0

    for index, date in enumerate(dates):
        if index < cfg.min_history:
            points.append(RegimePoint(date, False, None, None, None, 0.0, 0.0, "", "", 1.0))
            continue

        window = closes[index - cfg.ma_window + 1: index + 1]
        ma = statistics.fmean(window) if len(window) == cfg.ma_window else None
        trend_off = bool(ma and closes[index] < ma)

        fast = vol_fast_series[index]
        slow = vol_slow_series[index]
        fast_window = [
            value for value in vol_fast_series[max(0, index - cfg.vol_lookback + 1): index + 1]
            if value is not None
        ]
        slow_window = [
            value for value in vol_slow_series[max(0, index - cfg.vol_lookback + 1): index + 1]
            if value is not None
        ]
        fast_pct = percentile_rank(fast_window, fast) if fast is not None and len(fast_window) > 60 else None
        slow_pct = percentile_rank(slow_window, slow) if slow is not None and len(slow_window) > 60 else None
        raw_breadth = breadth_by_date.get(date)
        if raw_breadth is not None:
            breadth_history.append(raw_breadth)
        # 广度是"涨跌家数"类指标，日频抖动极大（一天的普涨就能把它从 0.2 推到 0.8），
        # 取 20 日均才是常见口径，也才配得上"慢信号"。
        breadth = (
            statistics.fmean(breadth_history[-cfg.breadth_smooth:])
            if breadth_history else None
        )

        votes = cfg.trend_weight * float(trend_off)
        if fast_pct is not None and fast_pct >= cfg.vol_high_pct:
            votes += cfg.vol_weight
        if breadth is not None and breadth < cfg.breadth_low:
            votes += cfg.breadth_weight

        vote_history.append(votes)
        window_votes = vote_history[-cfg.score_smooth:]
        slow_votes = statistics.fmean(window_votes)

        raw_state = "risk_off" if slow_votes >= 0.5 else "risk_on"
        raw_flipped = False
        flipped = False

        # 熔断腿：三票分类器是慢信号（MA200 要跌破才翻面、波动分位在泡沫期长期
        # 高位却凑不齐两票），对 2015-06 这种"从高位直接崩"的行情完全无效。
        # 这条只用"距 60 日高点的回撤"，仍然全是 T 日及以前的数据，但反应快得多。
        stop = False
        if cfg.stop_drawdown is not None and index >= cfg.stop_window:
            peak = max(closes[index - cfg.stop_window + 1: index + 1])
            drawdown = closes[index] / peak - 1.0
            stop = drawdown <= -cfg.stop_drawdown

        if stop:
            # 快速出场、慢速进场：熔断只负责把仓位压下来，恢复仍然要 5 日确认，
            # 否则会在暴跌后的反弹里反复进出来回打脸。
            if state != "risk_off":
                if state:
                    flipped = True
                state = "risk_off"
            raw_state = "risk_off"
            candidate = "risk_off"
            streak = 0
        elif state == "":
            # 预热结束后的第一天直接落位，不设确认期（此前没有仓位被判定的状态）
            state = raw_state
            candidate = raw_state
            streak = 0
        else:
            if raw_state != state:
                if candidate == raw_state:
                    streak += 1
                else:
                    candidate = raw_state
                    streak = 1
                if streak >= cfg.confirm_days:
                    state = raw_state
                    flipped = True
                    raw_flipped = True
                    streak = 0
            else:
                candidate = raw_state
                streak = 0

        if not flipped:
            raw_flipped = raw_state != state

        exposure = cfg.risk_off_exposure if state == "risk_off" else 1.0
        points.append(
            RegimePoint(
                date=date,
                trend_off=trend_off,
                vol_fast_pct=fast_pct,
                vol_slow_pct=slow_pct,
                breadth=breadth,
                votes=votes,
                slow_votes=slow_votes,
                raw_state=raw_state,
                state=state,
                exposure=exposure,
                flipped=flipped,
                raw_flipped=raw_flipped,
                stop=stop,
            )
        )
    return points


def exposure_for_open(points: Sequence[RegimePoint]) -> dict[str, float]:
    """把"收盘确认的状态"平移到"次日开盘可用的仓位"。

    ``simulate_t1`` 在 T+1 开盘执行 T 日收盘的信号，所以 T+1 开盘时能用的最新
    状态是 T 日收盘确认的那个；直接用当天的 ``state`` 会偷看当天收盘后才知道
    的信息。
    """
    mapping: dict[str, float] = {}
    previous = 1.0
    for point in points:
        mapping[point.date] = previous
        previous = point.exposure if point.state else previous
    return mapping


def switch_log(points: Sequence[RegimePoint]) -> list[dict[str, Any]]:
    """确认切换事件列表：用于统计切换频率与滞后成本。

    ``raw_date`` 是原始判断第一次翻面的日期，``confirm_date`` 是确认生效的那天，
    两者之间的交易日数就是 hysteresis 付出的滞后。

    注意两种"翻面失败"：原始判断翻过去又翻回来时，``divergence_start`` 必须
    清空，否则滞后天数会把几次无效抖动累加进去，把切换成本算大。
    """
    events: list[dict[str, Any]] = []
    divergence_start: str | None = None
    dates = [point.date for point in points]
    for index, point in enumerate(points):
        if not point.state:
            continue
        if point.flipped:
            start_date = divergence_start or point.date
            events.append({
                "raw_date": start_date,
                "confirm_date": point.date,
                "from": points[index - 1].state if index else "",
                "to": point.state,
                "lag_days": _gap(dates, start_date, point.date),
            })
            divergence_start = None
            continue
        if point.raw_state != point.state:
            if divergence_start is None:
                divergence_start = point.date
        else:
            divergence_start = None
    return events


def _gap(dates: Sequence[str], start: str, end: str) -> int:
    try:
        return dates.index(end) - dates.index(start)
    except ValueError:
        return 0


# --------------------------------------------------------------------------- #
# 因子倾斜（第二层，默认关闭）
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TiltConfig:
    """risk-off 时的权重倾斜：只压动量、抬防御，两个参数。

    ``damp`` < 1 表示 risk-off 时把整族动量的权重打折；``boost`` > 1 表示抬升
    反转与低波/质量。倾斜后按 L1 归一回原来的总权重，避免"顺手把总暴露也改了"。
    """

    damp_momentum: float = 0.40
    boost_defensive: float = 1.60
    momentum: tuple[str, ...] = ("mom12_1", "macd_hist", "rs20", "high52")
    defensive: tuple[str, ...] = ("rev5", "ivol", "downside20", "mdd60")


def tilt_weights(
    weights: dict[str, float],
    *,
    state: str,
    config: TiltConfig | None = None,
) -> dict[str, float]:
    """按状态返回一套权重；risk_on 原样返回，risk_off 才倾斜。

    只改**相对**权重、不改总 L1：否则"降暴露"和"换因子"两个动作会混在一起，
    回撤改善就无法归因。真正的降暴露由 ``exposure`` 单独负责。
    """
    if state != "risk_off":
        return dict(weights)
    cfg = config or TiltConfig()
    tilted: dict[str, float] = {}
    for key, value in weights.items():
        if key in cfg.momentum:
            tilted[key] = value * cfg.damp_momentum
        elif key in cfg.defensive:
            tilted[key] = value * cfg.boost_defensive
        else:
            tilted[key] = value
    total = sum(abs(value) for value in tilted.values())
    scale = sum(abs(value) for value in weights.values()) / total if total > 1e-12 else 1.0
    return {key: value * scale for key, value in tilted.items()}


# --------------------------------------------------------------------------- #
# 诊断：状态是否真的领先于风险
# --------------------------------------------------------------------------- #


def forward_stats(
    points: Sequence[RegimePoint],
    closes: Sequence[float],
    *,
    horizon: int = 20,
) -> dict[str, Any]:
    """按状态统计未来 ``horizon`` 日的已实现波动与最大回撤（用于验证领先性）。

    这是判断"regime 分类是不是在描述未来"的唯一诚实做法：分类器只看过去，
    所以只要 ``risk_off`` 组的未来波动显著更高、未来回撤更深，它就有信息量。
    拿它跟"事后最优状态"的相关性评估没有意义。
    """
    buckets: dict[str, list[dict[str, float]]] = {"risk_on": [], "risk_off": []}
    for index, point in enumerate(points):
        if not point.state or index + horizon >= len(closes):
            continue
        window = closes[index: index + horizon + 1]
        returns = [math.log(window[i] / window[i - 1]) for i in range(1, len(window))]
        vol = statistics.pstdev(returns) * math.sqrt(252) * 100
        peak = window[0]
        drawdown = 0.0
        for price in window:
            peak = max(peak, price)
            drawdown = min(drawdown, price / peak - 1.0)
        forward_return = (window[-1] / window[0] - 1.0) * 100
        buckets[point.state].append({
            "vol": vol,
            "mdd": drawdown * 100,
            "return": forward_return,
        })
    result: dict[str, Any] = {}
    for state, rows in buckets.items():
        if not rows:
            result[state] = {"n": 0}
            continue
        result[state] = {
            "n": len(rows),
            "vol": round(statistics.fmean([row["vol"] for row in rows]), 2),
            "mdd": round(statistics.fmean([row["mdd"] for row in rows]), 2),
            "return": round(statistics.fmean([row["return"] for row in rows]), 2),
            "negative_rate": round(sum(1 for row in rows if row["return"] < 0) / len(rows) * 100, 1),
        }
    if result.get("risk_on", {}).get("n") and result.get("risk_off", {}).get("n"):
        result["spread"] = {
            "vol": round(result["risk_off"]["vol"] - result["risk_on"]["vol"], 2),
            "mdd": round(result["risk_off"]["mdd"] - result["risk_on"]["mdd"], 2),
            "return": round(result["risk_off"]["return"] - result["risk_on"]["return"], 2),
        }
    return result


def state_by_year(points: Sequence[RegimePoint]) -> dict[str, dict[str, Any]]:
    """逐年状态占比与切换次数，用于检查规则在不同年份是否都成立。"""
    years: dict[str, dict[str, Any]] = {}
    for point in points:
        if not point.state:
            continue
        year = point.date[:4]
        bucket = years.setdefault(year, {"days": 0, "risk_off": 0, "flips": 0, "dates": []})
        bucket["days"] += 1
        bucket["risk_off"] += int(point.state == "risk_off")
        bucket["flips"] += int(point.flipped)
        bucket["dates"].append(point.date)
    for bucket in years.values():
        bucket["risk_off_share"] = round(bucket["risk_off"] / bucket["days"] * 100, 1)
        bucket.pop("dates", None)
    return years
