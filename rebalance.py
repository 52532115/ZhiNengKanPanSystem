"""降换手的组合构建：buffer 区间、最短持有期、分批调仓、成本感知选参、容量。

为什么需要这一层
----------------
Top 20% / 5 日调仓的问题不在信号，在**换手**：只要排名在阈值附近抖动，组合就
整进整出，年换手能到 1900%+，单边 20bp 就是 400bp/年的成本，把 5% 出头的毛
超额吃掉大半。四种手段，各自解决不同的问题：

1. **buffer zone**（进出阈值不对称 + 最短持有期）：把"排名在阈值附近"的抖动
   过滤掉。进入要进前 15%，被踢出要跌出前 40%，中间地带默认不动。
2. **分批调仓**（tranche）：每次只走完"现状 → 目标"差距的一部分，等于把一次
   调仓摊成几个调仓日，降低单日冲击，但会拖慢信号落地。
3. **信号慢化**（EWMA 平滑分位）：从信号端减少抖动，和 buffer 解决同一类问题
   但代价不同——平滑会真实降低 IC，buffer 不会。
4. **成本感知的目标**：选参时不用毛收益，用扣成本后的净 Sharpe，并把换手
   上限当作约束，而不是事后解释。

关键指标里特意加了「buffer 前后 IC 变化」：buffer 冻结了持仓的评分，等于用
"入场时的评分"替代"当前评分"。如果这个替代让 IC 掉太多，说明信号本身衰减快，
buffer 省下的成本是拿 alpha 换的，不划算。
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

TARGET = 0.20          # 目标持仓比例（Top 20%）
HARD_STOP = 0.70       # 排名跌出前 70% 且持有够久 → 无条件卖出


# --------------------------------------------------------------------------- #
# buffer 选择器
# --------------------------------------------------------------------------- #


@dataclass
class BufferConfig:
    entry_pct: float = 0.20      # 新进阈值：排名分位 < entry_pct 才买
    exit_pct: float = 0.20       # 保留阈值：排名分位 < exit_pct 才留（> entry 形成缓冲带）
    min_hold_rebalances: int = 0  # 最短持有期（按调仓次数计，0 = 不限）
    hard_stop_pct: float = HARD_STOP
    target_pct: float = TARGET
    weight_mode: str = "equal"   # equal | score | inv_vol

    @property
    def buffer_width(self) -> float:
        return round(max(0.0, self.exit_pct - self.entry_pct), 4)


@dataclass
class BufferState:
    held_since: dict[str, int] = field(default_factory=dict)   # symbol -> 入场调仓序号
    entry_score: dict[str, float] = field(default_factory=dict)
    last_holdings: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)


def select_with_buffer(
    ranked: Sequence[tuple[str, float]],
    *,
    step: int,
    state: BufferState,
    config: BufferConfig,
    volatilities: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, int]]:
    """按 buffer 规则给出目标权重，并更新持有期状态。

    ``ranked`` 是当日按分数降序的 ``(symbol, score)``；权重和 ≤ 1，其余留现金给
    分批调仓用。返回 ``(权重, 统计)``。
    """
    total = len(ranked)
    if total == 0:
        state.held_since.clear()
        state.last_holdings = []
        return {}, {"entry": 0, "exit": 0, "keep": 0, "n": 0}
    position = {symbol: index for index, (symbol, _score) in enumerate(ranked)}
    score_of = dict(ranked)
    entry_cut = max(1, int(round(total * config.entry_pct)))
    n_target = max(1, int(round(total * config.target_pct)))
    if entry_cut < n_target:
        # entry 比目标持仓还窄时，缓冲带实际上不存在，参数退化成"纯 Top N"。
        entry_cut = n_target
    exit_cut = max(entry_cut, int(round(total * config.exit_pct)))
    hard_cut = max(exit_cut, int(round(total * config.hard_stop_pct)))

    keep: list[str] = []
    exited = 0
    for symbol in list(state.held_since):
        if symbol not in position:
            state.held_since.pop(symbol, None)
            state.entry_score.pop(symbol, None)
            continue
        rank = position[symbol]
        held = step - state.held_since[symbol]
        if rank >= hard_cut and held >= config.min_hold_rebalances:
            state.held_since.pop(symbol, None)
            state.entry_score.pop(symbol, None)
            exited += 1
            continue
        if rank < exit_cut:
            keep.append(symbol)
            continue
        if held < config.min_hold_rebalances and rank < hard_cut:
            keep.append(symbol)          # 最短持有期保护
            continue
        state.held_since.pop(symbol, None)
        state.entry_score.pop(symbol, None)
        exited += 1

    keep.sort(key=lambda symbol: position[symbol])
    if len(keep) > n_target:
        for symbol in keep[n_target:]:
            state.held_since.pop(symbol, None)
            state.entry_score.pop(symbol, None)
            exited += 1
        keep = keep[:n_target]

    picked = list(keep)
    entered = 0
    for symbol, score in ranked[:entry_cut]:
        if len(picked) >= n_target:
            break
        if symbol in state.held_since:
            continue
        picked.append(symbol)
        state.held_since[symbol] = step
        state.entry_score[symbol] = score
        entered += 1

    # 名额没填满就空着，不从缓冲带外硬凑：硬凑等于把 entry 阈值作废，
    # 参数会失去意义（entry < target_pct 时尤其明显）。

    weights = _weights(picked, score_of, config.weight_mode, volatilities)
    state.last_holdings = sorted(picked)
    state.history.append({"step": step, "n": len(picked), "entered": entered, "exited": exited})
    return weights, {"entry": entered, "exit": exited, "keep": len(keep), "n": len(picked)}


def _weights(
    picked: Sequence[str],
    score_of: dict[str, float],
    mode: str,
    volatilities: dict[str, float] | None,
) -> dict[str, float]:
    if not picked:
        return {}
    if mode == "score":
        # 分数加权：把分数压到正区间再归一，避免负分导致负权重
        low = min(score_of.get(symbol, 0.0) for symbol in picked)
        raw = {symbol: max(1e-6, score_of.get(symbol, 0.0) - low + 1e-6) for symbol in picked}
    elif mode == "inv_vol" and volatilities:
        raw = {
            symbol: 1.0 / max(volatilities.get(symbol, 0.0), 0.01) for symbol in picked
        }
    else:
        raw = {symbol: 1.0 for symbol in picked}
    total = sum(raw.values())
    return {symbol: value / total for symbol, value in raw.items()}


# --------------------------------------------------------------------------- #
# 信号慢化
# --------------------------------------------------------------------------- #


class RankSmoother:
    """对"横截面分位"做 EWMA 平滑，而不是对原始分数——分数跨期不可比。"""

    def __init__(self, halflife: float) -> None:
        self.halflife = halflife
        self.state: dict[str, float] = {}

    def update(self, ranked: Sequence[tuple[str, float]]) -> list[tuple[str, float]]:
        total = max(len(ranked), 2)
        current = {
            symbol: 1.0 - index / (total - 1) for index, (symbol, _score) in enumerate(ranked)
        }
        if self.halflife <= 0:
            return [(symbol, current[symbol]) for symbol, _score in ranked]
        decay = 0.5 ** (protocol_rebalances(self.halflife))
        smoothed = {}
        for symbol, value in current.items():
            previous = self.state.get(symbol)
            smoothed[symbol] = value if previous is None else previous * decay + value * (1 - decay)
        self.state = smoothed
        ordered = sorted(smoothed.items(), key=lambda pair: -pair[1])
        return ordered


def protocol_rebalances(halflife_days: float, rebalance_days: int = 5) -> float:
    """把"日"半衰期换算成"调仓次数"半衰期。"""
    return max(1e-6, halflife_days / max(1, rebalance_days))


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #


def daily_returns(nav: Sequence[dict[str, Any]]) -> list[float]:
    values = [point["nav"] for point in nav]
    return [
        values[i] / values[i - 1] - 1.0
        for i in range(1, len(values))
        if values[i - 1]
    ]


def net_sharpe(nav: Sequence[dict[str, Any]], *, risk_free: float = 0.0) -> float | None:
    returns = daily_returns(nav)
    if len(returns) < 30:
        return None
    mean = statistics.fmean(returns) - risk_free / 252
    sd = statistics.pstdev(returns)
    return mean / sd * math.sqrt(252) if sd > 0 else None


def annual_turnover(rebalances: Sequence[dict[str, Any]], *, years: float) -> float:
    """年化单边换手（%）。"""
    if years <= 0 or not rebalances:
        return 0.0
    return sum(item["one_way_turnover"] for item in rebalances) / years * 100


def cost_drag_bps(rebalances: Sequence[dict[str, Any]], *, years: float) -> float:
    """年化成本拖累（bp）。直接累加每次调仓的 ``cost_bps``（已经是占当时净值的
    比例），不除以平均净值——净值翻倍后除平均值会把成本系统性低估一半。"""
    if years <= 0 or not rebalances:
        return 0.0
    return sum(item.get("cost_bps", 0.0) for item in rebalances) / years


def cap_trigger_rate(rebalances: Sequence[dict[str, Any]]) -> float:
    if not rebalances:
        return 0.0
    return sum(1 for item in rebalances if item.get("capped")) / len(rebalances) * 100


def jaccard(holdings: Sequence[Sequence[str]]) -> float:
    """相邻两期持仓的 Jaccard 相似度（1 = 完全没变）。"""
    values = []
    for previous, current in zip(holdings, holdings[1:]):
        left, right = set(previous), set(current)
        if not left and not right:
            continue
        values.append(len(left & right) / len(left | right))
    return statistics.fmean(values) if values else 0.0


def summary(result: dict[str, Any], *, years: float) -> dict[str, Any]:
    nav = result["nav"]
    holdings = [item["symbols"] for item in result.get("holdings", [])]
    return {
        "total_return": round((nav[-1]["nav"] - 1) * 100, 2) if nav else None,
        "annual_return": round(((nav[-1]["nav"]) ** (1 / years) - 1) * 100, 2)
        if nav and years > 0 and nav[-1]["nav"] > 0 else None,
        "max_drawdown": _max_drawdown(nav),
        "sharpe": round(net_sharpe(nav), 3) if net_sharpe(nav) is not None else None,
        "annual_turnover": round(annual_turnover(result["rebalances"], years=years), 1),
        "cost_drag_bps": round(cost_drag_bps(result["rebalances"], years=years), 1),
        "cap_trigger_rate": round(cap_trigger_rate(result["rebalances"]), 1),
        "rebalances": len(result["rebalances"]),
        "jaccard": round(jaccard(holdings), 3) if holdings else None,
        "mean_holdings": round(statistics.fmean([len(item) for item in holdings]), 1) if holdings else None,
        "cash_mean": round(result.get("cash_ratio_mean", 0.0) * 100, 1),
    }


def _max_drawdown(nav: Sequence[dict[str, Any]]) -> float | None:
    if not nav:
        return None
    peak = nav[0]["nav"]
    worst = 0.0
    for point in nav:
        peak = max(peak, point["nav"])
        if peak > 0:
            worst = min(worst, point["nav"] / peak - 1)
    return round(worst * 100, 2)


# --------------------------------------------------------------------------- #
# 容量
# --------------------------------------------------------------------------- #


def capacity(
    trades: Sequence[dict[str, Any]],
    adv: dict[str, float],
    *,
    participation: float = 0.10,
) -> dict[str, Any]:
    """给定参与率上限，反推能承载的规模。

    每笔交易的金额是"占组合净值的比例"，所以 ``AUM_max = 参与率 × ADV / 占比``；
    取所有交易的下分位，得到"几乎所有交易都能在限制内完成"的规模。
    """
    limits = []
    for trade in trades:
        symbol_adv = adv.get(trade["symbol"])
        if not symbol_adv or trade["fraction"] <= 0:
            continue
        limits.append(participation * symbol_adv / trade["fraction"])
    if not limits:
        return {"available": False}
    limits.sort()
    return {
        "available": True,
        "participation_limit": participation,
        "aum_p10": round(limits[len(limits) // 10] / 1e8, 2),
        "aum_p50": round(limits[len(limits) // 2] / 1e8, 2),
        "aum_p90": round(limits[len(limits) * 9 // 10] / 1e8, 2),
        "aum_binding": round(limits[0] / 1e8, 2),
        "n_trades": len(limits),
    }


def impact_cost_bp(participation: float, *, k: float = 10.0) -> float:
    """平方根冲击模型：``k * sqrt(参与率%)``，参与率 1% 时约 10bp。

    这是个**待校准的假设**，不是实测值；真实冲击要用逐笔成交或券商成交数据
    拟合。它只用来回答"规模上去以后成本怎么变"，不参与主线结论。
    """
    if participation <= 0:
        return 0.0
    return k * math.sqrt(participation * 100)


def adv_by_symbol(
    bars: dict[str, dict[str, dict[str, float]]],
    date: str,
    *,
    window: int = 20,
) -> dict[str, float]:
    """截至 date 的 20 日平均成交额（只用过去，不看未来）。"""
    result: dict[str, float] = {}
    for symbol, series in bars.items():
        dates = [day for day in sorted(series) if day <= date][-window:]
        amounts = [series[day].get("amount", 0.0) for day in dates]
        if amounts:
            result[symbol] = statistics.fmean(amounts)
    return result


# --------------------------------------------------------------------------- #
# 买持对照：区分"选股"与"再平衡"
# --------------------------------------------------------------------------- #


def buy_and_hold(
    symbols: Sequence[str],
    bars: dict[str, dict[str, dict[str, float]]],
    calendar: Sequence[str],
    *,
    entry_index: int = 1,
    cost: float = 0.0,
) -> dict[str, Any]:
    """把首期持仓原样拿到期末：两者之差就是"再平衡效应"，不是选股 alpha。"""
    entry_date = calendar[entry_index]
    weights = {symbol: 1.0 / len(symbols) for symbol in symbols if symbol in bars} if symbols else {}
    if not weights:
        return {"nav": [], "total_return": None}
    entry_prices = {
        symbol: bars[symbol][entry_date]["open"]
        for symbol in weights
        if entry_date in bars[symbol] and bars[symbol][entry_date]["open"] > 0
    }
    if not entry_prices:
        return {"nav": [], "total_return": None}
    shares = {symbol: weights[symbol] * (1 - cost) / entry_prices[symbol] for symbol in entry_prices}
    nav = []
    for date in calendar[entry_index:]:
        value = 0.0
        for symbol, quantity in shares.items():
            price = bars.get(symbol, {}).get(date, {}).get("close")
            if price:
                value += quantity * price
        nav.append({"date": date, "nav": round(value, 6)})
    return {"nav": nav, "total_return": round((nav[-1]["nav"] - 1) * 100, 2) if nav else None}


# --------------------------------------------------------------------------- #
# IC：buffer 前后
# --------------------------------------------------------------------------- #


def cross_sectional_ic(
    pairs_by_date: dict[str, list[tuple[float, float]]],
) -> list[float]:
    """逐期 Spearman IC 序列（输入已是 (信号, 未来收益) 对）。"""
    import evalkit as ek

    series = []
    for date in sorted(pairs_by_date):
        pairs = pairs_by_date[date]
        if len(pairs) < 10:
            continue
        ic = ek.spearman([p[0] for p in pairs], [p[1] for p in pairs])
        if ic is not None:
            series.append(ic)
    return series


def sticky_scores(
    ranked: Sequence[tuple[str, float]],
    state: BufferState,
    *,
    step: int,
) -> list[tuple[str, float]]:
    """buffer 生效后的"有效评分"：持仓用入场时的评分，其余用当前评分。

    这是 buffer 的真实代价——被冻结的持仓不再享受最新的排名信息。用它算 IC，
    掉多少就说明"省下的成本里有多少是拿 alpha 换的"。
    """
    effective = []
    for symbol, score in ranked:
        if symbol in state.held_since:
            effective.append((symbol, state.entry_score.get(symbol, score)))
        else:
            effective.append((symbol, score))
    return effective
