"""T+1 执行引擎：收盘出信号，次日开盘成交，并过滤实际不可成交的报价。

为什么不用 ``xsec.simulate``
---------------------------
``xsec.simulate`` 按当日收盘价成交，等于假设"收盘那一刻同时知道信号又完成
买卖"，对 A 股偏乐观两层：

1. **时点**：信号用的是当日收盘数据，只能次日开盘下单；
2. **可成交性**：涨停开盘买不进、跌停开盘卖不出、停牌当天根本不能交易。

本模块只换执行层，不动信号与成本口径：同样的单边费率、同样的换手上限、
同样的先卖后买记账。返回结构与 ``xsec.simulate`` 兼容，因此指标函数可以
直接复用，报告里不会出现两套净值口径。
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Callable, Iterable, Sequence

from market_data import is_bse

# A 股涨跌幅限制（按板块）。ST 单独 5%，由证券名称识别。
LIMIT_MAIN = 0.10
LIMIT_STAR = 0.20
LIMIT_BSE = 0.30
LIMIT_ST = 0.05
EPSILON = 0.005          # 价格比较容差（半个分位）
UNAVAILABLE = ("suspended", "limit_up", "limit_down", "no_bar")


def limit_ratio(symbol: str, name: str = "") -> float:
    """该标的当日的涨跌幅限制比例。"""
    if name and "ST" in name.upper():
        return LIMIT_ST
    if symbol.startswith(("300", "301", "302", "688", "689")):
        return LIMIT_STAR
    if is_bse(symbol):
        return LIMIT_BSE
    return LIMIT_MAIN


def _round_price(value: float) -> float:
    """涨跌停价按四舍五入到分。"""
    return math.floor(value * 100.0 + 0.5) / 100.0


def limit_prices(prev_close: float, ratio: float) -> tuple[float, float]:
    if not prev_close or prev_close <= 0:
        return (0.0, 0.0)
    return (_round_price(prev_close * (1.0 + ratio)), _round_price(prev_close * (1.0 - ratio)))


def build_bars(histories: Iterable[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    """``{symbol: {date: bar}}``，bar 保留开高低收与成交额/换手。"""
    result: dict[str, dict[str, dict[str, float]]] = {}
    for history in histories:
        series: dict[str, dict[str, float]] = {}
        for row in history["rows"]:
            series[row["date"]] = {
                "open": float(row["open"]),
                "close": float(row["close"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "volume": float(row.get("volume") or 0.0),
                "amount": float(row.get("amount") or 0.0),
                "turnover": float(row.get("turnover") or 0.0),
            }
        result[history["symbol"]] = series
    return result


def tradability(
    histories: Iterable[dict[str, Any]],
    *,
    strict_open: bool = True,
) -> dict[str, dict[str, str]]:
    """逐标的逐日标注开盘能否买入 / 卖出。

    返回 ``{symbol: {date: "" | "suspended" | "limit_up" | "limit_down"}}``。
    空字符串表示正常交易。判据都用**开盘价**，因为成交就发生在开盘：

    - 停牌：当日无成交（volume 为 0）；
    - 涨停：开盘价已触及涨停价 → 开盘买单排不进（``strict_open`` 时同时要求
      最高价 = 涨停价，避免把"高开未封板"误判成买不进）；
    - 跌停：开盘价触及跌停价 → 卖单排不出。
    """
    result: dict[str, dict[str, str]] = {}
    for history in histories:
        symbol = history["symbol"]
        name = history.get("name") or ""
        ratio = limit_ratio(symbol, name)
        rows = history["rows"]
        flags: dict[str, str] = {}
        for index, row in enumerate(rows):
            previous = rows[index - 1]["close"] if index else row["close"] - row.get("change", 0.0)
            up, down = limit_prices(float(previous), ratio)
            volume = float(row.get("volume") or 0.0)
            open_price = float(row["open"])
            high, low = float(row["high"]), float(row["low"])
            if volume <= 0:
                flags[row["date"]] = "suspended"
            elif up and open_price >= up - EPSILON and (not strict_open or high >= up - EPSILON and low >= up - EPSILON):
                flags[row["date"]] = "limit_up"
            elif down and open_price <= down + EPSILON and (not strict_open or high <= down + EPSILON and low <= down + EPSILON):
                flags[row["date"]] = "limit_down"
            else:
                flags[row["date"]] = ""
        result[symbol] = flags
    return result


def _board_of(symbol: str, name: str = "") -> str:
    if name and "ST" in name.upper():
        return "ST"
    if symbol.startswith(("300", "301", "302", "688", "689")):
        return "主板/双创（20%）"
    if is_bse(symbol):
        return "北交所（30%）"
    return "主板（10%）"


def simulate_t1(
    calendar: Sequence[str],
    bars: dict[str, dict[str, dict[str, float]]],
    pick: Callable[[str, list[str]], list[str]] | None = None,
    *,
    target: Callable[[str, list[str]], dict[str, float]] | None = None,
    tranche: float = 1.0,
    cost_fn: Callable[[str, str, float, str], float] | None = None,
    flags: dict[str, dict[str, str]] | None = None,
    rebalance_every: int = 5,
    cost_buy: float = 0.00076,
    cost_sell: float = 0.00126,
    max_turnover: float = 0.4,
    enforce_t1_sell: bool = True,
    exposure: Callable[[str], float] | None = None,
) -> dict[str, Any]:
    """T+1 开盘执行的回测。

    流程：``calendar[i]`` 收盘算出目标持仓 → ``calendar[i+1]`` 开盘成交。
    ``pick(date, available)`` 只拿得到收盘时可见的信息（可用标的 = 当日有行情
    且价格已知的标的）。

    记账约定与 ``xsec.simulate`` 一致：先卖后买、单边换手上限按组合净值折算、
    未成交仓位继续持有。额外的约束：

    - 涨停 / 停牌：当日不买；
    - 跌停 / 停牌：当日不卖（下一调仓日再试）；
    - ``enforce_t1_sell``：当日买入的份额当日不可卖（本函数里买入发生在开盘，
      卖出也在同一时点，因此只允许卖出"开盘前已持有"的份额）。
    - ``exposure``：可选的目标总仓位函数（regime 目标用），返回 (0, 1]，缺省满仓。
    - ``target``：可选的目标权重函数（buffer / 分批调仓用），返回 ``{symbol: 权重}``，
      权重和 ≤ 1，余额留现金。与 ``pick`` 二选一。
    - ``tranche``：部分调整系数 (0, 1]。1 = 一次到位；0.5 = 每次只走一半差距，
      等价于把一次调仓摊到多个调仓日。``target`` 与 ``pick`` 都适用。
    - ``cost_fn(symbol, date, 换手比例, side)``：按笔计费，返回单边费率。缺省用
      固定的 ``cost_buy`` / ``cost_sell``；容量测算用它注入冲击成本。
    """
    cash = 1.0
    shares: dict[str, float] = {}
    # 每笔持仓的买入日索引，用于 T+1 卖出限制
    bought_on: dict[str, int] = {}
    last_close: dict[str, float] = {}
    pending: list[str] | dict[str, float] | None = None
    pending_signal_date: str | None = None
    nav_series: list[dict[str, Any]] = []
    rebalances: list[dict[str, Any]] = []
    holdings: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    total_cost = 0.0
    total_turnover = 0.0
    cash_ratios: list[float] = []
    blocked = {"buy_limit_up": 0, "buy_suspended": 0, "sell_limit_down": 0, "sell_suspended": 0}
    last_rebalance = -10 ** 9

    for index, date in enumerate(calendar):
        # 1) 开盘执行上一个交易日收盘发出的目标
        executed_today: set[str] = set()
        if pending is not None:
            open_prices = {
                symbol: series[date]["open"]
                for symbol, series in bars.items()
                if date in series and series[date]["open"] > 0
            }
            def open_value() -> float:
                return cash + sum(
                    shares.get(symbol, 0.0) * open_prices.get(symbol, last_close.get(symbol, 0.0))
                    for symbol in shares
                )

            nav_before = open_value()
            gross = exposure(date) if exposure else 1.0
            gross = max(0.0, min(1.0, gross))
            if isinstance(pending, dict):
                # 权重目标：按权重折算金额，再乘总仓位（regime / 波动目标），其余留现金
                desired = {
                    symbol: weight * gross * nav_before
                    for symbol, weight in pending.items()
                    if symbol in open_prices and weight > 0
                }
            else:
                names = [symbol for symbol in pending if symbol in open_prices]
                weight = gross / len(names) if names else 0.0
                desired = {symbol: weight * nav_before for symbol in names}
            order = sorted(set(desired) | set(shares))
            current_value = {
                symbol: shares.get(symbol, 0.0) * open_prices.get(symbol, last_close.get(symbol, 0.0))
                for symbol in order
            }
            if 0 < tranche < 1:
                # 部分调整：只走完目标与现状差距的一部分，剩下的留到下次调仓
                desired = {
                    symbol: current_value[symbol]
                    + (desired.get(symbol, 0.0) - current_value[symbol]) * tranche
                    for symbol in order
                }
            delta = {
                symbol: desired.get(symbol, 0.0) - current_value[symbol] for symbol in order
            }
            buy_notional = sum(max(0.0, move) for move in delta.values())
            sell_notional = sum(max(0.0, -move) for move in delta.values())
            cap = max_turnover * nav_before
            scale = 1.0
            if max(buy_notional, sell_notional) > cap and max(buy_notional, sell_notional) > 0:
                scale = cap / max(buy_notional, sell_notional)

            sold = bought = 0.0
            cost_here = 0.0
            for symbol, move in delta.items():
                if move >= 0:
                    continue
                price = open_prices.get(symbol)
                if price is None:
                    continue
                if enforce_t1_sell and bought_on.get(symbol, -1) == index:
                    continue
                state = (flags or {}).get(symbol, {}).get(date, "")
                if state in ("limit_down", "suspended", "no_bar"):
                    blocked["sell_limit_down" if state == "limit_down" else "sell_suspended"] += 1
                    continue
                quantity = min(shares.get(symbol, 0.0), -scale * move / price)
                if quantity <= 0:
                    continue
                notional = quantity * price
                shares[symbol] = shares.get(symbol, 0.0) - quantity
                if shares[symbol] <= 1e-12:
                    shares.pop(symbol, None)
                    bought_on.pop(symbol, None)
                rate = cost_fn(symbol, date, notional / nav_before, "sell") if cost_fn else cost_sell
                cash += notional * (1 - rate)
                total_cost += notional * rate
                cost_here += notional * rate
                sold += notional
                trades.append({"date": date, "symbol": symbol, "side": "sell",
                               "fraction": round(notional / nav_before, 6), "rate": rate})
            for symbol, move in delta.items():
                if move <= 0:
                    continue
                price = open_prices.get(symbol)
                if price is None:
                    continue
                state = (flags or {}).get(symbol, {}).get(date, "")
                if state == "limit_up":
                    blocked["buy_limit_up"] += 1
                    continue
                if state in ("suspended", "no_bar"):
                    blocked["buy_suspended"] += 1
                    continue
                budget = min(scale * move, cash / (1 + cost_buy))
                if budget <= 1e-12:
                    continue
                quantity = budget / price
                if shares.get(symbol, 0.0) <= 1e-12:
                    bought_on[symbol] = index
                rate = cost_fn(symbol, date, budget / nav_before, "buy") if cost_fn else cost_buy
                shares[symbol] = shares.get(symbol, 0.0) + quantity
                cash -= budget * (1 + rate)
                total_cost += budget * rate
                cost_here += budget * rate
                bought += budget
                executed_today.add(symbol)
                trades.append({"date": date, "symbol": symbol, "side": "buy",
                               "fraction": round(budget / nav_before, 6), "rate": rate})
            one_way = 0.5 * (sold + bought) / nav_before if nav_before else 0.0
            total_turnover += one_way
            rebalances.append(
                {
                    "signal_date": pending_signal_date,
                    "date": date,
                    "one_way_turnover": round(one_way, 4),
                    "capped": scale < 1.0,
                    "cost_bps": round(cost_here / nav_before * 10000, 3) if nav_before else 0.0,
                    "traded": round((sold + bought) / nav_before, 6) if nav_before else 0.0,
                    "nav": round(cash + sum(shares.get(s, 0.0) * open_prices.get(s, 0.0) for s in shares), 4),
                    "gross": round(gross, 3),
                }
            )
            holdings.append(
                {
                    "signal_date": pending_signal_date,
                    "date": date,
                    "symbols": sorted(symbol for symbol, qty in shares.items() if qty > 1e-12),
                    # 权重快照：容量测算与买持对照都要用
                    "weights": {
                        symbol: qty * open_prices.get(symbol, last_close.get(symbol, 0.0)) / nav_before
                        for symbol, qty in shares.items()
                        if qty > 1e-12 and nav_before
                    },
                }
            )
            pending = None
            pending_signal_date = None

        # 2) 收盘估值
        for symbol, series in bars.items():
            if date in series and series[date]["close"] > 0:
                last_close[symbol] = series[date]["close"]
        total = cash + sum(shares.get(symbol, 0.0) * last_close.get(symbol, 0.0) for symbol in shares)
        nav_series.append({"date": date, "nav": round(total, 4)})
        if total:
            cash_ratios.append(cash / total)

        # 3) 收盘出信号，次日开盘执行
        if index - last_rebalance >= rebalance_every:
            available = [symbol for symbol in bars if date in bars[symbol] and bars[symbol][date]["close"] > 0]
            pending = dict(target(date, available)) if target is not None else list(pick(date, available))
            pending_signal_date = date
            last_rebalance = index

    return {
        "nav": nav_series,
        "rebalances": rebalances,
        "holdings": holdings,
        "trades": trades,
        "total_cost": total_cost,
        "total_turnover": total_turnover,
        "cash_ratio_mean": statistics.fmean(cash_ratios) if cash_ratios else 0.0,
        "final_nav": nav_series[-1]["nav"] if nav_series else 1.0,
        "blocked": blocked,
        "execution": {
            "fill": "next_open",
            "t1_sell": bool(enforce_t1_sell),
            "limit_filter": True,
        },
    }


def execution_summary(flags: dict[str, dict[str, str]]) -> dict[str, Any]:
    """执行层口径的可观测面：涨跌停 / 停牌各占多少标的日。"""
    counts: dict[str, int] = {}
    total = 0
    boards: dict[str, int] = {}
    for symbol, series in flags.items():
        for state in series.values():
            total += 1
            if state:
                counts[state] = counts.get(state, 0) + 1
    return {
        "symbol_days": total,
        "blocked_symbol_days": counts,
        "blocked_ratio": round(sum(counts.values()) / total * 100, 3) if total else 0.0,
    }
