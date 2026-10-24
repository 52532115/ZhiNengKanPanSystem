"""跨资产 ETF 时间序列配置：TSMOM + 双均线 + 波动目标 + 回撤止损 + ERC。

为什么这条线比个股选股更早能交付
--------------------------------
个股模型受困于"横截面只有 3.6 个有效 bet"：55 只股票的日频横截面 IC 的标准误约
0.02–0.03，要检出 0.03 的 IC 需要 500+ 个独立周期（约 10 年）。而跨资产配置的
标的只有十几个，**但每个标的自己就是一条独立的时间序列**，样本是"资产 × 时间"，
不需要横截面离散度来提供统计功效；评价口径也从"IC"换成"组合净值/回撤"——
这恰好是资金真正关心的量。

规则（全部预先写死，不做参数搜索）
----------------------------------
1. **入选**（收盘 T 日判定，四条全满足）：
   - 12–1 时序动量 > 0：``close[T-21] / close[T-252] - 1 > 0``
   - 收盘 > MA60 且 收盘 > MA120（双均线确认）
   - 距 252 日高点的回撤 > −15%（个股级别的止损，未触发）
2. **退出**（任一满足即出）：动量转负、收盘跌破 MA120、或回撤触 −15%。
   注意退出只看慢线（MA120），MA60 只用于入场——这是刻意的不对称：入场要确认，
   离场要容忍震荡。
3. **权重**：入选资产做等风险贡献（ERC），协方差用 60 日样本；单腿上限 30%，
   A 股合计 ≤ 50%、海外合计 ≤ 50%。
4. **波动目标**：组合事前波动 > 10% 时按比例缩到 10%，只降不升（ETF 不加杠杆）。
5. **现金**：未分配部分买货币 ETF（511880），按实际收益计息，不假设 0 收益。
6. **再平衡**：每月（21 个交易日）；单腿变化 < 3pp 不动（no-trade band），
   单次单边换手上限 30%。
7. **成本**：ETF 无印花税，单边 7.5bp（佣金 2.5 + 价差/滑点 5）。
8. **成交**：收盘出信号、次日开盘成交（``execution.simulate_t1``）。
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import execution as ex

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache" / "fields" / "etf"
REPORT = ROOT / "reports" / "etf"

CASH = "511880"

# 12 条腿 + 现金。class 用于约束与归因；note 写清入选理由与替代标的。
UNIVERSE: dict[str, dict[str, str]] = {
    "510300": {"name": "沪深300ETF", "class": "A股大盘", "bucket": "a_share",
               "note": "A 股核心 beta，流动性最好；可换 510310/159919"},
    "510500": {"name": "中证500ETF", "class": "A股中盘", "bucket": "a_share",
               "note": "中盘风格，与 300 相关性约 0.85；可换 512500"},
    "159915": {"name": "创业板ETF", "class": "A股成长", "bucket": "a_share",
               "note": "高 beta 成长，2015 起 3000 根；可换 159952/159949"},
    "512100": {"name": "中证1000ETF", "class": "A股小盘", "bucket": "a_share",
               "note": "2016-11 上市，小盘暴露；可换 159845"},
    "588000": {"name": "科创50ETF", "class": "A股硬科技", "bucket": "a_share",
               "note": "2020-11 上市，样本最短，权重会被 ERC 自动压低"},
    "511010": {"name": "国债ETF", "class": "利率债", "bucket": "defensive",
               "note": "5 年国债，与 A 股负相关为主；可换 511260/511020"},
    "518880": {"name": "黄金ETF", "class": "黄金", "bucket": "defensive",
               "note": "实物黄金，危机对冲；可换 159934"},
    "510170": {"name": "大宗商品ETF", "class": "商品（股票代理）", "bucket": "commodity",
               "note": "跟踪上证大宗商品股票指数，是商品**股**不是期货，与 A 股相关偏高；"
                       "真商品工具可换 159980（有色期货）/159985（豆粕），但 2019 才有数据"},
    "513100": {"name": "纳指ETF", "class": "海外·美股科技", "bucket": "overseas",
               "note": "美元资产 + 科技 beta；可换 513300"},
    "513500": {"name": "标普500ETF", "class": "海外·美股宽基", "bucket": "overseas",
               "note": "与纳指相关 0.8，冗余度较高，ERC 会自动降权"},
    "159920": {"name": "恒生ETF", "class": "海外·中国香港", "bucket": "overseas",
               "note": "港股，与 A 股相关性 0.6；可换 159742"},
    "513520": {"name": "日经ETF", "class": "海外·日本", "bucket": "overseas",
               "note": "2019-06 上市；可换 513000/513880"},
}
LIVE_LEGS = tuple(UNIVERSE)

BUY_COST = 0.00075      # 7.5bp：佣金 2.5 + 价差/滑点 5（ETF 免印花税）
SELL_COST = 0.00075

# 规则版本：任何阈值/权重法/成本口径的改动都必须升版本，纸面跟踪与回测才可比。
RULE_VERSION = "etf-tsmom-v1"
LATEST_PATH = REPORT / "latest.json"
PAPER_PATH = REPORT / "paper.jsonl"
DISCLAIMER = "研究结果，不是投资建议。"
STALE_TOLERANCE = 15    # 末端日期落后全样本超过 N 个交易日 → 判为停牌/清盘


@dataclass(frozen=True)
class Config:
    mom_lookback: int = 252
    mom_skip: int = 21
    ma_fast: int = 60
    ma_slow: int = 120
    stop_drawdown: float = -0.15
    stop_window: int = 252
    target_vol: float = 0.10
    vol_window: int = 60
    rebalance: int = 21
    max_weight: float = 0.30
    bucket_caps: tuple[tuple[str, float], ...] = (("a_share", 0.50), ("overseas", 0.50))
    no_trade_band: float = 0.03
    max_turnover: float = 0.30
    weighting: str = "erc"          # erc | inv_vol
    entry_requires_slow: bool = True   # 入场是否也要求站上 MA120
    use_stop: bool = True
    use_vol_target: bool = True


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #


def load_panel(symbols: Sequence[str] = tuple(UNIVERSE) + (CASH,)) -> dict[str, dict[str, Any]]:
    panel: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        path = CACHE / f"{symbol}.json"
        if not path.is_file():
            raise FileNotFoundError(f"缺少 {symbol} 的缓存（{path}）")
        with path.open("r", encoding="utf-8") as handle:
            panel[symbol] = json.load(handle)
    return panel


def align(panel: dict[str, dict[str, Any]], calendar: Sequence[str]) -> dict[str, list[float | None]]:
    """把收盘价对齐到统一日历；**上市前填 None**（不是 0、也不是首日价）。

    填首日价会让"未上市"变成"零波动"，逆波动权重会把它推到上限；填 0 会污染收益。
    这里显式保留 None，由信号层判断"当日是否可用"。
    """
    aligned: dict[str, list[float | None]] = {}
    for symbol, data in panel.items():
        lookup = {row["date"]: float(row["close"]) for row in data["rows"] if row["close"]}
        series: list[float | None] = []
        last: float | None = None
        for date in calendar:
            price = lookup.get(date)
            if price is not None:
                last = price
            series.append(last)
        aligned[symbol] = series
    return aligned


def returns_of(closes: Sequence[float | None]) -> list[float]:
    out = [0.0]
    for index in range(1, len(closes)):
        previous, current = closes[index - 1], closes[index]
        out.append(current / previous - 1 if previous and current else 0.0)
    return out


# --------------------------------------------------------------------------- #
# 信号
# --------------------------------------------------------------------------- #


def sma(values: Sequence[float | None], window: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    running = 0.0
    count = 0
    for index, value in enumerate(values):
        if value is not None:
            running += value
            count += 1
        if index >= window:
            dropped = values[index - window]
            if dropped is not None:
                running -= dropped
                count -= 1
        if index >= window - 1 and count == window:
            out[index] = running / window
    return out


def momentum(closes: Sequence[float | None], index: int, *, lookback: int = 252, skip: int = 21) -> float | None:
    if index < lookback or index - skip < 0:
        return None
    start, end = closes[index - lookback], closes[index - skip]
    if not start or not end:
        return None
    return end / start - 1.0


def drawdown_from_high(closes: Sequence[float | None], index: int, window: int = 252) -> float | None:
    if index < 5:
        return None
    sample = [value for value in closes[max(0, index - window + 1): index + 1] if value]
    if len(sample) < 20:
        return None
    peak = max(sample)
    return closes[index] / peak - 1 if peak and closes[index] else None


def eligibility(
    closes: Sequence[float | None],
    index: int,
    *,
    config: Config,
    ma_fast: Sequence[float | None],
    ma_slow: Sequence[float | None],
    holding: bool,
) -> tuple[bool, dict[str, Any]]:
    """返回（是否合格，诊断信息）。

    ``holding=True`` 时用"持有中"的退出规则（只看慢线与动量），否则用入场规则
    （四条全要满足）。这种不对称能把"入场晚"换成"离场稳"，是降低换手的主要来源。
    """
    price = closes[index]
    if not price:
        return False, {"reason": "no_price"}
    mom = momentum(closes, index, lookback=config.mom_lookback, skip=config.mom_skip)
    fast, slow = ma_fast[index], ma_slow[index]
    drawdown = drawdown_from_high(closes, index, config.stop_window)
    details = {"mom": mom, "above_fast": price > fast if fast else None,
               "above_slow": price > slow if slow else None, "drawdown": drawdown}
    if mom is None or fast is None or slow is None:
        return False, {**details, "reason": "warmup"}
    if config.use_stop and drawdown is not None and drawdown <= config.stop_drawdown:
        return False, {**details, "reason": "stop"}
    if holding:
        if mom < 0:
            return False, {**details, "reason": "mom_negative"}
        if price < slow:
            return False, {**details, "reason": "below_ma_slow"}
        return True, {**details, "reason": "hold"}
    if mom > 0 and price > fast and (price > slow or not config.entry_requires_slow):
        return True, {**details, "reason": "enter"}
    return False, {**details, "reason": "entry_filter"}


# --------------------------------------------------------------------------- #
# 权重：ERC 与逆波动
# --------------------------------------------------------------------------- #


def _covariance(
    returns: dict[str, list[float]],
    symbols: Sequence[str],
    index: int,
    window: int,
) -> list[list[float]]:
    series = {symbol: returns[symbol][index - window + 1: index + 1] for symbol in symbols}
    size = len(symbols)
    matrix = [[0.0] * size for _ in range(size)]
    for i, left in enumerate(symbols):
        for j in range(i, size):
            right = symbols[j]
            a, b = series[left], series[right]
            mean_a, mean_b = statistics.fmean(a), statistics.fmean(b)
            cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b)) / max(len(a) - 1, 1)
            matrix[i][j] = matrix[j][i] = cov * 252
    # 数值稳定：对角线加一点抖动量，避免单资产/共线时奇异
    for i in range(size):
        matrix[i][i] = max(matrix[i][i], 1e-8) + 1e-8
    return matrix


def erc_weights(cov: Sequence[Sequence[float]], *, iterations: int = 200, tol: float = 1e-10) -> list[float]:
    """等风险贡献（long-only）：不动点迭代 ``w_i ← w_i·(1/RC_i)`` 后归一。

    ERC 的解析解只在两资产时存在；多资产用这个迭代（收敛到所有 RC 相等）。
    相比逆波动，它会把相关性计入——两个高度相关的资产不会被同时给到高权重，
    这是跨资产池最需要的一步（纳指与标普、300 与 500 都高度相关）。
    """
    size = len(cov)
    if size == 0:
        return []
    volatility = [math.sqrt(max(cov[i][i], 1e-12)) for i in range(size)]
    weights = [1.0 / v for v in volatility]
    total = sum(weights)
    weights = [w / total for w in weights]
    for _ in range(iterations):
        marginal = [sum(cov[i][j] * weights[j] for j in range(size)) for i in range(size)]
        contribution = [weights[i] * marginal[i] for i in range(size)]
        total_risk = sum(contribution)
        if total_risk <= 1e-14:
            break
        target = total_risk / size
        updated = [
            weights[i] * ((target / contribution[i]) ** 0.5 if contribution[i] > 1e-14 else 1.0)
            for i in range(size)
        ]
        total = sum(updated)
        updated = [w / total for w in updated]
        if max(abs(updated[i] - weights[i]) for i in range(size)) < tol:
            weights = updated
            break
        weights = updated
    return weights


def inverse_vol_weights(cov: Sequence[Sequence[float]]) -> list[float]:
    inverse = [1.0 / math.sqrt(max(cov[i][i], 1e-12)) for i in range(len(cov))]
    total = sum(inverse)
    return [value / total for value in inverse] if total else []


def portfolio_vol(weights: Sequence[float], cov: Sequence[Sequence[float]]) -> float:
    total = sum(
        weights[i] * weights[j] * cov[i][j] for i in range(len(weights)) for j in range(len(weights))
    )
    return math.sqrt(max(0.0, total))


def apply_limits(
    weights: dict[str, float],
    *,
    config: Config,
    max_weight: float,
) -> dict[str, float]:
    """单腿上限 + 资产类别上限，超限部分按比例砍掉（不转移给别的腿，直接进现金）。"""
    out = dict(weights)
    for _ in range(8):
        changed = False
        for symbol, value in list(out.items()):
            if value > max_weight + 1e-12:
                out[symbol] = max_weight
                changed = True
        for bucket, cap in config.bucket_caps:
            members = [s for s in out if UNIVERSE[s]["bucket"] == bucket]
            total = sum(out[s] for s in members)
            if total > cap + 1e-12 and total > 0:
                scale = cap / total
                for symbol in members:
                    out[symbol] *= scale
                changed = True
        if not changed:
            break
    return {symbol: value for symbol, value in out.items() if value > 1e-6}


# --------------------------------------------------------------------------- #
# 目标权重
# --------------------------------------------------------------------------- #


def build_target(
    panel: dict[str, dict[str, Any]],
    calendar: Sequence[str],
    *,
    config: Config | None = None,
    start: str | None = None,
    stale: dict[str, str] | None = None,
) -> Callable[[str, list[str]], dict[str, float]]:
    """把信号层拼成 ``target(date, available) -> {symbol: weight}``。

    状态（``holding`` 与 ``last_weights``）在闭包里维护：入场用四条规则、持有只用
    两条，因此必须先知道"现在有没有这个腿"。这也是 no-trade band 能生效的前提。
    """
    cfg = config or Config()
    # 停牌/清盘的腿必须显式剔除：``align`` 会前向填充价格，留下的是一条"零波动"
    # 直线，动量恰好为 0 虽然进不了池，但它会污染协方差矩阵（方差≈0 → ERC 把它
    # 推到上限），所以不能靠"动量不为正"顺手挡掉。
    dead = set(stale or {})
    closes = align(panel, calendar)
    index_of = {date: i for i, date in enumerate(calendar)}
    returns = {symbol: returns_of(series) for symbol, series in closes.items()}
    ma_fast = {symbol: sma(series, cfg.ma_fast) for symbol, series in closes.items()}
    ma_slow = {symbol: sma(series, cfg.ma_slow) for symbol, series in closes.items()}
    state: dict[str, Any] = {"holding": set(), "last": {}, "dates": []}

    def target(date: str, available: list[str]) -> dict[str, float]:
        index = index_of.get(date)
        if index is None:
            return {}
        if start and date < start:
            return {}
        eligible: dict[str, float] = {}
        diagnostics: dict[str, Any] = {}
        for symbol in LIVE_LEGS:
            if symbol not in closes or symbol not in available or symbol in dead:
                continue
            holding = symbol in state["holding"]
            ok, info = eligibility(
                closes[symbol], index, config=cfg,
                ma_fast=ma_fast[symbol], ma_slow=ma_slow[symbol], holding=holding,
            )
            diagnostics[symbol] = info
            if ok:
                eligible[symbol] = 1.0
        state["holding"] = set(eligible)

        weights: dict[str, float] = {}
        if eligible:
            symbols = sorted(eligible)
            cov = _covariance(returns, symbols, index, cfg.vol_window)
            raw = erc_weights(cov) if cfg.weighting == "erc" else inverse_vol_weights(cov)
            weights = dict(zip(symbols, raw))
            weights = apply_limits(weights, config=cfg, max_weight=cfg.max_weight)
            if cfg.use_vol_target and weights:
                sigma = portfolio_vol(
                    [weights[s] for s in sorted(weights)],
                    _covariance(returns, sorted(weights), index, cfg.vol_window),
                )
                if sigma > cfg.target_vol > 0:
                    scale = cfg.target_vol / sigma
                    weights = {symbol: value * scale for symbol, value in weights.items()}

        # 无交易带：只对"两边都有仓位"的腿生效；清仓和新进都必须成交
        adjusted: dict[str, float] = {}
        for symbol, value in weights.items():
            previous = state["last"].get(symbol, 0.0)
            adjusted[symbol] = previous if previous > 0 and abs(value - previous) < cfg.no_trade_band else value
        for symbol, previous in state["last"].items():
            if symbol in adjusted or previous <= 0:
                continue
            # 已不在入选名单：这一轮直接清掉（band 不保护退出）
            continue

        invested = sum(adjusted.values())
        if invested > 1.0:
            adjusted = {symbol: value / invested for symbol, value in adjusted.items()}
            invested = 1.0
        cash = max(0.0, 1.0 - invested)
        if cash > 1e-9:
            adjusted[CASH] = adjusted.get(CASH, 0.0) + cash
        state["last"] = dict(adjusted)
        state["dates"].append({"date": date, "eligible": sorted(eligible),
                               "weights": dict(adjusted), "cash": cash,
                               "diagnostics": diagnostics})
        return adjusted

    target.diagnostics = state  # type: ignore[attr-defined]
    return target


# --------------------------------------------------------------------------- #
# 回测与基准
# --------------------------------------------------------------------------- #


def bars_of(panel: dict[str, dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    return ex.build_bars(panel.values())


def simulate(
    panel: dict[str, dict[str, Any]],
    calendar: Sequence[str],
    target: Callable[[str, list[str]], dict[str, float]],
    *,
    config: Config | None = None,
    cost_buy: float = BUY_COST,
    cost_sell: float = SELL_COST,
) -> dict[str, Any]:
    cfg = config or Config()
    bars = bars_of(panel)
    flags = ex.tradability(panel.values())
    result = ex.simulate_t1(
        calendar, bars, target=target, flags=flags,
        rebalance_every=cfg.rebalance,
        cost_buy=cost_buy, cost_sell=cost_sell,
        max_turnover=cfg.max_turnover, enforce_t1_sell=False,
    )
    return result


def equal_weight_target(
    calendar: Sequence[str],
    panel: dict[str, dict[str, Any]],
    *,
    start: str | None = None,
) -> Callable[[str, list[str]], dict[str, float]]:
    """同池等权：只拿当日可交易且已上市的腿，等权，月末再平衡。"""
    def target(date: str, available: list[str]) -> dict[str, float]:
        if start and date < start:
            return {}
        names = [symbol for symbol in LIVE_LEGS if symbol in available]
        if not names:
            return {}
        weight = 1.0 / len(names)
        return {symbol: weight for symbol in names}
    return target


def sixty_forty_target(start: str | None = None) -> Callable[[str, list[str]], dict[str, float]]:
    """60/40：60% 沪深300ETF + 40% 国债ETF，月度再平衡（经典对照组）。"""
    def target(date: str, available: list[str]) -> dict[str, float]:
        if start and date < start:
            return {}
        if "510300" not in available or "511010" not in available:
            return {}
        return {"510300": 0.60, "511010": 0.40}
    return target


def nav_metrics(nav: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values = [float(point["nav"]) for point in nav]
    if len(values) < 3:
        return {}
    years = (len(values) - 1) / 252
    returns = [values[i] / values[i - 1] - 1 for i in range(1, len(values)) if values[i - 1] > 0]
    total = values[-1] / values[0] - 1
    cagr = (values[-1] / values[0]) ** (1 / years) - 1 if years > 0 else 0.0
    vol = statistics.stdev(returns) * math.sqrt(252) if len(returns) > 1 else 0.0
    peak, worst = values[0], 0.0
    for value in values:
        peak = max(peak, value)
        worst = min(worst, value / peak - 1)
    return {
        "total_return": round(total * 100, 2),
        "cagr": round(cagr * 100, 2),
        "vol": round(vol * 100, 2),
        "sharpe": round(cagr / vol, 3) if vol > 1e-9 else None,
        "max_drawdown": round(worst * 100, 2),
        "calmar": round(cagr / abs(worst), 3) if worst < -1e-9 else None,
        "years": round(years, 2),
    }


def summarize(result: dict[str, Any], calendar: Sequence[str], name: str) -> dict[str, Any]:
    metrics = nav_metrics(result["nav"])
    years = max(len(calendar) / 252, 1e-9)
    metrics.update({
        "name": name,
        "annual_turnover": round(result["total_turnover"] / years * 100, 1),
        "cost_bps": round(result["total_cost"] / years * 10000, 1),
        "rebalances": len(result["rebalances"]),
        "cash_mean": round(result.get("cash_ratio_mean", 0.0) * 100, 1),
        "blocked": dict(result.get("blocked") or {}),
    })
    return {"nav": result["nav"], "metrics": metrics}


def window_metrics(nav: Sequence[dict[str, Any]], start: str, end: str) -> dict[str, Any]:
    block = [point for point in nav if start <= point["date"] <= end]
    if len(block) < 20:
        return {}
    values = [point["nav"] for point in block]
    peak, worst = values[0], 0.0
    for value in values:
        peak = max(peak, value)
        worst = min(worst, value / peak - 1)
    return {"return": round((values[-1] / values[0] - 1) * 100, 2),
            "max_drawdown": round(worst * 100, 2), "days": len(block)}


# --------------------------------------------------------------------------- #
# 验证
# --------------------------------------------------------------------------- #

PERIODS: tuple[tuple[str, str, str], ...] = (
    ("2015 泡沫与熔断", "2015-01-05", "2016-12-30"),
    ("2017 单边上涨", "2017-01-03", "2017-12-29"),
    ("2018 熊市", "2018-01-02", "2018-12-28"),
    ("2019–2021 结构牛", "2019-01-02", "2021-12-31"),
    ("2022 回撤", "2022-01-04", "2022-12-30"),
    ("2023–2024 震荡", "2023-01-03", "2024-12-31"),
    ("2025–2026", "2025-01-02", "2026-12-31"),
)


def run_study(
    panel: dict[str, dict[str, Any]],
    *,
    start: str,
    end: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    stale = stale_symbols(panel)
    calendar = _calendar(panel, start=start, as_of=end)
    if len(calendar) < 300:
        raise ValueError(f"交易日只有 {len(calendar)} 天，不足以回测")
    if verbose and stale:
        print("  剔除过期标的：" + "、".join(f"{code}({last})" for code, last in sorted(stale.items())),
              flush=True)

    variants: dict[str, Config] = {
        "BASE": Config(),
        "FASTIN": Config(entry_requires_slow=False),
        "NOSTOP": Config(use_stop=False),
        "NOVOL": Config(use_vol_target=False),
        "INVVOL": Config(weighting="inv_vol"),
        "BIWEEK": Config(rebalance=10),
    }
    results: dict[str, dict[str, Any]] = {}
    for name, config in variants.items():
        target = build_target(panel, calendar, config=config, stale=stale)
        sim = simulate(panel, calendar, target, config=config)
        results[name] = summarize(sim, calendar, name)
        if name == "BASE":
            results[name]["diagnostics"] = getattr(target, "diagnostics", {}).get("dates", [])
        if verbose:
            row = results[name]["metrics"]
            print(f"  {name:<7} 年化 {row['cagr']:>6.2f}%  Sharpe {row['sharpe']:>5.2f}  "
                  f"回撤 {row['max_drawdown']:>7.2f}%  Calmar {row['calmar']:>5.2f}  "
                  f"换手 {row['annual_turnover']:>6.1f}%  现金 {row['cash_mean']:>5.1f}%", flush=True)

    benchmarks = {
        "EW": summarize(simulate(panel, calendar, equal_weight_target(calendar, panel, start=start)),
                        calendar, "等权 ETF"),
        "6040": summarize(simulate(panel, calendar, sixty_forty_target(start)), calendar, "60/40"),
        "HOLD300": summarize(simulate(panel, calendar,
                                      lambda date, available: {} if date < start
                                      else ({"510300": 1.0} if "510300" in available else {})),
                             calendar, "沪深300 买入持有"),
    }
    if verbose:
        for name, row in benchmarks.items():
            metrics = row["metrics"]
            print(f"  {name:<7} 年化 {metrics['cagr']:>6.2f}%  Sharpe {metrics['sharpe']:>5.2f}  "
                  f"回撤 {metrics['max_drawdown']:>7.2f}%  Calmar {metrics['calmar']:>5.2f}", flush=True)

    periods = {}
    for label, pstart, pend in PERIODS:
        block = {}
        for name in ("BASE", "EW", "6040"):
            source = results.get(name) or benchmarks.get(name)
            if source:
                block[name] = window_metrics(source["nav"], pstart, pend)
        if block:
            periods[label] = block

    base = results["BASE"]["metrics"]
    equal = benchmarks["EW"]["metrics"]
    sixty = benchmarks["6040"]["metrics"]
    dd_gap = base["max_drawdown"] - equal["max_drawdown"]  # 正数 = 策略回撤更浅
    pass_sharpe = (base["sharpe"] or 0) > (equal["sharpe"] or 0)
    pass_dd = dd_gap >= 5.0
    crash = {}
    for label in ("2018 熊市", "2022 回撤"):
        if label in periods and "BASE" in periods[label] and "EW" in periods[label]:
            crash[label] = {
                "base": periods[label]["BASE"]["return"],
                "ew": periods[label]["EW"]["return"],
                "6040": periods[label].get("6040", {}).get("return"),
            }
    pass_crash = all(row["base"] > row["ew"] for row in crash.values()) if crash else False

    # 同风险等效收益：Sharpe × 策略波动。跨不同波动水平直接比年化是误导，
    # 这条把基准"折算到策略的风险预算"再比。
    base_vol = base["vol"] / 100
    matched = {
        "EW": round((equal["sharpe"] or 0) * base_vol * 100, 2),
        "6040": round((sixty["sharpe"] or 0) * base_vol * 100, 2),
        "HOLD300": round((benchmarks["HOLD300"]["metrics"]["sharpe"] or 0) * base_vol * 100, 2),
    }
    diagnostics = results["BASE"].get("diagnostics") or []
    legs = [len(row["eligible"]) for row in diagnostics]
    structure = {
        "rebalances": len(diagnostics),
        "avg_legs": round(statistics.fmean(legs), 2) if legs else None,
        "min_legs": min(legs) if legs else None,
        "full_cash_rebalances": sum(1 for row in diagnostics if row["cash"] > 0.95),
        "leverage_needed": round(0.10 / base_vol, 2) if base_vol else None,
        "projected": {
            "cagr": round(((1 + base["cagr"] / 100) ** (0.10 / base_vol) - 1) * 100, 2) if base_vol else None,
            "max_drawdown": round(base["max_drawdown"] * (0.10 / base_vol), 2) if base_vol else None,
        },
    }

    return {
        "calendar": calendar,
        "stale": stale,
        "results": results,
        "benchmarks": benchmarks,
        "matched_vol": matched,
        "structure": structure,
        "periods": periods,
        "verdict": {
            "sharpe": {"ok": pass_sharpe, "text": f"{base['sharpe']:.2f} vs 等权 {equal['sharpe']:.2f}"},
            "drawdown": {"ok": pass_dd,
                         "text": f"{base['max_drawdown']:.2f}% vs 等权 {equal['max_drawdown']:.2f}%"
                                 f"（差 {dd_gap:+.2f}pp）"},
            "crash": {"ok": pass_crash,
                      "text": " / ".join(f"{year} {row['base']:+.2f}% vs 等权 {row['ew']:+.2f}%"
                                         for year, row in crash.items()) or "样本不足"},
            "calmar": {"ok": (base["calmar"] or 0) > (equal["calmar"] or 0),
                       "text": f"{base['calmar']:.2f} vs 等权 {equal['calmar']:.2f}"},
        },
    }


# --------------------------------------------------------------------------- #
# 产品化：数据版本、当前配置快照、纸面跟踪
# --------------------------------------------------------------------------- #

DEFAULT_START = "2015-01-05"   # 留足 12-1 动量 + MA120 预热


def rules() -> dict[str, Any]:
    """把规则写成可序列化的块：页面展示与纸面跟踪都用它做"规则版本"证据。"""
    cfg = Config()
    return {
        "version": RULE_VERSION,
        "entry": "12-1 动量 > 0；收盘 > MA60；收盘 > MA120；距 252 日高点回撤 > -15%",
        "hold": "只需动量 > 0 且收盘 > MA120（MA60 只用于入场），止损仍生效",
        "weighting": "ERC（60 日协方差）",
        "max_weight": cfg.max_weight,
        "bucket_caps": {bucket: cap for bucket, cap in cfg.bucket_caps},
        "target_vol": cfg.target_vol,
        "rebalance_days": cfg.rebalance,
        "no_trade_band": cfg.no_trade_band,
        "max_turnover": cfg.max_turnover,
        "cost_bps": {"buy": BUY_COST * 10000, "sell": SELL_COST * 10000},
        "execution": "T 日收盘出信号，T+1 开盘成交",
        "cash_leg": CASH,
    }


def data_version(panel: dict[str, dict[str, Any]]) -> str:
    """由「标的 × 首末日 × 行数」算出的数据指纹。

    不含生成时间与字典顺序，所以同一份数据重复生成永远得到同一个版本号；
    页面与纸面跟踪都以它为准，避免"权重没变但数据换了"这种情况被忽略。
    """
    parts = []
    for symbol in sorted(panel):
        rows = panel[symbol].get("rows") or []
        if not rows:
            parts.append(f"{symbol}:empty")
            continue
        parts.append(f"{symbol}:{rows[0]['date']}:{rows[-1]['date']}:{len(rows)}")
    return "etf-" + hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:10]


def panel_window(panel: dict[str, dict[str, Any]]) -> dict[str, Any]:
    starts = [rows["rows"][0]["date"] for rows in panel.values() if rows.get("rows")]
    ends = [rows["rows"][-1]["date"] for rows in panel.values() if rows.get("rows")]
    return {
        "start": min(starts) if starts else None,
        "end": max(ends) if ends else None,
        "symbols": sorted(panel),
        "rows": sum(len(item.get("rows") or []) for item in panel.values()),
    }


def stale_symbols(
    panel: dict[str, dict[str, Any]],
    *,
    tolerance: int = STALE_TOLERANCE,
) -> dict[str, str]:
    """末端日期落后全样本最新日期超过 ``tolerance`` 个交易日的标的。

    清盘 ETF（510170 于 2022-03 清盘）不会报错，只会让"当日有行情"的集合悄悄变小。
    ``align`` 又会前向填充价格，于是它变成一条零波动直线常年挂在池边——必须显式
    剔除并写进产物，而不是等某个报告里发现"区间短了 4 年"。
    """
    union = sorted({row["date"] for history in panel.values() for row in history.get("rows") or []})
    if not union:
        return {}
    position = {date: index for index, date in enumerate(union)}
    last_index = len(union) - 1
    stale: dict[str, str] = {}
    for symbol, history in panel.items():
        rows = history.get("rows") or []
        if not rows:
            stale[symbol] = ""
            continue
        last = rows[-1]["date"]
        if last_index - position.get(last, last_index) > tolerance:
            stale[symbol] = last
    return stale


def _active_legs(panel: dict[str, dict[str, Any]], stale: dict[str, str]) -> list[str]:
    return [symbol for symbol in LIVE_LEGS if symbol in panel and symbol not in stale]


def _calendar(
    panel: dict[str, dict[str, Any]],
    *,
    start: str | None = None,
    as_of: str | None = None,
    core: str = "510300",
    require_cash: bool = True,
) -> list[str]:
    """交易日历只由「核心腿 ∩ 现金腿」决定，不用全部 12 条腿求交。

    用 12 腿求交的后果是：任何一条腿清盘，整段样本被截断，回测"少跑几年"却毫无提示。
    单腿的缺失由 ``stale_symbols`` 显式处理，日历本身不该被它绑架。
    """
    dates = {row["date"] for row in panel.get(core, {}).get("rows") or []}
    if require_cash and CASH in panel:
        dates &= {row["date"] for row in panel[CASH]["rows"]}
    calendar = sorted(dates)
    if start:
        calendar = [day for day in calendar if day >= start]
    if as_of:
        calendar = [day for day in calendar if day <= as_of]
    return calendar


def _availability(panel: dict[str, dict[str, Any]]) -> dict[str, set[str]]:
    """``{date: {有收盘价的标的}}``：与 ``execution.simulate_t1`` 的可用口径一致。"""
    table: dict[str, set[str]] = {}
    for symbol, history in panel.items():
        for row in history.get("rows") or []:
            if float(row.get("close") or 0) > 0:
                table.setdefault(row["date"], set()).add(symbol)
    return table


def _available(panel: dict[str, dict[str, Any]], day: str) -> list[str]:
    return sorted(_availability(panel).get(day, set()))


def cash_yield(panel: dict[str, dict[str, Any]], as_of: str, *, window: int = 60) -> float | None:
    """现金腿的年化收益（用 511880 自己近 60 个交易日算，不假设 0 收益）。"""
    rows = [row for row in panel.get(CASH, {}).get("rows") or [] if row["date"] <= as_of]
    if len(rows) < window + 1:
        return None
    closes = [float(row["close"]) for row in rows[-window - 1:]]
    if closes[0] <= 0:
        return None
    return round((closes[-1] / closes[0]) ** (252 / window) - 1, 6)


def check_constraints(weights: dict[str, float], *, config: Config | None = None) -> dict[str, Any]:
    """权重约束自检：和 ≤ 1、单腿 ≤ 30%、类上限 ≤ 50%、现金不为负。"""
    cfg = config or Config()
    legs = {symbol: value for symbol, value in weights.items() if symbol != CASH}
    total = sum(weights.values())
    violations: list[str] = []
    if total > 1 + 1e-9:
        violations.append(f"权重和 {total:.4f} > 1")
    for symbol, value in legs.items():
        if value > cfg.max_weight + 1e-9:
            violations.append(f"{symbol} 单腿 {value:.4f} > {cfg.max_weight:.2f}")
    for bucket, cap in cfg.bucket_caps:
        bucket_sum = sum(value for symbol, value in legs.items()
                         if UNIVERSE.get(symbol, {}).get("bucket") == bucket)
        if bucket_sum > cap + 1e-9:
            violations.append(f"{bucket} 合计 {bucket_sum:.4f} > {cap:.2f}")
    if weights.get(CASH, 0.0) < -1e-12:
        violations.append("现金腿为负")
    return {
        "ok": not violations,
        "violations": violations,
        "sum": round(total, 6),
        "max_leg": round(max(legs.values()), 6) if legs else 0.0,
        "cash": round(weights.get(CASH, 0.0), 6),
        "buckets": {
            bucket: round(sum(value for symbol, value in legs.items()
                              if UNIVERSE.get(symbol, {}).get("bucket") == bucket), 6)
            for bucket, _ in cfg.bucket_caps
        },
    }


def _weights_hash(weights: dict[str, float]) -> str:
    payload = json.dumps({key: round(value, 6) for key, value in sorted(weights.items())},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _turnover(previous: dict[str, float], current: dict[str, float]) -> dict[str, float]:
    """把两期权重的差拆成买卖两侧；与执行层 ``one_way = 0.5*(买+卖)`` 同口径。"""
    symbols = set(previous) | set(current)
    buy = sum(max(0.0, current.get(s, 0.0) - previous.get(s, 0.0)) for s in symbols)
    sell = sum(max(0.0, previous.get(s, 0.0) - current.get(s, 0.0)) for s in symbols)
    return {"buy": round(buy, 6), "sell": round(sell, 6),
            "one_way": round(0.5 * (buy + sell), 6),
            "cost_bps": round((buy * BUY_COST + sell * SELL_COST) * 10000, 3)}


def compute_snapshot(
    panel: dict[str, dict[str, Any]],
    *,
    as_of: str | None = None,
    config: Config | None = None,
    start: str | None = DEFAULT_START,
) -> dict[str, Any]:
    """当前配置：用**收盘信号**算出目标权重，执行日单独给（T+1 开盘）。

    关键点是"信号日"和"持仓日"分开：``signal_date`` 是算出这组权重的那根收盘，
    ``fill_date`` 是实际成交日（有下一交易日行情时给真实日期，否则为 ``None`` 并
    标记 ``pending_next_open``）。两者混成一个 ``as_of`` 就等于假装收盘价能成交。
    """
    cfg = config or Config()
    stale = stale_symbols(panel)
    active = _active_legs(panel, stale)
    if not active:
        raise ValueError("没有可用腿（全部缺失或被判为过期）")
    full_calendar = _calendar(panel, start=start)
    if not full_calendar:
        raise ValueError("交易日历为空")
    signal_date = as_of or full_calendar[-1]
    if signal_date not in full_calendar:
        prior = [day for day in full_calendar if day <= signal_date]
        if not prior:
            raise ValueError(f"{signal_date} 之前没有交易日")
        signal_date = prior[-1]
    calendar = [day for day in full_calendar if day <= signal_date]

    target = build_target(panel, calendar, config=cfg, start=start, stale=stale)
    available = _availability(panel)
    anchors = calendar[:: cfg.rebalance]
    history: list[dict[str, Any]] = []
    previous: dict[str, float] = {}
    for day in anchors:
        weights = target(day, sorted(available.get(day, set())))
        move = _turnover(previous, weights)
        history.append({"signal_date": day, "weights": dict(weights), **move})
        previous = dict(weights)
    if not history or history[-1]["signal_date"] != signal_date:
        target(signal_date, sorted(available.get(signal_date, set())))
    diagnostics = (getattr(target, "diagnostics", {}) or {}).get("dates") or []
    latest = diagnostics[-1] if diagnostics else {"date": signal_date, "eligible": [], "weights": {},
                                                  "cash": 0.0, "diagnostics": {}}
    # 权重规整：腿各自四舍五入到 1e-6，现金取「余数向下取整」。
    # 直接把 1 - Σ腿 四舍五入会得到 Σ = 1.000001 这种"超过 100%"的账面，
    # 对"权重和 ≤ 1"这条硬约束是假违反，但页面上看着像 bug，所以从源头修掉。
    raw = {symbol: float(value) for symbol, value in (latest.get("weights") or {}).items()
           if value > 1e-9}
    legs = {symbol: round(value, 6) for symbol, value in raw.items() if symbol != CASH}
    leg_total = sum(legs.values())
    if leg_total > 1:
        legs = {symbol: round(value / leg_total, 6) for symbol, value in legs.items()}
        leg_total = sum(legs.values())
    cash_residual = max(0.0, math.floor((1.0 - leg_total) * 1e6) / 1e6)
    weights = dict(legs)
    if cash_residual > 0:
        weights[CASH] = cash_residual

    future = [day for day in full_calendar if day > signal_date]
    fill_date = future[0] if future else None
    last_rebalance = history[-1] if history else None
    if last_rebalance:
        after = [day for day in full_calendar if day > last_rebalance["signal_date"]]
        last_rebalance = {**last_rebalance, "fill_date": after[0] if after else None,
                          "fill_status": "filled" if after else "pending_next_open"}

    holdings = []
    for symbol in sorted(weights, key=lambda code: (-weights[code], code)):
        if symbol == CASH:
            continue
        info = UNIVERSE.get(symbol, {})
        detail = (latest.get("diagnostics") or {}).get(symbol) or {}
        holdings.append({
            "symbol": symbol,
            "name": info.get("name", symbol),
            "class": info.get("class", ""),
            "bucket": info.get("bucket", ""),
            "weight": weights[symbol],
            "reason": {
                "rule": detail.get("reason"),
                "momentum_12_1": round(detail["mom"], 6) if detail.get("mom") is not None else None,
                "above_ma60": detail.get("above_fast"),
                "above_ma120": detail.get("above_slow"),
                "drawdown_from_high": round(detail["drawdown"], 6)
                if detail.get("drawdown") is not None else None,
            },
            "note": info.get("note", ""),
        })
    cash_weight = weights.get(CASH, 0.0)
    exposure = round(sum(item["weight"] for item in holdings), 6)
    passed = latest.get("eligible") or []
    rejected = []
    for symbol, detail in (latest.get("diagnostics") or {}).items():
        if symbol in weights:
            continue
        rejected.append({"symbol": symbol, "name": UNIVERSE.get(symbol, {}).get("name", symbol),
                         "reason": detail.get("reason"),
                         "momentum_12_1": round(detail["mom"], 6) if detail.get("mom") is not None else None})

    return {
        "schema": "etf-allocation-snapshot/1",
        "rule_version": RULE_VERSION,
        "data_version": data_version(panel),
        "as_of": signal_date,
        "signal_date": signal_date,
        "fill_date": fill_date,
        "fill_status": "effective" if fill_date else "pending_next_open",
        "holdings": holdings,
        "cash": {
            "symbol": CASH,
            "name": "货币ETF（现金腿）",
            "weight": round(cash_weight, 6),
            "yield_60d_annual": cash_yield(panel, signal_date),
        },
        "exposure": exposure,
        "weights": weights,
        "weights_sum": round(sum(weights.values()), 6),
        "weights_hash": _weights_hash(weights),
        "constraints": check_constraints(weights, config=cfg),
        "eligible": sorted(passed),
        "rejected": sorted(rejected, key=lambda item: item["symbol"]),
        "last_rebalance": last_rebalance,
        "rebalance_history": history[-6:],
        "stale": stale,
        "window": {"start": calendar[0], "end": calendar[-1], "days": len(calendar)},
        "panel": panel_window(panel),
        "rules": rules(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def read_paper(path: Path | None = None) -> list[dict[str, Any]]:
    target = Path(path) if path else PAPER_PATH
    if not target.is_file():
        return []
    records = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def record_paper(
    snapshot: dict[str, Any],
    *,
    path: Path | None = None,
    force: bool = False,
    calendar: Sequence[str] | None = None,
    rebalance: int | None = None,
) -> dict[str, Any]:
    """把一次快照写入纸面跟踪台账；同信号日 + 同数据版本 + 同规则版本只记一次。

    ``calendar`` 给出交易日序列时会额外检查"调仓是否到期"：距上一条记录不足
    ``rebalance`` 个交易日就跳过。不这么做的话，每天跑一次就会每天记一条，
    纸面换手与成本变成"日频"，与回测的月频口径无法对照。

    ``force=True`` 用于补记历史（回溯生成），同时绕过到期检查，不会覆盖已有记录。
    """
    target = Path(path) if path else PAPER_PATH
    records = read_paper(target)
    if not force and records and calendar:
        need = rebalance or Config().rebalance
        last = records[-1].get("signal_date") or ""
        between = [day for day in calendar if last < day <= snapshot["signal_date"]]
        if len(between) < need:
            return {"recorded": False, "path": str(target), "records": len(records),
                    "reason": f"not_due：距上次 {len(between)} 个交易日 < {need}"}
    key = (snapshot["signal_date"], snapshot["rule_version"], snapshot["data_version"])
    for record in records:
        if (record.get("signal_date"), record.get("rule_version"),
                record.get("data_version")) == key:
            return {"recorded": False, "reason": "duplicate", "path": str(target),
                    "records": len(records)}
        if not force and record.get("signal_date") == snapshot["signal_date"]:
            return {"recorded": False, "reason": "same_signal_date", "path": str(target),
                    "records": len(records)}
    previous = records[-1]["weights"] if records else {}
    move = _turnover(previous, snapshot["weights"])
    record = {
        "schema": "etf-paper/1",
        "signal_date": snapshot["signal_date"],
        "fill_date": snapshot["fill_date"],
        "fill_status": snapshot["fill_status"],
        "rule_version": snapshot["rule_version"],
        "data_version": snapshot["data_version"],
        "weights": snapshot["weights"],
        "weights_hash": snapshot["weights_hash"],
        "exposure": snapshot["exposure"],
        "cash_weight": snapshot["cash"]["weight"],
        "stale": snapshot["stale"],
        "one_way_turnover": move["one_way"],
        "cost_bps": move["cost_bps"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"recorded": True, "path": str(target), "records": len(records) + 1,
            "one_way_turnover": move["one_way"], "cost_bps": move["cost_bps"]}


def _open_price(panel: dict[str, dict[str, Any]], symbol: str, day: str) -> float | None:
    for row in panel.get(symbol, {}).get("rows") or []:
        if row["date"] == day:
            value = float(row.get("open") or 0)
            return value if value > 0 else None
    return None


def paper_summary(
    panel: dict[str, dict[str, Any]],
    *,
    path: Path | None = None,
    backtest: dict[str, Any] | None = None,
    start: str | None = DEFAULT_START,
) -> dict[str, Any]:
    """纸面跟踪的已实现结果：按"兑现一期"（每条记录持有到下一次信号）结算。

    收益用**开盘价**算：信号日 T 收盘出信号 → T+1 开盘买入 → 下次信号的 T'+1 开盘
    卖出，与回测的 ``simulate_t1`` 同一口径；成本用同一组费率按买卖两侧分别计。
    """
    records = read_paper(path)
    calendar = _calendar(panel, start=start)
    if len(records) < 2:
        return {"available": False, "records": len(records),
                "note": "纸面跟踪至少需要两条记录（两次调仓）才能结算"}
    periods = []
    for previous, current in zip(records, records[1:]):
        entry_candidates = [day for day in calendar if day > previous["signal_date"]]
        exit_candidates = [day for day in calendar if day > current["signal_date"]]
        if not entry_candidates or not exit_candidates:
            continue
        entry, exit_day = entry_candidates[0], exit_candidates[0]
        gross = 0.0
        missing = []
        for symbol, weight in (previous.get("weights") or {}).items():
            if weight <= 0:
                continue
            if symbol == CASH:
                entry_price = _open_price(panel, CASH, entry)
                exit_price = _open_price(panel, CASH, exit_day)
                if entry_price and exit_price:
                    gross += weight * (exit_price / entry_price - 1)
                continue
            entry_price = _open_price(panel, symbol, entry)
            exit_price = _open_price(panel, symbol, exit_day)
            if not entry_price or not exit_price:
                missing.append(symbol)
                continue
            gross += weight * (exit_price / entry_price - 1)
        cost = (current.get("cost_bps") or 0) / 10000
        periods.append({
            "signal_date": previous["signal_date"],
            "entry_date": entry,
            "exit_signal_date": current["signal_date"],
            "exit_date": exit_day,
            "gross_return": round(gross * 100, 4),
            "cost_bps": current.get("cost_bps"),
            "net_return": round((gross - cost) * 100, 4),
            "one_way_turnover": current.get("one_way_turnover"),
            "missing": sorted(set(missing)),
        })
    if not periods:
        return {"available": False, "records": len(records), "note": "还没有可结算的完整持有期"}
    net = [row["net_return"] / 100 for row in periods]
    cumulative = 1.0
    for value in net:
        cumulative *= 1 + value
    costs = [row["cost_bps"] or 0 for row in periods]
    turnovers = [row["one_way_turnover"] or 0 for row in periods]
    expected = None
    if backtest:
        metrics = backtest.get("metrics") or {}
        n = backtest.get("rebalances") or 0
        if n:
            # 回测的平均单次调仓成本：总成本 / 调仓次数（同一组费率、同一公式）
            expected = round((metrics.get("cost_bps") or 0) * (metrics.get("years") or 0) / n, 2)
    return {
        "available": True,
        "records": len(records),
        "periods": periods,
        "n_periods": len(periods),
        "cumulative_net_pct": round((cumulative - 1) * 100, 3),
        "avg_net_bps": round(statistics.fmean(net) * 10000, 1) if net else None,
        "avg_one_way_turnover": round(statistics.fmean(turnovers), 4) if turnovers else None,
        "avg_cost_bps": round(statistics.fmean(costs), 2) if costs else None,
        "backtest_avg_cost_bps": expected,
        "cost_consistent": (expected is None or not costs
                            or abs(statistics.fmean(costs) - expected) <= max(2.0, expected * 0.35)),
        "note": "纸面跟踪只验证规则可执行与成本口径一致；样本不足前不构成业绩证据",
    }


def write_latest(snapshot: dict[str, Any], *, outdir: Path | None = None,
                 backtest: dict[str, Any] | None = None,
                 paper: dict[str, Any] | None = None,
                 previous: dict[str, Any] | None = None) -> Path:
    """落 ``reports/etf/latest.json``：生产层（app.py）只读这个文件。"""
    target_dir = Path(outdir) if outdir else REPORT
    target_dir.mkdir(parents=True, exist_ok=True)
    # "与上期变化"的基准：优先用纸面跟踪的上一条记录；首期没有记录时退回
    # "上一次实际调仓"的目标权重——那正是"这次要动什么"的对照物。
    basis_weights: dict[str, float] = {}
    basis_label = "首期，无对照"
    if previous and previous.get("weights"):
        basis_weights = dict(previous["weights"])
        basis_label = f"上期快照 {previous.get('signal_date')}"
    elif snapshot.get("last_rebalance"):
        basis_weights = dict(snapshot["last_rebalance"].get("weights") or {})
        basis_label = f"上次调仓 {snapshot['last_rebalance'].get('signal_date')}"
    changes = []
    if basis_weights:
        for symbol in sorted(set(snapshot["weights"]) | set(basis_weights)):
            before = basis_weights.get(symbol, 0.0)
            after = snapshot["weights"].get(symbol, 0.0)
            delta = round(after - before, 6)
            if abs(delta) < 1e-4:      # 1bp 以下不进"变化"列表：那是舍入噪声
                continue
            changes.append({
                "symbol": symbol,
                "name": UNIVERSE.get(symbol, {}).get("name") or ("现金" if symbol == CASH else symbol),
                "before": round(before, 6),
                "after": round(after, 6),
                "delta": delta,
                "action": "新建" if before <= 1e-9 else ("清仓" if after <= 1e-9 else
                                                          ("加仓" if delta > 0 else "减仓")),
            })
    payload = {
        "schema": "etf-allocation/1",
        "rule_version": snapshot["rule_version"],
        "data_version": snapshot["data_version"],
        "as_of": snapshot["as_of"],
        "signal_date": snapshot["signal_date"],
        "fill_date": snapshot["fill_date"],
        "fill_status": snapshot["fill_status"],
        "holdings": snapshot["holdings"],
        "cash": snapshot["cash"],
        "exposure": snapshot["exposure"],
        "weights_sum": snapshot["weights_sum"],
        "constraints": snapshot["constraints"],
        "last_rebalance": snapshot["last_rebalance"],
        "rebalance_history": snapshot["rebalance_history"],
        "changes": changes,
        "changes_basis": basis_label,
        "previous_signal_date": (previous or {}).get("signal_date"),
        "eligible": snapshot["eligible"],
        "rejected": snapshot["rejected"],
        "stale": snapshot["stale"],
        "window": snapshot["window"],
        "panel": snapshot["panel"],
        "rules": snapshot["rules"],
        "backtest": backtest or {},
        "paper": paper or {},
        "observation": _observation_gate(paper),
        "risk": _risk_notes(snapshot, paper),
        "disclaimer": DISCLAIMER,
        "generated_at": snapshot["generated_at"],
    }
    path = target_dir / "latest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _observation_gate(paper: dict[str, Any] | None) -> dict[str, Any]:
    """观察期闸门：纸面跟踪不足 3 期时，产物里显式标记"不可据此调仓"。

    这不是文案问题而是数据契约问题——只要 ``status`` 还是 ``observing``，任何下游
    （页面、导出、别人的脚本）都能一眼看出"这份配置还没有执行证据"。
    """
    required = 3
    periods = int((paper or {}).get("n_periods") or 0) if (paper or {}).get("available") else 0
    ready = periods >= required
    return {
        "required_periods": required,
        "periods": periods,
        "status": "ready" if ready else "observing",
        "note": ("纸面跟踪已满 %d 期，可以评估执行偏差" % periods) if ready
        else f"纸面跟踪仅 {periods}/{required} 期：本配置仅用于观察，不作为调仓依据",
    }


def _risk_notes(snapshot: dict[str, Any], paper: dict[str, Any] | None) -> list[str]:
    """风险提示只写事实，不写建议。"""
    notes: list[str] = []
    cash = snapshot["cash"]["weight"]
    if cash >= 0.5:
        notes.append(f"多数腿未通过趋势确认，当前现金腿占 {cash * 100:.0f}%，暴露只有 {snapshot['exposure'] * 100:.0f}%")
    if snapshot["holdings"]:
        top = snapshot["holdings"][0]
        notes.append(f"最大单腿 {top['name']} {top['weight'] * 100:.0f}%（上限 {snapshot['rules']['max_weight'] * 100:.0f}%）")
    buckets = snapshot["constraints"]["buckets"]
    for bucket, value in buckets.items():
        if value >= 0.4:
            notes.append(f"{bucket} 合计 {value * 100:.0f}%，接近类别上限 {snapshot['rules']['bucket_caps'][bucket] * 100:.0f}%")
    if snapshot["stale"]:
        names = "、".join(f"{code}({last})" for code, last in sorted(snapshot["stale"].items()))
        notes.append(f"已剔除过期标的：{names}（清盘/长期停牌会让日历静默变短，已按规则排除）")
    if not paper or not paper.get("available"):
        notes.append("纸面跟踪样本不足，尚未验证实盘执行偏差")
    elif paper.get("n_periods", 0) < 3:
        notes.append(f"纸面跟踪仅 {paper.get('n_periods')} 期，未满 3 个月观察期")
    notes.append(DISCLAIMER)
    return notes


def evaluate(
    panel: dict[str, dict[str, Any]],
    *,
    start: str = DEFAULT_START,
    rebalance: int = 21,
) -> dict[str, Any]:
    """规则组 vs 基准的对照（供 validate_fields 与报告使用）。

    变体只改一个开关，用来回答"这条规则有没有贡献"：
    ``no_stop`` 去掉回撤止损、``no_voltarget`` 去掉波动目标、``inv_vol`` 换成逆波动、
    ``expensive`` 成本翻倍。基准是同池等权与沪深300 买入持有。
    """
    stale = stale_symbols(panel)
    active = _active_legs(panel, stale)
    calendar = _calendar(panel, start=start)
    if len(calendar) < 260:
        raise ValueError(f"共同交易日只有 {len(calendar)} 天，不足以回测")

    variants: dict[str, Config] = {
        "strategy": Config(rebalance=rebalance),
        "no_stop": Config(rebalance=rebalance, use_stop=False),
        "no_voltarget": Config(rebalance=rebalance, use_vol_target=False),
        "inv_vol": Config(rebalance=rebalance, weighting="inv_vol"),
    }
    results: dict[str, dict[str, Any]] = {}
    for name, config in variants.items():
        target = build_target(panel, calendar, config=config, start=start, stale=stale)
        results[name] = summarize(simulate(panel, calendar, target, config=config), calendar, name)
    expensive = summarize(
        simulate(panel, calendar, build_target(panel, calendar, start=start, stale=stale),
                 config=Config(rebalance=rebalance), cost_buy=BUY_COST * 2, cost_sell=SELL_COST * 2),
        calendar, "成本×2",
    )
    equal = summarize(simulate(panel, calendar, equal_weight_target(calendar, panel, start=start)),
                      calendar, "同池等权")
    hold_target = (lambda date, available: {} if date < start
                   else ({"510300": 1.0} if "510300" in available else {}))
    hold = summarize(simulate(panel, calendar, hold_target), calendar, "沪深300 买入持有")

    def row(name: str, metrics: dict[str, Any]) -> str:
        return (f"| {name} | {metrics.get('cagr', 0):+.2f}% | {metrics.get('vol', 0):.2f}% | "
                f"{metrics.get('sharpe', 0):+.2f} | {metrics.get('max_drawdown', 0):+.2f}% | "
                f"{metrics.get('calmar', 0):+.2f} | {metrics.get('annual_turnover', 0):.0f}% | "
                f"{metrics.get('cost_bps', 0):.0f}bp |")

    table = "\n".join([
        "| 方案 | 年化 | 年化波动 | 净 Sharpe | 最大回撤 | Calmar | 年换手 | 成本拖累 |",
        "|---|---|---|---|---|---|---|---|",
        row("规则组（本次）", results["strategy"]["metrics"]),
        row("去掉回撤止损", results["no_stop"]["metrics"]),
        row("去掉波动目标", results["no_voltarget"]["metrics"]),
        row("逆波动加权", results["inv_vol"]["metrics"]),
        row("成本×2", expensive["metrics"]),
        row("同池等权", equal["metrics"]),
        row("沪深300 买入持有", hold["metrics"]),
    ])

    base, bench = results["strategy"]["metrics"], equal["metrics"]
    dd_gap = base["max_drawdown"] - bench["max_drawdown"]      # 正数 = 回撤更浅
    reasons: list[str] = []
    if (base.get("sharpe") or 0) <= (bench.get("sharpe") or 0):
        reasons.append(f"净 Sharpe {base.get('sharpe')} 未高于同池等权 {bench.get('sharpe')}")
    if dd_gap < 5.0:
        reasons.append(f"最大回撤 {base.get('max_drawdown')}% 相对等权 {bench.get('max_drawdown')}% 只改善 {dd_gap:.2f}pp（要求 ≥5pp）")
    return {
        "available": True,
        "start": calendar[0],
        "end": calendar[-1],
        "rebalance": rebalance,
        "strategy": {**results["strategy"], "settings": {**rules(), "stale": stale}},
        "no_stop": results["no_stop"],
        "no_voltarget": results["no_voltarget"],
        "inv_vol": results["inv_vol"],
        "expensive": expensive,
        "equal_weight": equal,
        "buy_hold": hold,
        "table": table,
        "pass": not reasons,
        "reasons": reasons,
        "nav": results["strategy"]["nav"],
        "stale": stale,
        "active_legs": active,
    }


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #


def _num(value: Any, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _signed(value: Any, digits: int = 2) -> str:
    return "—" if value is None else f"{value:+.{digits}f}"


def _svg_nav(navs: dict[str, list[dict[str, Any]]], *, width: int = 920, height: int = 320) -> str:
    pad = 44
    values = [point["nav"] for series in navs.values() for point in series]
    if not values:
        return ""
    low, high = min(values), max(values)
    span = max(high - low, 1e-6)
    colors = {"策略（BASE）": "#c0392b", "等权 ETF": "#7f8c8d", "60/40": "#2980b9", "沪深300": "#bdc3c7"}
    paths, legend = [], []
    for index, (name, series) in enumerate(navs.items()):
        count = max(len(series) - 1, 1)
        color = colors.get(name, "#34495e")
        path = " ".join(
            f"{pad + i / count * (width - pad * 2):.1f},"
            f"{height - pad - (point['nav'] - low) / span * (height - pad * 1.55):.1f}"
            for i, point in enumerate(series)
        )
        paths.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="1.7"/>')
        legend.append(
            f'<rect x="{pad + index * 140}" y="10" width="9" height="9" fill="{color}"/>'
            f'<text x="{pad + index * 140 + 14}" y="18" font-size="10" fill="#34495e">{name}</text>'
        )
    return f"""<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="净值">
{''.join(paths)}{''.join(legend)}
<text x="{pad}" y="{height - 10}" font-size="9.5" fill="#7f8c8d">净值 {low:.2f} – {high:.2f}</text>
</svg>"""


def _short_period(label: str) -> str:
    head = label.split(" ")[0]
    if "–" in head:
        start, end = head.split("–")
        return f"{start[2:]}–{end[2:]}"
    return head


def _svg_periods(periods: dict[str, dict[str, Any]], *, width: int = 920, height: int = 330) -> str:
    labels = list(periods)
    tags = [("BASE", "#c0392b"), ("EW", "#7f8c8d"), ("6040", "#2980b9")]
    values = [row[tag]["return"] for row in periods.values() for tag, _ in tags if tag in row]
    low, high = min(values + [0.0]), max(values + [0.0])
    span = max(high - low, 1.0)
    top, bottom = 40, height - 62

    def y(value: float) -> float:
        return bottom - (value - low) / span * (bottom - top)

    slot = (width - 110) / max(len(labels), 1)
    bar = slot / (len(tags) + 0.6)
    rects = []
    for index, (tag, color) in enumerate(tags):
        for position, label in enumerate(labels):
            value = periods[label].get(tag, {}).get("return")
            if value is None:
                continue
            x = 70 + position * slot + index * bar
            y0, y1 = y(value), y(0.0)
            rects.append(
                f'<rect x="{x:.1f}" y="{min(y0, y1):.1f}" width="{bar * 0.8:.1f}" '
                f'height="{max(abs(y1 - y0), 0.8):.1f}" fill="{color}" opacity="0.88"/>'
            )
    ticks = "".join(
        f'<line x1="64" y1="{y(value):.1f}" x2="{width - 16}" y2="{y(value):.1f}" stroke="#eef1f5"/>'
        f'<text x="58" y="{y(value) + 3:.1f}" font-size="9.5" text-anchor="end" fill="#95a5a6">{value:.0f}%</text>'
        for value in (low + span * i / 4 for i in range(5))
    )
    names = "".join(
        f'<text x="{70 + i * slot + slot / 2 - bar * 0.3:.1f}" y="{height - 34}" font-size="9.5" '
        f'text-anchor="middle" fill="#34495e">{_short_period(label)}</text>'
        for i, label in enumerate(labels)
    )
    legend = "".join(
        f'<rect x="{70 + i * 120}" y="10" width="9" height="9" fill="{color}"/>'
        f'<text x="{70 + i * 120 + 14}" y="18" font-size="10" fill="#34495e">{name}</text>'
        for i, (name, color) in enumerate((("策略", "#c0392b"), ("等权", "#7f8c8d"), ("60/40", "#2980b9")))
    )
    return f"""<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="分区间收益">
{ticks}<line x1="64" y1="{y(0):.1f}" x2="{width - 16}" y2="{y(0):.1f}" stroke="#bdc3c7"/>
{''.join(rects)}{legend}{names}</svg>"""


def render_markdown(study: dict[str, Any], meta: dict[str, Any]) -> str:
    results, benchmarks, verdict = study["results"], study["benchmarks"], study["verdict"]
    table = "\n".join(
        f"| {row['metrics']['name']} | {row['metrics']['cagr']:+.2f}% | {row['metrics']['vol']:.2f}% | "
        f"{row['metrics']['sharpe']:.2f} | {row['metrics']['max_drawdown']:.2f}% | {row['metrics']['calmar']:.2f} | "
        f"{row['metrics']['annual_turnover']:.0f}% | {row['metrics']['cost_bps']:.0f}bp | "
        f"{row['metrics']['cash_mean']:.1f}% |"
        for row in list(results.values()) + list(benchmarks.values())
    )
    period_rows = "\n".join(
        f"| {label} | {_signed(row.get('BASE', {}).get('return'))}% / {_num(row.get('BASE', {}).get('max_drawdown'))}% | "
        f"{_signed(row.get('EW', {}).get('return'))}% / {_num(row.get('EW', {}).get('max_drawdown'))}% | "
        f"{_signed(row.get('6040', {}).get('return'))}% / {_num(row.get('6040', {}).get('max_drawdown'))}% |"
        for label, row in study["periods"].items()
    )
    universe_rows = "\n".join(
        f"| {symbol} | {info['name']} | {info['class']} | {info['note']} |" for symbol, info in UNIVERSE.items()
    )
    return f"""# 跨资产 ETF 时间序列配置 · 回测报告

标的：{len(UNIVERSE)} 只 ETF + 现金（{CASH}），区间 {meta['start']} → {meta['end']}，共 {len(study['calendar'])} 个交易日。
执行：收盘出信号、次日开盘成交；成本单边 {BUY_COST * 10000:.1f}bp（ETF 免印花税）。

## 结论

{verdict['sharpe']['text']}；回撤 {verdict['drawdown']['text']}；Calmar {verdict['calmar']['text']}。
下跌年份：{verdict['crash']['text']}。

| 通过标准 | 要求 | 实测 | 判定 |
| --- | --- | --- | --- |
| 净 Sharpe 高于等权 | 是 | {verdict['sharpe']['text']} | {'通过' if verdict['sharpe']['ok'] else '未通过'} |
| 最大回撤明显更低 | ≥5pp | {verdict['drawdown']['text']} | {'通过' if verdict['drawdown']['ok'] else '未通过'} |
| 2018 / 2022 降损 | 均优于等权 | {verdict['crash']['text']} | {'通过' if verdict['crash']['ok'] else '未通过'} |

## 1. 标的池与入选理由

| 代码 | 名称 | 类别 | 理由 / 可替代 |

{universe_rows}

## 2. 规则

- **入场**（四条全满足）：12–1 动量 > 0；收盘 > MA60；收盘 > MA120；距 252 日高点回撤 > −15%。
- **持有**：只看动量 > 0 与收盘 > MA120（MA60 只用于入场），回撤止损仍然有效。
- **权重**：入选腿做 ERC（60 日协方差），单腿 ≤30%，A 股 ≤50%、海外 ≤50%。
- **波动目标**：组合事前波动 >10% 时等比缩到 10%，只降不升。
- **现金**：未分配部分买 511880，按实际收益计息。
- **再平衡**：每月；单腿变动 <3pp 不动；单次单边换手上限 30%。
- **不做参数搜索**：以上阈值均为先验设定，没有在回测上挑过。

## 3. 结果对比

| 方案 | 年化 | 波动 | 净 Sharpe | 最大回撤 | Calmar | 年换手 | 成本 | 平均现金 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
{table}

## 4. 分区间（收益 / 区间最大回撤）

| 区间 | 策略 | 等权 ETF | 60/40 |
| --- | --- | --- | --- |
{period_rows}

## 5. 同风险对照（关键）

策略实现波动只有 {study['results']['BASE']['metrics']['vol']:.2f}%，而等权 ETF 是
{study['benchmarks']['EW']['metrics']['vol']:.2f}%——直接比年化会误导。把基准按 Sharpe
折算到策略的风险预算（同波动等效收益）：

| 方案 | 净 Sharpe | 实际波动 | 同风险等效收益 | 实际年化 |
| --- | --- | --- | --- | --- |
| 策略 | {study['results']['BASE']['metrics']['sharpe']:.2f} | {study['results']['BASE']['metrics']['vol']:.2f}% | {study['results']['BASE']['metrics']['cagr']:.2f}% | {study['results']['BASE']['metrics']['cagr']:.2f}% |
| 等权 ETF | {study['benchmarks']['EW']['metrics']['sharpe']:.2f} | {study['benchmarks']['EW']['metrics']['vol']:.2f}% | {study['matched_vol']['EW']:.2f}% | {study['benchmarks']['EW']['metrics']['cagr']:.2f}% |
| 60/40 | {study['benchmarks']['6040']['metrics']['sharpe']:.2f} | {study['benchmarks']['6040']['metrics']['vol']:.2f}% | {study['matched_vol']['6040']:.2f}% | {study['benchmarks']['6040']['metrics']['cagr']:.2f}% |
| 沪深300 | {study['benchmarks']['HOLD300']['metrics']['sharpe']:.2f} | {study['benchmarks']['HOLD300']['metrics']['vol']:.2f}% | {study['matched_vol']['HOLD300']:.2f}% | {study['benchmarks']['HOLD300']['metrics']['cagr']:.2f}% |

## 6. 结构诊断

- 平均入选 {study['structure']['avg_legs']} 条腿（最少 {study['structure']['min_legs']} 条），共 {study['structure']['rebalances']} 次调仓，
  其中 {study['structure']['full_cash_rebalances']} 次几乎全现金。
- **10% 的波动目标没有触发**：12 条低相关腿做 ERC 后组合波动天然只有 5% 左右，
  `目标/实际 = {study['structure']['leverage_needed']}×`，长-only 不加杠杆就上不去。
  这是"分散化的数学"，不是参数没调好。
- 若允许 {study['structure']['leverage_needed']}× 杠杆（股指期货/融资），按几何近似
  年化约 {study['structure']['projected']['cagr']:.1f}%、回撤约 {study['structure']['projected']['max_drawdown']:.1f}%——
  再扣融资成本（0.9× 约 1.5%/年）后约 11%–12%。这是一个 10% 波动产品的可行形态。

## 7. 为什么不用横截面 IC

这套策略的标的只有 {len(UNIVERSE)} 个，日频横截面 IC 的有效 bet 就是十几个，且大部分标的
长期同向（A 股内部相关 0.7–0.9），IC 的抽样误差远大于信号本身。跨资产配置的评价口径
换成**组合净值**：净 Sharpe、最大回撤、Calmar、分区间损失。这些量既不需要横截面离散度，
也是资金真正定价的东西。
"""


def render_html(study: dict[str, Any], meta: dict[str, Any]) -> str:
    results, benchmarks, verdict = study["results"], study["benchmarks"], study["verdict"]
    rows = "".join(
        f"<tr><td>{row['metrics']['name']}</td><td>{row['metrics']['cagr']:+.2f}%</td>"
        f"<td>{row['metrics']['vol']:.2f}%</td><td>{row['metrics']['sharpe']:.2f}</td>"
        f"<td>{row['metrics']['max_drawdown']:.2f}%</td><td>{row['metrics']['calmar']:.2f}</td>"
        f"<td>{row['metrics']['annual_turnover']:.0f}%</td><td>{row['metrics']['cost_bps']:.0f}bp</td>"
        f"<td>{row['metrics']['cash_mean']:.1f}%</td></tr>"
        for row in list(results.values()) + list(benchmarks.values())
    )
    period_rows = "".join(
        f"<tr><td>{label}</td>"
        f"<td>{_signed(row.get('BASE', {}).get('return'))}% / {_num(row.get('BASE', {}).get('max_drawdown'))}%</td>"
        f"<td>{_signed(row.get('EW', {}).get('return'))}% / {_num(row.get('EW', {}).get('max_drawdown'))}%</td>"
        f"<td>{_signed(row.get('6040', {}).get('return'))}% / {_num(row.get('6040', {}).get('max_drawdown'))}%</td></tr>"
        for label, row in study["periods"].items()
    )
    universe_rows = "".join(
        f"<tr><td>{symbol}</td><td>{info['name']}</td><td>{info['class']}</td><td>{info['note']}</td></tr>"
        for symbol, info in UNIVERSE.items()
    )
    nav_svg = _svg_nav({
        "策略（BASE）": results["BASE"]["nav"],
        "等权 ETF": benchmarks["EW"]["nav"],
        "60/40": benchmarks["6040"]["nav"],
        "沪深300": benchmarks["HOLD300"]["nav"],
    })
    period_svg = _svg_periods(study["periods"])
    flag = lambda ok: "pass" if ok else "fail"
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>跨资产 ETF 配置 · 回测</title>
<style>
:root{{--ink:#1f2a37;--muted:#6b7785;--line:#e3e8ef;--bg:#f7f9fc;--red:#c0392b;--green:#1e8449;--blue:#1f6fb2}}
*{{box-sizing:border-box}}
body{{margin:0;padding:32px 40px;background:var(--bg);color:var(--ink);font:14px/1.75 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}}
h1{{font-size:23px;margin:0 0 6px}}
h2{{font-size:15px;margin:30px 0 10px;padding-left:10px;border-left:3px solid var(--blue);color:#26364a}}
.sub{{color:var(--muted);font-size:12.5px}}
.card{{background:#fff;border:1px solid var(--line);border-radius:8px;padding:16px 20px;margin-bottom:14px}}
table{{width:100%;border-collapse:collapse;font-size:13px;background:#fff}}
th,td{{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right}}
th:first-child,td:first-child{{text-align:left}}
td:nth-child(4){{text-align:left}}
th{{color:var(--muted);font-weight:600;font-size:12px;background:#fbfcfe}}
.verdict{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px;margin:16px 0}}
.pill{{background:#fff;border:1px solid var(--line);border-radius:8px;padding:10px 14px}}
.pill b{{display:block;font-size:12px;color:var(--muted);margin-bottom:2px}}
.pass{{color:var(--green);font-weight:600}} .fail{{color:var(--red);font-weight:600}}
ul{{margin:0;padding-left:18px}} li{{font-size:12.9px;color:#3d4b5c;margin:3px 0}}
</style></head><body>
<h1>跨资产 ETF 时间序列配置 · 回测</h1>
<div class="sub">{len(UNIVERSE)} 只 ETF + 现金 · {meta['start']} → {meta['end']} · {len(study['calendar'])} 个交易日 ·
月末再平衡 · 成本 {BUY_COST * 10000:.1f}bp</div>

<div class="verdict">
  <div class="pill"><b>净 Sharpe</b><span class="{flag(verdict['sharpe']['ok'])}">{verdict['sharpe']['text']}</span></div>
  <div class="pill"><b>最大回撤</b><span class="{flag(verdict['drawdown']['ok'])}">{verdict['drawdown']['text']}</span></div>
  <div class="pill"><b>2018 / 2022</b><span class="{flag(verdict['crash']['ok'])}">{verdict['crash']['text']}</span></div>
  <div class="pill"><b>Calmar</b><span class="{flag(verdict['calmar']['ok'])}">{verdict['calmar']['text']}</span></div>
</div>

<h2>1. 净值</h2>
<div class="card">{nav_svg}</div>

<h2>2. 结果对比</h2>
<div class="card"><table>
<tr><th>方案</th><th>年化</th><th>波动</th><th>净 Sharpe</th><th>最大回撤</th><th>Calmar</th><th>年换手</th><th>成本</th><th>平均现金</th></tr>
{rows}</table></div>

<h2>3. 分区间收益</h2>
<div class="card">{period_svg}<table style="margin-top:10px">
<tr><th>区间</th><th>策略 收益/回撤</th><th>等权 收益/回撤</th><th>60/40 收益/回撤</th></tr>
{period_rows}</table></div>

<h2>4. 标的池</h2>
<div class="card"><table>
<tr><th>代码</th><th>名称</th><th>类别</th><th>入选理由 / 可替代</th></tr>
{universe_rows}</table></div>

<h2>5. 规则</h2>
<div class="card"><ul>
<li><b>入场</b>（四条全满足）：12–1 动量 &gt; 0；收盘 &gt; MA60；收盘 &gt; MA120；距 252 日高点回撤 &gt; −15%。</li>
<li><b>持有</b>：只看动量 &gt; 0 与收盘 &gt; MA120（MA60 只用于入场），回撤止损仍有效——入场要确认、离场要容忍震荡。</li>
<li><b>权重</b>：入选腿做 ERC（60 日协方差），单腿 ≤30%，A 股合计 ≤50%、海外合计 ≤50%。</li>
<li><b>波动目标</b>：组合事前波动 &gt;10% 时等比缩到 10%，只降不升（不加杠杆）。</li>
<li><b>现金</b>：未分配部分买 511880，按实际收益计息，不假设 0 收益。</li>
<li><b>再平衡</b>：每月；单腿变动 &lt;3pp 不动；单次单边换手上限 30%。</li>
<li><b>不做参数搜索</b>：阈值全部先验设定。</li>
</ul></div>

<h2>6. 同风险对照</h2>
<div class="card"><table>
<tr><th>方案</th><th>净 Sharpe</th><th>实际波动</th><th>同风险等效收益</th><th>实际年化</th></tr>
<tr><td>策略</td><td>{results['BASE']['metrics']['sharpe']:.2f}</td><td>{results['BASE']['metrics']['vol']:.2f}%</td>
<td>{results['BASE']['metrics']['cagr']:.2f}%</td><td>{results['BASE']['metrics']['cagr']:.2f}%</td></tr>
<tr><td>等权 ETF</td><td>{benchmarks['EW']['metrics']['sharpe']:.2f}</td><td>{benchmarks['EW']['metrics']['vol']:.2f}%</td>
<td>{study['matched_vol']['EW']:.2f}%</td><td>{benchmarks['EW']['metrics']['cagr']:.2f}%</td></tr>
<tr><td>60/40</td><td>{benchmarks['6040']['metrics']['sharpe']:.2f}</td><td>{benchmarks['6040']['metrics']['vol']:.2f}%</td>
<td>{study['matched_vol']['6040']:.2f}%</td><td>{benchmarks['6040']['metrics']['cagr']:.2f}%</td></tr>
<tr><td>沪深300</td><td>{benchmarks['HOLD300']['metrics']['sharpe']:.2f}</td><td>{benchmarks['HOLD300']['metrics']['vol']:.2f}%</td>
<td>{study['matched_vol']['HOLD300']:.2f}%</td><td>{benchmarks['HOLD300']['metrics']['cagr']:.2f}%</td></tr>
</table>
<p class="sub">策略波动只要 {results['BASE']['metrics']['vol']:.2f}%，等权是 {benchmarks['EW']['metrics']['vol']:.2f}%——
直接比年化会误导，所以把基准按 Sharpe 折算到同一风险预算。</p></div>

<h2>7. 结构诊断与杠杆投影</h2>
<div class="card"><ul>
<li>平均入选 <b>{study['structure']['avg_legs']}</b> 条腿（最少 {study['structure']['min_legs']} 条），
{study['structure']['rebalances']} 次调仓中 {study['structure']['full_cash_rebalances']} 次几乎全现金。</li>
<li><b>10% 波动目标没有触发</b>：12 条低相关腿做 ERC 后组合波动天然只有 5% 左右，
需要 {study['structure']['leverage_needed']}× 杠杆才能到 10%——长-only 不加杠杆上不去，这是分散化的数学。</li>
<li>若允许 {study['structure']['leverage_needed']}× 杠杆：几何近似年化约
{study['structure']['projected']['cagr']:.1f}%、回撤约 {study['structure']['projected']['max_drawdown']:.1f}%，
扣融资成本（0.9× 约 1.5%/年）后约 11%–12%。</li>
</ul></div>

<h2>8. 为什么不用横截面 IC</h2>
<div class="card"><p class="sub">标的只有 {len(UNIVERSE)} 个，日频横截面 IC 的有效 bet 就是十几个，
且 A 股内部相关 0.7–0.9，IC 的抽样误差远大于信号本身。跨资产配置用组合净值评价：
净 Sharpe、最大回撤、Calmar、分区间损失——这些量不需要横截面离散度，也是资金真正定价的东西。</p></div>
</body></html>"""


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="跨资产 ETF 时间序列配置：回测 + 当前配置快照")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=None)
    parser.add_argument("--as-of", default=None, help="快照的信号日（缺省用数据最后一天）")
    parser.add_argument("--outdir", default=str(REPORT))
    parser.add_argument("--no-paper", action="store_true", help="只生成快照，不写纸面跟踪台账")
    parser.add_argument("--latest-only", action="store_true", help="跳过回测，只刷新 latest.json")
    args = parser.parse_args(argv)

    print("[1/4] 载入 ETF 面板…", flush=True)
    panel = load_panel()
    stale = stale_symbols(panel)
    for symbol, data in panel.items():
        rows = data["rows"]
        flag = f"  ← 过期({stale[symbol]})" if symbol in stale else ""
        print(f"      {symbol} {data.get('name', ''):<12} {len(rows):>4} 根  "
              f"{rows[0]['date']} → {rows[-1]['date']}{flag}")
    print(f"      数据版本 {data_version(panel)}", flush=True)

    outdir = Path(args.outdir)
    study: dict[str, Any] | None = None
    backtest: dict[str, Any] = {}
    if not args.latest_only:
        print("[2/4] 回测…", flush=True)
        study = run_study(panel, start=args.start, end=args.end)
        meta = {"start": study["calendar"][0], "end": study["calendar"][-1]}
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "report.md").write_text(render_markdown(study, meta), encoding="utf-8")
        (outdir / "report.html").write_text(render_html(study, meta), encoding="utf-8")
        slim = {
            "meta": meta,
            "data_version": data_version(panel),
            "rule_version": RULE_VERSION,
            "matched_vol": study["matched_vol"],
            "structure": study["structure"],
            "results": {name: row["metrics"] for name, row in study["results"].items()},
            "benchmarks": {name: row["metrics"] for name, row in study["benchmarks"].items()},
            "periods": study["periods"],
            "verdict": study["verdict"],
            "stale": stale,
        }
        (outdir / "metrics.json").write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")
        backtest = {
            "start": meta["start"],
            "end": meta["end"],
            "metrics": study["results"]["BASE"]["metrics"],
            "benchmarks": {name: row["metrics"] for name, row in study["benchmarks"].items()},
            "verdict": study["verdict"],
            "matched_vol": study["matched_vol"],
            "structure": study["structure"],
        }
    else:
        metrics_path = outdir / "metrics.json"
        if metrics_path.is_file():
            saved = json.loads(metrics_path.read_text(encoding="utf-8"))
            backtest = {
                "start": (saved.get("meta") or {}).get("start"),
                "end": (saved.get("meta") or {}).get("end"),
                "metrics": (saved.get("results") or {}).get("BASE") or {},
                "benchmarks": saved.get("benchmarks") or {},
                "verdict": saved.get("verdict") or {},
                "matched_vol": saved.get("matched_vol") or {},
                "structure": saved.get("structure") or {},
            }

    print("[3/4] 当前配置快照…", flush=True)
    previous_records = read_paper()
    previous = previous_records[-1] if previous_records else None
    snapshot = compute_snapshot(panel, as_of=args.as_of, start=args.start)
    print(f"      信号日 {snapshot['signal_date']}　执行 {snapshot['fill_date'] or '下一交易日开盘'}"
          f"　入选 {len(snapshot['holdings'])} 腿 / 暴露 {snapshot['exposure'] * 100:.1f}%", flush=True)
    if snapshot["constraints"]["violations"]:
        print("      约束检查未通过：" + "；".join(snapshot["constraints"]["violations"]), flush=True)

    paper_record = {"recorded": False, "reason": "not_attempted"}
    if not args.no_paper:
        paper_record = record_paper(snapshot, calendar=_calendar(panel, start=args.start))
        print(f"      纸面跟踪：{'已记录' if paper_record.get('recorded') else paper_record.get('reason')}"
              f"（共 {paper_record.get('records')} 条）", flush=True)
    paper = paper_summary(panel, backtest=backtest, start=args.start)

    print("[4/4] 写产物…", flush=True)
    latest = write_latest(snapshot, outdir=outdir, backtest=backtest, paper=paper, previous=previous)
    print(f"      latest.json → {latest}", flush=True)
    if paper.get("available"):
        print(f"      纸面 {paper['n_periods']} 期：净 {paper['cumulative_net_pct']:+.2f}%，"
              f"平均换手 {paper['avg_one_way_turnover']:.3f}，平均成本 {paper['avg_cost_bps']}bp"
              f"（回测均值 {paper['backtest_avg_cost_bps']}bp，一致={paper['cost_consistent']}）", flush=True)
    print(f"\n完成 → {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
