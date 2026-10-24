"""已降级：跨资产 ETF 配置的唯一引擎是 :mod:`etf_allocation`。

这个模块以前是"跨资产 ETF 趋势配置"的第二套实现（12–1 动量 + MA60 + 逆波动 +
回撤减半），与 ``etf_allocation.py``（12–1 动量 + MA60/MA120 + ERC + 止损 + 波动
目标）在信号、权重、成本上各写一遍，导致同一件事有两个口径：

- ``validate_fields`` 验的是 crossasset 那套（7 条腿、逆波动、回撤减半），
  而页面要展示的是 etf_allocation 那套（12 条腿、ERC、单腿 30% 上限）；
- 两套的成本还不一样（7.6/12.6bp 对 7.5/7.5bp），"净 Sharpe"因此不可比。

现在只剩一个引擎。这里保留的只是**兼容壳**：名字还在、语义不再独立，
所有函数都转调 ``etf_allocation``，并在首次调用时抛 ``DeprecationWarning``。
新代码请直接 ``import etf_allocation``。

旧名 → 新实现的对应关系：

===========================  ==========================================
``crossasset`` 旧名           ``etf_allocation`` 现实现
===========================  ==========================================
``UNIVERSE`` / ``CASH``      同一份 12 腿定义 + 511880 现金腿
``stale_symbols``            ``etf_allocation.stale_symbols``
``build_target``             ``etf_allocation.build_target``（throttle /
                             overlay 以乘数叠加在目标权重上）
``run``                      ``etf_allocation.build_target`` + ``simulate``
``equal_weight``             ``etf_allocation.equal_weight_target``
``nav_metrics``              ``etf_allocation.nav_metrics``
``evaluate``                 ``etf_allocation.evaluate``
===========================  ==========================================
"""

from __future__ import annotations

import statistics
import warnings
from typing import Any, Callable, Sequence

import etf_allocation as ea
import execution as ex

UNIVERSE = ea.UNIVERSE
CASH = ea.CASH
BUY_COST = ea.BUY_COST
SELL_COST = ea.SELL_COST
TARGET_VOL = ea.Config().target_vol
_OLD_UNIVERSE_LEGS = ("510300", "510500", "511010", "518880", "510170", "513100", "159920")


def _deprecated(what: str) -> None:
    warnings.warn(
        f"crossasset.{what} 已降级为 etf_allocation 的兼容壳，新代码请直接用 etf_allocation",
        DeprecationWarning,
        stacklevel=3,
    )


def load_panel(cache: Any = None, *, symbols: Sequence[str] | None = None,
               offline: bool = False) -> dict[str, dict[str, Any]]:
    """兼容旧签名；实际读盘与缺失处理都交给 ``etf_allocation.load_panel``。"""
    _deprecated("load_panel")
    if symbols:
        try:
            return ea.load_panel(tuple(symbols))
        except FileNotFoundError:
            if offline:
                return {}
            raise
    return ea.load_panel()


def stale_symbols(panel: dict[str, dict[str, Any]], *, tolerance: int = ea.STALE_TOLERANCE) -> dict[str, str]:
    return ea.stale_symbols(panel, tolerance=tolerance)


def nav_metrics(nav: Sequence[dict[str, Any]], *, risk_free: float = 0.0) -> dict[str, float]:
    """保留旧字段（sortino / days / risk_free），数值一律由引擎算。"""
    _deprecated("nav_metrics")
    stats = dict(ea.nav_metrics(nav))
    values = [float(point["nav"]) for point in nav]
    if len(values) > 1:
        returns = [values[i] / values[i - 1] - 1 for i in range(1, len(values)) if values[i - 1] > 0]
        downside = [r for r in returns if r < 0]
        mean = statistics.fmean(returns) * 252 if returns else 0.0
        dev = (statistics.fmean([r * r for r in downside]) ** 0.5 * (252 ** 0.5)) if downside else 0.0
        stats["sortino"] = round((mean - risk_free) / dev, 3) if dev > 1e-12 else 0.0
        stats["days"] = len(values) - 1
    return stats


def build_target(
    panel: dict[str, dict[str, Any]],
    calendar: Sequence[str],
    *,
    throttle_by_date: dict[str, float] | None = None,
    overlay_by_date: dict[str, float] | None = None,
    target_vol: float = TARGET_VOL,
    max_assets: int | None = None,
) -> Callable[[str, list[str]], dict[str, float]]:
    """旧签名的目标权重函数；旧的两套"暴露乘数"以乘法叠加在引擎权重上。

    ``throttle_by_date`` / ``overlay_by_date`` 只保留接口：新引擎的暴露控制来自
    单腿止损与波动目标，不再有"影子净值回撤减半"。传进来的乘数仍然生效，
    未分配的部分进现金腿。
    """
    _deprecated("build_target")
    config = ea.Config(target_vol=target_vol)
    base = ea.build_target(panel, calendar, config=config, stale=ea.stale_symbols(panel))

    def target(date: str, available: list[str]) -> dict[str, float]:
        weights = base(date, available)
        exposure = 1.0
        if throttle_by_date is not None:
            exposure *= throttle_by_date.get(date, 1.0)
        if overlay_by_date is not None:
            exposure *= overlay_by_date.get(date, 1.0)
        if exposure >= 1.0:
            return weights
        scaled = {symbol: value * exposure for symbol, value in weights.items() if symbol != CASH}
        cash = max(0.0, 1.0 - sum(scaled.values()))
        if cash > 1e-12:
            scaled[CASH] = cash
        return scaled

    return target


def _calendar(panel: dict[str, dict[str, Any]], start: str | None) -> list[str]:
    return ea._calendar(panel, start=start)


def run(
    panel: dict[str, dict[str, Any]],
    *,
    rebalance: int = 21,
    target_vol: float = TARGET_VOL,
    drawdown_control: bool = True,
    overlay: bool = False,
    max_assets: int | None = None,
    start: str | None = None,
    cost_scale: float = 1.0,
) -> dict[str, Any]:
    """兼容旧签名：回测一遍并返回 ``nav / metrics / excluded / throttle_days / settings``。"""
    _deprecated("run")
    stale = ea.stale_symbols(panel)
    calendar = _calendar(panel, start)
    config = ea.Config(rebalance=rebalance, target_vol=target_vol)
    target = ea.build_target(panel, calendar, config=config, start=start, stale=stale)
    result = ea.simulate(panel, calendar, target, config=config,
                         cost_buy=ea.BUY_COST * cost_scale, cost_sell=ea.SELL_COST * cost_scale)
    summary = ea.summarize(result, calendar, "ETF 配置（etf_allocation）")
    metrics = dict(summary["metrics"])
    return {
        "nav": summary["nav"],
        "holdings": result.get("holdings") or [],
        "rebalances": result.get("rebalances") or [],
        "total_cost": result.get("total_cost"),
        "total_turnover": result.get("total_turnover"),
        "metrics": metrics,
        "excluded": stale,
        "throttle_days": 0,          # 新引擎没有"影子回撤减半"，这里恒为 0
        "settings": {
            "rebalance": rebalance,
            "target_vol": target_vol,
            "drawdown_control": False,   # 旧开关已由单腿止损替代
            "regime_overlay": False,
            "max_assets": max_assets,
            "throttle_days": 0,
            "cost_bps": [ea.BUY_COST * cost_scale * 10000, ea.SELL_COST * cost_scale * 10000],
            "engine": "etf_allocation",
        },
    }


def equal_weight(
    panel: dict[str, dict[str, Any]],
    *,
    rebalance: int = 21,
    start: str | None = None,
    cost_scale: float = 1.0,
) -> dict[str, Any]:
    _deprecated("equal_weight")
    calendar = _calendar(panel, start)
    target = ea.equal_weight_target(calendar, panel, start=start)
    result = ea.simulate(panel, calendar, target,
                         cost_buy=ea.BUY_COST * cost_scale, cost_sell=ea.SELL_COST * cost_scale)
    summary = ea.summarize(result, calendar, "同池等权")
    return {"nav": summary["nav"], "metrics": summary["metrics"],
            "total_cost": result.get("total_cost")}


def buy_and_hold(panel: dict[str, dict[str, Any]], symbol: str = "510300",
                 *, start: str | None = None) -> dict[str, Any]:
    _deprecated("buy_and_hold")
    rows = [row for row in panel[symbol]["rows"] if not start or row["date"] >= start]
    nav = [{"date": row["date"], "nav": round(float(row["close"]) / float(rows[0]["close"]), 4)}
           for row in rows]
    return {"nav": nav, "metrics": ea.nav_metrics(nav), "total_cost": 0.0}


def evaluate(panel: dict[str, dict[str, Any]], *, start: str, rebalance: int = 21) -> dict[str, Any]:
    """兼容壳：真正的对照在 ``etf_allocation.evaluate``。

    旧键名到新实现的映射写在返回值的 ``mapping`` 里——旧报告里的 "no_control"
    （去掉回撤减半）在新引擎里对应 "no_stop"（去掉单腿止损），不是同一个概念，
    所以这里不改名、只标注，避免读者把两套结论混起来。
    """
    _deprecated("evaluate")
    result = ea.evaluate(panel, start=start, rebalance=rebalance)
    return {
        "deprecated": True,
        "engine": "etf_allocation",
        "mapping": {"no_control": "no_stop", "overlay": "no_voltarget"},
        "start": result["start"],
        "rebalance": result["rebalance"],
        "strategy": result["strategy"],
        "no_control": result["no_stop"],
        "overlay": result["no_voltarget"],
        "expensive": result["expensive"],
        "equal_weight": result["equal_weight"],
        "buy_hold": result["buy_hold"],
        "table": result["table"],
        "nav": result["nav"],
        "pass": result["pass"],
        "reasons": result["reasons"],
    }


def main(argv: list[str] | None = None) -> int:
    _deprecated("main")
    print("crossasset 已降级：请运行 python -m etf_allocation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
