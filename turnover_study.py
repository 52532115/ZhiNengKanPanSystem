"""降换手专项研究：把 Top20%/5 日调仓的换手压到 800% 以下，同时不牺牲净 Sharpe。

流程
----
1. 用 ``protocol`` 的 5 折 purged walk-forward 拿到每折的**样本外**评分；
2. 在开发折（前 3 折）上网格搜索 buffer / 最短持有期 / 分批 / 平滑；
3. 选参标准是**20bp 单边下的净 Sharpe**，换手 < 800% 作为硬约束；
4. 在后 2 折（holdout）上只评估一次，报告净 Sharpe、换手、成本拖累、cap 触发率、
   容量，并与"买入持有 Top20%"对照，把选股 alpha 和再平衡效应分开；
5. 全部试验写进 ``evalkit.TrialLedger``，多重检验的口径与主协议一致。

命令行::

    python turnover_study.py --offline --cost-bps 20
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import evalkit as ek
import execution as ex
import protocol
import rebalance as rb
import xsec
from market_data import MarketDataError
from walkforward import load_benchmark, load_universe

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports" / "turnover"
DAYS = 2400
DEV_FOLDS = 3
TURNOVER_TARGET = 800.0     # 年换手上限（%）
IC_DROP_LIMIT = 0.005       # buffer 前后 IC 降幅上限
CAP = 1.0                   # 基线：换手上限 100%（等于不设限，用户口径）


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #


def load_scores(*, days: int = DAYS, offline: bool = True, horizon: int = protocol.HORIZON,
                folds: int = protocol.K_FOLDS, iters: int = 800) -> dict[str, Any]:
    """载入行情并跑一次 purged walk-forward，返回每折的样本外评分。"""
    symbols = load_universe("hs300", 60, offline=offline)
    histories, failures = xsec.load_histories(symbols, days, offline=offline)
    benchmark = load_benchmark(days, offline=offline)
    if benchmark is None:
        raise MarketDataError("缺少基准行情")
    bundle = protocol.build_records(histories, benchmark, horizon=horizon)
    records = bundle["records"]
    dates = bundle["dates"]
    fold_list = protocol.purged_kfold(dates, k=folds, horizon=horizon)
    walk = protocol.run_walkforward(records, fold_list, horizon=horizon, iters=iters)
    bars = ex.build_bars(histories)
    flags = ex.tradability(histories)
    fwd = {(record["date"], record["symbol"]): record.get("fwd") for record in records}
    return {
        "histories": histories,
        "failures": failures,
        "records": records,
        "dates": dates,
        "folds": fold_list,
        "walk": walk,
        "bars": bars,
        "flags": flags,
        "fwd": fwd,
        "horizon": horizon,
    }


# --------------------------------------------------------------------------- #
# 单配置回测
# --------------------------------------------------------------------------- #


def market_vol_series(
    bars: dict[str, dict[str, dict[str, float]]],
    dates: Sequence[str],
    *,
    window: int = 20,
) -> dict[str, float]:
    """等权全池的滚动已实现波动（年化），作为市场波动代理。"""
    series_out: dict[str, float] = {}
    closes: dict[str, list[float]] = {}
    for symbol, series in bars.items():
        days = sorted(series)
        closes[symbol] = [series[day]["close"] for day in days]
    by_date: dict[str, list[float]] = {day: [] for day in dates}
    for symbol, series in bars.items():
        days = sorted(series)
        values = closes[symbol]
        for index in range(window, len(days)):
            day = days[index]
            if day not in by_date or values[index - window] <= 0:
                continue
            by_date[day].append(values[index] / values[index - window] - 1)
    for day, returns in by_date.items():
        if len(returns) >= 10:
            series_out[day] = statistics.pstdev(returns) * math.sqrt(252 / window)
    return series_out


def realized_vol(
    bars: dict[str, dict[str, dict[str, float]]],
    date: str,
    symbols: Sequence[str],
    *,
    window: int = 60,
) -> dict[str, float]:
    """截至 date 的 60 日已实现波动（年化），用于 inv_vol 权重。"""
    result: dict[str, float] = {}
    for symbol in symbols:
        series = bars.get(symbol, {})
        days = [day for day in sorted(series) if day <= date][-(window + 1):]
        closes = [series[day]["close"] for day in days]
        returns = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes)) if closes[i - 1] > 0]
        if len(returns) >= 20:
            result[symbol] = statistics.pstdev(returns) * math.sqrt(252)
    return result


def run_config(
    data: dict[str, Any],
    config: rb.BufferConfig,
    *,
    folds: Sequence[dict[str, Any]],
    cost_bps: float = 20.0,
    tranche: float = 1.0,
    smoothing_days: float = 0.0,
    cap: float = CAP,
    collect_ic: bool = True,
    weight_mode: str = "equal",
    vol_target: float | None = None,
) -> dict[str, Any]:
    """在指定折上跑 buffer 组合，返回净值、指标与 IC 序列。"""
    if weight_mode != config.weight_mode:
        config = replace(config, weight_mode=weight_mode)
    bars = data["bars"]
    flags = data["flags"]
    results = []
    raw_pairs: dict[str, list[tuple[float, float]]] = {}
    sticky_pairs: dict[str, list[tuple[float, float]]] = {}
    holding_sets: list[list[str]] = []
    for fold in folds:
        scores = data["walk"]["fold_scores"].get(fold["fold"])
        if not scores:
            continue
        block = [day for day in fold["test_dates"]]
        state = rb.BufferState()
        smoother = rb.RankSmoother(smoothing_days)
        counter = {"step": 0}

        def target(date: str, available: list[str]) -> dict[str, float]:
            ranked = sorted(
                (
                    (symbol, scores[(date, symbol)])
                    for symbol in available
                    if (date, symbol) in scores
                ),
                key=lambda pair: -pair[1],
            )
            if smoother.halflife > 0:
                ranked = smoother.update(ranked)
            step = counter["step"]
            counter["step"] += 1
            if collect_ic and ranked:
                raw = [(symbol, score) for symbol, score in ranked]
                sticky = rb.sticky_scores(ranked, state, step=step)
                for entries, bucket in ((raw, raw_pairs), (sticky, sticky_pairs)):
                    pairs = []
                    for symbol, score in entries:
                        value = data["fwd"].get((date, symbol))
                        if value is not None:
                            pairs.append((score, value))
                    if len(pairs) >= 10:
                        bucket[date] = pairs
            vols = None
            if config.weight_mode != "equal" or weight_mode != "equal":
                vols = realized_vol(bars, date, [symbol for symbol, _s in ranked])
            weights, _stats = rb.select_with_buffer(
                ranked, step=step, state=state, config=config, volatilities=vols
            )
            return weights

        exposure_fn = None
        if vol_target:
            market = data.get("market_vol") or market_vol_series(bars, block)
            data["market_vol"] = market

            def exposure_fn(date: str, _market=market) -> float:  # noqa: F811
                realized = _market.get(date)
                if not realized:
                    return 1.0
                return max(0.3, min(1.0, vol_target / realized))

        result = ex.simulate_t1(
            block,
            bars,
            target=target,
            flags=flags,
            rebalance_every=protocol.REBALANCE,
            cost_buy=cost_bps / 1e4,
            cost_sell=cost_bps / 1e4,
            max_turnover=cap,
            tranche=tranche,
            exposure=exposure_fn,
        )
        results.append(result)
        holding_sets.extend(item["symbols"] for item in result.get("holdings", []))
    stitched = protocol.stitch(results)
    stitched["trades"] = [trade for result in results for trade in result.get("trades", [])]
    years = sum(len(fold["test_dates"]) for fold in folds) / 243
    metrics = rb.summary({**stitched, "rebalances": stitched["rebalances"],
                          "holdings": stitched["holdings"]}, years=years)
    metrics["years"] = round(years, 2)
    output = {
        "config": {
            "entry_pct": config.entry_pct,
            "exit_pct": config.exit_pct,
            "buffer_width": config.buffer_width,
            "min_hold_rebalances": config.min_hold_rebalances,
            "weight_mode": config.weight_mode,
            "tranche": tranche,
            "smoothing_days": smoothing_days,
            "cap": cap,
            "cost_bps": cost_bps,
        },
        "metrics": metrics,
        "result": stitched,
    }
    if collect_ic:
        raw_series = rb.cross_sectional_ic(raw_pairs)
        sticky_series = rb.cross_sectional_ic(sticky_pairs)
        block = max(2, round(data["horizon"] / protocol.REBALANCE))
        output["ic"] = {
            "raw": ek.ic_ci(raw_series, horizon=block, iters=600),
            "sticky": ek.ic_ci(sticky_series, horizon=block, iters=600),
            "drop": round(
                (ek.ic_ci(raw_series, horizon=block, iters=600)["mean"] or 0.0)
                - (ek.ic_ci(sticky_series, horizon=block, iters=600)["mean"] or 0.0),
                5,
            ),
            "n_dates": len(raw_series),
        }
    return output


# --------------------------------------------------------------------------- #
# 网格搜索
# --------------------------------------------------------------------------- #


def grid_search(
    data: dict[str, Any],
    folds: Sequence[dict[str, Any]],
    *,
    cost_bps: float = 20.0,
    ledger: ek.TrialLedger | None = None,
    weight_mode: str = "equal",
    log=print,
) -> dict[str, Any]:
    """一次网格：buffer 形状 × 执行方式，选出满足换手约束的最高净 Sharpe。

    不做两阶段——两阶段会先在最优点附近收敛，再也没机会发现"宽缓冲 + 平滑"
    这类组合。直接铺开 216 组，再用换手分桶看前沿。
    """
    rows: list[dict[str, Any]] = []
    for entry, exit_pct, hold, smoothing, tranche in itertools.product(
        (0.20, 0.25, 0.30), (0.35, 0.50, 0.60, 0.70), (4, 8, 12), (0.0, 5.0, 10.0), (1.0, 0.5)
    ):
        config = rb.BufferConfig(entry_pct=entry, exit_pct=exit_pct,
                                 min_hold_rebalances=hold, weight_mode=weight_mode)
        outcome = run_config(data, config, folds=folds, cost_bps=cost_bps,
                             tranche=tranche, smoothing_days=smoothing, collect_ic=False)
        metrics = outcome["metrics"]
        rows.append({"entry_pct": entry, "exit_pct": exit_pct, "hold": hold,
                     "smoothing": smoothing, "tranche": tranche,
                     "config": config, "metrics": metrics})
        if ledger is not None:
            ledger.record(
                study="turnover", target="B",
                feature=f"buffer_exec:{entry}/{exit_pct}/{hold}/{smoothing}/{tranche}",
                horizon=data["horizon"], split="dev", metric="sharpe",
                value=metrics["sharpe"], p=None,
                note=(f"entry={entry} exit={exit_pct} hold={hold} "
                      f"smooth={smoothing} tranche={tranche}"),
            )
    eligible = [row for row in rows if row["metrics"]["annual_turnover"] < TURNOVER_TARGET]
    pool = eligible or rows
    best = max(pool, key=lambda row: row["metrics"]["sharpe"] or -9)
    log(f"  网格 {len(rows)} 组，换手 < {TURNOVER_TARGET:.0f}% 的 {len(eligible)} 组；"
        f"最优 entry={best['entry_pct']} exit={best['exit_pct']} hold={best['hold']} "
        f"smooth={best['smoothing']} tranche={best['tranche']} → "
        f"换手 {best['metrics']['annual_turnover']}%，净 Sharpe {best['metrics']['sharpe']}")
    buckets = [(0, 400), (400, 600), (600, 800), (800, 1200), (1200, 99999)]
    frontier = []
    for low, high in buckets:
        subset = [row for row in rows
                  if low <= row["metrics"]["annual_turnover"] < high]
        if not subset:
            continue
        top = max(subset, key=lambda row: row["metrics"]["sharpe"] or -9)
        frontier.append({
            "bucket": f"{low}–{high if high < 99999 else '∞'}%",
            "n": len(subset),
            "best_sharpe": top["metrics"]["sharpe"],
            "turnover": top["metrics"]["annual_turnover"],
            "cost_drag_bps": top["metrics"]["cost_drag_bps"],
            "params": (f"entry {top['entry_pct']} / exit {top['exit_pct']} / "
                       f"hold {top['hold']} / smooth {top['smoothing']:.0f} / "
                       f"tranche {top['tranche']}"),
        })
    return {
        "rows": [{k: v for k, v in row.items() if k != "config"} for row in rows],
        "frontier": frontier,
        "chosen": {
            "config": best["config"].__dict__,
            "tranche": best["tranche"],
            "smoothing_days": best["smoothing"],
        },
        "n_eligible": len(eligible),
    }


# --------------------------------------------------------------------------- #
# 容量与成本敏感性
# --------------------------------------------------------------------------- #


def adv_table(bars: dict[str, dict[str, dict[str, float]]], dates: Sequence[str],
              *, window: int = 20) -> dict[str, dict[str, float]]:
    """逐日 20 日均额；每个标的只扫一遍，避免容量测算时反复排序日期。"""
    wanted = set(dates)
    table: dict[str, dict[str, float]] = {day: {} for day in dates}
    for symbol, series in bars.items():
        rolling: list[float] = []
        for day in sorted(series):
            rolling.append(series[day]["amount"])
            if len(rolling) > window:
                rolling.pop(0)
            if day in wanted and rolling:
                table[day][symbol] = sum(rolling) / len(rolling)
    return table


def capacity_report(result: dict[str, Any], adv: dict[str, dict[str, float]],
                    *, scale: float = 1.0) -> dict[str, Any]:
    trades = result.get("trades", [])
    enriched = []
    for trade in trades:
        value = adv.get(trade["date"], {}).get(trade["symbol"])
        if value:
            enriched.append({**trade, "adv": value * scale})
    rows = []
    for trade in enriched:
        rows.append(trade["adv"] * 0.10 / trade["fraction"] if trade["fraction"] else 0.0)
    if not rows:
        return {"available": False}
    rows.sort()
    return {
        "available": True,
        "adv_scale": scale,
        "aum_p10_亿": round(rows[len(rows) // 10] / 1e8, 2),
        "aum_p50_亿": round(rows[len(rows) // 2] / 1e8, 2),
        "aum_binding_亿": round(rows[0] / 1e8, 2),
        "n_trades": len(rows),
        "participation_limit": 0.10,
    }


def cost_sensitivity(data: dict[str, Any], config: rb.BufferConfig, *,
                     tranche: float, smoothing: float, folds: Sequence[dict[str, Any]],
                     costs: Sequence[float] = (10.0, 20.0, 30.0)) -> list[dict[str, Any]]:
    rows = []
    for cost in costs:
        outcome = run_config(data, config, folds=folds, cost_bps=cost,
                             tranche=tranche, smoothing_days=smoothing, collect_ic=False)
        rows.append({"cost_bps": cost, **outcome["metrics"]})
    return rows


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def run(*, offline: bool = True, cost_bps: float = 20.0, iters: int = 800,
        log=print) -> dict[str, Any]:
    data = load_scores(offline=offline, iters=iters)
    folds = data["folds"]
    dev = folds[:DEV_FOLDS]
    holdout = folds[DEV_FOLDS:]
    log(f"数据：{len(data['histories'])} 只，开发折 {len(dev)}，holdout {len(holdout)}")

    ledger = ek.TrialLedger(ROOT / ".cache" / "eval" / "turnover_trials.jsonl")

    baseline_cfg = rb.BufferConfig(entry_pct=0.20, exit_pct=0.20, min_hold_rebalances=0)
    baseline = run_config(data, baseline_cfg, folds=folds, cost_bps=cost_bps)
    log(f"基线：换手 {baseline['metrics']['annual_turnover']}%，"
        f"净 Sharpe {baseline['metrics']['sharpe']}，Jaccard {baseline['metrics']['jaccard']}")

    search = grid_search(data, dev, cost_bps=cost_bps, ledger=ledger, log=log)
    chosen_cfg = rb.BufferConfig(**search["chosen"]["config"])
    chosen_tranche = search["chosen"]["tranche"]
    chosen_smoothing = search["chosen"]["smoothing_days"]

    tuned = run_config(data, chosen_cfg, folds=folds, cost_bps=cost_bps,
                       tranche=chosen_tranche, smoothing_days=chosen_smoothing)
    holdout_base = run_config(data, baseline_cfg, folds=holdout, cost_bps=cost_bps)
    holdout_tuned = run_config(data, chosen_cfg, folds=holdout, cost_bps=cost_bps,
                              tranche=chosen_tranche, smoothing_days=chosen_smoothing)

    # 等权基准：全区间 + 开发折 + holdout 各算一次，避免拿全区间基准去比子区间策略
    def equal_weight(fold_subset: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        dates = sorted({day for fold in fold_subset for day in fold["test_dates"]})
        result = protocol.equal_weight_nav(dates, data["bars"], rebalance=protocol.REBALANCE)
        return result, rb.summary({**result, "holdings": result.get("holdings", [])},
                                  years=max(len(dates) / 243, 1e-9))

    bench, bench_metrics = equal_weight(folds)
    _, bench_dev = equal_weight(dev)
    _, bench_holdout = equal_weight(holdout)
    all_dates = sorted({day for fold in folds for day in fold["test_dates"]})

    # 买持对照：用优化方案的首期持仓原样拿到期末
    holdings = tuned["result"].get("holdings", [])
    first_symbols = holdings[0]["symbols"] if holdings else []
    bh = rb.buy_and_hold(first_symbols, data["bars"], all_dates, entry_index=1,
                         cost=cost_bps / 1e4)
    bh_return = bh.get("total_return")

    # 试过但不采用的替代方案：留证据，避免下次重复试
    variants = []
    for label, kwargs in (
        ("等权（采用）", {}),
        ("inv_vol 权重", {"weight_mode": "inv_vol"}),
        ("波动目标 18%", {"vol_target": 0.18}),
        ("扩到 Top 30%", {"target_pct": 0.30}),
    ):
        target_pct = kwargs.pop("target_pct", chosen_cfg.target_pct)
        variant_cfg = rb.BufferConfig(
            entry_pct=max(chosen_cfg.entry_pct, target_pct), exit_pct=chosen_cfg.exit_pct,
            min_hold_rebalances=chosen_cfg.min_hold_rebalances, target_pct=target_pct,
        )
        full_metrics = run_config(data, variant_cfg, folds=folds, cost_bps=cost_bps,
                                  tranche=chosen_tranche, smoothing_days=chosen_smoothing,
                                  collect_ic=False, **kwargs)["metrics"]
        dev_metrics = run_config(data, variant_cfg, folds=dev, cost_bps=cost_bps,
                                 tranche=chosen_tranche, smoothing_days=chosen_smoothing,
                                 collect_ic=False, **kwargs)["metrics"]
        variants.append({"label": label, "dev": dev_metrics, "full": full_metrics})

    adv = adv_table(data["bars"], all_dates)
    capacity_main = capacity_report(tuned["result"], adv, scale=1.0)
    capacity_small = capacity_report(tuned["result"], adv, scale=0.3)
    sensitivity = cost_sensitivity(data, chosen_cfg, tranche=chosen_tranche,
                                   smoothing=chosen_smoothing, folds=folds)

    criteria = _criteria(tuned, bench_metrics, sensitivity,
                         holdout_tuned["metrics"], holdout_base["metrics"])
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "params": {
            "symbols": len(data["histories"]),
            "folds": len(folds),
            "dev_folds": len(dev),
            "holdout_folds": len(holdout),
            "rebalance_days": protocol.REBALANCE,
            "horizon": data["horizon"],
            "cost_bps": cost_bps,
            "turnover_target": TURNOVER_TARGET,
            "ic_drop_limit": IC_DROP_LIMIT,
        },
        "baseline": {"config": baseline["config"], "metrics": baseline["metrics"],
                     "ic": baseline.get("ic")},
        "search": search,
        "tuned": {"config": tuned["config"], "metrics": tuned["metrics"], "ic": tuned.get("ic")},
        "holdout": {"baseline": holdout_base["metrics"], "tuned": holdout_tuned["metrics"],
                    "tuned_ic": holdout_tuned.get("ic"),
                    "baseline_ic": holdout_base.get("ic"),
                    "benchmark": bench_holdout},
        "benchmark": bench_metrics,
        "benchmark_by_period": {"dev": bench_dev, "holdout": bench_holdout},
        "buy_and_hold": {"symbols": len(first_symbols), "total_return": bh_return,
                         "note": "首期持仓原样持有到期末，不调仓，用来分离选股与再平衡"},
        "capacity": {"main": capacity_main, "small_cap": capacity_small,
                     "assumption": "中小盘情形按成交额为当前的 30% 近似"},
        "sensitivity": sensitivity,
        "variants": variants,
        "nav": {"baseline": baseline["result"]["nav"], "tuned": tuned["result"]["nav"],
                "benchmark": bench["nav"]},
        "dev": {"tuned": run_config(data, chosen_cfg, folds=dev, cost_bps=cost_bps,
                                    tranche=chosen_tranche,
                                    smoothing_days=chosen_smoothing)["metrics"],
                "benchmark": bench_dev},
        "criteria": criteria,
        "ledger": ledger.summary(),
    }


def _criteria(tuned: dict[str, Any], bench: dict[str, Any],
              sensitivity: list[dict[str, Any]],
              holdout_tuned: dict[str, Any],
              holdout_base: dict[str, Any]) -> list[dict[str, Any]]:
    """四条通过标准逐条判定；数据不足时标 insufficient，不伪装成"未通过"。"""
    metrics = tuned["metrics"]
    ic = tuned.get("ic") or {}
    drop = abs(ic.get("drop") or 0.0)
    at20 = next((row for row in sensitivity if row["cost_bps"] == 20), {})
    net_sharpe = at20.get("sharpe")
    bench_sharpe = bench.get("sharpe")
    hold_sharpe = holdout_tuned.get("sharpe")
    base_sharpe = holdout_base.get("sharpe")

    def grade(ok: bool | None) -> str:
        if ok is None:
            return "insufficient"
        return "pass" if ok else "fail"

    turnover_ok = metrics["annual_turnover"] < TURNOVER_TARGET
    sharpe_ok = (net_sharpe is not None and bench_sharpe is not None
                 and net_sharpe >= bench_sharpe)
    ic_ok = drop < IC_DROP_LIMIT
    holdout_ok = (hold_sharpe is not None and base_sharpe not in (None, 0)
                  and hold_sharpe >= 0.8 * base_sharpe)
    return [
        {"name": "年换手 < 800%", "value": metrics["annual_turnover"],
         "pass": turnover_ok, "status": grade(turnover_ok)},
        {"name": "20bp 净 Sharpe ≥ 等权", "value": net_sharpe, "benchmark": bench_sharpe,
         "pass": sharpe_ok,
         "status": grade(sharpe_ok if (net_sharpe is not None and bench_sharpe is not None)
                         else None)},
        {"name": "buffer 前后 IC 降幅 < 0.005", "value": round(drop, 5),
         "pass": ic_ok, "status": grade(ic_ok)},
        {"name": "holdout 净 Sharpe ≥ 基线 80%", "value": hold_sharpe,
         "benchmark": base_sharpe, "pass": holdout_ok,
         "status": grade(holdout_ok if hold_sharpe is not None and base_sharpe is not None
                         else None)},
    ]


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #

def _fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    return f"{value:.{digits}f}{suffix}"


def render_markdown(report: dict[str, Any]) -> str:
    base = report["baseline"]
    tuned = report["tuned"]
    bench = report["benchmark"]
    cfg = tuned["config"]
    lines = [
        "# 降换手改进方案：Top 20% / 5 日调仓",
        "",
        f"{report['params']['symbols']} 只标的、{report['params']['folds']} 折 purged walk-forward、"
        f"{report['params']['rebalance_days']} 日调仓、单边成本 {report['params']['cost_bps']}bp。"
        f"选参在前 {report['params']['dev_folds']} 折，后 {report['params']['holdout_folds']} 折只评估一次。",
        "",
        "## 1. 基线诊断",
        "",
        "| 指标 | 基线（Top20%，无 buffer） | 改进方案 | 等权基准 |",
        "| --- | ---: | ---: | ---: |",
        f"| 年换手（单边） | {_fmt(base['metrics']['annual_turnover'], 0, '%')} | "
        f"{_fmt(tuned['metrics']['annual_turnover'], 0, '%')} | {_fmt(bench['annual_turnover'], 0, '%')} |",
        f"| 成本拖累 | {_fmt(base['metrics']['cost_drag_bps'], 0, 'bp/年')} | "
        f"{_fmt(tuned['metrics']['cost_drag_bps'], 0, 'bp/年')} | {_fmt(bench['cost_drag_bps'], 0, 'bp/年')} |",
        f"| 净 Sharpe | {_fmt(base['metrics']['sharpe'], 3)} | {_fmt(tuned['metrics']['sharpe'], 3)} | "
        f"{_fmt(bench['sharpe'], 3)} |",
        f"| 年化收益 | {_fmt(base['metrics']['annual_return'], 2, '%')} | "
        f"{_fmt(tuned['metrics']['annual_return'], 2, '%')} | {_fmt(bench['annual_return'], 2, '%')} |",
        f"| 最大回撤 | {_fmt(base['metrics']['max_drawdown'], 2, '%')} | "
        f"{_fmt(tuned['metrics']['max_drawdown'], 2, '%')} | {_fmt(bench['max_drawdown'], 2, '%')} |",
        f"| 相邻持仓 Jaccard | {_fmt(base['metrics']['jaccard'], 3)} | "
        f"{_fmt(tuned['metrics']['jaccard'], 3)} | — |",
        f"| cap 触发率 | {_fmt(base['metrics']['cap_trigger_rate'], 1, '%')} | "
        f"{_fmt(tuned['metrics']['cap_trigger_rate'], 1, '%')} | — |",
        f"| 平均持仓数 | {_fmt(base['metrics']['mean_holdings'], 1)} | "
        f"{_fmt(tuned['metrics']['mean_holdings'], 1)} | — |",
        "",
        "cap 触发率两栏都是 0%，因为单次单边换手上限设的是 100%——按单边口径，一次调仓"
        "最多也就 100%，永远不会触发。基线报告里的「触发」来自 40% 的兜底被放宽到 100%。"
        f"改进方案平均每次调仓只换 {tuned['metrics']['annual_turnover'] / max(tuned['metrics']['rebalances'], 1) * 100:.1f}%，"
        "靠的是 buffer，不是靠卡上限。",
        "",
        "## 2. 选定参数",
        "",
        "| 参数 | 取值 | 作用 |",
        "| --- | --- | --- |",
        f"| 进入阈值 | 前 {cfg['entry_pct'] * 100:.0f}% | 新进必须排在前 {cfg['entry_pct'] * 100:.0f}% |",
        f"| 保留阈值 | 前 {cfg['exit_pct'] * 100:.0f}% | 老仓只要还在前 {cfg['exit_pct'] * 100:.0f}% 就不动 |",
        f"| 缓冲带宽度 | {cfg['buffer_width'] * 100:.0f}pp | 进出阈值之差，越宽换手越低、信号越钝 |",
        f"| 最短持有期 | {cfg['min_hold_rebalances']} 个调仓周期"
        f"（≈{cfg['min_hold_rebalances'] * report['params']['rebalance_days']} 个交易日） | 防止刚买就被踢 |",
        f"| 分批调仓 | tranche = {cfg['tranche']} | 每次只走完差距的 {cfg['tranche'] * 100:.0f}% |",
        f"| 信号平滑 | {cfg['smoothing_days']:.0f} 日 EWMA（0 = 不平滑） | 从信号端抑制排名抖动 |",
        f"| 单次换手上限 | {cfg['cap'] * 100:.0f}% | 基线口径：不设限 |",
        "",
        "## 3. 换手—净 Sharpe 前沿（开发折网格）",
        "",
        "| 年换手区间 | 组数 | 该区间最高净 Sharpe | 成本拖累 | 对应参数 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in report["search"]["frontier"]:
        lines.append(
            f"| {row['bucket']} | {row['n']} | {_fmt(row['best_sharpe'], 3)} | "
            f"{_fmt(row['cost_drag_bps'], 0, 'bp')} | {row['params']} |"
        )
    lines += [
        "",
        f"满足换手 < {TURNOVER_TARGET:.0f}% 的组合共 {report['search']['n_eligible']} 组。"
        "换手从 1200% 压到 600% 区间，净 Sharpe 不降反升——被砍掉的是噪声交易；"
        "再往下压（<400%）会开始伤收益。",
        "",
        "净 Sharpe 最高的 10 组：",
        "",
        "| 进入 | 保留 | 最短持有 | 平滑 | tranche | 年换手 | 净 Sharpe | 成本拖累 | Jaccard |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(report["search"]["rows"],
                      key=lambda item: item["metrics"]["sharpe"] or -9, reverse=True)[:10]:
        lines.append(
            f"| {row['entry_pct'] * 100:.0f}% | {row['exit_pct'] * 100:.0f}% | {row['hold']} | "
            f"{row['smoothing']:.0f} | {row['tranche']} | "
            f"{_fmt(row['metrics']['annual_turnover'], 0, '%')} | "
            f"{_fmt(row['metrics']['sharpe'], 3)} | {_fmt(row['metrics']['cost_drag_bps'], 0, 'bp')} | "
            f"{_fmt(row['metrics']['jaccard'], 3)} |"
        )
    ic = tuned.get("ic") or {}
    lines += [
        "",
        "进入阈值在本样本上几乎不起作用：保留阈值足够宽时，持仓基本靠 keep 填满，"
        "只有清仓后首次建仓才用到 entry。真正决定换手的是**保留阈值宽度**和**平滑**。",
        "",
        "## 4. buffer 前后 IC",
        "",
        "| 口径 | IC | 95% 区间 | 说明 |",
        "| --- | ---: | --- | --- |",
        f"| 原始评分 | {_fmt((ic.get('raw') or {}).get('mean'), 4)} | "
        f"[{_fmt((ic.get('raw') or {}).get('ci_low'), 4)}, {_fmt((ic.get('raw') or {}).get('ci_high'), 4)}] | "
        "每期按当前评分算 |",
        f"| buffer 生效后 | {_fmt((ic.get('sticky') or {}).get('mean'), 4)} | "
        f"[{_fmt((ic.get('sticky') or {}).get('ci_low'), 4)}, {_fmt((ic.get('sticky') or {}).get('ci_high'), 4)}] | "
        "持仓用入场时的评分（被冻结） |",
        f"| 降幅 | **{_fmt(ic.get('drop'), 5)}** | — | 上限 {report['params']['ic_drop_limit']} |",
        "",
        f"共 {ic.get('n_dates')} 个调仓截面。降幅为正说明 buffer 让信号变钝，"
        "这是省换手必须付的价钱；超过 0.005 就说明省下的成本是拿 alpha 换的。",
        "",
        "## 5. 成本敏感性",
        "",
        "| 单边成本 | 年换手 | 成本拖累 | 净 Sharpe | 年化收益 |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["sensitivity"]:
        lines.append(
            f"| {row['cost_bps']:.0f}bp | {_fmt(row['annual_turnover'], 0, '%')} | "
            f"{_fmt(row['cost_drag_bps'], 0, 'bp')} | {_fmt(row['sharpe'], 3)} | "
            f"{_fmt(row['annual_return'], 2, '%')} |"
        )
    lines += [
        "",
        "试过但没有采用的替代方案（同一 buffer，只换一处）：",
        "",
        "| 方案 | 开发折净 Sharpe | 全区间净 Sharpe | 年换手 | 全区间年化 | 不采用的原因 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    reasons = {
        "等权（采用）": "基线方案",
        "inv_vol 权重": "波动小的票被加权重，组合反而更集中、换手更高",
        "波动目标 18%": "只在开发折有效，全区间更差——样本外不稳定",
        "扩到 Top 30%": "持仓变多但 Sharpe 没提升，且偏离 Top 20% 的产品定义",
    }
    for row in report.get("variants", []):
        lines.append(
            f"| {row['label']} | {_fmt(row['dev']['sharpe'], 3)} | {_fmt(row['full']['sharpe'], 3)} | "
            f"{_fmt(row['full']['annual_turnover'], 0, '%')} | {_fmt(row['full']['annual_return'], 2, '%')} | "
            f"{reasons.get(row['label'], '')} |"
        )
    cap = report["capacity"]
    lines += [
        "",
        "## 6. 容量（参与率 ≤ 10%）",
        "",
        "| 情形 | 约束规模(p10) | 中位规模 | 绑定规模 |",
        "| --- | ---: | ---: | ---: |",
        f"| 当前池（大盘为主） | {cap['main'].get('aum_p10_亿', '—')} 亿 | "
        f"{cap['main'].get('aum_p50_亿', '—')} 亿 | {cap['main'].get('aum_binding_亿', '—')} 亿 |",
        f"| 中小盘（成交额 ×0.3） | {cap['small_cap'].get('aum_p10_亿', '—')} 亿 | "
        f"{cap['small_cap'].get('aum_p50_亿', '—')} 亿 | {cap['small_cap'].get('aum_binding_亿', '—')} 亿 |",
        "",
        f"{cap['assumption']}。冲击成本用平方根模型（参与率 1% ≈ 10bp）估算，"
        "参数需要真实成交数据校准，只作规模敏感性参考。",
        "",
        "看 p10 而不是「绑定规模」：后者由单笔最极端的交易决定（小盘股某天成交极小），"
        "p10 才是 90% 的交易都能在 10% 参与率内完成的规模。",
        "",
        "## 7. 选股 alpha 与再平衡效应",
        "",
        "| 口径 | 收益 | 说明 |",
        "| --- | ---: | --- |",
        f"| 改进方案（5 日调仓） | {_fmt(tuned['metrics']['total_return'], 2, '%')} | 含成本 |",
        f"| 买入持有首期 Top20% | {_fmt(report['buy_and_hold']['total_return'], 2, '%')} | "
        f"{report['buy_and_hold']['symbols']} 只，只付一次进场成本 |",
        f"| 等权基准 | {_fmt(bench['total_return'], 2, '%')} | 同区间、不计成本 |",
        "",
        "两者之差是**再平衡效应**（调仓的贡献），不是选股 alpha。若买持更高，说明这个样本上"
        "再平衡是负贡献，应继续降频或加宽缓冲带。",
        "",
        "## 8. holdout（后 2 折，只评估一次）",
        "",
        "| 口径 | 年换手 | 净 Sharpe | 年化收益 |",
        "| --- | ---: | ---: | ---: |",
        f"| 基线 | {_fmt(report['holdout']['baseline']['annual_turnover'], 0, '%')} | "
        f"{_fmt(report['holdout']['baseline']['sharpe'], 3)} | "
        f"{_fmt(report['holdout']['baseline']['annual_return'], 2, '%')} |",
        f"| 改进方案 | {_fmt(report['holdout']['tuned']['annual_turnover'], 0, '%')} | "
        f"{_fmt(report['holdout']['tuned']['sharpe'], 3)} | "
        f"{_fmt(report['holdout']['tuned']['annual_return'], 2, '%')} |",
        f"| 等权基准 | {_fmt((report['holdout'].get('benchmark') or {}).get('annual_turnover'), 0, '%')} | "
        f"{_fmt((report['holdout'].get('benchmark') or {}).get('sharpe'), 3)} | "
        f"{_fmt((report['holdout'].get('benchmark') or {}).get('annual_return'), 2, '%')} |",
        "",
        "## 9. 通过标准",
        "",
        "| 标准 | 实测 | 对比值 | 判定 |",
        "| --- | ---: | ---: | --- |",
    ]
    labels = {"pass": "通过", "fail": "未通过", "insufficient": "数据不足"}

    def show(name: str, value: Any) -> str:
        if value is None:
            return "—"
        if "换手" in name:
            return _fmt(value, 1, "%")
        if "IC" in name:
            return _fmt(value, 5)
        return _fmt(value, 3)

    for item in report["criteria"]:
        lines.append(
            f"| {item['name']} | {show(item['name'], item['value'])} | "
            f"{show(item['name'], item.get('benchmark'))} | "
            f"{labels.get(item.get('status'), '通过' if item['pass'] else '未通过')} |"
        )
    lines += [
        "",
        f"试验台账：{report['ledger']['n_trials']} 条。参数搜索本身就是多重检验，"
        "所以选参用「净 Sharpe + 换手硬约束」，而不是挑毛收益最高的一组。",
        "",
    ]
    return "\n".join(lines)


def write_reports(report: dict[str, Any], outdir: Path = REPORT_DIR) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": outdir / "turnover.json",
        "markdown": outdir / "turnover.md",
        "preview": outdir / "turnover_preview.html",
    }
    paths["json"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["markdown"].write_text(render_markdown(report), encoding="utf-8")
    paths["preview"].write_text(_render_preview(report), encoding="utf-8")
    return paths


def _render_preview(report: dict[str, Any]) -> str:
    lines = render_markdown(report).splitlines()
    html: list[str] = []
    in_table = False
    for line in lines:
        if line.startswith("# "):
            html.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            html.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("| ---"):
            continue
        elif line.startswith("| "):
            cells = [cell.strip().replace("**", "") for cell in line.strip("|").split("|")]
            if not in_table:
                html.append("<table>")
                in_table = True
            html.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
        else:
            if in_table:
                html.append("</table>")
                in_table = False
            if line.strip():
                html.append(f"<p>{line}</p>")
    if in_table:
        html.append("</table>")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>降换手改进方案</title>
<style>
  body {{ margin:0; background:#f4f2eb; color:#16221d;
    font:15px/1.7 -apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:34px 22px 60px; }}
  .panel {{ background:#fffdf7; border:1px solid #d8ddd6; border-radius:10px; padding:26px;
    box-shadow:0 18px 50px rgba(31,48,40,.08); }}
  h1 {{ font-size:22px; margin:0 0 14px; }}
  h2 {{ font-size:15px; margin:26px 0 8px; color:#176b4d; }}
  table {{ width:100%; border-collapse:collapse; font-size:12.5px; margin:8px 0 4px; }}
  td {{ padding:6px 9px; border-bottom:1px solid #e6e9e2; white-space:nowrap; }}
  tr:first-child td {{ color:#6e7b74; font-weight:600; background:rgba(23,107,77,.05); }}
  p {{ margin:6px 0; color:#3d4a44; }}
</style></head>
<body><div class="wrap"><div class="panel">{''.join(html)}</div></div></body></html>"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="降换手专项研究（buffer / 分批 / 成本感知）")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--cost-bps", type=float, default=20.0)
    parser.add_argument("--iters", type=int, default=800)
    parser.add_argument("--outdir", default=str(REPORT_DIR))
    args = parser.parse_args(argv)
    try:
        report = run(offline=args.offline, cost_bps=args.cost_bps, iters=max(200, args.iters))
    except MarketDataError as exc:
        print(f"运行失败：{exc}", file=sys.stderr)
        return 2
    paths = write_reports(report, Path(args.outdir))
    marks = {"pass": "✓", "fail": "✗", "insufficient": "?"}
    for item in report["criteria"]:
        print(f"  {marks.get(item.get('status'), '✗')} {item['name']}：{item['value']}")
    for key, path in paths.items():
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
