"""评估协议：三个目标各自定标签、各自定指标，所有结论必须带区间。

这个模块替代"一次 60/40 切分 + 点估计 IC"的旧做法，把用户提出的协议落到
可执行代码上：

===========  ==========================  ==================  ==========================
目标          标签                        指标                 通过标准
===========  ==========================  ==================  ==========================
A 描述分      未来 20 日波动 / 最大回撤 /  分位 IC + 块自助 CI   不要求收益 IC，只要求"分数
              下行波动 / 5% VaR                                对风险状态有可区分信息"
B 选股分      次日开盘进、20 日后开盘出    IC + 净净值(T+1)      IC 的 CI 不含 0 且净净值
              的横截面分位收益                                  跑赢等权
C 市场 regime 组合回撤与仓位暴露           回撤改善、切换成本    回撤改善 > 切换成本
===========  ==========================  ==================  ==========================

另外四件事也在这里做成固定流程，而不是靠自觉：

1. **5 折 purged walk-forward** 取代单次 60/40，末 2 年作为 lockbox 只在
   最后评估一次。
2. **多重检验记账**：本轮真正试过的因子表达 / horizon / 换手上限 / universe
   全部写进 ``evalkit.TrialLedger``，再用 BH-FDR 与 deflated Sharpe 校正。
3. **执行口径**：信号收盘出、次日开盘成交，过滤涨跌停与停牌，T+1 不得卖出
   当日买入的份额。
4. **Harness 校准**：随机信号跑同一套流程应当 IC≈0 且 95% 区间覆盖 0；
   12-1 动量作为已知弱信号用来校准统计功效。

用法::

    python protocol.py --offline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import evalkit as ek
import execution as ex
import style as st
import xsec
from analyzer import DEFAULT_WEIGHTS
from market_data import MarketDataError
from xsec import build_panel, fit_rank_model, metrics as nav_metrics, model_scores, screen_features
from walkforward import REPORT_DIR, load_benchmark, load_universe

ROOT = Path(__file__).resolve().parent
DAYS = 2400          # 约 9.6 年，够 5 折 + 2 年 lockbox + 12-1 预热
HORIZON = 20
WARMUP = 130
K_FOLDS = 5
EMBARGO = 5
LOCKBOX_YEARS = 2
TRADING_DAYS_PER_YEAR = 243
REBALANCE = 5
TOP_FRACTION = 0.2
MAX_TURNOVER = 0.4
MIN_XSEC = 20
DEFAULT_SYMBOLS_N = 60
BOOTSTRAP_ITERS = 2000

TARGETS: dict[str, dict[str, Any]] = {
    "A": {
        "name": "描述分（趋势 / 风险状态）",
        "label": "未来 20 日波动率、最大回撤、下行波动、5% VaR",
        "metric": "横截面分位 IC（分数 vs 风险标签）+ 块自助 CI",
        "pass": "至少一个风险标签的 IC 区间不含 0，且方向可解释；不要求收益 IC",
    },
    "B": {
        "name": "选股分（横截面分位收益）",
        "label": "次日开盘进场、20 日后开盘出场的横截面分位收益",
        "metric": "IC + 95% block bootstrap CI + 扣费净净值（T+1 开盘成交）",
        "pass": "IC 的 CI 不含 0，且净净值跑赢同池等权",
    },
    "C": {
        "name": "市场 regime（回撤与暴露）",
        "label": "组合层回撤与目标仓位",
        "metric": "最大回撤改善、年化换手与切换成本",
        "pass": "回撤改善的幅度大于切换成本，且不显著牺牲年化收益",
    },
}


# --------------------------------------------------------------------------- #
# 日历、切分
# --------------------------------------------------------------------------- #


def purged_kfold(
    dates: Sequence[str],
    *,
    k: int = K_FOLDS,
    horizon: int = HORIZON,
    embargo: int = EMBARGO,
) -> list[dict[str, Any]]:
    """连续分块的 K 折，训练集与测试块之间做 purge + embargo。

    训练样本 t 的标签覆盖 ``[t, t+horizon]``，所以只有 ``t + horizon < 测试块首日``
    的训练样本可用；embargo 再让出若干交易日隔离缓慢变量。测试块之后的训练
    样本同样要求 ``t >= 测试块末日 + embargo``。
    """
    n = len(dates)
    bounds = [round(n * i / k) for i in range(k + 1)]
    folds: list[dict[str, Any]] = []
    for index in range(k):
        start, end = bounds[index], bounds[index + 1]
        if end - start < horizon:      # 测试块太短，无法形成一次完整调仓周期
            continue
        test_dates = list(dates[start:end])
        left_end = max(0, start - horizon - embargo)
        right_start = min(n, end + embargo)
        train_dates = list(dates[:left_end]) + list(dates[right_start:])
        folds.append(
            {
                "fold": index + 1,
                "test_start": test_dates[0],
                "test_end": test_dates[-1],
                "test_dates": test_dates,
                "train_dates": train_dates,
                "purged_left": start - left_end,
                "purged_right": min(n, end + embargo) - end,
            }
        )
    return folds


def lockbox_split(dates: Sequence[str], *, years: int = LOCKBOX_YEARS) -> dict[str, Any]:
    """末 ``years`` 年作为 lockbox：开发阶段完全不参与任何选择。"""
    keep = years * TRADING_DAYS_PER_YEAR
    if len(dates) <= keep + 120:
        return {
            "dev_dates": list(dates),
            "lockbox_dates": [],
            "lockbox_start": None,
            "note": f"样本仅 {len(dates)} 个交易日，不足以留出 {years} 年 lockbox",
        }
    return {
        "dev_dates": list(dates[:-keep]),
        "lockbox_dates": list(dates[-keep:]),
        "lockbox_start": dates[-keep],
        "note": f"末 {keep} 个交易日（约 {years} 年）锁定",
    }


# --------------------------------------------------------------------------- #
# 标签
# --------------------------------------------------------------------------- #


def forward_open_returns(
    histories: Sequence[dict[str, Any]],
    *,
    horizon: int = HORIZON,
) -> dict[str, dict[str, float]]:
    """与 T+1 执行一致的收益标签：``open[t+1+H] / open[t+1] - 1``（百分数）。"""
    result: dict[str, dict[str, float]] = {}
    for history in histories:
        rows = history["rows"]
        series: dict[str, float] = {}
        for index in range(len(rows) - horizon - 1):
            entry = float(rows[index + 1]["open"])
            exit_price = float(rows[index + 1 + horizon]["open"])
            if entry > 0 and exit_price > 0:
                series[rows[index]["date"]] = (exit_price / entry - 1.0) * 100.0
        result[history["symbol"]] = series
    return result


def risk_labels(
    histories: Sequence[dict[str, Any]],
    *,
    horizon: int = HORIZON,
) -> dict[str, dict[str, dict[str, float]]]:
    """A 目标的风险标签：未来 ``horizon`` 日的波动、回撤、下行波动、VaR。"""
    result: dict[str, dict[str, dict[str, float]]] = {}
    for history in histories:
        rows = history["rows"]
        closes = [float(row["close"]) for row in rows]
        series: dict[str, dict[str, float]] = {}
        for index in range(len(closes) - horizon - 1):
            window = closes[index: index + horizon + 1]
            returns = [
                window[i + 1] / window[i] - 1.0
                for i in range(len(window) - 1)
                if window[i] > 0
            ]
            if len(returns) < max(6, horizon // 3):
                continue
            peak = window[0]
            worst = 0.0
            for price in window:
                peak = max(peak, price)
                if peak > 0:
                    worst = min(worst, price / peak - 1.0)
            downside = [r for r in returns if r < 0]
            sorted_returns = sorted(returns)
            var_index = max(0, min(len(sorted_returns) - 1, int(0.05 * len(sorted_returns))))
            series[rows[index]["date"]] = {
                "vol": statistics.stdev(returns) * math.sqrt(252) * 100 if len(returns) > 1 else 0.0,
                "mdd": worst * 100,
                "downside": (
                    math.sqrt(statistics.fmean([r * r for r in downside])) * math.sqrt(252) * 100
                    if downside else 0.0
                ),
                "var95": -sorted_returns[var_index] * 100,
            }
        result[history["symbol"]] = series
    return result


# --------------------------------------------------------------------------- #
# 面板：特征沿用 xsec，标签换成 T+1 口径
# --------------------------------------------------------------------------- #


def build_records(
    histories: Sequence[dict[str, Any]],
    benchmark: dict[str, Any],
    *,
    horizon: int = HORIZON,
) -> dict[str, Any]:
    """在 ``xsec.build_panel`` 之上把标签换成"次日开盘进、H 日后开盘出"的分位。

    特征与选因子流程保持不变，只换标签口径——这样"同日收盘"和"T+1 开盘"
    两种口径可以直接对比，差额只来自执行时点。
    """
    panel = build_panel(histories, benchmark, horizon=horizon, warmup=WARMUP)
    exec_returns = forward_open_returns(histories, horizon=horizon)
    by_date: dict[str, list[dict[str, Any]]] = {}
    for record in panel["records"]:
        by_date.setdefault(record["date"], []).append(record)

    kept: list[dict[str, Any]] = []
    for date, group in by_date.items():
        labeled = [
            record for record in group
            if exec_returns.get(record["symbol"], {}).get(date) is not None
        ]
        if len(labeled) < MIN_XSEC:
            continue
        values = [exec_returns[record["symbol"]][date] for record in labeled]
        ranks = xsec._rank_pct(values)
        labels = dict(zip([record["symbol"] for record in labeled], ranks))
        for record in group:
            record["fwd_exec"] = exec_returns.get(record["symbol"], {}).get(date)
            record["label"] = labels.get(record["symbol"])
            record["fwd"] = record["fwd_exec"]       # 下游 quintile/screen 统一看 exec 口径
            kept.append(record)
    kept.sort(key=lambda record: (record["date"], record["symbol"]))
    dates = sorted({record["date"] for record in kept})
    return {"records": kept, "dates": dates, "panel_dates": panel["dates"]}


def daily_ic_series(
    records: Sequence[dict[str, Any]],
    score_map: Callable[[dict[str, Any]], float | None] | dict[tuple[str, str], float],
    *,
    actual_key: str = "fwd",
    min_xsec: int = MIN_XSEC,
) -> list[float]:
    """逐日横截面 Spearman IC 序列（缺值日跳过）。"""
    by_date: dict[str, list[tuple[float, float]]] = {}
    for record in records:
        value = record.get(actual_key)
        if value is None:
            continue
        score = (
            score_map(record)
            if callable(score_map)
            else score_map.get((record["date"], record["symbol"]))
        )
        if score is None:
            continue
        by_date.setdefault(record["date"], []).append((float(score), float(value)))
    series: list[float] = []
    for date in sorted(by_date):
        pairs = by_date[date]
        if len(pairs) < min_xsec:
            continue
        ic = ek.spearman([p[0] for p in pairs], [p[1] for p in pairs])
        if ic is not None:
            series.append(ic)
    return series


def _rank_column(
    records: Sequence[dict[str, Any]],
    key: str,
) -> dict[tuple[str, str], float]:
    """把任意因子在当日横截面内转成分位，供 daily_ic_series 使用。"""
    by_date: dict[str, list[tuple[str, float]]] = {}
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        by_date.setdefault(record["date"], []).append((record["symbol"], float(value)))
    ranks: dict[tuple[str, str], float] = {}
    for date, pairs in by_date.items():
        if len(pairs) < MIN_XSEC:
            continue
        values = [value for _symbol, value in pairs]
        for (symbol, _value), rank in zip(pairs, xsec._rank_pct(values)):
            ranks[(date, symbol)] = rank
    return ranks


# --------------------------------------------------------------------------- #
# B 目标：5 折 purged walk-forward
# --------------------------------------------------------------------------- #


def run_walkforward(
    records: Sequence[dict[str, Any]],
    folds: Sequence[dict[str, Any]],
    *,
    horizon: int = HORIZON,
    iters: int = BOOTSTRAP_ITERS,
) -> dict[str, Any]:
    """逐折：训练集选表达 + 拟合 → 测试块打分 → 测试块 IC。"""
    rows: list[dict[str, Any]] = []
    pooled: list[float] = []
    within_variance: list[float] = []
    fold_scores: dict[int, dict[tuple[str, str], float]] = {}
    fold_models: dict[int, dict[str, Any]] = {}
    for fold in folds:
        train_set = set(fold["train_dates"])
        test_set = set(fold["test_dates"])
        train = [r for r in records if r["date"] in train_set and r.get("label") is not None]
        test = [r for r in records if r["date"] in test_set]
        if not train or not test:
            continue
        screening = screen_features(train, horizon=horizon)
        model = fit_rank_model(train, screening["features"])
        scores = model_scores(test, model)
        series = daily_ic_series(test, scores)
        stat = ek.ic_ci(series, horizon=horizon, iters=iters)
        pooled.extend(series)
        if len(series) > 2:
            within_variance.append(statistics.stdev(series) ** 2)
        fold_scores[fold["fold"]] = scores
        fold_models[fold["fold"]] = model
        rows.append(
            {
                "fold": fold["fold"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "train_start": fold["train_dates"][0] if fold["train_dates"] else None,
                "n_train": len(train),
                "n_test": len(test),
                "features": screening["features"],
                "ic": stat["mean"],
                "ci_low": stat["ci_low"],
                "ci_high": stat["ci_high"],
                "p": stat["p"],
                "n_dates": stat["n"],
                "significant": stat["significant"],
                "coefficients": model["coefficients"],
            }
        )
    pooled_stat = ek.ic_ci(pooled, horizon=horizon, iters=iters)
    return {
        "folds": rows,
        "pooled": pooled_stat,
        "daily_ic": pooled,
        # 折内方差取平均：跨折直接算会把"不同年份的水平漂移"当成抽样噪声，
        # 从而把 σ 抬高（8 年样本里各折 IC 均值本来就不同）。
        "sigma_within": math.sqrt(statistics.fmean(within_variance)) if within_variance else None,
        "sigma_pooled": statistics.stdev(pooled) if len(pooled) > 1 else None,
        "fold_scores": fold_scores,
        "fold_models": fold_models,
    }


def feature_ic_table(
    records: Sequence[dict[str, Any]],
    folds: Sequence[dict[str, Any]],
    *,
    horizon: int = HORIZON,
    iters: int = 800,
) -> list[dict[str, Any]]:
    """每个候选表达的样本外（跨折拼接）IC 与区间，供多重检验记账。"""
    rows = []
    for key in xsec.FEATURE_LABELS:
        pooled: list[float] = []
        for fold in folds:
            test_set = set(fold["test_dates"])
            test = [r for r in records if r["date"] in test_set]
            ranks = _rank_column(test, key)
            pooled.extend(daily_ic_series(test, ranks))
        stat = ek.ic_ci(pooled, horizon=horizon, iters=iters)
        rows.append(
            {
                "key": key,
                "label": xsec.FEATURE_LABELS[key],
                "family": next(
                    (family for family, keys in xsec.FEATURE_FAMILIES.items() if key in keys), ""
                ),
                **{k: stat[k] for k in ("mean", "ci_low", "ci_high", "p", "n", "significant")},
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# 净值：T+1 拼接
# --------------------------------------------------------------------------- #


def stitch(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """把各折净值首尾相接，拼成一条连续的样本外净值。"""
    nav: list[dict[str, float]] = []
    base = 1.0
    for result in results:
        points = result["nav"]
        if not points:
            continue
        offset = base / points[0]["nav"] if points[0]["nav"] else 1.0
        for point in points:
            nav.append({"date": point["date"], "nav": point["nav"] * offset})
        base = nav[-1]["nav"]
    return {
        "nav": nav,
        "rebalances": [item for result in results for item in result["rebalances"]],
        "holdings": [item for result in results for item in result.get("holdings", [])],
        "total_cost": sum(result["total_cost"] for result in results),
        "total_turnover": sum(result["total_turnover"] for result in results),
        "cash_ratio_mean": statistics.fmean(
            [result["cash_ratio_mean"] for result in results if result["cash_ratio_mean"] is not None]
        ) if results else 0.0,
        "blocked": {
            key: sum(result.get("blocked", {}).get(key, 0) for result in results)
            for key in ("buy_limit_up", "buy_suspended", "sell_limit_down", "sell_suspended")
        },
    }


def walkforward_nav(
    calendar: Sequence[str],
    bars: dict[str, dict[str, dict[str, float]]],
    flags: dict[str, dict[str, str]],
    walk: dict[str, Any],
    folds: Sequence[dict[str, Any]],
    *,
    rebalance: int = REBALANCE,
    top_fraction: float = TOP_FRACTION,
    max_turnover: float = MAX_TURNOVER,
    cost_buy: float = xsec.BUY_COST,
    cost_sell: float = xsec.SELL_COST,
    members: Callable[[str], set[str]] | None = None,
) -> dict[str, Any]:
    """逐折在自己的测试块上跑 T+1 回测，再拼接成完整样本外净值。

    ``members`` 给出"当日允许持有的标的集合"，用于 PIT 成分回测：信号只从当日
    真实成分里选，避免用今天的名单去持有当年还没进指数的票。
    """
    results = []
    for fold in folds:
        scores = walk["fold_scores"].get(fold["fold"])
        if not scores:
            continue
        block = [date for date in calendar if fold["test_start"] <= date <= fold["test_end"]]
        if len(block) < rebalance * 2:
            continue

        def pick(date: str, available: list[str], _scores=scores) -> list[str]:
            allowed = available if members is None else [
                symbol for symbol in available if symbol in members(date)
            ]
            day = {symbol: _scores[(date, symbol)] for symbol in allowed if (date, symbol) in _scores}
            return xsec.top_n(day, top_fraction)

        results.append(
            ex.simulate_t1(
                block, bars, pick,
                flags=flags,
                rebalance_every=rebalance,
                cost_buy=cost_buy,
                cost_sell=cost_sell,
                max_turnover=max_turnover,
            )
        )
        # 打上折号：风格归因必须跳过折与折之间的空档，否则会凭空多出
        # 一段跨年的"持有期"，把单期收益算成几十个百分点。
        for item in results[-1].get("holdings", []):
            item["fold"] = fold["fold"]
    return stitch(results)


def equal_weight_nav(
    calendar: Sequence[str],
    bars: dict[str, dict[str, dict[str, float]]],
    *,
    rebalance: int = REBALANCE,
    with_cost: bool = False,
    members: Callable[[str], set[str]] | None = None,
) -> dict[str, Any]:
    """同池等权基准。

    ``members`` 与 ``walkforward_nav`` 用同一个回调：PIT 口径下等权基准也必须
    只买当日真实成分，否则"跑赢等权"会比的是一个包含当年还没进指数的票的池子。
    """

    def pick(date: str, available: list[str]) -> list[str]:
        if members is None:
            return sorted(available)
        return sorted(symbol for symbol in available if symbol in members(date))

    return ex.simulate_t1(
        calendar, bars, pick,
        rebalance_every=rebalance,
        cost_buy=xsec.BUY_COST if with_cost else 0.0,
        cost_sell=xsec.SELL_COST if with_cost else 0.0,
        max_turnover=1.0,
    )


def period_excess(
    strategy: dict[str, Any],
    benchmark: dict[str, Any],
    *,
    step: int,
) -> list[float]:
    """按调仓窗口切净值，返回逐窗口超额（%）。"""
    rows = xsec.period_breakdown(strategy, benchmark, step=step)
    return [row["excess"] for row in rows]


# --------------------------------------------------------------------------- #
# A 目标：描述分是否携带风险信息
# --------------------------------------------------------------------------- #


def evaluate_target_a(
    histories: Sequence[dict[str, Any]],
    benchmark: dict[str, Any],
    dates: Sequence[str],
    *,
    horizon: int = HORIZON,
    stride: int = 5,
    iters: int = BOOTSTRAP_ITERS,
) -> dict[str, Any]:
    """用规则评分对风险标签做横截面 IC（分数越高应当越"危险"）。"""
    labels = risk_labels(histories, horizon=horizon)
    scorer = xsec.RuleScorer(histories, benchmark)
    sample = list(dates[::stride])
    series: dict[str, list[float]] = {key: [] for key in ("vol", "mdd", "downside", "var95")}
    coverage = 0
    for date in sample:
        available = [
            symbol for symbol in scorer.histories
            if labels.get(symbol, {}).get(date) is not None
        ]
        if len(available) < MIN_XSEC:
            continue
        scores = scorer.score(date, available)
        symbols = [symbol for symbol in available if symbol in scores]
        if len(symbols) < MIN_XSEC:
            continue
        coverage += 1
        score_values = [scores[symbol] for symbol in symbols]
        for key in series:
            target = [labels[symbol][date][key] for symbol in symbols]
            ic = ek.spearman(score_values, target)
            if ic is not None:
                series[key].append(ic)
    return {
        "dates": coverage,
        "stride": stride,
        "metrics": {
            key: ek.ic_ci(values, horizon=horizon, iters=iters) for key, values in series.items()
        },
        "interpretation": "IC>0 表示分数越高、未来风险越大；IC<0 表示高分对应低风险",
    }


# --------------------------------------------------------------------------- #
# C 目标：regime 暴露
# --------------------------------------------------------------------------- #


def evaluate_target_c(
    histories: Sequence[dict[str, Any]],
    benchmark: dict[str, Any],
    calendar: Sequence[str],
    bars: dict[str, dict[str, dict[str, float]]],
    flags: dict[str, dict[str, str]],
    *,
    rebalance: int = REBALANCE,
    iters: int = 800,
) -> dict[str, Any]:
    """把全池平均评分当仓位信号：分数低就降仓，看回撤能否改善、切换成本多大。

    信号必须因果：``date`` 的目标仓位只用 ``date`` 及之前的评分，靠
    ``RuleScorer`` 的前缀切片保证。仓位映射用滚动 250 日分位（低分位 → 0.3
    仓，高分位 → 1.0 仓），避免用全样本分位偷看未来。
    """
    scorer = xsec.RuleScorer(histories, benchmark)
    sample = list(calendar[::rebalance])
    mean_scores: dict[str, float] = {}
    for date in sample:
        available = [symbol for symbol in scorer.histories if scorer.index[symbol].get(date) is not None]
        scores = scorer.score(date, available)
        if scores:
            mean_scores[date] = statistics.fmean(scores.values())
    ordered = sorted(mean_scores)
    if len(ordered) < 60:
        return {"available": False, "note": "样本不足，无法构造 regime 信号"}
    sorted_values = [mean_scores[date] for date in ordered]
    positions = {date: index for index, date in enumerate(ordered)}

    def exposure(date: str) -> float:
        position = positions.get(date)
        if position is None:
            # 非信号日：沿用最近一次信号日的仓位
            previous = [d for d in ordered if d <= date]
            position = positions[previous[-1]] if previous else 0
        window = sorted_values[max(0, position - 250): position + 1]
        value = mean_scores[ordered[position]]
        rank = sum(1 for item in window if item <= value) / len(window)
        return 0.3 + 0.7 * rank

    def pick(_date: str, available: list[str]) -> list[str]:
        return sorted(available)

    regime = ex.simulate_t1(
        calendar, bars, pick,
        flags=flags,
        rebalance_every=rebalance,
        exposure=exposure,
        max_turnover=1.0,
    )
    full = equal_weight_nav(calendar, bars, rebalance=rebalance)
    regime_stat = nav_metrics(regime, len(calendar))
    full_stat = nav_metrics(full, len(calendar))
    excess = period_excess(regime, full, step=horizon_window(rebalance))
    return {
        "available": True,
        "signal_dates": len(sample),
        "exposure_mean": round(statistics.fmean(
            [item.get("gross", 1.0) for item in regime["rebalances"]]
        ) * 100, 1),
        "strategy": regime_stat,
        "benchmark": full_stat,
        "drawdown_improvement": round(full_stat["max_drawdown"] - regime_stat["max_drawdown"], 2),
        "return_change": round(regime_stat["total_return"] - full_stat["total_return"], 2),
        "annual_turnover": regime_stat["annual_turnover"],
        "switch_cost_bps": regime_stat["cost_drag_bps"],
        "excess_ci": ek.block_bootstrap_mean(excess, block=max(2, len(excess) // 3), iters=iters),
        "calmar": {
            "strategy": round(regime_stat["annual_return"] / abs(regime_stat["max_drawdown"]), 2)
            if regime_stat["max_drawdown"] else None,
            "benchmark": round(full_stat["annual_return"] / abs(full_stat["max_drawdown"]), 2)
            if full_stat["max_drawdown"] else None,
        },
        "nav": regime["nav"],
        "benchmark_nav": full["nav"],
        "note": "回撤改善为正=策略回撤更浅；切换成本为年化成本拖累，需小于回撤改善带来的收益",
    }


def horizon_window(rebalance: int) -> int:
    return max(rebalance, HORIZON)


# --------------------------------------------------------------------------- #
# Harness 校准
# --------------------------------------------------------------------------- #


def _pseudo_random(seed: int, date: str, symbol: str) -> float:
    digest = hashlib.md5(f"{seed}|{date}|{symbol}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def random_signal_harness(
    records: Sequence[dict[str, Any]],
    folds: Sequence[dict[str, Any]],
    *,
    seeds: int = 20,
    horizon: int = HORIZON,
    iters: int = 400,
    smooth: int = 5,
) -> dict[str, Any]:
    """随机信号跑同一套 harness：IC 应当≈0，95% 区间应当以 95% 概率覆盖 0。

    跑两种随机信号，因为它们的失效模式不同：

    - **iid 随机**：逐日独立乱序。检验的是"没有任何结构时不会报显著"；
    - **平滑随机**：``smooth`` 日移动平均的乱序，等于给信号加上自相关，
      让逐日 IC 序列像真实因子那样连续同号。这才是块自助法真正要保护的
      场景——如果块长设错，第二种会大面积误报。
    """
    by_symbol: dict[str, list[str]] = {}
    for record in records:
        by_symbol.setdefault(record["symbol"], []).append(record["date"])
    position = {
        (symbol, date): index
        for symbol, dates in by_symbol.items()
        for index, date in enumerate(sorted(dates))
    }
    sorted_dates = {symbol: sorted(dates) for symbol, dates in by_symbol.items()}

    def build(seed: int) -> list[dict[str, Any]]:
        base = {
            (record["date"], record["symbol"]): _pseudo_random(seed, record["date"], record["symbol"])
            for record in records
        }
        if smooth <= 1:
            return base
        smoothed = {}
        for (date, symbol), _value in base.items():
            index = position[(symbol, date)]
            window = sorted_dates[symbol][max(0, index - smooth + 1): index + 1]
            smoothed[(date, symbol)] = statistics.fmean(base[(d, symbol)] for d in window)
        return smoothed

    def run(score_builder) -> dict[str, Any]:
        means: list[float] = []
        coverage = false_positive = 0
        widths: list[float] = []
        for seed in range(seeds):
            scores = score_builder(seed)
            pooled: list[float] = []
            for fold in folds:
                test_set = set(fold["test_dates"])
                test = [r for r in records if r["date"] in test_set and r.get("fwd") is not None]
                pooled.extend(daily_ic_series(test, scores))
            stat = ek.ic_ci(pooled, horizon=horizon, iters=iters, seed=seed + 7)
            if stat["mean"] is not None:
                means.append(stat["mean"])
            if stat["covers_zero"]:
                coverage += 1
            if stat["significant"]:
                false_positive += 1
            if stat["ci_low"] is not None:
                widths.append(stat["ci_high"] - stat["ci_low"])
        return {
            "mean_ic": round(statistics.fmean(means), 4) if means else None,
            "mean_abs_ic": round(statistics.fmean([abs(v) for v in means]), 4) if means else None,
            "coverage": round(coverage / seeds * 100, 1),
            "false_positive_rate": round(false_positive / seeds * 100, 1),
            "ci_width_mean": round(statistics.fmean(widths), 4) if widths else None,
        }

    iid = run(
        lambda seed: {
            (record["date"], record["symbol"]): _pseudo_random(seed, record["date"], record["symbol"])
            for record in records
        }
    )
    persistent = run(build)
    return {
        "seeds": seeds,
        "iid": iid,
        "persistent": persistent,
        "pass": bool(
            iid["coverage"] >= 85 and persistent["coverage"] >= 85
            and iid["false_positive_rate"] <= 15 and persistent["false_positive_rate"] <= 15
        ),
    }


def weak_signal_harness(
    histories: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    folds: Sequence[dict[str, Any]],
    *,
    horizon: int = HORIZON,
    iters: int = 800,
) -> dict[str, Any]:
    """已知弱信号（12-1 动量）校准统计功效：真实存在的弱 edge 能不能被看见。"""
    momentum = {history["symbol"]: st.momentum_series(history) for history in histories}
    pooled: list[float] = []
    within: list[float] = []
    for fold in folds:
        test = [record for record in records if record["date"] in set(fold["test_dates"])]
        score_map = {
            (record["date"], record["symbol"]): momentum.get(record["symbol"], {}).get(record["date"])
            for record in test
        }
        series = daily_ic_series(test, {k: v for k, v in score_map.items() if v is not None})
        pooled.extend(series)
        if len(series) > 2:
            within.append(statistics.stdev(series) ** 2)
    stat = ek.ic_ci(pooled, horizon=horizon, iters=iters)
    sigma = math.sqrt(statistics.fmean(within)) if within else (
        statistics.stdev(pooled) if len(pooled) > 1 else None
    )
    n_blocks = stat.get("n_blocks")
    return {
        "ic": stat,
        "sigma": round(sigma, 4) if sigma else None,
        "detectable_ic": (
            round(ek.detectable_ic(n_blocks, sigma), 4) if sigma and n_blocks else None
        ),
        "n_blocks": n_blocks,
        "detected": stat["significant"],
        "note": "12-1 是文献里稳定存在的弱信号；若它都检不出，说明样本量不足以支撑更强的结论",
    }


def power_analysis(
    sigma: float,
    *,
    n_blocks: float,
    horizon: int = HORIZON,
    sigma_theory: float | None = None,
) -> dict[str, Any]:
    blocks = ek.required_independent_blocks(0.03, sigma)
    sensitivity = []
    for label, value in (
        ("理论下限 1/√N", sigma_theory),
        ("实测（本报告采用）", sigma),
        ("文献常用 0.13", 0.13),
    ):
        if not value:
            continue
        need = ek.required_independent_blocks(0.03, value)
        sensitivity.append(
            {
                "label": label,
                "sigma": round(value, 4),
                "n_blocks": round(need, 1),
                "n_years": round(need * horizon / TRADING_DAYS_PER_YEAR, 1),
            }
        )
    return {
        "sigma_ic": round(sigma, 4),
        "sigma_theory": round(sigma_theory, 4) if sigma_theory else None,
        "sensitivity": sensitivity,
        "n_blocks_now": round(n_blocks, 1),
        "detectable_now": round(ek.detectable_ic(n_blocks, sigma), 4),
        "table": ek.power_table(sigma, ics=(0.02, 0.03, 0.05, 0.10), horizons=(5, 20, 60)),
        "for_ic_003": {
            "n_blocks": round(ek.required_independent_blocks(0.03, sigma), 1),
            "n_days": round(ek.required_independent_blocks(0.03, sigma) * horizon),
            "n_years": round(ek.required_independent_blocks(0.03, sigma) * horizon / TRADING_DAYS_PER_YEAR, 1),
        },
        "horizon": horizon,
    }


# --------------------------------------------------------------------------- #
# 风格归因：组合超额里有多少是风格、有多少是选股
# --------------------------------------------------------------------------- #


def _open(bar: dict[str, dict[str, float]], date: str) -> float | None:
    value = bar.get(date, {}).get("open")
    return value if value and value > 0 else None


def style_periods(
    holdings: Sequence[dict[str, Any]],
    bars: dict[str, dict[str, dict[str, float]]],
    panel: dict[str, dict[str, dict[str, float]]],
    *,
    min_xsec: int = MIN_XSEC,
) -> list[dict[str, Any]]:
    """把逐次调仓的持仓转成"一个区间一份截面"的归因输入。"""
    panel_dates = sorted(panel)
    periods: list[dict[str, Any]] = []
    for current, following in zip(holdings, holdings[1:]):
        if current.get("fold") != following.get("fold"):
            continue
        start, end = current["date"], following["date"]
        if start >= end:
            continue
        exposures_date = next((d for d in reversed(panel_dates) if d <= start), None)
        if exposures_date is None:
            continue
        exposures = panel[exposures_date]
        returns: dict[str, float] = {}
        for symbol, series in bars.items():
            entry, exit_price = _open(series, start), _open(series, end)
            if entry and exit_price:
                # 归因层统一用小数（0.02 = 2%），百分号只在展示时乘。
                # 这里若先乘 100，attribute_period 再乘一次，超额会放大 100 倍。
                returns[symbol] = exit_price / entry - 1.0
        universe = sorted(s for s in returns if s in exposures)
        if len(universe) < min_xsec:
            continue
        periods.append(
            {
                "start": start,
                "end": end,
                "returns": {s: returns[s] for s in universe},
                "exposures": {s: exposures[s] for s in universe},
                "portfolio": [s for s in current["symbols"] if s in returns],
                "benchmark": universe,
            }
        )
    return periods


def evaluate_style(
    holdings: Sequence[dict[str, Any]],
    bars: dict[str, dict[str, dict[str, float]]],
    histories: Sequence[dict[str, Any]],
    *,
    clusters: int = st.DEFAULT_CLUSTERS,
) -> dict[str, Any]:
    grouping = st.correlation_clusters(histories, clusters=clusters)
    panel = st.style_panel(histories, grouping)
    periods = style_periods(holdings, bars, panel)
    result = st.attribute(periods)
    sizes = [len(item["symbols"]) for item in holdings]
    return {
        "clusters": grouping["table"],
        "summary": result["summary"],
        "periods": result["periods"][-12:],
        "n_periods": len(result["periods"]),
        "mean_holdings": round(statistics.fmean(sizes), 1) if sizes else None,
        "max_holdings": max(sizes) if sizes else None,
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def _bootstrap_excess(excess: list[float], *, iters: int = 800) -> dict[str, Any]:
    if not excess:
        return {"mean": None, "p": None}
    block = max(2, min(len(excess) // 3, 12))
    return ek.block_bootstrap_mean(excess, block=block, iters=iters)


def run(
    *,
    days: int = DAYS,
    universe: str = "hs300",
    max_symbols: int = DEFAULT_SYMBOLS_N,
    horizon: int = HORIZON,
    k_folds: int = K_FOLDS,
    rebalance: int = REBALANCE,
    top_fraction: float = TOP_FRACTION,
    max_turnover: float = MAX_TURNOVER,
    offline: bool = False,
    refresh: bool = False,
    iters: int = BOOTSTRAP_ITERS,
    symbols: Sequence[str] | None = None,
    pit: str | None = None,
) -> dict[str, Any]:
    members: Callable[[str], set[str]] | None = None
    pit_histories: list[dict[str, Any]] | None = None
    pit_benchmark: dict[str, Any] | None = None
    if pit:
        # PIT 池：面板用成分并集（含退市股），下单与等权基准都按当日成分过滤。
        # 不传 members 就等于"用今天的名单回看历史"，这正是旧口径的问题所在。
        from research import universe as ru   # 延迟导入：研究层可选依赖

        info = ru.prepare_cache(pit)
        cache_dir = Path(info["dir"])
        # 面板直接从 PIT 缓存读：xsec.load_histories 走的是"当前成分"的抓取缓存，
        # 里面没有退市股，等于把幸存者偏差又带回来了。
        pit_histories = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(cache_dir.glob("*.json"))
        ]
        bench_path = Path(info["members"]).parent / "benchmark.json"
        dataset_bench = Path(".cache") / "research" / pit / "benchmark.json"
        if not bench_path.is_file() and dataset_bench.is_file():
            bench_path = dataset_bench
        if bench_path.is_file():
            payload = json.loads(bench_path.read_text(encoding="utf-8"))
            # baostock 原始行是字符串：不转 float，xsec 里一除就 TypeError
            payload["rows"] = [
                {**row, **{key: float(row[key]) for key in
                           ("open", "high", "low", "close", "volume", "amount")
                           if row.get(key) not in (None, "")}}
                for row in payload.get("rows", [])[-days:]
            ]
            payload.setdefault("asset_type", "index")
            pit_benchmark = payload
        symbols = [history["symbol"] for history in pit_histories]
        members = ru.members_loader(Path(info["members"]))
        universe = f"pit:{pit}"
    elif symbols is None:
        symbols = load_universe(universe, max_symbols, refresh=refresh, offline=offline)
    else:
        universe = "custom"

    if pit_histories is not None:
        histories, failures = pit_histories, []
    else:
        histories, failures = xsec.load_histories(symbols, days, refresh=refresh, offline=offline)
    if len(histories) < MIN_XSEC:
        raise MarketDataError(f"可用个股只有 {len(histories)} 只，横截面至少需要 {MIN_XSEC} 只")
    benchmark = pit_benchmark or load_benchmark(days, refresh=refresh, offline=offline)
    if benchmark is None:
        raise MarketDataError("协议需要基准行情（相对强弱与 regime 信号都要用）")

    bars = ex.build_bars(histories)
    flags = ex.tradability(histories)
    bundle = build_records(histories, benchmark, horizon=horizon)
    records = bundle["records"]
    dates = bundle["dates"]
    lockbox = lockbox_split(dates)
    dev_dates = lockbox["dev_dates"]
    dev_set = set(dev_dates)
    dev_records = [record for record in records if record["date"] in dev_set]
    folds = purged_kfold(dev_dates, k=k_folds, horizon=horizon)

    walk = run_walkforward(dev_records, folds, horizon=horizon, iters=iters)
    calendar = [row["date"] for row in benchmark["rows"]]
    dev_calendar = [date for date in calendar if dev_dates and date <= dev_dates[-1]]
    dev_calendar = [date for date in dev_calendar if date >= dev_dates[0]]

    strategy = walkforward_nav(dev_calendar, bars, flags, walk, folds, rebalance=rebalance,
                               top_fraction=top_fraction, max_turnover=max_turnover,
                               members=members)
    benchmark_ew = equal_weight_nav(dev_calendar, bars, rebalance=rebalance, members=members)

    strategy_stat = nav_metrics(strategy, len(strategy["nav"]))
    bench_stat = nav_metrics(benchmark_ew, len(benchmark_ew["nav"]))
    excess_periods = period_excess(strategy, benchmark_ew, step=max(rebalance, horizon))

    # 同日收盘对照：同一批分数，只改成交时点
    close_results = []
    for fold in folds:
        scores = walk["fold_scores"].get(fold["fold"])
        if not scores:
            continue
        block = [date for date in calendar if fold["test_start"] <= date <= fold["test_end"]]
        prices = {symbol: {d: series[d]["close"] for d in series} for symbol, series in bars.items()}

        def pick(date: str, available: list[str], _scores=scores) -> list[str]:
            day = {s: _scores[(date, s)] for s in available if (date, s) in _scores}
            return xsec.top_n(day, top_fraction)

        close_results.append(
            xsec.simulate(block, prices, pick, rebalance_every=rebalance, max_turnover=max_turnover)
        )
    strategy_close = stitch(close_results)
    close_stat = nav_metrics(strategy_close, len(strategy_close["nav"]))

    target_a = evaluate_target_a(histories, benchmark, dev_dates, horizon=horizon, iters=iters)
    target_c = evaluate_target_c(histories, benchmark, dev_calendar, bars, flags,
                                 rebalance=rebalance, iters=800)
    feature_table = feature_ic_table(dev_records, folds, horizon=horizon)
    randomness = random_signal_harness(dev_records, folds, horizon=horizon, seeds=20)
    weak = weak_signal_harness(histories, dev_records, folds, horizon=horizon)
    sigma = walk["sigma_within"] or weak["sigma"] or 0.13
    sigma_theory = 1.0 / math.sqrt(len(histories))

    # MDI 对照：模型因子对标签分位的解释份额
    test_records = [r for r in dev_records if r["date"] in {d for fold in folds for d in fold["test_dates"]}]
    importance = ek.mdi_r2(test_records, walk["fold_models"][folds[0]["fold"]]["features"]) if folds and walk["fold_models"] else []

    style_result = evaluate_style(
        strategy["holdings"], bars, histories, clusters=st.DEFAULT_CLUSTERS
    )

    # ---------------- 多重检验记账 ----------------
    ledger = ek.TrialLedger()
    base_key = {
        "study": "protocol",
        "horizon": horizon,
        "universe": f"{universe}:{len(histories)}",
        "execution": "t1_open",
    }
    ledger.record(**base_key, target="B", feature="model_3f", split="wf5",
                  metric="ic", value=walk["pooled"]["mean"], p=walk["pooled"]["p"],
                  ci_low=walk["pooled"]["ci_low"], ci_high=walk["pooled"]["ci_high"],
                  n_blocks=walk["pooled"]["n_blocks"],
                  note="5 折 purged walk-forward 汇总 IC")
    for row in feature_table:
        ledger.record(**base_key, target="B", feature=row["key"], split="wf5-oos",
                      metric="ic", value=row["mean"], p=row["p"],
                      ci_low=row["ci_low"], ci_high=row["ci_high"], note="单因子样本外 IC")
    ledger.record(**{**base_key, "execution": "same_close"}, target="B", feature="model_3f",
                  split="same_close",
                  metric="excess", value=close_stat["total_return"] - bench_stat["total_return"],
                  note="同日收盘成交（旧口径）")

    horizon_variants = []
    for variant in sorted({5, horizon, 60}):
        if variant == horizon:
            pooled = walk["pooled"]
        else:
            bundle_v = build_records(histories, benchmark, horizon=variant)
            dev_v = [r for r in bundle_v["records"] if r["date"] in dev_set]
            folds_v = purged_kfold(dev_dates, k=k_folds, horizon=variant)
            pooled = run_walkforward(dev_v, folds_v, horizon=variant, iters=800)["pooled"]
        horizon_variants.append(
            {
                "horizon": variant,
                "ic": pooled["mean"],
                "ci_low": pooled["ci_low"],
                "ci_high": pooled["ci_high"],
                "p": pooled["p"],
                "n_blocks": pooled["n_blocks"],
            }
        )
        ledger.record(**{**base_key, "horizon": variant}, target="B", feature="model_3f",
                      split="wf5", metric="ic", value=pooled["mean"], p=pooled["p"],
                      ci_low=pooled["ci_low"], ci_high=pooled["ci_high"],
                      note="标签周期变体")

    cap_variants = []
    for cap in sorted({0.2, 0.4, 0.6, 1.0}):
        run_v = walkforward_nav(dev_calendar, bars, flags, walk, folds, rebalance=rebalance,
                                top_fraction=top_fraction, max_turnover=cap)
        stat_v = nav_metrics(run_v, len(run_v["nav"]))
        excess_v = period_excess(run_v, benchmark_ew, step=max(rebalance, horizon))
        boot = _bootstrap_excess(excess_v)
        cap_variants.append(
            {
                "cap": cap,
                "total_return": stat_v["total_return"],
                "annual_turnover": stat_v["annual_turnover"],
                "cost_drag_bps": stat_v["cost_drag_bps"],
                "excess": round(stat_v["total_return"] - bench_stat["total_return"], 2),
                "p": boot["p"],
            }
        )
        ledger.record(**base_key, target="B", feature="model_3f", split="wf5",
                      metric="excess", value=cap_variants[-1]["excess"], p=boot["p"],
                      turnover=cap, note="换手上限变体")

    # 第二个股票池：只用于检验结论是否依赖 60 只这个抽样
    smaller = sorted(h["symbol"] for h in histories)[:30]
    histories_s, _ = xsec.load_histories(smaller, days, offline=True)
    if len(histories_s) >= MIN_XSEC:
        bundle_s = build_records(histories_s, benchmark, horizon=horizon)
        dev_s = [r for r in bundle_s["records"] if r["date"] in dev_set]
        folds_s = purged_kfold(dev_dates, k=k_folds, horizon=horizon)
        pooled_s = run_walkforward(dev_s, folds_s, horizon=horizon, iters=800)["pooled"]
        universe_variant = {"symbols": len(histories_s), **{
            k: pooled_s[k] for k in ("mean", "ci_low", "ci_high", "p")
        }}
        ledger.record(**{**base_key, "universe": f"{universe}:30"}, target="B", feature="model_3f",
                      split="wf5", metric="ic", value=pooled_s["mean"], p=pooled_s["p"],
                      ci_low=pooled_s["ci_low"], ci_high=pooled_s["ci_high"],
                      note="股票池缩小到 30 只")
    else:
        universe_variant = {"symbols": 0, "note": "样本不足"}

    ledger_summary = ledger.summary()
    # 同名次试验的夏普也要校正：台账里所有收益类试验都算"试过的次数"
    daily_returns = [
        strategy["nav"][i]["nav"] / strategy["nav"][i - 1]["nav"] - 1.0
        for i in range(1, len(strategy["nav"]))
        if strategy["nav"][i - 1]["nav"]
    ]
    sharpe = ek.deflated_sharpe(
        daily_returns,
        n_trials=max(1, len(cap_variants) + len(horizon_variants),
                     ledger_summary["n_sharpe_trials"]),
    )

    # ---------------- Lockbox（只评估一次） ----------------
    lockbox_result: dict[str, Any] = {"available": False}
    if lockbox["lockbox_dates"]:
        lock_set = set(lockbox["lockbox_dates"])
        train_all = [r for r in records if r["date"] in dev_set and r.get("label") is not None]
        screening_all = screen_features(train_all, horizon=horizon)
        model_all = fit_rank_model(train_all, screening_all["features"])
        lock_records = [r for r in records if r["date"] in lock_set]
        lock_scores = model_scores(lock_records, model_all)
        lock_ic = ek.ic_ci(daily_ic_series(lock_records, lock_scores), horizon=horizon, iters=800)
        lock_calendar = [d for d in calendar if lockbox["lockbox_start"] <= d]
        prices = {s: {d: v[d]["close"] for d in v} for s, v in bars.items()}

        def lock_pick(date: str, available: list[str]) -> list[str]:
            day = {s: lock_scores[(date, s)] for s in available if (date, s) in lock_scores}
            return xsec.top_n(day, top_fraction)

        lock_nav = ex.simulate_t1(lock_calendar, bars, lock_pick, flags=flags,
                                  rebalance_every=rebalance, cost_buy=xsec.BUY_COST,
                                  cost_sell=xsec.SELL_COST, max_turnover=max_turnover)
        lock_bench = equal_weight_nav(lock_calendar, bars, rebalance=rebalance)
        lockbox_result = {
            "available": True,
            "start": lockbox["lockbox_start"],
            "end": lockbox["lockbox_dates"][-1],
            "days": len(lockbox["lockbox_dates"]),
            "features": screening_all["features"],
            "ic": lock_ic,
            "strategy": nav_metrics(lock_nav, len(lock_nav["nav"])),
            "benchmark": nav_metrics(lock_bench, len(lock_bench["nav"])),
            "n_runs": 1,
        }

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "params": {
            "universe": universe,
            "symbols": len(histories),
            "days": days,
            "horizon": horizon,
            "k_folds": k_folds,
            "embargo": EMBARGO,
            "lockbox_years": LOCKBOX_YEARS,
            "rebalance": rebalance,
            "top_fraction": top_fraction,
            "max_turnover": max_turnover,
            "min_xsec": MIN_XSEC,
            "bootstrap_iters": iters,
            "cost_buy_bps": round(xsec.BUY_COST * 10000, 2),
            "cost_sell_bps": round(xsec.SELL_COST * 10000, 2),
            "execution": "T+1 开盘成交，过滤涨跌停/停牌，T+1 卖出限制",
            "benchmark": {"symbol": benchmark["symbol"], "name": benchmark["name"]},
            "failures": failures,
        },
        "targets": TARGETS,
        "coverage": {
            "dates_all": len(dates),
            "dates_dev": len(dev_dates),
            "dates_lockbox": len(lockbox["lockbox_dates"]),
            "lockbox_start": lockbox["lockbox_start"],
            "records_dev": len(dev_records),
            "trading_days": len(dev_calendar),
        },
        "folds": [{k: v for k, v in fold.items() if k not in ("train_dates", "test_dates")} for fold in folds],
        "walkforward": {
            "folds": walk["folds"],
            "pooled": walk["pooled"],
            "sigma_within": walk["sigma_within"],
            "sigma_pooled": walk["sigma_pooled"],
            "n_folds": len(walk["folds"]),
            "features_per_fold": [row["features"] for row in walk["folds"]],
        },
        "features": feature_table,
        "importance": importance,
        "stats": {
            "strategy": strategy_stat,
            "strategy_same_close": close_stat,
            "benchmark": bench_stat,
            "excess_vs_bench": round(strategy_stat["total_return"] - bench_stat["total_return"], 2),
            "execution_gap": round(strategy_stat["total_return"] - close_stat["total_return"], 2),
            "blocked": strategy["blocked"],
            "rebalances": len(strategy["rebalances"]),
            "excess_ci": _bootstrap_excess(excess_periods),
            "excess_win_rate": round(
                sum(1 for value in excess_periods if value > 0) / len(excess_periods) * 100, 1
            ) if excess_periods else None,
            "deflated_sharpe": sharpe,
        },
        "nav": {"strategy": strategy["nav"], "benchmark": benchmark_ew["nav"],
                "same_close": strategy_close["nav"]},
        "target_a": target_a,
        "target_c": {k: v for k, v in target_c.items() if k not in ("nav", "benchmark_nav")},
        "target_c_nav": {"strategy": target_c.get("nav", []), "benchmark": target_c.get("benchmark_nav", [])},
        "style": style_result,
        "harness": {"random": randomness, "weak": weak},
        "power": power_analysis(sigma, n_blocks=walk["pooled"]["n_blocks"] or 0,
                                sigma_theory=sigma_theory),
        "horizon_variants": horizon_variants,
        "cap_variants": cap_variants,
        "universe_variant": universe_variant,
        "ledger": ledger_summary,
        "lockbox": lockbox_result,
        "execution_summary": ex.execution_summary(flags),
        "board_mix": _board_mix(histories),
    }
    return report


def _board_mix(histories: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for history in histories:
        board = ex._board_of(history["symbol"], history.get("name") or "")
        counts[board] = counts.get(board, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# 呈现
# --------------------------------------------------------------------------- #

DASH = "—"


def _fmt(value: float | None, digits: int = 3, suffix: str = "", signed: bool = True) -> str:
    if value is None:
        return DASH
    template = f"{{:+.{digits}f}}" if signed else f"{{:.{digits}f}}"
    return template.format(value) + suffix


def _ci(stat: dict[str, Any], digits: int = 3) -> str:
    if stat.get("ci_low") is None:
        return DASH
    return f"[{_fmt(stat['ci_low'], digits)}, {_fmt(stat['ci_high'], digits)}]"


def _verdict(stat: dict[str, Any]) -> str:
    if stat.get("ci_low") is None:
        return "样本不足"
    if stat["ci_low"] > 0:
        return "显著为正"
    if stat["ci_high"] < 0:
        return "显著为负"
    return "与 0 无法区分"


def _target_pass(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """三条通过标准逐条判定，报告里只认这个结果。"""
    pooled = report["walkforward"]["pooled"]
    stats = report["stats"]
    target_a = report["target_a"]["metrics"]
    strongest_a = max(
        (item for item in target_a.values() if item.get("ci_low") is not None),
        key=lambda item: abs(item["mean"]),
        default=None,
    )
    target_c = report["target_c"]
    verdicts = {}
    verdicts["B"] = {
        "pass": bool(pooled["significant"] and stats["excess_vs_bench"] > 0),
        "why": (
            f"IC {_fmt(pooled['mean'], 4)} 的 95% 区间 {_ci(pooled, 4)} "
            f"{'不含 0' if pooled['significant'] else '覆盖 0'}；"
            f"净净值相对等权 {_fmt(stats['excess_vs_bench'], 2, '%')}"
        ),
    }
    verdicts["A"] = {
        "pass": bool(strongest_a and strongest_a["significant"]),
        "why": (
            f"最强风险标签 IC {_fmt(strongest_a['mean'])} 区间 {_ci(strongest_a)}"
            if strongest_a else "风险标签无有效样本"
        ),
    }
    calmar = target_c.get("calmar", {}) if target_c.get("available") else {}
    verdicts["C"] = {
        "pass": bool(
            target_c.get("available")
            and target_c["drawdown_improvement"] > 0
            and (calmar.get("strategy") or 0) > (calmar.get("benchmark") or 0)
        ),
        "why": (
            f"回撤改善 {_fmt(target_c['drawdown_improvement'], 2, 'pct')}，"
            f"切换成本 {_fmt(target_c['switch_cost_bps'], 0, 'bp/年')}，"
            f"Calmar {calmar.get('strategy')} vs {calmar.get('benchmark')}"
        ) if target_c.get("available") else "regime 信号样本不足",
    }
    return verdicts


def render_markdown(report: dict[str, Any]) -> str:
    params = report["params"]
    coverage = report["coverage"]
    pooled = report["walkforward"]["pooled"]
    stats = report["stats"]
    power = report["power"]
    harness = report["harness"]
    ledger = report["ledger"]
    verdicts = _target_pass(report)
    lines = [
        "# 评估协议：三目标、带区间的 IC、5 折 purged walk-forward",
        "",
        f"生成时间：{report['generated_at']}　股票池：{params['symbols']} 只 A 股个股"
        f"（{params['universe']}）　"
        f"开发区间：{coverage['dates_dev']} 个交易日　lockbox："
        f"{coverage['lockbox_start'] or DASH} 起 {coverage['dates_lockbox']} 个交易日",
        "",
        "## 0. 一句话结论",
        "",
        f"- 目标 B（选股）：IC {_fmt(pooled['mean'], 4)}，95% 区间 {_ci(pooled, 4)} → "
        f"**{_verdict(pooled)}**；净净值相对等权 {_fmt(stats['excess_vs_bench'], 2, '%')}。"
        f"通过标准：{'达标' if verdicts['B']['pass'] else '未达标'}。",
        f"- 目标 A（描述分）：{verdicts['A']['why']} → "
        f"通过标准：{'达标' if verdicts['A']['pass'] else '未达标'}。",
        f"- 目标 C（regime）：{verdicts['C']['why']} → "
        f"通过标准：{'达标' if verdicts['C']['pass'] else '未达标'}。",
        f"- 随机信号校准：{harness['random']['seeds']} 组 iid 随机信号，95% 区间覆盖 0 的比例 "
        f"{harness['random']['iid']['coverage']}%（误报 "
        f"{harness['random']['iid']['false_positive_rate']}%）；平滑随机（自相关）覆盖 "
        f"{harness['random']['persistent']['coverage']}%（误报 "
        f"{harness['random']['persistent']['false_positive_rate']}%）→ "
        f"{'通过' if harness['random']['pass'] else '未通过'}。",
        f"- 多重检验：台账共 {ledger['n_trials']} 次试验，原始 p<0.05 的 {ledger['n_raw_p05']} 个，"
        f"BH-FDR(q={ledger['q']}) 后剩 {ledger['n_bh_reject']} 个。",
        "",
        "## 1. 三个目标的定义（口径唯一来源：`protocol.TARGETS`）",
        "",
        "| 目标 | 标签 | 指标 | 通过标准 |",
        "| --- | --- | --- | --- |",
    ]
    for key, item in report["targets"].items():
        lines.append(f"| {key} {item['name']} | {item['label']} | {item['metric']} | {item['pass']} |")

    lines += [
        "",
        "## 2. 样本与功效：先问这套数据能不能看见 IC=0.03",
        "",
        f"- 开发区间 {coverage['dates_dev']} 个交易日，{params['horizon']} 日 horizon 下相当于 "
        f"{pooled['n_blocks']} 个独立块（重叠标签按块长 = horizon 折算）。",
        f"- 实测 IC 抽样标准差 σ = {power['sigma_ic']}（折内逐日 IC 的标准差；"
        f"理论下限 1/√N = {power['sigma_theory']}），"
        f"当前 {pooled['n_blocks']} 个独立块能检出的最小 IC ≈ "
        f"**{_fmt(power['detectable_now'], 3)}**（α=0.05，power=0.8）。",
        f"- 要检测 IC=0.03，需要 **{power['for_ic_003']['n_blocks']} 个独立 "
        f"{params['horizon']} 日周期**，约 {power['for_ic_003']['n_days']} 个交易日 / "
        f"{power['for_ic_003']['n_years']} 年——不是 20 个。",
        "",
        "| 想检出的 IC | 需要的独立块 | 按 20 日 / 5 日 / 60 日折算的交易日 |",
        "| ---: | ---: | --- |",
    ]
    for row in power["table"]:
        lines.append(
            f"| {row['ic']:.2f} | {row['n_blocks']} | "
            f"{row['n_days']['h20']} / {row['n_days']['h5']} / {row['n_days']['h60']} |"
        )

    lines += [
        "",
        "σ 取值的敏感性（都按检测 IC=0.03、power=0.8 折算）：",
        "",
        "| σ 口径 | σ | 需要独立块 | 折合年数 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in power["sensitivity"]:
        lines.append(f"| {row['label']} | {row['sigma']} | {row['n_blocks']} | {row['n_years']} |")

    lines += [
        "",
        f"σ 的含义：{params['symbols']} 只股票的横截面里，单日 IC 本身的抽样噪声下限约 "
        f"1/√N ≈ {round(1 / math.sqrt(params['symbols']), 3)}，小 IC 因此需要很长的历史。"
        "降低 σ 的唯一有效手段是扩大横截面（更多股票），而不是多跑几个 horizon。"
        "这份数据是 5 折拼接的样本外 IC，块数按开发期交易日 / horizon 计。",
        "",
        "## 3. 目标 B：5 折 purged walk-forward 的选股分",
        "",
        f"切分：{report['walkforward']['n_folds']} 折连续分块，训练集在测试块两侧各剔除 "
        f"horizon {params['horizon']} 日 + embargo {params['embargo']} 日。"
        f"每折独立做因子表达筛选与拟合，测试块不参与任何选择。",
        "",
        "| 折 | 测试区间 | 训练样本 | 入选因子 | 测试 IC | 95% 区间 | p |",
        "| ---: | --- | ---: | --- | ---: | --- | ---: |",
    ]
    for row in report["walkforward"]["folds"]:
        lines.append(
            f"| {row['fold']} | {row['test_start']} → {row['test_end']} | {row['n_train']} | "
            f"{', '.join(row['features'])} | {_fmt(row['ic'], 4)} | "
            f"[{_fmt(row['ci_low'], 4)}, {_fmt(row['ci_high'], 4)}] | {_fmt(row['p'], 3, '', False)} |"
        )
    lines += [
        "",
        f"**汇总（跨折拼接）**：IC = **{_fmt(pooled['mean'], 4)}**，95% 区间 **{_ci(pooled, 4)}**，"
        f"p = {_fmt(pooled['p'], 3, '', False)}，正 IC 日占比 {pooled['positive_rate']}% → "
        f"**{_verdict(pooled)}**。",
        "",
        "### 3.1 净净值（T+1 开盘成交、扣费）",
        "",
        "| 组合 | 总收益 | 年化 | 最大回撤 | 夏普 | 年换手 | 成本(bp/年) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, stat in (
        ("新模型（T+1 开盘）", stats["strategy"]),
        ("新模型（同日收盘，旧口径）", stats["strategy_same_close"]),
        ("等权基准（T+1，不计成本）", stats["benchmark"]),
    ):
        lines.append(
            f"| {label} | {_fmt(stat['total_return'], 2, '%')} | {_fmt(stat['annual_return'], 2, '%')} | "
            f"{_fmt(stat['max_drawdown'], 2, '%')} | {stat['sharpe'] if stat['sharpe'] is not None else DASH} | "
            f"{_fmt(stat['annual_turnover'], 0, '%', False)} | {stat['cost_drag_bps']:.0f} |"
        )
    blocked = stats["blocked"]
    lines += [
        "",
        f"- 执行差异：同日收盘口径 {_fmt(stats['strategy_same_close']['total_return'], 2, '%')} → "
        f"T+1 开盘口径 {_fmt(stats['strategy']['total_return'], 2, '%')}，"
        f"差 **{_fmt(stats['execution_gap'], 2, '%')}**。"
        + ("T+1 更差，这部分就是旧口径高估的量级。"
           if stats["execution_gap"] < 0
           else "这次 T+1 反而略好：执行口径的影响方向不稳定，"
                "所以两个口径都要报，不能只挑好看的那个。"),
        f"- 被过滤的委托：涨停买不进 {blocked['buy_limit_up']} 次、停牌买不进 "
        f"{blocked['buy_suspended']} 次、跌停卖不出 {blocked['sell_limit_down']} 次、"
        f"停牌卖不出 {blocked['sell_suspended']} 次。",
        f"- 逐窗口超额 95% 区间 {_ci(stats['excess_ci'])}，超额为正的窗口占 "
        f"{stats['excess_win_rate']}%。",
        f"- 通货紧缩夏普：SR={stats['deflated_sharpe']['sr']}，"
        f"{stats['deflated_sharpe']['n_trials']} 次试验的期望最大值 SR0="
        f"{stats['deflated_sharpe']['sr0']}，DSR={stats['deflated_sharpe']['dsr']}。",
        "",
        "### 3.2 单因子样本外 IC（多重检验的输入）",
        "",
        "| 因子族 | 表达 | IC | 95% 区间 | p | 判定 |",
        "| --- | --- | ---: | --- | ---: | --- |",
    ]
    for row in report["features"]:
        lines.append(
            f"| {row['family']} | {row['key']} | {_fmt(row['mean'], 4)} | "
            f"[{_fmt(row['ci_low'], 4)}, {_fmt(row['ci_high'], 4)}] | "
            f"{_fmt(row['p'], 3, '', False)} | {_verdict(row)} |"
        )
    if report["importance"]:
        lines += [
            "",
            "MDI 对照（线性模型版的 R² 损失，不是树模型不纯度）：",
            "",
            "| 因子 | 全模型 R² | 剔除后 ΔR² | 占比 | 置换 ΔR² |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for row in report["importance"]:
            lines.append(
                f"| {row['feature']} | {row['full_r2']} | {row['drop_r2']} | "
                f"{row['drop_share']}% | {row['permutation_drop']} |"
            )

    target_a = report["target_a"]
    lines += [
        "",
        "## 4. 目标 A：描述分对风险状态有没有信息",
        "",
        f"用现役 v3 规则评分对四个风险标签做横截面 IC（每 {target_a['stride']} 个交易日一个截面，"
        f"共 {target_a['dates']} 个截面）。{target_a['interpretation']}。",
        "",
        "| 风险标签 | IC | 95% 区间 | p | 判定 |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for key, label in (("vol", "未来波动率"), ("mdd", "未来最大回撤"),
                       ("downside", "未来下行波动"), ("var95", "未来 5% VaR")):
        stat = target_a["metrics"][key]
        lines.append(
            f"| {label} | {_fmt(stat['mean'], 4)} | {_ci(stat, 4)} | "
            f"{_fmt(stat['p'], 3, '', False)} | {_verdict(stat)} |"
        )

    target_c = report["target_c"]
    lines += ["", "## 5. 目标 C：regime 仓位与回撤", ""]
    if target_c.get("available"):
        calmar = target_c["calmar"]
        lines += [
            f"信号：全池平均评分 → 滚动 250 日分位 → 目标仓位 0.3–1.0，平均仓位 "
            f"{target_c['exposure_mean']}%，每 {params['rebalance']} 日调仓，T+1 开盘生效。",
            "",
            "| 指标 | regime 组合 | 满仓等权 | 差 |",
            "| --- | ---: | ---: | ---: |",
            f"| 总收益 | {_fmt(target_c['strategy']['total_return'], 2, '%')} | "
            f"{_fmt(target_c['benchmark']['total_return'], 2, '%')} | "
            f"{_fmt(target_c['return_change'], 2, '%')} |",
            f"| 最大回撤 | {_fmt(target_c['strategy']['max_drawdown'], 2, '%')} | "
            f"{_fmt(target_c['benchmark']['max_drawdown'], 2, '%')} | "
            f"改善 {_fmt(target_c['drawdown_improvement'], 2, 'pct')} |",
            f"| Calmar | {calmar['strategy']} | {calmar['benchmark']} | "
            f"{_fmt((calmar['strategy'] or 0) - (calmar['benchmark'] or 0), 2)} |",
            "",
            f"- 切换成本 {_fmt(target_c['switch_cost_bps'], 0, 'bp/年')}，年换手 "
            f"{_fmt(target_c['annual_turnover'], 0, '%', False)}。",
            f"- 逐窗口相对满仓等权的超额 95% 区间 {_ci(target_c['excess_ci'])}。",
            f"- {target_c['note']}",
        ]
    else:
        lines.append(f"- {target_c.get('note', '未评估')}")

    style = report["style"]
    summary = style["summary"]
    lines += [
        "",
        "## 6. 风格基准：超额里有多少是风格",
        "",
        f"行业代理用日收益相关性聚类（{len(style['clusters'])} 簇），市值用 成交额/换手率 反推，"
        "动量用 12-1。每期做一次 Fama-MacBeth 横截面回归，把超额拆成风格贡献 + 残差。",
        "",
        "| 来源 | 逐期加总(%) | 均值(%) | t | 为正的期数 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    if summary:
        for key, label in (("contrib_size", "市值风格"), ("contrib_momentum", "动量风格"),
                           ("contrib_industry", "行业聚类"), ("alpha", "残差 alpha"),
                           ("excess", "合计超额")):
            item = summary.get(key, {})
            lines.append(
                f"| {label} | {item.get('sum')} | {item.get('mean')} | {item.get('t')} | "
                f"{item.get('positive_rate')}% |"
            )
        lines += [
            "",
            f"- 归因期数 {summary['n_periods']}（5 日一期，逐期加总不是复利，看均值和 t 更靠得住），"
            f"残差 alpha 占超额 {summary['alpha_share']}%；"
            f"平均市值暴露差 {summary['exposure_size']}、动量暴露差 {summary['exposure_momentum']}"
            "（正=组合比基准更偏小盘 / 更高动量）。",
            f"- 实际持仓数平均 {style['mean_holdings']} 只（最多 {style['max_holdings']}）："
            "受单次单边换手上限约束，卖不完的旧仓会留在组合里，所以这里的组合并不是严格的"
            " Top 20%，而是一个缓慢换手的宽篮子。",
            f"- 横截面回归平均 R² = {summary['mean_r2']}。",
        ]
    else:
        lines.append("- 归因样本不足")

    lines += [
        "",
        "| 簇 | 代表 | 成员数 | 成员 |",
        "| ---: | --- | ---: | --- |",
    ]
    for row in style["clusters"]:
        lines.append(
            f"| {row['cluster']} | {row['label']} | {row['size']} | "
            f"{', '.join(row['members'][:8])}{' …' if row['size'] > 8 else ''} |"
        )

    lines += [
        "",
        "## 7. 多重检验台账",
        "",
        f"本轮把试过的因子表达 / horizon / 换手上限 / universe / 执行口径全部记账，"
        f"共 **{ledger['n_trials']}** 条（同配置重复跑只更新数值）。",
        "",
        f"- 原始 p<0.05：{ledger['n_raw_p05']} 条；BH-FDR(q={ledger['q']}) 后："
        f"**{ledger['n_bh_reject']}** 条。",
        f"- 其中收益类试验 {ledger['n_sharpe_trials']} 次，已用于通货紧缩夏普的门槛。",
        f"- 按股票池分组的试验数：{ledger['by_universe']}。换股票池本身也是一种选择自由，"
        "所以同一套假设在不同抽样上各算一次；本轮不同抽样版本之间的超额动辄相差上百个百分点，"
        "而 IC 区间全都覆盖 0——这正是「点估计不能当结论」的实证。",
        "",
        "| 目标 | 因子 | horizon | 换手 | universe | 指标 | 数值 | p | q | BH 通过 |",
        "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for entry in ledger["entries"]:
        lines.append(
            f"| {entry.get('target')} | {entry.get('feature')} | {entry.get('horizon')} | "
            f"{entry.get('turnover') or DASH} | {entry.get('universe')} | {entry.get('metric')} | "
            f"{_fmt(entry.get('value'), 4, '', False)} | "
            f"{_fmt(entry.get('p'), 3, '', False)} | {_fmt(entry.get('qvalue'), 3, '', False)} | "
            f"{'是' if entry.get('bh_reject') else '否'} |"
        )

    harness = report["harness"]
    lines += [
        "",
        "## 8. Harness 校准",
        "",
        f"- **iid 随机信号**（{harness['random']['seeds']} 组，确定性哈希）：平均 IC "
        f"{_fmt(harness['random']['iid']['mean_ic'], 4)}，|IC| 均值 {harness['random']['iid']['mean_abs_ic']}，"
        f"覆盖 0 的比例 {harness['random']['iid']['coverage']}%，误报率 "
        f"{harness['random']['iid']['false_positive_rate']}%，区间宽度 "
        f"{harness['random']['iid']['ci_width_mean']}。",
        f"- **平滑随机信号**（5 日移动平均，人为制造自相关）：平均 IC "
        f"{_fmt(harness['random']['persistent']['mean_ic'], 4)}，覆盖 0 的比例 "
        f"{harness['random']['persistent']['coverage']}%，误报率 "
        f"{harness['random']['persistent']['false_positive_rate']}%，区间宽度 "
        f"{harness['random']['persistent']['ci_width_mean']}。块自助法主要就是要挡住这一种。"
        f"→ {'区间与显著性判定可信' if harness['random']['pass'] else '区间偏窄，显著性别当真'}。",
        f"- **12-1 动量**（已知弱信号）：IC {_fmt(harness['weak']['ic']['mean'])}，95% 区间 "
        f"{_ci(harness['weak']['ic'])}，p = {_fmt(harness['weak']['ic']['p'], 3, '', False)}，"
        f"检出 = {'是' if harness['weak']['detected'] else '否'}；此刻可检出下限 "
        f"{_fmt(harness['weak']['detectable_ic'], 3)}。"
        f"→ {harness['weak']['note']}",
        "",
        "## 9. Lockbox（只在最后评估一次）",
        "",
    ]
    lock = report["lockbox"]
    if lock.get("available"):
        lines += [
            f"区间 {lock['start']} → {lock['end']}（{lock['days']} 个交易日），"
            f"用开发期全部数据拟合、不做任何再选择，只跑一次。",
            "",
            f"- IC {_fmt(lock['ic']['mean'])}，95% 区间 {_ci(lock['ic'])} → {_verdict(lock['ic'])}。",
            f"- 净净值 {_fmt(lock['strategy']['total_return'], 2, '%')} vs 等权 "
            f"{_fmt(lock['benchmark']['total_return'], 2, '%')}（差 "
            f"{_fmt(lock['strategy']['total_return'] - lock['benchmark']['total_return'], 2, '%')}），"
            f"最大回撤 {_fmt(lock['strategy']['max_drawdown'], 2, '%')} vs "
            f"{_fmt(lock['benchmark']['max_drawdown'], 2, '%')}。",
        ]
    else:
        lines.append(f"- {report['coverage']['lockbox_start'] or '无 lockbox'}：样本深度不足。")

    lines += [
        "",
        "## 10. 与旧口径的对照",
        "",
        "旧做法：单次 60/40 时间切分、同日收盘成交、IC 只有点估计。新做法改变了三件事，"
        "每一项都会单独影响结论：",
        "",
        "1. 切分：单次 60/40 → 5 折 purged walk-forward（每折都在没见过的测试块上评估，"
        "再拼接）＋ 末 2 年 lockbox。",
        "2. 执行：同日收盘 → T+1 开盘，并过滤涨跌停/停牌。",
        f"3. 判断：IC 与 0 比较 → IC 的 95% 块自助区间，且按试验次数做 BH-FDR / 通货紧缩夏普。",
        "",
        "## 11. 通过标准与后续约束",
        "",
        "| 目标 | 通过 | 依据 |",
        "| --- | --- | --- |",
    ]
    for key, item in verdicts.items():
        lines.append(f"| {key} | {'达标' if item['pass'] else '未达标'} | {item['why']} |")
    lines += [
        "",
        "后续所有方向只引用：",
        "",
        "- 带 95% 区间的 IC（`evalkit.ic_ci`），不再出现「IC 为负所以无效」这种没有区间的结论；",
        "- T+1 开盘、扣费后的净净值，而不是同日收盘的名义净值；",
        "- 风格归因后的残差 alpha，而不是未剥离风格暴露的原始超额；",
        "- 记进台账并做过 BH-FDR / 通货紧缩的显著性，而不是单次 p 值。",
        "",
    ]
    return "\n".join(lines)


def _svg_nav(report: dict[str, Any], *, width: int = 880, height: int = 300) -> str:
    series = [
        ("新模型（T+1 开盘）", report["nav"]["strategy"], "#b34938", 1.9, ""),
        ("新模型（同日收盘）", report["nav"]["same_close"], "#8a5cc4", 1.4, ' stroke-dasharray="5 3"'),
        ("等权基准", report["nav"]["benchmark"], "#176b4d", 1.6, ""),
    ]
    series = [item for item in series if item[1]]
    if not series:
        return ""
    left, right, top, bottom = 52, 14, 26, 30
    values = [point["nav"] for _name, points, _c, _w, _d in series for point in points]
    low, high = min(values), max(values)
    span = max(high - low, 1e-6)
    low -= span * 0.08
    high += span * 0.08
    total = max(len(points) for _name, points, _c, _w, _d in series)

    def x_at(index: int) -> float:
        return left + (width - left - right) * (index / max(total - 1, 1))

    def y_at(value: float) -> float:
        return top + (height - top - bottom) * (1 - (value - low) / (high - low))

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="净值曲线">']
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fffdf7" rx="8"/>')
    for step in range(5):
        value = low + (high - low) * step / 4
        y = y_at(value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e2e5de"/>')
        parts.append(
            f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end" fill="#6e7b74" font-size="10" '
            f'font-family="ui-monospace,Menlo,monospace">{value:.2f}</text>'
        )
    for name, points, color, weight, dash in series:
        path = " ".join(
            f'{"M" if i == 0 else "L"}{x_at(i):.1f},{y_at(p["nav"]):.1f}'
            for i, p in enumerate(points)
        )
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{weight}"{dash}/>')
    for order, (name, _p, color, _w, _d) in enumerate(series):
        x = left + 4 + order * 170
        parts.append(f'<rect x="{x}" y="{top - 16}" width="10" height="3" fill="{color}"/>')
        parts.append(
            f'<text x="{x + 14}" y="{top - 11}" fill="#16221d" font-size="11" '
            f'font-family="-apple-system,Segoe UI,PingFang SC,sans-serif">{name}</text>'
        )
    dates = [p["date"] for p in series[0][1]]
    for step in range(4):
        index = round((len(dates) - 1) * step / 3)
        parts.append(
            f'<text x="{x_at(index):.1f}" y="{height - 10}" text-anchor="middle" fill="#6e7b74" '
            f'font-size="10" font-family="ui-monospace,Menlo,monospace">{dates[index]}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


STYLE = """
  .wf { margin: 0; padding: 0; color: var(--ink, #16221d); }
  .wf h3 { margin: 0 0 6px; font-size: 17px; }
  .wf h4 { margin: 18px 0 6px; font-size: 13.5px; }
  .wf .wf-sub { margin: 0 0 12px; color: var(--muted, #6e7b74); font-size: 12.5px; line-height: 1.7; }
  .wf .wf-scroll { overflow-x: auto; border: 1px solid var(--line, #d8ddd6); border-radius: 8px; background: var(--panel, rgba(255,255,252,.84)); }
  .wf table { width: 100%; border-collapse: collapse; font-size: 12.5px; white-space: nowrap; }
  .wf thead th { padding: 8px 10px; color: var(--muted, #6e7b74); font-weight: 600; font-size: 11px; border-bottom: 1px solid var(--line, #d8ddd6); background: rgba(23,107,77,.04); }
  .wf tbody td { padding: 7px 10px; border-bottom: 1px solid rgba(216,221,214,.7); }
  .wf tbody tr:last-child td { border-bottom: 0; }
  .wf .num { text-align: right; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-variant-numeric: tabular-nums; }
  .wf .up { color: var(--red, #b34938); }
  .wf .down { color: var(--green, #176b4d); }
  .wf .mute { color: var(--muted, #6e7b74); }
  .wf .wf-chart { border: 1px solid var(--line, #d8ddd6); border-radius: 8px; background: #fffdf7; padding: 6px 4px 0; }
  .wf .wf-warn { margin: 12px 0 0; padding: 9px 12px; border-left: 3px solid var(--amber, #aa7627); border-radius: 0 6px 6px 0; background: rgba(170,118,39,.09); color: #7a5418; font-size: 12px; line-height: 1.75; }
  .wf .wf-tag { display: inline-block; margin-right: 6px; padding: 1px 6px; border-radius: 4px; font: 700 10px/1.6 ui-monospace, Menlo, monospace; }
  .wf .ok { background: rgba(23,107,77,.12); color: #176b4d; }
  .wf .no { background: rgba(179,73,56,.12); color: #b34938; }
  .wf .wf-factor { margin-top: 18px; }
"""


def _tone(value: float | None) -> str:
    if value is None:
        return ""
    return "up" if value > 0 else "down" if value < 0 else ""


def render_html(report: dict[str, Any]) -> str:
    params = report["params"]
    coverage = report["coverage"]
    pooled = report["walkforward"]["pooled"]
    stats = report["stats"]
    power = report["power"]
    harness = report["harness"]
    ledger = report["ledger"]
    verdicts = _target_pass(report)

    fold_rows = "".join(
        f"<tr><td class='num'>{row['fold']}</td><td>{row['test_start']} → {row['test_end']}</td>"
        f"<td class='num'>{row['n_train']}</td><td>{', '.join(row['features'])}</td>"
        f"<td class='num'>{_fmt(row['ic'], 4)}</td>"
        f"<td class='num mute'>[{_fmt(row['ci_low'], 4)}, {_fmt(row['ci_high'], 4)}]</td>"
        f"<td class='num'>{_fmt(row['p'], 3, '', False)}</td></tr>"
        for row in report["walkforward"]["folds"]
    )
    feature_rows = "".join(
        f"<tr><td>{row['family']}</td><td>{row['key']}</td>"
        f"<td class='num {_tone(row['mean'])}'>{_fmt(row['mean'], 4)}</td>"
        f"<td class='num mute'>[{_fmt(row['ci_low'], 4)}, {_fmt(row['ci_high'], 4)}]</td>"
        f"<td class='num'>{_fmt(row['p'], 3, '', False)}</td>"
        f"<td>{_verdict(row)}</td></tr>"
        for row in report["features"]
    )
    power_rows = "".join(
        f"<tr><td class='num'>{row['ic']:.2f}</td><td class='num'>{row['n_blocks']}</td>"
        f"<td class='num'>{row['n_days']['h20']}</td><td class='num'>{row['n_days']['h5']}</td>"
        f"<td class='num'>{row['n_days']['h60']}</td></tr>"
        for row in power["table"]
    )
    risk_rows = "".join(
        f"<tr><td>{label}</td><td class='num {_tone(stat['mean'])}'>{_fmt(stat['mean'], 4)}</td>"
        f"<td class='num mute'>{_ci(stat, 4)}</td>"
        f"<td class='num'>{_fmt(stat['p'], 3, '', False)}</td><td>{_verdict(stat)}</td></tr>"
        for key, label in (("vol", "未来波动率"), ("mdd", "未来最大回撤"),
                           ("downside", "未来下行波动"), ("var95", "未来 5% VaR"))
        for stat in [report["target_a"]["metrics"][key]]
    )
    ledger_rows = "".join(
        f"<tr><td>{entry.get('target')}</td><td>{entry.get('feature')}</td>"
        f"<td class='num'>{entry.get('horizon')}</td><td class='num'>{entry.get('turnover') or DASH}</td>"
        f"<td class='mute'>{entry.get('universe')}</td><td>{entry.get('metric')}</td>"
        f"<td class='num'>{_fmt(entry.get('value'), 4, '', False)}</td>"
        f"<td class='num'>{_fmt(entry.get('p'), 3, '', False)}</td>"
        f"<td class='num'>{_fmt(entry.get('qvalue'), 3, '', False)}</td>"
        f"<td>{'是' if entry.get('bh_reject') else '否'}</td></tr>"
        for entry in ledger["entries"]
    )
    style = report["style"]
    summary = style["summary"]
    style_rows = "".join(
        f"<tr><td>{label}</td><td class='num {_tone(summary.get(key, {}).get('sum'))}'>"
        f"{summary.get(key, {}).get('sum')}</td>"
        f"<td class='num'>{summary.get(key, {}).get('mean')}</td>"
        f"<td class='num'>{summary.get(key, {}).get('t')}</td>"
        f"<td class='num'>{summary.get(key, {}).get('positive_rate')}%</td></tr>"
        for key, label in (("contrib_size", "市值风格"), ("contrib_momentum", "动量风格"),
                           ("contrib_industry", "行业聚类"), ("alpha", "残差 alpha"),
                           ("excess", "合计超额"))
    ) if summary else "<tr><td colspan='5'>归因样本不足</td></tr>"
    cluster_rows = "".join(
        f"<tr><td class='num'>{row['cluster']}</td><td>{row['label']}</td>"
        f"<td class='num'>{row['size']}</td>"
        f"<td class='mute'>{', '.join(row['members'][:10])}{' …' if row['size'] > 10 else ''}</td></tr>"
        for row in style["clusters"]
    )
    verdict_rows = "".join(
        f"<tr><td>{key}</td><td><span class='wf-tag {'ok' if item['pass'] else 'no'}'>"
        f"{'达标' if item['pass'] else '未达标'}</span></td><td class='mute'>{item['why']}</td></tr>"
        for key, item in verdicts.items()
    )
    lock = report["lockbox"]
    lock_html = (
        f"<p class='wf-sub'>区间 {lock['start']} → {lock['end']}（{lock['days']} 个交易日），"
        f"用开发期全部数据拟合后只评估这一次。</p>"
        f"<ul class='wf-sub'><li>IC {_fmt(lock['ic']['mean'])}，95% 区间 {_ci(lock['ic'])} → "
        f"{_verdict(lock['ic'])}</li>"
        f"<li>净净值 {_fmt(lock['strategy']['total_return'], 2, '%')} vs 等权 "
        f"{_fmt(lock['benchmark']['total_return'], 2, '%')}，最大回撤 "
        f"{_fmt(lock['strategy']['max_drawdown'], 2, '%')} vs "
        f"{_fmt(lock['benchmark']['max_drawdown'], 2, '%')}</li></ul>"
        if lock.get("available")
        else "<p class='wf-sub'>lockbox 不可用（样本深度不足）。</p>"
    )
    target_c = report["target_c"]
    if target_c.get("available"):
        calmar = target_c["calmar"]
        c_html = (
            f"<p class='wf-sub'>全池平均评分 → 滚动 250 日分位 → 目标仓位 0.3–1.0，平均仓位 "
            f"{target_c['exposure_mean']}%，T+1 开盘生效。</p>"
            "<div class='wf-scroll'><table><thead><tr><th>指标</th><th>regime</th>"
            "<th>满仓等权</th><th>差</th></tr></thead><tbody>"
            f"<tr><td>总收益</td><td class='num'>{_fmt(target_c['strategy']['total_return'], 2, '%')}</td>"
            f"<td class='num'>{_fmt(target_c['benchmark']['total_return'], 2, '%')}</td>"
            f"<td class='num {_tone(target_c['return_change'])}'>{_fmt(target_c['return_change'], 2, '%')}</td></tr>"
            f"<tr><td>最大回撤</td><td class='num'>{_fmt(target_c['strategy']['max_drawdown'], 2, '%')}</td>"
            f"<td class='num'>{_fmt(target_c['benchmark']['max_drawdown'], 2, '%')}</td>"
            f"<td class='num'>改善 {_fmt(target_c['drawdown_improvement'], 2, 'pct')}</td></tr>"
            f"<tr><td>Calmar</td><td class='num'>{calmar['strategy']}</td>"
            f"<td class='num'>{calmar['benchmark']}</td><td class='num'>—</td></tr>"
            "</tbody></table></div>"
            f"<p class='wf-sub'>切换成本 {_fmt(target_c['switch_cost_bps'], 0, 'bp/年')}，"
            f"逐窗口超额 95% 区间 {_ci(target_c['excess_ci'])}。</p>"
        )
    else:
        c_html = f"<p class='wf-sub'>{target_c.get('note', '未评估')}</p>"

    return f"""<section class="wf" aria-labelledby="protocolTitle">
<style>{STYLE}</style>
<h3 id="protocolTitle">评估协议：带区间的 IC 与 5 折 purged walk-forward</h3>
<p class="wf-sub">{params['symbols']} 只 A 股 · 开发期 {coverage['dates_dev']} 个交易日 ·
lockbox {coverage['lockbox_start'] or DASH} 起 {coverage['dates_lockbox']} 个交易日 ·
{params['execution']} · 自助重采样 {params['bootstrap_iters']} 次</p>

<div class="wf-warn">
目标 B（选股）IC <b>{_fmt(pooled['mean'], 4)}</b>，95% 区间 <b>{_ci(pooled, 4)}</b> →
<b>{_verdict(pooled)}</b>；净净值相对等权 {_fmt(stats['excess_vs_bench'], 2, '%')}。
目标 A {('达标' if verdicts['A']['pass'] else '未达标')}，
目标 C {('达标' if verdicts['C']['pass'] else '未达标')}。
台账 {ledger['n_trials']} 次试验，BH-FDR 后显著 {ledger['n_bh_reject']} 个。
</div>

<div class="wf-chart" style="margin-top:14px">{_svg_nav(report)}</div>

<div class="wf-factor">
<h4>1. 样本量与功效：这套数据能看见多大的 IC</h4>
<p class="wf-sub">{params['horizon']} 日 horizon 下开发期相当于 {pooled['n_blocks']} 个独立块；
实测折内逐日 IC 标准差 σ={power['sigma_ic']}（理论下限 1/√N={power['sigma_theory']}），
当前可检出下限 ≈ {_fmt(power['detectable_now'], 3)}。
若要检测 IC=0.03，需要 {power['for_ic_003']['n_blocks']} 个独立周期
（≈{power['for_ic_003']['n_days']} 个交易日 / {power['for_ic_003']['n_years']} 年）。</p>
<div class="wf-scroll"><table>
<thead><tr><th>目标 IC</th><th>需要的独立块</th><th>20 日折算交易日</th><th>5 日</th><th>60 日</th></tr></thead>
<tbody>{power_rows}</tbody></table></div>
</div>

<div class="wf-factor">
<h4>2. 目标 B：逐折 IC（每折都在没见过的测试块上）</h4>
<div class="wf-scroll"><table>
<thead><tr><th>折</th><th>测试区间</th><th>训练样本</th><th>入选因子</th><th>IC</th><th>95% 区间</th><th>p</th></tr></thead>
<tbody>{fold_rows}</tbody></table></div>
</div>

<div class="wf-factor">
<h4>3. 净净值：T+1 开盘 vs 同日收盘</h4>
<div class="wf-scroll"><table>
<thead><tr><th>组合</th><th>总收益</th><th>年化</th><th>最大回撤</th><th>夏普</th><th>年换手</th><th>成本(bp/年)</th></tr></thead>
<tbody>
<tr><td><b>新模型（T+1 开盘）</b></td>
<td class="num {_tone(stats['strategy']['total_return'])}">{_fmt(stats['strategy']['total_return'], 2, '%')}</td>
<td class="num">{_fmt(stats['strategy']['annual_return'], 2, '%')}</td>
<td class="num">{_fmt(stats['strategy']['max_drawdown'], 2, '%')}</td>
<td class="num">{stats['strategy']['sharpe']}</td>
<td class="num">{_fmt(stats['strategy']['annual_turnover'], 0, '%', False)}</td>
<td class="num">{stats['strategy']['cost_drag_bps']:.0f}</td></tr>
<tr><td>新模型（同日收盘，旧口径）</td>
<td class="num {_tone(stats['strategy_same_close']['total_return'])}">{_fmt(stats['strategy_same_close']['total_return'], 2, '%')}</td>
<td class="num">{_fmt(stats['strategy_same_close']['annual_return'], 2, '%')}</td>
<td class="num">{_fmt(stats['strategy_same_close']['max_drawdown'], 2, '%')}</td>
<td class="num">{stats['strategy_same_close']['sharpe']}</td>
<td class="num">{_fmt(stats['strategy_same_close']['annual_turnover'], 0, '%', False)}</td>
<td class="num">{stats['strategy_same_close']['cost_drag_bps']:.0f}</td></tr>
<tr><td>等权基准（T+1，不计成本）</td>
<td class="num {_tone(stats['benchmark']['total_return'])}">{_fmt(stats['benchmark']['total_return'], 2, '%')}</td>
<td class="num">{_fmt(stats['benchmark']['annual_return'], 2, '%')}</td>
<td class="num">{_fmt(stats['benchmark']['max_drawdown'], 2, '%')}</td>
<td class="num">{stats['benchmark']['sharpe']}</td>
<td class="num">{_fmt(stats['benchmark']['annual_turnover'], 0, '%', False)}</td>
<td class="num">{stats['benchmark']['cost_drag_bps']:.0f}</td></tr>
</tbody></table></div>
<p class="wf-sub">执行口径差额 {_fmt(stats['execution_gap'], 2, '%')}；被过滤委托：涨停买不进
{stats['blocked']['buy_limit_up']}、停牌买不进 {stats['blocked']['buy_suspended']}、
跌停卖不出 {stats['blocked']['sell_limit_down']}、停牌卖不出 {stats['blocked']['sell_suspended']}。
逐窗口超额 95% 区间 {_ci(stats['excess_ci'])}，为正窗口 {stats['excess_win_rate']}%。
通货紧缩夏普 SR0={stats['deflated_sharpe']['sr0']}，DSR={stats['deflated_sharpe']['dsr']}。</p>
</div>

<div class="wf-factor">
<h4>4. 单因子样本外 IC（带区间）</h4>
<div class="wf-scroll"><table>
<thead><tr><th>因子族</th><th>表达</th><th>IC</th><th>95% 区间</th><th>p</th><th>判定</th></tr></thead>
<tbody>{feature_rows}</tbody></table></div>
</div>

<div class="wf-factor">
<h4>5. 目标 A：描述分 vs 风险标签</h4>
<div class="wf-scroll"><table>
<thead><tr><th>风险标签</th><th>IC</th><th>95% 区间</th><th>p</th><th>判定</th></tr></thead>
<tbody>{risk_rows}</tbody></table></div>
<p class="wf-sub">{report['target_a']['interpretation']}；共 {report['target_a']['dates']} 个截面。</p>
</div>

<div class="wf-factor">
<h4>6. 目标 C：regime 仓位</h4>
{c_html}
</div>

<div class="wf-factor">
<h4>7. 风格归因：风格贡献 vs 残差 alpha</h4>
<div class="wf-scroll"><table>
<thead><tr><th>来源</th><th>逐期加总(%)</th><th>均值(%)</th><th>t</th><th>为正期数占比</th></tr></thead>
<tbody>{style_rows}</tbody></table></div>
<p class="wf-sub">识别"选股"的只有残差 alpha；市值/动量/行业聚类三项都是可以被风格基准解释掉的部分。
残差占超额 {summary.get('alpha_share') if summary else DASH}%。5 日一期、逐期加总非复利。
实际持仓数平均 {style['mean_holdings']} 只（最多 {style['max_holdings']}），受换手上限约束，
组合不是严格的 Top 20%。</p>
<div class="wf-scroll" style="margin-top:10px"><table>
<thead><tr><th>簇</th><th>代表</th><th>成员数</th><th>成员</th></tr></thead>
<tbody>{cluster_rows}</tbody></table></div>
</div>

<div class="wf-factor">
<h4>8. 多重检验台账</h4>
<p class="wf-sub">{ledger['n_trials']} 次试验，原始 p&lt;0.05 共 {ledger['n_raw_p05']} 条，
BH-FDR(q={ledger['q']}) 后剩 {ledger['n_bh_reject']} 条。</p>
<div class="wf-scroll"><table>
<thead><tr><th>目标</th><th>因子</th><th>horizon</th><th>换手</th><th>universe</th>
<th>指标</th><th>数值</th><th>p</th><th>q</th><th>BH 通过</th></tr></thead>
<tbody>{ledger_rows}</tbody></table></div>
</div>

<div class="wf-factor">
<h4>9. Harness 校准</h4>
<p class="wf-sub">随机信号 {harness['random']['seeds']} 组 ×2（iid / 5 日平滑自相关）：
平均 IC {_fmt(harness['random']['iid']['mean_ic'], 4)} / {_fmt(harness['random']['persistent']['mean_ic'], 4)}，
覆盖 0 的比例 {harness['random']['iid']['coverage']}% / {harness['random']['persistent']['coverage']}%，
误报率 {harness['random']['iid']['false_positive_rate']}% / {harness['random']['persistent']['false_positive_rate']}%
→ {'区间可信' if harness['random']['pass'] else '区间偏窄，显著性别当真'}。<br>
12-1 动量（已知弱信号）：IC {_fmt(harness['weak']['ic']['mean'])}，区间 {_ci(harness['weak']['ic'])}，
检出 {'是' if harness['weak']['detected'] else '否'}；当前可检出下限
{_fmt(harness['weak']['detectable_ic'], 3)}。</p>
</div>

<div class="wf-factor">
<h4>10. Lockbox（只评估一次）</h4>
{lock_html}
</div>

<div class="wf-factor">
<h4>11. 通过标准</h4>
<div class="wf-scroll"><table>
<thead><tr><th>目标</th><th>判定</th><th>依据</th></tr></thead>
<tbody>{verdict_rows}</tbody></table></div>
</div>
</section>
"""


def render_preview(report: dict[str, Any]) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>评估协议 · 预览</title>
<style>
  :root {{
    --ink: #16221d; --muted: #6e7b74; --paper: #f4f2eb; --panel: rgba(255,255,252,.84);
    --line: #d8ddd6; --green: #176b4d; --red: #b34938; --amber: #aa7627;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--paper); color: var(--ink);
    font: 15px/1.65 -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; }}
  .wrap {{ max-width: 1080px; margin: 0 auto; padding: 34px 22px 56px; }}
  .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
    padding: 22px 22px 18px; box-shadow: 0 18px 50px rgba(31,48,40,.08); }}
  .label {{ color: var(--green); font: 700 11px/1.4 ui-monospace, Menlo, monospace;
    letter-spacing: .18em; margin-bottom: 4px; }}
  .title {{ margin: 0 0 16px; font-size: 20px; font-weight: 650; }}
</style>
</head>
<body>
  <div class="wrap"><div class="panel">
    <div class="label">EVALUATION PROTOCOL</div>
    <h2 class="title">三目标 · 带区间的 IC · 5 折 purged walk-forward</h2>
    {render_html(report)}
  </div></div>
</body>
</html>
"""


def write_reports(report: dict[str, Any], outdir: Path) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": outdir / "protocol.json",
        "markdown": outdir / "protocol.md",
        "html": outdir / "protocol_table.html",
        "preview": outdir / "protocol_preview.html",
    }
    paths["json"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["markdown"].write_text(render_markdown(report), encoding="utf-8")
    paths["html"].write_text(render_html(report), encoding="utf-8")
    paths["preview"].write_text(render_preview(report), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="评估协议：三目标 + 5 折 purged walk-forward（离线）")
    parser.add_argument("--days", type=int, default=DAYS, help="日线根数，默认 2400")
    parser.add_argument("--universe", default="hs300", help="股票池")
    parser.add_argument("--pit", default=None, help="用 PIT 池（如 pit-v1）：面板按成分并集、下单按当日成分")
    parser.add_argument("--max-symbols", type=int, default=DEFAULT_SYMBOLS_N, help="抽样数量")
    parser.add_argument("--horizon", type=int, default=HORIZON, help="主标签周期")
    parser.add_argument("--folds", type=int, default=K_FOLDS, help="折数")
    parser.add_argument("--rebalance", type=int, default=REBALANCE, help="调仓间隔")
    parser.add_argument("--top", type=float, default=TOP_FRACTION, help="持仓比例")
    parser.add_argument("--turnover", type=float, default=MAX_TURNOVER, help="单次单边换手上限")
    parser.add_argument("--iters", type=int, default=BOOTSTRAP_ITERS, help="自助重采样次数")
    parser.add_argument("--refresh", action="store_true", help="忽略缓存重新拉取")
    parser.add_argument("--offline", action="store_true", help="只用本地缓存")
    parser.add_argument("--outdir", default=str(REPORT_DIR / "protocol"), help="报告目录")
    args = parser.parse_args(argv)

    try:
        report = run(
            days=max(400, min(args.days, 3000)),
            universe=args.universe,
            max_symbols=args.max_symbols,
            horizon=max(2, args.horizon),
            k_folds=max(2, min(args.folds, 10)),
            rebalance=max(1, args.rebalance),
            top_fraction=min(0.9, max(0.05, args.top)),
            max_turnover=min(1.0, max(0.05, args.turnover)),
            offline=args.offline,
            refresh=args.refresh,
            iters=max(200, args.iters),
            pit=args.pit,
        )
    except (MarketDataError, ValueError) as exc:
        print(f"运行失败：{exc}", file=sys.stderr)
        return 2

    paths = write_reports(report, Path(args.outdir))
    pooled = report["walkforward"]["pooled"]
    stats = report["stats"]
    print(
        f"股票池 {report['params']['symbols']} 只，开发期 {report['coverage']['dates_dev']} 天，"
        f"lockbox {report['coverage']['dates_lockbox']} 天，{report['walkforward']['n_folds']} 折"
    )
    print(
        f"  目标 B：IC {pooled['mean']:+.4f} [{pooled['ci_low']:+.4f}, {pooled['ci_high']:+.4f}] "
        f"p={pooled['p']:.3f} → {_verdict(pooled)}"
    )
    print(
        f"  净净值：策略 {stats['strategy']['total_return']:+.2f}% vs 等权 "
        f"{stats['benchmark']['total_return']:+.2f}%（同日收盘口径 "
        f"{stats['strategy_same_close']['total_return']:+.2f}%）"
    )
    for key, item in _target_pass(report).items():
        print(f"  目标 {key}：{'达标' if item['pass'] else '未达标'} —— {item['why']}")
    random_harness = report["harness"]["random"]
    print(
        f"  试验台账 {report['ledger']['n_trials']} 条，BH-FDR 后显著 "
        f"{report['ledger']['n_bh_reject']} 条；随机信号覆盖 0 比例 "
        f"{random_harness['iid']['coverage']}% / {random_harness['persistent']['coverage']}%"
        f"（iid / 平滑）"
    )
    for key, path in paths.items():
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


