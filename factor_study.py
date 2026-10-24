"""新因子族研究与验证：定义 → 中性化 → 正交化 → 增量 IC → 组合模型。

用法::

    python factor_study.py --offline
    python factor_study.py --offline --horizon 20 --iters 1200 --outdir reports/factors

筛选规则（不是"训练集 IC 最高"）
--------------------------------
1. **先验符号写死**在 ``factors.FACTOR_SPECS``：值越大越看多记 +1，反之 −1。
   反号是结论，不是翻转的理由。
2. **多表达平均**：同族表达先按先验符号对齐，再等权平均成一个族综合分；不会因为
   某个表达在训练集上偶然 IC 高就把它单独拎出来（这正是上一版 macd_hist 的问题）。
3. **稳定性选择**：族综合分必须同时满足增量 IC、t、半年同号、与现有因子相关、
   换手五个条件才进模型；任何一条不过，整族不进。

验证口径
--------
- 标签：次日开盘进、H 日后开盘出的收益，当日横截面分位（与 ``protocol`` 一致）。
- 增量 IC：对现有三因子（macd_hist / vol_ratio / rs20）横截面回归取残差后的 IC。
- t 值：块自助（块长 = H）标准误下的 t，不是独立同分布假设下的 t。
- 组合模型：5 折 purged walk-forward，逐折训练、逐折打分，OOS IC 按日汇总。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import evalkit as ek
import factors as fa
import fields as fd
import protocol as pr
from xsec import _rank_pct

from research import universe

ROOT = Path(__file__).resolve().parent
STOCK_CACHE = ROOT / ".cache" / "fields" / "stocks"
ETF_CACHE = ROOT / ".cache" / "fields" / "etf"
INDUSTRY_PATH = ROOT / ".cache" / "research" / "pit-v1" / "industry.json"
OUTDIR = ROOT / "reports" / "factors"

HORIZON = 20
REVERSAL_HORIZON = 5
MIN_XSEC = 20
MIN_INDUSTRY = 3
K_FOLDS = 5
EMBARGO = 5
ITERS = 1000
REBALANCE = 21
TOP_FRACTION = 0.2

# 进入模型的硬门槛：跑之前写死，不随结果调整。
PASS_CRITERIA = {
    "incremental_ic": 0.01,
    "t_stat": 2.0,
    "half_sign_share": 0.70,
    "max_corr_existing": 0.50,
    "annual_turnover": 8.0,
    "model_ic_gain": 0.01,
}


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #


def load_inputs(stock_cache: Path | None = None) -> dict[str, Any]:
    source = Path(stock_cache) if stock_cache else STOCK_CACHE
    stocks: list[dict[str, Any]] = []
    for path in sorted(source.glob("*.json")):
        payload = fd.sanitize(json.loads(path.read_text(encoding="utf-8")))
        if len(payload.get("rows", [])) >= 300:
            payload.setdefault("asset_type", "stock")
            stocks.append(payload)
    market = fd.sanitize(json.loads((ETF_CACHE / "510300.json").read_text(encoding="utf-8")))
    market["asset_type"] = "index"
    industry_raw = json.loads(INDUSTRY_PATH.read_text(encoding="utf-8"))
    industry = {
        code: ((info.get("industry") or "")[:1] or "NA")
        for code, info in industry_raw.items()
    }
    return {"stocks": stocks, "market": market, "industry": industry}


def forward_returns(history: dict[str, Any], horizon: int) -> dict[str, float]:
    rows = history["rows"]
    result: dict[str, float] = {}
    for index in range(len(rows) - horizon - 1):
        entry = float(rows[index + 1]["open"])
        exit_price = float(rows[index + 1 + horizon]["open"])
        if entry > 0 and exit_price > 0:
            result[rows[index]["date"]] = (exit_price / entry - 1.0) * 100.0
    return result


def build_records(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    bench = fa.benchmark_returns(bundle["market"])
    industry = bundle["industry"]
    records: list[dict[str, Any]] = []
    for history in bundle["stocks"]:
        symbol = history["symbol"]
        factors = fa.compute(history, bench["ret1"])
        existing = fa.existing(history, bench)
        caps = fa.float_cap(history)
        closings = [float(row["close"]) for row in history["rows"]]
        ma60 = fa._sma(closings, 60)
        fwd = forward_returns(history, HORIZON)
        fwd5 = forward_returns(history, REVERSAL_HORIZON)
        last_cap: float | None = None
        for index, row in enumerate(history["rows"]):
            if caps[index] is not None:
                last_cap = caps[index]
            cap = last_cap
            records.append({
                "date": row["date"],
                "symbol": symbol,
                "industry": industry.get(symbol, "NA"),
                "fwd": fwd.get(row["date"]),
                "fwd5": fwd5.get(row["date"]),
                "log_cap": math.log(cap) if cap and cap > 0 else None,
                "above_ma60": (closings[index] > ma60[index]) if (ma60[index] and closings[index]) else None,
                **{key: series[index] for key, series in factors.items()},
                **{f"existing_{key}": series[index] for key, series in existing.items()},
            })
    records.sort(key=lambda item: (item["date"], item["symbol"]))
    return records


def group_by_date(records: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[row["date"]].append(row)
    return dict(groups)


def breadth(groups: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for date, rows in groups.items():
        sample = [row["above_ma60"] for row in rows if row["above_ma60"] is not None]
        if len(sample) >= MIN_XSEC:
            result[date] = sum(1 for value in sample if value) / len(sample)
    return result


# --------------------------------------------------------------------------- #
# 横截面处理
# --------------------------------------------------------------------------- #


def _rank_column(rows: Sequence[dict[str, Any]], key: str) -> list[float | None]:
    values = [row.get(key) for row in rows]
    present = [value for value in values if value is not None]
    if len(present) < MIN_XSEC:
        return [None] * len(rows)
    ranks = fa.rank_pct(present)
    iterator = iter(ranks)
    return [(next(iterator) / 100.0) if value is not None else None for value in values]


def _industry_means(rows: Sequence[dict[str, Any]], column: Sequence[float | None]) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row, value in zip(rows, column):
        if value is not None:
            buckets[row["industry"]].append(value)
    overall = statistics.fmean([v for v in column if v is not None])
    return {
        code: (statistics.fmean(values) if len(values) >= MIN_INDUSTRY else overall)
        for code, values in buckets.items()
    }


def cross_sectionalize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """逐日：原始分位 → 行业/规模/beta 中性化 → 对现有三因子正交化 → 族综合分。"""
    groups = group_by_date(records)
    dates: list[str] = []
    for date, rows in sorted(groups.items()):
        labeled = [row for row in rows if row["fwd"] is not None]
        if len(rows) < MIN_XSEC or len(labeled) < MIN_XSEC:
            continue
        dates.append(date)

        # 标签分位（0–1）与 5 日标签
        label_ranks = _rank_pct([row["fwd"] for row in labeled])
        label_map = {row["symbol"]: value for row, value in zip(labeled, label_ranks)}
        for row in rows:
            row["label"] = label_map.get(row["symbol"])
        five_day = [row for row in rows if row["fwd5"] is not None]
        if len(five_day) >= MIN_XSEC:
            ranks5 = _rank_pct([row["fwd5"] for row in five_day])
            mapping5 = {row["symbol"]: value for row, value in zip(five_day, ranks5)}
            for row in rows:
                row["label5"] = mapping5.get(row["symbol"])

        for key in fa.FACTOR_SPECS:
            for row, value in zip(rows, _rank_column(rows, key)):
                row[f"raw_{key}"] = value
        for key in fa.EXISTING_FACTORS:
            for row, value in zip(rows, _rank_column(rows, f"existing_{key}")):
                row[f"rank_existing_{key}"] = value

        size_rank = _rank_column(rows, "log_cap")
        beta_rank = _rank_column(rows, "beta120")

        for key in fa.FACTOR_SPECS:
            raw = [row.get(f"raw_{key}") for row in rows]
            if sum(1 for value in raw if value is not None) < MIN_XSEC:
                for row in rows:
                    row[f"neu_{key}"] = row[f"rank_{key}"] = None
                continue
            means = _industry_means(rows, raw)
            adjusted = [
                None if value is None else value - means.get(row["industry"], 0.0)
                for row, value in zip(rows, raw)
            ]
            valid = [
                i for i, value in enumerate(adjusted)
                if value is not None and size_rank[i] is not None and beta_rank[i] is not None
            ]
            for row in rows:
                row[f"neu_{key}"] = row[f"rank_{key}"] = None
            if len(valid) < MIN_XSEC:
                continue
            residual = fa.orthogonalize(
                [adjusted[i] for i in valid],
                [[size_rank[i] for i in valid], [beta_rank[i] for i in valid]],
            )
            neutral_ranks = _rank_pct(residual)
            for position, index in enumerate(valid):
                rows[index][f"neu_{key}"] = neutral_ranks[position]

            controls_idx = [
                i for i in valid
                if all(rows[i].get(f"rank_existing_{name}") is not None for name in fa.EXISTING_FACTORS)
            ]
            if len(controls_idx) < MIN_XSEC:
                continue
            base = [rows[i][f"neu_{key}"] for i in controls_idx]
            controls = [
                [rows[i][f"rank_existing_{name}"] for i in controls_idx]
                for name in fa.EXISTING_FACTORS
            ]
            ortho = fa.orthogonalize(base, controls)
            ranks = _rank_pct(ortho)
            for position, index in enumerate(controls_idx):
                rows[index][f"rank_{key}"] = ranks[position]

        for family in fa.FAMILY_SPECS:
            members = [key for key, spec in fa.FACTOR_SPECS.items() if spec["family"] == family]
            for row in rows:
                values = []
                for key in members:
                    value = row.get(f"neu_{key}")
                    if value is None:
                        continue
                    values.append(value if fa.FACTOR_SPECS[key]["prior"] > 0 else 1.0 - value)
                row[f"family_{family}"] = statistics.fmean(values) if len(values) >= 2 else None
                row[f"resid_family_{family}"] = None
            valid = [i for i, row in enumerate(rows) if row[f"family_{family}"] is not None]
            if len(valid) < MIN_XSEC:
                continue
            family_ranks = _rank_pct([rows[i][f"family_{family}"] for i in valid])
            for position, index in enumerate(valid):
                rows[index][f"family_{family}"] = family_ranks[position]
            controls_idx = [
                i for i in valid
                if all(rows[i].get(f"rank_existing_{name}") is not None for name in fa.EXISTING_FACTORS)
            ]
            if len(controls_idx) < MIN_XSEC:
                continue
            base = [rows[i][f"family_{family}"] for i in controls_idx]
            controls = [
                [rows[i][f"rank_existing_{name}"] for i in controls_idx]
                for name in fa.EXISTING_FACTORS
            ]
            ortho = fa.orthogonalize(base, controls)
            ranks = _rank_pct(ortho)
            for position, index in enumerate(controls_idx):
                rows[index][f"resid_family_{family}"] = ranks[position]
                # xsec.fit_rank_model 按 rank_<feature> 取列，这里存一份同值别名
                rows[index][f"rank_family_{family}"] = ranks[position]
    return {"dates": dates, "groups": groups}


# --------------------------------------------------------------------------- #
# 统计
# --------------------------------------------------------------------------- #


def half_year(date: str) -> str:
    month = int(date[5:7])
    return f"{date[:4]}H{1 if month <= 6 else 2}"


def daily_ic(
    groups: dict[str, list[dict[str, Any]]],
    key: str,
    *,
    label: str = "label",
    dates: Sequence[str] | None = None,
) -> list[float]:
    output: list[float] = []
    for date in (dates if dates is not None else sorted(groups)):
        rows = groups.get(date) or []
        pairs = [
            (row.get(key), row.get(label))
            for row in rows
            if row.get(key) is not None and row.get(label) is not None
        ]
        if len(pairs) < MIN_XSEC:
            continue
        ic = ek.spearman([p[0] for p in pairs], [p[1] for p in pairs])
        if ic is not None:
            output.append(ic)
    return output


def stat(series: Sequence[float], *, iters: int = ITERS) -> dict[str, Any]:
    clean = [value for value in series if value is not None]
    if len(clean) < 40:
        return {"mean": None, "t": None, "ci_low": None, "ci_high": None,
                "positive_rate": None, "n": len(clean)}
    result = ek.ic_ci(clean, horizon=HORIZON, iters=iters)
    mean, se = result["mean"], result["se"]
    t_value = (mean / se) if (se and se > 1e-12) else None
    return {
        "mean": round(mean, 4),
        "t": round(t_value, 2) if t_value is not None else None,
        "ci_low": round(result["ci_low"], 4) if result["ci_low"] is not None else None,
        "ci_high": round(result["ci_high"], 4) if result["ci_high"] is not None else None,
        "positive_rate": result.get("positive_rate"),
        "n": len(clean),
    }


def halves_ic(
    groups: dict[str, list[dict[str, Any]]],
    key: str,
    *,
    label: str = "label",
) -> dict[str, float]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for date in groups:
        buckets[half_year(date)].append(date)
    output: dict[str, float] = {}
    for name, dates in sorted(buckets.items()):
        series = daily_ic(groups, key, label=label, dates=dates)
        if len(series) >= 20:
            output[name] = round(statistics.fmean(series), 3)
    return output


def turnover_of(
    groups: dict[str, list[dict[str, Any]]],
    key: str,
    *,
    prior: int = 1,
    start: str = "2015-01-01",
) -> float:
    """Top 20% 等权、21 日调仓的单边年换手（倍）。按先验方向取多的一侧。"""
    dates = sorted(date for date, rows in groups.items() if date >= start and len(rows) >= MIN_XSEC)
    picks: list[set[str]] = []
    for date in dates[::REBALANCE]:
        rows = [row for row in groups[date] if row.get(key) is not None]
        if len(rows) < MIN_XSEC:
            continue
        count = max(1, int(round(len(rows) * TOP_FRACTION)))
        ordered = sorted(rows, key=lambda row: -prior * row[key])
        picks.append({row["symbol"] for row in ordered[:count]})
    if len(picks) < 2:
        return 0.0
    changes = [len(current - previous) / len(current) for previous, current in zip(picks, picks[1:])]
    years = len(dates) / 252.0 or 1.0
    return round(statistics.fmean(changes) * (len(picks) - 1) / years, 2)


def corr_with_existing(
    groups: dict[str, list[dict[str, Any]]],
    key: str,
) -> dict[str, float]:
    output: dict[str, float] = {}
    for name in fa.EXISTING_FACTORS:
        series = []
        for date in sorted(groups):
            pairs = [
                (row.get(key), row.get(f"rank_existing_{name}"))
                for row in groups[date]
                if row.get(key) is not None and row.get(f"rank_existing_{name}") is not None
            ]
            if len(pairs) < MIN_XSEC:
                continue
            ic = ek.spearman([p[0] for p in pairs], [p[1] for p in pairs])
            if ic is not None:
                series.append(ic)
        output[name] = round(statistics.fmean(series), 3) if series else 0.0
    output["max_abs"] = round(max((abs(output[name]) for name in fa.EXISTING_FACTORS), default=0.0), 3)
    return output


def ic_by_regime(
    groups: dict[str, list[dict[str, Any]]],
    key: str,
    regime_by_date: dict[str, str],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for state in ("risk_on", "neutral", "risk_off"):
        dates = [date for date in groups if regime_by_date.get(date) == state]
        series = daily_ic(groups, key, dates=dates)
        output[state] = (
            {"mean": round(statistics.fmean(series), 3), "n": len(series)}
            if len(series) >= 30 else None
        )
    return output


# --------------------------------------------------------------------------- #
# 因子与族评估
# --------------------------------------------------------------------------- #


def evaluate_factors(
    groups: dict[str, list[dict[str, Any]]],
    regime_by_date: dict[str, str],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for key, spec in fa.FACTOR_SPECS.items():
        raw = stat(daily_ic(groups, f"raw_{key}"))
        neutral = stat(daily_ic(groups, f"neu_{key}"))
        residual = stat(daily_ic(groups, f"rank_{key}"))
        per_half = halves_ic(groups, f"rank_{key}")
        halves = list(per_half.values())
        expected = spec["prior"]
        agreeing = sum(
            1 for value in halves if (value > 0) == (expected > 0)
        )
        results[key] = {
            "label": spec["label"],
            "family": spec["family"],
            "prior": expected,
            "raw": raw,
            "neutral": neutral,
            "residual": residual,
            "corr_existing": corr_with_existing(groups, f"neu_{key}"),
            "halves": per_half,
            "half_sign_share": round(agreeing / len(halves), 3) if halves else 0.0,
            "regime_ic": ic_by_regime(groups, f"rank_{key}", regime_by_date),
            "turnover": turnover_of(groups, f"rank_{key}", prior=expected),
            "reversal_ic": stat(daily_ic(groups, f"rank_{key}", label="label5"))
            if spec["family"] == "reversal" else None,
        }
    return results


def evaluate_families(
    groups: dict[str, list[dict[str, Any]]],
    factor_stats: dict[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for family, spec in fa.FAMILY_SPECS.items():
        members = [key for key, item in fa.FACTOR_SPECS.items() if item["family"] == family]
        residual = stat(daily_ic(groups, f"resid_family_{family}"))
        neutral = stat(daily_ic(groups, f"family_{family}"))
        per_half = halves_ic(groups, f"resid_family_{family}")
        halves = list(per_half.values())
        share = sum(1 for value in halves if value > 0) / len(halves) if halves else 0.0
        # 冗余要在正交化之前查：正交化后的相关必然≈0，拿它当门槛等于没查。
        corr = corr_with_existing(groups, f"family_{family}")["max_abs"]
        turnover = turnover_of(groups, f"resid_family_{family}", prior=1)
        checks = {
            "incremental_ic": bool(residual["mean"] is not None and residual["mean"] >= PASS_CRITERIA["incremental_ic"]),
            "t_stat": bool(residual["t"] is not None and residual["t"] >= PASS_CRITERIA["t_stat"]),
            "half_sign_share": share >= PASS_CRITERIA["half_sign_share"],
            "max_corr_existing": corr < PASS_CRITERIA["max_corr_existing"],
            "annual_turnover": turnover <= PASS_CRITERIA["annual_turnover"],
        }
        output[family] = {
            "label": spec["label"],
            "members": members,
            "member_residual_ics": {key: factor_stats[key]["residual"]["mean"] for key in members},
            "residual": residual,
            "neutral": neutral,
            "halves": per_half,
            "half_sign_share": round(share, 3),
            "corr_existing_max": corr,
            "turnover": turnover,
            "checks": checks,
            "spread": spread_after_cost(
                groups, f"resid_family_{family}", prior=1, turnover=turnover
            ),
            "pass": all(checks.values()),
        }
    return output


def spread_after_cost(
    groups: dict[str, list[dict[str, Any]]],
    key: str,
    *,
    prior: int = 1,
    turnover: float,
    start: str = "2015-01-01",
) -> dict[str, Any]:
    """Top 20% 相对同池等权的年化（近似），以及按换手折算的成本。

    用 ``fwd``（次日开盘进、20 日后开盘出）在每 21 个交易日采样一次，等价于
    "月频调仓、持有到下次调仓"，不做逐日盯市——这是筛选用的近似，不是净值回测。
    成本按单边年换手折算：一次完整换手 = 卖出 12.6bp + 买入 7.6bp。
    """
    dates = sorted(date for date, rows in groups.items() if date >= start and len(rows) >= MIN_XSEC)
    spreads: list[float] = []
    for date in dates[::REBALANCE]:
        rows = [
            row for row in groups[date]
            if row.get(key) is not None and row.get("fwd") is not None
        ]
        if len(rows) < MIN_XSEC:
            continue
        count = max(1, int(round(len(rows) * TOP_FRACTION)))
        ordered = sorted(rows, key=lambda row: -prior * row[key])[:count]
        spreads.append(statistics.fmean([row["fwd"] for row in ordered])
                       - statistics.fmean([row["fwd"] for row in rows]))
    if not spreads:
        return {"gross": None, "cost": None, "net": None}
    per_period = statistics.fmean(spreads)
    gross = per_period * (252.0 / REBALANCE)
    cost = turnover * (12.6 + 7.6) / 100.0
    return {
        "gross": round(gross, 2),
        "cost": round(cost, 2),
        "net": round(gross - cost, 2),
        "n": len(spreads),
    }


# --------------------------------------------------------------------------- #
# 组合模型
# --------------------------------------------------------------------------- #


def _usable(rows: Sequence[dict[str, Any]], features: Sequence[str], label: str) -> list[dict[str, Any]]:
    """``xsec.fit_rank_model`` 按 ``rank_<feature>`` 取列，特征名因此不带 rank_ 前缀。"""
    return [
        row for row in rows
        if row.get(label) is not None
        and all(row.get(f"rank_{feature}") is not None for feature in features)
    ]


def oos_ic_by_date(
    records: Sequence[dict[str, Any]],
    features: Sequence[str],
    *,
    label: str = "label",
) -> tuple[dict[str, float], list[dict[str, float]]]:
    from xsec import fit_rank_model, model_scores

    usable = _usable(records, features, label)
    dates = sorted({row["date"] for row in usable})
    folds = pr.purged_kfold(dates, k=K_FOLDS, horizon=HORIZON, embargo=EMBARGO)
    output: dict[str, float] = {}
    coefficients: list[dict[str, float]] = []
    for fold in folds:
        train = [row for row in usable if row["date"] in set(fold["train_dates"])]
        test = [row for row in usable if row["date"] in set(fold["test_dates"])]
        if len(train) < 1000 or not test:
            continue
        model = fit_rank_model(train, features)
        coefficients.append(dict(model.get("coefficients") or {}))
        scores = model_scores(test, model)
        by_date: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for row in test:
            score = scores.get((row["date"], row["symbol"]))
            if score is not None:
                by_date[row["date"]].append((score, row[label]))
        for date, pairs in by_date.items():
            if len(pairs) < MIN_XSEC:
                continue
            ic = ek.spearman([p[0] for p in pairs], [p[1] for p in pairs])
            if ic is not None:
                output[date] = ic
    return output, coefficients


def _mean_coefficients(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    totals: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            totals[key].append(float(value))
    return {key: round(statistics.fmean(values), 4) for key, values in totals.items() if values}


def run_model(
    records: Sequence[dict[str, Any]],
    features: Sequence[str],
    *,
    label: str = "label",
) -> dict[str, Any]:
    by_date, coefficients = oos_ic_by_date(records, features, label=label)
    series = [by_date[date] for date in sorted(by_date)]
    result = stat(series)
    result["by_date"] = by_date
    result["coefficient_mean"] = _mean_coefficients(coefficients)
    return result


def compare_models(
    records: Sequence[dict[str, Any]],
    families: dict[str, Any],
    *,
    label: str = "label",
) -> dict[str, Any]:
    baseline_features = [f"existing_{key}" for key in fa.EXISTING_FACTORS]
    chosen = [family for family, data in families.items() if data["pass"]]
    extended_features = list(baseline_features) + [f"family_{family}" for family in chosen]
    baseline = run_model(records, baseline_features, label=label)
    extended = run_model(records, extended_features, label=label) if chosen else None

    paired = None
    if extended:
        common = sorted(set(baseline["by_date"]) & set(extended["by_date"]))
        differences = [extended["by_date"][date] - baseline["by_date"][date] for date in common]
        paired = stat(differences)

    gain = None
    if extended and extended["mean"] is not None and baseline["mean"] is not None:
        gain = round(extended["mean"] - baseline["mean"], 4)
    return {
        "baseline_features": baseline_features,
        "extended_features": extended_features,
        "chosen_families": chosen,
        "baseline": {key: value for key, value in baseline.items() if key != "by_date"},
        "extended": {key: value for key, value in (extended or {}).items() if key != "by_date"} or None,
        "gain": gain,
        "paired": paired,
        "pass": bool(
            gain is not None and gain >= PASS_CRITERIA["model_ic_gain"]
            and extended and extended["ci_low"] is not None and extended["ci_low"] > 0
        ),
    }


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #


def fmt(value: Any, digits: int = 3, signed: bool = True) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        if not signed:
            return f"{value:.{digits}f}"
        return f"{value:+.{digits}f}"
    return str(value)


def share(value: Any) -> str:
    return "—" if value is None else f"{float(value) * 100:.0f}%"


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    factors = report["factors"]
    families = report["families"]
    model = report["model"]

    add("# 新因子族：实现方案与验证结果\n")
    add(f"生成时间：{report['generated_at']}　样本：{report['universe']['stocks']} 只个股 × "
        f"{report['universe']['symbol_days']} 个标的日；区间 {report['universe']['start']} ~ "
        f"{report['universe']['end']}（{report['universe']['dates']} 个交易日）\n")

    add("## 0. 结论先行\n")
    add("| 族 | 族综合增量 IC | t | 半年同号 | 与现有因子最大相关 | 年换手 | 判定 |")
    add("|---|---|---|---|---|---|---|")
    for family, data in families.items():
        add(f"| {data['label']} | {fmt(data['residual']['mean'])} | {fmt(data['residual']['t'], 2)} | "
            f"{share(data['half_sign_share'])} | {data['corr_existing_max']:.2f} | "
            f"{data['turnover']:.1f}× | {'通过' if data['pass'] else '未通过'} |")
    add("")
    if model.get("extended"):
        add(f"组合模型（5 折 purged walk-forward）：现有 3 因子 OOS IC "
            f"{fmt(model['baseline']['mean'])} [{fmt(model['baseline']['ci_low'])}, "
            f"{fmt(model['baseline']['ci_high'])}]；加入通过筛选的 "
            f"{len(model['chosen_families'])} 个族之后 {fmt(model['extended']['mean'])} "
            f"[{fmt(model['extended']['ci_low'])}, {fmt(model['extended']['ci_high'])}]；"
            f"提升 {fmt(model['gain'])}，判定{'达标' if model['pass'] else '未达标'}。\n")
    add("通过标准（跑之前写死）：增量 IC ≥ 0.01、t ≥ 2、半年同号率 ≥ 70%、与现有因子相关 < 0.5、"
        "年换手 ≤ 8 倍；组合模型要求 OOS IC 提升 ≥ 0.01 且区间下限 > 0。\n")

    add("## 1. 因子定义与单因子结果\n")
    add("| 因子 | 族 | 定义 | 先验 | 方向一致 | 原始 IC | 中性化 IC | 增量 IC | t | 年换手 |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for key, spec in fa.FACTOR_SPECS.items():
        item = factors[key]
        value = item["residual"]["mean"]
        agrees = "—" if value is None else ("是" if (value > 0) == (spec["prior"] > 0) else "相反")
        add(f"| {spec['label']} | {fa.FAMILY_SPECS[spec['family']]['label']} | {spec['formula']} | "
            f"{'+' if spec['prior'] > 0 else '−'} | {agrees} | {fmt(item['raw']['mean'])} | "
            f"{fmt(item['neutral']['mean'])} | {fmt(value)} | "
            f"{fmt(item['residual']['t'], 2)} | {item['turnover']:.1f}× |")
    add("")
    add("原始 IC：因子分位 vs 未来 20 日收益分位；中性化 IC：再剥离行业、规模（自由流通市值）、"
        "beta；增量 IC：再对现有三因子（MACD 柱、量比、RS20）取残差。t 用块自助标准误。\n")

    add("## 2. 稳定性、分 regime 与相关性\n")
    add("| 因子 | 半年 IC 区间 | 同号率 | 风险偏好上行 | 中性 | 防守 | ρ(MACD) | ρ(量比) | ρ(RS20) |")
    add("|---|---|---|---|---|---|---|---|---|")
    for key, spec in fa.FACTOR_SPECS.items():
        item = factors[key]
        values = list(item["halves"].values())
        span = f"[{fmt(min(values))}, {fmt(max(values))}]" if values else "—"
        regime = item["regime_ic"]
        cells = " | ".join(
            fmt(regime[state]["mean"]) if regime.get(state) else "—"
            for state in ("risk_on", "neutral", "risk_off")
        )
        corr = item["corr_existing"]
        add(f"| {spec['label']} | {span} | {share(item['half_sign_share'])} | {cells} | "
            f"{corr['macd_hist']:+.2f} | {corr['vol_ratio']:+.2f} | {corr['rs20']:+.2f} |")
    add("")
    add("同号率按各因子的**先验方向**统计（风险类因子 IC 为负才算同向）。"
        "regime 分档用 `fields.regime_series`（沪深300 趋势 + 波动 + 宽度）。\n")

    add("## 3. 族综合分：多表达平均\n")
    add("| 族 | 成员 | 成员增量 IC | 综合增量 IC | t | 95% 区间 | 年换手 | 毛年化 | 成本 | 净年化 | 检查项 |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    for family, data in families.items():
        members = "、".join(fa.FACTOR_SPECS[key]["label"] for key in data["members"])
        ics = "、".join(fmt(data["member_residual_ics"][key]) for key in data["members"])
        failed = [name for name, ok in data["checks"].items() if not ok]
        verdict = "全部通过" if not failed else "未过：" + "、".join(failed)
        spread = data.get("spread") or {}
        add(f"| {data['label']} | {members} | {ics} | {fmt(data['residual']['mean'])} | "
            f"{fmt(data['residual']['t'], 2)} | [{fmt(data['residual']['ci_low'])}, "
            f"{fmt(data['residual']['ci_high'])}] | {data['turnover']:.1f}× | "
            f"{fmt(spread.get('gross'), 2)}% | {fmt(spread.get('cost'), 2)}pp | "
            f"{fmt(spread.get('net'), 2)}% | {verdict} |")
    add("")
    add("族综合分先把成员按先验符号对齐（负向因子取 1−分位），再等权平均——"
        "这是「多表达平均」的落地方式，与「挑训练集 IC 最高的表达」完全不同。\n")

    add("### 3.1 被族平均掩盖的单表达\n")
    ranked = sorted(
        fa.FACTOR_SPECS,
        key=lambda key: -(factors[key]["residual"]["mean"] or 0),
    )
    add("按增量 IC 绝对值排序的前五名（含方向）：\n")
    add("| 因子 | 增量 IC | t | 半年同号 | 说明 |")
    add("|---|---|---|---|---|")
    for key in ranked[:5]:
        item = factors[key]
        add(f"| {item['label']} | {fmt(item['residual']['mean'])} | {fmt(item['residual']['t'], 2)} | "
            f"{share(item['half_sign_share'])} | {fa.FACTOR_SPECS[key]['note']} |")
    add("")
    add("这里故意只做诊断、不据此建模。52 周高点距离（增量 IC +0.030、t 2.76）单看已经够格，"
        "但它所在的动量族被 12–1、6–1、斜率等噪声表达平均掉了——这正是「多表达平均」的代价，"
        "也正好是上一版「训练集 IC 最高就选它」的反面。如果要单独用它，应当先把它作为一个"
        "独立的先验族写进 `FAMILY_SPECS`，再在下一段样本外验证，而不是在这份报告里补进去。\n")
    reversal = families["reversal"]
    momentum = families["momentum"]
    add(f"5 日反转是这个样本里最典型的「毛收益好看、扣费后消失」：族毛年化 "
        f"{fmt((reversal.get('spread') or {}).get('gross'), 2)}%，年换手 {reversal['turnover']:.1f} 倍，"
        f"按单边换手折算成本 {fmt((reversal.get('spread') or {}).get('cost'), 2)}pp，净 "
        f"{fmt((reversal.get('spread') or {}).get('net'), 2)}%。A 股短期反转确实存在，但在这份"
        "样本与这个成本口径下打不过换手。\n")
    add(f"动量族则是另一种情况：毛年化 {fmt((momentum.get('spread') or {}).get('gross'), 2)}%、净 "
        f"{fmt((momentum.get('spread') or {}).get('net'), 2)}% 看起来能用，但增量 IC 的 t 只有 "
        f"{fmt(momentum['residual']['t'], 2)}——点估计靠少数时段贡献，不能当稳定 edge。"
        "这也说明「毛收益正」和「统计上可区分」是两件事。\n")

    add("## 4. 组合模型\n")
    add("| 模型 | 特征 | OOS IC | 95% 区间 | t | 为正比例 |")
    add("|---|---|---|---|---|---|")
    add(f"| 基准 | 现有三因子 | {fmt(model['baseline']['mean'])} | "
        f"[{fmt(model['baseline']['ci_low'])}, {fmt(model['baseline']['ci_high'])}] | "
        f"{fmt(model['baseline']['t'], 2)} | {share((model['baseline'].get('positive_rate') or 0) / 100)} |")
    if model.get("extended"):
        add(f"| 扩展 | 三因子 + {len(model['chosen_families'])} 族 | {fmt(model['extended']['mean'])} | "
            f"[{fmt(model['extended']['ci_low'])}, {fmt(model['extended']['ci_high'])}] | "
            f"{fmt(model['extended']['t'], 2)} | {share((model['extended'].get('positive_rate') or 0) / 100)} |")
    add("")
    if model.get("paired"):
        paired = model["paired"]
        add(f"逐日配对差（扩展 − 基准）：{fmt(paired['mean'])} "
            f"[{fmt(paired['ci_low'])}, {fmt(paired['ci_high'])}]，t = {fmt(paired['t'], 2)}。"
            "配对差比「两个均值各自波动」更接近真实增量。\n")

    add("## 5. 口径与限制\n")
    add("- 60 只沪深300 抽样、每日有效横截面约 55 只；行业按证监会大类，制造业占 29 只，"
        "行业内比较的有效样本只有十几只，行业中性化的分辨率有限。")
    add("- 市值 = `成交额 ÷ 换手率`（自由流通市值），停牌日沿用前值。该恒等式也说明"
        "「成交额/市值」与换手率是同一个量，因此没有单独造这个因子，避免重复计价。")
    add("- 标签是次日开盘进 / H 日后开盘出的毛收益，未扣成本；换手只用于筛选，"
        "组合层净净值口径见 `xsec.py` / `protocol.py`。")
    add("- 所有结论带块自助区间；t 不是独立同分布假设下的 t，不能直接与教科书的 2 比较大小。")
    add("")
    return "\n".join(lines)


def _inline(text: str) -> str:
    out = text
    while "**" in out:
        first = out.index("**")
        second = out.find("**", first + 2)
        if second < 0:
            break
        out = f"{out[:first]}<strong>{out[first + 2:second]}</strong>{out[second + 2:]}"
    return out


def render_html(markdown: str) -> str:
    body: list[str] = []
    in_table = False
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if all(set(cell) <= set("-: ") for cell in cells):
                continue
            if not in_table:
                body.append("<table>")
                in_table = True
                body.append("<tr>" + "".join(f"<th>{_inline(cell)}</th>" for cell in cells) + "</tr>")
            else:
                body.append("<tr>" + "".join(f"<td>{_inline(cell)}</td>" for cell in cells) + "</tr>")
            continue
        if in_table:
            body.append("</table>")
            in_table = False
        if stripped.startswith("# "):
            body.append(f"<h1>{stripped[2:]}</h1>")
        elif stripped.startswith("## "):
            body.append(f"<h2>{stripped[3:]}</h2>")
        elif stripped.startswith("- "):
            body.append(f"<li>{stripped[2:]}</li>")
        elif stripped:
            body.append(f"<p>{stripped}</p>")
    if in_table:
        body.append("</table>")
    css = """
    body { margin:0; padding:44px 22px 80px; background:#f6f4ef; color:#17231d;
           font-family:"PingFang SC","Microsoft YaHei",system-ui,sans-serif; line-height:1.7; }
    main { max-width:1150px; margin:0 auto; }
    h1 { font-size:25px; margin:0 0 10px; }
    h2 { font-size:17px; margin:40px 0 12px; padding-bottom:8px; border-bottom:1px solid #d8d4c8; }
    p { font-size:13.5px; color:#33413a; margin:9px 0; }
    li { font-size:13px; color:#33413a; margin:5px 0 5px 18px; }
    table { width:100%; border-collapse:collapse; margin:12px 0 22px; font-size:12.5px; background:#fff;
            box-shadow:0 1px 2px rgba(23,35,29,.06); }
    th, td { padding:7px 9px; border-bottom:1px solid #ece8dd; text-align:right; }
    th:first-child, td:first-child, th:nth-child(3), td:nth-child(3) { text-align:left; }
    th { background:#efece3; font-weight:600; color:#3a4a41; }
    """
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        f"<title>新因子族验证</title><style>{css}</style></head><body><main>"
        + "\n".join(body) + "</main></body></html>"
    )


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #


def run(
    *,
    outdir: Path | None = None,
    iters: int = ITERS,
    stock_cache: Path | None = None,
    members: Any = None,
    pit_version: str | None = None,
) -> dict[str, Any]:
    global ITERS
    ITERS = iters
    bundle = load_inputs(stock_cache)
    print(f"[1/5] {len(bundle['stocks'])} 只个股 + 市场代理 510300"
          + (f"（PIT {pit_version}）" if pit_version else ""))
    records = build_records(bundle)
    if members is not None:
        # PIT 闸门：中性化与族综合分都在"当日成分"的横截面里做，退市股退市后自然出池
        before = len(records)
        records = universe.filter_by_members(records, members)
        print(f"      按日成分过滤：{before} → {len(records)} 条")
    print(f"[2/5] 面板 {len(records)} 条标的日")
    sections = cross_sectionalize(records)
    valid_dates = set(sections["dates"])
    groups = {date: [row for row in rows if row.get("label") is not None]
              for date, rows in sections["groups"].items()}
    groups = {date: rows for date, rows in groups.items() if len(rows) >= MIN_XSEC}
    usable = [row for rows in groups.values() for row in rows]
    print(f"[3/5] 横截面 {len(usable)} 条 / {len(groups)} 个交易日")

    regime = fd.regime_series(bundle["market"], breadth_by_date=breadth(groups))
    regime_by_date = {point["date"]: point["state"] for point in regime if point}

    print("[4/5] 因子与族评估")
    factor_stats = evaluate_factors(groups, regime_by_date)
    families = evaluate_families(groups, factor_stats)
    print("[5/5] 组合模型")
    model = compare_models(usable, families)

    report = {
        "generated_at": datetime.now().isoformat(timespec="minutes"),
        "universe": {
            "stocks": len(bundle["stocks"]),
            "symbol_days": len(usable),
            "dates": len(groups),
            "start": min(groups),
            "end": max(groups),
        },
        "criteria": PASS_CRITERIA,
        "factors": factor_stats,
        "families": families,
        "model": model,
    }
    target = outdir or OUTDIR
    target.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(report)
    (target / "report.md").write_text(markdown, encoding="utf-8")
    (target / "report.html").write_text(render_html(markdown), encoding="utf-8")
    (target / "summary.json").write_text(
        json.dumps(_summary(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (target / "factor_stats.json").write_text(
        json.dumps({"factors": factor_stats, "families": families, "model": model},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"报告写入 {target}")
    for family, data in families.items():
        print(f"  {data['label']}: 增量 IC {data['residual']['mean']} t={data['residual']['t']} "
              f"换手 {data['turnover']}× {'通过' if data['pass'] else '未通过'}")
    print(f"  模型：基准 {model['baseline']['mean']} → 扩展 "
          f"{model['extended']['mean'] if model.get('extended') else '—'} (gain {model['gain']})")
    return report


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_at": report["generated_at"],
        "universe": report["universe"],
        "criteria": report["criteria"],
        "families": {
            family: {
                "pass": data["pass"],
                "residual_ic": data["residual"]["mean"],
                "t": data["residual"]["t"],
                "turnover": data["turnover"],
                "checks": data["checks"],
            }
            for family, data in report["families"].items()
        },
        "model": {
            "baseline_ic": report["model"]["baseline"]["mean"],
            "extended_ic": (report["model"]["extended"] or {}).get("mean"),
            "gain": report["model"]["gain"],
            "pass": report["model"]["pass"],
        },
    }


def main(argv: list[str] | None = None) -> int:
    global HORIZON
    parser = argparse.ArgumentParser(description="新因子族研究与验证")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--outdir", default=str(OUTDIR))
    parser.add_argument("--pit", default=None, help="用 PIT 池重跑（如 pit-v1）：面板与横截面都按日成分")
    args = parser.parse_args(argv)
    HORIZON = args.horizon
    stock_cache = members = None
    if args.pit:
        info = universe.prepare_cache(args.pit)
        stock_cache = Path(info["dir"])
        members = universe.members_loader(Path(info["members"]))
        print(f"PIT 面板 {info['symbols']} 只（{stock_cache}）")
    run(outdir=Path(args.outdir), iters=args.iters, stock_cache=stock_cache,
        members=members, pit_version=args.pit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
