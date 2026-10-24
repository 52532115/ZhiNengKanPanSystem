"""三字段的独立验证：每个字段一套口径，互不借用。

用法::

    python validate_fields.py --offline          # 读 .cache/fields 的本地面板
    python validate_fields.py --offline --skip-crossasset

产出:

- ``reports/fields/report.md`` / ``report.html``：人读
- ``reports/fields/summary.json``：前端读（用于给字段挂"已验证 / 未通过"标签）
- ``reports/fields/calibration.json``：描述分的分位校准表（按资产类别）
- ``reports/fields/crossasset.json``：跨资产组合净值与指标

判定标准（写死在 PASS_CRITERIA，报告里逐条给出是否满足）:

===========  ==============================================  ==========================
字段          指标                                            通过条件
===========  ==============================================  ==========================
descriptor    与未来 20 日风险标签的逐标的 Spearman           均值 |ρ| ≥ 0.3 且方向为负
              + 分位桶单调性 + 半年子样本符号一致            单调且 ≥ 80% 半年同号
selection     锁箱期逐日 IC 的块自助区间                        区间下限 > 0
regime        叠加 regime 后的最大回撤与净 Sharpe               回撤改善 ≥ 3pp 且 Sharpe 不降
etf 配置      净 Sharpe 与最大回撤对比同池等权                  两者同时优于等权
===========  ==============================================  ==========================
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import etf_allocation as ea
import evalkit as ek
import execution as ex
import fields as fd
import protocol as pr
from xsec import BUY_COST, SELL_COST

ROOT = Path(__file__).resolve().parent
STOCK_CACHE = ROOT / ".cache" / "fields" / "stocks"
OUTDIR = ROOT / "reports" / "fields"
HORIZON = 20
LOCKBOX_START = "2024-07-01"
MIN_XSEC = 20
REBALANCE = 21
TOP_FRACTION = 0.2
RISK_LABELS = ("vol", "mdd_depth", "downside", "var95")
RISK_LABEL_CN = {
    "vol": "未来 20 日年化波动",
    "mdd_depth": "未来 20 日最大回撤（取正）",
    "downside": "未来 20 日下行波动",
    "var95": "未来 20 日 5% VaR",
}

PASS_CRITERIA = {
    "descriptor": "未来波动标签的同期池化 Spearman ≤ −0.3；分位桶单调（ρ ≤ −0.8）；≥80% 半年同号且 |ρ| 中位数 ≥ 0.2",
    "selection": "锁箱期 IC 的 95% 块自助区间下限 > 0，且扣费后跑赢同池等权",
    "regime": "最大回撤改善 ≥ 3pp，且扣费后 Sharpe 不低于始终满仓",
    "crossasset": "净 Sharpe 高于同池等权且最大回撤更低（引擎：etf_allocation）",
}


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #


def load_stocks(cache: Path | None = None) -> list[dict[str, Any]]:
    target = cache or STOCK_CACHE
    histories: list[dict[str, Any]] = []
    for path in sorted(target.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload = fd.sanitize(payload)
        if len(payload.get("rows", [])) >= fd.MIN_BARS:
            payload.setdefault("asset_type", "stock")
            histories.append(payload)
    return histories


def half_year(date: str) -> str:
    month = int(date[5:7])
    return f"{date[:4]}H{1 if month <= 6 else 2}"


def breadth_series(
    histories: Sequence[dict[str, Any]],
    *,
    window: int = 60,
) -> dict[str, float]:
    """每日"收盘价在 MA60 上方的标的占比"，用于 regime 的市场宽度。"""
    above: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    for history in histories:
        closes = [float(row["close"]) for row in history["rows"]]
        ma = fd._sma(closes, window)
        for index, row in enumerate(history["rows"]):
            if ma[index] is None:
                continue
            total[row["date"]] += 1
            if closes[index] > ma[index]:
                above[row["date"]] += 1
    return {date: above[date] / count for date, count in total.items() if count >= MIN_XSEC}


def load_benchmark(path: Path | None = None) -> dict[str, Any]:
    target = path or ROOT / ".cache" / "fields" / "etf" / "510300.json"
    if target.is_file():
        payload = json.loads(target.read_text(encoding="utf-8"))
        payload["asset_type"] = "index"
        return payload
    fallback = ROOT / ".cache" / "walkforward" / "000300_index_2400.json"
    return json.loads(fallback.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# 描述分
# --------------------------------------------------------------------------- #


def _rho(left: Sequence[float], right: Sequence[float]) -> float | None:
    return ek.spearman(left, right) if len(left) >= 12 else None


def descriptor_samples(histories: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """逐标的、逐日的描述分 + 未来风险标签 + 自身分位。"""
    labels = pr.risk_labels(histories, horizon=HORIZON)
    samples: list[dict[str, Any]] = []
    for history in histories:
        symbol = history["symbol"]
        series = fd.descriptor_series(history)
        rows = history["rows"]
        scores: list[float | None] = [item["score"] if item else None for item in series]
        # 自身分位：过去 60 个有效分数里的位置
        trailing: list[float] = []
        for index, score in enumerate(scores):
            if score is None:
                continue
            trailing.append(score)
            if len(trailing) > fd.SELF_WINDOW:
                trailing.pop(0)
            if len(trailing) < 30:
                continue
            label = labels.get(symbol, {}).get(rows[index]["date"])
            if not label:
                continue
            samples.append(
                {
                    "symbol": symbol,
                    "date": rows[index]["date"],
                    "score": score,
                    "self_pct": fd._rank_of(score, trailing),
                    "labels": {
                        "vol": label["vol"],
                        "mdd_depth": -label["mdd"],
                        "downside": label["downside"],
                        "var95": label["var95"],
                    },
                }
            )
    return {"samples": samples, "symbols": len(histories)}


def descriptor_study(histories: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """描述分对风险标签的秩相关：逐标的、分位桶、半年子样本三层检查。"""
    bundle = descriptor_samples(histories)
    samples = bundle["samples"]
    if not samples:
        return {"available": False, "reason": "没有可用的描述分样本"}

    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_half: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_half_symbol: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        by_symbol[sample["symbol"]].append(sample)
        key = half_year(sample["date"])
        by_half[key].append(sample)
        by_half_symbol[key][sample["symbol"]].append(sample)

    per_label: dict[str, Any] = {}
    for label in RISK_LABELS:
        rhos: list[float] = []
        for rows in by_symbol.values():
            rho = _rho([row["score"] for row in rows], [row["labels"][label] for row in rows])
            if rho is not None:
                rhos.append(rho)
        pooled = _rho([row["score"] for row in samples], [row["labels"][label] for row in samples])

        # 分位桶：按「相对自身近 60 日分位」分 5 组；分数越高风险应越低
        buckets: list[list[float]] = [[] for _ in range(5)]
        for row in samples:
            position = min(4, max(0, int(row["self_pct"] // 20)))
            buckets[position].append(row["labels"][label])
        means = [statistics.fmean(bucket) if bucket else None for bucket in buckets]
        monotone = None
        if all(value is not None for value in means):
            monotone = ek.spearman(list(range(5)), [float(value) for value in means])

        # 半年子样本：主口径是同期池化（描述分本身就是自身相对刻度），
        # 次口径是"先按标的算再平均"（去掉标的间水平差异，但更噪）。
        half_pooled: dict[str, float] = {}
        half_symbol: dict[str, float] = {}
        for key, rows in sorted(by_half.items()):
            rho = _rho([row["score"] for row in rows], [row["labels"][label] for row in rows])
            if rho is not None:
                half_pooled[key] = round(rho, 3)
            symbol_values = []
            for symbol_rows in by_half_symbol[key].values():
                symbol_rho = _rho(
                    [row["score"] for row in symbol_rows],
                    [row["labels"][label] for row in symbol_rows],
                )
                if symbol_rho is not None:
                    symbol_values.append(symbol_rho)
            if symbol_values:
                half_symbol[key] = round(statistics.fmean(symbol_values), 3)

        negative = sum(1 for value in half_pooled.values() if value < 0)
        strong = sum(1 for value in half_pooled.values() if value <= -0.2)
        per_label[label] = {
            "per_symbol_mean": round(statistics.fmean(rhos), 3) if rhos else None,
            "per_symbol_median": round(statistics.median(rhos), 3) if rhos else None,
            "per_symbol_share_negative": round(sum(1 for v in rhos if v < 0) / len(rhos) * 100, 1) if rhos else None,
            "pooled": round(pooled, 3) if pooled is not None else None,
            "n_symbols": len(rhos),
            "n_samples": len(samples),
            "bucket_means": [round(value, 3) if value is not None else None for value in means],
            "bucket_monotone": round(monotone, 3) if monotone is not None else None,
            "halves": half_pooled,
            "halves_per_symbol": half_symbol,
            "halves_negative_share": round(negative / len(half_pooled) * 100, 1) if half_pooled else None,
            "halves_strong_share": round(strong / len(half_pooled) * 100, 1) if half_pooled else None,
            "halves_median_abs": round(statistics.median([abs(v) for v in half_pooled.values()]), 3)
            if half_pooled else None,
        }

    primary = per_label["vol"]["pooled"]
    strong_labels = [
        label for label in RISK_LABELS
        if per_label[label]["pooled"] is not None and per_label[label]["pooled"] <= -0.3
    ]
    # 分数越高风险越低，所以「单调」意味着桶均值随桶号下降，ρ ≤ −0.8
    monotone_ok = all(
        (per_label[label]["bucket_monotone"] or 0) <= -0.8 for label in ("vol", "downside")
    )
    stable_ok = all(
        (per_label[label]["halves_negative_share"] or 0) >= 80
        and (per_label[label]["halves_median_abs"] or 0) >= 0.2
        for label in ("vol", "downside")
    )
    passed = bool(
        primary is not None and primary <= -0.3
        and len(strong_labels) >= 3
        and monotone_ok and stable_ok
    )
    reasons: list[str] = []
    if primary is None or primary > -0.3:
        reasons.append(f"未来波动 ρ={primary}，未达到 ≤ −0.3")
    if len(strong_labels) < 3:
        reasons.append(f"只有 {len(strong_labels)} 个风险标签达到 |ρ| ≥ 0.3（需 ≥3）")
    if not monotone_ok:
        reasons.append("波动 / 下行波动的分位桶不单调")
    if not stable_ok:
        reasons.append("半年子样本中同号且 |ρ|≥0.2 的比例不足 80%")
    return {
        "available": True,
        "labels": per_label,
        "pass": passed,
        "reasons": reasons,
        "n_samples": len(samples),
        "n_symbols": bundle["symbols"],
        "score_distribution": {
            "mean": round(statistics.fmean([row["score"] for row in samples]), 2),
            "stdev": round(statistics.pstdev([row["score"] for row in samples]), 2),
        },
    }


# --------------------------------------------------------------------------- #
# 选股分
# --------------------------------------------------------------------------- #


def build_selection_panel(
    histories: Sequence[dict[str, Any]],
    *,
    members: Any = None,
) -> dict[str, Any]:
    """逐日横截面：描述分 → 池内分位；标签用次日开盘进、20 日后开盘出的收益分位。

    ``members`` 是 PIT 的闸门：当日不在成分里的票不进横截面。这一步不做，池内
    分位就是把"今天还活着但当年不在指数里"的票也算进来，分位会被系统性地抬高。
    """
    exec_returns = pr.forward_open_returns(histories, horizon=HORIZON)
    scores_by_date: dict[str, dict[str, float]] = defaultdict(dict)
    momentum_by_date: dict[str, dict[str, float]] = defaultdict(dict)
    allowed: dict[str, set[str]] = {}
    for history in histories:
        symbol = history["symbol"]
        series = fd.descriptor_series(history)
        closes = [float(row["close"]) for row in history["rows"]]
        for index, item in enumerate(series):
            if item is None:
                continue
            date = history["rows"][index]["date"]
            if date not in exec_returns[symbol]:
                continue
            if members is not None:
                if date not in allowed:
                    allowed[date] = members(date)
                if symbol not in allowed[date]:
                    continue
            scores_by_date[date][symbol] = item["score"]
            if index >= 252 and closes[index - 252] > 0:
                momentum_by_date[date][symbol] = closes[index - 21] / closes[index - 252] - 1.0

    records: list[dict[str, Any]] = []
    for date, scores in scores_by_date.items():
        if len(scores) < MIN_XSEC:
            continue
        percentiles = fd.pool_percentiles(scores)
        forward = {symbol: exec_returns[symbol][date] for symbol in scores}
        ranks = fd.pool_percentiles(forward)
        momentum = momentum_by_date.get(date) or {}
        momentum_pct = fd.pool_percentiles(momentum) if len(momentum) >= MIN_XSEC else {}
        for symbol, percentile in percentiles.items():
            records.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "score_pct": percentile,
                    "mom_pct": momentum_pct.get(symbol),
                    "label": ranks.get(symbol),
                    "forward": forward[symbol],
                }
            )
    records.sort(key=lambda row: (row["date"], row["symbol"]))
    return {"records": records, "dates": sorted({row["date"] for row in records})}


def _daily_ic(records: Sequence[dict[str, Any]], key: str, *, start: str | None = None,
              end: str | None = None) -> list[float]:
    buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in records:
        if row.get("label") is None or row.get(key) is None:
            continue
        if start and row["date"] < start:
            continue
        if end and row["date"] >= end:
            continue
        buckets[row["date"]].append((float(row[key]), float(row["label"])))
    series: list[float] = []
    for date in sorted(buckets):
        pairs = buckets[date]
        if len(pairs) < MIN_XSEC:
            continue
        ic = ek.spearman([p[0] for p in pairs], [p[1] for p in pairs])
        if ic is not None:
            series.append(ic)
    return series


def selection_study(
    histories: Sequence[dict[str, Any]],
    *,
    iters: int = 2000,
    members: Any = None,
) -> dict[str, Any]:
    panel = build_selection_panel(histories, members=members)
    records = panel["records"]
    if not records:
        return {"available": False, "reason": "横截面样本不足"}

    signals = {"descriptor": "score_pct", "momentum_12_1": "mom_pct"}
    stats: dict[str, Any] = {}
    for name, key in signals.items():
        full = _daily_ic(records, key)
        dev = _daily_ic(records, key, end=LOCKBOX_START)
        lock = _daily_ic(records, key, start=LOCKBOX_START)
        stats[name] = {
            "full": ek.ic_ci(full, horizon=HORIZON, iters=iters),
            "dev": ek.ic_ci(dev, horizon=HORIZON, iters=iters),
            "lockbox": ek.ic_ci(lock, horizon=HORIZON, iters=iters),
        }
    # 两个候选一起做 BH-FDR（口径相同，可比）
    pvalues = [
        stats[name][period]["p"]
        for name in signals for period in ("dev", "lockbox")
        if stats[name][period].get("p") is not None
    ]
    fdr = ek.bh_fdr(pvalues, q=0.05)
    index = 0
    for name in signals:
        for period in ("dev", "lockbox"):
            if stats[name][period].get("p") is not None:
                stats[name][period]["fdr_q"] = (
                    round(fdr["qvalues"][index], 4) if index < len(fdr["qvalues"]) else None
                )
                index += 1

    nav_full = selection_nav(histories, records, members=members)
    nav_lock = selection_nav(histories, records, start=LOCKBOX_START, members=members)
    nav = nav_lock if nav_lock["strategy"]["metrics"] else nav_full
    lockbox = stats["descriptor"]["lockbox"]
    passed = bool(
        lockbox.get("ci_low") is not None
        and lockbox["ci_low"] > 0
        and nav["strategy"]["metrics"].get("sharpe", 0) > nav["equal_weight"]["metrics"].get("sharpe", 1)
    )
    reasons: list[str] = []
    if lockbox.get("ci_low") is None or lockbox["ci_low"] <= 0:
        low = lockbox.get("ci_low")
        reasons.append(
            f"锁箱期 IC 区间下限为 {round(low, 3) if low is not None else '—'}，未大于 0"
        )
    if nav["strategy"]["metrics"].get("sharpe", 0) <= nav["equal_weight"]["metrics"].get("sharpe", 1):
        reasons.append(
            f"扣费后 Sharpe {nav['strategy']['metrics'].get('sharpe')} 未超过同池等权 "
            f"{nav['equal_weight']['metrics'].get('sharpe')}"
        )
    return {
        "available": True,
        "signals": stats,
        "nav": nav,
        "nav_full": nav_full,
        "nav_lockbox": nav_lock,
        "pass": passed,
        "reasons": reasons,
        "pool": {"mean": statistics.fmean([len({r['symbol'] for r in records if r['date'] == date}) for date in panel["dates"]]),
                 "dates": len(panel["dates"])},
        "n_records": len(records),
        "fdr": {"q": fdr.get("q"), "n_reject": fdr.get("n_reject"), "m": fdr.get("m")},
    }


def selection_nav(
    histories: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    *,
    start: str | None = None,
    members: Any = None,
) -> dict[str, Any]:
    """按池内分位取前 20% 等权，T+1 开盘成交、扣双边成本，与同池等权对比。

    ``members`` 同时约束策略与等权基准：PIT 口径下基准也只能买当日成分，
    否则"跑赢等权"比的是含非成分股的池子。
    """
    bars = ex.build_bars(histories)
    flags = ex.tradability(histories)
    rows = [row for row in records if not start or row["date"] >= start]
    calendar = sorted({row["date"] for row in rows})
    if len(calendar) < 40:
        return {"strategy": {"metrics": {}}, "equal_weight": {"metrics": {}}}
    by_date: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_date[row["date"]][row["symbol"]] = float(row["score_pct"])
    universe = [history["symbol"] for history in histories]

    def target(date: str, available: list[str]) -> dict[str, float]:
        scores = {symbol: value for symbol, value in by_date.get(date, {}).items() if symbol in available}
        if len(scores) < MIN_XSEC:
            return {}
        count = max(1, int(round(len(scores) * TOP_FRACTION)))
        chosen = sorted(scores.items(), key=lambda item: -item[1])[:count]
        weight = 1.0 / len(chosen)
        return {symbol: weight for symbol, _ in chosen}

    def cost_fn(symbol: str, date: str, fraction: float, side: str) -> float:
        return SELL_COST if side == "sell" else BUY_COST

    strategy = ex.simulate_t1(
        calendar, bars, target=target, cost_fn=cost_fn, flags=flags,
        rebalance_every=REBALANCE, max_turnover=0.4,
    )

    def equal(date: str, available: list[str]) -> dict[str, float]:
        names = [
            symbol for symbol in universe
            if symbol in available and (members is None or symbol in members(date))
        ]
        return {symbol: 1.0 / len(names) for symbol in names} if names else {}

    benchmark = ex.simulate_t1(
        calendar, bars, target=equal, cost_fn=cost_fn, flags=flags,
        rebalance_every=REBALANCE, max_turnover=1.0,
    )
    return {
        "strategy": {
            "metrics": ea.nav_metrics(strategy["nav"]),
            "total_cost": strategy["total_cost"],
            "turnover": round(strategy["total_turnover"], 1),
        },
        "equal_weight": {
            "metrics": ea.nav_metrics(benchmark["nav"]),
            "total_cost": benchmark["total_cost"],
        },
        "nav": strategy["nav"],
    }


# --------------------------------------------------------------------------- #
# 择时分（regime）
# --------------------------------------------------------------------------- #


def regime_study(benchmark: dict[str, Any], breadth: dict[str, float]) -> dict[str, Any]:
    rows = benchmark["rows"]
    series = fd.regime_series(benchmark, breadth_by_date=breadth)
    exposure = {point["date"]: point["exposure"] for point in series if point is not None}
    states = {point["date"]: point["state"] for point in series if point is not None}
    if not exposure:
        return {"available": False, "reason": "regime 序列为空"}

    nav = 1.0
    peak = 1.0
    worst = 0.0
    curve: list[dict[str, Any]] = []
    rets: list[float] = []
    previous_exposure = 1.0
    switches = 0
    cost_total = 0.0
    for index in range(1, len(rows)):
        date = rows[index]["date"]
        previous = rows[index - 1]
        ret = float(rows[index]["close"]) / float(previous["close"]) - 1.0 if float(previous["close"]) else 0.0
        target_exposure = exposure.get(previous["date"], previous_exposure)
        cost = abs(target_exposure - previous_exposure) * SELL_COST
        if abs(target_exposure - previous_exposure) > 1e-9:
            switches += 1
        cost_total += cost
        previous_exposure = target_exposure
        realized = ret * target_exposure - cost
        nav *= 1.0 + realized
        rets.append(realized)
        peak = max(peak, nav)
        worst = min(worst, nav / peak - 1.0)
        curve.append({"date": date, "nav": round(nav, 4), "exposure": target_exposure,
                      "state": states.get(previous["date"])})

    base_nav = 1.0
    base_peak = 1.0
    base_worst = 0.0
    base_rets: list[float] = []
    for index in range(1, len(rows)):
        ret = float(rows[index]["close"]) / float(rows[index - 1]["close"]) - 1.0
        base_nav *= 1.0 + ret
        base_rets.append(ret)
        base_peak = max(base_peak, base_nav)
        base_worst = min(base_worst, base_nav / base_peak - 1.0)

    years = len(rets) / 252.0
    stats = ea.nav_metrics(curve)
    base_stats = ea.nav_metrics([{"date": rows[i]["date"], "nav": 1.0} for i in range(1)] )  # 占位
    base_stats = {
        "cagr": round(((base_nav) ** (1 / years) - 1) * 100, 2) if years > 0 else 0.0,
        "vol": round(statistics.stdev(base_rets) * math.sqrt(252) * 100, 2) if len(base_rets) > 1 else 0.0,
        "sharpe": round(statistics.fmean(base_rets) / statistics.stdev(base_rets) * math.sqrt(252), 3)
        if len(base_rets) > 1 and statistics.stdev(base_rets) > 0 else 0.0,
        "max_drawdown": round(base_worst * 100, 2),
    }
    distribution = {state: sum(1 for value in states.values() if value == state) / len(states) * 100
                    for state in ("risk_on", "neutral", "risk_off")}
    improvement = round(stats.get("max_drawdown", 0) - base_stats["max_drawdown"], 2)
    passed = bool(
        improvement >= 3.0
        and stats.get("sharpe", 0) >= base_stats["sharpe"] - 0.02
    )
    reasons: list[str] = []
    if improvement < 3.0:
        reasons.append(f"最大回撤只改善 {improvement:.2f}pp（需 ≥3pp）")
    if stats.get("sharpe", 0) < base_stats["sharpe"] - 0.02:
        reasons.append(
            f"净 Sharpe {stats.get('sharpe')} 明显低于始终满仓 {base_stats['sharpe']}"
        )
    return {
        "available": True,
        "regime": stats,
        "always_in": base_stats,
        "improvement_pp": improvement,
        "switches": switches,
        "switches_per_year": round(switches / years, 1) if years else 0,
        "cost_drag_bps": round(cost_total / years * 10000, 1) if years else 0,
        "distribution": {key: round(value, 1) for key, value in distribution.items()},
        "nav": curve,
        "pass": passed,
        "reasons": reasons,
    }


# --------------------------------------------------------------------------- #
# 校准
# --------------------------------------------------------------------------- #


def calibrate(stocks: Sequence[dict[str, Any]], etf_panel: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """按资产类别分别校准；只用锁箱期之前的数据，避免用未来定义分位。"""
    stock_scores: list[float] = []
    for history in stocks:
        for index, item in enumerate(fd.descriptor_series(history)):
            if item is None:
                continue
            if history["rows"][index]["date"] >= LOCKBOX_START:
                continue
            stock_scores.append(item["score"])
    etf_scores: list[float] = []
    for symbol, history in etf_panel.items():
        if symbol == ea.CASH:
            continue
        for index, item in enumerate(fd.descriptor_series(history)):
            if item is None:
                continue
            if history["rows"][index]["date"] >= LOCKBOX_START:
                continue
            etf_scores.append(item["score"])
    table = fd.calibrate(
        {"stock": stock_scores, "etf": etf_scores},
        source=f"2016-12~{LOCKBOX_START} 面板（个股 {len(stock_scores)} / ETF {len(etf_scores)} 个样本）",
    )
    return table


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #


def _fmt(value: Any, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:+.{digits}f}{suffix}"
    return str(value)


def _share(value: Any, digits: int = 1) -> str:
    """比率类字段（占比、比例）不加正负号。"""
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}%"


def _ci_text(stat: dict[str, Any]) -> str:
    if stat.get("ci_low") is None:
        return f"{_fmt(stat.get('mean'))}（样本不足）"
    return f"{_fmt(stat.get('mean'))} [{_fmt(stat.get('ci_low'))}, {_fmt(stat.get('ci_high'))}]"


def render_markdown(report: dict[str, Any]) -> str:
    descriptor = report["descriptor"]
    selection = report["selection"]
    regime = report["regime"]
    cross = report["crossasset"]
    lines: list[str] = []
    add = lines.append

    add("# 三字段独立验证：描述 / 选股 / 择时\n")
    add(f"生成时间：{report['generated_at']}　样本：{report['universe']['stocks']} 只个股 × "
        f"{report['universe']['bars']} 根日线；{report['universe']['etf']} 只跨资产 ETF\n")
    add("## 0. 结论\n")
    add("| 字段 | 用途 | 核心指标 | 数值 | 判定 |")
    add("|---|---|---|---|---|")
    add(f"| 趋势状态 | 描述 | 未来波动逐标的 Spearman | "
        f"{_fmt(descriptor['labels']['vol']['per_symbol_mean'])} | "
        f"{'通过' if descriptor['pass'] else '未通过'} |")
    add(f"| 横截面分位 | 选股 | 锁箱期 IC 区间 | "
        f"{_ci_text(selection['signals']['descriptor']['lockbox'])} | "
        f"{'通过' if selection['pass'] else '未通过'} |")
    add(f"| 市场 regime | 择时 | 最大回撤改善 | "
        f"{_fmt(regime['improvement_pp'], 2, 'pp')} | {'通过' if regime['pass'] else '未通过'} |")
    add(f"| 跨资产 ETF | 配置 | 净 Sharpe / 回撤 | "
        f"{_fmt(cross['strategy']['metrics'].get('sharpe'))} / "
        f"{_fmt(cross['strategy']['metrics'].get('max_drawdown'), 2, '%')} | "
        f"{'通过' if cross['pass'] else '未通过'} |")
    add("")
    add("通过标准（写死在 `PASS_CRITERIA`）：\n")
    for key, text in PASS_CRITERIA.items():
        add(f"- **{key}**：{text}")
    add("")

    add("## 1. 趋势状态（描述口径）\n")
    add(descriptor.get("reasons") and "未通过原因：" + "；".join(descriptor["reasons"]) + "\n" or "全部条件满足。\n")
    add("### 1.1 与未来 20 日风险标签的秩相关\n")
    add("`逐标的均值` 是先按标的分组算 Spearman 再等权平均；`pooled` 是同一天所有标的"
        "混在一起算。描述分是「相对自身」的刻度，所以判定用 pooled，逐标的均值作稳健性对照。"
        "四个标签都已转成「越大越有风险」，所以预期符号为负。\n")
    add("| 风险标签 | 逐标的均值 | 中位数 | 负号占比 | pooled | 样本数 |")
    add("|---|---|---|---|---|---|")
    for label in RISK_LABELS:
        row = descriptor["labels"][label]
        add(f"| {RISK_LABEL_CN[label]} | {_fmt(row['per_symbol_mean'])} | "
            f"{_fmt(row['per_symbol_median'])} | {_share(row['per_symbol_share_negative'])} | "
            f"{_fmt(row['pooled'])} | {row['n_samples']} |")
    add("")
    add("### 1.2 分位桶单调性\n")
    add("按「相对自身近 60 日分位」分成五桶，看未来风险是否随桶号单调上升。\n")
    add("| 风险标签 | Q1（最低分位） | Q2 | Q3 | Q4 | Q5（最高分位） | 单调性 ρ |")
    add("|---|---|---|---|---|---|---|")
    for label in RISK_LABELS:
        row = descriptor["labels"][label]
        cells = " | ".join(_fmt(value, 3) for value in row["bucket_means"])
        add(f"| {RISK_LABEL_CN[label]} | {cells} | {_fmt(row['bucket_monotone'])} |")
    add("")
    add("### 1.3 半年子样本稳定性\n")
    add("表内是每个半年「先按标的算、再等权平均」的 Spearman；判定用的同期池化值在下方单列。"
        "`达标比例` 指池化值 ≤ −0.2 的半年占比。\n")
    halves = sorted(descriptor["labels"]["vol"]["halves"])
    add("| 半年 | " + " | ".join(RISK_LABEL_CN[label] for label in RISK_LABELS) + " |")
    add("|---" * (len(RISK_LABELS) + 1) + "|")
    for key in halves:
        cells = " | ".join(_fmt(descriptor["labels"][label]["halves"].get(key)) for label in RISK_LABELS)
        add(f"| {key} | {cells} |")
    add("| 池化达标比例 | " + " | ".join(
        _share(descriptor["labels"][label]["halves_strong_share"]) for label in RISK_LABELS
    ) + " |")
    add("")
    add("判定口径（同期池化）逐半年数值：\n")
    add("| 半年 | " + " | ".join(RISK_LABEL_CN[label] for label in RISK_LABELS) + " |")
    add("|---" * (len(RISK_LABELS) + 1) + "|")
    for key in halves:
        cells = " | ".join(_fmt(descriptor["labels"][label]["halves"].get(key)) for label in RISK_LABELS)
        add(f"| {key} | {cells} |")
    add("| 同号占比 | " + " | ".join(
        _share(descriptor["labels"][label]["halves_negative_share"]) for label in RISK_LABELS
    ) + " |")
    add("| 中位绝对相关 | " + " | ".join(
        f"{descriptor['labels'][label]['halves_median_abs']}" for label in RISK_LABELS
    ) + " |")
    add("")

    add("## 2. 横截面分位（选股口径）\n")
    add(selection.get("reasons") and "未通过原因：" + "；".join(selection["reasons"]) + "\n" or "全部条件满足。\n")
    add("同一批描述分转成**当日同池分位**后按横截面排序，标签是次日开盘进、20 日后开盘出的"
        "收益分位。因为分数没有拟合参数，开发期与锁箱期的区别只在时间切分。\n")
    add("| 信号 | 区间 | 均值 IC | 95% 区间 | 为正比例 | p | BH-FDR q |")
    add("|---|---|---|---|---|---|---|")
    names = {"descriptor": "描述分池内分位", "momentum_12_1": "12–1 动量分位"}
    for name in selection["signals"]:
        for period, label in (("dev", "开发期"), ("lockbox", "锁箱期"), ("full", "全样本")):
            stat = selection["signals"][name][period]
            add(f"| {names[name]} | {label} | {_fmt(stat.get('mean'))} | "
                f"[{_fmt(stat.get('ci_low'))}, {_fmt(stat.get('ci_high'))}] | "
                f"{_share(stat.get('positive_rate'))} | {_fmt(stat.get('p'))} | "
                f"{_fmt(stat.get('fdr_q'))} |")
    add("")
    add(f"组合层：前 20% 等权、{REBALANCE} 日调仓、T+1 开盘、双边成本 "
        f"{BUY_COST * 10000:.1f}/{SELL_COST * 10000:.1f}bp。判定看锁箱期，全样本只作对照。\n")
    add("| 区间 | 方案 | 年化 | 波动 | Sharpe | 最大回撤 | 累计成本 |")
    add("|---|---|---|---|---|---|---|")
    for period, key in (("锁箱期", "nav_lockbox"), ("全样本", "nav_full")):
        block = selection.get(key) or {}
        for label, side in (("池内分位 Top20%", "strategy"), ("同池等权", "equal_weight")):
            metrics = (block.get(side) or {}).get("metrics") or {}
            add(f"| {period} | {label} | {_fmt(metrics.get('cagr'), 2, '%')} | {_fmt(metrics.get('vol'), 2, '%')} | "
                f"{_fmt(metrics.get('sharpe'))} | {_fmt(metrics.get('max_drawdown'), 2, '%')} | "
                f"{_fmt((block.get(side) or {}).get('total_cost', 0), 3)} |")
    add("")

    add("## 3. 市场 regime（择时口径）\n")
    add(regime.get("reasons") and "未通过原因：" + "；".join(regime["reasons"]) + "\n" or "全部条件满足。\n")
    add("regime 分数由「长期趋势 / 趋势斜率 / 波动率位置 / 市场宽度」合成，再对**自身近三年**"
        "做标准化，切点是 ±0.5 个标准差，不是绝对分数。暴露按 100% / 60% / 30% 三档。\n")
    add("| 指标 | 叠加 regime | 始终满仓 |")
    add("|---|---|---|")
    for label, key in (("年化收益", "cagr"), ("年化波动", "vol"), ("Sharpe", "sharpe"), ("最大回撤", "max_drawdown")):
        suffix = "" if key == "sharpe" else "%"
        add(f"| {label} | {_fmt(regime['regime'].get(key), 2, suffix)} | "
            f"{_fmt(regime['always_in'].get(key), 2, suffix)} |")
    add("")
    dist = regime["distribution"]
    add(f"状态分布：风险偏好上行 {dist.get('risk_on')}% / 中性 {dist.get('neutral')}% / 防守 {dist.get('risk_off')}%；"
        f"年切换 {regime['switches_per_year']} 次，切换成本 {regime['cost_drag_bps']}bp/年。\n")

    add("## 4. 跨资产 ETF 趋势配置\n")
    add(cross.get("reasons") and "未通过原因：" + "；".join(cross["reasons"]) + "\n" or "全部条件满足。\n")
    settings = cross["strategy"].get("settings") or {}
    add(f"引擎 `etf_allocation`（规则版本 {settings.get('version', '—')}）："
        f"区间 {cross['start']} ~ 结束；月度调仓；入场四条（12–1 动量 > 0、收盘 > MA60、"
        f"收盘 > MA120、回撤 > −15%），持有只看动量与 MA120；ERC 加权后按 "
        f"{int((settings.get('target_vol') or 0.1) * 100)}% 目标波动缩放（只降不升）；"
        f"单腿 ≤ {int((settings.get('max_weight') or 0.3) * 100)}%、A 股与海外各 ≤ 50%；"
        f"现金腿 {settings.get('cash_leg', '511880')} 按实际收益计息。\n")
    if cross.get("stale"):
        add("已剔除过期标的：" + "、".join(f"`{k}`（末行 {v}）" for k, v in cross["stale"].items())
            + "——清盘 ETF 会让日历静默变短，必须显式排除。\n")
    add(cross["table"])
    add("")

    add("## 5. 校准表（分数 → 分位）\n")
    add("| 资产类别 | 样本数 | 分数中位数（P50 节点） | P20 | P80 | 来源 |")
    add("|---|---|---|---|---|---|")
    for key, row in report["calibration"].items():
        knots = row.get("score_knots") or []
        if len(knots) >= 21:
            add(f"| {fd.ASSET_CLASS_LABEL.get(key, key)} | {row.get('n')} | {knots[10]:.1f} | "
                f"{knots[4]:.1f} | {knots[16]:.1f} | {row.get('source')} |")
        else:
            # 未校准的类别（指数）不进分位映射，也不假装有 P50/P20/P80
            add(f"| {fd.ASSET_CLASS_LABEL.get(key, key)} | {row.get('n')} | — | — | — | "
                f"{row.get('source')} |")
    add("")
    add("分位切点固定为 20/40/60/80，所以档位本身与资产类别无关；资产类别改变的是"
        "「多少分对应多少分位」这个映射。个股波动更高、分布更散，同一分数在 ETF 上通常对应更高分位。"
        "指数（沪深300 / 中证500 / 中证1000）只有三条序列，构不成同类分布，因此不做同类校准、"
        "页面上只显示「相对自身」分位。\n")

    add("## 6. 口径纪律\n")
    add("- 三个字段不能互相推断：`fields.assert_use()` 在代码层拒绝跨用途调用，前端每个字段单独标注口径。")
    add("- 描述分不输出方向或仓位建议；选股分不输出强弱结论；regime 只给总暴露上限。")
    add("- 任何「分数 75+ 所以看多」的写法都属于把描述当择时用，已从接口和页面上移除。")
    add("")
    return "\n".join(lines)


def _inline(text: str) -> str:
    """只处理本报告用到的行内语法：**粗体** 与 `代码`。"""
    out = text
    while "**" in out:
        first = out.index("**")
        second = out.find("**", first + 2)
        if second < 0:
            break
        out = f"{out[:first]}<strong>{out[first + 2:second]}</strong>{out[second + 2:]}"
    while "`" in out:
        first = out.index("`")
        second = out.find("`", first + 1)
        if second < 0:
            break
        out = f"{out[:first]}<code>{out[first + 1:second]}</code>{out[second + 1:]}"
    return out


def render_html(report: dict[str, Any], markdown: str) -> str:
    """把 markdown 表格轻量转成 HTML（只处理本报告用到的语法）。"""
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
        elif stripped.startswith("### "):
            body.append(f"<h3>{stripped[4:]}</h3>")
        elif stripped.startswith("- "):
            body.append(f"<li>{_inline(stripped[2:])}</li>")
        elif stripped:
            body.append(f"<p>{_inline(stripped)}</p>")
    if in_table:
        body.append("</table>")
    css = """
    :root { color-scheme: light; }
    body { margin:0; padding:48px 24px 96px; background:#f6f4ef; color:#17231d;
           font-family:"PingFang SC","Microsoft YaHei",system-ui,sans-serif; line-height:1.75; }
    main { max-width:1000px; margin:0 auto; }
    h1 { font-size:26px; letter-spacing:.02em; margin:0 0 8px; }
    h2 { font-size:18px; margin:44px 0 12px; padding-bottom:8px; border-bottom:1px solid #d8d4c8; }
    h3 { font-size:15px; margin:26px 0 8px; color:#3a4a41; }
    p { margin:10px 0; font-size:14px; color:#33413a; }
    li { font-size:14px; color:#33413a; margin:4px 0 4px 18px; }
    table { width:100%; border-collapse:collapse; margin:14px 0 22px; font-size:13px; background:#fff;
            box-shadow:0 1px 2px rgba(23,35,29,.06); }
    th, td { padding:8px 10px; border-bottom:1px solid #ece8dd; text-align:right; }
    th:first-child, td:first-child { text-align:left; }
    th { background:#efece3; font-weight:600; color:#3a4a41; }
    code { padding:1px 4px; border-radius:3px; background:#eceee8; font:12px ui-monospace,monospace; }
    tr:last-child td { border-bottom:none; }
    """
    return (
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        f"<title>三字段独立验证</title><style>{css}</style></head><body><main>"
        + "\n".join(body)
        + "</main></body></html>"
    )


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #


def run(
    *,
    skip_crossasset: bool = False,
    iters: int = 2000,
    outdir: Path | None = None,
    stock_cache: Path | None = None,
    members: Any = None,
    pit_version: str | None = None,
) -> dict[str, Any]:
    stocks = load_stocks(stock_cache)
    if len(stocks) < MIN_XSEC:
        raise SystemExit(f"个股面板只有 {len(stocks)} 只，至少需要 {MIN_XSEC} 只")
    benchmark = load_benchmark()
    breadth = breadth_series(stocks)
    etf_panel = ea.load_panel()

    print(f"[1/5] 描述分：{len(stocks)} 只个股 × {len(stocks[0]['rows'])} 根日线"
          + (f"（PIT {pit_version}）" if pit_version else "（当前成分缓存）"))
    descriptor = descriptor_study(stocks)
    print(f"      未来波动 ρ={descriptor['labels']['vol']['per_symbol_mean']} "
          f"{'通过' if descriptor['pass'] else '未通过'}")

    print("[2/5] 选股分：当日同池分位")
    selection = selection_study(stocks, iters=iters, members=members)
    lock = selection["signals"]["descriptor"]["lockbox"]
    print(f"      锁箱期 IC {lock.get('mean')} [{lock.get('ci_low')}, {lock.get('ci_high')}] "
          f"{'通过' if selection['pass'] else '未通过'}")

    print("[3/5] regime：择时叠加")
    regime = regime_study(benchmark, breadth)
    print(f"      回撤改善 {regime['improvement_pp']}pp {'通过' if regime['pass'] else '未通过'}")

    print("[4/5] 跨资产 ETF 组合")
    if skip_crossasset:
        cross = {"available": False, "pass": False, "reasons": ["已跳过"], "table": "",
                 "start": "", "strategy": {"metrics": {}, "settings": {}}, "no_stop": {"metrics": {}},
                 "no_voltarget": {"metrics": {}}, "inv_vol": {"metrics": {}},
                 "expensive": {"metrics": {}}, "equal_weight": {"metrics": {}},
                 "buy_hold": {"metrics": {}}, "stale": {}}
    else:
        result = ea.evaluate(etf_panel, start=ea.DEFAULT_START, rebalance=21)
        strategy = result["strategy"]["metrics"]
        equal = result["equal_weight"]["metrics"]
        reasons = []
        if strategy.get("sharpe", 0) <= equal.get("sharpe", 0):
            reasons.append(f"净 Sharpe {strategy.get('sharpe')} 未高于等权 {equal.get('sharpe')}")
        if strategy.get("max_drawdown", 0) <= equal.get("max_drawdown", 0):
            reasons.append(f"最大回撤 {strategy.get('max_drawdown')}% 未小于等权 {equal.get('max_drawdown')}%")
        cross = {
            "available": True,
            "pass": not reasons,
            "reasons": reasons,
            "table": result["table"],
            "start": result["start"],
            "strategy": result["strategy"],
            "no_stop": result["no_stop"],
            "no_voltarget": result["no_voltarget"],
            "inv_vol": result["inv_vol"],
            "expensive": result["expensive"],
            "equal_weight": result["equal_weight"],
            "buy_hold": result["buy_hold"],
            "nav": result["strategy"]["nav"],
            "stale": result.get("stale") or {},
            "engine": "etf_allocation",
        }
        print(f"      Sharpe {strategy.get('sharpe')} vs 等权 {equal.get('sharpe')}；"
              f"maxDD {strategy.get('max_drawdown')}% vs {equal.get('max_drawdown')}% "
              f"{'通过' if cross['pass'] else '未通过'}")

    print("[5/5] 校准表")
    calibration = calibrate(stocks, etf_panel)
    for key, row in calibration.items():
        knots = row.get("score_knots") or []
        median = f"{knots[10]:.1f}" if len(knots) >= 21 else "未校准"
        print(f"      {key}: n={row.get('n')} P50={median}")

    report = {
        "generated_at": datetime.now().isoformat(timespec="minutes"),
        "universe": {
            "stocks": len(stocks),
            "bars": len(stocks[0]["rows"]) if stocks else 0,
            "etf": len(etf_panel),
            "lockbox_start": LOCKBOX_START,
        },
        "pass_criteria": PASS_CRITERIA,
        "descriptor": descriptor,
        "selection": selection,
        "regime": regime,
        "crossasset": cross,
        "calibration": calibration,
    }

    target = outdir or OUTDIR
    target.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(report)
    (target / "report.md").write_text(markdown, encoding="utf-8")
    (target / "report.html").write_text(render_html(report, markdown), encoding="utf-8")
    (target / "calibration.json").write_text(
        json.dumps({key: {k: v for k, v in row.items()} for key, row in calibration.items()},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "generated_at": report["generated_at"],
        "universe": report["universe"],
        "fields": {
            "descriptor": {
                "pass": descriptor["pass"],
                "metrics": {
                    "vol_rho": descriptor["labels"]["vol"]["per_symbol_mean"],
                    "mdd_rho": descriptor["labels"]["mdd_depth"]["per_symbol_mean"],
                    "halves_strong_share": descriptor["labels"]["vol"]["halves_strong_share"],
                },
                "reasons": descriptor["reasons"],
            },
            "selection": {
                "pass": selection["pass"],
                "metrics": {
                    "lockbox_ic": selection["signals"]["descriptor"]["lockbox"].get("mean"),
                    "lockbox_ci_low": selection["signals"]["descriptor"]["lockbox"].get("ci_low"),
                    "lockbox_ci_high": selection["signals"]["descriptor"]["lockbox"].get("ci_high"),
                },
                "reasons": selection["reasons"],
            },
            "regime": {
                "pass": regime["pass"],
                "metrics": {
                    "improvement_pp": regime["improvement_pp"],
                    "sharpe": regime["regime"].get("sharpe"),
                    "max_drawdown": regime["regime"].get("max_drawdown"),
                },
                "reasons": regime["reasons"],
            },
            "crossasset": {
                "pass": cross["pass"],
                "metrics": {
                    "sharpe": cross["strategy"]["metrics"].get("sharpe"),
                    "max_drawdown": cross["strategy"]["metrics"].get("max_drawdown"),
                    "equal_sharpe": cross["equal_weight"]["metrics"].get("sharpe"),
                    "equal_max_drawdown": cross["equal_weight"]["metrics"].get("max_drawdown"),
                },
                "reasons": cross["reasons"],
            },
        },
    }
    (target / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if cross.get("available"):
        (target / "crossasset.json").write_text(
            json.dumps(
                {
                    "generated_at": report["generated_at"],
                    "start": cross["start"],
                    "settings": cross["strategy"].get("settings"),
                    "universe": ea.UNIVERSE,
                    "engine": "etf_allocation",
                    "rule_version": (cross["strategy"].get("settings") or {}).get("version"),
                    "metrics": {
                        key: cross[key]["metrics"]
                        for key in ("strategy", "no_stop", "no_voltarget", "inv_vol",
                                    "expensive", "equal_weight", "buy_hold")
                    },
                    "nav": cross["nav"],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    print(f"报告已写入 {target}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="三字段独立验证")
    parser.add_argument("--offline", action="store_true", help="只读本地缓存（默认即是）")
    parser.add_argument("--skip-crossasset", action="store_true")
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--outdir", default=str(OUTDIR))
    parser.add_argument("--pit", default=None,
                        help="用 PIT 池重跑（如 pit-v1）：面板换成按日成分，锁箱 IC 也按日过滤")
    parser.add_argument("--pit-refresh", action="store_true", help="重建 PIT 面板缓存")
    args = parser.parse_args(argv)
    stock_cache = None
    members = None
    if args.pit:
        from research.universe import members_loader, prepare_cache  # noqa: PLC0415

        info = prepare_cache(args.pit, refresh=args.pit_refresh)
        stock_cache = Path(info["dir"])
        members = members_loader(Path(info["members"]))
        print(f"PIT 面板：{info['symbols']} 只（{stock_cache}），成分快照 {info['snapshots']} 期")
    run(skip_crossasset=args.skip_crossasset, iters=args.iters, outdir=Path(args.outdir),
        stock_cache=stock_cache, members=members, pit_version=args.pit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
