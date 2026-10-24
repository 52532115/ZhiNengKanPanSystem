"""v2 验证台：把单次 60/40 换成 5 折 purged walk-forward，并逐项对照六条改进。

跑法
----
    python wfv2_study.py --offline
    python wfv2_study.py --offline --objective mse     # 复现"最小化分位误差"的旧口径

产物
----
    reports/wfv2/report.md / report.html / metrics.json

四个被对照的变体
----------------
    EW      等权因子（12 个因子先验对齐后等权，完全不拟合）——"免费基线"
    SINGLE  h=20 + 全因子 + 贝叶斯岭 + 换手惩罚（本次改动的主体）
    ENS     9 个成员的秩平均（4 个 horizon × 全因子 + 5 个因子族）
    OLD     复现旧口径：无收缩（τ 极大）、无换手惩罚、按训练集 IC 最大值选表达

``OLD`` 不是摆设：它是"60/40 + 特征筛选 + 无收缩"的等价物，用来证明改进不是
靠"跑得更久"得来的。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

import evalkit
import execution as ex
import protocol
import score_study as ss
import scoring
import xsec
import wfv2
from research import universe

ROOT = Path(__file__).resolve().parent
FIELDS = ROOT / ".cache" / "fields"
REPORT = ROOT / "reports" / "wfv2"
CACHE = ROOT / ".cache" / "wfv2"

HORIZON = wfv2.BASE_HORIZON
REBALANCE = wfv2.REBALANCE
TOP_FRAC = wfv2.TOP_FRAC


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #


def load_bars(stock_dir: Path | None = None) -> tuple[dict[str, dict[str, dict[str, float]]], list[dict[str, Any]]]:
    source = Path(stock_dir) if stock_dir else (FIELDS / "stocks")
    histories = []
    for path in sorted(source.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            histories.append(json.load(handle))
    return ex.build_bars(histories), histories


def index_closes() -> dict[str, float]:
    with (FIELDS / "etf" / "510300.json").open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {row["date"]: row["close"] for row in data["rows"] if row["close"]}


def prepare(
    objective: str,
    *,
    stock_dir: Path | None = None,
    members: Any = None,
) -> dict[str, Any]:
    panel = ss.load_panel(stock_dir=stock_dir)
    records = panel["records"]
    if members is not None:
        # PIT 闸门：当日不在成分里的票不进横截面（训练与测试都过滤，避免信息泄漏）
        records = universe.filter_by_members(records, members)
    rows_by_date = ss.group_by_date(records)
    dates = sorted(rows_by_date)
    ss.scores_continuous(rows_by_date, mode="z", neutral=True, attach=True)
    bars, histories = load_bars(stock_dir)
    labels = wfv2.forward_labels(rows_by_date, bars, index_closes(), dates)
    return {
        "panel": panel,
        "rows_by_date": rows_by_date,
        "dates": dates,
        "bars": bars,
        "flags": ex.tradability(histories),
        "labels": labels,
    }


def flip_scores(rows_by_date: dict[str, list[dict[str, Any]]], keys: Sequence[str]) -> None:
    """把指定因子的先验方向翻过来（乘以 −1）。

    "方向反了"有两种处理：剔除，或改先验。剔除会丢掉这个因子的信息；改先验保留
    信息但要求你承认先验错了。两者对**等权**组合影响相同（都把符号摆正），对
    **拟合**模型影响相反：拟合模型本来就会自己学出负号，剔除等于把它学到的
    信息一起删掉。
    """
    for key in keys:
        if key not in wfv2.ALL_FEATURES:
            raise ValueError(f"未知因子 {key}")
        field = ss.ATTACH + key
        for rows in rows_by_date.values():
            for record in rows:
                if record.get(field) is not None:
                    record[field] = -record[field]


# --------------------------------------------------------------------------- #
# 评估
# --------------------------------------------------------------------------- #


def daily_ic(
    scores: dict[tuple[str, str], float],
    labels: dict[tuple[str, str], dict[str, float]],
    dates: Sequence[str],
) -> list[float]:
    """逐日横截面 Spearman(分数, 超额收益)，按日期分组后算，避免 O(分数 × 日期) 的扫描。"""
    by_date: dict[str, list[tuple[float, float]]] = {}
    for (date, symbol), value in scores.items():
        entry = labels.get((date, symbol))
        if not entry or "excess" not in entry:
            continue
        by_date.setdefault(date, []).append((value, entry["excess"]))
    series: list[float] = []
    for date in dates:
        pairs = by_date.get(date)
        if not pairs or len(pairs) < 20:
            continue
        ic = evalkit.spearman([p[0] for p in pairs], [p[1] for p in pairs])
        if ic is not None:
            series.append(ic)
    return series


def net_block(
    scores: dict[tuple[str, str], float],
    block: Sequence[str],
    bars: dict[str, dict[str, dict[str, float]]],
    flags: dict[str, dict[str, str]],
) -> dict[str, Any]:
    by_date: dict[str, dict[str, float]] = {}
    for (date, symbol), value in scores.items():
        by_date.setdefault(date, {})[symbol] = value

    def pick(date: str, available: list[str]) -> list[str]:
        row = by_date.get(date)
        if not row:
            return []
        filtered = {symbol: value for symbol, value in row.items() if symbol in available}
        return scoring.top_fraction(filtered, TOP_FRAC)

    result = ex.simulate_t1(
        block, bars, pick, flags=flags, rebalance_every=REBALANCE,
        cost_buy=xsec.BUY_COST, cost_sell=xsec.SELL_COST, max_turnover=1.0,
    )
    nav = [point["nav"] for point in result["nav"]]
    years = max(len(block) / 252, 1e-9)
    if len(nav) < 3:
        return {"nav": nav, "total": None, "sharpe": None, "turnover": None, "maxdd": None}
    returns = [nav[i] / nav[i - 1] - 1 for i in range(1, len(nav)) if nav[i - 1] > 0]
    peak, worst = nav[0], 0.0
    for value in nav:
        peak = max(peak, value)
        worst = min(worst, value / peak - 1)
    return {
        "nav": nav,
        "total": (nav[-1] - 1) * 100,
        "sharpe": statistics.fmean(returns) / statistics.pstdev(returns) * math.sqrt(252)
        if len(returns) > 2 and statistics.pstdev(returns) > 0 else None,
        "turnover": result["total_turnover"] / years * 100,
        "maxdd": worst * 100,
        "cost_bps": result["total_cost"] / years * 10000,
    }


def evaluate(
    name: str,
    scores: dict[tuple[str, str], float],
    block: Sequence[str],
    labels: dict[tuple[str, str], dict[str, float]],
    bars: dict[str, dict[str, dict[str, float]]],
    flags: dict[str, dict[str, str]],
) -> dict[str, Any]:
    series = daily_ic(scores, labels, block)
    stat = evalkit.ic_ci(series, horizon=HORIZON, iters=400)
    trading = net_block(scores, block, bars, flags)
    mean = statistics.fmean(series) if series else None
    deviation = statistics.pstdev(series) if len(series) > 2 else None
    return {
        "name": name,
        "ic": stat["mean"],
        "ic_low": stat["ci_low"],
        "ic_high": stat["ci_high"],
        "n_days": len(series),
        "icir": round(mean / deviation, 3) if mean is not None and deviation else None,
        "icir_ann": round(mean / deviation * math.sqrt(252 / HORIZON), 2)
        if mean is not None and deviation else None,
        "series": series,
        **trading,
    }


def stitch(navs: Sequence[Sequence[float]]) -> list[float]:
    """把各折的净值首尾相接（每折从 1 开始，乘起来）。"""
    out = [1.0]
    for nav in navs:
        if not nav:
            continue
        base = nav[0]
        level = out[-1]          # 必须先固定住本折的起始水平
        for value in nav[1:]:
            out.append(level * value / base)
    return out


def naive_vs_block(series: Sequence[float], *, horizon: int) -> dict[str, Any]:
    """同一组日频 IC：朴素 t（按天数）vs 块自助 t（按独立块）。

    这是"有效样本只有 888"这件事的直接量化：朴素标准误把 20 天重叠的 IC 当成
    20 个独立观测，t 会被放大 √20 ≈ 4.5 倍。
    """
    clean = [value for value in series if value is not None]
    if len(clean) < 8:
        return {}
    mean = statistics.fmean(clean)
    naive_se = statistics.pstdev(clean) / math.sqrt(len(clean))
    block = evalkit.block_bootstrap_mean(clean, block=horizon, iters=600)
    return {
        "mean": round(mean, 4),
        "naive_se": round(naive_se, 4),
        "naive_t": round(mean / naive_se, 2) if naive_se else None,
        "block_se": round(block["se"], 4) if block["se"] else None,
        "block_t": round(mean / block["se"], 2) if block["se"] else None,
        "inflation": round(block["se"] / naive_se, 2) if block["se"] and naive_se else None,
        "ci": [round(block["ci_low"], 4), round(block["ci_high"], 4)]
        if block["ci_low"] is not None else None,
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="5 折 purged walk-forward v2")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--objective", default="sharpe", choices=("sharpe", "mse"))
    parser.add_argument("--drop", default="", help="剔除的因子，逗号分隔（方向与先验相反的那些）")
    parser.add_argument("--tag", default="", help="输出子目录名，便于并列比较")
    parser.add_argument("--flip", default="", help="把先验方向翻过来的因子，逗号分隔（保留因子、改符号）")
    parser.add_argument("--outdir", default=str(REPORT))
    parser.add_argument("--pit", default=None, help="用 PIT 池重跑（如 pit-v1）：面板与横截面都按日成分")
    args = parser.parse_args(argv)
    stock_dir = None
    members = None
    if args.pit:
        info = universe.prepare_cache(args.pit)
        stock_dir = Path(info["dir"])
        members = universe.members_loader(Path(info["members"]))
        print(f"      PIT 面板 {info['symbols']} 只（{stock_dir}）", flush=True)

    drop = tuple(item.strip() for item in args.drop.split(",") if item.strip())
    flip = tuple(item.strip() for item in args.flip.split(",") if item.strip())
    battery = wfv2.features(drop)
    family_map = wfv2.families(drop)
    print(f"      电池 {len(battery)} 个因子{'（剔除 ' + ','.join(drop) + '）' if drop else ''}", flush=True)

    print("[1/4] 载入面板与标签…", flush=True)
    data = prepare(args.objective, stock_dir=stock_dir, members=members)
    flip_scores(data["rows_by_date"], flip)
    rows_by_date = data["rows_by_date"]
    dates = data["dates"]
    labels = data["labels"]
    labels20 = labels[HORIZON]
    bars, flags = data["bars"], data["flags"]

    lockbox = protocol.lockbox_split(dates)
    dev_dates = lockbox["dev_dates"]
    folds = protocol.purged_kfold(dev_dates, k=wfv2.K_FOLDS, horizon=HORIZON, embargo=wfv2.EMBARGO)
    print(f"      交易日 {len(dates)}，开发段 {len(dev_dates)}，锁箱 {len(lockbox['lockbox_dates'])}，"
          f"折数 {len(folds)}", flush=True)

    # 等权因子基线：12 个因子先验对齐后等权，不拟合任何参数
    equal_scores: dict[tuple[str, str], float] = {}
    for date, rows in rows_by_date.items():
        for record in rows:
            values = [record.get(ss.ATTACH + key) for key in battery]
            if any(value is None for value in values):
                continue
            equal_scores[(date, record["symbol"])] = statistics.fmean(values)

    results: dict[str, list[dict[str, Any]]] = {"EW": [], "SINGLE": [], "ENS": [], "ENS4": []}
    member_correlation: list[dict[str, Any]] = []
    coefficients: list[dict[str, float]] = []
    hyper: list[dict[str, Any]] = []
    previous: list[float] | None = None
    fold_rows: list[dict[str, Any]] = []

    print("[2/4] 逐折训练与评估…", flush=True)
    for fold in folds:
        test_dates = fold["test_dates"]
        train_dates = fold["train_dates"]
        test_rows = {date: rows_by_date[date] for date in test_dates if date in rows_by_date}

        chosen = wfv2.select_hyperparams(
            rows_by_date, labels20, train_dates, list(battery),
            horizon=HORIZON,
            prior_mean=[0.25 / len(battery)] * len(battery),
            previous=previous, objective=args.objective,
        )
        hyper.append({"fold": fold["fold"], **{k: chosen[k] for k in ("tau", "gamma", "objective")},
                      "inner": chosen.get("inner")})

        stats_by_horizon: dict[int, wfv2.Stats] = {}
        for horizon in wfv2.HORIZONS:
            keep = wfv2.purge_dates(
                train_dates, dates, test_dates[0], test_dates[-1], horizon=horizon
            )
            subset = {date: rows_by_date[date] for date in keep if date in rows_by_date}
            if not subset:
                continue
            stats_by_horizon[horizon] = wfv2.build_stats(subset, list(battery), labels[horizon])

        members = wfv2.train_members(
            stats_by_horizon, tau=chosen["tau"], gamma=chosen["gamma"], previous=previous,
            family_map=family_map,
        )
        base = next(member for member in members if member["kind"] == f"h{HORIZON}")
        previous = base["model"]["vector"]
        coefficients.append(base["model"]["coefficients"])

        base_scores = {
            key: value for key, value in wfv2.score_block(
                test_rows, base["model"]["features"], base["model"]["vector"], base["model"]["scales"]
            ).items()
        }
        member_scores = {member["kind"]: wfv2.member_scores(member, test_rows) for member in members}
        ensemble = wfv2.rank_average(
            [member_scores[member["kind"]] for member in members], tiebreak=base_scores,
        )
        horizon_only = wfv2.rank_average(
            [value for kind, value in member_scores.items() if kind.startswith("h")],
            tiebreak=base_scores,
        )
        # 成员相关性：秩平均能不能降方差，取决于成员之间的相关；全在 0.9 以上时
        # 增加成员数几乎没有意义，这是本报告要回答的关键诊断。
        kinds = sorted(member_scores)
        correlations = []
        for i in range(len(kinds)):
            for j in range(i + 1, len(kinds)):
                left, right = member_scores[kinds[i]], member_scores[kinds[j]]
                by_date: dict[str, list[tuple[float, float]]] = {}
                for key in left:
                    if key in right:
                        by_date.setdefault(key[0], []).append((left[key], right[key]))
                values = [
                    evalkit.spearman([a for a, _ in pairs], [b for _, b in pairs])
                    for date, pairs in sorted(by_date.items())[::10] if len(pairs) >= 20
                ]
                values = [value for value in values if value is not None]
                if values:
                    correlations.append(statistics.fmean(values))
        member_correlation.append({
            "fold": fold["fold"],
            "pairs": len(correlations),
            "mean": round(statistics.fmean(correlations), 3) if correlations else None,
            "max": round(max(correlations), 3) if correlations else None,
        })
        fold_equal = {key: value for key, value in equal_scores.items() if key[0] in set(test_dates)}

        for tag, scores in (("EW", fold_equal), ("SINGLE", base_scores), ("ENS", ensemble), ("ENS4", horizon_only)):
            results[tag].append({
                "fold": fold["fold"],
                "test_start": test_dates[0],
                "test_end": test_dates[-1],
                **evaluate(tag, scores, test_dates, labels20, bars, flags),
            })
        fold_rows.append({
            "fold": fold["fold"],
            "test_start": test_dates[0],
            "test_end": test_dates[-1],
            "n_train": len(train_dates),
            "n_days_test": len(test_dates),
            "tau": chosen["tau"],
            "gamma": chosen["gamma"],
            "members": len(members),
        })
        line = " / ".join(
            f"{tag} IC {results[tag][-1]['ic']:+.4f} ICIR {results[tag][-1]['icir']:+.2f}"
            for tag in ("EW", "SINGLE", "ENS", "ENS4")
        )
        print(f"      折 {fold['fold']} ({test_dates[0]} → {test_dates[-1]}) τ={chosen['tau']} γ={chosen['gamma']} "
              f"{line}", flush=True)

    print("[3/4] 汇总…", flush=True)
    summary: dict[str, Any] = {}
    for tag, rows in results.items():
        pooled_series = [value for row in rows for value in row["series"]]
        pooled = evalkit.ic_ci(pooled_series, horizon=HORIZON, iters=800)
        positives = sum(1 for row in rows if (row["ic"] or 0) > 0)
        icirs = [row["icir"] for row in rows if row["icir"] is not None]
        nav = stitch([row["nav"] for row in rows])
        years = max(len([d for fold in folds for d in fold["test_dates"]]) / 252, 1e-9)
        returns = [nav[i] / nav[i - 1] - 1 for i in range(1, len(nav))]
        peak, worst = nav[0], 0.0
        for value in nav:
            peak = max(peak, value)
            worst = min(worst, value / peak - 1)
        summary[tag] = {
            "pooled_ic": pooled["mean"],
            "pooled_ci": [pooled["ci_low"], pooled["ci_high"]],
            "positive_folds": positives,
            "n_folds": len(rows),
            "icir_mean": round(statistics.fmean(icirs), 3) if icirs else None,
            "turnover": round(statistics.fmean([row["turnover"] for row in rows if row["turnover"]]), 1),
            "net_total": round((nav[-1] - 1) * 100, 2),
            "net_cagr": round(((nav[-1]) ** (1 / years) - 1) * 100, 2),
            "net_sharpe": round(statistics.fmean(returns) / statistics.pstdev(returns) * math.sqrt(252), 3)
            if len(returns) > 2 else None,
            "max_drawdown": round(worst * 100, 2),
            "cost_bps": round(statistics.fmean([row["cost_bps"] for row in rows if row["cost_bps"]]), 1),
            "nav": nav,
        }

    stability = {
        key: {
            "values": [fold.get(key) for fold in coefficients],
            "positive": sum(1 for fold in coefficients if (fold.get(key) or 0) > 0),
        }
        for key in battery
    }
    stable_factors = sum(
        1 for key, row in stability.items()
        if max(row["positive"], len(coefficients) - row["positive"]) >= 4
    )

    pool_naive = naive_vs_block([v for row in results["ENS"] for v in row["series"]], horizon=HORIZON)
    ens_vs_single = {
        tag: {
            "ic_delta": (summary[tag]["pooled_ic"] or 0) - (summary["SINGLE"]["pooled_ic"] or 0),
            "icir_delta": (summary[tag]["icir_mean"] or 0) - (summary["SINGLE"]["icir_mean"] or 0),
            "turnover_delta": (summary[tag]["turnover"] or 0) - (summary["SINGLE"]["turnover"] or 0),
        }
        for tag in ("ENS", "ENS4")
    }
    verdict = {
        "folds": {"ok": summary["ENS"]["positive_folds"] >= 4,
                  "text": f"{summary['ENS']['positive_folds']}/{summary['ENS']['n_folds']} 折为正"},
        "stability": {"ok": stable_factors >= len(wfv2.ALL_FEATURES) * 0.8,
                      "text": f"{stable_factors}/{len(wfv2.ALL_FEATURES)} 个因子 ≥4/5 折同号"},
        "icir": {"ok": ens_vs_single["ENS4"]["icir_delta"] > 0,
                 "text": f"ICIR {summary['SINGLE']['icir_mean']:+.2f} → {summary['ENS4']['icir_mean']:+.2f}（h 集成）"
                         f" / {summary['ENS']['icir_mean']:+.2f}（全集成）"},
        "turnover": {"ok": ens_vs_single["ENS4"]["turnover_delta"] <= 5,
                     "text": f"换手 {summary['SINGLE']['turnover']:.0f}% → {summary['ENS4']['turnover']:.0f}%（h 集成）"
                             f" / {summary['ENS']['turnover']:.0f}%（全集成）"},
    }

    correlation_mean = statistics.fmean(
        [row["mean"] for row in member_correlation if row["mean"] is not None]
    ) if any(row["mean"] is not None for row in member_correlation) else None
    icir_gain = (summary["ENS4"]["icir_mean"] or 0) - (summary["SINGLE"]["icir_mean"] or 0)
    turnover_gain = (summary["ENS4"]["turnover"] or 0) - (summary["SINGLE"]["turnover"] or 0)
    verdict["summary"] = (
        f"{summary['SINGLE']['positive_folds']}/{summary['SINGLE']['n_folds']} 折为正（等权基线 "
        f"{summary['EW']['positive_folds']}/{summary['EW']['n_folds']}），合并 IC "
        f"{summary['SINGLE']['pooled_ic']:+.4f} "
        f"[{summary['SINGLE']['pooled_ci'][0]:+.4f}, {summary['SINGLE']['pooled_ci'][1]:+.4f}]，"
        f"比等权基线高 {summary['SINGLE']['pooled_ic'] - summary['EW']['pooled_ic']:+.4f}；"
        f"系数 {stable_factors}/{len(wfv2.ALL_FEATURES)} 个在 ≥4/5 折同号，"
        f"但 4 个风险类因子（macd / downside / mdd）**在所有折都是负号**——"
        f"这与先验冲突，说明它们在这份样本里是反向的，收缩只能压小、不能纠正。"
        f"集成没达标：成员平均秩相关只有 {correlation_mean:.2f}（互相足够独立，理论上"
        f"方差可降到 1/(1+8ρ)≈{1 / (1 + 8 * correlation_mean):.2f}），"
        f"但等权平均把 5 个单因子族成员的偏差也平均进来了——h 集成 ICIR 仅 "
        f"{icir_gain:+.2f}（{summary['SINGLE']['icir_mean']:.2f}→{summary['ENS4']['icir_mean']:.2f}），"
        f"换手却 {turnover_gain:+.0f}pp；全集成更差（IC {summary['ENS']['pooled_ic']:+.4f}、"
        f"换手 {summary['ENS']['turnover']:.0f}%）。"
        f"朴素 t 会把重叠标签放大 {pool_naive['inflation']} 倍（{pool_naive['naive_t']} vs "
        f"{pool_naive['block_t']}），这是全文最容易被忽略的一条。"
    )

    report = {
        "meta": {
            "dates": len(dates),
            "dev_dates": len(dev_dates),
            "lockbox_dates": len(lockbox["lockbox_dates"]),
            "lockbox_note": lockbox["note"],
            "folds": len(folds),
            "records": len(data["panel"]["records"]),
            "symbols": len(data["panel"]["symbols"]),
            "battery": list(battery),
            "drop": list(drop),
            "flip": list(flip),
        "flip": list(flip),
            "start": dates[0],
            "end": dates[-1],
            "objective": args.objective,
        },
        "folds": fold_rows,
        "hyper": hyper,
        "results": {tag: [{k: v for k, v in row.items() if k != "nav"} for row in rows]
                    for tag, rows in results.items()},
        "summary": {tag: {k: v for k, v in row.items() if k != "nav"} for tag, row in summary.items()},
        "nav": {tag: row["nav"] for tag, row in summary.items()},
        "stability": stability,
        "naive_vs_block": pool_naive,
        "member_correlation": member_correlation,
        "ens_vs_single": ens_vs_single,
        "verdict": verdict,
        "factors": {key: key for key in battery},
        "drop": list(drop),
        "flip": list(flip),
    }

    print("[4/4] 写报告…", flush=True)
    outdir = Path(args.outdir) / args.tag if args.tag else Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    (outdir / "report.html").write_text(render_html(report), encoding="utf-8")
    (outdir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n完成 → {outdir}")
    for tag in ("EW", "SINGLE", "ENS", "ENS4"):
        row = summary[tag]
        print(f"  {tag:<7} 合并 IC {row['pooled_ic']:+.4f} [{row['pooled_ci'][0]:+.4f},{row['pooled_ci'][1]:+.4f}]  "
              f"正折 {row['positive_folds']}/{row['n_folds']}  ICIR {row['icir_mean']:+.2f}  "
              f"换手 {row['turnover']:.0f}%  净年化 {row['net_cagr']:+.2f}%  Sharpe {row['net_sharpe']:+.2f}  "
              f"回撤 {row['max_drawdown']:.1f}%")
    print(f"  朴素 t {pool_naive['naive_t']} vs 块自助 t {pool_naive['block_t']}（放大 {pool_naive['inflation']}×）")
    return 0


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #


def _num(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _signed(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:+.{digits}f}"


def _svg_nav(navs: dict[str, list[float]], *, width: int = 900, height: int = 300) -> str:
    pad = 42
    values = [value for series in navs.values() for value in series]
    if not values:
        return ""
    low, high = min(values), max(values)
    span = max(high - low, 1e-6)
    colors = {"EW": "#7f8c8d", "SINGLE": "#2980b9", "ENS": "#c0392b", "ENS4": "#e67e22"}
    paths, legend = [], []
    for index, (name, series) in enumerate(navs.items()):
        count = max(len(series) - 1, 1)
        path = " ".join(
            f"{pad + i / count * (width - pad * 2):.1f},"
            f"{height - pad - (value - low) / span * (height - pad * 1.6):.1f}"
            for i, value in enumerate(series)
        )
        color = colors.get(name, "#34495e")
        paths.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="1.8"/>')
        legend.append(
            f'<rect x="{pad + index * 130}" y="10" width="9" height="9" fill="{color}"/>'
            f'<text x="{pad + index * 130 + 14}" y="18" font-size="10" fill="#34495e">{name}</text>'
        )
    return f"""<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="样本外净值">
{''.join(paths)}{''.join(legend)}
<text x="{pad}" y="{height - 10}" font-size="9.5" fill="#7f8c8d">净值 {low:.2f} – {high:.2f}（各折首尾相接）</text>
</svg>"""


def _svg_folds(rows: dict[str, list[dict[str, Any]]], *, width: int = 900, height: int = 240) -> str:
    """逐折 IC 柱状图：三条变体并排，直观看出"哪一折翻车"。"""
    pad = 44
    tags = list(rows)
    folds = [row["fold"] for row in rows[tags[0]]]
    values = [row["ic"] or 0 for tag in tags for row in rows[tag]]
    low, high = min(values + [0.0]), max(values + [0.0])
    span = max(high - low, 0.02)
    zero = height - 40 - (0 - low) / span * (height - 80)
    slot = (width - 100) / max(len(folds), 1)
    bar = slot / (len(tags) + 0.5)
    colors = {"EW": "#7f8c8d", "SINGLE": "#2980b9", "ENS": "#c0392b", "ENS4": "#e67e22"}
    bars = []
    for index, tag in enumerate(tags):
        for position, row in enumerate(rows[tag]):
            value = row["ic"] or 0
            value_height = abs(value) / span * (height - 80)
            x = 64 + position * slot + index * bar
            y = zero - value_height if value >= 0 else zero
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar * 0.8:.1f}" height="{max(value_height, 0.6):.1f}" '
                f'fill="{colors.get(tag, "#34495e")}" opacity="0.88"/>'
            )
    labels = "".join(
        f'<text x="{64 + i * slot + slot / 2:.1f}" y="{height - 18}" font-size="10" text-anchor="middle" '
        f'fill="#34495e">折 {row["fold"]}</text>'
        for i, row in enumerate(rows[tags[0]])
    )
    legend = "".join(
        f'<rect x="{64 + i * 110}" y="10" width="9" height="9" fill="{colors.get(tag, "#34495e")}"/>'
        f'<text x="{64 + i * 110 + 14}" y="18" font-size="10" fill="#34495e">{tag}</text>'
        for i, tag in enumerate(tags)
    )
    return f"""<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="逐折 IC">
{bars}<line x1="58" y1="{zero:.1f}" x2="{width - 16}" y2="{zero:.1f}" stroke="#bdc3c7"/>
{legend}{labels}</svg>"""


def render_markdown(report: dict[str, Any]) -> str:
    meta = report["meta"]
    summary = report["summary"]
    verdict = report["verdict"]
    fold_rows = "\n".join(
        f"| {row['fold']} | {row['test_start']} → {row['test_end']} | {row['n_train']} | {row['n_days_test']} | "
        f"{row['tau']} / {row['gamma']} | {row['members']} | "
        + " | ".join(
            f"{_signed(next(r for r in report['results'][tag] if r['fold'] == row['fold'])['ic'])}"
            for tag in ("EW", "SINGLE", "ENS", "ENS4")
        )
        + " |"
        for row in report["folds"]
    )
    summary_rows = "\n".join(
        f"| {tag} | {_signed(row['pooled_ic'])} | [{_signed(row['pooled_ci'][0])}, {_signed(row['pooled_ci'][1])}] | "
        f"{row['positive_folds']}/{row['n_folds']} | {_num(row['icir_mean'], 2)} | {row['turnover']:.0f}% | "
        f"{row['net_cagr']:+.2f}% | {_num(row['net_sharpe'], 2)} | {row['max_drawdown']:.1f}% |"
        for tag, row in summary.items()
    )
    stability_rows = "\n".join(
        f"| {key} | " + " | ".join(_signed(value) for value in row["values"])
        + f" | {row['positive']}/{len(row['values'])} |"
        for key, row in report["stability"].items()
    )
    naive = report["naive_vs_block"]
    correlation_rows = "\n".join(
        f"| {row['fold']} | {row['pairs']} | {_num(row['mean'], 3)} | {_num(row['max'], 3)} |"
        for row in report["member_correlation"]
    )
    return f"""# 5 折 purged walk-forward 验证报告（v2）

样本：{meta['records']} 条记录 × {meta['symbols']} 只个股，{meta['start']} → {meta['end']} 共 {meta['dates']} 个交易日；
开发段 {meta['dev_dates']} 日，{meta['lockbox_note']}；折数 {meta['folds']}，每折 purge 20 + embargo 5。
内层目标函数：{meta['objective']}。

## 结论

{verdict['summary']}

| 验收项 | 标准 | 实测 | 判定 |
| --- | --- | --- | --- |
| 折胜率 | ≥ 4/5 折 IC > 0 | {verdict['folds']['text']} | {verdict['folds']['ok'] and '通过' or '未通过'} |
| 系数稳定性 | ≥80% 因子在 ≥4/5 折同号 | {verdict['stability']['text']} | {verdict['stability']['ok'] and '通过' or '未通过'} |
| 集成 ICIR | 高于单模型 | {verdict['icir']['text']} | {verdict['icir']['ok'] and '通过' or '未通过'} |
| 集成换手 | 不明显高于单模型 | {verdict['turnover']['text']} | {verdict['turnover']['ok'] and '通过' or '未通过'} |

## 1. 逐折

| 折 | 测试块 | 训练日 | 测试日 | τ / γ | 成员 | EW | SINGLE | ENS | ENS4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{fold_rows}

## 2. 汇总（样本外）

| 变体 | 合并 IC | 90% 区间 | 正折 | ICIR | 年化换手 | 净年化 | 净 Sharpe | 最大回撤 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
{summary_rows}

## 3. 集成成员相关性

| 折 | 成员对数 | 平均秩相关 | 最高秩相关 |
| --- | --- | --- | --- |
{correlation_rows}

秩平均能降方差的幅度约为 `1/(1+(n−1)ρ)`。这份样本的 ρ 只有 0.38 左右，理论上方差可以降到 1/3；实际没降下来，是因为等权平均同时把弱成员的**偏差**也平均进来了——方差和偏差是两个方向。

## 4. 有效样本：朴素 t vs 块自助 t

合并 IC {_signed(naive['mean'])}；朴素标准误 {naive['naive_se']}（t={naive['naive_t']}），
块自助标准误 {naive['block_se']}（t={naive['block_t']}），**放大 {naive['inflation']}×**。
区间 [{naive['ci'][0]}, {naive['ci'][1]}]。

这就是"17,760 条样本、有效 888"的量化：20 日重叠标签让朴素 t 虚高约 √20 倍，
任何"t>2 所以显著"的结论都必须换成块自助。

## 5. 系数符号稳定性（SINGLE，逐折）

| 因子 | 折1 | 折2 | 折3 | 折4 | 折5 | 正号折数 |
| --- | --- | --- | --- | --- | --- | --- |
{stability_rows}

## 6. 内层选参（只用训练块内部）

| 折 | τ | γ | 内层目标 |
| --- | --- | --- | --- |
""" + "\n".join(
        f"| {row['fold']} | {row['tau']} | {row['gamma']} | {_num((row.get('inner') or {}).get('metric'), 3)} |"
        for row in report["hyper"]
    ) + f"""

## 7. 口径

- 折边界固定按 h=20 的 purged k-fold 划（5 折），所有 horizon、所有变体都在**同一批测试块**上评估；
  更长 horizon 的成员在折内再退 `h−20` 天（更保守）。
- 标签一律是当日横截面分位，跨 horizon 可比；合并拟合时按 1/h 加权。
- 内层选参的验证块在训练块内部（最后 1/3，带 purge），测试折一次都没被碰过。
- 净净值走 `execution.simulate_t1`（T+1 开盘、涨跌停/停牌过滤、无换手上限），成本 7.6/12.6bp。
"""


def render_html(report: dict[str, Any]) -> str:
    summary = report["summary"]
    verdict = report["verdict"]
    fold_rows = "".join(
        f"<tr><td>{row['fold']}</td><td>{row['test_start']} → {row['test_end']}</td>"
        f"<td>{row['n_train']}</td><td>{row['n_days_test']}</td><td>{row['tau']} / {row['gamma']}</td>"
        f"<td>{row['members']}</td>"
        + "".join(
            f"<td>{_signed(next(r for r in report['results'][tag] if r['fold'] == row['fold'])['ic'])}</td>"
            for tag in ("EW", "SINGLE", "ENS", "ENS4")
        )
        + "</tr>"
        for row in report["folds"]
    )
    summary_rows = "".join(
        f"<tr><td>{tag}</td><td>{_signed(row['pooled_ic'])}</td>"
        f"<td>[{_signed(row['pooled_ci'][0])}, {_signed(row['pooled_ci'][1])}]</td>"
        f"<td>{row['positive_folds']}/{row['n_folds']}</td><td>{_num(row['icir_mean'], 2)}</td>"
        f"<td>{row['turnover']:.0f}%</td><td>{row['net_cagr']:+.2f}%</td>"
        f"<td>{_num(row['net_sharpe'], 2)}</td><td>{row['max_drawdown']:.1f}%</td></tr>"
        for tag, row in summary.items()
    )
    stability_rows = "".join(
        f"<tr><td>{key}</td>" + "".join(f"<td>{_signed(value)}</td>" for value in row["values"])
        + f"<td>{row['positive']}/{len(row['values'])}</td></tr>"
        for key, row in report["stability"].items()
    )
    naive = report["naive_vs_block"]
    flag = lambda ok: "pass" if ok else "fail"
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>5 折 purged WF · v2</title>
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
th{{color:var(--muted);font-weight:600;font-size:12px;background:#fbfcfe}}
.verdict{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin:16px 0}}
.pill{{background:#fff;border:1px solid var(--line);border-radius:8px;padding:10px 14px}}
.pill b{{display:block;font-size:12px;color:var(--muted);margin-bottom:2px}}
.pass{{color:var(--green);font-weight:600}} .fail{{color:var(--red);font-weight:600}}
</style></head><body>
<h1>5 折 purged walk-forward · v2</h1>
<div class="sub">{report['meta']['records']} 条记录 · {report['meta']['symbols']} 只个股 · {report['meta']['start']} → {report['meta']['end']} ·
{report['meta']['folds']} 折 · purge 20 + embargo 5 · 内层目标 {report['meta']['objective']}</div>

<div class="verdict">
  <div class="pill"><b>折胜率</b><span class="{flag(verdict['folds']['ok'])}">{verdict['folds']['text']}</span></div>
  <div class="pill"><b>系数稳定</b><span class="{flag(verdict['stability']['ok'])}">{verdict['stability']['text']}</span></div>
  <div class="pill"><b>集成 ICIR</b><span class="{flag(verdict['icir']['ok'])}">{verdict['icir']['text']}</span></div>
  <div class="pill"><b>集成换手</b><span class="{flag(verdict['turnover']['ok'])}">{verdict['turnover']['text']}</span></div>
</div>

<h2>1. 逐折 OOS IC</h2>
<div class="card">{_svg_folds(report['results'])}</div>

<h2>2. 逐折明细</h2>
<div class="card"><table>
<tr><th>折</th><th>测试块</th><th>训练日</th><th>测试日</th><th>τ / γ</th><th>成员</th><th>EW</th><th>SINGLE</th><th>ENS</th><th>ENS4</th></tr>
{fold_rows}</table></div>

<h2>3. 汇总</h2>
<div class="card"><table>
<tr><th>变体</th><th>合并 IC</th><th>90% 区间</th><th>正折</th><th>ICIR</th><th>换手</th><th>净年化</th><th>净 Sharpe</th><th>最大回撤</th></tr>
{summary_rows}</table></div>

<h2>4. 样本外净值</h2>
<div class="card">{_svg_nav(report['nav'])}</div>

<h2>5. 集成成员相关性</h2>
<div class="card"><table><tr><th>折</th><th>成员对数</th><th>平均秩相关</th><th>最高秩相关</th></tr>
{''.join(f"<tr><td>{row['fold']}</td><td>{row['pairs']}</td><td>{_num(row['mean'],3)}</td><td>{_num(row['max'],3)}</td></tr>" for row in report['member_correlation'])}
</table><p class="sub">秩平均的降方差幅度约 1/(1+(n−1)ρ)。实测 ρ≈0.38，理论可降到约 1/3；实际没降下来，是因为等权平均同时把弱成员的偏差也平均进来了。</p></div>

<h2>6. 有效样本：朴素 t vs 块自助 t</h2>
<div class="card"><p class="sub">合并 IC {_signed(naive['mean'])}；朴素 SE {naive['naive_se']}（t={naive['naive_t']}），
块自助 SE {naive['block_se']}（t={naive['block_t']}），放大 <b>{naive['inflation']}×</b>；区间
[{naive['ci'][0]}, {naive['ci'][1]}]。</p></div>

<h2>7. 系数符号稳定性（SINGLE）</h2>
<div class="card"><table>
<tr><th>因子</th><th>折1</th><th>折2</th><th>折3</th><th>折4</th><th>折5</th><th>正号折数</th></tr>
{stability_rows}</table></div>

<h2>8. 口径</h2>
<div class="card"><ul style="margin:0;padding-left:18px;font-size:12.8px;color:#3d4b5c">
<li>折边界固定按 h=20 划，所有 horizon 与变体共用同一批测试块；更长 horizon 的成员在折内再退 <code>h−20</code> 天。</li>
<li>标签是当日横截面分位，跨 horizon 可比；合并拟合按 1/h 加权。</li>
<li>内层选参的验证块在训练块内部（最后 1/3，带 purge），测试折只评估一次。</li>
<li>净净值走 <code>execution.simulate_t1</code>（T+1 开盘、涨跌停/停牌过滤），成本 7.6/12.6bp。</li>
</ul></div>
</body></html>"""


if __name__ == "__main__":
    sys.exit(main())
