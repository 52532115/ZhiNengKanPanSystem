"""池层验证：把"股票池怎么建"当成变量，而不是背景假设。

四组对比，每组只改一个东西：

- **A 当前成分 vs PIT 成分**：都用今天的名单回看历史（幸存者）对照每日真实名单。
- **B 含退市 vs 不含退市**：PIT 池里保留后来退市/被并购的票，还是提前踢掉。
- **C 60 只大盘 vs 300–400 只分层**：旧口径（市值前 60）对照按行业×市值分层抽样。
- **D 分层回测**：同一策略分别跑沪深300 / 中证500 / 小盘层，看是否只在某层有效。

每组输出横截面 IC（带块自助区间）与 T+1 扣费净净值；差异超过 30% 直接判定
旧结论不可用。同时报告行业/市值覆盖与有效宽度，避免"结论对但池子歪"。

命令行::

    python -m research.pool_validate --version pit-v1 --target 400
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import evalkit as ek
import execution as ex
import protocol
import xsec
from market_data import MarketDataError

from . import clean, pit
from .manifest import CACHE_ROOT, ROOT, Dataset, latest_version
from .universe import PitUniverse, field_audit, to_history

REPORT_DIR = ROOT / "reports" / "pool"
EVAL_START = "2018-01-01"
LAYER_LABELS = {
    "hs300": "沪深300（PIT）",
    "zz500": "中证500（PIT）",
    "small": "小盘层（PIT 成分内市值后 1/3）",
    "all": "分层抽样（PIT 成分并集）",
}
INDEX_BY_LAYER = {"hs300": "000300", "zz500": "000905"}


# --------------------------------------------------------------------------- #
# 数据准备
# --------------------------------------------------------------------------- #


def load_context(version: str, *, root: Path | None = None) -> dict[str, Any]:
    universe = PitUniverse(version, root=root)
    dataset = universe.dataset
    return {
        "dataset": dataset,
        "universe": universe,
        "bars": universe.bars,
        "industry": universe.industry,
        "book": universe.book,
        "listed": dataset.read_table("listed") if (dataset.root / "listed.json").is_file() else {},
        "calendar": universe.calendar,
        "benchmark": json.loads((dataset.root / "benchmark.json").read_text(encoding="utf-8"))
        if (dataset.root / "benchmark.json").is_file() else None,
        "verify": dataset.verify(),
    }


def alive_at_end(bars: dict[str, list[dict[str, Any]]], *, min_symbols: int = 50) -> set[str]:
    """样本末期仍有行情的标的（用来近似"在市"）。

    只有在日线覆盖足够广时这个近似才成立：10 只样本算出来的"在市"名单会把
    几乎所有被换出的票都判成退市，所以在覆盖不足时直接返回空集，让报告显示
    "未判定"而不是给一个错的数。
    """
    if len(bars) < min_symbols:
        return set()
    ends = {symbol: rows[-1]["date"] for symbol, rows in bars.items() if rows}
    if not ends:
        return set()
    last = max(ends.values())
    return {symbol for symbol, end in ends.items() if end >= last[:4] + "-06-01"}


# --------------------------------------------------------------------------- #
# 单次评估
# --------------------------------------------------------------------------- #


def _returns_from_rows(rows: Sequence[dict[str, Any]], *, skip: int = 2) -> list[float]:
    """用后复权收盘算日收益；跳过上市头两天（首行是发行价 → 首日收盘）。"""
    closes = [float(row["close"]) for row in rows]
    result = []
    for index in range(max(skip, 1), len(closes)):
        if closes[index - 1] > 0:
            value = closes[index] / closes[index - 1] - 1.0
            if abs(value) <= 0.25:
                result.append(value)
    return result


def evaluate(
    *,
    histories: Sequence[dict[str, Any]],
    benchmark: dict[str, Any],
    label: str,
    eval_start: str,
    eval_end: str,
    horizon: int = protocol.HORIZON,
    folds: int = protocol.K_FOLDS,
    rebalance: int = protocol.REBALANCE,
    top_fraction: float = protocol.TOP_FRACTION,
    max_turnover: float = protocol.MAX_TURNOVER,
    iters: int = 800,
    members: Any = None,
    bundle: dict[str, Any] | None = None,
    bars: dict[str, Any] | None = None,
    flags: dict[str, Any] | None = None,
    min_xsec: int = protocol.MIN_XSEC,
) -> dict[str, Any]:
    """在一个给定股票池上跑 5 折 purged walk-forward：IC + T+1 净净值。

    ``members`` 是**逐日**回调，不是一份固定名单：PIT 口径下同一天里能持有的票
    随成分调整而变，用末期成分回跑全历史会把中途退出的样本整段删掉。
    ``bundle`` / ``bars`` / ``flags`` 允许外部复用同一次面板构建——9 次评估各建
    一遍 500 只 × 2600 行的字典是纯浪费。
    """
    if len(histories) < min_xsec:
        return {"label": label, "available": False, "symbols": len(histories),
                "note": f"有日线的标的只有 {len(histories)} 只，不足 {min_xsec} 只"}
    if bundle is None:
        bundle = protocol.build_records(histories, benchmark, horizon=horizon)
    if bars is None:
        bars = ex.build_bars(histories)
    if flags is None:
        flags = ex.tradability(histories)
    records = [
        record for record in bundle["records"]
        if eval_start <= record["date"] <= eval_end
        and (members is None or record["symbol"] in members(record["date"]))
    ]
    dates = sorted({record["date"] for record in records})
    if len(dates) < horizon * (folds + 1):
        return {"label": label, "available": False, "note": f"评估期只有 {len(dates)} 个交易日"}
    fold_list = protocol.purged_kfold(dates, k=folds, horizon=horizon)
    walk = protocol.run_walkforward(records, fold_list, horizon=horizon, iters=iters)

    calendar = sorted({row["date"] for history in histories for row in history["rows"]
                       if eval_start <= row["date"] <= eval_end})
    strategy = protocol.walkforward_nav(calendar, bars, flags, walk, fold_list,
                                       rebalance=rebalance, top_fraction=top_fraction,
                                       max_turnover=max_turnover, members=members)
    # 等权基准必须用同一个逐日回调：否则"跑赢等权"比的是含非成分股的池子。
    benchmark_nav = protocol.equal_weight_nav(calendar, bars, rebalance=rebalance,
                                              members=members)
    stat = xsec.metrics(strategy, len(strategy["nav"]))
    bench_stat = xsec.metrics(benchmark_nav, len(benchmark_nav["nav"]))
    daily = protocol.period_excess(strategy, benchmark_nav, step=max(rebalance, horizon))
    holdings = [item for item in strategy["holdings"] if item.get("symbols")]
    returns = {history["symbol"]: _returns_from_rows(history["rows"]) for history in histories}
    last_holdings = holdings[-1]["symbols"] if holdings else []
    breadth = pit.breadth_metrics(last_holdings, returns)
    coverage = None
    if members is not None:
        checked = dates[::5]
        ratios = [
            sum(1 for record in bundle["records"]
                if record["date"] == day and record["symbol"] in members(day))
            / max(len(members(day)), 1)
            for day in checked
        ]
        coverage = round(statistics.fmean(ratios) * 100, 1) if ratios else None
    per_day = [len({record["symbol"] for record in records if record["date"] == day})
               for day in dates]
    return {
        "label": label,
        "available": True,
        "symbols": len(histories),
        "xsec_mean": round(statistics.fmean(per_day), 1) if per_day else None,
        "member_coverage": coverage,
        "dates": len(dates),
        "folds": len(fold_list),
        "ic": walk["pooled"],
        "sigma_within": walk["sigma_within"],
        "nav": {
            "total_return": stat["total_return"],
            "annual": stat["annual_return"],
            "max_drawdown": stat["max_drawdown"],
            "sharpe": stat["sharpe"],
        },
        "benchmark": {
            "total_return": bench_stat["total_return"],
            "annual": bench_stat["annual_return"],
            "max_drawdown": bench_stat["max_drawdown"],
        },
        "excess": round(stat["total_return"] - bench_stat["total_return"], 2),
        "excess_ci": ek.block_bootstrap_mean(daily, block=max(2, len(daily) // 3), iters=iters),
        "breadth": breadth,
        "holdings_mean": round(statistics.fmean([len(item["symbols"]) for item in holdings]), 1)
        if holdings else None,
    }


def diff_pct(new: float | None, old: float | None) -> float | None:
    """相对差异：以旧口径为分母，超过 30% 判定旧结论不可用。"""
    if new is None or old is None or old == 0:
        return None
    return round((new - old) / abs(old) * 100, 1)


# --------------------------------------------------------------------------- #
# 池子构造
# --------------------------------------------------------------------------- #


def universe_codes(
    day: str,
    *,
    layer: str,
    ctx: dict[str, Any],
    target: int,
    exclude_delisted: bool = False,
    use_current_members: bool = False,
    top_by_size: int | None = None,
) -> list[str]:
    """按层给出**当日**标的列表（PIT 口径；``use_current_members`` 才是幸存者口径）。

    ``small`` 的定义：PIT 成分并集里当日自由流通市值最低的 1/3。原先想要的是
    "非成分小盘股抽样"，但那需要全市场日线（4000+ 只），免费源拉不完；用成分内的
    市值后 1/3 至少保证样本都真实存在于当日池子里，不会被"当年还没上市/已退市"
    这类空洞污染。报告里会写明这个替代口径。
    """
    bars, industry, book = ctx["bars"], ctx["industry"], ctx["book"]
    snapshot = clean.panel_slice(bars, day)
    alive = ctx["alive"]
    if use_current_members:
        # 幸存者偏差口径：始终用最后一期成分（"今天还在指数里"的票）
        members = set(book.members(book.days[-1], INDEX_BY_LAYER.get(layer, "000300")))
    elif layer in INDEX_BY_LAYER:
        members = set(book.members(day, INDEX_BY_LAYER[layer]))
    else:                                    # small / all：PIT 成分并集
        members = set()
        for index in ("000300", "000905"):
            members |= set(book.members(day, index))

    pool = pit.build_pool(
        day,
        snapshot=snapshot,
        members=members,
        industry=industry,
        exclude_delisted=(lambda code: code not in alive) if exclude_delisted else None,
    )
    if layer == "small":
        sized = sorted(item["mcap"] or 0.0 for item in pool)
        cut = sized[len(sized) // 3] if sized else 0.0
        codes = sorted(item["symbol"] for item in pool if (item["mcap"] or 0.0) <= cut)
    elif layer == "all":
        codes = pit.stratified_sample(pool, target=target, seed=_seed(day))
    elif top_by_size:
        ranked = sorted(pool, key=lambda item: -(item["mcap"] or 0))[:top_by_size]
        codes = sorted(item["symbol"] for item in ranked)
    else:
        codes = sorted(item["symbol"] for item in pool)
    return codes


def _seed(day: str) -> int:
    return 20261021 + int(day.replace("-", "")) % 100000


def member_closure(
    layer: str,
    ctx: dict[str, Any],
    target: int,
    *,
    exclude_delisted: bool = False,
    use_current_members: bool = False,
    top_by_size: int | None = None,
) -> Any:
    """把"当日池子"包成 ``date -> set`` 回调，供 protocol 逐日过滤。

    逐日计算 + 缓存：同一天可能被 IC 计算、下单、等权基准三处各问一次，缓一下
    能把 9 次评估的池子构造从几万次降到几千次。
    """
    cache: dict[str, set[str]] = {}

    def fn(day: str) -> set[str]:
        if day not in cache:
            cache[day] = set(
                universe_codes(
                    day, layer=layer, ctx=ctx, target=target,
                    exclude_delisted=exclude_delisted,
                    use_current_members=use_current_members,
                    top_by_size=top_by_size,
                )
            )
        return cache[day]

    return fn


def build_histories(codes: Sequence[str], ctx: dict[str, Any]) -> list[dict[str, Any]]:
    bars, industry = ctx["bars"], ctx["industry"]
    histories = []
    for code in codes:
        rows = bars.get(code)
        if not rows:
            continue
        name = (industry.get(code) or {}).get("name") or code
        histories.append(to_history(code, rows, name))
    return histories


def coverage_at(day: str, codes: Sequence[str], ctx: dict[str, Any]) -> dict[str, Any]:
    snapshot = clean.panel_slice(ctx["bars"], day)
    pool = [
        {
            "symbol": code,
            "date": day,
            "mcap": snapshot[code]["mcap"] if code in snapshot else None,
            "industry": pit.industry_key(code, ctx["industry"]),
        }
        for code in codes
    ]
    return pit.coverage(pool, ctx["industry"])


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def run(
    version: str,
    *,
    target: int = 400,
    eval_start: str = EVAL_START,
    eval_end: str | None = None,
    iters: int = 800,
    layers: Sequence[str] = ("hs300", "zz500", "small"),
    root: Path | None = None,
    log=print,
) -> dict[str, Any]:
    ctx = load_context(version, root=root)
    if not ctx["verify"]["ok"]:
        raise MarketDataError(f"数据版本 {version} 校验失败：{ctx['verify']}")
    calendar = ctx["calendar"]
    eval_end = eval_end or calendar[-1]
    window = [day for day in calendar if eval_start <= day <= eval_end]
    ctx["alive"] = ctx["universe"].alive_at_end()
    book = ctx["book"]
    log(f"版本 {version}：{len(ctx['bars'])} 只日线，评估期 {window[0]} → {window[-1]}"
        f"（{len(window)} 个交易日）")

    benchmark = ctx["benchmark"]
    if benchmark is None:
        raise MarketDataError("缺少基准（benchmark.json），先跑 research.fetch_benchmark")
    bench_history = to_history(benchmark["symbol"], benchmark["rows"], benchmark.get("name", "沪深300"))

    sample_days = window[::21]          # 每月抽一个截面做池子统计
    observations = []
    for day in sample_days:
        row = {"date": day, "layers": {}}
        for layer in layers:
            codes = universe_codes(day, layer=layer, ctx=ctx, target=target)
            row["layers"][layer] = {
                "n": len(codes),
                "coverage": coverage_at(day, codes, ctx),
            }
        observations.append(row)
    log(f"池子截面统计完成：{len(observations)} 个截面")

    # ---------- 面板：一次建好，所有组共用 ----------
    # 面板 = 评估期内 PIT 成分（300 ∪ 500）的并集，且必须有日线。这样退市股在
    # 退市前照常参与截面，退市后自然消失——不靠"末期名单"决定谁能进样本。
    union = sorted({
        code for day in window
        for index in ("000300", "000905")
        for code in book.members(day, index)
    })
    available_union = [code for code in union if code in ctx["bars"]]
    hs300_union = sorted({code for day in window for code in book.members(day, "000300")})
    available_hs300 = [code for code in hs300_union if code in ctx["bars"]]
    log(f"PIT 并集 {len(union)} 只（有日线 {len(available_union)}）；"
        f"其中沪深300 并集 {len(hs300_union)} 只（有日线 {len(available_hs300)}）")

    panel = build_histories(available_union, ctx)
    if len(panel) < protocol.MIN_XSEC:
        raise MarketDataError(f"日线只有 {len(panel)} 只，先补齐 research.fetch_bars")
    bundle = protocol.build_records(panel, bench_history, horizon=protocol.HORIZON)
    bars_map = ex.build_bars(panel)
    flags_map = ex.tradability(panel)
    log(f"面板构建完成：{len(panel)} 只，记录 {len(bundle['records'])} 条")

    def ev(label: str, members: Any, **kwargs: Any) -> dict[str, Any]:
        return evaluate(
            histories=panel, benchmark=bench_history, label=label,
            eval_start=window[0], eval_end=window[-1], iters=iters,
            members=members, bundle=bundle, bars=bars_map, flags=flags_map, **kwargs,
        )

    # ---------- 四项对比 ----------
    comparisons: dict[str, Any] = {}

    def run_pair(name: str, left: dict[str, Any], right: dict[str, Any], note: str) -> None:
        left_ic = (left.get("ic") or {})
        right_ic = (right.get("ic") or {})
        comparisons[name] = {
            "note": note,
            "pit": left,
            "legacy": right,
            "ic_diff_pct": diff_pct(left_ic.get("mean"), right_ic.get("mean")),
            "nav_diff_pct": diff_pct((left.get("nav") or {}).get("total_return"),
                                     (right.get("nav") or {}).get("total_return")),
        }
        same_sign = (
            left_ic.get("mean") is not None and right_ic.get("mean") is not None
            and (left_ic["mean"] * right_ic["mean"] > 0)
        )
        comparisons[name]["same_sign"] = bool(same_sign)
        log(f"  {name}: IC {left_ic.get('mean')} vs {right_ic.get('mean')}，"
            f"净值 {left.get('nav', {}).get('total_return')} vs "
            f"{right.get('nav', {}).get('total_return')}")

    # A：PIT（逐日换名单）vs 当前成分（幸存者）
    current_members = set(book.members(book.days[-1], "000300"))
    log(f"A 组：当前成分 {len(current_members)} 只")
    run_pair(
        "A_current_vs_pit",
        ev("HS300 PIT（逐日成分）", member_closure("hs300", ctx, target)),
        ev("HS300 当前成分（幸存者）", lambda day: current_members),
        "同一策略、同一区间、同一面板：左边按日取真实成分，右边固定用今天的成分回看",
    )

    # B：含退市 vs 不含退市（同一逐日口径，只改"是否允许退市股留在样本里"）
    with_delisted = member_closure("hs300", ctx, target)
    without_delisted = member_closure("hs300", ctx, target, exclude_delisted=True)
    log(f"B 组：在市判定 {len(ctx['alive'])} 只（覆盖不足时为 0，报告里会标注未判定）")
    run_pair(
        "B_with_vs_without_delisted",
        ev("HS300 PIT（含退市）", with_delisted),
        ev("HS300 PIT（剔除已退市）", without_delisted),
        "左边保留后来退市/被并购的票（退市前照常参与），右边只留活到今天的",
    )

    # C：60 大盘 vs 300–400 分层（都按日重算，不用末期名单）
    top60 = member_closure("hs300", ctx, target, top_by_size=60)
    stratified = member_closure("all", ctx, target)
    log(f"C 组：分层目标 {target} 只 / 市值前 60")
    run_pair(
        "C_top60_vs_stratified",
        ev(f"分层抽样（目标 {target} 只）", stratified),
        ev("市值前 60", top60),
        "旧口径（市值前 60）对照按行业×市值三分位分层抽样，两侧都逐日重算",
    )

    # D：分层回测（三层同面板、同区间，只换当日名单）
    layer_results = {}
    for layer in layers:
        result = ev(LAYER_LABELS.get(layer, layer), member_closure(layer, ctx, target))
        layer_results[layer] = result
        if result.get("available"):
            log(f"  D {layer}: IC {round(result['ic']['mean'], 4)} / "
                f"净值 {result['nav']['total_return']}% / 横截面 {result.get('xsec_mean')} 只")
        else:
            log(f"  D {layer}: 跳过（{result.get('note')}）")

    coverage_summary = {}
    for layer in layers:
        rows = [item["layers"][layer]["coverage"] for item in observations if layer in item["layers"]]
        coverage_summary[layer] = {
            "n_mean": round(statistics.fmean([row["n"] for row in rows]), 1) if rows else 0,
            "industries_mean": round(statistics.fmean([row["industries"] for row in rows]), 1) if rows else 0,
            "top3_share_mean": round(statistics.fmean([row["top3_share"] for row in rows]), 1) if rows else 0,
            "hhi_mean": round(statistics.fmean([row["hhi"] for row in rows]), 3) if rows else 0,
            "mcap_p50_亿": round(statistics.median([row["mcap_p50_亿"] for row in rows if row["mcap_p50_亿"]]), 1)
            if any(row["mcap_p50_亿"] for row in rows) else None,
            "top_industries": rows[0]["top"] if rows else [],
        }

    breadth = {layer: result.get("breadth") for layer, result in layer_results.items()}
    criteria = _criteria(len(window), coverage_summary, breadthe=breadth,
                         comparisons=comparisons)
    survivorship = {
        index: pit.survivors_bias(book, index, alive_end=ctx["alive"])
        for index in ("000300", "000905")
    }
    fields = field_audit(ctx["universe"])
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "version": version,
        "verify": ctx["verify"],
        "eval": {"start": window[0], "end": window[-1], "days": len(window),
                 "years": round(len(window) / 243, 2)},
        "params": {"target": target, "horizon": protocol.HORIZON, "folds": protocol.K_FOLDS,
                   "rebalance": protocol.REBALANCE, "top": protocol.TOP_FRACTION,
                   "turnover": protocol.MAX_TURNOVER, "iters": iters},
        "survivorship": survivorship,
        "quality": clean.quality_report(ctx["bars"]),
        "fields": fields,
        "panel": {"union": len(union), "with_bars": len(available_union),
                  "hs300_union": len(hs300_union), "hs300_with_bars": len(available_hs300),
                  "records": len(bundle["records"])},
        "coverage": coverage_summary,
        "comparisons": comparisons,
        "layers": layer_results,
        "breadth": breadth,
        "criteria": criteria,
    }


def _criteria(
    days: int,
    coverage: dict[str, Any],
    *,
    breadthe: dict[str, Any],
    comparisons: dict[str, Any],
) -> list[dict[str, Any]]:
    """四条通过标准。

    IC 与净值必须分开判：**IC 方向不变、净值差几倍**是幸存者偏差的典型症状——
    策略的选股逻辑没变，变的是"池子里那些票本身赚了多少"。把两者合成一个数字，
    就会得出"差异 749% 所以通过"这种荒谬结论。
    """
    years = days / 243
    width = [item.get("effective_holdings") for item in breadthe.values() if item]
    top3 = [item["top3_share_mean"] for item in coverage.values() if item]
    # 只有 A（当前成分 vs PIT）和 B（含退市 vs 不含）是"口径之争"；
    # C（60 只 vs 300 只）比的是两种池子设计，差异本身就是结论，不该拿来判定
    # "旧结论是否可用"——否则池子越宽越容易被判成"不一致"。
    caliber = {name: item for name, item in comparisons.items()
               if name.startswith(("A_", "B_"))}
    ic_diffs = [abs(item["ic_diff_pct"]) for item in caliber.values()
                if item.get("ic_diff_pct") is not None]
    nav_diffs = [abs(item["nav_diff_pct"]) for item in caliber.values()
                 if item.get("nav_diff_pct") is not None]
    same_signs = [item["same_sign"] for item in caliber.values()
                  if item.get("same_sign") is not None]

    # 幸存者偏差的方向是可检验的：旧口径（当期成分回看）净值应系统性高于 PIT。
    survivor = comparisons.get("A_current_vs_pit") or {}
    legacy_nav = (survivor.get("legacy") or {}).get("nav", {}).get("total_return")
    pit_nav = (survivor.get("pit") or {}).get("nav", {}).get("total_return")
    inflated = (legacy_nav is not None and pit_nav is not None and legacy_nav > pit_nav)

    def status(value: Any, ok: bool) -> str:
        return "insufficient" if value is None else ("pass" if ok else "fail")

    ic_ok = bool(ic_diffs) and (all(same_signs) if same_signs else True) and max(ic_diffs) <= 30
    nav_close = bool(nav_diffs) and max(nav_diffs) <= 30
    return [
        {"name": "有效宽度 > 10", "value": max(width) if width else None,
         "pass": bool(width and max(width) > 10),
         "status": status(max(width) if width else None, bool(width and max(width) > 10))},
        {"name": "行业不集中（前三行业 ≤ 50%）", "value": max(top3) if top3 else None,
         "pass": bool(top3 and max(top3) <= 50),
         "status": status(max(top3) if top3 else None, bool(top3 and max(top3) <= 50))},
        {"name": "历史 ≥ 8 年", "value": round(years, 2), "pass": years >= 8,
         "status": "pass" if years >= 8 else "fail"},
        {"name": "PIT 与旧口径 IC 同向且差异 ≤ 30%", "value": max(ic_diffs) if ic_diffs else None,
         "pass": ic_ok, "status": status(max(ic_diffs) if ic_diffs else None, ic_ok)},
        {"name": "PIT 与旧口径净值差异 ≤ 30%（或方向可归因于幸存者偏差）",
         "value": max(nav_diffs) if nav_diffs else None,
         "pass": nav_close,
         "status": "insufficient" if not nav_diffs else ("pass" if nav_close else
                                                         ("explain" if (all(same_signs) and inflated) else "fail")),
         "note": "旧口径净值更高、IC 同向 → 差异来自幸存者偏差，旧净值不可引用" if inflated else ""},
    ]


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #

def _fmt(value: Any, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    return f"{value:+.{digits}f}{suffix}" if digits else f"{value}{suffix}"


def render_markdown(report: dict[str, Any]) -> str:
    criteria = report["criteria"]
    survivorship = report["survivorship"]
    bars = report["quality"]["symbols"]
    lines = [
        "# 股票池重建与池层验证",
        "",
        f"数据版本 `{report['version']}`（校验 {'通过' if report['verify']['ok'] else '失败'}），"
        f"评估期 {report['eval']['start']} → {report['eval']['end']}（{report['eval']['days']} 个交易日，"
        f"{report['eval']['years']} 年）。",
        "",
    ]
    if bars < 300:
        lines += [
            f"> **数据不足，本报告不作结论**：当前版本只有 {bars} 只标的的日线"
            f"（PIT 并集 {report['panel']['union']} 只），下面的 IC 与净值只用于验证管线可跑。"
            "补齐方法：`python -m research.fetch_bars --shard 0/1 --limit N` 多轮续跑后 "
            "`--reconcile` 收尾。",
            "",
        ]
    lines += [
        "## 1. 幸存者偏差有多大",
        "",
        "| 指数 | 首末快照 | 期初成分 | 期末成分 | 中途离开 | 占比 | 其中退市 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, item in survivorship.items():
        if not item:
            continue
        lines.append(
            f"| {index} | {item['start']} → {item['end']} | {item['start_size']} | "
            f"{item['end_size']} | {item['turnover_names']} | {item['turnover_share']}% | "
            f"{item['delisted_among_removed'] if item['delisted_among_removed'] is not None else '未判定'} |"
        )
    lines += [
        "",
        "用今天的成分回看历史，等于把中途被剔除/退市的票整段删掉——上面的『中途离开』"
        "就是被静默删掉的样本量。",
        "",
        "## 2. 数据体检",
        "",
        f"- 标的 {report['quality']['symbols']} 只、{report['quality']['rows']} 行，"
        f"区间 {report['quality']['window']}。",
        f"- 成交额缺失（已用 量×价 回填）：{report['quality']['amount_missing_pct']}%；"
        f"换手率缺失：{report['quality']['turnover_missing_pct']}%；"
        f"异常收益率被剔除：{report['quality']['invalid_return_pct']}%。",
        f"- 一字板日数 {report['quality']['one_word_days']}；"
        f"末期已无行情（退市/长停）{report['quality']['delisted_like']} 只；"
        f"历史不足 400 日 {report['quality']['short_history']} 只。",
        f"- 面板：PIT 并集 {report['panel']['union']} 只，其中有日线 "
        f"{report['panel']['with_bars']} 只（覆盖率 "
        f"{round(report['panel']['with_bars'] / max(report['panel']['union'], 1) * 100, 1)}%），"
        f"记录 {report['panel']['records']} 条。",
        "",
        "## 3. 字段体检（中性化与可交易性过滤的前提）",
        "",
        "| 日期 | 成分数 | 有行情 | 无行情（停牌/退市） | 行业已知 | 市值已知 | 上市天数已知 | ST 数 | 触板数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    fields = report.get("fields") or {}
    for row in (fields.get("rows") or [])[-6:]:
        lines.append(
            f"| {row['date']} | {row['members']} | {row['traded']} | {row['no_bar_or_suspended']} | "
            f"{row['industry_known_pct']}% | {row['mcap_known_pct']}% | "
            f"{row['listing_days_known_pct']}% | {row['st_count']} | {row['limit_count']} |"
        )
    if fields:
        lines += [
            "",
            f"样本均值：成分 {fields['members_mean']} 只 / 有行情 {fields['traded_mean']} 只；"
            f"行业已知 {fields['industry_known_pct']}%，自由流通市值已知 {fields['mcap_known_pct']}%，"
            f"上市天数已知 {fields['listing_days_known_pct']}%；"
            f"ST 平均 {fields['st_mean']} 只，触板平均 {fields['limit_mean']} 只。",
            "",
            "行业缺失按 `UNKNOWN` 落在一档、市值缺失不进分层——两者都会让「前三行业占比」"
            "与分层抽样失真，所以必须先看这张表再看下面的池子覆盖。",
            "",
        ]
    lines += [
        "## 4. 池子覆盖",
        "",
        "分层（small）的定义：PIT 成分并集里当日自由流通市值最低的 1/3。"
        "全市场小盘股需要 4000+ 只日线，免费源拉不完，故用成分内下三分之一近似，"
        "报告里的「小盘层」一律按此理解。",
        "",
        "| 层 | 平均标的数 | 行业数 | 前三行业占比 | HHI | 市值中位数(亿) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for layer, item in report["coverage"].items():
        lines.append(
            f"| {LAYER_LABELS.get(layer, layer)} | {item['n_mean']} | {item['industries_mean']} | "
            f"{item['top3_share_mean']}% | {item['hhi_mean']} | {item['mcap_p50_亿']} |"
        )
    lines += ["", "## 5. 四组对比", "",
              "| 对比 | PIT / 新口径 IC | 旧口径 IC | IC 差异 | 新净值 | 旧净值 | 净值差异 | 同向 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for name, item in report["comparisons"].items():
        left, right = item["pit"], item["legacy"]
        lines.append(
            f"| {name} | {_fmt((left.get('ic') or {}).get('mean'), 4)} | "
            f"{_fmt((right.get('ic') or {}).get('mean'), 4)} | "
            f"{_fmt(item['ic_diff_pct'], 1, '%')} | "
            f"{_fmt((left.get('nav') or {}).get('total_return'), 2, '%')} | "
            f"{_fmt((right.get('nav') or {}).get('total_return'), 2, '%')} | "
            f"{_fmt(item['nav_diff_pct'], 1, '%')} | {'是' if item['same_sign'] else '否'} |"
        )
    lines += ["", "## 6. 分层回测", "",
              "| 层 | IC | 95% 区间 | 横截面均只数 | 净净值 | 等权基准 | 超额 | 最大回撤 | 有效宽度 | 平均持仓 |",
              "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for layer, item in report["layers"].items():
        ic = item.get("ic") or {}
        nav = item.get("nav") or {}
        lines.append(
            f"| {LAYER_LABELS.get(layer, layer)} | {_fmt(ic.get('mean'), 4)} | "
            f"[{_fmt(ic.get('ci_low'), 4)}, {_fmt(ic.get('ci_high'), 4)}] | "
            f"{item.get('xsec_mean')} | "
            f"{_fmt(nav.get('total_return'), 2, '%')} | "
            f"{_fmt((item.get('benchmark') or {}).get('total_return'), 2, '%')} | "
            f"{_fmt(item.get('excess'), 2, '%')} | {_fmt(nav.get('max_drawdown'), 2, '%')} | "
            f"{(item.get('breadth') or {}).get('effective_holdings')} | {item.get('holdings_mean')} |"
        )
    lines += ["", "## 7. 通过标准", "", "| 标准 | 实测 | 判定 |", "| --- | ---: | --- |"]
    labels = {"pass": "通过", "fail": "未通过", "explain": "可解释", "insufficient": "数据不足"}
    for item in criteria:
        lines.append(f"| {item['name']} | {item['value']} | {labels.get(item.get('status'), '未通过')} |")
    lines += [
        "",
        "## 8. 结论",
        "",
    ]
    insufficient = [item["name"] for item in criteria if item.get("status") == "insufficient"]
    failed = [item["name"] for item in criteria if item.get("status") == "fail"]
    explained = [item for item in criteria if item.get("status") == "explain"]
    if insufficient:
        lines.append(
            "数据不足，暂不判定：" + "、".join(insufficient) + "。"
            "这几项需要日线齐全后才能算，当前只完成了成分层面的审计；"
            "在补齐数据之前不要引用池层验证的 IC / 净值结论。"
        )
    elif failed:
        lines.append(
            "未通过：" + "、".join(failed) + "。"
            "差异超过 30% 的对比说明旧结论不可用，后续研究只引用新口径。"
        )
    else:
        lines.append("各项标准全部通过：可以按新口径继续研究，旧口径结论仅作历史对照。")
    for item in explained:
        lines += [
            "",
            f"**净值差异 {item['value']}% 判定为「可解释」**：" + (item.get("note") or "")
            + "。IC 同向说明选股逻辑没变，净值差的是池子里那些票本身的收益——"
            "这正是幸存者偏差的作用方式，所以旧口径的净值（以及基于它的收益率结论）一律不可引用，"
            "但不必因此否定策略的方向。",
        ]
    return "\n".join(lines)


def render_preview(report: dict[str, Any]) -> str:
    body = render_markdown(report)
    rows = []
    for line in body.splitlines():
        if line.startswith("# "):
            rows.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            rows.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("| ---"):
            continue
        elif line.startswith("| "):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
        elif line.strip() == "":
            rows.append("")
        else:
            rows.append(f"<p>{line}</p>")
    html: list[str] = []
    in_table = False
    for item in rows:
        if item.startswith("<tr>") and not in_table:
            html.append("<table>")
            in_table = True
        elif not item.startswith("<tr>") and in_table:
            html.append("</table>")
            in_table = False
        html.append(item)
    if in_table:
        html.append("</table>")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>股票池重建与池层验证</title>
<style>
  body {{ margin:0; background:#f4f2eb; color:#16221d;
    font:15px/1.7 -apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; }}
  .wrap {{ max-width:1100px; margin:0 auto; padding:34px 22px 60px; }}
  .panel {{ background:#fffdf7; border:1px solid #d8ddd6; border-radius:10px; padding:26px;
    box-shadow:0 18px 50px rgba(31,48,40,.08); }}
  h1 {{ font-size:22px; margin:0 0 16px; }}
  h2 {{ font-size:15px; margin:26px 0 8px; color:#176b4d; }}
  table {{ width:100%; border-collapse:collapse; font-size:12.5px; margin:8px 0 4px; }}
  td {{ padding:6px 9px; border-bottom:1px solid #e6e9e2; white-space:nowrap; }}
  tr:first-child td {{ color:#6e7b74; font-weight:600; background:rgba(23,107,77,.05); }}
  p {{ margin:6px 0; color:#3d4a44; }}
</style></head>
<body><div class="wrap"><div class="panel">{''.join(html)}</div></div></body></html>"""


def write_reports(report: dict[str, Any], outdir: Path = REPORT_DIR) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": outdir / "pool.json",
        "markdown": outdir / "pool.md",
        "preview": outdir / "pool_preview.html",
    }
    paths["json"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["markdown"].write_text(render_markdown(report), encoding="utf-8")
    paths["preview"].write_text(render_preview(report), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="股票池池层验证（PIT / 退市 / 分层 / 三指数）")
    parser.add_argument("--version", default=None, help="数据版本，缺省用最新")
    parser.add_argument("--target", type=int, default=300, help="分层抽样目标只数")
    parser.add_argument("--eval-start", default=EVAL_START)
    parser.add_argument("--eval-end", default=None)
    parser.add_argument("--iters", type=int, default=800)
    parser.add_argument("--layers", default="hs300,zz500,small")
    parser.add_argument("--outdir", default=str(REPORT_DIR))
    args = parser.parse_args(argv)
    version = args.version or latest_version()
    if not version:
        print("没有可用的数据版本，先跑 python -m research.fetch", file=sys.stderr)
        return 2
    try:
        report = run(
            version,
            target=max(20, args.target),
            eval_start=args.eval_start,
            eval_end=args.eval_end,
            iters=max(200, args.iters),
            layers=[item for item in args.layers.split(",") if item],
        )
    except (MarketDataError, FileNotFoundError) as exc:
        print(f"验证失败：{exc}", file=sys.stderr)
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
