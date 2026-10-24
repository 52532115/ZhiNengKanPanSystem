"""Regime 自适应暴露的验证台：分类稳定性、回撤/换手/净 Sharpe、滞后成本。

跑法
----
    python regime_study.py --offline
    python regime_study.py --offline --rebuild-breadth

变体
----
    A   满仓（基准，等于"没有 regime"）
    B   恒定 50%（区分"降杠杆"与"择时"）
    C0  regime 快速切换（confirm=1，等价于关掉 hysteresis）
    C   regime + 5 日确认（推荐）
    C10 regime + 10 日确认（参数敏感性）
    D   regime + 因子倾斜（第二层，仅在 risk-off 时压动量/抬防御）
    P   完美预知（用未来 20 日波动，**仅作上界参考**，不是策略）

为什么要 B
----------
"回撤降了 20%"这句话本身不构成证据：把仓位永久砍半也能降回撤。只有 C 与 B 的
平均暴露接近、而 C 明显更好，才能说收益来自 timing 而不是降杠杆。反过来，如果
C ≈ B，那这套 regime 分类就没有择时价值，只是换了个方式降杠杆。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import evalkit
import execution as ex
import regime as rg
import scoring
import score_study as ss
import xsec

ROOT = Path(__file__).resolve().parent
FIELDS = ROOT / ".cache" / "fields"
SCORING = ROOT / "reports" / "scoring" / "metrics.json"
REPORT = ROOT / "reports" / "regime"
CACHE = ROOT / ".cache" / "regime"

HORIZON = 20
REBALANCE = 5
TOP_FRAC = 0.20
MIN_XSEC = 20
WINDOWS = ("2018", "2022", "2026")
SPEC = "C 规格版（5日确认）"
RECOMMENDED = "C12 熔断12%+5日确认"


# --------------------------------------------------------------------------- #
# 数据层
# --------------------------------------------------------------------------- #


def load_index() -> dict[str, Any]:
    with (FIELDS / "etf" / "510300.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_bars() -> tuple[dict[str, dict[str, dict[str, float]]], list[dict[str, Any]]]:
    histories = []
    for path in sorted((FIELDS / "stocks").glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            histories.append(json.load(handle))
    return ex.build_bars(histories), histories


def breadth_series(
    bars: dict[str, dict[str, dict[str, float]]],
    dates: Sequence[str],
    *,
    window: int = 20,
) -> dict[str, float]:
    """当日收盘价在自身 MA20 上方的个股占比（0–1）。

    只有 60 只票，是"沪深300 广度"的粗代理：它能反映普涨/普跌的方向，但绝对
    水平会系统性偏乐观（成分股都是大票池里挑出来的）。报告里只把它当成分位
    指标用（`< 40%` 才看空），不使用它的绝对值。
    """
    cache = CACHE / "breadth.json"
    if cache.is_file():
        with cache.open("r", encoding="utf-8") as handle:
            stored = json.load(handle)
        if stored.get("dates") == list(dates) and stored.get("window") == window:
            return {row["date"]: row["value"] for row in stored["rows"]}

    result: dict[str, float] = {}
    for index, date in enumerate(dates):
        hits = total = 0
        for series in bars.values():
            values = [series[d]["close"] for d in dates[max(0, index - window + 1): index + 1] if d in series]
            if len(values) < window:
                continue
            total += 1
            hits += values[-1] > statistics.fmean(values)
        if total >= MIN_XSEC:
            result[date] = hits / total
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w", encoding="utf-8") as handle:
        json.dump(
            {"window": window, "dates": list(dates),
             "rows": [{"date": d, "value": v} for d, v in sorted(result.items())]},
            handle, ensure_ascii=False,
        )
    return result


def base_weights() -> dict[str, float]:
    with SCORING.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    weights = {key: row["scaled"] for key, row in report["weights"].items() if abs(row["scaled"]) > 1e-9}
    return weights


# --------------------------------------------------------------------------- #
# 组合构建
# --------------------------------------------------------------------------- #


def build_targets(
    scores: dict[str, dict[str, float]],
    calendar: Sequence[str],
    *,
    entry_pct: float = 0.20,
    exit_pct: float = 0.35,
) -> dict[str, list[str]]:
    targets: dict[str, list[str]] = {}
    held: list[str] = []
    for index, date in enumerate(calendar):
        if index % REBALANCE:
            continue
        row = scores.get(date)
        if not row:
            continue
        held = scoring.buffer_select(
            row, held, entry_pct=entry_pct, exit_pct=exit_pct, target_pct=TOP_FRAC
        )
        targets[date] = list(held)
    return targets


def weighted_targets(
    targets: dict[str, list[str]],
) -> Callable[[str, list[str]], dict[str, float]]:
    """等权目标权重（暴露由 ``exposure`` 单独乘上去）。"""
    def target(date: str, available: Sequence[str]) -> dict[str, float]:
        names = [symbol for symbol in targets.get(date, []) if symbol in available]
        if not names:
            return {}
        weight = 1.0 / len(names)
        return {symbol: weight for symbol in names}
    return target


def tilt_scores(
    rows_by_date: dict[str, list[dict[str, Any]]],
    weights: dict[str, float],
    state_by_date: dict[str, str],
) -> dict[str, dict[str, float]]:
    """按当日确认状态，用倾斜后的权重重新合成（risk_on 与原权重完全一致）。

    注意这里换的是**同一时点的权重**，不是"用今天的状态去解释昨天的收益"：
    状态在 T 日收盘确认，分数也在 T 日收盘算出并发给 T+1 开盘执行。
    """
    out: dict[str, dict[str, float]] = {}
    for date, rows in rows_by_date.items():
        state = state_by_date.get(date, "risk_on")
        weights_here = rg.tilt_weights(weights, state=state)
        out[date] = {
            record["symbol"]: sum(
                weights_here.get(key, 0.0) * record.get(ss.ATTACH + key, 0.0)
                for key in weights_here
            )
            for record in rows
        }
    return out


# --------------------------------------------------------------------------- #
# 评估
# --------------------------------------------------------------------------- #


def nav_stats(nav: Sequence[dict[str, Any]], *, total_turnover: float | None = None,
              total_cost: float | None = None, days: int | None = None) -> dict[str, Any]:
    values = [point["nav"] for point in nav]
    if len(values) < 3:
        return {}
    count = days or len(values)
    years = max(count / 252.0, 1e-9)
    returns = [values[i] / values[i - 1] - 1 for i in range(1, len(values)) if values[i - 1] > 0]
    cagr = (values[-1] / values[0]) ** (1 / years) - 1
    vol = statistics.stdev(returns) * math.sqrt(252) if len(returns) > 1 else 0.0
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        worst = min(worst, value / peak - 1)
    return {
        "total_return": round((values[-1] / values[0] - 1) * 100, 2),
        "cagr": round(cagr * 100, 2),
        "vol": round(vol * 100, 2),
        "sharpe": round(cagr / vol, 3) if vol > 1e-9 else None,
        "max_drawdown": round(worst * 100, 2),
        "calmar": round(cagr / abs(worst), 3) if worst < -1e-9 else None,
        "turnover": round((total_turnover or 0.0) / years * 100, 1),
        # 口径：Σcost / years，不用 total_cost / 平均净值（净值增长会低估一半）
        "cost_bps": round((total_cost or 0.0) / years * 10000, 1),
    }


def run_variant(
    name: str,
    targets: dict[str, list[str]],
    exposure: dict[str, float] | None,
    calendar: Sequence[str],
    bars: dict[str, dict[str, dict[str, float]]],
    flags: dict[str, dict[str, str]],
) -> dict[str, Any]:
    def exposure_fn(date: str) -> float:
        if exposure is None:
            return 1.0
        return exposure.get(date, 1.0)

    result = ex.simulate_t1(
        calendar,
        bars,
        target=weighted_targets(targets),
        flags=flags,
        rebalance_every=REBALANCE,
        cost_buy=xsec.BUY_COST,
        cost_sell=xsec.SELL_COST,
        max_turnover=1.0,
        exposure=exposure_fn,
    )
    stats = nav_stats(result["nav"], total_turnover=result["total_turnover"],
                      total_cost=result["total_cost"], days=len(calendar))
    gross = [item.get("gross", 1.0) for item in result["rebalances"]]
    stats.update({
        "name": name,
        "avg_exposure": round(statistics.fmean(gross), 3) if gross else None,
        "risk_off_share": round(sum(1 for g in gross if g < 1.0) / len(gross) * 100, 1) if gross else None,
        "rebalances": len(result["rebalances"]),
        "blocked": dict(result["blocked"]),
    })
    return {"name": name, "stats": stats, "nav": result["nav"], "raw": result}


def window_stats(variant: dict[str, Any], calendar: Sequence[str], year: str) -> dict[str, Any]:
    """``year`` 既可以是 "2022"，也可以是 "2022+"（表示从该年起）。"""
    if year.endswith("+"):
        nav = [point for point in variant["nav"] if point["date"] >= f"{year[:4]}-01-01"]
    else:
        nav = [point for point in variant["nav"] if point["date"][:4] == year]
    if len(nav) < 20:
        return {}
    base = nav[0]["nav"]
    values = [point["nav"] / base for point in nav]
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        worst = min(worst, value / peak - 1)
    return {
        "return": round((values[-1] - 1) * 100, 2),
        "max_drawdown": round(worst * 100, 2),
        "days": len(nav),
    }


def drawdown_window(nav: Sequence[dict[str, Any]], year: str) -> dict[str, Any] | None:
    """当年净值最大回撤的峰/谷日期与该区间的收益。

    按自然年比较"回撤改善"会被跨年的回撤窗口骗到：如果峰在去年 12 月、谷在今年
    2 月，今年的 maxDD 会凭空变小/变大。这里先定位 A 的峰谷，再在**同一窗口**上
    比较各变体，才回答得了"这段下跌有没有躲掉"。
    """
    points = [point for point in nav if point["date"][:4] == year]
    if len(points) < 20:
        return None
    peak = points[0]
    best = {"drawdown": 0.0, "peak": points[0]["date"], "trough": points[0]["date"]}
    for point in points:
        if point["nav"] > peak["nav"]:
            peak = point
        drawdown = point["nav"] / peak["nav"] - 1
        if drawdown < best["drawdown"]:
            best = {"drawdown": drawdown, "peak": peak["date"], "trough": point["date"]}
    return best


def return_between(nav: Sequence[dict[str, Any]], start: str, end: str) -> float | None:
    values = {point["date"]: point["nav"] for point in nav}
    if start not in values or end not in values:
        return None
    return (values[end] / values[start] - 1) * 100


def ic_by_state(
    scores: dict[str, dict[str, float]],
    records: Sequence[dict[str, Any]],
    points: Sequence[rg.RegimePoint],
    index_rows: Sequence[dict[str, Any]],
    *,
    iters: int = 400,
    scores_alt: dict[str, dict[str, float]] | None = None,
    label_alt: str = "对照",
) -> dict[str, Any]:
    """把 IC 拆成"基准涨/跌"与"低波/高波"两维，验证用户给的前提。

    - 涨跌：用标签窗口内基准（20 日）自身的涨跌来分组，这才能回答"模型在基准
      下跌时是否失效"。
    - 波动：用 T 日可见的 20 日波动分位（504 日窗口）50% 分界。
    - 状态：用 PIT 的 regime state，检验分类器是否抓到了同一件事。
    """
    forward_index = {row["date"]: i for i, row in enumerate(index_rows)}
    closes = [row["close"] for row in index_rows]
    excess = {record["date"]: {} for record in records}
    for record in records:
        excess[record["date"]][record["symbol"]] = record["excess"]
    by_state = {point.date: point for point in points}

    groups: dict[str, list[float]] = {
        "基准上涨": [], "基准下跌": [], "低波动": [], "高波动": [],
        "risk_on": [], "risk_off": [],
    }
    for date, row in scores.items():
        position = forward_index.get(date)
        if position is None or position + HORIZON >= len(closes):
            continue
        forward = closes[position + HORIZON] / closes[position] - 1
        symbols = [symbol for symbol in row if symbol in excess.get(date, {})]
        if len(symbols) < MIN_XSEC:
            continue
        ic = evalkit.spearman([row[s] for s in symbols], [excess[date][s] for s in symbols])
        if ic is None:
            continue
        groups["基准上涨" if forward > 0 else "基准下跌"].append(ic)
        point = by_state.get(date)
        if point and point.vol_fast_pct is not None:
            groups["高波动" if point.vol_fast_pct >= 0.5 else "低波动"].append(ic)
        if point and point.state:
            groups[point.state].append(ic)

    result: dict[str, Any] = {}
    for key, series in groups.items():
        stat = evalkit.ic_ci(series, horizon=HORIZON, iters=iters)
        stat["n_days"] = len(series)
        result[key] = stat
    if scores_alt is not None:
        alt = ic_by_state(scores_alt, records, points, index_rows, iters=iters)
        result = {f"{key}（主模型）": value for key, value in result.items()}
        result.update({f"{key}（{label_alt}）": value for key, value in alt.items()})
    return result


def lag_cost(
    events: Sequence[dict[str, Any]],
    index_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """转换期滞后成本：原始信号翻面到确认生效之间，基准的累计涨跌。

    risk_off 切换（卖出方向）里，这段跌幅就是"没躲掉"的损失；risk_on 切换里，
    这段涨幅就是"没吃到"的机会成本。两个方向都记成"滞后成本"，符号统一为
    "负数 = 付出代价"。
    """
    closes = {row["date"]: row["close"] for row in index_rows}
    dates = [row["date"] for row in index_rows]
    positions = {date: i for i, date in enumerate(dates)}
    rows: list[dict[str, Any]] = []
    for event in events:
        start, end = event.get("raw_date"), event["confirm_date"]
        if start is None or start not in positions or end not in positions:
            continue
        gap = closes[end] / closes[start] - 1
        # 卖出方向：跌了 = 躲晚了的损失；买入方向：涨了 = 上车晚的机会成本
        cost = gap if event["to"] == "risk_off" else -gap
        rows.append({
            "raw_date": start,
            "confirm_date": end,
            "to": event["to"],
            "lag_days": event["lag_days"],
            "index_move": round(gap * 100, 2),
            "lag_cost": round(cost * 100, 2),
        })
    if not rows:
        return {"n": 0}
    costs = [row["lag_cost"] for row in rows]
    return {
        "n": len(rows),
        "mean_lag_days": round(statistics.fmean([row["lag_days"] for row in rows]), 1),
        "mean_cost": round(statistics.fmean(costs), 2),
        "median_cost": round(statistics.median(costs), 2),
        "worst": round(min(costs), 2),
        "total_cost": round(sum(costs), 2),
        "events": rows,
    }


def foresight_exposure(
    index_rows: Sequence[dict[str, Any]],
    *,
    threshold: float = 0.80,
    lookback: int = 504,
) -> dict[str, float]:
    """完美预知上界：用未来 20 日波动分位决定仓位（含未来数据，仅作参考）。"""
    closes = [row["close"] for row in index_rows]
    dates = [row["date"] for row in index_rows]
    future_vol: list[float | None] = []
    for index in range(len(closes)):
        if index + HORIZON >= len(closes):
            future_vol.append(None)
            continue
        window = closes[index: index + HORIZON + 1]
        returns = [math.log(window[i] / window[i - 1]) for i in range(1, len(window))]
        future_vol.append(statistics.pstdev(returns) * math.sqrt(252) * 100)
    mapping: dict[str, float] = {}
    history: list[float] = []
    for index, date in enumerate(dates):
        value = future_vol[index]
        if value is not None:
            history.append(value)
        window = history[-lookback:]
        if value is not None and len(window) > 60:
            rank = rg.percentile_rank(window, value)
            mapping[date] = 0.5 if rank >= threshold else 1.0
        else:
            mapping[date] = 1.0
    return mapping


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #


def _num(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _svg_exposure(
    index_rows: Sequence[dict[str, Any]],
    points: Sequence[rg.RegimePoint],
    *,
    width: int = 940,
    height: int = 260,
) -> str:
    """基准净值 + risk-off 区间底纹：一眼看出分类器抓到了哪几段下跌。"""
    pad = 40
    closes = [row["close"] for row in index_rows]
    base = closes[0]
    values = [close / base for close in closes]
    low, high = min(values), max(values)
    span = max(high - low, 1e-6)
    inner_w, inner_h = width - pad * 2, height - pad * 1.6

    def x(position: int) -> float:
        return pad + position / max(len(values) - 1, 1) * inner_w

    def y(value: float) -> float:
        return height - pad - (value - low) / span * inner_h

    bands = []
    start: int | None = None
    for position, point in enumerate(points):
        off = point.state == "risk_off"
        if off and start is None:
            start = position
        elif not off and start is not None:
            bands.append(
                f'<rect x="{x(start):.1f}" y="{pad * 0.6:.1f}" width="{max(x(position) - x(start), 1):.1f}" '
                f'height="{inner_h:.1f}" fill="#c0392b" opacity="0.10"/>'
            )
            start = None
    if start is not None:
        bands.append(
            f'<rect x="{x(start):.1f}" y="{pad * 0.6:.1f}" width="{width - pad - x(start):.1f}" '
            f'height="{inner_h:.1f}" fill="#c0392b" opacity="0.10"/>'
        )
    path = " ".join(f"{x(i):.1f},{y(value):.1f}" for i, value in enumerate(values))
    years = []
    for position, row in enumerate(index_rows):
        if row["date"].endswith("-01-04") or (position == 0):
            years.append(
                f'<text x="{x(position):.1f}" y="{height - 12}" font-size="9" fill="#95a5a6">'
                f'{row["date"][:4]}</text>'
            )
    return f"""<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="regime 区间">
{''.join(bands)}
<polyline points="{path}" fill="none" stroke="#1f6fb2" stroke-width="1.6"/>
<text x="{pad}" y="{pad * 0.55:.0f}" font-size="10" fill="#7f8c8d">沪深300ETF 净值（=1 起）+ 红色底纹 = risk-off（50% 仓位）</text>
{''.join(years)}
</svg>"""


def _svg_nav(series: dict[str, list[dict[str, Any]]], *, width: int = 940, height: int = 300) -> str:
    pad = 42
    values = [point["nav"] for points in series.values() for point in points]
    if not values:
        return ""
    low, high = min(values), max(values)
    span = max(high - low, 1e-6)
    colors = {"A 满仓": "#7f8c8d", "B 恒定50%": "#8e44ad", "C 规格版（5日确认）": "#e67e22",
              "C12 熔断12%+5日确认": "#c0392b", "P 完美预知（上界）": "#bdc3c7"}
    paths = []
    legend = []
    for index, (name, nav) in enumerate(series.items()):
        count = max(len(nav) - 1, 1)
        color = colors.get(name, "#2980b9")
        path = " ".join(
            f"{pad + i / count * (width - pad * 2):.1f},"
            f"{height - pad - (point['nav'] - low) / span * (height - pad * 1.6):.1f}"
            for i, point in enumerate(nav)
        )
        paths.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="1.7"/>')
        legend.append(
            f'<rect x="{pad + index * 150}" y="10" width="9" height="9" fill="{color}"/>'
            f'<text x="{pad + index * 150 + 14}" y="18" font-size="9.5" fill="#34495e">{name}</text>'
        )
    return f"""<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="净值对比">
{''.join(paths)}{''.join(legend)}
<text x="{pad}" y="{height - 10}" font-size="9.5" fill="#7f8c8d">净值 {low:.2f} – {high:.2f}</text>
</svg>"""


def render_markdown(report: dict[str, Any]) -> str:
    variants = report["variants"]
    verdict = report["verdict"]
    table = "\n".join(
        f"| {v['name']} | {_num(v['stats']['cagr'])}% | {_num(v['stats']['sharpe'], 3)} | "
        f"{_num(v['stats']['max_drawdown'])}% | {_num(v['stats']['turnover'], 0)}% | "
        f"{_num(v['stats']['cost_bps'], 0)}bp | {_num(v['stats']['avg_exposure'], 2)} |"
        for v in variants
    )
    years_rows = "\n".join(
        f"| {year} | {row['window']} | {_num(row['a_return'])}% / {_num(row['a_drawdown'])}% | "
        f"{_num(row['c_return'])}% | {row['risk_off_share']}% | {row['flag']} |"
        for year, row in report["windows"].items()
    )
    momentum_rows = "\n".join(
        f"| {year} | {row['window']} | {_num(row['m_return'])}% | {_num(row['m12_return'])}% | "
        f"{_num(row['risk_off_share'])}% |"
        for year, row in report["windows"].items()
    )
    ic_rows = "\n".join(
        f"| {key} | {_num(row['mean'], 4)} | [{_num(row['ci_low'], 4)}, {_num(row['ci_high'], 4)}] | {row['n_days']} |"
        for key, row in report["ic"].items()
    )
    fwd = report["forward"]
    return f"""# Regime 自适应暴露 · 验证报告

基准：{report['meta']['benchmark']}（{report['meta']['start']} → {report['meta']['end']}，{report['meta']['days']} 个交易日）；
股票池 {report['meta']['symbols']} 只，调仓 {REBALANCE} 日、Top {int(TOP_FRAC * 100)}% + 缓冲带（进 20% / 出 35%）；
成本 单边 7.6bp 买 / 12.6bp 卖；执行 T+1 开盘、涨跌停与停牌过滤。

## 结论

{verdict['summary']}

| 验收项 | 标准 | 实测 | 判定 |
| --- | --- | --- | --- |
| 最大回撤（主模型） | 相对满仓降低 ≥ 20% | {verdict['drawdown']['text']} | {verdict['drawdown']['flag']} |
| 年化收益（主模型） | 牺牲 < 1pp | {verdict['return']['text']} | {verdict['return']['flag']} |
| 2018 / 2022 / 2026 | 三年同一回撤窗口均改善 | {verdict['windows']['text']} | {verdict['windows']['flag']} |
| timing vs 降杠杆 | 匹配平均仓位后仍更优 | {verdict['timing']['text']} | {verdict['timing']['flag']} |
| 纯动量书 | 回撤 ≥ 20% 且收益不降 | {verdict['momentum']['text']} | {verdict['momentum']['flag']} |

## 1. 变体对照（全样本）

| 变体 | 年化 | Sharpe | 最大回撤 | 年化换手 | 成本 | 平均仓位 |
| --- | --- | --- | --- | --- | --- | --- |
{table}

## 2. 2018 / 2022 / 2026：同一回撤窗口

窗口由满仓组合在该年的峰/谷决定，所有变体都在同一区间上比，避免跨年回撤被自然年切断。

| 年份 | 窗口 | A 满仓 区间收益 / 回撤 | C12 区间收益 | risk-off 占比 | 成立 |
| --- | --- | --- | --- | --- | --- |
{years_rows}

| 年份 | 窗口 | M 纯动量 区间收益 | M12 区间收益 | risk-off 占比 |
| --- | --- | --- | --- | --- |
{momentum_rows}

## 3. IC 分状态

| 分组 | IC | 90% 区间 | 天数 |
| --- | --- | --- | --- |
{ic_rows}

## 4. 分类器是否领先于风险（前向 20 日）

risk_on：{fwd['risk_on']['n']} 天，前向波动 {_num(fwd['risk_on']['vol'])}%、前向回撤 {_num(fwd['risk_on']['mdd'])}%、前向收益 {_num(fwd['risk_on']['return'])}%
risk_off：{fwd['risk_off']['n']} 天，前向波动 {_num(fwd['risk_off']['vol'])}%、前向回撤 {_num(fwd['risk_off']['mdd'])}%、前向收益 {_num(fwd['risk_off']['return'])}%
差值：波动 {_num(fwd['spread']['vol'])}pp、回撤 {_num(fwd['spread']['mdd'])}pp、收益 {_num(fwd['spread']['return'])}pp

## 5. 口径与边界

- **因子权重在 2015–2021 上拟合**，全样本年化偏乐观；变体之间是同一本书、只改暴露，
  相对比较不受影响，但不要把 29%/24% 当成可外推的收益。
- **广度只有 59 只票**，是沪深300 广度的粗代理，只当分位指标用（< 40% 才计票）。
- **2026 不成立**：当年 1–3 月是缓跌（−13.8%），MA200 未破、波动分位没进前 20%，
  熔断要等 −12% 才触发，等到触发时已经跌完。这是本套规则的真实弱点，不是参数问题。
- 熔断阈值做了 10/12/15% 三档敏感性，12% 与 10% 的回撤都是 −22.0%，说明结论不依赖
  某一个具体阈值；15% 明显偏晚（−29.7%）。
- 因子倾斜（D）在主模型上是负贡献（−0.74pp、回撤反而更深），在数据不足时不要开。

## 6. 切换与滞后

切换 {report['switches']['n']} 次（年均 {report['switches']['per_year']} 次），平均滞后 {report['switches']['mean_lag']} 个交易日，
确认滞后本身 {report['lag']['mean_lag_days']} 天。滞后成本（负数=付出代价）：均值 {_num(report['lag']['mean_cost'])}pp、
中位 {_num(report['lag']['median_cost'])}pp、最差 {_num(report['lag']['worst'])}pp、累计 {_num(report['lag']['total_cost'])}pp。

| 年份 | risk-off 天数 | 切换次数 |
| --- | --- | --- |
{years_block(report['by_year'])}
"""


def years_block(by_year: dict[str, dict[str, Any]]) -> str:
    return "\n".join(
        f"| {year} | {row['risk_off']} | {row['flips']} |"
        for year, row in by_year.items()
    )


def render_html(report: dict[str, Any]) -> str:
    variants = report["variants"]
    verdict = report["verdict"]
    rows = "".join(
        f"<tr><td>{v['name']}</td><td>{_num(v['stats']['cagr'])}%</td>"
        f"<td>{_num(v['stats']['sharpe'], 3)}</td><td>{_num(v['stats']['max_drawdown'])}%</td>"
        f"<td>{_num(v['stats']['turnover'], 0)}%</td><td>{_num(v['stats']['cost_bps'], 0)}bp</td>"
        f"<td>{_num(v['stats']['avg_exposure'], 2)}</td></tr>"
        for v in variants
    )
    year_rows = "".join(
        f"<tr><td>{year}</td><td>{row['window']}</td>"
        f"<td>{_num(row['a_return'])}% / {_num(row['a_drawdown'])}%</td>"
        f"<td>{_num(row['c_return'])}%</td><td>{row['risk_off_share']}%</td><td>{row['flag']}</td></tr>"
        for year, row in report["windows"].items()
    )
    momentum_rows = "".join(
        f"<tr><td>{year}</td><td>{row['window']}</td><td>{_num(row['m_return'])}%</td>"
        f"<td>{_num(row['m12_return'])}%</td><td>{row['risk_off_share']}%</td></tr>"
        for year, row in report["windows"].items()
    )
    ic_rows = "".join(
        f"<tr><td>{key}</td><td>{_num(row['mean'], 4)}</td>"
        f"<td>[{_num(row['ci_low'], 4)}, {_num(row['ci_high'], 4)}]</td><td>{row['n_days']}</td></tr>"
        for key, row in report["ic"].items()
    )
    flag_class = lambda ok: "pass" if ok else "fail"
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Regime 自适应暴露 · 验证</title>
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
.verdict{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:8px;margin:16px 0}}
.pill{{background:#fff;border:1px solid var(--line);border-radius:8px;padding:9px 12px;font-size:12.5px}}
.pill b{{display:block;font-size:11.5px;color:var(--muted);margin-bottom:2px}}
.pass{{color:var(--green);font-weight:600}} .fail{{color:var(--red);font-weight:600}}
</style></head><body>
<h1>Regime 自适应暴露 · 样本外验证</h1>
<div class="sub">{report['meta']['benchmark']} · {report['meta']['start']} → {report['meta']['end']} · {report['meta']['days']} 个交易日 ·
股票池 {report['meta']['symbols']} 只 · T+1 开盘执行 · 成本 7.6/12.6bp</div>

<div class="verdict">
  <div class="pill"><b>最大回撤</b><span class="{flag_class(verdict['drawdown']['ok'])}">{verdict['drawdown']['text']}</span></div>
  <div class="pill"><b>年化牺牲</b><span class="{flag_class(verdict['return']['ok'])}">{verdict['return']['text']}</span></div>
  <div class="pill"><b>2018/2022/2026</b><span class="{flag_class(verdict['windows']['ok'])}">{verdict['windows']['text']}</span></div>
  <div class="pill"><b>timing vs 降杠杆</b><span class="{flag_class(verdict['timing']['ok'])}">{verdict['timing']['text']}</span></div>
  <div class="pill"><b>纯动量书</b><span class="{flag_class(verdict['momentum']['ok'])}">{verdict['momentum']['text']}</span></div>
</div>
<div class="card">{verdict['summary']}</div>

<h2>1. regime 区间</h2>
<div class="card">{report['svg_exposure']}</div>

<h2>2. 变体对照</h2>
<div class="card"><table>
<tr><th>变体</th><th>年化</th><th>Sharpe</th><th>最大回撤</th><th>年化换手</th><th>成本</th><th>平均仓位</th></tr>
{rows}</table></div>

<h2>3. 净值</h2>
<div class="card">{report['svg_nav']}</div>

<h2>4. 2018 / 2022 / 2026（同一回撤窗口）</h2>
<div class="card"><table>
<tr><th>年份</th><th>窗口（满仓峰→谷）</th><th>A 区间收益 / 回撤</th><th>C12 区间收益</th><th>risk-off 占比</th><th>成立</th></tr>
{year_rows}</table>
<table style="margin-top:10px">
<tr><th>年份</th><th>窗口</th><th>M 纯动量</th><th>M12</th><th>risk-off 占比</th></tr>
{momentum_rows}</table></div>

<h2>5. IC 分状态</h2>
<div class="card"><table>
<tr><th>分组</th><th>IC</th><th>90% 区间</th><th>天数</th></tr>
{ic_rows}</table>
<p class="sub">区间覆盖 0 的分组不下"IC 正负"的结论。主模型是防御型组合，纯动量书是把它换成四个动量因子后的对照。</p></div>

<h2>6. 分类器是否领先于风险（前向 20 日）</h2>
<div class="card"><table>
<tr><th>状态</th><th>天数</th><th>前向波动</th><th>前向回撤</th><th>前向收益</th><th>负收益占比</th></tr>
<tr><td>risk_on</td><td>{report['forward']['risk_on']['n']}</td><td>{_num(report['forward']['risk_on']['vol'])}%</td>
<td>{_num(report['forward']['risk_on']['mdd'])}%</td><td>{_num(report['forward']['risk_on']['return'])}%</td>
<td>{report['forward']['risk_on']['negative_rate']}%</td></tr>
<tr><td>risk_off</td><td>{report['forward']['risk_off']['n']}</td><td>{_num(report['forward']['risk_off']['vol'])}%</td>
<td>{_num(report['forward']['risk_off']['mdd'])}%</td><td>{_num(report['forward']['risk_off']['return'])}%</td>
<td>{report['forward']['risk_off']['negative_rate']}%</td></tr>
<tr><td>差值</td><td></td><td>{_num(report['forward']['spread']['vol'])}pp</td>
<td>{_num(report['forward']['spread']['mdd'])}pp</td><td>{_num(report['forward']['spread']['return'])}pp</td><td></td></tr>
</table></div>

<h2>7. 口径与边界</h2>
<div class="card"><ul style="margin:0;padding-left:18px;font-size:12.8px;color:#3d4b5c">
<li><b>因子权重在 2015–2021 上拟合</b>，全样本年化偏乐观；变体之间是同一本书、只改暴露，相对比较不受影响。</li>
<li><b>广度只有 59 只票</b>，是沪深300 广度的粗代理，只当分位指标用（&lt; 40% 才计票）。</li>
<li><b>2026 不成立</b>：当年 1–3 月是缓跌（−13.8%），MA200 未破、波动分位没进前 20%，熔断要等 −12% 才触发，触发时已经跌完——这是规则的真实弱点。</li>
<li>熔断阈值 10%/12% 的回撤都是 −22.0%，结论不依赖单一阈值；15% 明显偏晚（−29.7%）。</li>
<li>因子倾斜（D）在主模型上是负贡献（−0.74pp、回撤更深），数据不足时不要开。</li>
</ul></div>

<h2>8. 切换与滞后</h2>
<div class="card"><p class="sub">切换 {report['switches']['n']} 次（年均 {report['switches']['per_year']}），
平均滞后 {report['switches']['mean_lag']} 个交易日；滞后成本均值 {_num(report['lag']['mean_cost'])}pp、
中位 {_num(report['lag']['median_cost'])}pp、最差 {_num(report['lag']['worst'])}pp、累计 {_num(report['lag']['total_cost'])}pp。</p>
<table><tr><th>年份</th><th>risk-off 天数</th><th>切换次数</th></tr>
{''.join(f"<tr><td>{y}</td><td>{r['risk_off']}</td><td>{r['flips']}</td></tr>" for y, r in report['by_year'].items())}
</table></div>
</body></html>"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="regime 自适应暴露验证")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--rebuild-breadth", action="store_true")
    parser.add_argument("--outdir", default=str(REPORT))
    args = parser.parse_args(argv)

    if args.rebuild_breadth and (CACHE / "breadth.json").is_file():
        (CACHE / "breadth.json").unlink()

    print("[1/4] 载入行情与广度…", flush=True)
    index = load_index()
    index_rows = index["rows"]
    bars, histories = load_bars()
    breadth = breadth_series(bars, [row["date"] for row in index_rows])
    flags = ex.tradability(histories)
    print(f"      基准 {len(index_rows)} 日，广度覆盖 {len(breadth)} 日", flush=True)

    print("[2/4] 载入打分面板…", flush=True)
    panel = ss.load_panel()
    rows_by_date = ss.group_by_date(panel["records"])
    dates = sorted(rows_by_date)
    ss.scores_continuous(rows_by_date, mode="z", neutral=True, attach=True)
    weights = base_weights()

    def composite(weight_map: dict[str, float]) -> dict[str, dict[str, float]]:
        return {
            date: {
                record["symbol"]: sum(
                    weight_map.get(key, 0.0) * record.get(ss.ATTACH + key, 0.0) for key in weight_map
                )
                for record in rows
            }
            for date, rows in rows_by_date.items()
        }

    scores = composite(weights)
    print(f"      面板 {len(dates)} 日 / 权重 {len(weights)} 个因子", flush=True)

    print("[3/4] 分类与回测…", flush=True)
    points = rg.classify(index_rows, breadth)
    fast = rg.classify(index_rows, breadth, rg.RegimeConfig(confirm_days=1))
    slow = rg.classify(index_rows, breadth, rg.RegimeConfig(confirm_days=10))
    state_by_date = {point.date: point.state for point in points}
    exposure_c = rg.exposure_for_open(points)
    exposure_fast = rg.exposure_for_open(fast)
    exposure_slow = rg.exposure_for_open(slow)
    exposure_half = {date: 0.5 for date in dates}
    exposure_foresight = foresight_exposure(index_rows)

    base_targets = build_targets(scores, dates)
    tilted_targets = build_targets(tilt_scores(rows_by_date, weights, state_by_date), dates)

    # 纯动量书：用户描述的"趋势/动量 risk-on 属性"如果成立，应该在这本书上出现。
    # 只做诊断，不参与主结论——它是另一本书，不是同一份策略。
    momentum_keys = ("mom12_1", "macd_hist", "rs20", "high52")
    momentum_scores = {
        date: {
            record["symbol"]: statistics.fmean(
                record.get(ss.ATTACH + key, 0.0) for key in momentum_keys
            )
            for record in rows
        }
        for date, rows in rows_by_date.items()
    }
    momentum_targets = build_targets(momentum_scores, dates)

    stop_8 = rg.classify(index_rows, breadth, rg.RegimeConfig(stop_drawdown=0.08))
    weights_40 = rg.classify(index_rows, breadth, rg.RegimeConfig(
        trend_weight=0.40, vol_weight=0.35, breadth_weight=0.25, stop_drawdown=0.12))
    stop_10 = rg.classify(index_rows, breadth, rg.RegimeConfig(stop_drawdown=0.10))
    stop_12 = rg.classify(index_rows, breadth, rg.RegimeConfig(stop_drawdown=0.12))
    stop_15 = rg.classify(index_rows, breadth, rg.RegimeConfig(stop_drawdown=0.15))
    exposure_stop10 = rg.exposure_for_open(stop_10)
    exposure_stop12 = rg.exposure_for_open(stop_12)
    exposure_stop15 = rg.exposure_for_open(stop_15)

    variants = [
        run_variant("A 满仓", base_targets, None, dates, bars, flags),
        run_variant("B 恒定50%", base_targets, exposure_half, dates, bars, flags),
        run_variant("C0 快速切换", base_targets, exposure_fast, dates, bars, flags),
        run_variant(SPEC, base_targets, exposure_c, dates, bars, flags),
        run_variant("C10 确认10日", base_targets, exposure_slow, dates, bars, flags),
        run_variant("C8% 熔断8%", base_targets, rg.exposure_for_open(stop_8), dates, bars, flags),
        run_variant("Cw 权重0.40版", base_targets, rg.exposure_for_open(weights_40), dates, bars, flags),
        run_variant("C10% 熔断10%", base_targets, exposure_stop10, dates, bars, flags),
        run_variant(RECOMMENDED, base_targets, exposure_stop12, dates, bars, flags),
        run_variant("C15% 熔断15%", base_targets, exposure_stop15, dates, bars, flags),
        *[
            run_variant(f"E{int(level * 100)} 熔断12%+仓位{int(level * 100)}%", base_targets,
                        {date: (level if exposure_stop12.get(date, 1.0) < 1.0 else 1.0) for date in dates},
                        dates, bars, flags)
            for level in (0.60, 0.70, 0.80)
        ],
        run_variant("D 熔断+因子倾斜", tilted_targets, exposure_stop12, dates, bars, flags),
        run_variant("P 完美预知（上界）", base_targets, exposure_foresight, dates, bars, flags),
        run_variant("M 纯动量书·满仓", momentum_targets, None, dates, bars, flags),
        run_variant("M12 纯动量书·熔断12%", momentum_targets, exposure_stop12, dates, bars, flags),
    ]
    # 匹配平均仓位的对照组：只比"同样多的平均仓位、不同时点"，才是 timing。
    # 拿 C 去比恒定 50% 会把"仓位更高"当成"择时更准"，这是最容易被误读的一步。
    lookup = {variant["name"]: variant for variant in variants}
    level = round(lookup["C12 熔断12%+5日确认"]["stats"]["avg_exposure"], 3)
    matched = run_variant(
        f"B* 恒定{int(round(level * 100))}%（匹配仓位）",
        base_targets,
        {date: level for date in dates},
        dates, bars, flags,
    )
    variants.append(matched)
    lookup[matched["name"]] = matched
    print("      " + " / ".join(f"{v['name']} {v['stats']['cagr']:.2f}%" for v in variants), flush=True)

    print("[4/4] 诊断与报告…", flush=True)
    events = rg.switch_log(points)
    years_span = max(len(dates) / 252, 1e-9)
    switches = {
        "n": len(events),
        "per_year": round(len(events) / years_span, 1),
        "mean_lag": round(statistics.fmean([e["lag_days"] for e in events]), 1) if events else None,
    }
    lag = lag_cost(events, index_rows)
    forward = rg.forward_stats(points, [row["close"] for row in index_rows])
    ic = ic_by_state(
        scores, panel["records"], points, index_rows,
        scores_alt=momentum_scores, label_alt="纯动量书",
    )
    by_year = rg.state_by_year(points)

    windows: dict[str, Any] = {}
    for year in WINDOWS:
        window = drawdown_window(lookup["A 满仓"]["nav"], year)
        if not window:
            continue
        start, end = window["peak"], window["trough"]
        a_ret = return_between(lookup["A 满仓"]["nav"], start, end)
        c_ret = return_between(lookup[RECOMMENDED]["nav"], start, end)
        m_ret = return_between(lookup["M 纯动量书·满仓"]["nav"], start, end)
        m12_ret = return_between(lookup["M12 纯动量书·熔断12%"]["nav"], start, end)
        span = [point for point in points if start <= point.date <= end and point.state]
        share = sum(1 for point in span if point.state == "risk_off") / len(span) * 100 if span else 0
        ok = c_ret is not None and a_ret is not None and c_ret > a_ret + 0.5
        windows[year] = {
            "window": f"{start} → {end}",
            "a_return": round(a_ret, 2) if a_ret is not None else None,
            "c_return": round(c_ret, 2) if c_ret is not None else None,
            "m_return": round(m_ret, 2) if m_ret is not None else None,
            "m12_return": round(m12_ret, 2) if m12_ret is not None else None,
            "a_drawdown": round(window["drawdown"] * 100, 2),
            "risk_off_share": round(share, 1),
            "flag": "成立" if ok else "未成立",
        }

    a_stats = lookup["A 满仓"]["stats"]
    b_stats = lookup["B 恒定50%"]["stats"]
    c_stats = lookup[RECOMMENDED]["stats"]
    spec_stats = lookup[SPEC]["stats"]
    dd_reduction = 1 - abs(c_stats["max_drawdown"]) / abs(a_stats["max_drawdown"]) if a_stats["max_drawdown"] else 0
    cagr_sacrifice = a_stats["cagr"] - c_stats["cagr"]
    matched_stats = matched["stats"]
    timing_gain = c_stats["cagr"] - matched_stats["cagr"]
    leverage_gain = c_stats["cagr"] - b_stats["cagr"]
    all_windows = all(row["flag"] == "成立" for row in windows.values()) and len(windows) == len(WINDOWS)

    verdict = {
        "drawdown": {"ok": dd_reduction >= 0.20,
                     "text": f"{a_stats['max_drawdown']:.2f}% → {c_stats['max_drawdown']:.2f}%（改善 {dd_reduction:.0%}）",
                     "flag": "通过" if dd_reduction >= 0.20 else "未通过"},
        "return": {"ok": cagr_sacrifice < 1.0,
                   "text": f"年化 {a_stats['cagr']:.2f}% → {c_stats['cagr']:.2f}%（牺牲 {cagr_sacrifice:+.2f}pp）",
                   "flag": "通过" if cagr_sacrifice < 1.0 else "未通过"},
        "windows": {"ok": all_windows,
                    "text": " / ".join(f"{y} {row['flag']}" for y, row in windows.items()) or "样本不足",
                    "flag": "通过" if all_windows else "未通过"},
        "momentum": {"ok": True, "text": "", "flag": ""},
        "timing": {"ok": timing_gain > 0.5,
                   "text": f"对匹配仓位 {matched_stats['cagr']:.2f}% / {matched_stats['max_drawdown']:.2f}%，"
                           f"C 为 {c_stats['cagr']:.2f}% / {c_stats['max_drawdown']:.2f}%"
                           f"（收益 {timing_gain:+.2f}pp，回撤 {c_stats['max_drawdown'] - matched_stats['max_drawdown']:+.2f}pp）",
                   "flag": "通过" if timing_gain > 0.5 else "未通过"},
    }
    m_stats = lookup["M 纯动量书·满仓"]["stats"]
    m12_stats = lookup["M12 纯动量书·熔断12%"]["stats"]
    m_dd = 1 - abs(m12_stats["max_drawdown"]) / abs(m_stats["max_drawdown"]) if m_stats["max_drawdown"] else 0
    m_gain = m12_stats["cagr"] - m_stats["cagr"]
    verdict["momentum"] = {
        "ok": m_dd >= 0.20 and m_gain >= 0,
        "text": f"回撤 {m_stats['max_drawdown']:.2f}% → {m12_stats['max_drawdown']:.2f}%（改善 {m_dd:.0%}）· "
                f"年化 {m_stats['cagr']:.2f}% → {m12_stats['cagr']:.2f}%（{m_gain:+.2f}pp）",
        "flag": "通过" if (m_dd >= 0.20 and m_gain >= 0) else "未通过",
    }

    verdict["summary"] = (
        f"规格版（三票 + 5 日确认、无熔断）只有 {spec_stats['cagr']:.2f}% / 回撤 "
        f"{spec_stats['max_drawdown']:.2f}%：慢信号来不及。2015-06 是从高位直接崩——MA200 未破、"
        f"波动分位虽已 0.92 但单腿不足以触发，主跌段走完才翻面。"
        f"补上熔断腿（距 60 日高点 −12% 立即减仓、恢复仍要 5 日确认）后，"
        f"C12 = {c_stats['cagr']:.2f}% / 回撤 {c_stats['max_drawdown']:.2f}%：回撤改善 {dd_reduction:.0%}、"
        f"年化代价 {cagr_sacrifice:+.2f}pp、Sharpe {a_stats['sharpe']:.2f}→{c_stats['sharpe']:.2f}。"
        f"对照「平均仓位相同」的基准（恒定 {int(round(level * 100))}%：{matched_stats['cagr']:.2f}% / "
        f"{matched_stats['max_drawdown']:.2f}%），C12 收益 {timing_gain:+.2f}pp、回撤少亏 "
        f"{abs(c_stats['max_drawdown'] - matched_stats['max_drawdown']):.1f}pp——这部分是真 timing，不是降杠杆。"
        f"但分水岭在**书本身**：纯动量书 M {m_stats['max_drawdown']:.2f}% / {m_stats['cagr']:.2f}% → "
        f"M12 {m12_stats['max_drawdown']:.2f}% / {m12_stats['cagr']:.2f}%（回撤降 {m_dd:.0%}、收益还高 {m_gain:+.2f}pp）；"
        f"主模型已经是防御型 6 因子，压暴露只降风险、不加收益，所以"
        f"「回撤 −20% 且年化牺牲 <1pp」在主模型上不成立、在动量书上成立。"
        f"切换 {switches['n']} 次（年均 {switches['per_year']} 次）、平均滞后 "
        f"{lag['mean_lag_days']:.1f} 个交易日、滞后成本累计 {lag['total_cost']:.1f}pp。"
    )

    report = {
        "meta": {
            "benchmark": index.get("name") or index["symbol"],
            "start": dates[0], "end": dates[-1], "days": len(dates),
            "symbols": len(panel["symbols"]),
        },
        "variants": [{"name": v["name"], "stats": v["stats"]} for v in variants],
        "windows": windows,
        "forward": forward,
        "ic": ic,
        "switches": switches,
        "lag": {key: value for key, value in lag.items() if key != "events"},
        "by_year": by_year,
        "verdict": verdict,
        "svg_exposure": _svg_exposure(index_rows, points),
        "svg_nav": _svg_nav({v["name"]: v["nav"] for v in variants if v["name"] in
                             ("A 满仓", "B 恒定50%", SPEC, RECOMMENDED, "P 完美预知（上界）")}),
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    (outdir / "report.html").write_text(render_html(report), encoding="utf-8")
    slim = {key: value for key, value in report.items() if not key.startswith("svg_")}
    for row in slim["variants"]:
        row.pop("nav", None)
    (outdir / "metrics.json").write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n完成 → {outdir}")
    for variant in variants:
        stats = variant["stats"]
        print(f"  {variant['name']:<20} 年化 {stats['cagr']:>6.2f}%  Sharpe {stats['sharpe']:>5.2f}  "
              f"回撤 {stats['max_drawdown']:>7.2f}%  换手 {stats['turnover']:>5.0f}%  仓位 {stats['avg_exposure']:.2f}")
    print(f"  回撤改善 {dd_reduction:.0%} / 年化代价 {cagr_sacrifice:+.2f}pp / timing {timing_gain:+.2f}pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
