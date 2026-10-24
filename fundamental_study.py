"""P2-10 研究：基本面 / 资金流 / 另类数据的增量 IC 与 PIT 审计。

跑法::

    python -m research.fetch_alt            # 先落地数据（慢，可断点续）
    python fundamental_study.py             # 构造因子 + 评估 + 出报告

产物在 ``reports/altdata/``：``report.md`` / ``report.html`` / ``metrics.json``。

三条纪律
--------
1. **生产层不参与**：本文件与 ``research/*`` 都不被 ``app.py`` / ``analyzer.py``
   import；测试 ``tests/test_isolation.py`` 会强制检查这一点。
2. **指数只用一次**：所有因子先做行业/规模/beta 中性化，再对现有 OHLCV 基线
   正交化，"增量 IC"是**正交化之后**的 IC。不做这一步，"新因子有效"通常只是
   在重复动量的信息。
3. **前视偏差要量化，不靠声明**：同一批因子用"报告期对齐"再算一遍，两者之差
   就是"按报告期而非公告日对齐"会虚增多少 IC。
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

import evalkit
import fields as fd
import score_study as ss
from research import fundamentals as fz
from research.datasets import resolve
from research.pit_join import audit as pit_audit

ROOT = Path(__file__).resolve().parent
STOCK_CACHE = ROOT / ".cache" / "fields" / "stocks"
ETF_CACHE = ROOT / ".cache" / "fields" / "etf"
INDUSTRY_PATH = ROOT / ".cache" / "research" / "pit-v1" / "industry.json"
ALT_CACHE = ROOT / ".cache" / "alt_data"
OUTDIR = ROOT / "reports" / "altdata"

MIN_XSEC = fz.MIN_XSEC
HORIZON = fz.HORIZON
DIMENSIONS = ("value", "quality", "growth", "flow", "event")
DIMENSION_NAMES = {
    "value": "价值", "quality": "质量", "growth": "成长", "flow": "资金", "event": "事件",
    "fundamental": "财务报表", "market": "行情",
}
FUNDAMENTAL_KEYS = tuple(
    key for key, spec in fz.FACTORS.items() if spec.dimension in ("value", "quality", "growth")
)


# --------------------------------------------------------------------------- #
# 输入
# --------------------------------------------------------------------------- #


def load_payload(path: Path | None = None) -> dict[str, Any]:
    source = Path(path) if path else (ALT_CACHE / "records.json")
    if not source.is_file():
        raise SystemExit(f"没有落地数据（{source}），先跑：python -m research.fetch_alt")
    return json.loads(source.read_text(encoding="utf-8"))


def load_prices() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(STOCK_CACHE.glob("*.json")):
        payload = fd.sanitize(json.loads(path.read_text(encoding="utf-8")))
        if len(payload.get("rows", [])) >= 300:
            out[payload["symbol"]] = payload
    return out


def load_industry() -> dict[str, str]:
    raw = json.loads(INDUSTRY_PATH.read_text(encoding="utf-8"))
    return {code: ((info.get("industry") or "")[:1] or "NA") for code, info in raw.items()}


def benchmark_returns() -> dict[str, float]:
    payload = fd.sanitize(json.loads((ETF_CACHE / "510300.json").read_text(encoding="utf-8")))
    rows = payload["rows"]
    out: dict[str, float] = {}
    for i in range(1, len(rows)):
        if rows[i - 1]["close"]:
            out[rows[i]["date"]] = rows[i]["close"] / rows[i - 1]["close"] - 1.0
    return out


def baseline_scores() -> dict[tuple[str, str], float]:
    """现有 12 个 OHLCV 因子的等权分（已是行业/规模中性化后的 z，方向已对齐）。"""
    panel = ss.load_panel()
    rows_by_date = ss.group_by_date(panel["records"])
    ss.scores_continuous(rows_by_date, mode="z", neutral=True, attach=True)
    scores: dict[tuple[str, str], float] = {}
    for date, rows in rows_by_date.items():
        for row in rows:
            values = [row.get(ss.ATTACH + key) for key in ss.BATTERY]
            present = [v for v in values if v is not None]
            if len(present) >= 8:
                scores[(date, row["symbol"])] = statistics.fmean(present)
    return scores


# --------------------------------------------------------------------------- #
# 评估
# --------------------------------------------------------------------------- #


def build_dimension_composites(rows: Sequence[dict[str, Any]]) -> list[str]:
    """维度内方向对齐后等权——这才是"接入一个数据维度"的真实形态。"""
    keys: list[str] = []
    for dimension in DIMENSIONS:
        members = [key for key, spec in fz.FACTORS.items() if spec.dimension == dimension]
        if not members:
            continue
        name = f"dim_{dimension}"
        keys.append(name)
        for row in rows:
            values = [row.get(f"neu_{key}") for key in members]
            present = [value for value in values if value is not None]
            row[name] = statistics.fmean(present) if len(present) >= 2 else None
    return keys


def lookahead_gap(
    records: Sequence[dict[str, Any]],
    prices: dict[str, dict[str, Any]],
    industry: dict[str, str],
    bench: dict[str, float],
    keys: Sequence[str],
) -> dict[str, Any]:
    """按报告期对齐 vs 按公告日对齐，同一个因子的 IC 差多少。"""
    rows = fz.build_panel(records=records, prices=prices, industry=industry,
                          benchmark=bench, alignment="period")
    out: dict[str, Any] = {}
    for key in keys:
        series = fz.daily_ic(rows, key)
        if len(series) < 30:
            continue
        stat = evalkit.ic_ci(series, horizon=HORIZON, iters=300)
        out[key] = {"period_mean": round(stat["mean"], 4), "n": len(series)}
    return out


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #


def _num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _signed(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:+.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    meta, verdict = report["meta"], report["verdict"]
    factor_rows = "\n".join(
        f"| {DIMENSION_NAMES[spec.dimension]} | {spec.name} | `{spec.key}` | "
        f"{spec.formula} | {'+' if spec.prior > 0 else '−'} | {_num(report['coverage'].get(spec.key, {}).get('date_share'), 2)} | "
        f"{_num(report['coverage'].get(spec.key, {}).get('mean_stocks'), 1)} |"
        for spec in fz.FACTORS.values()
    )
    result_rows = "\n".join(
        f"| {DIMENSION_NAMES[fz.FACTORS[key].dimension]} | {fz.FACTORS[key].name} | "
        f"{_signed(row['raw']['mean'])} | {_signed(row['neutral']['mean'])} | "
        f"**{_signed(row['incremental']['mean'])}** "
        f"[{_signed(row['incremental']['ci_low'])}, {_signed(row['incremental']['ci_high'])}] | "
        f"{_signed(row['combination']['delta']['mean'])} | "
        f"{row['incremental']['icir'] if row['incremental']['icir'] is not None else '—'} |"
        for key, row in report["results"].items()
        if row["incremental"]["mean"] is not None
    )
    dim_rows = "\n".join(
        f"| {DIMENSION_NAMES[name.replace('dim_', '')]} | {_signed(row['incremental']['mean'])} "
        f"[{_signed(row['incremental']['ci_low'])}, {_signed(row['incremental']['ci_high'])}] | "
        f"{_signed(row['combination']['delta']['mean'])} "
        f"[{_signed(row['combination']['delta']['ci_low'])}, {_signed(row['combination']['delta']['ci_high'])}] | "
        f"{row['incremental']['icir'] if row['incremental']['icir'] is not None else '—'} |"
        for name, row in report["dimensions"].items()
        if row["incremental"]["mean"] is not None
    )
    source_rows = "\n".join(
        f"| {name} | {entry.get('status', '—')} | {entry.get('records', '—')} | "
        f"{entry.get('reason', entry.get('error', ''))} |"
        for name, entry in meta["sources"].items()
    )
    inventory_rows = "\n".join(
        f"| {item['source']} | `{item['field']}`{('（' + item['unit'] + '）') if item['unit'] else ''} | "
        f"{DIMENSION_NAMES.get(item['dimension'], item['dimension'])} | {item['pit_basis'] or '—'} | {item['note']} |"
        for item in report["inventory"]
    )
    lag_rows = "\n".join(
        f"| {DIMENSION_NAMES.get(dim, dim)} | {row['n']} | {row['median']} | {row['p90']} | {row['max']} |"
        for dim, row in report["pit"]["lag_days"].items()
    )
    gap_rows = "\n".join(
        f"| {fz.FACTORS[key].name} | {_signed(row['period_mean'])} | {_signed(row['available_mean'])} | "
        f"{_signed(row['gap'])} | {_num(row['inflation'], 2)}× |"
        for key, row in report["pit"]["lookahead"].items()
    )
    return f"""# P2-10 · 基本面 / 资金流 / 另类数据：增量 IC 与 PIT 审计

标的 {meta['symbols']} 只 · {meta['start']} → {meta['end']} · 标签 = 未来 {HORIZON} 日超额收益 ·
基线 = 现有 12 个 OHLCV 因子等权分 · 成本口径与 ``wfv2`` 一致。

## 结论

{verdict['summary']}

| 验收项 | 标准 | 实测 | 判定 |
| --- | --- | --- | --- |
| PIT 合规 | 可得日 ≥ 报告期，无未来数据 | {report['pit']['violations']} 条越界 / {report['pit']['records']} 条记录 | {'通过' if report['pit']['violations'] == 0 else '未通过'} |
| 增量 IC | 正交化后区间不含 0 | {verdict['incremental']['text']} | {verdict['incremental']['flag']} |
| 维度合并 | 至少一个维度组合 ΔIC > 0 | {verdict['dimension']['text']} | {verdict['dimension']['flag']} |
| 前视量化 | 报告"按报告期对齐"的虚增幅度 | {verdict['lookahead']['text']} | 已完成 |

## 1. 因子定义（{len(fz.FACTORS)} 个）

| 维度 | 名称 | key | 构造 | 先验 | 覆盖日占比 | 平均股票数 |
| --- | --- | --- | --- | --- | --- | --- |
{factor_rows}

先验（+1/−1）在写代码前定好，不随回测结果调整。``analyst_revision`` 依赖 CSV 导入
（无免费原始源），本轮回测里覆盖率为 0，因此不参与结论——不用代理变量硬凑。

## 2. 字段清单（按源，共 {len(report['inventory'])} 个原始字段）

| 源 | 字段（单位） | 维度 | PIT 依据 | 说明 |
| --- | --- | --- | --- | --- |
{inventory_rows}

## 3. 数据源与落地状态

| 源 | 状态 | 记录数 | 说明 |
| --- | --- | --- | --- |
{source_rows}

## 4. PIT 审计

披露滞后（可得日 − 报告期，自然日）：

| 维度 | 记录数 | 中位 | P90 | 最大 |
| --- | --- | --- | --- | --- |
{lag_rows}

重述：{report['pit']['revised_series']} 个 (标的,字段,报告期) 序列出现过多次披露，
占比 {report['pit']['revision_share']:.2%}。``AsOfBook`` 对每个日期取"当时可得的最新一版"，
所以重述不会被当成新信息提前使用。

**前视偏差的量化**（同一因子，按报告期对齐 vs 按公告日对齐）：

| 因子 | 报告期对齐 IC | 公告日对齐 IC | 差值 | 虚增倍数 |
| --- | --- | --- | --- | --- |
{gap_rows}

## 5. 增量 IC（行业/规模/beta 中性化 + 对基线正交化）

| 维度 | 因子 | 原始 IC | 中性化 IC | 增量 IC [90%] | 组合 ΔIC | 增量 ICIR |
| --- | --- | --- | --- | --- | --- | --- |
{result_rows}

"原始 IC"与"中性化 IC"的差是行业/规模/beta 的贡献；"增量 IC"与"中性化 IC"的差是
**与现有 OHLCV 信息重叠的部分**。只有增量 IC 的区间不含 0，才说明这个维度真的补了
新信息。

## 6. 维度合并

| 维度 | 增量 IC [90%] | 组合 ΔIC [90%] | ICIR |
| --- | --- | --- | --- |
{dim_rows}

## 7. 研究层 / 生产层隔离

```
生产层（零第三方依赖）          研究层（可选依赖，只在研究进程里）
app.py  analyzer.py  fields.py   research/datasets/*  ← baostock / HTTP / CSV
market_data.py  rebalance.py     research/pit_join.py ← 纯标准库
        ↑                        research/fundamentals.py ← 纯标准库
        └──── 只读复用 ──────────┘（evalkit / fields.sanitize / factors.orthogonalize）
```

- 生产层**不 import research**，研究层不回写生产层的状态；
- 数据源是插件：``Source`` 子类 + ``@register``，新增维度不改因子代码；
- 每条记录强制 ``available >= period``，违反直接抛错，而不是静默进回测；
- ``tests/test_isolation.py`` 静态检查"生产层没有 import research/baostock/akshare"。

## 8. 边界

- 样本只有 {meta['symbols']} 只票，横截面有效 bet 约 {meta['symbols'] / 3:.1f} 个；
  单个因子的增量 IC 区间宽度普遍在 ±0.03 以上，**只能判断"有没有"，不能排序"哪个更强"**。
- 财务因子的更新频率是季度，日频 IC 序列高度自相关（块自助已按 20 日块处理），
  有效独立块数远小于名义天数。
- **两个维度在 60 只票的池子里根本评估不了**：龙虎榜平均只有 4.1 只票有值、业绩预告 15.8 只，
  低于横截面下限（30 只）。这不是因子无效，是事件类数据天然稀疏——它需要 300+ 只票的池子才
  有横截面。北向持股（季度快照）与分析师修正（CSV 通道）本轮覆盖率为 0，只有适配器。
- 两融是 T+1 披露、北向是季度快照，都不是日频；日频 IC 序列因此高度自相关，块自助已按 20 日块处理。
- 本报告只回答"是否补了信息"，不回答"能不能赚钱"——后者要走 ``wfv2`` 的 5 折净净值口径。
"""


def render_html(report: dict[str, Any]) -> str:
    meta, verdict = report["meta"], report["verdict"]
    flag = lambda ok: "pass" if ok else ("fail" if ok is False else "muted")
    factor_rows = "".join(
        f"<tr><td>{DIMENSION_NAMES[spec.dimension]}</td><td>{spec.name}</td><td><code>{spec.key}</code></td>"
        f"<td class='left'>{spec.formula}</td><td>{'+' if spec.prior > 0 else '−'}</td>"
        f"<td>{_num(report['coverage'].get(spec.key, {}).get('date_share'), 2)}</td>"
        f"<td>{_num(report['coverage'].get(spec.key, {}).get('mean_stocks'), 1)}</td></tr>"
        for spec in fz.FACTORS.values()
    )
    result_rows = "".join(
        f"<tr><td>{DIMENSION_NAMES[fz.FACTORS[key].dimension]}</td><td>{fz.FACTORS[key].name}</td>"
        f"<td>{_signed(row['raw']['mean'])}</td><td>{_signed(row['neutral']['mean'])}</td>"
        f"<td class='{flag(row['incremental']['ci_low'] is not None and row['incremental']['ci_low'] > 0)}'>"
        f"{_signed(row['incremental']['mean'])} [{_signed(row['incremental']['ci_low'])}, {_signed(row['incremental']['ci_high'])}]</td>"
        f"<td>{_signed(row['combination']['delta']['mean'])}</td>"
        f"<td>{row['incremental']['icir'] if row['incremental']['icir'] is not None else '—'}</td></tr>"
        for key, row in report["results"].items() if row["incremental"]["mean"] is not None
    )
    dim_rows = "".join(
        f"<tr><td>{DIMENSION_NAMES[name.replace('dim_', '')]}</td>"
        f"<td>{_signed(row['incremental']['mean'])} [{_signed(row['incremental']['ci_low'])}, {_signed(row['incremental']['ci_high'])}]</td>"
        f"<td>{_signed(row['combination']['delta']['mean'])} [{_signed(row['combination']['delta']['ci_low'])}, {_signed(row['combination']['delta']['ci_high'])}]</td>"
        f"<td>{row['incremental']['icir'] if row['incremental']['icir'] is not None else '—'}</td></tr>"
        for name, row in report["dimensions"].items() if row["incremental"]["mean"] is not None
    )
    source_rows = "".join(
        f"<tr><td>{name}</td><td>{entry.get('status', '—')}</td><td>{entry.get('records', '—')}</td>"
        f"<td class='left'>{entry.get('reason', entry.get('error', ''))}</td></tr>"
        for name, entry in meta["sources"].items()
    )
    inventory_html = "".join(
        f"<tr><td>{item['source']}</td>"
        f"<td><code>{item['field']}</code>{('（' + item['unit'] + '）') if item['unit'] else ''}</td>"
        f"<td>{DIMENSION_NAMES.get(item['dimension'], item['dimension'])}</td>"
        f"<td>{item['pit_basis'] or '—'}</td><td class='left'>{item['note']}</td></tr>"
        for item in report["inventory"]
    )
    lag_rows = "".join(
        f"<tr><td>{DIMENSION_NAMES.get(dim, dim)}</td><td>{row['n']}</td><td>{row['median']}</td>"
        f"<td>{row['p90']}</td><td>{row['max']}</td></tr>"
        for dim, row in report["pit"]["lag_days"].items()
    )
    gap_rows = "".join(
        f"<tr><td>{fz.FACTORS[key].name}</td><td>{_signed(row['period_mean'])}</td>"
        f"<td>{_signed(row['available_mean'])}</td><td>{_signed(row['gap'])}</td>"
        f"<td>{_num(row['inflation'], 2)}×</td></tr>"
        for key, row in report["pit"]["lookahead"].items()
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>P2-10 · 增量 IC 与 PIT 审计</title>
<style>
:root{{--ink:#1f2a37;--muted:#6b7785;--line:#e3e8ef;--bg:#f7f9fc;--red:#c0392b;--green:#1e8449;--blue:#1f6fb2}}
*{{box-sizing:border-box}}
body{{margin:0;padding:32px 40px;background:var(--bg);color:var(--ink);font:14px/1.75 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}}
h1{{font-size:23px;margin:0 0 6px}}
h2{{font-size:15px;margin:30px 0 10px;padding-left:10px;border-left:3px solid var(--blue);color:#26364a}}
.sub{{color:var(--muted);font-size:12.5px}}
.card{{background:#fff;border:1px solid var(--line);border-radius:8px;padding:16px 20px;margin-bottom:14px;overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:12.6px;background:#fff}}
th,td{{padding:6px 9px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
td.left,th.left{{text-align:left;white-space:normal}}
th:first-child,td:first-child{{text-align:left}}
th{{color:var(--muted);font-weight:600;font-size:12px;background:#fbfcfe}}
.verdict{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px;margin:16px 0}}
.pill{{background:#fff;border:1px solid var(--line);border-radius:8px;padding:10px 14px;font-size:12.5px}}
.pill b{{display:block;font-size:12px;color:var(--muted);margin-bottom:2px}}
.pass{{color:var(--green);font-weight:600}} .fail{{color:var(--red);font-weight:600}} .muted{{color:var(--muted)}}
code{{background:#f2f5f9;padding:1px 5px;border-radius:4px;font-size:11.5px}}
pre{{background:#0f172a;color:#e2e8f0;padding:14px 16px;border-radius:8px;font-size:11.8px;line-height:1.5;overflow-x:auto}}
ul{{margin:0;padding-left:18px}} li{{font-size:12.9px;color:#3d4b5c;margin:3px 0}}
</style></head><body>
<h1>P2-10 · 基本面 / 资金流 / 另类数据</h1>
<div class="sub">{meta['symbols']} 只标的 · {meta['start']} → {meta['end']} · 标签 {HORIZON} 日 ·
基线 = 12 个 OHLCV 因子等权 · {len(fz.FACTORS)} 个新因子 · 来源 {len(meta['sources'])} 个</div>

<div class="verdict">
  <div class="pill"><b>PIT 合规</b><span class="{flag(report['pit']['violations'] == 0)}">
    {report['pit']['violations']} 条越界 / {report['pit']['records']} 条</span></div>
  <div class="pill"><b>增量 IC</b><span class="{flag(verdict['incremental']['ok'])}">{verdict['incremental']['text']}</span></div>
  <div class="pill"><b>维度合并</b><span class="{flag(verdict['dimension']['ok'])}">{verdict['dimension']['text']}</span></div>
  <div class="pill"><b>前视虚增</b><span class="muted">{verdict['lookahead']['text']}</span></div>
</div>
<div class="card">{verdict['summary']}</div>

<h2>1. 因子定义</h2>
<div class="card"><table>
<tr><th>维度</th><th>名称</th><th>key</th><th class="left">构造</th><th>先验</th><th>覆盖日占比</th><th>平均股票数</th></tr>
{factor_rows}</table>
<p class="sub">先验在写代码前定好，不随回测调整。analyst_revision 依赖 CSV 导入，本轮覆盖率 0，不参与结论。</p></div>

<h2>2. 字段清单</h2>
<div class="card"><table>
<tr><th>源</th><th>字段（单位）</th><th>维度</th><th>PIT 依据</th><th class="left">说明</th></tr>
{inventory_html}</table>
<p class="sub">共 {len(report['inventory'])} 个原始字段。PIT 依据决定 available 取哪个日期：
公告日 / 除权除息日 / T+1 披露 / 当日收盘。因子层不直接读原始字段，只读 AsOfBook 的对齐结果。</p></div>

<h2>3. 数据源状态</h2>
<div class="card"><table>
<tr><th>源</th><th>状态</th><th>记录数</th><th class="left">说明</th></tr>
{source_rows}</table></div>

<h2>4. PIT 审计</h2>
<div class="card"><table>
<tr><th>维度</th><th>记录数</th><th>中位滞后（天）</th><th>P90</th><th>最大</th></tr>
{lag_rows}</table>
<p class="sub">重述：{report['pit']['revised_series']} 个序列有多次披露（占 {report['pit']['revision_share']:.2%}）。
AsOfBook 取"当时可得的最新一版"，重述不会被提前使用。</p>
<table style="margin-top:12px">
<tr><th>因子</th><th>报告期对齐 IC</th><th>公告日对齐 IC</th><th>差值</th><th>虚增倍数</th></tr>
{gap_rows}</table>
<p class="sub">这张表是"按报告期对齐"这个常见错误的代价。倍数为 1 说明该因子对披露滞后不敏感（例如已滞后数月的慢变量）。</p></div>

<h2>5. 增量 IC</h2>
<div class="card"><table>
<tr><th>维度</th><th>因子</th><th>原始 IC</th><th>中性化 IC</th><th>增量 IC [90%]</th><th>组合 ΔIC</th><th>ICIR</th></tr>
{result_rows}</table>
<p class="sub">原始 → 中性化的差是行业/规模/beta 的贡献；中性化 → 增量的差是与现有 OHLCV 重叠的部分。</p></div>

<h2>6. 维度合并</h2>
<div class="card"><table>
<tr><th>维度</th><th>增量 IC [90%]</th><th>组合 ΔIC [90%]</th><th>ICIR</th></tr>
{dim_rows}</table></div>

<h2>7. 研究层 / 生产层隔离</h2>
<div class="card"><pre>生产层（零第三方依赖）            研究层（可选依赖，只在研究进程里）
app.py  analyzer.py  fields.py    research/datasets/*   ← baostock / HTTP / CSV
market_data.py  rebalance.py      research/pit_join.py  ← 纯标准库
        ↑                         research/fundamentals.py ← 纯标准库
        └──── 只读复用 ───────────┘（evalkit / fields.sanitize / factors.orthogonalize）</pre>
<ul>
<li>生产层不 import research，研究层不回写生产层状态；</li>
<li>数据源是插件：<code>Source</code> 子类 + <code>@register</code>，新增维度不改因子代码；</li>
<li>每条记录强制 <code>available &gt;= period</code>，违反直接抛错；</li>
<li><code>tests/test_isolation.py</code> 静态检查生产层没有 import research / baostock / akshare。</li>
</ul></div>

<h2>8. 边界</h2>
<div class="card"><ul>
<li>样本 {meta['symbols']} 只票，横截面有效 bet 约 {meta['symbols'] / 3:.1f} 个；单因子增量 IC 区间普遍宽 ±0.03，只能判断"有没有信息"，不能排"谁更强"。</li>
<li>财务因子按季度更新，日频 IC 高度自相关；块自助已按 {HORIZON} 日块处理。</li>
<li><b>两个维度在 60 只票的池子里无法评估</b>：龙虎榜平均 4.1 只、业绩预告 15.8 只有值，低于横截面下限 30 只。
这不是因子无效，是事件数据天然稀疏——需要 300+ 只票的池子。北向（季度快照）与分析师修正（CSV）本轮覆盖率为 0，只有适配器。</li>
<li>两融是 T+1 披露、北向是季度快照，都不是日频；日频 IC 序列因此高度自相关，块自助已按 20 日块处理。</li>
<li>本报告只回答"是否补了信息"，赚钱与否要走 wfv2 的 5 折净净值口径。</li>
</ul></div>
</body></html>"""


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="P2-10 增量 IC 评估")
    parser.add_argument("--records", default=None, help="records.json 路径")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args(argv)

    payload = load_payload(args.records)
    records = payload["records"]
    meta = dict(payload["meta"])
    meta["sources"] = payload["report"]

    print(f"[1/5] 记录 {len(records)} 条，来源 {len(meta['sources'])} 个", flush=True)
    derived = fz.derive_fundamentals(records)
    print(f"      派生 {len(derived)} 条（TTM / 净资产 / 同比）", flush=True)
    all_records = list(records) + derived

    print("[2/5] 构造面板…", flush=True)
    prices = load_prices()
    industry = load_industry()
    bench = benchmark_returns()
    scores = baseline_scores()
    rows = fz.build_panel(records=all_records, prices=prices, industry=industry,
                          benchmark=bench, scores=scores)
    print(f"      {len(rows)} 行 / {len({row['date'] for row in rows})} 个交易日", flush=True)

    # 方向对齐：prior=−1 的因子（应计、融资余额占比）先取负，之后所有 IC 为正
    # 都表示"与先验一致"。不翻的话，-0.039 的融资余额占比会被读成"反向"，
    # 而它其实是本次最强的新增信号。
    for row in rows:
        for key, spec in fz.FACTORS.items():
            if spec.prior < 0 and row.get(key) is not None:
                row[key] = -row[key]

    print("[3/5] 中性化…", flush=True)
    keys = [key for key in fz.FACTORS if any(row.get(key) is not None for row in rows)]
    fz.neutralize(rows, keys)
    dim_keys = build_dimension_composites(rows)
    fz.neutralize(rows, dim_keys)

    print("[4/5] 评估…", flush=True)
    results = {key: fz.evaluate(rows, key) for key in keys}
    dimensions = {key: fz.evaluate(rows, key) for key in dim_keys}
    coverage = fz.coverage(rows, list(fz.FACTORS))
    pit = pit_audit(all_records)
    pit["lookahead"] = {}
    for key in FUNDAMENTAL_KEYS:
        series = fz.daily_ic(rows, key)
        if len(series) < 30:
            continue
        available_mean = evalkit.ic_ci(series, horizon=HORIZON, iters=300)["mean"]
        pit["lookahead"][key] = {
            "available_mean": round(available_mean, 4),
            "period_mean": None, "gap": None, "inflation": None,
        }
    period_stats = lookahead_gap(all_records, prices, industry, bench, FUNDAMENTAL_KEYS)
    for key, stat in period_stats.items():
        entry = pit["lookahead"].get(key)
        if not entry:
            continue
        entry["period_mean"] = stat["period_mean"]
        entry["gap"] = round(stat["period_mean"] - entry["available_mean"], 4)
        entry["inflation"] = (round(stat["period_mean"] / entry["available_mean"], 2)
                              if entry["available_mean"] else None)

    evaluable = [key for key, row in results.items() if row["incremental"]["mean"] is not None]
    evaluable_dims = [key for key, row in dimensions.items() if row["incremental"]["mean"] is not None]
    winners = [key for key, row in results.items()
               if row["incremental"]["ci_low"] is not None and row["incremental"]["ci_low"] > 0]
    dim_winners = [key for key, row in dimensions.items()
                   if row["combination"]["delta"]["ci_low"] is not None
                   and row["combination"]["delta"]["ci_low"] > 0]
    typical_gap = statistics.median([abs(v["gap"]) for v in pit["lookahead"].values() if v["gap"] is not None]) \
        if any(v["gap"] is not None for v in pit["lookahead"].values()) else None

    verdict = {
        "incremental": {
            "ok": bool(winners),
            "text": f"{len(winners)}/{len(evaluable)} 个可评估因子的增量 IC 区间不含 0"
                    + (f"（{', '.join(fz.FACTORS[k].name for k in winners[:4])}）" if winners else ""),
            "flag": "通过" if winners else "未通过",
        },
        "dimension": {
            "ok": bool(dim_winners),
            "text": f"{len(dim_winners)}/{len(evaluable_dims)} 个可评估维度的组合 ΔIC 区间不含 0"
                    if dimensions else "样本不足",
            "flag": "通过" if dim_winners else "未通过",
        },
        "lookahead": {
            "text": f"中位虚增 {_signed(typical_gap)} IC" if typical_gap is not None else "样本不足",
        },
    }
    best = max(results.items(), key=lambda kv: (kv[1]["incremental"]["mean"] or -9))
    # 披露滞后要用**财务维度**的中位数，不能对五个维度取中位数：行情/资金是 0–1 天，
    # 混进去会把"财报要等 51 天"稀释成"8 天"，正好把最该被看见的数字抹掉。
    median_lag = pit["lag_days"].get("fundamental", {}).get("median", 0.0)
    pit_line = (f"PIT 侧没有越界记录（{pit['records']} 条）" if pit["violations"] == 0
                else f"PIT 侧有 {pit['violations']} 条越界记录（共 {pit['records']} 条），必须先修数据")
    verdict["summary"] = (
        f"{pit_line}，财报的披露滞后中位数为 "
        f"{median_lag:.0f} 天——"
        f"这就是「按报告期对齐」会提前知道的天数。"
        f"增量侧：{len(winners)}/{len(evaluable)} 个可评估因子的增量 IC 区间不含 0，"
        f"最好是 {fz.FACTORS[best[0]].name}（{best[1]['incremental']['mean']:+.4f}）；"
        f"维度合并后 {len(dim_winners)}/{len(evaluable_dims)} 个可评估维度的组合 ΔIC 区间不含 0。"
        f"但所有区间宽度都在 ±0.03 量级，样本只有 {meta['symbols']} 只票，"
        f"这个结论只够回答「该不该接」，不足以回答「接多少」。"
    )

    inventory: list[dict[str, Any]] = []
    for source in resolve(["baostock_fundamental", "baostock_events",
                           "eastmoney_flows", "csv_drop"]):
        for spec in source.fields:
            inventory.append({
                "source": source.name, "field": spec.name, "dimension": spec.dimension,
                "unit": spec.unit, "pit_basis": spec.pit_basis, "note": spec.note,
            })

    report = {
        "inventory": inventory,
        "meta": {**meta, "rows": len(rows), "days": len({row['date'] for row in rows}),
                 "factors": len(fz.FACTORS), "derived": len(derived)},
        "results": results,
        "dimensions": dimensions,
        "coverage": coverage,
        "pit": pit,
        "verdict": verdict,
    }

    outdir = Path(args.outdir) if args.outdir else Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    (outdir / "report.html").write_text(render_html(report), encoding="utf-8")
    (outdir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[5/5] 完成 → {outdir}", flush=True)
    for key, row in sorted(results.items(), key=lambda kv: -(kv[1]["incremental"]["mean"] or -9)):
        inc = row["incremental"]
        print(f"      {fz.FACTORS[key].name:<12} raw {_signed(inc['mean'])} "
              f"[{_signed(inc['ci_low'])}, {_signed(inc['ci_high'])}] "
              f"Δ组合 {_signed(row['combination']['delta']['mean'])}", flush=True)
    print(f"      越界 {pit['violations']} 条；中位滞后 {median_lag:.0f} 天", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
