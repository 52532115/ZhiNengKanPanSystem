"""成分审计：只用 PIT 成分快照回答"幸存者偏差有多大"。

这一步不依赖日线，所以在限流严重、日线还没抓齐的时候也能先出结论；
它同时给池层验证提供分母——到底有多少标的在样本期内被换出/退市。

命令行::

    python -m research.pit_audit --version pit-v1
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from .manifest import ROOT, Dataset, latest_version
from .pit import CSRC_L1, industry_key

REPORT_DIR = ROOT / "reports" / "pool"
INDEX_NAMES = {"000300": "沪深300", "000905": "中证500"}


def audit(version: str, *, root: Path | None = None, since: str | None = None) -> dict[str, Any]:
    dataset = Dataset.open(version, root=root)
    membership: dict[str, dict[str, list[str]]] = dataset.read_table("membership")
    days = [day for day in sorted(membership) if since is None or day >= since]
    industry = dataset.read_table("industry") if (dataset.root / "industry.json").is_file() else {}
    report: dict[str, Any] = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "version": version,
        "since": since,
        "verify": dataset.verify(),
        "snapshots": len(days),
        "window": [days[0], days[-1]] if days else [],
        "indexes": {},
    }
    for index, name in INDEX_NAMES.items():
        first = set(membership[days[0]].get(index, []))
        last = set(membership[days[-1]].get(index, []))
        union = {code for day in days for code in membership[day].get(index, [])}
        removed = first - last
        added = last - first
        # 每次快照的换手：与上一期相比进出多少
        churn = []
        for previous_day, day in zip(days, days[1:]):
            previous = set(membership[previous_day].get(index, []))
            current = set(membership[day].get(index, []))
            if previous and current:
                churn.append(len(current - previous) / len(current) * 100)
        # 逐年净变动
        yearly: dict[str, dict[str, int]] = {}
        for day in days:
            year = day[:4]
            current = set(membership[day].get(index, []))
            entry = yearly.setdefault(year, {"names": len(current), "added": 0, "removed": 0})
            entry["names"] = len(current)
        for previous_day, day in zip(days, days[1:]):
            previous = set(membership[previous_day].get(index, []))
            current = set(membership[day].get(index, []))
            entry = yearly.setdefault(day[:4], {"names": len(current), "added": 0, "removed": 0})
            entry["added"] += len(current - previous)
            entry["removed"] += len(previous - current)
        # 只出现过一次就再没回来的标的：多半是退市/被并购，而不是指数调仓
        occurrences: dict[str, int] = {}
        last_seen: dict[str, str] = {}
        for day in days:
            for code in membership[day].get(index, []):
                occurrences[code] = occurrences.get(code, 0) + 1
                last_seen[code] = day
        one_shot = [code for code, count in occurrences.items()
                    if count == 1 and last_seen[code] < days[-1]]
        composition = _composition(membership, days, index, industry)
        report["indexes"][index] = {
            "composition": composition,
            "name": name,
            "start": days[0],
            "end": days[-1],
            "start_size": len(first),
            "end_size": len(last),
            "union": len(union),
            "removed": len(removed),
            "added": len(added),
            "removed_share": round(len(removed) / max(len(first), 1) * 100, 1),
            "churn_mean": round(statistics.fmean(churn), 2) if churn else None,
            "churn_max": round(max(churn), 1) if churn else None,
            "one_shot_names": len(one_shot),
            "yearly": OrderedDict(sorted(yearly.items())),
        }
    return report


def _composition(
    membership: dict[str, dict[str, list[str]]],
    days: list[str],
    index: str,
    industry: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """行业覆盖：首末快照的行业数、前三集中度、HHI，以及占比最高的行业。"""
    def snap(day: str) -> dict[str, Any]:
        codes = membership.get(day, {}).get(index, [])
        counts: dict[str, int] = {}
        for code in codes:
            key = industry_key(code, industry)
            counts[key] = counts.get(key, 0) + 1
        total = len(codes) or 1
        shares = sorted((value / total for value in counts.values()), reverse=True)
        top = sorted(counts.items(), key=lambda pair: -pair[1])[:5]
        return {
            "snapshot": day,
            "industries": len(counts),
            "top3_share": round(sum(shares[:3]) * 100, 1),
            "hhi": round(sum(value ** 2 for value in shares), 4),
            "top": [
                {"industry": f"{CSRC_L1.get(key[0], key[0])}·{key}",
                 "count": count, "share": round(count / total * 100, 1)}
                for key, count in top
            ],
        }

    if not days or not industry:
        return {}
    return {"first": snap(days[0]), "last": snap(days[-1])}


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 指数成分审计：幸存者偏差有多大",
        "",
        f"数据版本 `{report['version']}`（校验 {'通过' if report['verify']['ok'] else '失败'}），"
        f"{report['snapshots']} 期月末快照，{report['window'][0]} → {report['window'][1]}"
        + (f"（自 {report['since']} 起）" if report.get("since") else "") + "。",
        "",
        "| 指数 | 期初成分 | 期末成分 | 期间出现过 | 期初被换出 | 换出占比 | 单期换手均值 | 只出现一期 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, item in report["indexes"].items():
        lines.append(
            f"| {item['name']} | {item['start_size']} | {item['end_size']} | {item['union']} | "
            f"{item['removed']} | {item['removed_share']}% | {item['churn_mean']}% | "
            f"{item['one_shot_names']} |"
        )
    lines += [
        "",
        "## 行业分布（首末快照）",
        "",
        "| 指数 | 快照 | 行业数 | 前三行业占比 | HHI | 占比最高的三个行业 |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for index, item in report["indexes"].items():
        composition = item.get("composition") or {}
        for tag in ("first", "last"):
            entry = composition.get(tag)
            if not entry:
                continue
            top = "、".join(f"{row['industry']} {row['share']}%" for row in entry["top"][:3])
            lines.append(
                f"| {item['name']} | {entry['snapshot']} | {entry['industries']} | "
                f"{entry['top3_share']}% | {entry['hhi']} | {top} |"
            )
    lines += [
        "",
        "读法：「期间出现过」是并集，「期初被换出」是期初那批里今天已不在名单的数量。"
        "用今天的成分回看历史，等于把后者整段删除——这一列就是幸存者偏差的直接规模。",
        "",
        "## 逐年成分变化",
        "",
    ]
    lines += ["调入与调出每年相等，是因为成分数恒定（300/500），换手体现的是"
              "「进一个、出一个」，不是净增。", ""]
    for index, item in report["indexes"].items():
        lines += [f"### {item['name']}", "",
                  "| 年份 | 成分数 | 当年新进 | 当年退出 |", "| --- | ---: | ---: | ---: |"]
        for year, entry in item["yearly"].items():
            lines.append(f"| {year} | {entry['names']} | {entry['added']} | {entry['removed']} |")
        lines.append("")
    return "\n".join(lines)


def render_preview(report: dict[str, Any]) -> str:
    """把审计结果渲染成可直接打开的独立页面。"""
    lines = render_markdown(report).splitlines()
    html: list[str] = []
    in_table = False
    for line in lines:
        if line.startswith("# "):
            html.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            html.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            html.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("| ---"):
            continue
        elif line.startswith("| "):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
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
<title>指数成分审计 · 幸存者偏差</title>
<style>
  body {{ margin:0; background:#f4f2eb; color:#16221d;
    font:15px/1.7 -apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; }}
  .wrap {{ max-width:1040px; margin:0 auto; padding:34px 22px 60px; }}
  .panel {{ background:#fffdf7; border:1px solid #d8ddd6; border-radius:10px; padding:26px;
    box-shadow:0 18px 50px rgba(31,48,40,.08); }}
  h1 {{ font-size:22px; margin:0 0 14px; }}
  h2 {{ font-size:15px; margin:26px 0 8px; color:#176b4d; }}
  h3 {{ font-size:13px; margin:18px 0 6px; color:#3d4a44; }}
  table {{ width:100%; border-collapse:collapse; font-size:12.5px; margin:8px 0 4px; }}
  td {{ padding:6px 9px; border-bottom:1px solid #e6e9e2; white-space:nowrap; }}
  tr:first-child td {{ color:#6e7b74; font-weight:600; background:rgba(23,107,77,.05); }}
  p {{ margin:6px 0; color:#3d4a44; }}
</style></head>
<body><div class="wrap"><div class="panel">{''.join(html)}</div></div></body></html>"""


def write_reports(report: dict[str, Any], outdir: Path = REPORT_DIR) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": outdir / "pit_audit.json",
        "markdown": outdir / "pit_audit.md",
        "preview": outdir / "pit_audit_preview.html",
    }
    paths["json"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["markdown"].write_text(render_markdown(report), encoding="utf-8")
    paths["preview"].write_text(render_preview(report), encoding="utf-8")
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PIT 成分审计（不依赖日线）")
    parser.add_argument("--version", default=None)
    parser.add_argument("--since", default=None, help="只看这个日期之后的快照（对齐回测区间）")
    parser.add_argument("--outdir", default=str(REPORT_DIR))
    args = parser.parse_args(argv)
    version = args.version or latest_version()
    if not version:
        print("没有可用数据版本，先跑 python -m research.fetch", file=sys.stderr)
        return 2
    report = audit(version, since=args.since)
    paths = write_reports(report, Path(args.outdir))
    for index, item in report["indexes"].items():
        print(f"  {item['name']}：期初 {item['start_size']} → 期末 {item['end_size']}，"
              f"换出 {item['removed']} 只（{item['removed_share']}%），并集 {item['union']} 只")
    for key, path in paths.items():
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
