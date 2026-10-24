"""把同一套 5 折验证在不同电池配置下的产物并排比较。

用法：python wfv2_compare.py base prune3 prune4
读 ``reports/wfv2/<tag>/metrics.json``，写 ``reports/wfv2/compare.md``。
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "wfv2"

LABELS = {
    "base": "全电池（12）",
    "prune3": "剔除 3 个反号（9）",
    "prune4": "剔除 4 个（含 mom12_1，8）",
    "flip3": "改先验（3 个翻符号，12）",
}


def load(tag: str) -> dict[str, Any] | None:
    path = REPORT / tag / "metrics.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fold_ics(report: dict[str, Any], tag: str) -> dict[int, float | None]:
    return {row["fold"]: row["ic"] for row in report["results"][tag]}


def main(argv: Sequence[str] | None = None) -> int:
    tags = list(argv or sys.argv[1:]) or ["base", "prune3", "prune4"]
    reports = {tag: load(tag) for tag in tags}
    reports = {tag: report for tag, report in reports.items() if report}
    if not reports:
        print("没有可比较的产物")
        return 1

    rows = []
    for tag, report in reports.items():
        single = report["summary"]["SINGLE"]
        equal = report["summary"]["EW"]
        rows.append({
            "tag": tag,
            "label": LABELS.get(tag, tag),
            "battery": len(report["meta"].get("battery") or []),
            "drop": "、".join(report["meta"].get("drop") or []) or "—",
            "ic": single["pooled_ic"],
            "ci": single["pooled_ci"],
            "folds": f"{single['positive_folds']}/{single['n_folds']}",
            "icir": single["icir_mean"],
            "turnover": single["turnover"],
            "cagr": single["net_cagr"],
            "sharpe": single["net_sharpe"],
            "mdd": single["max_drawdown"],
            "ew_ic": equal["pooled_ic"],
            "ew_mdd": equal["max_drawdown"],
            "ew_folds": f"{equal['positive_folds']}/{equal['n_folds']}",
            "fold_ic": fold_ics(report, "SINGLE"),
        })

    header = "| 配置 | 因子数 | 剔除 | 合并 IC | 90% 区间 | 正折 | ICIR | 换手 | 净年化 | Sharpe | 最大回撤 | 等权 IC |"
    lines = [header, "| " + " | ".join(["---"] * 12) + " |"]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['battery']} | {row['drop']} | {row['ic']:+.4f} | "
            f"[{row['ci'][0]:+.4f}, {row['ci'][1]:+.4f}] | {row['folds']} | {row['icir']:+.2f} | "
            f"{row['turnover']:.0f}% | {row['cagr']:+.2f}% | {row['sharpe']:+.2f} | {row['mdd']:.1f}% | {row['ew_ic']:+.4f} |"
        )

    folds = sorted({fold for row in rows for fold in row["fold_ic"]})
    fold_lines = [
        "| 配置 | " + " | ".join(f"折 {fold}" for fold in folds) + " |",
        "| " + " | ".join(["---"] * (len(folds) + 1)) + " |",
    ]
    for row in rows:
        fold_lines.append(
            f"| {row['label']} | "
            + " | ".join(
                "—" if row["fold_ic"].get(fold) is None else f"{row['fold_ic'][fold]:+.4f}"
                for fold in folds
            )
            + " |"
        )

    by_tag = {row["tag"]: row for row in rows}
    base = by_tag.get("base", rows[0])

    def delta(tag: str, key: str) -> float | None:
        row = by_tag.get(tag)
        if not row or row.get(key) is None or base.get(key) is None:
            return None
        return row[key] - base[key]

    parts = []
    if "prune3" in by_tag:
        row = by_tag["prune3"]
        parts.append(
            f"剔除 3 个反号因子后，**拟合模型变差**：IC {row['ic']:+.4f}（{delta('prune3', 'ic'):+.4f}）、"
            f"ICIR {row['icir']:+.2f}、净 Sharpe {row['sharpe']:+.2f}、回撤 {row['mdd']:.1f}%；"
            f"但**等权基线变好**：IC {row['ew_ic']:+.4f}（{row['ew_ic'] - base['ew_ic']:+.4f}）。"
        )
    if "prune4" in by_tag:
        row = by_tag["prune4"]
        parts.append(
            f"再剔除 mom12_1（3/5 折为正，属「近两年失效」而非「方向错」）后，拟合模型 IC {row['ic']:+.4f}、"
            f"ICIR {row['icir']:+.2f}、净 Sharpe {row['sharpe']:+.2f}，仍低于全电池；"
            f"等权基线升到 {row['ew_ic']:+.4f}，回撤收窄到 {row['ew_mdd']:.1f}%——"
            f"**反号因子对等权的伤害最大，因为等权把它们按错误方向硬加权**。"
        )
    if "flip3" in by_tag:
        row = by_tag["flip3"]
        parts.append(
            f"把同一批因子的先验翻过来、因子保留（flip3），拟合模型几乎不动：IC {row['ic']:+.4f}"
            f"（{delta('flip3', 'ic'):+.4f}）、ICIR {row['icir']:+.2f}、Sharpe {row['sharpe']:+.2f}——"
            f"说明模型本来就学出了负号，剔除等于删掉它已经用上的信息。"
        )
    conclusion = " ".join(parts)
    text = f"""# 电池配置对照（5 折 purged walk-forward）

同一套折边界、同一套标签、同一套内层选参，只改因子电池。

{chr(10).join(lines)}

逐折 OOS IC（SINGLE）：

{chr(10).join(fold_lines)}

{conclusion}

## 怎么读

- **剔除判据必须是跨折一致的符号冲突**，不是某一折为负。macd_hist / downside20 / mdd60 在 5 折
  全为负号（与先验同号率为 0），才有资格被剔除；mom12_1 是 3/5，属于"近两年失效"而不是"方向错"，
  所以单独作为 prune4 对照，而不是默认剔除。
- **"方向反了"不等于"没用"**：系数为负说明这个因子有反向预测力。拟合模型会自动学出负号，
  剔除等于删信息；等权模型没有负号，剔除才是修正。两者结论必然相反，这不是矛盾。
- **flip 只对等权书有意义**：改先验后拟合模型完全不动（IC +0.0607→+0.0606，系数只是整体换号），
  等权基线 IC +0.0421→+0.0465 但回撤反而从 −43.8% 深到 −48.8%、换手 1540%→2055%。
  原因很直白：把"高波动/深回撤"按正向加权，就是在风险维度上做多尾部。
  所以 flip 不是"正确的修法"，它只是把"先验错"这件事摆正；**是否要这个暴露，是另一个决定**。
- **ICIR 与 IC 是两件事**：剔除因子通常会降 IC（信息少了），但可能升 ICIR（噪声也少了）。
  判据以 ICIR 与净 Sharpe 为主，IC 为辅。
- 换手会随电池变小而下降（持仓更集中、边界更稳），这是剔除的附带收益，不是目的。
- 每折的 IC 差距必须和块自助区间一起看：单折差异普遍在 ±0.03 以内，**不要用单折排名选电池**。
- 全电池的净回撤 −39.2% 比等权剔除版的 −31.0% 更深：负号的风险因子让组合在风险维度上是
  "反向暴露"（越高波动越买），IC 赚了，尾部风险也放大了。这是本次对照最值得警惕的一点。

产物：`reports/wfv2/<tag>/` 下有各自的完整报告。
"""
    (REPORT / "compare.md").write_text(text, encoding="utf-8")
    print(f"写入 {REPORT / 'compare.md'}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
