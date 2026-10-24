"""手工/终端导出的 CSV 落地面：分析师修正、回购、解禁、ETF 份额。

为什么需要这一层
----------------
P2-10 里最值钱的三个字段恰好没有稳定免费源：

- 分析师一致预期修正（Wind/朝阳永续/Choice 才有，免费爬取既不稳定也不合规）；
- 回购公告金额（交易所逐条披露，聚合需要自己维护）；
- 限售解禁明细（可算，但需要股本变动历史）；
- ETF 申购赎回份额（东财只给当前值，没有历史序列）。

与其编一个"看起来有数"的爬虫，不如把**接口留出来**：任何终端导出的 CSV 只要
列名对齐，就能进同一条 PIT 管线。这也是"可插拔"最实际的价值——新增一个维度
不需要改因子代码，只要丢一个文件进 ``.cache/datasets/csv/``。

CSV 约定（表头必须完全一致）::

    symbol,field,period,available,value
    600519,analyst_revision,2024-03-31,2024-04-08,0.021
    600519,buyback_amount,2024-06-30,2024-07-02,150000000
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .base import FetchContext, FieldSpec, Source, register

CSV_DIR = Path(".cache") / "datasets" / "csv"

# 允许从 CSV 进来的字段白名单：字段名 → (维度, 单位, PIT 依据, 说明)
MANUAL_FIELDS: dict[str, tuple[str, str, str, str]] = {
    "analyst_revision": ("growth", "小数", "研报发布日", "一致预期 EPS 的 30 日修正幅度"),
    "analyst_count": ("growth", "人", "研报发布日", "覆盖分析师数，用于置信度加权"),
    "buyback_amount": ("event", "元", "回购公告日", "当日公告的回购金额上限"),
    "unlock_ratio": ("event", "小数", "解禁公告日", "未来 30 日解禁市值/流通市值"),
    "etf_share_change": ("flow", "份", "T+1 披露", "ETF 份额日变化（申赎代理）"),
    "etf_premium": ("flow", "%", "当日收盘", "ETF 折溢价率（申赎压力的日内代理）"),
}

FIELDS = tuple(
    FieldSpec(name=name, dimension=dimension, unit=unit, pit_basis=basis, note=note)
    for name, (dimension, unit, basis, note) in MANUAL_FIELDS.items()
)


@register
class CsvDrop(Source):
    name = "csv_drop"
    dimensions = ("growth", "event", "flow")
    fields = FIELDS
    optional_deps = ()

    def probe(self) -> tuple[bool, str]:
        directory = self._dir(None)
        if not directory.is_dir():
            return False, f"目录不存在：{directory}"
        files = sorted(directory.glob("*.csv"))
        if not files:
            return False, f"{directory} 里没有 CSV"
        return True, f"{len(files)} 个文件"

    def _dir(self, ctx: FetchContext | None) -> Path:
        if ctx is not None:
            return Path(ctx.cache) / "csv"
        return CSV_DIR

    def fetch(self, ctx: FetchContext) -> list[dict[str, Any]]:
        directory = self._dir(ctx)
        if not directory.is_dir():
            ctx.log(f"      CSV 目录不存在，跳过：{directory}")
            return []
        wanted = set(ctx.symbols)
        records: list[dict[str, Any]] = []
        for path in sorted(directory.glob("*.csv")):
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    field = (row.get("field") or "").strip()
                    if field not in MANUAL_FIELDS:
                        continue
                    symbol = (row.get("symbol") or "").strip().zfill(6)
                    if symbol not in wanted:
                        continue
                    period = (row.get("period") or "").strip()[:10]
                    available = (row.get("available") or period).strip()[:10]
                    try:
                        value = float(row.get("value") or "")
                    except ValueError:
                        continue
                    records.append({
                        "symbol": symbol,
                        "dimension": MANUAL_FIELDS[field][0],
                        "field": field,
                        "value": value,
                        "period": period,
                        "available": available,
                        "source": f"csv:{path.name}",
                    })
        ctx.log(f"      CSV 记录 {len(records)} 条")
        return records


def write_template(path: Path | None = None) -> Path:
    """生成一份带示例行的模板，方便终端导出后对齐列名。"""
    target = path or (CSV_DIR / "template.csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "symbol,field,period,available,value\n"
        "600519,analyst_revision,2024-03-31,2024-04-08,0.021\n"
        "600519,buyback_amount,2024-06-30,2024-07-02,150000000\n",
        encoding="utf-8",
    )
    return target
