"""研究层数据集：可插拔数据源集合。

对外只有三个入口：

```python
from research.datasets import catalog, resolve, collect

catalog()                      # 所有已注册源的能力与可用性（不取数）
collect(names, ctx)            # 按名单取数 → 规范长表 + 每个源的取数报告
```

新增一个维度只需要：写一个 ``Source`` 子类并 ``@register``，在本文件底部 import
一次。因子层不用改——它只认规范长表里的 ``field`` 名。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

from .base import (
    FetchContext,
    FieldSpec,
    Source,
    SourceError,
    catalog,
    compact,
    register,
    resolve,
    validate,
)

# 注册顺序 = 报告里的展示顺序；import 即注册
from . import baostock_fundamental  # noqa: F401,E402
from . import baostock_events       # noqa: F401,E402
from . import eastmoney_flows       # noqa: F401,E402
from . import csv_drop              # noqa: F401,E402


def collect(
    names: Sequence[str] | None,
    ctx: FetchContext,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """按名单依次取数，返回 ``{"records": [...], "report": {...}}``。

    单个源失败不阻断其余源：报告里记 ``status=error`` 与原因，让"这一维没跑到"
    在结论里可见，而不是让整条流水线静默降级。
    """
    report: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for source in resolve(names):
        ok, reason = source.probe()
        entry: dict[str, Any] = {"available": ok, "reason": reason, "dimensions": list(source.dimensions)}
        if not ok:
            entry["status"] = "skipped"
            report[source.name] = entry
            ctx.log(f"  - {source.name}: 跳过（{reason}）")
            continue
        ctx.log(f"  - {source.name}: 取数中…")
        try:
            payload = source.fetch(ctx)
            problems = validate(payload, strict=strict)
            cleaned = compact(payload)
            entry.update({"status": "ok", "records": len(cleaned), "problems": len(problems)})
            records.extend(cleaned)
        except Exception as exc:
            entry.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        report[source.name] = entry
    return {"records": records, "report": report}


def group(records: Iterable[dict[str, Any]], key: str = "dimension") -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        out.setdefault(record[key], []).append(record)
    return out


__all__ = [
    "FetchContext", "FieldSpec", "Source", "SourceError",
    "catalog", "collect", "compact", "group", "register", "resolve", "validate",
    "write_template",
]
from .csv_drop import write_template  # noqa: E402
