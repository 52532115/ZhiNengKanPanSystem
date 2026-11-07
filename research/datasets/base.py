"""研究层数据源插件协议：所有外部数据都必须落成同一种长表记录。

为什么要有这层
--------------
P2-10 要接的维度（财务、资金、事件）各有一个免费源、各自一套字段名，如果直接在
因子代码里``import akshare``再就地清洗，会出现三个问题：口径散落、PIT 无法审计、
生产层被第三方依赖污染。这里把"取数"与"用数"切开：

- 每个数据源是一个 ``Source`` 子类，只负责把外部格式翻译成**规范长表**；
- 规范长表里每条记录都带 ``period``（所属报告期/交易日）与 ``available``
  （公开可得日），PIT 对齐由 ``research.pit_join`` 统一做，因子层不允许自己查表；
- 源可以声明 ``optional_deps``，缺依赖时 ``probe()`` 返回原因而不是抛异常，
  这样整条流水线在没装第三方库的机器上仍然能跑其余维度。

规范记录（长表，一条记录 = 一个标的的一个字段在一个报告期的一次披露）
--------------------------------------------------------------------
```python
{
  "symbol":    "600519",       # 6 位本地代码
  "dimension": "fundamental",  # fundamental | event | flow | market
  "field":     "roe",          # 规范字段名（见各 Source.fields）
  "value":     0.3617,         # 数值；类目字段可为 str
  "period":    "2023-12-31",   # 数据所属期（statDate / 交易日）
  "available": "2024-04-03",   # 公开可得日（pubDate / 公告日 / 披露日）
  "source":    "baostock:query_profit_data",
}
```

``available >= period`` 是硬约束：违反说明抓取层把"报告期"当成了"可得日"，
这类记录会被 ``validate`` 直接拒绝，而不是悄悄流进回测。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

SCHEMA_VERSION = 1
REQUIRED_KEYS = ("symbol", "dimension", "field", "value", "period", "available", "source")


class SourceError(RuntimeError):
    """数据源不可用或返回结构不符合约定。"""


@dataclass(frozen=True)
class FieldSpec:
    """一个规范字段的元信息。``pit_basis`` 说明这个字段的可得日以什么为准。"""

    name: str
    dimension: str
    kind: str = "numeric"          # numeric | date | category
    unit: str = ""
    pit_basis: str = ""            # 例：'公告日 pubDate' / '除权除息日' / 'T+1 披露'
    note: str = ""


@dataclass
class FetchContext:
    symbols: Sequence[str]
    start: str
    end: str
    cache: Path
    pause: float = 0.0
    log: Callable[[str], None] = print
    limit: int | None = None
    extra: dict[str, Any] = dc_field(default_factory=dict)


class Source:
    """数据源基类。子类只需实现 ``probe`` 与 ``fetch``。"""

    name: str = ""
    dimensions: tuple[str, ...] = ()
    fields: tuple[FieldSpec, ...] = ()
    optional_deps: tuple[str, ...] = ()

    # ------------------------------------------------------------------ #
    def probe(self) -> tuple[bool, str]:
        """返回 (是否可用, 原因)。默认只看 optional_deps 能否导入。"""
        for module in self.optional_deps:
            try:
                __import__(module)
            except Exception as exc:  # pragma: no cover - 取决于环境
                return False, f"缺少依赖 {module}: {exc}"
        return True, "ok"

    def fetch(self, ctx: FetchContext) -> list[dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    def describe(self) -> dict[str, Any]:
        ok, reason = self.probe()
        return {
            "name": self.name,
            "dimensions": list(self.dimensions),
            "available": ok,
            "reason": reason,
            "fields": [
                {"name": spec.name, "dimension": spec.dimension, "kind": spec.kind,
                 "unit": spec.unit, "pit_basis": spec.pit_basis, "note": spec.note}
                for spec in self.fields
            ],
        }


# --------------------------------------------------------------------------- #
# 注册表
# --------------------------------------------------------------------------- #

_REGISTRY: dict[str, type[Source]] = {}


def register(cls: type[Source]) -> type[Source]:
    if not cls.name:
        raise ValueError("Source 必须有 name")
    _REGISTRY[cls.name] = cls
    return cls


def catalog() -> dict[str, dict[str, Any]]:
    """所有已注册源的能力清单（供文档与报告使用，不触发取数）。"""
    return {name: cls().describe() for name, cls in sorted(_REGISTRY.items())}


def resolve(names: Iterable[str] | None = None) -> list[Source]:
    if names is None:
        return [cls() for _, cls in sorted(_REGISTRY.items())]
    out: list[Source] = []
    for name in names:
        if name not in _REGISTRY:
            raise SourceError(f"未注册的数据源：{name}（可选 {', '.join(sorted(_REGISTRY))}）")
        out.append(_REGISTRY[name]())
    return out


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #


def validate(records: Sequence[dict[str, Any]], *, strict: bool = True) -> list[str]:
    """结构 + PIT 不变量检查，返回问题列表（strict 时直接抛）。

    检查项：
    1. 必备键齐全；
    2. ``available >= period``（否则就是把报告期当可得日）。**事件类例外**：
       业绩预告在报告期结束前就发布，``available < period`` 是正常语义（预告的是
       未来那一期），所以 ``dimension == "event"`` 不做这条检查；
    3. 数值字段必须有限，``None`` 直接拒绝（缺失就**不要写这条记录**，
       让上层用"没有记录"表示缺失，而不是用 0 或者 NaN 表示）；
    4. ``(symbol, field, period, available)`` 不重复。
    """
    problems: list[str] = []
    seen: set[tuple[str, str, str, str]] = set()
    for index, record in enumerate(records):
        missing = [key for key in REQUIRED_KEYS if key not in record]
        if missing:
            problems.append(f"#{index} 缺字段 {missing}")
            continue
        period, available = str(record["period"]), str(record["available"])
        if len(period) < 10 or len(available) < 10:
            problems.append(f"#{index} 日期格式不对：period={period} available={available}")
        if available < period and record["dimension"] != "event":
            problems.append(
                f"#{index} {record['symbol']}.{record['field']} 可得日 {available} 早于报告期 {period}"
            )
        value = record["value"]
        if isinstance(value, bool):
            problems.append(f"#{index} value 不能是 bool")
        elif isinstance(value, (int, float)):
            if not math.isfinite(float(value)):
                problems.append(f"#{index} value 非有限数")
        elif not isinstance(value, str):
            problems.append(f"#{index} value 类型不支持：{type(value).__name__}")
        key = (str(record["symbol"]), str(record["field"]), period, available)
        if key in seen:
            problems.append(f"#{index} 重复记录 {key}")
        seen.add(key)
    if problems and strict:
        head = "\n".join(problems[:8])
        raise SourceError(f"数据校验失败（{len(problems)} 处）：\n{head}")
    return problems


def compact(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """去掉空值、统一代码与日期格式，便于落盘与比较。"""
    out: list[dict[str, Any]] = []
    for record in records:
        value = record.get("value")
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            continue
        item = {
            "symbol": str(record["symbol"]).zfill(6),
            "dimension": record["dimension"],
            "field": record["field"],
            "value": value,
            "period": str(record["period"])[:10],
            "available": str(record["available"])[:10],
            "source": record["source"],
        }
        if record.get("vintage"):
            item["vintage"] = str(record["vintage"])[:10]
        out.append(item)
    return out
