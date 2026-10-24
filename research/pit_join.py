"""PIT 对齐引擎：只有这一处允许"按日期取历史值"。

三条规则
--------
1. **以 ``available`` 为准，不以 ``period`` 为准**。财报的报告期是 3/31，但 4 月底
   才公告；用报告期对齐等于提前一个月知道业绩。资金流数据同理：两融是 T+1 披露，
   北向持股是当日盘后，龙虎榜是当日盘后。
2. **同一报告期有多条时取"当时可得的那一版"**。财报会重述（同一 ``period`` 对应
   多个 ``available``），如果一律取最新版就是前视；``AsOfBook`` 在 ``available <= T``
   的记录里取 ``available`` 最大的一条，等价于"T 日能看到的最新一版"。
3. **允许过期上限**。公司停止披露后，三年旧的 ROE 不该继续参与横截面排序；
   ``max_stale`` 一到就返回 ``None``，让缺失如实暴露，而不是用陈旧值填坑。
"""

from __future__ import annotations

import bisect
import statistics
from collections import defaultdict
from datetime import date
from typing import Any, Callable, Iterable, Sequence

Record = dict[str, Any]


def _shift(day: str, offset: int) -> str:
    from datetime import timedelta
    try:
        return (date.fromisoformat(day[:10]) + timedelta(days=offset)).isoformat()
    except ValueError:
        return day


def _days(later: str, earlier: str) -> int:
    try:
        return (date.fromisoformat(later[:10]) - date.fromisoformat(earlier[:10])).days
    except ValueError:
        return 0


class AsOfBook:
    """``(symbol, field)`` → 按 ``available`` 排序的时间序列，支持 as-of 查询。"""

    def __init__(self, records: Iterable[Record] = ()) -> None:
        self._index: dict[tuple[str, str], list[tuple[str, str, Any]]] = defaultdict(list)
        self._built = False
        self.stats = {"records": 0, "fields": 0, "symbols": set()}
        for record in records:
            self.add(record)

    def add(self, record: Record) -> None:
        key = (str(record["symbol"]), str(record["field"]))
        self._index[key].append(
            (str(record["available"])[:10], str(record["period"])[:10], record["value"])
        )
        self._built = False
        self.stats["records"] += 1
        self.stats["symbols"].add(str(record["symbol"]))

    def _ensure(self) -> None:
        if self._built:
            return
        for key, rows in self._index.items():
            # 先按可得日、再按报告期：同一可得日有多期时取较新的一期
            rows.sort(key=lambda item: (item[0], item[1]))
        self._built = True
        self.stats["fields"] = len(self._index)

    # ------------------------------------------------------------------ #
    def at(self, symbol: str, field: str, day: str) -> tuple[Any, str, int] | None:
        """返回 ``(value, available, stale_days)``，没有可用记录时 ``None``。"""
        self._ensure()
        rows = self._index.get((str(symbol), field))
        if not rows:
            return None
        day = str(day)[:10]
        position = bisect.bisect_right(rows, (day, "9999-12-31", None)) - 1
        if position < 0:
            return None
        available, _period, value = rows[position]
        if value is None:
            return None
        return value, available, _days(day, available)

    def value(self, symbol: str, field: str, day: str, *, max_stale: int | None = None) -> Any:
        hit = self.at(symbol, field, day)
        if hit is None:
            return None
        value, _available, stale = hit
        if max_stale is not None and stale > max_stale:
            return None
        return value

    def series(
        self,
        field: str,
        dates: Sequence[str],
        symbols: Sequence[str],
        *,
        max_stale: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        """``{date: {symbol: value}}``，缺失的 (date, symbol) 不出现。"""
        out: dict[str, dict[str, Any]] = {}
        for day in dates:
            row: dict[str, Any] = {}
            for symbol in symbols:
                value = self.value(symbol, field, day, max_stale=max_stale)
                if value is not None:
                    row[symbol] = value
            if row:
                out[day] = row
        return out

    def window_sum(self, symbol: str, field: str, day: str, window_days: int) -> float | None:
        """``(day - window_days, day]`` 内所有记录的求和（分红、龙虎榜净买入都用它）。

        与 ``value`` 的区别：``value`` 只看最新一条，``window_sum`` 要累计窗口内的
        全部事件。窗口按**可得日**切，不按报告期，这样"12 个月股息"是"过去 12 个月
        已经实施的分红"，而不是"报告期落在过去 12 个月的分红"。
        """
        self._ensure()
        rows = self._index.get((str(symbol), field))
        if not rows:
            return None
        day = str(day)[:10]
        right = bisect.bisect_right(rows, (day, "9999-12-31", None))
        if right == 0:
            return None
        left_day = _shift(day, -window_days)
        left = bisect.bisect_right(rows, (left_day, "9999-12-31", None))
        chunk = rows[left:right]
        if not chunk:
            return None
        return float(sum(float(item[2]) for item in chunk))

    def window_count(self, symbol: str, field: str, day: str, window_days: int) -> int:
        self._ensure()
        rows = self._index.get((str(symbol), field))
        if not rows:
            return 0
        day = str(day)[:10]
        right = bisect.bisect_right(rows, (day, "9999-12-31", None))
        left = bisect.bisect_right(rows, (_shift(day, -window_days), "9999-12-31", None))
        return max(0, right - left)

    def latest_with_age(self, symbol: str, field: str, day: str) -> tuple[Any, int] | None:
        hit = self.at(symbol, field, day)
        if hit is None:
            return None
        return hit[0], hit[2]

    def coverage(self, field: str, dates: Sequence[str], symbols: Sequence[str]) -> dict[str, Any]:
        counts = []
        for day in dates:
            counts.append(sum(1 for symbol in symbols if self.value(symbol, field, day) is not None))
        if not counts:
            return {"dates": 0, "mean": 0.0, "min": 0, "max": 0, "first": None}
        covered = [date for date, count in zip(dates, counts) if count > 0]
        return {
            "dates": len(covered),
            "share": round(len(covered) / max(len(dates), 1), 3),
            "mean": round(statistics.fmean(counts), 1),
            "min": min(counts),
            "max": max(counts),
            "first": covered[0] if covered else None,
        }


# --------------------------------------------------------------------------- #
# 派生字段：在同一次披露的"版本"上算，避免混用不同版本的原始值
# --------------------------------------------------------------------------- #


def vintages(records: Sequence[Record]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """``(symbol, period, available)`` → {field: value}。"""
    frames: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(dict)
    for record in records:
        key = (str(record["symbol"]), str(record["period"])[:10], str(record["available"])[:10])
        frames[key][str(record["field"])] = record["value"]
    return dict(frames)


def derive(
    records: Sequence[Record],
    builder: Callable[[str, str, str, dict[str, Any]], dict[str, Any]],
    *,
    dimension: str,
    source: str = "derived",
) -> list[Record]:
    """按披露版本生成派生字段。

    ``builder(symbol, period, available, frame)`` 返回 ``{field: value}``；返回空字典
    表示这一版算不出来（例如亏损股的 BP），此时**不写记录**——缺失要能被上层看见。

    派生只在"这一版可见的原始字段"上做：同一 ``period`` 有两个 ``available`` 时，
    各自算各自的，不前视借用后面那版的数字。
    """
    out: list[Record] = []
    for (symbol, period, available), frame in sorted(vintages(records).items()):
        values = builder(symbol, period, available, frame)
        for field, value in (values or {}).items():
            if value is None:
                continue
            out.append({
                "symbol": symbol,
                "dimension": dimension,
                "field": field,
                "value": value,
                "period": period,
                "available": available,
                "source": source,
            })
    return out


# --------------------------------------------------------------------------- #
# 审计
# --------------------------------------------------------------------------- #


def audit(records: Sequence[Record]) -> dict[str, Any]:
    """PIT 体检：披露滞后、重述次数、可得日越界。

    报告里必须出现"中位披露滞后"——它直接决定"按报告期对齐"会虚增多少 IC。
    """
    lags: dict[str, list[int]] = defaultdict(list)
    revisions: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    violations: list[str] = []
    for record in records:
        period = str(record["period"])[:10]
        available = str(record["available"])[:10]
        lag = _days(available, period)
        # 事件类（业绩预告）本来就发布在报告期结束之前，负滞后是语义正确的
        if lag < 0 and record["dimension"] != "event":
            violations.append(f"{record['symbol']}.{record['field']} period={period} available={available}")
        lags[record["dimension"]].append(lag)
        revisions[(str(record["symbol"]), str(record["field"]), period)].add(available)
    revised = {key: len(value) for key, value in revisions.items() if len(value) > 1}
    return {
        "records": len(records),
        "violations": len(violations),
        "violation_samples": violations[:5],
        "lag_days": {
            dimension: {
                "median": round(statistics.median(values), 1),
                "p90": round(sorted(values)[int(0.9 * (len(values) - 1))], 1),
                "max": max(values),
                "n": len(values),
            }
            for dimension, values in lags.items() if values
        },
        "revised_series": len(revised),
        "revision_share": round(len(revised) / max(len(revisions), 1), 4),
        "max_revisions": max(revised.values()) if revised else 1,
    }
