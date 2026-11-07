"""PIT 股票池的唯一入口：按日取成分、按日取可交易集合、按日取字段。

为什么要有这一层
----------------
之前的研究脚本各自从 ``.cache/fields/stocks`` 读"当前成分的 60 只"，再拿它回跑
2018 年——等于只研究活到今天的赢家。池子错在入口，后面所有 IC / 净值都跟着错。
这个模块把"当日池子"收敛成一处：

- ``members(day)``：当日真实成分（沪深300 ∪ 中证500，来自 baostock 的按期名单）；
- ``tradable(day)``：在成分基础上剔 ST / 停牌 / 涨跌停 / 次新 / 流动性枯竭；
- ``fields(day)``：行业（证监会 L1）、自由流通市值代理、ST、停牌、上市天数，
  中性化与可交易性过滤都从这里取，不再各算一套；
- ``histories()``：把日线转成回测层认识的格式（价格全部后复权）。

三条硬约束
----------
1. **按日取名单**：``MembershipBook`` 只用快照日 ≤ 目标日的最近一期，不偷看下一期。
2. **退市股留在池子里**：只要当日有行情就保留，否则幸存者偏差会被"剔除退市"这个
   动作悄悄做掉；要量化偏差时用 ``exclude_delisted=True`` 单独跑一遍。
3. **字段缺失要说出来**：``audit()`` 报告行业/市值/ST/停牌的缺失比例，缺失的标的
   在分层时落到 ``UNKNOWN``，不静默按 0 处理。
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from . import clean, pit
from .manifest import CACHE_ROOT, ROOT, Dataset, latest_version

DEFAULT_INDEXES = ("000300", "000905")
MIN_LISTING_DAYS = pit.MIN_LISTING_DAYS
MIN_AMOUNT = pit.MIN_AMOUNT


def to_history(symbol: str, bars: Sequence[dict[str, Any]], name: str = "") -> dict[str, Any]:
    """清洗后的 bar → 回测层认识的 history 结构（价格全部为后复权）。"""
    rows = []
    for bar in bars:
        close = float(bar["close"])
        volume = bar.get("volume_adj")
        if volume is None:
            volume = bar.get("volume") or 0.0
        ret = bar.get("ret")
        rows.append(
            {
                "date": bar["date"],
                "open": float(bar.get("open") or close),
                "close": close,
                "high": float(bar.get("high") or close),
                "low": float(bar.get("low") or close),
                "volume": float(volume),
                "amount": float(bar.get("amount") or 0.0),
                "turnover": float(bar.get("turnover") or 0.0),
                "pct_change": (ret or 0.0) * 100,
                "change": 0.0,
            }
        )
    return {
        "symbol": symbol,
        "name": name or symbol,
        "asset_type": "stock",
        "data_source": "baostock-hfq",
        "rows": rows,
    }


class PitUniverse:
    """一个数据版本对应的 PIT 池子（懒加载 + 缓存按日截面）。"""

    def __init__(
        self,
        version: str | None = None,
        *,
        root: Path | None = None,
        indexes: Sequence[str] = DEFAULT_INDEXES,
    ) -> None:
        self.version = version or latest_version()
        if not self.version:
            raise FileNotFoundError("没有可用的数据版本，先跑 python -m research.fetch")
        self.dataset = Dataset.open(self.version, root=root)
        self.indexes = tuple(indexes)
        self._bars: dict[str, list[dict[str, Any]]] | None = None
        self._industry: dict[str, dict[str, Any]] | None = None
        self._book: pit.MembershipBook | None = None
        self._calendar: list[str] | None = None
        self._slices: dict[str, dict[str, dict[str, Any]]] = {}
        self._members: dict[tuple[str, str], set[str]] = {}

    # ------------------------------------------------------------------ #
    # 基础数据
    # ------------------------------------------------------------------ #
    @property
    def bars(self) -> dict[str, list[dict[str, Any]]]:
        if self._bars is None:
            self._bars = clean.load_universe_bars(self.dataset)
        return self._bars

    @property
    def industry(self) -> dict[str, dict[str, Any]]:
        if self._industry is None:
            path = self.dataset.root / "industry.json"
            self._industry = (
                json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            )
        return self._industry

    @property
    def book(self) -> pit.MembershipBook:
        if self._book is None:
            path = self.dataset.root / "membership.json"
            payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            self._book = pit.MembershipBook(payload)
        return self._book

    @property
    def calendar(self) -> list[str]:
        if self._calendar is None:
            path = self.dataset.root / "calendar.json"
            self._calendar = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
        return self._calendar

    def snapshot(self, day: str) -> dict[str, dict[str, Any]]:
        """当日截面（有行情且未停牌）：行业无关的原始字段。"""
        if day not in self._slices:
            self._slices[day] = clean.panel_slice(self.bars, day)
        return self._slices[day]

    # ------------------------------------------------------------------ #
    # 成分与可交易集合
    # ------------------------------------------------------------------ #
    def members(self, day: str, *, kind: str = "union") -> set[str]:
        """当日真实成分；``kind`` 取 ``union`` / ``hs300`` / ``zz500`` / ``all``。"""
        key = (day, kind)
        if key in self._members:
            return self._members[key]
        if kind == "all":                       # 当日全市场（近似：有行情就算）
            result = set(self.snapshot(day))
        elif kind == "hs300":
            result = set(self.book.members(day, "000300"))
        elif kind == "zz500":
            result = set(self.book.members(day, "000905"))
        else:
            result = set()
            for index in self.indexes:
                result |= set(self.book.members(day, index))
        result &= set(self.bars) if kind != "all" else result
        self._members[key] = result
        return result

    def tradable(
        self,
        day: str,
        *,
        kind: str = "union",
        exclude_st: bool = True,
        exclude_limit: bool = False,
        min_listing_days: int = MIN_LISTING_DAYS,
        min_amount: float = MIN_AMOUNT,
        exclude_delisted: bool = False,
        alive_end: set[str] | None = None,
    ) -> list[str]:
        """当日池子：成分 → 剔 ST / 停牌 / 涨跌停 / 次新 / 低流动性。"""
        snapshot = self.snapshot(day)
        pool = pit.build_pool(
            day,
            snapshot=snapshot,
            members=self.members(day, kind=kind),
            industry=self.industry,
            exclude_st=exclude_st,
            exclude_limit=exclude_limit,
            min_listing_days=min_listing_days,
            min_amount=min_amount,
            exclude_delisted=(lambda code: code not in (alive_end or set())) if exclude_delisted else None,
        )
        return sorted(item["symbol"] for item in pool)

    def members_fn(self, *, kind: str = "union") -> Callable[[str], set[str]]:
        """给 ``protocol`` / ``xsec`` 用的回调：``date -> 当日允许持有的标的``。"""

        def fn(day: str) -> set[str]:
            return self.members(day, kind=kind)

        return fn

    # ------------------------------------------------------------------ #
    # 字段
    # ------------------------------------------------------------------ #
    def fields(self, day: str) -> dict[str, dict[str, Any]]:
        """当日字段表：行业、自由流通市值、ST、停牌、涨跌停、上市天数。

        市值用 ``成交额 / 换手率`` 反推自由流通市值（元），缺失则为 ``None``；
        停牌标的当日不在 ``panel_slice`` 里，因此 ``fields`` 天然只含可交易标的，
        停牌信息由 ``field_audit`` 从原始 bar 单独统计。
        """
        snapshot = self.snapshot(day)
        table: dict[str, dict[str, Any]] = {}
        for code, info in snapshot.items():
            entry = self.industry.get(code) or {}
            table[code] = {
                "symbol": code,
                "industry": pit.industry_key(code, self.industry),
                "industry_l1": (entry.get("industry") or "")[:1] or "?",
                "name": entry.get("name") or code,
                "mcap": info["mcap"],
                "is_st": info["is_st"],
                "listing_days": info["listing_days"],
                "at_limit_up": info["at_limit_up"],
                "at_limit_down": info["at_limit_down"],
                "turnover": info["turnover"],
                "amount": info["amount"],
                "board": info["board"],
            }
        return table

    def alive_at_end(self, *, min_symbols: int = 50) -> set[str]:
        """样本末期仍在市（未退市/未被并购）的标的。

        优先用抓取时存下的全市场名单（``alive.json``，一次 ``query_all_stock``）；
        没有这份名单时才退回"末期仍有日线"的近似——但那个近似会把"名单里有、
        只是没抓日线"的票误判成退市，所以覆盖不足时宁可返回空集。
        """
        path = self.dataset.root / "alive.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            symbols = payload.get("symbols") or []
            if symbols:
                return set(symbols)
        if len(self.bars) < min_symbols:
            return set()
        ends = {symbol: rows[-1]["date"] for symbol, rows in self.bars.items() if rows}
        if not ends:
            return set()
        last = max(ends.values())
        return {symbol for symbol, end in ends.items() if end >= last[:4] + "-06-01"}

    # ------------------------------------------------------------------ #
    # 输出
    # ------------------------------------------------------------------ #
    def histories(
        self,
        codes: Iterable[str] | None = None,
        *,
        min_rows: int = 250,
    ) -> list[dict[str, Any]]:
        """转成回测层的 history 列表；``min_rows`` 以下的不进池（预热不足）。"""
        picked = sorted(codes) if codes is not None else sorted(self.bars)
        result = []
        for code in picked:
            rows = self.bars.get(code)
            if not rows or len(rows) < min_rows:
                continue
            name = (self.industry.get(code) or {}).get("name") or code
            result.append(to_history(code, rows, name))
        return result

    def write_cache(self, outdir: Path, *, min_rows: int = 250) -> dict[str, Any]:
        """把 PIT 面板写成 ``.cache/fields/stocks_pit`` 那种一标的一文件的格式。

        这样 ``validate_fields`` / ``factor_study`` / ``wfv2_study`` 只要换缓存目录就能
        用上 PIT 池，不必每个脚本各写一遍加载逻辑。
        """
        outdir.mkdir(parents=True, exist_ok=True)
        written = 0
        for history in self.histories(min_rows=min_rows):
            entry = self.industry.get(history["symbol"]) or {}
            payload = dict(history)
            payload["industry"] = pit.industry_key(history["symbol"], self.industry)
            payload["industry_name"] = entry.get("industry") or ""
            (outdir / f"{history['symbol']}.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            written += 1
        members_path = outdir.parent / "members.json"
        members_path.write_text(
            json.dumps(
                {day: sorted(self.members(day)) for day in self.book.days},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return {"symbols": written, "snapshots": len(self.book.days),
                "members_snapshots": len(self.book.days),
                "dir": str(outdir), "members": str(members_path)}


def members_loader(path: Path) -> Callable[[str], set[str]]:
    """读 ``members.json``：快照日 ≤ 目标日的最近一期成分（与 MembershipBook 同规则）。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    days = sorted(payload)

    def fn(day: str) -> set[str]:
        low, high, found = 0, len(days) - 1, None
        while low <= high:
            middle = (low + high) // 2
            if days[middle] <= day:
                found = days[middle]
                low = middle + 1
            else:
                high = middle - 1
        return set(payload[found]) if found else set()

    return fn


PIT_CACHE = ROOT / ".cache" / "fields" / "stocks_pit"


def prepare_cache(
    version: str | None = None,
    *,
    outdir: Path | None = None,
    refresh: bool = False,
    min_rows: int = 250,
) -> dict[str, Any]:
    """把 PIT 池落成 ``.cache/fields/stocks_pit``，供各研究脚本直接换目录使用。

    返回 ``{"dir", "members", "symbols", "snapshots", "version"}``；``refresh=False``
    且目录已存在时直接复用，避免每次跑研究都重写 400 多个 JSON。
    """
    target = Path(outdir) if outdir else PIT_CACHE
    members_path = target.parent / "members.json"
    stamp_path = target.parent / "pit_stamp.json"
    if not refresh and target.is_dir() and members_path.is_file() and stamp_path.is_file():
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
        # 缓存必须和当前数据版本、日线数量对得上：抓完新日线却复用旧面板，
        # 会让"PIT 重跑"悄悄跑在 99 只的旧样本上（踩过一次）。
        current = len(list((CACHE_ROOT / (version or stamp.get("version", "")) / "bars").glob("*.csv.gz")))
        if stamp.get("bars") == current and current:
            symbols = len(list(target.glob("*.json")))
            if symbols:
                return {"dir": target, "members": members_path, "symbols": symbols,
                        "snapshots": len(json.loads(members_path.read_text(encoding="utf-8"))),
                        "version": version or stamp.get("version"), "reused": True}
    universe = PitUniverse(version)
    info = universe.write_cache(target, min_rows=min_rows)
    info["dir"] = target
    info["members"] = members_path
    info["reused"] = False
    stamp_path.write_text(
        json.dumps({"version": universe.version,
                    "bars": len(list((universe.dataset.root / "bars").glob("*.csv.gz"))),
                    "symbols": info["symbols"],
                    "written_at": universe.version}, ensure_ascii=False),
        encoding="utf-8",
    )
    return info


def filter_by_members(
    records: Iterable[dict[str, Any]],
    members: Callable[[str], set[str]],
) -> list[dict[str, Any]]:
    """按 ``date -> 当日成分`` 过滤记录；PIT 口径的最后一道闸门。"""
    cache: dict[str, set[str]] = {}
    kept: list[dict[str, Any]] = []
    for record in records:
        date = record.get("date")
        if date not in cache:
            cache[date] = members(date)
        if record.get("symbol") in cache[date]:
            kept.append(record)
    return kept



# --------------------------------------------------------------------------- #
# 字段体检（第 2 步的验收材料）
# --------------------------------------------------------------------------- #


def field_audit(universe: PitUniverse, *, sample_days: int = 40) -> dict[str, Any]:
    """行业 / 市值 / ST / 停牌 / 涨跌停 / 上市天数的覆盖率。

    分母分两种，必须分开看：**成分内**（池子本身该有的）与**当日有行情**（可算的）。
    行业缺失按 ``UNKNOWN`` 落档，市值缺失不进分层，两者都会写进报告。
    """
    calendar = [day for day in universe.calendar if day >= "2018-01-01"]
    if not calendar:
        calendar = universe.calendar
    step = max(1, len(calendar) // max(1, sample_days))
    days = calendar[::step]
    rows: list[dict[str, Any]] = []
    for day in days:
        members = universe.members(day)
        snapshot = universe.snapshot(day)
        listed = members & set(snapshot)
        table = universe.fields(day)
        industry_known = [code for code in listed if table[code]["industry"] != "UNKNOWN"]
        mcap_known = [code for code in listed if table[code]["mcap"]]
        st_count = sum(1 for code in listed if table[code]["is_st"])
        limit_count = sum(
            1 for code in listed
            if table[code]["at_limit_up"] or table[code]["at_limit_down"]
        )
        suspended_like = len(members) - len(listed)
        rows.append(
            {
                "date": day,
                "members": len(members),
                "traded": len(listed),
                "no_bar_or_suspended": suspended_like,
                "industry_known_pct": round(len(industry_known) / max(len(listed), 1) * 100, 1),
                "mcap_known_pct": round(len(mcap_known) / max(len(listed), 1) * 100, 1),
                "st_count": st_count,
                "limit_count": limit_count,
                "listing_days_known_pct": round(
                    sum(1 for code in listed if table[code]["listing_days"] is not None)
                    / max(len(listed), 1) * 100, 1
                ),
            }
        )

    def mean(key: str) -> float | None:
        values = [row[key] for row in rows if row.get(key) is not None]
        return round(statistics.fmean(values), 1) if values else None

    return {
        "version": universe.version,
        "days": days,
        "samples": len(rows),
        "members_mean": mean("members"),
        "traded_mean": mean("traded"),
        "industry_known_pct": mean("industry_known_pct"),
        "mcap_known_pct": mean("mcap_known_pct"),
        "listing_days_known_pct": mean("listing_days_known_pct"),
        "st_mean": mean("st_count"),
        "limit_mean": mean("limit_count"),
        "rows": rows,
    }
