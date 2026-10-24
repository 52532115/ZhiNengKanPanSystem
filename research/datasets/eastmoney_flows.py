"""东方财富资金流数据源：两融、龙虎榜、北向持股。

三个接口的共同点是**都带前视陷阱**：

- 两融是 T 日收盘后披露、T+1 才能用于交易，所以 ``available = T+1``；
- 龙虎榜同理，而且原始返回里直接带 ``D1/D5/D10_CLOSE_ADJCHRATE``（未来 1/5/10 日
  涨跌幅）——这是最危险的一类字段，谁不小心把它当特征，回测立刻"很赚钱"。
  本适配器**白名单取列**，未来收益一律不进长表；
- 北向持股是 CCASS 按券商拆分的明细，同一只票一天有十几到几十行，日频全量拉
  60 只票十年要上万次请求。这里改成**季度末快照**（``--north-snapshots``），
  承认它是慢变量，不假装日频。

ETF 份额变化没有稳定的免费接口，走 ``csv_drop``（券商终端导出即可），
不在本模块里编造数据。
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from .base import FetchContext, FieldSpec, Source, SourceError, register

BASE = "https://datacenter-web.eastmoney.com/api/data/v1/get"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

FIELDS = (
    FieldSpec("margin_balance", "flow", "numeric", "元", "T+1 披露", "融资余额 RZYE"),
    FieldSpec("margin_net_buy", "flow", "numeric", "元", "T+1 披露", "融资净买入 RZJME"),
    FieldSpec("short_balance", "flow", "numeric", "元", "T+1 披露", "融券余额 RQYE"),
    FieldSpec("margin_ratio", "flow", "numeric", "%", "T+1 披露", "融资余额/流通市值 RZYEZB"),
    FieldSpec("lhb_net", "flow", "numeric", "元", "T+1 披露", "龙虎榜净买入"),
    FieldSpec("lhb_amount", "flow", "numeric", "元", "T+1 披露", "龙虎榜成交额"),
    FieldSpec("north_hold_num", "flow", "numeric", "股", "T+1 披露（季度快照）", "北向持股合计"),
)


def _get_json(url: str, *, retries: int = 3, timeout: int = 20) -> dict[str, Any]:
    last = ""
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            last = str(exc)
            time.sleep(0.5 * (attempt + 1))
    raise SourceError(f"请求失败：{last}")


def _paged(report: str, flt: str, *, page_size: int = 500, max_pages: int = 40,
           sort: str | None = None, pause: float = 0.0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        query = {
            "reportName": report, "columns": "ALL", "filter": flt,
            "pageNumber": str(page), "pageSize": str(page_size),
        }
        if sort:
            query["sortColumns"] = sort
            query["sortTypes"] = "1"
        payload = _get_json(f"{BASE}?{urllib.parse.urlencode(query, safe='(),=')}")
        result = payload.get("result") or {}
        data = result.get("data") or []
        rows.extend(data)
        pages = result.get("pages") or 1
        if page >= pages or not data:
            break
        if pause:
            time.sleep(pause)
    return rows


def _next_day(day: str) -> str:
    from datetime import date, timedelta
    return (date.fromisoformat(day[:10]) + timedelta(days=1)).isoformat()


@register
class EastmoneyFlows(Source):
    name = "eastmoney_flows"
    dimensions = ("flow",)
    fields = FIELDS
    optional_deps = ()          # 纯标准库 HTTP，无第三方依赖

    def probe(self) -> tuple[bool, str]:
        try:
            _paged("RPTA_WEB_RZRQ_GGMX", '(scode="600519")', page_size=1, max_pages=1)
            return True, "ok"
        except Exception as exc:
            return False, f"接口不可达：{exc}"

    def fetch(self, ctx: FetchContext) -> list[dict[str, Any]]:
        cache = Path(ctx.cache) / "em_flows"
        cache.mkdir(parents=True, exist_ok=True)
        snapshot_mode = ctx.extra.get("north_snapshots", "quarter")
        records: list[dict[str, Any]] = []
        for symbol in ctx.symbols:
            path = cache / f"{symbol}.json"
            if path.is_file():
                records.extend(json.loads(path.read_text(encoding="utf-8")))
                continue
            payload: list[dict[str, Any]] = []
            payload.extend(self._margin(symbol, ctx))
            payload.extend(self._lhb(symbol, ctx))
            if ctx.extra.get("north", True):
                payload.extend(self._north(symbol, ctx, mode=snapshot_mode))
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            records.extend(payload)
            ctx.log(f"      {symbol} 资金 {len(payload)} 条")
        return records

    # ------------------------------------------------------------------ #
    def _margin(self, symbol: str, ctx: FetchContext) -> list[dict[str, Any]]:
        rows = _paged("RPTA_WEB_RZRQ_GGMX", f'(scode="{symbol}")', sort="DATE", pause=ctx.pause)
        out: list[dict[str, Any]] = []
        for row in rows:
            day = (row.get("DATE") or "")[:10]
            if not day or day < ctx.start or day > ctx.end:
                continue
            available = _next_day(day)
            for field, column in (("margin_balance", "RZYE"), ("margin_net_buy", "RZJME"),
                                  ("short_balance", "RQYE"), ("margin_ratio", "RZYEZB")):
                value = row.get(column)
                if value is None:
                    continue
                out.append({
                    "symbol": symbol, "dimension": "flow", "field": field,
                    "value": float(value), "period": day, "available": available,
                    "source": "eastmoney:RPTA_WEB_RZRQ_GGMX",
                })
        return out

    def _lhb(self, symbol: str, ctx: FetchContext) -> list[dict[str, Any]]:
        rows = _paged("RPT_DAILYBILLBOARD_DETAILSNEW", f'(SECURITY_CODE="{symbol}")',
                      sort="TRADE_DATE", pause=ctx.pause)
        # 同一只票同一天可能因多条"上榜原因"各有一行，必须先按日聚合：
        # 不去重会让 validate 报"重复记录"，去重（取第一条）会漏掉另一条原因的金额。
        daily: dict[str, dict[str, float]] = {}
        for row in rows:
            day = (row.get("TRADE_DATE") or "")[:10]
            if not day or day < ctx.start or day > ctx.end:
                continue
            # 白名单：只取成交与买卖金额，D1/D5/D10 未来收益绝不入库
            buy = row.get("BILLBOARD_BUY_AMT")
            sell = row.get("BILLBOARD_SELL_AMT")
            net = row.get("BILLBOARD_NET_AMT")
            if net is None and buy is not None and sell is not None:
                net = float(buy) - float(sell)
            bucket = daily.setdefault(day, {"lhb_net": 0.0, "lhb_amount": 0.0})
            if net is not None:
                bucket["lhb_net"] += float(net)
            if row.get("BILLBOARD_DEAL_AMT") is not None:
                bucket["lhb_amount"] += float(row["BILLBOARD_DEAL_AMT"])
        out: list[dict[str, Any]] = []
        for day, bucket in sorted(daily.items()):
            available = _next_day(day)
            for field, value in bucket.items():
                out.append({
                    "symbol": symbol, "dimension": "flow", "field": field,
                    "value": value, "period": day, "available": available,
                    "source": "eastmoney:RPT_DAILYBILLBOARD_DETAILSNEW",
                })
        return out

    def _north(self, symbol: str, ctx: FetchContext, *, mode: str = "quarter") -> list[dict[str, Any]]:
        dates = _snapshot_dates(ctx.start, ctx.end, mode=mode,
                                calendar=ctx.extra.get("calendar") or ())
        if not dates:
            return []
        out: list[dict[str, Any]] = []
        for day in dates:
            flt = f'(SECURITY_CODE="{symbol}")(HOLD_DATE=\'{day}\')'
            try:
                rows = _paged("RPT_MUTUAL_HOLD_DET", flt, page_size=500, max_pages=4)
            except SourceError:
                continue
            if not rows:
                continue
            total = sum(float(row.get("HOLD_NUM") or 0.0) for row in rows)
            if total <= 0:
                continue
            out.append({
                "symbol": symbol, "dimension": "flow", "field": "north_hold_num",
                "value": total, "period": day, "available": _next_day(day),
                "source": "eastmoney:RPT_MUTUAL_HOLD_DET",
            })
        return out


def _snapshot_dates(start: str, end: str, *, mode: str = "quarter",
                    calendar: Iterable[str] = ()) -> list[str]:
    """季度/月末快照日，取该月**最后一个交易日**。

    固定用 28 号会大量落在周末（接口查不到数据、覆盖率白白掉一半），所以有
    交易日历时按日历回退，没有时退化为 28 号。
    """
    days = sorted(day for day in calendar if start <= day <= end)
    months = (3, 6, 9, 12) if mode == "quarter" else tuple(range(1, 13))
    out: list[str] = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        for month in months:
            prefix = f"{year}-{month:02d}"
            if days:
                candidates = [day for day in days if day.startswith(prefix)]
                if candidates:
                    out.append(candidates[-1])
            else:
                day = f"{prefix}-28"
                if start <= day <= end:
                    out.append(day)
    return out
