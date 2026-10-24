"""baostock 财务数据源：报表三张 + 不复权收盘价，全部以 ``pubDate`` 为可得日。

为什么是这三个接口
------------------
- ``query_profit_data``：ROE、毛利率、净利率、净利润、EPS(TTM)、主营收入、总股本、
  流通股本——价值与质量两族的原料几乎都在这里；
- ``query_balance_data``：资产负债率、权益乘数（用来还原净资产，进而算 BP）；
- ``query_cash_flow_data``：经营现金流/净利润（CFOToNP），应计因子的分母。

增速不取 ``query_growth_data``：那张表的同比是"单季/累计"口径混杂，与我们要的
TTM 同比不一致；直接从累计值拼 TTM 再同比，只有一个事实来源，也少 1/4 的请求。

不复权价是价值因子的必需品
--------------------------
项目里的 ``.cache/fields`` 是**后复权**价（2016 年的 10 元被复权成 3 元），
拿它乘总股本算市值会得到完全错误的 EP/BP。所以这里额外抓一份 ``adjustflag=3``
的原始收盘价，只在"算估值比率"时使用，收益计算仍走 ``.cache/fields``。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .base import FetchContext, FieldSpec, Source, SourceError, register

QUARTERS = (1, 2, 3, 4)

# 规范字段名 ← baostock 列名
PROFIT_COLUMNS = {
    "roe": "roeAvg",              # 平均净资产收益率（累计）
    "np_margin": "npMargin",
    "gp_margin": "gpMargin",
    "net_profit": "netProfit",    # 累计归母净利润
    "eps_ttm": "epsTTM",
    "revenue": "MBRevenue",       # 累计主营收入
    "total_share": "totalShare",
    "float_share": "liqaShare",
}
BALANCE_COLUMNS = {
    "current_ratio": "currentRatio",
    "quick_ratio": "quickRatio",
    "cash_ratio": "cashRatio",
    "liability_to_asset": "liabilityToAsset",
    "asset_to_equity": "assetToEquity",
}
CASHFLOW_COLUMNS = {
    "cfo_to_np": "CFOToNP",
    "cfo_to_or": "CFOToOR",
    "cfo_to_gr": "CFOToGr",
}

FIELDS = tuple(
    FieldSpec(name=name, dimension="fundamental", pit_basis="公告日 pubDate", note=column)
    for name, column in
    [(name, col) for name, col in PROFIT_COLUMNS.items()]
    + [(name, col) for name, col in BALANCE_COLUMNS.items()]
    + [(name, col) for name, col in CASHFLOW_COLUMNS.items()]
) + (
    FieldSpec(name="close_raw", dimension="market", pit_basis="当日收盘",
              note="不复权收盘价，只用于估值比率与市值"),
)


@register
class BaoStockFundamental(Source):
    name = "baostock_fundamental"
    dimensions = ("fundamental", "market")
    fields = FIELDS
    optional_deps = ("baostock",)

    # ------------------------------------------------------------------ #
    def fetch(self, ctx: FetchContext) -> list[dict[str, Any]]:
        from ..bs_source import _load_baostock, to_bs  # 只有这里碰 baostock

        cache = Path(ctx.cache) / "bs_fundamental"
        cache.mkdir(parents=True, exist_ok=True)
        bs = _load_baostock()
        login = bs.login()
        if login.error_code != "0":
            raise SourceError(f"baostock 登录失败：{login.error_msg}")
        budget = {"calls": 0}
        records: list[dict[str, Any]] = []
        try:
            for symbol in ctx.symbols:
                path = cache / f"{symbol}.json"
                if path.is_file():
                    records.extend(json.loads(path.read_text(encoding="utf-8")))
                    continue
                payload = self._one(bs, symbol, ctx, budget)
                path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                records.extend(payload)
                ctx.log(f"      {symbol} {len(payload)} 条")
                if ctx.limit and budget["calls"] >= ctx.limit:
                    ctx.log(f"      达到本次请求上限 {ctx.limit}，先落盘")
                    break
        finally:
            try:
                bs.logout()
            except Exception:  # pragma: no cover
                pass
        return records

    # ------------------------------------------------------------------ #
    def _one(self, bs, symbol: str, ctx: FetchContext, budget: dict[str, int]) -> list[dict[str, Any]]:
        from ..bs_source import to_bs

        code = to_bs(symbol)
        start_year = int(ctx.start[:4]) - 1        # 多取一年，TTM 与同比要用
        end_year = int(ctx.end[:4])
        out: list[dict[str, Any]] = []

        for year in range(start_year, end_year + 1):
            for quarter in QUARTERS:
                frame: dict[str, Any] = {}
                pub_dates: list[str] = []
                for method, columns in (
                    ("query_profit_data", PROFIT_COLUMNS),
                    ("query_balance_data", BALANCE_COLUMNS),
                    ("query_cash_flow_data", CASHFLOW_COLUMNS),
                ):
                    budget["calls"] += 1
                    try:
                        result = getattr(bs, method)(code=code, year=year, quarter=quarter)
                    except Exception:
                        continue
                    if result.error_code != "0":
                        continue
                    while result.next():
                        row = dict(zip(result.fields, result.get_row_data()))
                        if not row.get("statDate"):
                            continue
                        frame.update({
                            name: _number(row.get(column))
                            for name, column in columns.items()
                        })
                        if row.get("pubDate"):
                            pub_dates.append(row["pubDate"])
                        break                      # 一次查询只有一个 statDate
                if not frame or not pub_dates:
                    continue
                period = _quarter_end(year, quarter)
                available = max(pub_dates)
                for name, value in frame.items():
                    if value is None:
                        continue
                    out.append({
                        "symbol": symbol, "dimension": "fundamental", "field": name,
                        "value": value, "period": period, "available": available,
                        "source": "baostock:quarterly",
                    })

        # 不复权收盘价：分段取，整段十年一次会整段不回包
        for chunk_start, chunk_end in _segments(ctx.start, ctx.end, years=3):
            budget["calls"] += 1
            try:
                result = bs.query_history_k_data_plus(
                    code, "date,close", start_date=chunk_start, end_date=chunk_end,
                    frequency="d", adjustflag="3",
                )
            except Exception:
                continue
            if result.error_code != "0":
                continue
            while result.next():
                day, close = result.get_row_data()[:2]
                value = _number(close)
                if day and value and value > 0:
                    out.append({
                        "symbol": symbol, "dimension": "market", "field": "close_raw",
                        "value": value, "period": day, "available": day,
                        "source": "baostock:query_history_k_data_plus(adjustflag=3)",
                    })
        return out


def _segments(start: str, end: str, *, years: int = 3) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    cursor = start
    while cursor <= end:
        year_end = f"{int(cursor[:4]) + years - 1}-12-31"
        stop = min(year_end, end)
        out.append((cursor, stop))
        if stop == end:
            break
        cursor = f"{int(stop[:4]) + 1}-01-01"
    return out


def _quarter_end(year: int, quarter: int) -> str:
    return {1: f"{year}-03-31", 2: f"{year}-06-30", 3: f"{year}-09-30", 4: f"{year}-12-31"}[quarter]


def _number(text: Any) -> float | None:
    if text is None:
        return None
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value
