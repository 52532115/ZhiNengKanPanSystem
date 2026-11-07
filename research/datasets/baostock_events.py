"""baostock 事件源：业绩预告、业绩快报、分红方案。

事件的 PIT 口径比财务数据更苛刻：**公告日才是可得日**。业绩预告在 1 月底就发了，
报告期却是 12/31；用报告期对齐等于提前一个月知道公司自己要报多少。

分红用**除权除息日**（``dividOperateDate``）而不是预案公告日：预案可能被股东大
会否掉或调整金额，只有实施了才算"过去 12 个月拿到手的股息"。代价是股息率因子
天然滞后 1–2 个月，这个滞后是真实的，不该被优化掉。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import FetchContext, FieldSpec, Source, SourceError, register

# 预告类型 → 方向分（-1..+1）。分档来自交易所披露口径，不是拟合出来的。
FORECAST_DIRECTION = {
    "预增": 1.0, "略增": 0.6, "扭亏": 0.8, "续盈": 0.4,
    "预减": -1.0, "略减": -0.6, "首亏": -0.8, "续亏": -0.8,
    "不确定": 0.0,
}

FIELDS = (
    FieldSpec("forecast_direction", "event", "numeric", pit_basis="预告公告日", note="类型映射到 -1..+1"),
    FieldSpec("forecast_chg_mid", "event", "numeric", "%", "预告公告日", "预告同比增速中值"),
    FieldSpec("express_roe", "event", "numeric", pit_basis="快报公告日", note="业绩快报加权 ROE"),
    FieldSpec("express_rev_yoy", "event", "numeric", "%", "快报公告日", "快报营收同比"),
    FieldSpec("div_cash_ps", "event", "numeric", "元/股", "除权除息日", "每股税前现金分红"),
)


@register
class BaoStockEvents(Source):
    name = "baostock_events"
    dimensions = ("event",)
    fields = FIELDS
    optional_deps = ("baostock",)

    def fetch(self, ctx: FetchContext) -> list[dict[str, Any]]:
        from ..bs_source import _load_baostock, to_bs

        cache = Path(ctx.cache) / "bs_events"
        cache.mkdir(parents=True, exist_ok=True)
        bs = _load_baostock()
        if bs.login().error_code != "0":
            raise SourceError("baostock 登录失败")
        records: list[dict[str, Any]] = []
        try:
            for symbol in ctx.symbols:
                path = cache / f"{symbol}.json"
                if path.is_file():
                    records.extend(json.loads(path.read_text(encoding="utf-8")))
                    continue
                payload = self._one(bs, symbol, ctx)
                path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                records.extend(payload)
                ctx.log(f"      {symbol} 事件 {len(payload)} 条")
        finally:
            try:
                bs.logout()
            except Exception:  # pragma: no cover
                pass
        return records

    def _one(self, bs, symbol: str, ctx: FetchContext) -> list[dict[str, Any]]:
        from ..bs_source import to_bs

        code = to_bs(symbol)
        out: list[dict[str, Any]] = []
        try:
            result = bs.query_forecast_report(code=code, start_date=ctx.start, end_date=ctx.end)
        except Exception:
            result = None
        if result is not None and result.error_code == "0":
            while result.next():
                row = dict(zip(result.fields, result.get_row_data()))
                published = row.get("profitForcastExpPubDate") or ""
                period = row.get("profitForcastExpStatDate") or ""
                if not published or not period:
                    continue
                direction = FORECAST_DIRECTION.get((row.get("profitForcastType") or "").strip(), 0.0)
                out.append({
                    "symbol": symbol, "dimension": "event", "field": "forecast_direction",
                    "value": direction, "period": period, "available": published,
                    "source": "baostock:query_forecast_report",
                })
                up = _number(row.get("profitForcastChgPctUp"))
                down = _number(row.get("profitForcastChgPctDwn"))
                if up is not None or down is not None:
                    middle = ((up if up is not None else down) + (down if down is not None else up)) / 2
                    out.append({
                        "symbol": symbol, "dimension": "event", "field": "forecast_chg_mid",
                        "value": middle, "period": period, "available": published,
                        "source": "baostock:query_forecast_report",
                    })

        try:
            result = bs.query_performance_express_report(code=code, start_date=ctx.start, end_date=ctx.end)
        except Exception:
            result = None
        if result is not None and result.error_code == "0":
            while result.next():
                row = dict(zip(result.fields, result.get_row_data()))
                published = row.get("performanceExpPubDate") or ""
                period = row.get("performanceExpStatDate") or ""
                if not published or not period:
                    continue
                for field, column in (("express_roe", "performanceExpressROEWa"),
                                      ("express_rev_yoy", "performanceExpressGRYOY")):
                    value = _number(row.get(column))
                    if value is None:
                        continue
                    out.append({
                        "symbol": symbol, "dimension": "event", "field": field,
                        "value": value, "period": period, "available": published,
                        "source": "baostock:query_performance_express_report",
                    })

        # 同一天可能有多笔分红（年度 + 中期，或 A/B 股分拆披露）：按除权除息日汇总，
        # 否则 validate 会报重复记录，而"去重取第一条"会漏掉另一笔现金。
        dividends: dict[str, float] = {}
        for year in range(int(ctx.start[:4]), int(ctx.end[:4]) + 1):
            try:
                result = bs.query_dividend_data(code=code, year=year, yearType="report")
            except Exception:
                continue
            if result.error_code != "0":
                continue
            while result.next():
                row = dict(zip(result.fields, result.get_row_data()))
                operated = row.get("dividOperateDate") or ""
                cash = _number(row.get("dividCashPsBeforeTax"))
                if not operated or cash is None or cash <= 0:
                    continue
                dividends[operated] = dividends.get(operated, 0.0) + cash
        for operated, cash in sorted(dividends.items()):
            out.append({
                "symbol": symbol, "dimension": "event", "field": "div_cash_ps",
                "value": cash, "period": operated, "available": operated,
                "source": "baostock:query_dividend_data",
            })
        return out


def _number(text: Any) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None
