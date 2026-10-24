"""基本面/资金/事件因子的构造与增量评估（纯计算，不取数、不 import 生产层）。

三个层次严格分开
----------------
1. **派生**（``derive_fundamentals``）：把报表累计值拼成 TTM、还原净资产、算同比。
   只在"某一次披露版本"上做，所以同一报告期的重述会产出两条派生记录，回测时
   由 ``AsOfBook`` 取"当时能看到的那一版"。
2. **构造**（``build_panel``）：按交易日把 PIT 值对齐成 (date, symbol) 行，同时
   给出市值、beta 和未来 20 日收益。
3. **评估**（``neutralize`` / ``evaluate``）：行业 + 规模 + beta 中性化，再对
   现有 OHLCV 基线正交化，得到"增量 IC"；组合增量用固定 80/20 权重，不调参。

为什么增量 IC 要正交化两次
--------------------------
把新因子对基线正交化，回答的是"**扣掉 OHLCV 已经解释的部分之后**，它还能预测
什么"。不做这一步的"因子 IC 为正"没有意义——任何一个与动量相关的财务指标都会
继承动量的 IC。第二次正交化（对行业/规模/beta）是惯例，必须在正交基线之前做，
否则行业效应会被算成"增量"。
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from .pit_join import AsOfBook, vintages

import evalkit
import factors as fa

HORIZON = 20
MIN_XSEC = 30
MIN_INDUSTRY = 3
MAX_STALE_DAYS = 500          # 超过一年半没有新报表就不再用它的质量指标
EVENT_DECAY_DAYS = 120        # 业绩预告的有效期（约 4 个月，覆盖下一个报告期）
FLOW_WINDOW_DAYS = 30         # 资金流累计窗口（≈20 个交易日）
DIVIDEND_WINDOW_DAYS = 365


@dataclass(frozen=True)
class FactorDef:
    key: str
    name: str
    dimension: str          # value | quality | growth | flow | event
    prior: int              # +1 越大越好 / −1 越小越好
    formula: str
    inputs: tuple[str, ...]
    note: str = ""
    proxy: bool = False     # True 表示用代理变量（无免费原始数据）

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "name": self.name, "dimension": self.dimension,
            "prior": self.prior, "formula": self.formula, "inputs": list(self.inputs),
            "note": self.note, "proxy": self.proxy,
        }


FACTORS: dict[str, FactorDef] = {spec.key: spec for spec in (
    # ---------------- 价值 ----------------
    FactorDef("ep", "盈利收益率", "value", +1,
              "eps_ttm / close_raw", ("eps_ttm", "close_raw"),
              "亏损股 eps_ttm 为负仍保留（负 EP 本身就是排序信息），不截断"),
    FactorDef("bp", "账面市值比", "value", +1,
              "bps / close_raw，bps = 净资产 / 总股本", ("net_profit", "roe", "total_share", "close_raw"),
              "净资产用 net_profit/roe 还原（平均净资产口径）；亏损或 roe<=0 时缺失"),
    FactorDef("sp", "营收市值比", "value", +1,
              "revenue_ttm / (close_raw × total_share)", ("revenue", "total_share", "close_raw"),
              "营收用自拼 TTM，不取报表累计值，避免季度间不可比"),
    FactorDef("div_yield", "股息率", "value", +1,
              "过去 365 天已实施每股现金分红 / close_raw", ("div_cash_ps", "close_raw"),
              "以除权除息日为可得日；预案/股东大会通过但未实施的不计入"),

    # ---------------- 质量 ----------------
    FactorDef("roe_ttm", "ROE(TTM)", "quality", +1,
              "np_ttm / equity", ("net_profit", "roe"),
              "用 TTM 净利润除以平均净资产，避免 Q1 的 ROE 只有全年 1/4"),
    FactorDef("gp_margin", "毛利率", "quality", +1,
              "gp_margin（累计口径）", ("gp_margin",),
              "累计口径在 Q4 最可比；季报间横向比较会有季节性，靠行业中性化吸收一部分"),
    FactorDef("cfo_np", "现金流/利润", "quality", +1,
              "cfo_to_np（经营现金流 / 净利润）", ("cfo_to_np",),
              "净利润为负时该比值失去意义，直接置缺失"),
    FactorDef("accrual", "应计利润", "quality", -1,
              "roe_ttm × (1 − cfo_to_np)", ("roe_ttm", "cfo_to_np"),
              "应计越高盈利质量越差，先验为负；这是 Sloan(1996) 的简化版"),

    # ---------------- 成长 ----------------
    FactorDef("rev_yoy", "营收增速", "growth", +1,
              "rev_ttm / rev_ttm(t−4q) − 1", ("revenue",),
              "TTM 同比，剔除季节性；基期为负时缺失"),
    FactorDef("np_yoy", "利润增速", "growth", +1,
              "np_ttm / np_ttm(t−4q) − 1", ("net_profit",),
              "同上；扭亏（基期为负）不参与排序"),
    FactorDef("analyst_revision", "一致预期修正", "growth", +1,
              "30 日一致预期 EPS 修正幅度（CSV 导入）", ("analyst_revision",),
              "无免费原始源；缺失时该因子整体不参与评估，不用代理硬凑", proxy=True),

    # ---------------- 资金 ----------------
    FactorDef("margin_ratio", "融资余额占比", "flow", -1,
              "融资余额 / 流通市值", ("margin_ratio",),
              "杠杆拥挤度，先验为负（拥挤交易后续收益差）"),
    FactorDef("margin_chg20", "融资净买入强度", "flow", +1,
              "20 日融资余额变化 / 流通市值", ("margin_balance",),
              "短期资金流入，先验为正；与动量相关性高，增量要打折扣"),
    FactorDef("north_hold_ratio", "北向持股比例", "flow", +1,
              "北向持股合计 / 流通股本（季度快照）", ("north_hold_num", "float_share"),
              "季度快照而非日频：CCASS 按券商拆分，日频全量拉取代价过高"),
    FactorDef("lhb_net_20", "龙虎榜净买入", "flow", +1,
              "30 日龙虎榜净买入 / 龙虎榜成交额", ("lhb_net", "lhb_amount"),
              "上榜是极端事件，取净买入强度而不是绝对额，避免大票占优"),

    # ---------------- 事件 ----------------
    FactorDef("event_forecast", "业绩预告方向", "event", +1,
              "预告类型映射到 −1..+1，有效期 120 天", ("forecast_direction",),
              "预增/略增/扭亏/续盈为正，预减/略减/首亏/续亏为负"),
    FactorDef("event_forecast_gap", "预告超预期", "event", +1,
              "预告同比增速中值 − 上一期已实现同比", ("forecast_chg_mid", "np_yoy"),
              "衡量\"公司自己说的\"比\"上一份报表\"加速多少"),
)}


# --------------------------------------------------------------------------- #
# 派生：TTM 与净资产
# --------------------------------------------------------------------------- #


def _quarter(period: str) -> tuple[int, int]:
    month = int(period[5:7])
    return int(period[:4]), {3: 1, 6: 2, 9: 3, 12: 4}.get(month, 4)


def _period_of(year: int, quarter: int) -> str:
    return {1: f"{year}-03-31", 2: f"{year}-06-30", 3: f"{year}-09-30", 4: f"{year}-12-31"}[quarter]


def derive_fundamentals(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """把累计报表拼成 TTM，并还原净资产、同比。

    每个派生值都带原披露的 ``(period, available)``，所以重述会自然产生新的
    时间点记录，不需要额外处理。
    """
    frames = vintages(records)
    # (symbol, period) → [(available, frame)]
    index: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = {}
    for (symbol, period, available), frame in frames.items():
        index.setdefault((symbol, period), []).append((available, frame))
    for rows in index.values():
        rows.sort(key=lambda item: item[0])

    def as_of(symbol: str, period: str, available: str) -> dict[str, Any] | None:
        rows = index.get((symbol, period))
        if not rows:
            return None
        best = None
        for stamp, frame in rows:
            if stamp <= available:
                best = frame
            else:
                break
        return best

    def ttm(symbol: str, period: str, available: str, field: str) -> float | None:
        frame = as_of(symbol, period, available)
        if not frame:
            return None
        current = frame.get(field)
        if current is None:
            return None
        year, quarter = _quarter(period)
        if quarter == 4:
            return float(current)
        annual = as_of(symbol, f"{year - 1}-12-31", available)
        prior = as_of(symbol, _period_of(year - 1, quarter), available)
        if not annual or not prior:
            return None
        base_annual, base_prior = annual.get(field), prior.get(field)
        if base_annual is None or base_prior is None:
            return None
        return float(current) + float(base_annual) - float(base_prior)

    out: list[dict[str, Any]] = []
    for (symbol, period, available), frame in sorted(frames.items()):
        if "net_profit" not in frame:
            continue
        np_ttm = ttm(symbol, period, available, "net_profit")
        rev_ttm = ttm(symbol, period, available, "revenue")
        year, quarter = _quarter(period)
        np_prev = ttm(symbol, _period_of(year - 1, quarter), available, "net_profit")
        rev_prev = ttm(symbol, _period_of(year - 1, quarter), available, "revenue")

        roe = frame.get("roe")
        net_profit = frame.get("net_profit")
        equity = None
        if roe and net_profit and roe > 1e-4 and net_profit > 0:
            equity = float(net_profit) / float(roe)
        shares = frame.get("total_share")

        derived: dict[str, float | None] = {
            "np_ttm": np_ttm,
            "rev_ttm": rev_ttm,
            "np_ttm_prev": np_prev,
            "rev_ttm_prev": rev_prev,
            "equity": equity,
            "bps": (equity / float(shares)) if (equity and shares) else None,
            "roe_ttm": (np_ttm / equity) if (np_ttm and equity) else None,
            "np_yoy": (np_ttm / np_prev - 1.0) if (np_ttm and np_prev and np_prev > 0) else None,
            "rev_yoy": (rev_ttm / rev_prev - 1.0) if (rev_ttm and rev_prev and rev_prev > 0) else None,
        }
        cfo_np = frame.get("cfo_to_np")
        if derived["roe_ttm"] is not None and cfo_np is not None:
            derived["accrual"] = derived["roe_ttm"] * (1.0 - float(cfo_np))
        for field, value in derived.items():
            if value is None or not math.isfinite(float(value)):
                continue
            out.append({
                "symbol": symbol, "dimension": "fundamental", "field": field,
                "value": float(value), "period": period, "available": available,
                "source": "derived:ttm",
            })
    return out


# --------------------------------------------------------------------------- #
# 面板
# --------------------------------------------------------------------------- #


def _beta(closes: Sequence[float], benchmark: Sequence[float | None], window: int = 120) -> list[float | None]:
    out: list[float | None] = [None] * len(closes)
    rets = [None] + [
        (closes[i] / closes[i - 1] - 1.0) if closes[i - 1] else None for i in range(1, len(closes))
    ]
    for i in range(window, len(closes)):
        xs = [benchmark[j] for j in range(i - window + 1, i + 1)]
        ys = [rets[j] for j in range(i - window + 1, i + 1)]
        pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
        if len(pairs) < window * 0.7:
            continue
        mean_x = statistics.fmean(p[0] for p in pairs)
        mean_y = statistics.fmean(p[1] for p in pairs)
        var = sum((p[0] - mean_x) ** 2 for p in pairs)
        if var <= 1e-12:
            continue
        cov = sum((p[0] - mean_x) * (p[1] - mean_y) for p in pairs)
        out[i] = cov / var
    return out


def build_panel(
    *,
    records: Sequence[dict[str, Any]],
    prices: dict[str, dict[str, Any]],
    industry: dict[str, str],
    benchmark: dict[str, float],
    scores: dict[tuple[str, str], float] | None = None,
    horizon: int = HORIZON,
    max_stale: int = MAX_STALE_DAYS,
    alignment: str = "available",
) -> list[dict[str, Any]]:
    """按交易日对齐成 ``(date, symbol)`` 行。

    ``alignment="period"`` 只用于**前视偏差的对照实验**：把可得日换成报告期，
    复现"按报告期对齐"这个常见错误，量化它会把 IC 抬高多少。生产口径永远是
    ``"available"``。
    """
    if alignment == "period":
        # 只改财务类记录的可得日；行情/资金记录本来就是"当日可得"，一起改会把
        # close_raw 也搬到报告期，测试里表现为整列变成 None。
        records = [
            {**record, "available": record["period"]} if record["dimension"] == "fundamental"
            else record
            for record in records
        ]
    book = AsOfBook(records)
    rows: list[dict[str, Any]] = []

    for symbol, payload in prices.items():
        bars = payload["rows"]
        if len(bars) < horizon + 60:
            continue
        closes = [float(row["close"]) for row in bars]
        dates = [row["date"] for row in bars]
        beta = _beta(closes, [benchmark.get(day) for day in dates])
        group = industry.get(symbol, "NA")

        for index, date in enumerate(dates):
            if index + horizon + 1 >= len(bars):
                continue
            entry = float(bars[index + 1]["open"])
            exit_price = float(bars[index + 1 + horizon]["open"])
            if entry <= 0 or exit_price <= 0:
                continue
            raw_close = book.value(symbol, "close_raw", date)
            row: dict[str, Any] = {
                "date": date, "symbol": symbol, "industry": group,
                "fwd": (exit_price / entry - 1.0) * 100.0,
                "beta120": beta[index],
                "base_score": (scores or {}).get((date, symbol)),
                "raw_close": raw_close,
            }
            if raw_close:
                shares = book.value(symbol, "total_share", date, max_stale=max_stale)
                float_shares = book.value(symbol, "float_share", date, max_stale=max_stale)
                if shares:
                    row["log_cap"] = math.log(raw_close * float(shares or shares))
                row["float_mcap"] = raw_close * float(float_shares or 0) if float_shares else None

            # ---- 价值 ----
            eps = book.value(symbol, "eps_ttm", date, max_stale=max_stale)
            bps = book.value(symbol, "bps", date, max_stale=max_stale)
            rev_ttm = book.value(symbol, "rev_ttm", date, max_stale=max_stale)
            shares = book.value(symbol, "total_share", date, max_stale=max_stale)
            dividend = book.window_sum(symbol, "div_cash_ps", date, DIVIDEND_WINDOW_DAYS)
            if raw_close:
                row["ep"] = eps / raw_close if eps is not None else None
                row["bp"] = bps / raw_close if bps else None
                row["sp"] = (rev_ttm / (raw_close * shares)) if (rev_ttm and shares) else None
                row["div_yield"] = dividend / raw_close if dividend else None
            # ---- 质量 / 成长 ----
            row["roe_ttm"] = book.value(symbol, "roe_ttm", date, max_stale=max_stale)
            row["gp_margin"] = book.value(symbol, "gp_margin", date, max_stale=max_stale)
            cfo_np = book.value(symbol, "cfo_to_np", date, max_stale=max_stale)
            row["cfo_np"] = cfo_np if (cfo_np is not None and row["roe_ttm"] is not None and row["roe_ttm"] > 0) else None
            row["accrual"] = book.value(symbol, "accrual", date, max_stale=max_stale)
            row["rev_yoy"] = book.value(symbol, "rev_yoy", date, max_stale=max_stale)
            row["np_yoy"] = book.value(symbol, "np_yoy", date, max_stale=max_stale)
            row["analyst_revision"] = book.value(symbol, "analyst_revision", date, max_stale=90)

            # ---- 资金 ----
            row["margin_ratio"] = book.value(symbol, "margin_ratio", date, max_stale=10)
            balance = book.value(symbol, "margin_balance", date, max_stale=10)
            if balance is not None and index >= 20:
                past = book.value(symbol, "margin_balance", dates[index - 20], max_stale=10)
                float_mcap = row.get("float_mcap")
                if past is not None and float_mcap:
                    row["margin_chg20"] = (balance - past) / float_mcap
            north = book.value(symbol, "north_hold_num", date, max_stale=200)
            float_shares = book.value(symbol, "float_share", date, max_stale=max_stale)
            row["north_hold_ratio"] = (north / float_shares) if (north and float_shares) else None
            lhb_net = book.window_sum(symbol, "lhb_net", date, FLOW_WINDOW_DAYS)
            lhb_amount = book.window_sum(symbol, "lhb_amount", date, FLOW_WINDOW_DAYS)
            row["lhb_net_20"] = (lhb_net / lhb_amount) if (lhb_net is not None and lhb_amount) else None

            # ---- 事件 ----
            forecast = book.latest_with_age(symbol, "forecast_direction", date)
            if forecast and forecast[1] <= EVENT_DECAY_DAYS:
                row["event_forecast"] = forecast[0]
                gap = book.value(symbol, "forecast_chg_mid", date, max_stale=EVENT_DECAY_DAYS)
                if gap is not None and row.get("np_yoy") is not None:
                    row["event_forecast_gap"] = (gap / 100.0) - row["np_yoy"]
            rows.append(row)

    rows.sort(key=lambda item: (item["date"], item["symbol"]))
    return rows


# --------------------------------------------------------------------------- #
# 中性化与增量
# --------------------------------------------------------------------------- #


def _rank01(values: Sequence[float | None]) -> list[float | None]:
    present = [(i, v) for i, v in enumerate(values) if v is not None and math.isfinite(v)]
    if len(present) < 2:
        return [None] * len(values)
    ranks = fa.rank_pct([v for _, v in present])
    out: list[float | None] = [None] * len(values)
    for (index, _), rank in zip(present, ranks):
        out[index] = rank / 100.0
    return out


def _industry_demean(rows: Sequence[dict[str, Any]], values: Sequence[float | None]) -> list[float | None]:
    buckets: dict[str, list[float]] = {}
    for row, value in zip(rows, values):
        if value is not None:
            buckets.setdefault(row["industry"], []).append(value)
    overall = statistics.fmean([v for v in values if v is not None]) if any(v is not None for v in values) else 0.0
    means = {
        code: (statistics.fmean(items) if len(items) >= MIN_INDUSTRY else overall)
        for code, items in buckets.items()
    }
    return [None if v is None else v - means.get(row["industry"], overall)
            for row, v in zip(rows, values)]


def neutralize(
    rows: Sequence[dict[str, Any]],
    keys: Iterable[str],
    *,
    baseline_key: str = "base_score",
) -> dict[str, Any]:
    """逐日：分位 → 行业去均值 → 规模/beta 残差 → 再分位 → 对基线正交化。

    只改行内 ``neu_<key>`` 与 ``inc_<key>``，不返回新对象，便于按日期分组反复调用。
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["date"], []).append(row)
    stats = {"dates": 0, "skipped": 0}
    for date, chunk in sorted(groups.items()):
        labeled = [row for row in chunk if row.get("fwd") is not None]
        if len(labeled) < MIN_XSEC:
            stats["skipped"] += 1
            continue
        stats["dates"] += 1
        size = _rank01([row.get("log_cap") for row in chunk])
        beta = _rank01([row.get("beta120") for row in chunk])
        base = _rank01([row.get(baseline_key) for row in chunk]) if baseline_key else None
        for key in keys:
            values = _rank01([row.get(key) for row in chunk])
            adjusted = _industry_demean(chunk, values)
            valid = [
                i for i, value in enumerate(adjusted)
                if value is not None and size[i] is not None and beta[i] is not None
            ]
            for row in chunk:
                row[f"neu_{key}"] = None
                row[f"inc_{key}"] = None
            if len(valid) < MIN_XSEC:
                continue
            residual = fa.orthogonalize([adjusted[i] for i in valid],
                                        [[size[i] for i in valid], [beta[i] for i in valid]])
            ranks = _rank01(residual)
            for position, index in enumerate(valid):
                chunk[index][f"neu_{key}"] = ranks[position]
            if base is None:
                continue
            second = [i for i in valid if base[i] is not None]
            if len(second) < MIN_XSEC:
                continue
            residual2 = fa.orthogonalize([chunk[i][f"neu_{key}"] for i in second], [[base[i] for i in second]])
            ranks2 = _rank01(residual2)
            for position, index in enumerate(second):
                chunk[index][f"inc_{key}"] = ranks2[position]
    return stats


def daily_ic(rows: Sequence[dict[str, Any]], column: str) -> list[float]:
    groups: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        value, target = row.get(column), row.get("fwd")
        if value is None or target is None:
            continue
        groups.setdefault(row["date"], []).append((value, target))
    series: list[float] = []
    for date in sorted(groups):
        pairs = groups[date]
        if len(pairs) < MIN_XSEC:
            continue
        ic = evalkit.spearman([p[0] for p in pairs], [p[1] for p in pairs])
        if ic is not None:
            series.append(ic)
    return series


def evaluate(
    rows: Sequence[dict[str, Any]],
    key: str,
    *,
    horizon: int = HORIZON,
    iters: int = 600,
    combo_weight: float = 0.2,
) -> dict[str, Any]:
    """一个因子的三项 IC：原始 / 中性化 / 增量，以及组合后 ΔIC。"""
    raw = daily_ic(rows, key)
    neu = daily_ic(rows, f"neu_{key}")
    inc = daily_ic(rows, f"inc_{key}")

    # 组合：0.8 × 基线 + 0.2 × 增量因子（权重先验固定，不搜索）
    by_date: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_date.setdefault(row["date"], []).append(row)
    combo_ics: list[float] = []
    delta: list[float] = []
    for date in sorted(by_date):
        chunk = by_date[date]
        usable = [row for row in chunk
                  if row.get("fwd") is not None and row.get("base_score") is not None
                  and row.get(f"inc_{key}") is not None]
        if len(usable) < MIN_XSEC:
            continue
        base_rank = _rank01([row["base_score"] for row in usable])
        inc_rank = _rank01([row[f"inc_{key}"] for row in usable])
        if any(v is None for v in base_rank) or any(v is None for v in inc_rank):
            continue
        combined = [
            (1 - combo_weight) * b + combo_weight * i for b, i in zip(base_rank, inc_rank)
        ]
        fwd = [row["fwd"] for row in usable]
        ic_base = evalkit.spearman(base_rank, fwd)
        ic_combo = evalkit.spearman(combined, fwd)
        if ic_base is None or ic_combo is None:
            continue
        combo_ics.append(ic_combo)
        delta.append(ic_combo - ic_base)

    def stat(series: Sequence[float]) -> dict[str, Any]:
        if len(series) < 30:
            return {"mean": None, "ci_low": None, "ci_high": None, "icir": None, "n": len(series)}
        result = evalkit.ic_ci(series, horizon=horizon, iters=iters)
        deviation = statistics.pstdev(series) if len(series) > 2 else 0.0
        return {
            "mean": round(result["mean"], 4) if result["mean"] is not None else None,
            "ci_low": round(result["ci_low"], 4) if result["ci_low"] is not None else None,
            "ci_high": round(result["ci_high"], 4) if result["ci_high"] is not None else None,
            # ICIR 按"年化"口径：日均 IC / 日 IC 标准差 × sqrt(252/h)，与 wfv2 一致
            "icir": round(result["mean"] / deviation * math.sqrt(252 / horizon), 2)
            if deviation > 1e-12 else None,
            "positive_rate": result.get("positive_rate"),
            "n": len(series),
        }

    inc_stat, delta_stat = stat(inc), stat(delta)
    return {
        "key": key,
        "raw": stat(raw),
        "neutral": stat(neu),
        "incremental": inc_stat,
        "combination": {"weight": combo_weight, "combo": stat(combo_ics), "delta": delta_stat},
        "verdict": _verdict(inc_stat, delta_stat),
    }


def _verdict(inc: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    """判定只回答"有没有增量"，不回答"能不能赚钱"。"""
    if inc.get("mean") is None:
        return {"ok": None, "text": "样本不足"}
    low = inc.get("ci_low") or 0.0
    gain = delta.get("mean") or 0.0
    gain_low = delta.get("ci_low") if delta.get("ci_low") is not None else None
    ok = low > 0 and gain > 0
    text = f"增量 IC {inc['mean']:+.4f} [{low:+.4f}, {inc['ci_high']:+.4f}]"
    if gain_low is not None:
        text += f"，组合 ΔIC {gain:+.4f} [{gain_low:+.4f}, {delta['ci_high']:+.4f}]"
    return {"ok": ok, "text": text}


def coverage(
    rows: Sequence[dict[str, Any]],
    keys: Iterable[str],
    *,
    calendar: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    dates = sorted({row["date"] for row in rows}) if calendar is None else list(calendar)
    total = {date: 0 for date in dates}
    present: dict[str, dict[str, int]] = {key: {date: 0 for date in dates} for key in keys}
    for row in rows:
        if row["date"] not in total:
            continue
        total[row["date"]] += 1
        for key in present:
            if row.get(key) is not None:
                present[key][row["date"]] += 1
    out: dict[str, dict[str, Any]] = {}
    for key, counts in present.items():
        ratios = [counts[d] / total[d] for d in dates if total[d] >= MIN_XSEC]
        active = [d for d in dates if total[d] >= MIN_XSEC and counts[d] >= MIN_XSEC]
        out[key] = {
            "date_share": round(len(active) / max(1, len(ratios)), 3),
            "mean_stocks": round(statistics.fmean([counts[d] for d in dates if total[d] >= MIN_XSEC]), 1)
            if ratios else 0.0,
            "first": active[0] if active else None,
            "last": active[-1] if active else None,
        }
    return out
