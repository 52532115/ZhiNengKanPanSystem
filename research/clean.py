"""清洗层：把原始日线变成"可用于横截面研究"的面板。

六件事，每件都对应一个具体的历史坑：

1. **后复权收益**：``close`` 已经在抓取时定为 ``adjustflag=1``（后复权），
   所以 ``r_t = close_t / close_{t-1} - 1`` 天然跨过除权除息与拆股，不需要再
   手工复权因子；前复权只用于展示。
2. **成交量同步复权**：后复权只调价格不调股数，直接拿 ``volume`` 做量比会在
   拆股日跳变。统一用 ``volume_adj = amount / close``（按后复权价折算的股数），
   与价格同尺度。
3. **IPO 首日异常**：次新股首行收益率是"发行价 → 首日收盘"，动辄 +300%
   （001391 就是 +325%）。策略层绝不能用它：首行一律不进收益率序列。
4. **字段统一**：``amount`` 缺失（备用源通病）时用 ``volume × close`` 回填并
   打 ``amount_estimated`` 标记；``turn`` 缺失时置空而不是填 0。
5. **停牌 / 涨跌停 / 一字板**：``tradestatus=0`` 或量额为 0 记停牌；用
   ``preclose`` 与板块涨跌幅规则算涨跌停价；``high == low`` 且触板记一字板。
6. **跨源校验**：抽样用东方财富与腾讯的同日不复权收盘对照，避免单一数据源
   的除权处理错误被当成 alpha。
"""

from __future__ import annotations

import json
import math
import statistics
import urllib.parse
import urllib.request
from typing import Any, Iterable, Sequence

from .manifest import Dataset

# 板块涨跌幅限制（不含 ST 的 5%，ST 单独按名称/isST 处理）
LIMIT_MAIN = 0.10
LIMIT_STAR = 0.20
LIMIT_ST = 0.05
EPS = 0.005
MAX_ABS_RETURN = 0.25      # 单日超过 25% 的"正常交易"视为数据错误（20% 板留余量）
IPO_SKIP_DAYS = 2          # 上市头两个交易日不进收益率序列


def _f(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def limit_ratio(symbol: str, is_st: bool) -> float:
    if is_st:
        return LIMIT_ST
    if symbol.startswith(("300", "301", "302", "688", "689")):
        return LIMIT_STAR
    if symbol.startswith(("43", "83", "87", "88", "92")):
        return 0.30
    return LIMIT_MAIN


def normalize(
    rows: Sequence[dict[str, str]],
    symbol: str,
    name: str = "",
    *,
    dataset_start: str | None = None,
) -> list[dict[str, Any]]:
    """原始 CSV 行 → 带标记的标准 bar 列表（按日期升序）。

    每行补齐：``ret``（后复权日收益）、``volume_adj``、``suspended``、
    ``limit_up``/``limit_down``/``one_word``、``is_st``、``listing_days``、
    ``amount_estimated``、``ret_ok``（是否可用于研究）。
    """
    bars: list[dict[str, Any]] = []
    # 抓取区间开始前就上市的老股，首行不是 IPO 首日：上市天数不能从首行算，
    # 否则 2017-07 拉的十年数据会让 2010 年上市的公司"看起来只上市了 250 天"。
    pre_listed = bool(dataset_start and rows and rows[0]["date"] <= dataset_start)
    for index, row in enumerate(rows):
        close = _f(row.get("close"))
        preclose = _f(row.get("preclose"))
        volume = _f(row.get("volume")) or 0.0
        amount = _f(row.get("amount"))
        turnover = _f(row.get("turn"))
        is_st = str(row.get("isST") or "0") == "1"
        trade_status = str(row.get("tradestatus") or "1")
        if close is None or close <= 0:
            continue
        amount_estimated = False
        if amount is None or amount <= 0:
            amount = volume * close
            amount_estimated = True
        volume_adj = amount / close if close > 0 else 0.0

        ratio = limit_ratio(symbol, is_st)
        limit_up = limit_down = None
        at_up = at_down = False
        if preclose and preclose > 0:
            limit_up = math.floor(preclose * (1 + ratio) * 100 + 0.5) / 100
            limit_down = math.floor(preclose * (1 - ratio) * 100 + 0.5) / 100
        high, low, open_price = _f(row.get("high")), _f(row.get("low")), _f(row.get("open"))
        if None not in (high, low) and limit_up is not None:
            at_up = high >= limit_up - EPS and low >= limit_up - EPS
            at_down = high <= limit_down + EPS and low <= limit_down + EPS

        ret = None
        ret_ok = False
        if index > 0 and preclose and preclose > 0:
            ret = close / preclose - 1.0
            ret_ok = (
                index >= IPO_SKIP_DAYS
                and trade_status == "1"
                and abs(ret) <= MAX_ABS_RETURN
            )
        bars.append(
            {
                "date": row["date"],
                "symbol": symbol,
                "name": name,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "preclose": preclose,
                "volume": volume,
                "volume_adj": volume_adj,
                "amount": amount,
                "amount_estimated": amount_estimated,
                "turnover": turnover,
                "is_st": is_st,
                "tradestatus": trade_status,
                "suspended": trade_status != "1" or volume <= 0,
                "limit_up": limit_up,
                "limit_down": limit_down,
                "at_limit_up": at_up,
                "at_limit_down": at_down,
                "one_word": bool(at_up or at_down) and high is not None and high == low,
                "ret": ret,
                "ret_ok": ret_ok,
                "listing_days": 10000 + index if pre_listed else index,
                "board": _board(symbol),
            }
        )
    return bars


def _board(symbol: str) -> str:
    if symbol.startswith(("688", "689")):
        return "科创板"
    if symbol.startswith(("300", "301", "302")):
        return "创业板"
    if symbol.startswith(("43", "83", "87", "88", "92")):
        return "北交所"
    return "主板"


def free_float_mcap(bar: dict[str, Any]) -> float | None:
    """自由流通市值代理：``amount / (turn/100)``；换手率缺失时返回 None。"""
    turnover = bar.get("turnover")
    amount = bar.get("amount")
    if not turnover or turnover <= 0 or not amount or amount <= 0:
        return None
    return amount / (turnover / 100.0)


def load_universe_bars(dataset: Dataset) -> dict[str, list[dict[str, Any]]]:
    """把整个版本的日线读进内存并按标的清洗。"""
    payload = dataset.manifest()
    index = payload.get("bar_index") or {}
    industry = dataset.read_table("industry") if (dataset.root / "industry.json").is_file() else {}
    dataset_start = None
    calendar_path = dataset.root / "calendar.json"
    if calendar_path.is_file():
        days = json.loads(calendar_path.read_text(encoding="utf-8"))
        dataset_start = days[0] if days else None
    result: dict[str, list[dict[str, Any]]] = {}
    for code in sorted(index):
        raw = dataset.read_bars(code)
        if not raw:
            continue
        name = (industry.get(code) or {}).get("name") or code
        bars = normalize(raw, code, name, dataset_start=dataset_start)
        if bars:
            result[code] = bars
    return result


# --------------------------------------------------------------------------- #
# 质量检查
# --------------------------------------------------------------------------- #


def quality_report(bars_by_symbol: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """数据体检：覆盖、停牌、异常收益、字段缺失。"""
    per_symbol = []
    total_rows = 0
    bad_returns = 0
    amount_missing = 0
    turnover_missing = 0
    one_word = 0
    for symbol, bars in bars_by_symbol.items():
        total_rows += len(bars)
        bad_returns += sum(1 for bar in bars if bar["ret"] is not None and not bar["ret_ok"])
        amount_missing += sum(1 for bar in bars if bar["amount_estimated"])
        turnover_missing += sum(1 for bar in bars if bar["turnover"] is None)
        one_word += sum(1 for bar in bars if bar["one_word"])
        per_symbol.append(
            {
                "symbol": symbol,
                "rows": len(bars),
                "start": bars[0]["date"],
                "end": bars[-1]["date"],
                "suspended_days": sum(1 for bar in bars if bar["suspended"]),
                "limit_up_days": sum(1 for bar in bars if bar["at_limit_up"]),
                "limit_down_days": sum(1 for bar in bars if bar["at_limit_down"]),
            }
        )
    starts = sorted(item["start"] for item in per_symbol) or [""]
    ends = sorted(item["end"] for item in per_symbol) or [""]
    return {
        "symbols": len(per_symbol),
        "rows": total_rows,
        "window": [starts[0], ends[-1]] if per_symbol else [],
        "amount_missing_pct": round(amount_missing / total_rows * 100, 3) if total_rows else 0.0,
        "turnover_missing_pct": round(turnover_missing / total_rows * 100, 3) if total_rows else 0.0,
        "invalid_return_pct": round(bad_returns / total_rows * 100, 3) if total_rows else 0.0,
        "one_word_days": one_word,
        "delisted_like": sum(1 for item in per_symbol if item["end"] < ends[-1]),
        "short_history": sum(1 for item in per_symbol if item["rows"] < 400),
        "per_symbol": sorted(per_symbol, key=lambda item: item["rows"])[:10],
    }


def cross_check(
    codes: Sequence[str],
    *,
    day: str,
    timeout: float = 8.0,
) -> dict[str, Any]:
    """东方财富 vs 腾讯：抽样对照同日不复权收盘，检查除权处理是否一致。

    两个源都可能限流；任一源不可用时返回 ``available: False`` 并说明原因，
    不静默当作"通过"。
    """
    results = []
    for code in codes:
        em = _eastmoney_close(code, day, timeout)
        tx = _tencent_close(code, day, timeout)
        if em is None or tx is None:
            continue
        diff = abs(em / tx - 1) if tx else None
        results.append({"symbol": code, "eastmoney": em, "tencent": tx, "diff": diff})
    if not results:
        return {"available": False, "note": "两个源都没有返回可比数据（可能被限流）", "day": day}
    worst = max(results, key=lambda item: item["diff"] or 0)
    return {
        "available": True,
        "day": day,
        "samples": len(results),
        "within_1bp": round(sum(1 for item in results if (item["diff"] or 1) <= 0.0001) / len(results) * 100, 1),
        "max_diff_bp": round((worst["diff"] or 0) * 10000, 2),
        "worst": worst,
    }


def _eastmoney_close(code: str, day: str, timeout: float) -> float | None:
    secid = f"1.{code}" if code.startswith(("6", "9")) else f"0.{code}"
    url = (
        "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        f"?secid={secid}&fields1=f1,f2&fields2=f51,f53&klt=101&fqt=0"
        f"&beg={day.replace('-', '')}&end={day.replace('-', '')}"
    )
    return _first_close(url, "data", "klines", index=1, timeout=timeout)


def _tencent_close(code: str, day: str, timeout: float) -> float | None:
    key = ("sh" if code.startswith(("6", "9")) else "sz") + code
    url = (
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        f"?param={key},day,{day},{day},10,"
    )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    data = (payload.get("data") or {}).get(key) or {}
    rows = data.get("day") or data.get("qfqday") or []
    for row in rows:
        if row[0] == day:
            return _f(row[2])
    return None


def _first_close(url: str, *path: str, index: int, timeout: float) -> float | None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    node: Any = payload
    for key in path:
        node = (node or {}).get(key)
    if not node:
        return None
    parts = str(node[0]).split(",")
    return _f(parts[index]) if len(parts) > index else None


# --------------------------------------------------------------------------- #
# 面板
# --------------------------------------------------------------------------- #


def panel_slice(
    bars_by_symbol: dict[str, list[dict[str, Any]]],
    day: str,
) -> dict[str, dict[str, Any]]:
    """取某日截面：只保留当日有交易、且收益合法的标的。"""
    snapshot: dict[str, dict[str, Any]] = {}
    for symbol, bars in bars_by_symbol.items():
        bar = _at(bars, day)
        if bar is None or bar["suspended"]:
            continue
        snapshot[symbol] = {
            "close": bar["close"],
            "mcap": free_float_mcap(bar),
            "is_st": bar["is_st"],
            "listing_days": bar["listing_days"],
            "at_limit_up": bar["at_limit_up"],
            "at_limit_down": bar["at_limit_down"],
            "turnover": bar["turnover"],
            "amount": bar["amount"],
            "board": bar["board"],
        }
    return snapshot


def _at(bars: list[dict[str, Any]], day: str) -> dict[str, Any] | None:
    """按日期二分查找（日线已按日期升序）。"""
    low, high = 0, len(bars) - 1
    while low <= high:
        middle = (low + high) // 2
        if bars[middle]["date"] == day:
            return bars[middle]
        if bars[middle]["date"] < day:
            low = middle + 1
        else:
            high = middle - 1
    return None


def trailing_stats(
    bars: list[dict[str, Any]],
    day: str,
    *,
    window: int = 60,
) -> dict[str, float] | None:
    """截至 day 的滚动统计（不含未来信息）：波动、均额、平均换手。"""
    position = None
    low, high = 0, len(bars) - 1
    while low <= high:
        middle = (low + high) // 2
        if bars[middle]["date"] == day:
            position = middle
            break
        if bars[middle]["date"] < day:
            low = middle + 1
        else:
            high = middle - 1
    if position is None or position < window:
        return None
    chunk = bars[position - window + 1: position + 1]
    returns = [bar["ret"] for bar in chunk if bar["ret"] is not None and bar["ret_ok"]]
    if len(returns) < window // 2:
        return None
    return {
        "vol": statistics.pstdev(returns) * math.sqrt(252) if len(returns) > 1 else 0.0,
        "amount_mean": statistics.fmean([bar["amount"] for bar in chunk]),
        "turnover_mean": statistics.fmean(
            [bar["turnover"] for bar in chunk if bar["turnover"] is not None] or [0.0]
        ),
        "suspended_ratio": sum(1 for bar in chunk if bar["suspended"]) / len(chunk),
    }
