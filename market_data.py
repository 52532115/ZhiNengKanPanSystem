"""A-share and exchange-traded fund data using only the standard library.

The upstream endpoint is a public quote interface.  It is deliberately isolated in
this module so it can be replaced by a licensed data vendor without touching the
analysis or UI layers.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


class MarketDataError(RuntimeError):
    pass


INDEXES = {
    "000001": ("1.000001", "上证指数"),
    "399001": ("0.399001", "深证成指"),
    "399006": ("0.399006", "创业板指"),
}

# 相对强弱基准池。市场环境只遍历 INDEXES，因此向基准池追加品种不会改变
# /api/market 的口径。
BENCHMARKS = {
    **INDEXES,
    "000300": ("1.000300", "沪深300"),
    "000905": ("1.000905", "中证500"),
    "000852": ("1.000852", "中证1000"),
}


def normalize_symbol(raw: str) -> str:
    value = raw.strip().upper().replace("SH", "").replace("SZ", "").replace("BJ", "")
    if not (len(value) == 6 and value.isdigit()):
        raise ValueError("请输入 6 位证券代码，例如 600519 或 510300")
    if value[0] not in "01345689":
        raise ValueError("暂不支持该证券代码，仅支持沪深京 A 股和场内 ETF/基金")
    return value


def is_bse(symbol: str) -> bool:
    """北交所代码段：43/83/87/88，以及 2024 年启用的 92 段。

    92 段如果按首位 9 处理会落到沪市通道（1.920xxx），上游直接返回空数据。
    """
    return symbol.startswith(("43", "83", "87", "88", "92"))


def security_id(symbol: str) -> str:
    if is_bse(symbol):
        return f"0.{symbol}"
    market = "1" if symbol.startswith(("5", "6", "9")) else "0"
    return f"{market}.{symbol}"


def instrument_type(symbol: str) -> str:
    """Classify common exchange-traded instruments from their exchange code."""
    if symbol.startswith(("51", "56", "58", "15")):
        return "etf"
    if symbol.startswith(("16", "50")):
        return "fund"
    return "stock"


def clean_name(value: Any, fallback: str) -> str:
    """Remove spacing artifacts present in a few quote-source stock names."""
    if not isinstance(value, str) or not value.strip():
        return fallback
    return "".join(value.split())


@dataclass
class CacheItem:
    expires_at: float
    value: Any


class MarketDataClient:
    KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    QUOTE_URL = "https://push2.eastmoney.com/api/qt/stock/get"
    FALLBACK_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    SEARCH_URL = "https://searchapi.eastmoney.com/api/suggest/get"
    SEARCH_TOKEN = "D43BF722C8E33BDC906FB84D85E326E8"
    CONSTITUENT_URL = "https://push2.eastmoney.com/api/qt/clist/get"
    CONSTITUENT_BOARDS = {"hs300": ("BK0500", "沪深300")}
    # 只接受 A 股与场内基金；指数/港股/美股一律过滤，避免与个股代码冲突。
    SEARCH_CLASSIFY = {"AStock": "stock", "Fund": "fund"}

    def __init__(self, timeout: float = 8.0) -> None:
        self.timeout = timeout
        self._cache: dict[str, CacheItem] = {}
        self._lock = threading.Lock()
        # The public endpoint occasionally drops bursts of parallel connections.
        # Keep the dashboard's first-load fan-out polite and retry transient drops.
        self._outbound_slots = threading.BoundedSemaphore(2)

    def _get_json(self, url: str, params: dict[str, str]) -> dict[str, Any]:
        target = f"{url}?{urllib.parse.urlencode(params)}"
        last_error: Exception | None = None
        for attempt in range(3):
            request = urllib.request.Request(
                target,
                headers={
                    "User-Agent": "Mozilla/5.0 AShareResearchDashboard/1.0",
                    "Accept": "application/json,text/plain,*/*",
                    "Referer": "https://quote.eastmoney.com/",
                    "Connection": "close",
                },
            )
            try:
                with self._outbound_slots:
                    with urllib.request.urlopen(request, timeout=self.timeout) as response:
                        return json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ConnectionError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.25 * (attempt + 1))
        raise MarketDataError(f"行情服务暂时不可用：{last_error}") from last_error

    def _cached(self, key: str, ttl: int, loader: Any) -> Any:
        now = time.time()
        with self._lock:
            item = self._cache.get(key)
            if item and item.expires_at > now:
                return item.value
        value = loader()
        with self._lock:
            self._cache[key] = CacheItem(now + ttl, value)
        return value

    def _fallback_history(
        self,
        symbol: str,
        limit: int,
        *,
        index: bool = False,
        market: str | None = None,
    ) -> dict[str, Any]:
        if market is None:
            market = "1" if (index and symbol == "000001") or symbol.startswith(("5", "6", "9")) else "0"
        # 北交所备用源用 bj 前缀；沪市用 sh，其余用 sz。
        prefix = "bj" if is_bse(symbol) else ("sh" if market == "1" else "sz")
        key = f"{prefix}{symbol}"
        payload = self._get_json(
            self.FALLBACK_KLINE_URL,
            {"param": f"{key},day,,,{limit},qfq"},
        )
        data = (payload.get("data") or {}).get(key) or {}
        lines = data.get("qfqday") or data.get("day") or []
        quote_row = (data.get("qt") or {}).get(key) or []
        rows: list[dict[str, Any]] = []
        previous_close: float | None = None
        for line in lines:
            if len(line) < 6:
                continue
            try:
                open_price, close, high, low = map(float, line[1:5])
                volume = float(line[5])
            except (TypeError, ValueError):
                continue
            pct_change = ((close / previous_close - 1) * 100) if previous_close else 0.0
            amplitude = ((high - low) / previous_close * 100) if previous_close else 0.0
            rows.append(
                {
                    "date": line[0],
                    "open": open_price,
                    "close": close,
                    "high": high,
                    "low": low,
                    "volume": volume,
                    "amount": 0.0,
                    "amplitude": amplitude,
                    "pct_change": pct_change,
                    "change": close - previous_close if previous_close else 0.0,
                    "turnover": 0.0,
                }
            )
            previous_close = close
        if len(rows) < 60:
            raise MarketDataError("备用行情源返回的有效历史数据不足 60 个交易日")
        fallback_name = quote_row[1] if len(quote_row) > 1 else symbol
        return {
            "symbol": symbol,
            "name": clean_name(fallback_name, symbol),
            "rows": rows,
            "data_source": "fallback",
        }

    def history(self, symbol: str, limit: int = 300, *, index: bool = False) -> dict[str, Any]:
        symbol = normalize_symbol(symbol)
        limit = max(80, min(limit, 600))
        if index and symbol not in BENCHMARKS:
            raise ValueError("不支持的市场指数代码")
        secid = BENCHMARKS[symbol][0] if index else security_id(symbol)
        market = secid.split(".", 1)[0]
        asset_type = "index" if index else instrument_type(symbol)

        def load() -> dict[str, Any]:
            try:
                payload = self._get_json(
                    self.KLINE_URL,
                    {
                        "secid": secid,
                        "klt": "101",
                        "fqt": "1",
                        "lmt": str(limit),
                        "end": "20500101",
                        "fields1": "f1,f2,f3,f4,f5,f6,f7,f8",
                        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                    },
                )
            except MarketDataError:
                fallback = self._fallback_history(symbol, limit, index=index, market=market)
                return {**fallback, "asset_type": asset_type}
            data = payload.get("data")
            if not data or not data.get("klines"):
                raise MarketDataError("未找到该代码的行情，请确认它是已上市的 A 股或场内 ETF/基金")
            rows = []
            for line in data["klines"]:
                parts = line.split(",")
                if len(parts) < 11:
                    continue
                try:
                    rows.append(
                        {
                            "date": parts[0],
                            "open": float(parts[1]),
                            "close": float(parts[2]),
                            "high": float(parts[3]),
                            "low": float(parts[4]),
                            "volume": float(parts[5]),
                            "amount": float(parts[6]),
                            "amplitude": float(parts[7]),
                            "pct_change": float(parts[8]),
                            "change": float(parts[9]),
                            "turnover": float(parts[10]),
                        }
                    )
                except ValueError:
                    continue
            if len(rows) < 60:
                raise MarketDataError("有效历史行情不足 60 个交易日，暂时无法分析")
            return {
                "symbol": symbol,
                "name": clean_name(data.get("name"), symbol),
                "asset_type": asset_type,
                "data_source": "primary",
                "rows": rows,
            }

        return self._cached(f"history:{asset_type}:{symbol}:{limit}", 300, load)

    def quote(self, symbol: str) -> dict[str, Any]:
        symbol = normalize_symbol(symbol)

        def load() -> dict[str, Any]:
            payload = self._get_json(
                self.QUOTE_URL,
                {
                    "secid": security_id(symbol),
                    "fields": "f57,f58,f59,f43,f44,f45,f46,f47,f48,f60,f116,f117,f162,f167,f170,f171",
                },
            )
            data = payload.get("data") or {}

            # f59 是价格小数位数：A 股 2 位，场内基金 3 位。f43 等价格字段
            # 必须按 10**digits 缩放；统一除以 100 会把 ETF 价格放大 10 倍。
            try:
                digits = int(data.get("f59"))
            except (TypeError, ValueError):
                digits = 2
            if not 0 <= digits <= 4:
                digits = 2
            price_scale = 10 ** digits

            def scaled(field: str, divisor: float = 100.0, precision: int = 3) -> float | None:
                value = data.get(field)
                return None if value in (None, "-") else round(float(value) / divisor, precision)

            return {
                "symbol": data.get("f57") or symbol,
                "name": clean_name(data.get("f58"), symbol),
                "asset_type": instrument_type(symbol),
                "price": scaled("f43", price_scale, digits),
                "high": scaled("f44", price_scale, digits),
                "low": scaled("f45", price_scale, digits),
                "open": scaled("f46", price_scale, digits),
                "volume": data.get("f47"),
                "amount": data.get("f48"),
                "previous_close": scaled("f60", price_scale, digits),
                "market_cap": data.get("f116"),
                "float_market_cap": data.get("f117"),
                "pe_ttm": scaled("f162"),
                "pb": scaled("f167"),
                "pct_change": scaled("f170"),
                "amplitude": scaled("f171"),
            }

        return self._cached(f"quote:{symbol}", 20, load)

    def search(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        """Resolve a name / pinyin / partial code into tradeable symbols.

        The upstream suggest endpoint is best-effort: when it is unreachable we
        return an empty list so the search box degrades to "code only" silently
        instead of blocking the whole page with an error banner.
        """
        query = (query or "").strip()[:24]
        if not query:
            return []
        limit = max(1, min(limit, 12))

        def load() -> list[dict[str, Any]]:
            try:
                payload = self._get_json(
                    self.SEARCH_URL,
                    {"input": query, "type": "14", "token": self.SEARCH_TOKEN, "count": "12"},
                )
            except MarketDataError:
                return []
            rows = (payload.get("QuotationCodeTable") or {}).get("Data") or []
            items: list[dict[str, Any]] = []
            seen: set[str] = set()
            for row in rows:
                if row.get("Classify") not in self.SEARCH_CLASSIFY:
                    continue
                code = str(row.get("Code") or "").strip()
                if not (len(code) == 6 and code.isdigit()) or code[0] not in "01345689":
                    continue
                if code in seen:
                    continue
                seen.add(code)
                items.append(
                    {
                        "symbol": code,
                        "name": clean_name(row.get("Name"), code),
                        "asset_type": instrument_type(code),
                        "market": "沪市" if security_id(code).startswith("1.") else "深市",
                        "type_label": clean_name(row.get("SecurityTypeName"), ""),
                    }
                )
                if len(items) >= limit:
                    break
            return items

        return self._cached(f"search:{query}:{limit}", 600, load)

    def constituents(self, board: str = "hs300", limit: int = 300) -> list[dict[str, Any]]:
        """按市值降序取指数成分股，供 walk-forward 扩样本使用。

        上游 clist 接口单页最多 100 条，这里自动翻页并缓存一小时。
        """
        if board not in self.CONSTITUENT_BOARDS:
            raise ValueError(f"不支持的成分股池：{board}")
        board_id, board_name = self.CONSTITUENT_BOARDS[board]
        limit = max(1, min(limit, 500))

        def load() -> list[dict[str, Any]]:
            items: list[dict[str, Any]] = []
            seen: set[str] = set()
            page = 1
            while len(items) < limit:
                payload = self._get_json(
                    self.CONSTITUENT_URL,
                    {
                        "pn": str(page),
                        "pz": "100",
                        "po": "1",
                        "np": "1",
                        "fltt": "2",
                        "invt": "2",
                        "fid": "f20",
                        "fs": f"b:{board_id}",
                        "fields": "f12,f14,f20",
                    },
                )
                data = payload.get("data") or {}
                rows = data.get("diff") or []
                if not rows:
                    break
                for row in rows:
                    code = str(row.get("f12") or "").strip()
                    if not (len(code) == 6 and code.isdigit()) or code[0] not in "01345689":
                        continue
                    if code in seen:
                        continue
                    seen.add(code)
                    items.append(
                        {
                            "symbol": code,
                            "name": clean_name(row.get("f14"), code),
                            "market_cap": row.get("f20"),
                        }
                    )
                    if len(items) >= limit:
                        break
                total = int(data.get("total") or 0)
                if len(rows) < 100 or page * 100 >= total:
                    break
                page += 1
            if not items:
                raise MarketDataError(f"未取得{board_name}成分股列表")
            return items

        return self._cached(f"constituents:{board}:{limit}", 3600, load)
