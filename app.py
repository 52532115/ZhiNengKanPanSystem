"""Local A-share and ETF research dashboard server."""

from __future__ import annotations

import json
import mimetypes
import os
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from analyzer import analyze, market_regime, relative_strength
from market_data import BENCHMARKS, INDEXES, MarketDataClient, MarketDataError, normalize_symbol


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
DATA = MarketDataClient()

APP_VERSION = "1.4"
# 相对强弱因子使用的基准；与 /api/rs 的默认基准保持一致。
BENCHMARK_SYMBOL = "000300"
# 供前端和 /api/health 自检：旧进程没有这些路由时会以 JSON 404 明确报错，
# 而不是退回静态文件服务返回 HTML。
API_ROUTES = ("/api/health", "/api/analyze", "/api/search", "/api/rs", "/api/market", "/api/batch")

# 自选股组合面板：单次上限 20 个代码；真正的出站限流在 MarketDataClient 的
# 信号量里（同时最多 2 个上行请求），这里的线程池只负责让等待时间重叠。
MAX_BATCH_SYMBOLS = 20
BATCH_WORKERS = 4
POOL = ThreadPoolExecutor(max_workers=BATCH_WORKERS, thread_name_prefix="batch")
MAX_BODY_BYTES = 8 * 1024


def json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def analyze_symbol(symbol: str, limit: int = 320) -> dict[str, Any]:
    history = DATA.history(symbol, limit)
    try:
        # 基准只用于相对强弱因子；取不到时该因子记 0 分，不影响主分析。
        benchmark = DATA.history(BENCHMARK_SYMBOL, max(limit, 260), index=True)
    except MarketDataError:
        benchmark = None
    try:
        quote = DATA.quote(symbol)
    except MarketDataError:
        quote = None
    return analyze(history, quote, benchmark=benchmark)


def batch_item(symbol: str) -> dict[str, Any]:
    """One row of the watchlist panel: quote fields + rule-model score.

    异动只在"确实偏离常态"时置顶，避免把每只股票都标成异动：涨跌幅阈值按
    股票 5% / 场内基金 3% 区分，量比 2.2 倍以上视为放量，评分仅取极值。
    """
    result = analyze_symbol(symbol, 200)
    day_change = result["day_change"] or 0.0
    volume_ratio = result["metrics"]["volume_ratio"] or 1.0
    move_threshold = 3.0 if result["asset_type"] in ("etf", "fund") else 5.0
    alerts: list[str] = []
    if abs(day_change) >= move_threshold:
        alerts.append(f"涨跌幅 {day_change:+.2f}% 超过 {move_threshold:.0f}% 异动阈值")
    if volume_ratio >= 2.2:
        alerts.append(f"量比 {volume_ratio:.2f} 倍显著放大")
    if result["score"] >= 85:
        alerts.append(f"评分 {result['score']} 进入极强区间")
    elif result["score"] <= 15:
        alerts.append(f"评分 {result['score']} 进入极弱区间")
    return {
        "symbol": result["symbol"],
        "name": result["name"],
        "asset_type": result["asset_type"],
        "asset_label": result["asset_label"],
        "price": result["price"],
        "price_precision": result["price_precision"],
        "day_change": day_change,
        "score": result["score"],
        "view": result["view"],
        "tone": result["tone"],
        "alert": bool(alerts),
        "alert_reason": "；".join(alerts),
        "as_of": result["as_of"],
        "data_source": result["data_source"],
    }


def batch_payload(symbols: list[Any]) -> dict[str, Any]:
    cleaned: list[str] = []
    failed: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in symbols:
        try:
            code = normalize_symbol(str(raw))
        except ValueError as exc:
            failed.append({"symbol": str(raw), "error": str(exc)})
            continue
        if code in seen:
            continue
        seen.add(code)
        cleaned.append(code)

    def work(code: str) -> dict[str, Any]:
        try:
            return batch_item(code)
        except (MarketDataError, ValueError) as exc:
            return {"symbol": code, "error": str(exc)}

    items: list[dict[str, Any]] = []
    for row in POOL.map(work, cleaned):
        if "error" in row:
            failed.append(row)  # type: ignore[arg-type]
        else:
            items.append(row)

    items.sort(key=lambda item: (not item["alert"], -item["score"]))
    as_of = max((item["as_of"] for item in items), default=None)
    return {
        "items": items,
        "failed": failed,
        "as_of": as_of,
        "alert_count": sum(1 for item in items if item["alert"]),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = f"AShareResearch/{APP_VERSION}"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stdout.write(f"[{self.log_date_time_string()}] {fmt % args}\n")

    def version_string(self) -> str:
        """Expose the running version so a stale process is easy to spot."""
        return self.server_version

    def _send_json(self, status: int, payload: Any) -> None:
        body = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, route: str) -> None:
        relative = "index.html" if route == "/" else route.lstrip("/")
        target = (STATIC / relative).resolve()
        if STATIC.resolve() not in target.parents and target != STATIC.resolve():
            self.send_error(403)
            return
        if not target.is_file():
            self.send_error(404)
            return
        body = target.read_bytes()
        mime = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", f"{mime}; charset=utf-8" if mime.startswith("text/") else mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError as exc:
            raise ValueError("请求长度无效") from exc
        if length <= 0:
            raise ValueError("请求体为空")
        if length > MAX_BODY_BYTES:
            raise ValueError("请求体过大")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("请求体必须是合法的 JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if parsed.path == "/api/health":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "version": APP_VERSION,
                        "routes": list(API_ROUTES),
                        "features": ["search", "relative_strength", "rs_factor", "batch", "candlestick", "score_factors"],
                    },
                )
                return
            if parsed.path == "/api/analyze":
                symbol = normalize_symbol(query.get("symbol", ["600519"])[0])
                self._send_json(200, {"ok": True, "data": analyze_symbol(symbol)})
                return
            if parsed.path == "/api/search":
                keyword = (query.get("q", [""])[0] or "").strip()
                items = DATA.search(keyword, 8) if keyword else []
                self._send_json(200, {"ok": True, "data": {"query": keyword, "items": items}})
                return
            if parsed.path == "/api/rs":
                symbol = normalize_symbol(query.get("symbol", ["600519"])[0])
                benchmark = (query.get("benchmark", ["000300"])[0] or "000300").strip()
                if benchmark not in BENCHMARKS:
                    raise ValueError("基准仅支持 " + "、".join(f"{code} {name}" for code, (_, name) in BENCHMARKS.items()))
                days = int(query.get("days", ["120"])[0])
                days = max(40, min(days, 250))
                stock = DATA.history(symbol, days + 80)
                bench = DATA.history(benchmark, days + 80, index=True)
                self._send_json(200, {"ok": True, "data": relative_strength(stock, bench, days)})
                return
            if parsed.path == "/api/market":
                items = []
                for symbol, (_, name) in INDEXES.items():
                    result = analyze(DATA.history(symbol, 260, index=True))
                    items.append(
                        {
                            **result,
                            "name": name,
                            "chart": [],
                            "reasons": [],
                        }
                    )
                self._send_json(200, {"ok": True, "data": market_regime(items)})
                return
            if parsed.path.startswith("/api/"):
                self._send_json(
                    404,
                    {"ok": False, "error": f"接口不存在：{parsed.path}（当前服务版本 {APP_VERSION}，请重启 python app.py）"},
                )
                return
            self._serve_static(parsed.path)
        except ValueError as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
        except MarketDataError as exc:
            self._send_json(502, {"ok": False, "error": str(exc)})
        except Exception as exc:  # keep UI errors actionable without a debug traceback leak
            self._send_json(500, {"ok": False, "error": f"分析失败：{exc}"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/batch":
            self._send_json(
                404,
                {"ok": False, "error": f"接口不存在：{parsed.path}（当前服务版本 {APP_VERSION}，请重启 python app.py）"},
            )
            return
        try:
            payload = self._read_json_body()
            symbols = payload.get("symbols")
            if not isinstance(symbols, list) or not symbols:
                raise ValueError("symbols 必须是非空数组")
            if len(symbols) > MAX_BATCH_SYMBOLS:
                raise ValueError(f"单次最多查询 {MAX_BATCH_SYMBOLS} 个代码")
            self._send_json(200, {"ok": True, "data": batch_payload(symbols)})
        except ValueError as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
        except MarketDataError as exc:
            self._send_json(502, {"ok": False, "error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": f"批量查询失败：{exc}"})


def main() -> None:
    host = os.getenv("ASHARE_HOST", "127.0.0.1")
    port = int(os.getenv("ASHARE_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"A 股研究仪表盘已启动：http://{host}:{port}（v{APP_VERSION}）")
    print("按 Ctrl+C 停止服务")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
    finally:
        server.server_close()
        POOL.shutdown(wait=False)


if __name__ == "__main__":
    main()
