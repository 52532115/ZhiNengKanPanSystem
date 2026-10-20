"""Local A-share and ETF research dashboard server."""

from __future__ import annotations

import json
import mimetypes
import os
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from analyzer import analyze, market_regime
from market_data import INDEXES, MarketDataClient, MarketDataError, normalize_symbol


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
DATA = MarketDataClient()


def json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "AShareResearch/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stdout.write(f"[{self.log_date_time_string()}] {fmt % args}\n")

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

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if parsed.path == "/api/health":
                self._send_json(200, {"status": "ok"})
                return
            if parsed.path == "/api/analyze":
                symbol = normalize_symbol(query.get("symbol", ["600519"])[0])
                history = DATA.history(symbol, 320)
                try:
                    quote = DATA.quote(symbol)
                except MarketDataError:
                    quote = None
                self._send_json(200, {"ok": True, "data": analyze(history, quote)})
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
            self._serve_static(parsed.path)
        except ValueError as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
        except MarketDataError as exc:
            self._send_json(502, {"ok": False, "error": str(exc)})
        except Exception as exc:  # keep UI errors actionable without a debug traceback leak
            self._send_json(500, {"ok": False, "error": f"分析失败：{exc}"})


def main() -> None:
    host = os.getenv("ASHARE_HOST", "127.0.0.1")
    port = int(os.getenv("ASHARE_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"A 股研究仪表盘已启动：http://{host}:{port}")
    print("按 Ctrl+C 停止服务")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
