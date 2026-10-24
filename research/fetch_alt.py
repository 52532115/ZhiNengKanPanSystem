"""P2-10 数据落地：把五个维度的数据抓成规范长表，写进版本化数据集。

用法::

    python -m research.fetch_alt                       # 全部源（北向默认关）
    python -m research.fetch_alt --sources baostock_fundamental,baostock_events
    python -m research.fetch_alt --north               # 额外抓北向季度快照
    python -m research.fetch_alt --list                # 只看源能力清单

为什么单独一个入口
------------------
抓取是慢且会限流的一步（财务 8000+ 次请求、资金流上千次分页），分析是快且要反复
迭代的一步。分开之后，``fundamental_study.py`` 可以反复改口径而不用重抓；
``--list`` 让"哪个维度没数据、为什么"在第一秒就能看到。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .datasets import FetchContext, catalog, collect
from .manifest import ROOT

STOCK_CACHE = ROOT / ".cache" / "fields" / "stocks"
ETF_CACHE = ROOT / ".cache" / "fields" / "etf"
OUT = ROOT / ".cache" / "alt_data"
ALL_SOURCES = ("baostock_fundamental", "baostock_events", "eastmoney_flows", "csv_drop")


def symbols() -> list[str]:
    return sorted(path.stem for path in STOCK_CACHE.glob("*.json"))


def calendar() -> list[str]:
    payload = json.loads((ETF_CACHE / "510300.json").read_text(encoding="utf-8"))
    return [row["date"] for row in payload["rows"]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P2-10 数据落地")
    parser.add_argument("--sources", default=",".join(ALL_SOURCES))
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--north", action="store_true", help="抓北向季度快照（请求量大）")
    parser.add_argument("--north-mode", default="quarter", choices=("quarter", "month"))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard", default=None, help="分片 k/n，用于并行抓取（k 从 0 计）")
    args = parser.parse_args(argv)

    if args.list:
        print(json.dumps(catalog(), ensure_ascii=False, indent=2))
        return 0

    days = calendar()
    first, last = days[0], days[-1]
    start = max(args.start, first)
    end = args.end or last
    names = [name.strip() for name in args.sources.split(",") if name.strip()]
    universe = symbols()
    if args.shard:
        index, total = (int(part) for part in args.shard.split("/"))
        universe = [symbol for position, symbol in enumerate(universe) if position % total == index]
    ctx = FetchContext(
        symbols=universe, start=start, end=end, cache=OUT,
        limit=args.limit, log=lambda text: print(text, flush=True),
        extra={"calendar": days, "north_snapshots": args.north_mode},
    )
    ctx.extra["north"] = bool(args.north)

    print(f"标的 {len(ctx.symbols)} 只，区间 {start} → {end}，源：{', '.join(names)}", flush=True)
    began = time.time()
    result = collect(names, ctx)
    elapsed = time.time() - began

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "start": start, "end": end, "symbols": len(ctx.symbols),
            "sources": names, "seconds": round(elapsed, 1),
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "report": result["report"],
        "records": result["records"],
    }
    path = OUT / "records.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    print(f"\n完成：{len(result['records'])} 条记录，{elapsed:.0f}s → {path}", flush=True)
    for name, entry in result["report"].items():
        status = entry.get("status")
        detail = entry.get("records", entry.get("error", entry.get("reason", "")))
        print(f"  {name:<22} {status:<8} {detail}", flush=True)
    by_dimension: dict[str, int] = {}
    for record in result["records"]:
        by_dimension[record["dimension"]] = by_dimension.get(record["dimension"], 0) + 1
    print("  维度分布：" + ", ".join(f"{k}={v}" for k, v in sorted(by_dimension.items())), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
