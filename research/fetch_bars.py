"""日线抓取的分片工作进程：把"一次拉几百只"拆成可并行、可中断、可核对的小步。

为什么单独开一个入口
--------------------
``research.fetch`` 的 ``--chunk/--resume`` 是**串行**的：单进程一次只发一个查询，
免费源每段 3 年日线要 3 秒左右，500 只就是两小时。瓶颈在网络往返而不是本地 CPU，
所以把标的按 ``index % n == k`` 切片，起 n 个进程各拉一片，再把结果合并。

**分片进程只写自己的日线文件，不碰 manifest**：``manifest.json`` 是读-改-写，
多进程同时写会互相覆盖。文件名不同所以日线本身没有冲突，收尾统一由
``reconcile()`` 扫目录、逐只登记行数与 sha256，这才有"每轮可核对"的账。

命令行::

    # 4 片并行，每片每次最多 120 只
    python -m research.fetch_bars --version pit-v1 --shard 0/4 --limit 120
    ...
    # 全部拉完后收尾：登记行数/sha256、更新 progress
    python -m research.fetch_bars --version pit-v1 --reconcile
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from .bs_source import BAR_FIELDS, BaoStockError, Client, DEFAULT_END, DEFAULT_START, to_bs
from .manifest import Dataset, latest_version


def universe_codes(dataset: Dataset, indexes: Sequence[str] = ("000300", "000905")) -> list[str]:
    """待抓清单：**按指数轮流**取，而不是先把沪深300 拉满再拉中证500。

    顺序会决定"只拉得动 300 只时拉到谁"。按指数内任期排序、两个指数交替取值，
    能保证样本里同时有沪深300 与中证500 的票——否则前 300 只全是沪深300，
    `zz500` 那一层在池层验证里会直接是空的，三指数分层就退化成一层。

    顺序必须确定：分片是在这个列表上取模的，换了顺序等于换了某一片的内容。
    """
    membership = dataset.read_table("membership")
    ranked: list[list[str]] = []
    for index in indexes:
        counts: dict[str, int] = {}
        for payload in membership.values():
            for code in payload.get(index, []) or []:
                counts[code] = counts.get(code, 0) + 1
        ranked.append(sorted(counts, key=lambda code: (-counts[code], code)))

    merged: list[str] = []
    seen: set[str] = set()
    depth = max((len(items) for items in ranked), default=0)
    for position in range(depth):
        for items in ranked:
            if position < len(items):
                code = items[position]
                if code not in seen:
                    seen.add(code)
                    merged.append(code)
    return merged


def write_bars_atomic(dataset: Dataset, code: str, rows: Iterable[tuple[str, ...]]) -> int:
    """先写 ``.tmp`` 再改名：进程被杀时不会留下半截 gzip 被当成"已下载"。"""
    path = dataset.bar_path(code)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with gzip.open(tmp, "wt", encoding="utf-8", newline="") as handle:
        handle.write(",".join(dataset.BAR_COLUMNS) + "\n")
        for row in rows:
            handle.write(",".join(row) + "\n")
            count += 1
    os.replace(tmp, path)
    return count


def fetch_shard(
    version: str,
    *,
    shard: str = "0/1",
    limit: int | None = None,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    segment_years: int = 3,
    pause: float = 0.2,
    indexes: Sequence[str] = ("000300", "000905"),
    retries: int = 1,
    timeout: float = 18.0,
    log=print,
) -> dict[str, Any]:
    """拉本片内尚未下载的日线；返回本轮的战果（供外层核对）。

    ``retries`` 默认只试 1 次、``timeout`` 18 秒：批量抓取时"某只票反复超时"比
    "失败一次留待下轮"更贵——免费源偶发整段不回包，把重试预算花在同一只上会让
    整个进程卡住。失败的标的记进 ``errors``，下一轮自然会再排到。
    """
    dataset = Dataset.open(version)
    k, _, n = shard.partition("/")
    index, total = int(k), max(1, int(n or 1))
    codes = universe_codes(dataset, indexes)
    todo = [code for code in codes if not dataset.has_bars(code)]
    mine = [code for position, code in enumerate(todo) if position % total == index]
    if limit:
        mine = mine[:limit]
    log(f"[{index}/{total}] 待拉 {len(todo)} 只，本片本轮 {len(mine)} 只")
    done = failed = 0
    errors: list[dict[str, str]] = []
    started = time.time()
    with Client(pause=pause, retries=max(1, retries)) as client:
        socket.setdefaulttimeout(max(5.0, timeout))
        for code in mine:
            try:
                rows = client.bars(code, start, end, segment_years=segment_years)
            except BaoStockError as exc:
                failed += 1
                errors.append({"symbol": code, "error": str(exc)[:160]})
                log(f"[{index}/{total}] 失败 {code}：{str(exc)[:80]}")
                continue
            if not rows:                       # 全区间无行情：上市更晚或已退市
                write_bars_atomic(dataset, code, [])
                failed += 1
                errors.append({"symbol": code, "error": "empty"})
                continue
            write_bars_atomic(dataset, code, [tuple(row[:11]) for row in rows])
            done += 1
            if done % 20 == 0:
                rate = (time.time() - started) / done
                log(f"[{index}/{total}] {done}/{len(mine)}（最近 {code} {len(rows)} 行，{rate:.1f}s/只）")
    return {
        "shard": shard, "requested": len(mine), "done": done, "failed": failed,
        "errors": errors[:20], "seconds": round(time.time() - started, 1),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def fetch_alive(version: str, *, day: str | None = None, log=print) -> dict[str, Any]:
    """抓样本末期"在全市场名单里"的标的一次，用来判定谁真的退市了。

    没有这份名单时，"中途被换出的票"里有多少是退市、多少只是调出指数，根本分不开，
    报告里的「其中退市」就只能写"未判定"。一次 ``query_all_stock`` 就能把它变成实数。
    """
    dataset = Dataset.open(version)
    if day is None:
        calendar = json.loads((dataset.root / "calendar.json").read_text(encoding="utf-8"))
        day = calendar[-1]
    with Client(pause=0.2, retries=2) as client:
        codes = client.listed(day)
    dataset.write_table("alive", {"date": day, "symbols": sorted(codes)},
                        source="baostock:query_all_stock", rows=len(codes))
    log(f"在市名单 {day}：{len(codes)} 只")
    return {"date": day, "symbols": len(codes)}


def reconcile(version: str, *, log=print) -> dict[str, Any]:
    """扫日线目录 → 登记每只的行数与 sha256 → 更新 progress（单进程收尾）。"""
    dataset = Dataset.open(version)
    payload = dataset.manifest()
    index: dict[str, dict[str, Any]] = {}
    empty: list[str] = []
    for path in sorted((dataset.root / "bars").glob("*.csv.gz")):
        code = path.name.replace("_", ".").replace(".csv.gz", "")
        rows = dataset.read_bars(code)
        if not rows:
            empty.append(code)
        index[code] = {
            "rows": len(rows),
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
            "first": rows[0]["date"] if rows else None,
            "last": rows[-1]["date"] if rows else None,
        }
    payload["bar_index"] = index
    payload["bars_total"] = len(index)
    progress = payload.get("progress") or {}
    progress.update({
        "bars_done": len(index),
        "bars_empty": len(empty),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    })
    payload["progress"] = progress
    if (dataset.root / "industry.json").is_file():
        payload.setdefault("files", {}).setdefault("industry", {})
    dataset.write_manifest(payload)
    stats = {
        "version": version, "bars": len(index), "empty": len(empty),
        "rows_total": sum(item["rows"] for item in index.values()),
        "reconciled_at": progress["updated_at"],
    }
    log(json.dumps(stats, ensure_ascii=False))
    return stats


def _sha(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="分片拉 PIT 日线（可并行、可续跑）")
    parser.add_argument("--version", default=None)
    parser.add_argument("--shard", default="0/1", help="第 k 片 / 共 n 片，如 0/4")
    parser.add_argument("--limit", type=int, default=None, help="本轮本片最多拉多少只")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--segment-years", type=int, default=3)
    parser.add_argument("--pause", type=float, default=0.2)
    parser.add_argument("--retries", type=int, default=1, help="单只重试次数（批量抓取时越少越快）")
    parser.add_argument("--timeout", type=float, default=18.0, help="单次查询的 socket 超时秒数")
    parser.add_argument("--indexes", default="000300,000905")
    parser.add_argument("--reconcile", action="store_true", help="只做收尾登记")
    parser.add_argument("--alive", action="store_true", help="抓样本末期的全市场在市名单")
    args = parser.parse_args(argv)
    version = args.version or latest_version()
    if not version:
        print("没有可用的数据版本，先跑 python -m research.fetch", file=sys.stderr)
        return 2
    if args.reconcile:
        reconcile(version)
        return 0
    if args.alive:
        print(json.dumps(fetch_alive(version), ensure_ascii=False))
        return 0
    try:
        report = fetch_shard(
            version,
            shard=args.shard,
            limit=args.limit,
            start=args.start,
            end=args.end,
            segment_years=max(1, args.segment_years),
            pause=max(0.0, args.pause),
            retries=max(1, args.retries),
            timeout=max(5.0, args.timeout),
            indexes=[item.strip() for item in args.indexes.split(",") if item.strip()],
        )
    except (BaoStockError, FileNotFoundError) as exc:
        print(f"分片抓取失败：{exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
