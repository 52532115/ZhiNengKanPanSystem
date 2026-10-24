"""抓取层：baostock 提供 PIT 成分与后复权日线，Eastmoney/Tencent 只做交叉校验。

为什么换数据源
--------------
前一版用东方财富当前快照 + 现存标的列表回测，等于只看得见"活到今天的赢家"；
而且东方财富免费接口没有历史成分，批量长历史还会限流。baostock 恰好补上三块：

- ``query_hs300_stocks(date)`` / ``query_zz500_stocks(date)``：**按日期**的真实
  成分，2020 年初的 300 只里有 114 只今天已不在名单里，这就是幸存者偏差的量级；
- ``query_history_k_data_plus(adjustflag="1")``：**后复权**日线，带 ``turn``
  （换手率）、``tradestatus``（停牌）、``isST``，量与额齐全；
- ``query_stock_industry()``：证监会行业分类（含退市股的历史归类）。

本模块是研究层唯一 import ``baostock`` 的地方；抓下来的东西落进
``research.manifest.Dataset``，回测与清洗都不再依赖第三方库。

命令行::

    python -m research.fetch --start 2016-01-01 --end 2026-10-21
    python -m research.fetch --resume --version pit-20261021T1400
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from datetime import date, datetime, timedelta
from typing import Any, Iterable

from .manifest import CACHE_ROOT, Dataset, latest_version

INDEXES = {
    "000300": ("沪深300", "query_hs300_stocks"),
    "000905": ("中证500", "query_zz500_stocks"),
}
BAR_FIELDS = (
    "date,open,high,low,close,preclose,volume,amount,turn,tradestatus,isST,pctChg"
)
DEFAULT_START = "2016-01-01"     # 2018 年评估期前留两年做动量/均线预热
DEFAULT_END = date.today().isoformat()


class BaoStockError(RuntimeError):
    pass


def _load_baostock():
    try:
        import baostock as bs  # noqa: PLC0415  研究层可选依赖
    except ImportError as exc:  # pragma: no cover - 依赖缺失时的提示
        raise BaoStockError(
            "研究层抓取需要 baostock："
            "python -m pip install --target <ASCII路径> baostock，"
            "再把该路径加进 PYTHONPATH"
        ) from exc
    return bs


# --------------------------------------------------------------------------- #
# 代码格式
# --------------------------------------------------------------------------- #


def to_bs(symbol: str) -> str:
    """``600519`` → ``sh.600519``；已带前缀的原样返回。"""
    if "." in symbol:
        return symbol
    if symbol.startswith(("5", "6", "9")):
        return f"sh.{symbol}"
    if symbol.startswith(("4", "8")):
        return f"bj.{symbol}"
    return f"sz.{symbol}"


def to_local(code: str) -> str:
    return code.split(".", 1)[1] if "." in code else code


def market_of(code: str) -> str:
    return code.split(".", 1)[0] if "." in code else ""


# --------------------------------------------------------------------------- #
# 基础查询
# --------------------------------------------------------------------------- #


class Client:
    """对 baostock 的薄封装：统一错误码、重试与节流。"""

    def __init__(self, *, pause: float = 0.02, retries: int = 3) -> None:
        # 免费源偶尔整段不回包，给 socket 一个默认超时，避免进程无限挂住
        socket.setdefaulttimeout(30)
        self.bs = _load_baostock()
        self.pause = pause
        self.retries = retries
        result = self.bs.login()
        if result.error_code != "0":
            raise BaoStockError(f"登录失败：{result.error_msg}")

    def close(self) -> None:
        try:
            self.bs.logout()
        except Exception:  # pragma: no cover - 退出阶段不抛
            pass

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _drain(self, result) -> list[list[str]]:
        rows: list[list[str]] = []
        while result.next():
            rows.append(result.get_row_data())
        return rows

    def query(self, method: str, **kwargs: Any) -> list[list[str]]:
        last = ""
        for attempt in range(self.retries):
            try:
                result = getattr(self.bs, method)(**kwargs)
                if result.error_code != "0":
                    last = result.error_msg
                else:
                    rows = self._drain(result)
                    time.sleep(self.pause)
                    return rows
            except Exception as exc:  # 网络抖动：退避重试
                last = str(exc)
            time.sleep(0.4 * (attempt + 1))
        raise BaoStockError(f"{method} 失败：{last}")

    # ---------------- 具体接口 ---------------- #
    def trading_days(self, start: str, end: str) -> list[str]:
        rows = self.query("query_trade_dates", start_date=start, end_date=end)
        return [row[0] for row in rows if row[1] == "1"]

    def index_members(self, index: str, day: str) -> list[str]:
        method = INDEXES[index][1]
        rows = self.query(method, date=day)
        return sorted(to_local(row[1]) for row in rows if len(row) > 1)

    def listed(self, day: str) -> list[str]:
        """当日全市场在市 A 股（用于构造小盘层抽样池）。"""
        rows = self.query("query_all_stock", day=day)
        return sorted(
            to_local(row[0])
            for row in rows
            if len(row) > 0 and _is_ashare(row[0])
        )

    def bars(self, code: str, start: str, end: str, *, segment_years: int = 3) -> list[list[str]]:
        """分段取日线。整段十年一次的响应体太大，免费源经常整段不回包；
        拆成三年一段后单次行数降到千行级，成功率明显高，且行序仍是升序。"""
        rows: list[list[str]] = []
        cursor = start
        while cursor <= end:
            stop = _add_years(cursor, segment_years) if segment_years else end
            stop = min(stop, end)
            chunk = self.query(
                "query_history_k_data_plus",
                code=to_bs(code),
                fields=BAR_FIELDS,
                start_date=cursor,
                end_date=stop,
                frequency="d",
                adjustflag="1",      # 1 = 后复权（计算口径）；2 = 前复权（仅展示）
            )
            rows.extend(chunk)
            if stop == end:
                break
            cursor = _next_day(stop)
        return rows

    def industry(self) -> list[list[str]]:
        return self.query("query_stock_industry")

    def basic(self, code: str) -> list[str] | None:
        rows = self.query("query_stock_basic", code=to_bs(code))
        return rows[0] if rows else None


def _is_ashare(code: str) -> bool:
    """过滤指数、B 股、基金：只留沪深 A 股主板/中小/创业/科创。"""
    market, _, symbol = code.partition(".")
    if market not in ("sh", "sz"):
        return False
    if market == "sh":
        return symbol.startswith(("600", "601", "603", "605", "688", "689"))
    return symbol.startswith(("000", "001", "002", "003", "300", "301"))


def _add_years(day: str, years: int) -> str:
    year, month, day_of_month = (int(part) for part in day.split("-"))
    target = year + years
    try:
        return datetime(target, month, day_of_month).strftime("%Y-%m-%d")
    except ValueError:                       # 2 月 29 日
        return datetime(target, month, 28).strftime("%Y-%m-%d")


def _next_day(day: str) -> str:
    return (datetime.strptime(day, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")


def month_ends(days: Iterable[str]) -> list[str]:
    """每个自然月的最后一个交易日，用作成分快照日。"""
    picked: dict[str, str] = {}
    for day in days:
        picked[day[:7]] = day
    return [picked[key] for key in sorted(picked)]


# --------------------------------------------------------------------------- #
# 抓取编排
# --------------------------------------------------------------------------- #


def fetch_dataset(
    *,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    indexes: Iterable[str] = ("000300", "000905"),
    version: str | None = None,
    resume: bool = False,
    bar_limit: int | None = None,
    chunk: int | None = None,
    max_bars: int | None = None,
    listed_stride: int = 3,
    pause: float = 1.0,
    log=print,
) -> Dataset:
    """抓一份完整的 PIT 数据集：日历 + 成分快照 + 全市场名单 + 日线 + 行业。

    **按小步落盘、可随时中断**：成分快照与全市场名单都先写成 ``parts/`` 下的
    单日文件，进程被杀（数据源限流时会挂住）也不会丢掉已完成的查询；
    ``chunk`` 限制本次最多发多少个请求，配合外层 ``timeout`` 与 ``--resume``
    就能在一个会抽风的免费源上把数据慢慢攒齐。
    """
    indexes = list(indexes)
    if resume:
        version = version or latest_version()
        if not version:
            raise BaoStockError("没有可续跑的版本，先跑一次完整抓取")
        dataset = Dataset.open(version)
        log(f"续跑版本 {version}")
    else:
        dataset = Dataset.create(
            version=version,
            params={
                "start": start, "end": end, "indexes": indexes,
                "adjust": "hfq(adjustflag=1)", "snapshot": "month-end trading day",
            },
        )
        log(f"新版本 {dataset.version}")

    parts = dataset.root / "parts"
    (parts / "membership").mkdir(parents=True, exist_ok=True)
    (parts / "listed").mkdir(parents=True, exist_ok=True)
    budget = {"queries": 0, "limit": chunk or 10 ** 9}

    def spend() -> bool:
        budget["queries"] += 1
        return budget["queries"] < budget["limit"]

    calendar_path = dataset.root / "calendar.json"
    with Client(pause=pause) as client:
        if calendar_path.is_file():
            days = json.loads(calendar_path.read_text(encoding="utf-8"))
            log(f"复用交易日历 {len(days)} 个")
        else:
            days = client.trading_days(start, end)
            dataset.write_table("calendar", days, source="baostock:query_trade_dates", rows=len(days))
            dataset.register_source("baostock", api="query_trade_dates", start=start, end=end)
            log(f"交易日 {len(days)} 个：{days[0]} → {days[-1]}")
            if not spend():
                dataset.update_manifest(progress=_progress(dataset, days))
                return dataset

        snapshots = month_ends(days)
        pending = [day for day in snapshots
                   if not (parts / "membership" / f"{day}.json").is_file()]
        log(f"成分快照：{len(snapshots) - len(pending)}/{len(snapshots)} 已完成")
        for day in pending:
            payload = {"date": day}
            for index in indexes:
                payload[index] = client.index_members(index, day)
                budget["queries"] += 1
                if not spend():
                    break
            (parts / "membership" / f"{day}.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            if budget["queries"] >= budget["limit"]:
                log(f"达到 chunk {chunk}，先落盘退出")
                dataset.update_manifest(progress=_progress(dataset, days))
                return dataset

        listed_targets = snapshots[::max(1, listed_stride)]
        pending_listed = [day for day in listed_targets
                          if not (parts / "listed" / f"{day}.json").is_file()]
        log(f"全市场名单：{len(listed_targets) - len(pending_listed)}/{len(listed_targets)} 已完成")
        for day in pending_listed:
            codes = client.listed(day)
            (parts / "listed" / f"{day}.json").write_text(
                json.dumps(codes), encoding="utf-8"
            )
            if not spend():
                log(f"达到 chunk {chunk}，先落盘退出")
                dataset.update_manifest(progress=_progress(dataset, days))
                return dataset

        membership = {
            path.stem: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((parts / "membership").glob("*.json"))
        }
        dataset.write_table("membership", membership,
                            source="baostock:query_hs300/zz500_stocks", rows=len(membership))
        listed = {
            path.stem: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((parts / "listed").glob("*.json"))
        }
        if listed:
            dataset.write_table("listed", listed, source="baostock:query_all_stock", rows=len(listed))

        # 日线按"在指数里出现的月数"排序：先保证沪深300 的 PIT 并集完整，
        # 中证500 的补充标的在后面。抓不完时至少 A/B 组对比还是干净的。
        in300: dict[str, int] = {}
        in500: dict[str, int] = {}
        for day in membership.values():
            for code in day.get("000300", []) or []:
                in300[code] = in300.get(code, 0) + 1
            for code in day.get("000905", []) or []:
                in500[code] = in500.get(code, 0) + 1
        universe = sorted(set(in300) | set(in500),
                          key=lambda code: (-in300.get(code, 0), -in500.get(code, 0), code))
        log(f"PIT 成分并集 {len(universe)} 只，拉后复权日线（可 --resume 续跑）")
        done = skipped = 0
        for code in universe:
            if dataset.has_bars(code):
                done += 1
                continue
            if budget["queries"] >= budget["limit"]:
                log(f"达到 chunk {chunk}，先落盘退出（日线 {done}/{len(universe)}）")
                dataset.finalize_bars()
                dataset.update_manifest(progress=_progress(dataset, days))
                return dataset
            try:
                rows = client.bars(code, start, end)
            except BaoStockError as exc:
                skipped += 1
                log(f"  跳过 {code}：{exc}")
                budget["queries"] += 1
                continue
            budget["queries"] += 1
            count = dataset.write_bars(code, [tuple(row[:11]) for row in rows])
            dataset.register_bars(code, count)
            done += 1
            if done % 100 == 0:
                log(f"  日线 {done}/{len(universe)}（最近 {code} {count} 行）")
            if max_bars and done >= max_bars:
                log(f"到达 --max-bars {max_bars}，剩 {len(universe) - done} 只留待下次续跑")
                break
            if bar_limit and done >= bar_limit:
                log(f"到达 --bar-limit {bar_limit}，停止")
                break

        if not (dataset.root / "industry.json").is_file():
            rows = client.industry()
            industry = {
                to_local(row[1]): {
                    "name": row[2], "industry": row[3],
                    "classification": row[4], "updated": row[0],
                }
                for row in rows if len(row) >= 5
            }
            dataset.write_table("industry", industry,
                                source="baostock:query_stock_industry", rows=len(industry))
            log(f"行业分类 {len(industry)} 条")

    dataset.finalize_bars()
    dataset.update_manifest(finished_at=datetime.now().isoformat(timespec="seconds"),
                            progress=_progress(dataset, days, done=True))
    log(f"完成：{dataset.root}（{dataset.manifest().get('bars_total')} 只日线）")
    return dataset


def _progress(dataset: Dataset, days: list[str], done: bool = False) -> dict[str, Any]:
    parts = dataset.root / "parts"
    total = len(month_ends(days))
    have = len(list((parts / "membership").glob("*.json")))
    bars = len(list((dataset.root / "bars").glob("*.csv.gz")))
    return {
        "snapshots_done": have, "snapshots_total": total,
        "bars_done": bars, "updated_at": datetime.now().isoformat(timespec="seconds"),
        "complete": bool(done and have >= total and bars > 0),
    }


def fetch_benchmark(
    *,
    version: str | None = None,
    codes: Iterable[str] = ("sh.000300", "sh.000905", "sh.000852"),
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    log=print,
) -> dict[str, Any]:
    """抓指数基准（不复权即可，指数无除权），供相对强弱因子与 regime 使用。"""
    version = version or latest_version()
    if not version:
        raise BaoStockError("先跑一次 python -m research.fetch")
    dataset = Dataset.open(version)
    payload: dict[str, Any] = {}
    with Client(pause=0.6) as client:
        for code in codes:
            rows = client.query(
                "query_history_k_data_plus", code=code,
                fields="date,open,high,low,close,preclose,volume,amount",
                start_date=start, end_date=end, frequency="d", adjustflag="3",
            )
            bars = [
                {
                    "date": row[0], "open": row[1], "high": row[2], "low": row[3],
                    "close": row[4], "preclose": row[5], "volume": row[6], "amount": row[7],
                    "turnover": None,
                }
                for row in rows if len(row) >= 8
            ]
            payload[to_local(code)] = {"symbol": to_local(code), "name": INDEXES.get(
                to_local(code), (to_local(code), ""))[0] if to_local(code) in INDEXES else to_local(code),
                "rows": bars}
            log(f"  {code} {len(bars)} 行")
    dataset.write_table("benchmark_by_code", payload, source="baostock:index_k_data",
                        rows=len(payload))
    # 默认基准 = 沪深300，单独落一份，回测入口直接读
    default = payload.get("000300")
    if default:
        (dataset.root / "benchmark.json").write_text(
            json.dumps(default, ensure_ascii=False), encoding="utf-8"
        )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="抓取 PIT 股票池数据集（研究层）")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--indexes", default="000300,000905")
    parser.add_argument("--version", default=None)
    parser.add_argument("--resume", action="store_true", help="续跑已有版本，跳过已下载的日线")
    parser.add_argument("--bar-limit", type=int, default=None, help="最多下载多少只（调试用）")
    parser.add_argument("--max-bars", type=int, default=None, help="本次最多新增多少只日线")
    parser.add_argument("--chunk", type=int, default=None, help="本次最多发多少个请求，用完就退出（配合 --resume 迭代）")
    parser.add_argument("--pause", type=float, default=1.0, help="每次查询后的停顿秒数，免费源要客气一点")
    parser.add_argument("--listed-stride", type=int, default=3, help="全市场名单快照间隔（月）")
    parser.add_argument("--with-benchmark", action="store_true", help="顺带抓指数基准")
    args = parser.parse_args(argv)
    try:
        dataset = fetch_dataset(
            start=args.start,
            end=args.end,
            indexes=[item.strip() for item in args.indexes.split(",") if item.strip()],
            version=args.version,
            resume=args.resume,
            bar_limit=args.bar_limit,
            chunk=args.chunk,
            max_bars=args.max_bars,
            pause=max(0.0, args.pause),
            listed_stride=max(1, args.listed_stride),
        )
    except BaoStockError as exc:
        print(f"抓取失败：{exc}", file=sys.stderr)
        return 2
    if args.with_benchmark:
        try:
            fetch_benchmark(version=dataset.version, start=args.start, end=args.end)
        except BaoStockError as exc:
            print(f"基准抓取失败：{exc}", file=sys.stderr)
    print(dataset.verify())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
