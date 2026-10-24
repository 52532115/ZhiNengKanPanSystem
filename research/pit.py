"""PIT 股票池：按日期取真实成分，过滤不可交易标的，再按行业×市值分层抽样。

三个容易做错的地方，这里都显式处理：

1. **成分必须按日期取**。用"今天的沪深300"回测 2018 年，等于只研究活到今天的
   赢家。``MembershipBook`` 只用快照日 ≤ 目标日的最近一期，绝不偷看下一期。
2. **退市股要在池子里**。成分快照本身包含后来退市/被剔除的票，只要它有当日
   行情就保留；同时提供"剔除已退市"的开关，用来量化幸存者偏差。
3. **抽样要分层**。按市值取前 N 只会得到清一色大盘，行业也会挤在两三个板块里；
   这里按"行业(CSRC 大类) × 自由流通市值三分位"分层，按层内标的数比例分配。
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Any, Callable, Iterable, Sequence

CSRC_L1 = {
    "A": "农林牧渔", "B": "采矿业", "C": "制造业", "D": "电力热力燃气水",
    "E": "建筑业", "F": "批发零售", "G": "交通运输仓储", "H": "住宿餐饮",
    "I": "信息技术服务", "J": "金融业", "K": "房地产", "L": "租赁商务服务",
    "M": "科学研究", "N": "水利环境公共设施", "O": "居民服务修理",
    "P": "教育", "Q": "卫生社会工作", "R": "文化体育娱乐", "S": "综合",
}

DEFAULT_TARGET = 400
MIN_LISTING_DAYS = 250        # 上市满一年才进池，避开次新股上市初期的定价异常
MIN_AMOUNT = 2e7              # 当日成交额下限（元），过滤流动性枯竭
MIN_MCAP = 1e9


# --------------------------------------------------------------------------- #
# 成分
# --------------------------------------------------------------------------- #


class MembershipBook:
    """按日期查询 PIT 成分（快照日 ≤ 目标日，最近一期）。"""

    def __init__(self, membership: dict[str, dict[str, list[str]]]) -> None:
        self.snapshots = {day: data for day, data in sorted(membership.items())}
        self.days = sorted(self.snapshots)

    def snapshot_for(self, day: str) -> str | None:
        low, high = 0, len(self.days) - 1
        found = None
        while low <= high:
            middle = (low + high) // 2
            if self.days[middle] <= day:
                found = self.days[middle]
                low = middle + 1
            else:
                high = middle - 1
        return found

    def members(self, day: str, index: str) -> list[str]:
        snapshot = self.snapshot_for(day)
        if snapshot is None:
            return []
        return list(self.snapshots[snapshot].get(index, []))

    def union(self, index: str) -> set[str]:
        return {code for data in self.snapshots.values() for code in data.get(index, [])}

    def ever_later_delisted(self, day: str, index: str, *, alive_end: set[str]) -> set[str]:
        """当日成分里、在样本末期已不在市（退市或被并购）的标的。"""
        return {code for code in self.members(day, index) if code not in alive_end}


# --------------------------------------------------------------------------- #
# 池子
# --------------------------------------------------------------------------- #


def industry_key(code: str, industry: dict[str, dict[str, Any]]) -> str:
    """证监会行业大类（如 ``C39``），缺失时用 ``UNKNOWN``。"""
    info = industry.get(code) or {}
    raw = (info.get("industry") or "").strip()
    if len(raw) >= 3 and raw[0].isalpha():
        return raw[:3]
    return "UNKNOWN"


def industry_label(key: str, industry: dict[str, dict[str, Any]]) -> str:
    if key == "UNKNOWN":
        return "未分类"
    return f"{CSRC_L1.get(key[0], key[0])}·{key}"


def build_pool(
    day: str,
    *,
    snapshot: dict[str, dict[str, Any]],
    members: Iterable[str] | None,
    industry: dict[str, dict[str, Any]],
    exclude_st: bool = True,
    exclude_limit: bool = False,
    min_listing_days: int = MIN_LISTING_DAYS,
    min_amount: float = MIN_AMOUNT,
    exclude_delisted: Callable[[str], bool] | None = None,
) -> list[dict[str, Any]]:
    """当日可交易池：成分 → 剔 ST/次新/停牌/低流动性 → 附带行业与市值。"""
    pool = []
    for code in (snapshot if members is None else members):
        info = snapshot.get(code)
        if info is None:                       # 当日无行情（停牌或已退市）
            continue
        if info["listing_days"] < min_listing_days:
            continue
        if info["is_st"] and exclude_st:
            continue
        if exclude_limit and (info["at_limit_up"] or info["at_limit_down"]):
            continue
        if info["amount"] < min_amount:
            continue
        if exclude_delisted is not None and exclude_delisted(code):
            continue
        pool.append(
            {
                "symbol": code,
                "date": day,
                "mcap": info["mcap"],
                "industry": industry_key(code, industry),
                "board": info["board"],
                "is_st": info["is_st"],
                "turnover": info["turnover"],
                "amount": info["amount"],
            }
        )
    return pool


def size_bucket(pool: Sequence[dict[str, Any]]) -> dict[str, str]:
    """自由流通市值三分位标签；市值缺失的单独一档。"""
    values = sorted(item["mcap"] for item in pool if item["mcap"])
    if len(values) < 6:
        return {item["symbol"]: "NA" for item in pool}
    low = values[len(values) // 3]
    high = values[2 * len(values) // 3]
    labels: dict[str, str] = {}
    for item in pool:
        if not item["mcap"]:
            labels[item["symbol"]] = "NA"
        elif item["mcap"] <= low:
            labels[item["symbol"]] = "小"
        elif item["mcap"] <= high:
            labels[item["symbol"]] = "中"
        else:
            labels[item["symbol"]] = "大"
    return labels


def stratified_sample(
    pool: Sequence[dict[str, Any]],
    *,
    target: int = DEFAULT_TARGET,
    seed: int = 20261021,
    size_weight: bool = True,
) -> list[str]:
    """按「行业 × 市值档」分层，按层内标的数比例抽 ``target`` 只。

    层内随机（固定种子），保证同一份数据、同一个种子抽出来的池子完全一致；
    ``size_weight=False`` 时忽略市值档，只按行业分层。
    """
    if not pool:
        return []
    labels = size_bucket(pool)
    strata: dict[tuple[str, str], list[str]] = {}
    for item in pool:
        key = (item["industry"], labels[item["symbol"]] if size_weight else "-")
        strata.setdefault(key, []).append(item["symbol"])
    total = sum(len(members) for members in strata.values())
    target = min(target, total)
    rng = random.Random(seed)

    # 先按比例给每个层分配名额（至少 1 只，避免小行业被整体抹掉）
    allocation = {
        key: max(1, round(len(members) / total * target)) for key, members in strata.items()
    }
    while sum(allocation.values()) > target:
        # 从超出比例最多的层里回收名额
        worst = max(allocation, key=lambda k: allocation[k] - len(strata[k]) / total * target)
        if allocation[worst] <= 1:
            worst = max(allocation, key=lambda k: allocation[k])
            if allocation[worst] <= 1:
                break
        allocation[worst] -= 1
    while sum(allocation.values()) < target:
        best = max(
            (k for k in strata if allocation[k] < len(strata[k])),
            key=lambda k: len(strata[k]) / total * target - allocation[k],
            default=None,
        )
        if best is None:
            break
        allocation[best] += 1

    picked: list[str] = []
    for key in sorted(strata):
        members = sorted(strata[key])
        count = min(allocation.get(key, 0), len(members))
        picked.extend(rng.sample(members, count))
    return sorted(picked)


# --------------------------------------------------------------------------- #
# 覆盖度与"有效宽度"
# --------------------------------------------------------------------------- #


def coverage(pool: Sequence[dict[str, Any]], industry: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """行业分布与市值分布：判断池子是不是挤在两三个行业里。"""
    counts: dict[str, int] = {}
    for item in pool:
        counts[item["industry"]] = counts.get(item["industry"], 0) + 1
    total = len(pool) or 1
    shares = sorted((count / total for count in counts.values()), reverse=True)
    mcaps = [item["mcap"] for item in pool if item["mcap"]]
    return {
        "n": len(pool),
        "industries": len(counts),
        "top3_share": round(sum(shares[:3]) * 100, 1),
        "hhi": round(sum(share ** 2 for share in shares), 4),
        "top": [
            {
                "industry": industry_label(key, industry),
                "count": count,
                "share": round(count / total * 100, 1),
            }
            for key, count in sorted(counts.items(), key=lambda pair: -pair[1])[:5]
        ],
        "mcap_p50_亿": round(statistics.median(mcaps) / 1e8, 1) if mcaps else None,
        "mcap_p10_亿": round(sorted(mcaps)[len(mcaps) // 10] / 1e8, 1) if mcaps else None,
        "mcap_p90_亿": round(sorted(mcaps)[9 * len(mcaps) // 10] / 1e8, 1) if mcaps else None,
    }


def breadth_metrics(
    symbols: Sequence[str],
    returns: dict[str, list[float]],
    *,
    window: int = 60,
) -> dict[str, Any]:
    """持仓的有效宽度。

    - ``effective_holdings`` = 1/Σw²（等权时等于持仓数），回答"真正分散在几只上"；
    - ``grinold_breadth`` = N/(1+(N-1)ρ̄)，ρ̄ 为窗口内两两相关系数均值，回答
      "相关性扣掉以后还剩几个独立注"；选股 IR 的上限由它决定；
    - ``corr_mean`` 同时报告，方便解释为什么广度会塌。
    """
    n = len(symbols)
    if n == 0:
        return {"n": 0, "effective_holdings": 0, "grinold_breadth": 0, "corr_mean": None}
    weight = 1.0 / n
    effective = 1.0 / (n * weight ** 2)
    series = {symbol: returns.get(symbol, [])[-window:] for symbol in symbols}
    length = min((len(values) for values in series.values()), default=0)
    corr = None
    if n > 1 and length >= 10:
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                left = series[symbols[i]][-length:]
                right = series[symbols[j]][-length:]
                value = _corr(left, right)
                if value is not None:
                    pairs.append(value)
        if pairs:
            corr = statistics.fmean(pairs)
    grinold = n / (1 + (n - 1) * corr) if corr is not None and (1 + (n - 1) * corr) > 0 else None
    return {
        "n": n,
        "effective_holdings": round(effective, 1),
        "grinold_breadth": round(grinold, 1) if grinold else None,
        "corr_mean": round(corr, 3) if corr is not None else None,
    }


def _corr(left: Sequence[float], right: Sequence[float]) -> float | None:
    n = min(len(left), len(right))
    if n < 10:
        return None
    xs, ys = list(left[-n:]), list(right[-n:])
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    if dx <= 0 or dy <= 0:
        return None
    return num / (dx * dy)


def survivors_bias(membership: MembershipBook, index: str, *, alive_end: set[str]) -> dict[str, Any]:
    """量化"当前成分"与"PIT 成分"的差距：有多少票在中途被换掉/退市。"""
    if not membership.days:
        return {}
    first, last = membership.days[0], membership.days[-1]
    start_members = set(membership.members(first, index))
    end_members = set(membership.members(last, index))
    removed = start_members - end_members
    return {
        "index": index,
        "snapshots": len(membership.days),
        "start": first,
        "end": last,
        "start_size": len(start_members),
        "end_size": len(end_members),
        "turnover_names": len(removed),
        "turnover_share": round(len(removed) / max(len(start_members), 1) * 100, 1),
        # alive_end 为空表示"没有全市场在市名单"，退市数无法判定，宁可返回 None
        "delisted_among_removed": len(removed - alive_end) if alive_end else None,
    }
