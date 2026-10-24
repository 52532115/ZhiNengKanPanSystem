"""横截面分位建模与净值回测（离线研究链路，不影响线上评分）。

与 ``walkforward.py`` 的区别
---------------------------
``walkforward.py`` 验证的是"绝对分数 → 未来绝对收益"这条链，标签是收益本身，
样本按标的合并，因此混入了品种间差异。本模块换成标准的多因子横截面口径：

1. **标签**：t 日全股票池按未来 H 日收益排名，取横截面分位 ``(rank-1)/(n-1)``，
   而不是收益数值。这样每天的可比性来自"当日相对强弱"，天然剥离 beta 与
   行情整体涨跌，也不会被不同价位的股票放大。
2. **特征**：同样先转成当日横截面分位（0–1），再做线性拟合。分位化让特征
   对极端值、量纲、行业价格水平都不敏感。
3. **切分**：严格按时间顺序切训练 / 验证，并在边界做 purge（剔除标签跨越
   切分点的训练样本）+ embargo（额外隔离若干交易日）。
4. **回测**：组合层面计入双边成本、印花税、滑点与单次换手上限，先看净值
   曲线能否跑赢"同股票池等权持有"基准，再谈模型指标。

只依赖标准库。行情复用 ``walkforward`` 的本地缓存，避免重复开发数据层。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from analyzer import DEFAULT_WEIGHTS, ema, score_factors, sma
from market_data import MarketDataClient, MarketDataError
from walkforward import (
    CACHE_DIR,
    REPORT_DIR,
    _pearson,
    _rank,
    load_benchmark,
    load_history,
    load_universe,
)

# --------------------------------------------------------------------------- #
# 口径常量：报告里逐项展示，改这里就等于改口径
# --------------------------------------------------------------------------- #

HORIZON = 20          # 标签与持仓的目标周期（交易日）
WARMUP = 60           # 特征预热：MACD(26,9)、量比(20)、RS(60) 取最长窗口
MIN_XSEC = 20         # 当日至少多少只标的才构成一个有效横截面
SPLIT_RATIO = 0.6     # 时间顺序切分：前 60% 训练，后 40% 验证
EMBARGO = 5           # 切分点额外隔离的交易日

# 成本：A 股实际费率（单边）
COMMISSION_BPS = 2.5      # 佣金 万 2.5
TRANSFER_BPS = 0.1        # 过户费 0.001%
STAMP_DUTY_BPS = 5.0      # 印花税 0.05%，仅卖出
SLIPPAGE_BPS = 5.0        # 滑点（含冲击成本）0.05%
BUY_COST = (COMMISSION_BPS + TRANSFER_BPS + SLIPPAGE_BPS) / 10000      # 0.00076
SELL_COST = (COMMISSION_BPS + TRANSFER_BPS + SLIPPAGE_BPS + STAMP_DUTY_BPS) / 10000

DEFAULT_REBALANCE = 5     # 调仓间隔（交易日）
DEFAULT_TOP_FRAC = 0.2    # 每次持有当日得分最高的 20%
DEFAULT_MAX_TURNOVER = 0.4  # 单次调仓最多换掉 40% 的仓位（单边）

# 候选特征：三个因子族（上一轮唯一仍有微弱正 edge 的三项），
# 每族给两种连续化表达，最终表达由训练集 IC 决定，验证集不参与选择。
FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "macd": ("macd_hist", "macd_slope"),
    "volume": ("vol_ratio", "vol_sync"),
    "rs": ("rs20", "rs60"),
}
FEATURE_LABELS: dict[str, str] = {
    "macd_hist": "MACD 柱（占价格比）",
    "macd_slope": "MACD 柱斜率",
    "vol_ratio": "量比（20 日）",
    "vol_sync": "量价配合（量比×涨跌方向）",
    "rs20": "相对强弱（20 日超额）",
    "rs60": "相对强弱（60 日超额）",
}
ELLIPSIS = "—"


# --------------------------------------------------------------------------- #
# 数据读取
# --------------------------------------------------------------------------- #


def _symbol_series(history: dict[str, Any], bench: dict[str, dict[str, float]]) -> dict[str, Any]:
    """把一只标的的日线转换成因果特征序列。

    所有值都只用到 ``index`` 及之前的数据：EMA / SMA 本身就只回看，RS 用的是
    当日与前 20 / 60 个交易日的收盘价，量比用的是含当日的 20 日均量。
    """
    rows = history["rows"]
    closes = [row["close"] for row in rows]
    volumes = [row["volume"] for row in rows]
    dates = [row["date"] for row in rows]
    count = len(rows)

    dif = [fast - slow for fast, slow in zip(ema(closes, 12), ema(closes, 26))]
    dea = ema(dif, 9)
    hist = [(d - e) * 2 for d, e in zip(dif, dea)]
    volume_ma = sma(volumes, 20)

    series: dict[str, list[float | None]] = {
        key: [None] * count for key in FEATURE_LABELS
    }
    for index in range(count):
        close = closes[index]
        series["macd_hist"][index] = hist[index] / close * 100 if close else None
        if index >= 1 and close:
            series["macd_slope"][index] = (hist[index] - hist[index - 1]) / close * 100

        if volume_ma[index]:
            ratio = volumes[index] / volume_ma[index]
            series["vol_ratio"][index] = ratio
            change = rows[index]["pct_change"]
            direction = 1 if change > 0 else -1 if change < 0 else 0
            series["vol_sync"][index] = (ratio - 1) * direction

        if index >= 20:
            ret20 = (close / closes[index - 20] - 1) * 100
            bench20 = bench["ret20"].get(dates[index])
            if bench20 is not None:
                series["rs20"][index] = ret20 - bench20
        if index >= 60:
            ret60 = (close / closes[index - 60] - 1) * 100
            bench60 = bench["ret60"].get(dates[index])
            if bench60 is not None:
                series["rs60"][index] = ret60 - bench60

    return {
        "symbol": history["symbol"],
        "name": history["name"],
        "dates": dates,
        "closes": closes,
        "features": series,
    }


def _benchmark_returns(benchmark: dict[str, Any] | None) -> dict[str, dict[str, float]]:
    """基准的 20 / 60 日收益，按日期索引，供 RS 使用。"""
    result: dict[str, dict[str, float]] = {"ret20": {}, "ret60": {}}
    if not benchmark:
        return result
    rows = benchmark["rows"]
    closes = [row["close"] for row in rows]
    for index, row in enumerate(rows):
        if index >= 20 and closes[index - 20]:
            result["ret20"][row["date"]] = (closes[index] / closes[index - 20] - 1) * 100
        if index >= 60 and closes[index - 60]:
            result["ret60"][row["date"]] = (closes[index] / closes[index - 60] - 1) * 100
    return result


def load_histories(
    symbols: Iterable[str],
    days: int,
    *,
    refresh: bool = False,
    offline: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """只保留 A 股个股：ETF 与指数不参与横截面选股。"""
    client = MarketDataClient()
    histories: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for symbol in symbols:
        try:
            history = load_history(client, symbol, days, refresh=refresh, offline=offline)
        except (MarketDataError, ValueError) as exc:
            failures.append({"symbol": symbol, "error": str(exc)})
            continue
        if history.get("asset_type") != "stock":
            failures.append({"symbol": symbol, "error": f"{history.get('asset_type')} 不参与个股横截面"})
            continue
        if len(history["rows"]) < WARMUP + HORIZON + 5:
            failures.append({"symbol": symbol, "error": f"历史仅 {len(history['rows'])} 根，不足预热 + 标签"})
            continue
        histories.append(history)
    return histories, failures


# --------------------------------------------------------------------------- #
# 横截面面板：特征分位 + 分位标签
# --------------------------------------------------------------------------- #


def _rank_pct(values: Sequence[float]) -> list[float]:
    """把一组数值映射到 0–1 的分位，最小为 0、最大为 1（平均秩处理并列）。"""
    if not values:
        return []
    ranks = _rank(list(values))
    span = max(len(values) - 1, 1)
    return [(rank - 1) / span for rank in ranks]


def build_panel(
    histories: list[dict[str, Any]],
    benchmark: dict[str, Any] | None,
    *,
    horizon: int = HORIZON,
    warmup: int = WARMUP,
    min_xsec: int = MIN_XSEC,
) -> dict[str, Any]:
    """构造横截面面板。

    返回 ``records``（每个交易日的每只标的）与 ``dates``（有效横截面日期）。
    每条记录包含：原始特征、当日特征分位（``rank_*``）、未来 H 日收益及其分位。
    未来收益只用于训练标签与评估，不进入任何特征。
    """
    bench = _benchmark_returns(benchmark)
    prepared = [_symbol_series(history, bench) for history in histories]

    # 1) 先把原始特征按 (日期, 标的) 摊平
    raw: dict[str, dict[str, dict[str, Any]]] = {}
    for item in prepared:
        features = item["features"]
        for index, date in enumerate(item["dates"]):
            if index < warmup - 1:
                continue
            # 至少要有 MACD / 量比 / RS 三个基座特征，否则当天无法打分
            if not (
                features["macd_hist"][index] is not None
                and features["vol_ratio"][index] is not None
                and features["rs20"][index] is not None
            ):
                continue
            count = len(item["dates"])
            forward = None
            if index + horizon < count:
                entry = item["closes"][index]
                if entry:
                    forward = (item["closes"][index + horizon] / entry - 1) * 100
            raw.setdefault(date, {})[item["symbol"]] = {
                "symbol": item["symbol"],
                "name": item["name"],
                "fwd": forward,
                **{key: features[key][index] for key in FEATURE_LABELS},
            }

    # 2) 每个交易日内部做横截面分位（特征与标签各自独立分位）
    records: list[dict[str, Any]] = []
    dates: list[str] = []
    for date in sorted(raw):
        members = raw[date]
        if len(members) < min_xsec:
            continue
        symbols = sorted(members)
        dates.append(date)

        # 标签分位：只在当日有未来收益的标的之间排，避免末期样本被单独分位
        labeled = [symbol for symbol in symbols if members[symbol]["fwd"] is not None]
        label_rank: dict[str, float] = {}
        if len(labeled) >= min_xsec:
            for symbol, value in zip(
                labeled, _rank_pct([members[symbol]["fwd"] for symbol in labeled])
            ):
                label_rank[symbol] = value

        feature_rank: dict[str, dict[str, float]] = {}
        for key in FEATURE_LABELS:
            present = [symbol for symbol in symbols if members[symbol][key] is not None]
            if len(present) < min_xsec:
                feature_rank[key] = {}
                continue
            ranked = _rank_pct([members[symbol][key] for symbol in present])
            feature_rank[key] = dict(zip(present, ranked))

        for symbol in symbols:
            member = members[symbol]
            record = {
                "date": date,
                "symbol": symbol,
                "name": member["name"],
                "fwd": member["fwd"],
                "label": label_rank.get(symbol),
                "n_xsec": len(labeled),
            }
            for key in FEATURE_LABELS:
                record[key] = member[key]
                record[f"rank_{key}"] = feature_rank[key].get(symbol)
            records.append(record)

    return {"records": records, "dates": dates, "symbols": [h["symbol"] for h in histories]}


# --------------------------------------------------------------------------- #
# 时间顺序切分
# --------------------------------------------------------------------------- #


def time_split(
    dates: Sequence[str],
    *,
    split_ratio: float = SPLIT_RATIO,
    horizon: int = HORIZON,
    embargo: int = EMBARGO,
) -> dict[str, Any]:
    """按时间顺序切分，并剔除标签跨越边界 / 紧邻边界的训练样本。

    purge：训练集最后一个样本的标签终点必须早于验证集第一天，所以训练日期
    在切分点前 ``horizon`` 天截止；embargo 再往后让出 ``embargo`` 个交易日，
    避免停牌、复牌等慢变量把验证集信息带回训练集。
    """
    split_index = int(len(dates) * split_ratio)
    train_end = max(1, split_index - horizon - embargo)
    return {
        "split_index": split_index,
        "train_dates": list(dates[:train_end]),
        "valid_dates": list(dates[split_index:]),
        "purged": split_index - train_end,
        "train_start": dates[0] if dates else None,
        "train_end": dates[train_end - 1] if train_end else None,
        "valid_start": dates[split_index] if split_index < len(dates) else None,
        "valid_end": dates[-1] if dates else None,
    }


# --------------------------------------------------------------------------- #
# 特征筛选：只在训练集上算横截面 IC
# --------------------------------------------------------------------------- #


def _daily_ic(
    records: Sequence[dict[str, Any]],
    key: str,
    *,
    target: str = "fwd",
    horizon: int = HORIZON,
) -> dict[str, Any]:
    """逐日计算 (特征分位, 未来收益) 的 Spearman IC，再对日期取均值。"""
    by_date: dict[str, list[tuple[float, float]]] = {}
    for record in records:
        value = record.get(target)
        rank = record.get(f"rank_{key}") if target == "fwd" else record.get("label")
        if value is None or rank is None:
            continue
        by_date.setdefault(record["date"], []).append((float(rank), float(value)))

    daily: list[float] = []
    for pairs in by_date.values():
        if len(pairs) < MIN_XSEC:
            continue
        ic = _pearson(_rank([p[0] for p in pairs]), _rank([p[1] for p in pairs]))
        if ic is not None:
            daily.append(ic)

    if len(daily) < 10:
        return {"ic": None, "n_dates": len(daily), "t": None, "positive_rate": None}
    mean = statistics.fmean(daily)
    deviation = statistics.stdev(daily) if len(daily) > 1 else 0.0
    # 相邻日期的 H 日标签窗口重叠，有效样本按 horizon 折算
    effective = max(3, round(len(daily) / max(horizon, 1)))
    t_stat = mean / (deviation / math.sqrt(effective)) if deviation > 0 else None
    return {
        "ic": round(mean, 4),
        "n_dates": len(daily),
        "n_effective": effective,
        "t": round(t_stat, 2) if t_stat is not None else None,
        "positive_rate": round(sum(1 for v in daily if v > 0) / len(daily) * 100, 1),
    }


def screen_features(
    train_records: Sequence[dict[str, Any]],
    *,
    horizon: int = HORIZON,
) -> dict[str, Any]:
    """每个因子族保留训练集 IC 最高的表达；族内所有候选的 IC 都如实列出。"""
    families: list[dict[str, Any]] = []
    selected: dict[str, str] = {}
    for family, candidates in FEATURE_FAMILIES.items():
        rows = []
        for key in candidates:
            stat = _daily_ic(train_records, key, horizon=horizon)
            rows.append({"key": key, "label": FEATURE_LABELS[key], **stat})
        valid = [row for row in rows if row["ic"] is not None]
        best = max(valid, key=lambda row: row["ic"]) if valid else None
        chosen = best["key"] if best and best["ic"] > 0 else candidates[0]
        selected[family] = chosen
        for row in rows:
            row["selected"] = row["key"] == chosen
            row["note"] = (
                "训练集 IC 最高，入选"
                if row["selected"] and best and best["ic"] > 0
                else ("无正 IC，退回基准表达" if row["selected"] else "")
            )
        families.append({"family": family, "candidates": rows})
    return {"families": families, "selected": selected, "features": list(selected.values())}


# --------------------------------------------------------------------------- #
# 线性拟合（分位 → 分位），闭式解 + 极小岭项
# --------------------------------------------------------------------------- #


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    aug = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(aug[r][column]))
        if abs(aug[pivot][column]) < 1e-12:
            continue
        aug[column], aug[pivot] = aug[pivot], aug[column]
        divisor = aug[column][column]
        for c in range(column, size + 1):
            aug[column][c] /= divisor
        for r in range(size):
            if r == column:
                continue
            factor = aug[r][column]
            if not factor:
                continue
            for c in range(column, size + 1):
                aug[r][c] -= factor * aug[column][c]
    return [aug[i][size] for i in range(size)]


def fit_rank_model(
    records: Sequence[dict[str, Any]],
    features: Sequence[str],
    *,
    ridge: float = 1e-4,
) -> dict[str, Any]:
    """用 OLS 拟合 ``标签分位 ~ 特征分位``。

    特征与标签都是当日分位（均值约 0.5），因此截距近似 0.5、系数可以理解为
    "该特征分位每提高 1 个横截面档位，标签分位平均提高多少"。岭项只用于
    防止共线导致的正则矩阵不可逆。
    """
    columns = len(features) + 1
    xtx = [[0.0] * columns for _ in range(columns)]
    xty = [0.0] * columns
    rows = 0
    for record in records:
        if record.get("label") is None:
            continue
        values = [record.get(f"rank_{key}") for key in features]
        if any(value is None for value in values):
            continue
        vector = [1.0] + [float(value) for value in values]
        label = float(record["label"])
        rows += 1
        for i in range(columns):
            xty[i] += vector[i] * label
            for j in range(columns):
                xtx[i][j] += vector[i] * vector[j]
    if rows < columns * 10:
        raise ValueError("训练样本不足，无法拟合")
    for i in range(1, columns):
        xtx[i][i] += ridge
    coefficients = _solve(xtx, xty)
    return {
        "features": list(features),
        "intercept": coefficients[0],
        "coefficients": {key: coefficients[i + 1] for i, key in enumerate(features)},
        "n_train": rows,
        "ridge": ridge,
    }


def model_scores(
    records: Sequence[dict[str, Any]],
    model: dict[str, Any],
) -> dict[tuple[str, str], float]:
    """给每条记录打分：score = b0 + Σ b_i × 特征分位。"""
    features = model["features"]
    coefficients = model["coefficients"]
    scores: dict[tuple[str, str], float] = {}
    for record in records:
        values = [record.get(f"rank_{key}") for key in features]
        if any(value is None for value in values):
            continue
        total = model["intercept"]
        for key, value in zip(features, values):
            total += coefficients[key] * float(value)
        scores[(record["date"], record["symbol"])] = total
    return scores


def quintile_returns(
    records: Sequence[dict[str, Any]],
    scores: dict[tuple[str, str], float],
    *,
    groups: int = 5,
) -> list[dict[str, Any]]:
    """验证集里按模型分数分五组，看未来收益是否单调。"""
    by_date: dict[str, list[tuple[float, float]]] = {}
    for record in records:
        if record.get("fwd") is None:
            continue
        key = (record["date"], record["symbol"])
        if key not in scores:
            continue
        by_date.setdefault(record["date"], []).append((scores[key], record["fwd"]))

    buckets: list[list[float]] = [[] for _ in range(groups)]
    for pairs in by_date.values():
        if len(pairs) < groups * 4:
            continue
        pairs.sort(key=lambda item: item[0])
        for index, (_score, forward) in enumerate(pairs):
            bucket = min(groups - 1, index * groups // len(pairs))
            buckets[bucket].append(forward)
    return [
        {
            "group": f"Q{index + 1}",
            "n": len(values),
            "mean": round(statistics.fmean(values), 2) if values else None,
            "win_rate": round(sum(1 for v in values if v > 0) / len(values) * 100, 1) if values else None,
        }
        for index, values in enumerate(buckets)
    ]


# --------------------------------------------------------------------------- #
# 回测：成本 + 换手上限 + 等权基准
# --------------------------------------------------------------------------- #


def price_maps(histories: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        history["symbol"]: {row["date"]: row["close"] for row in history["rows"] if row["close"]}
        for history in histories
    }


def simulate(
    calendar: Sequence[str],
    prices: dict[str, dict[str, float]],
    pick: Callable[[str, list[str]], list[str]],
    *,
    rebalance_every: int = DEFAULT_REBALANCE,
    cost_buy: float = BUY_COST,
    cost_sell: float = SELL_COST,
    max_turnover: float = DEFAULT_MAX_TURNOVER,
) -> dict[str, Any]:
    """按市值记账回测。

    ``pick(date, available)`` 返回当日目标持仓；调仓按收盘价成交，卖出先于买入，
    单边换手不超过 ``max_turnover``（占当日组合净值），超出部分按比例缩减。
    未成交的旧仓位继续持有，不会为了"凑满目标"再交易一次。
    """
    cash = 1.0
    shares: dict[str, float] = {}
    last_price: dict[str, float] = {}
    nav_series: list[dict[str, Any]] = []
    rebalances: list[dict[str, Any]] = []
    total_cost = 0.0
    total_turnover = 0.0
    cash_ratios: list[float] = []
    last_rebalance = -10 ** 9

    for index, date in enumerate(calendar):
        for symbol, series in prices.items():
            if date in series:
                last_price[symbol] = series[date]
        available = [symbol for symbol in prices if symbol in last_price]

        def value() -> float:
            return cash + sum(shares.get(symbol, 0.0) * last_price[symbol] for symbol in shares)

        if index - last_rebalance >= rebalance_every and available:
            nav_before = value()
            target = pick(date, available)
            if not target:
                target = []
            weight = 1.0 / len(target) if target else 0.0
            desired = {symbol: weight * nav_before for symbol in target}
            # 排序后遍历：set 的迭代顺序受哈希随机化影响，会让"现金不足时谁先成交"
            # 每次运行都不一样，回测结果不可复现。
            order = sorted(set(desired) | set(shares))
            delta = {
                symbol: desired.get(symbol, 0.0) - shares.get(symbol, 0.0) * last_price[symbol]
                for symbol in order
            }
            buy = sum(max(0.0, move) for move in delta.values())
            sell = sum(max(-0.0, -move) for move in delta.values())
            cap = max_turnover * nav_before
            scale = 1.0
            if max(buy, sell) > cap and max(buy, sell) > 0:
                scale = cap / max(buy, sell)

            # 先卖后买：否则先执行的买单会因现金未到账被跳过，
            # 高换手组合会被动留出大量现金，净值被系统性压低。
            sold = bought = 0.0
            for symbol, move in delta.items():
                if move >= 0:
                    continue
                price = last_price[symbol]
                quantity = min(shares.get(symbol, 0.0), -scale * move / price)
                if quantity <= 0:
                    continue
                notional = quantity * price
                shares[symbol] = shares.get(symbol, 0.0) - quantity
                cash += notional * (1 - cost_sell)
                total_cost += notional * cost_sell
                sold += notional
            for symbol, move in delta.items():
                if move <= 0:
                    continue
                price = last_price[symbol]
                budget = min(scale * move, cash / (1 + cost_buy))
                if budget <= 1e-12:
                    continue
                quantity = budget / price
                shares[symbol] = shares.get(symbol, 0.0) + quantity
                cash -= budget * (1 + cost_buy)
                total_cost += budget * cost_buy
                bought += budget
            one_way = 0.5 * (sold + bought) / nav_before if nav_before else 0.0
            total_turnover += one_way
            rebalances.append(
                {
                    "date": date,
                    "one_way_turnover": round(one_way, 4),
                    "capped": scale < 1.0,
                    "cost_bps": round(
                        (sold * cost_sell + bought * cost_buy) / nav_before * 10000, 1
                    )
                    if nav_before
                    else 0.0,
                    "nav": round(value(), 4),
                }
            )
            last_rebalance = index

        nav_series.append({"date": date, "nav": round(value(), 4)})
        total = value()
        if total:
            cash_ratios.append(cash / total)

    return {
        "nav": nav_series,
        "rebalances": rebalances,
        "total_cost": total_cost,
        "total_turnover": total_turnover,
        "cash_ratio_mean": statistics.fmean(cash_ratios) if cash_ratios else 0.0,
        "final_nav": nav_series[-1]["nav"] if nav_series else 1.0,
    }


def metrics(result: dict[str, Any], calendar_len: int) -> dict[str, Any]:
    nav = [point["nav"] for point in result["nav"]]
    if len(nav) < 2:
        return {}
    years = max(calendar_len / 252, 1e-9)
    total = nav[-1] / nav[0] - 1
    annual = (nav[-1] / nav[0]) ** (1 / years) - 1
    returns = [nav[i] / nav[i - 1] - 1 for i in range(1, len(nav)) if nav[i - 1]]
    volatility = statistics.stdev(returns) * math.sqrt(252) if len(returns) > 1 else 0.0
    peak = 0.0
    drawdown = 0.0
    for value in nav:
        peak = max(peak, value)
        if peak:
            drawdown = min(drawdown, value / peak - 1)
    average_nav = statistics.fmean(nav)
    return {
        "total_return": round(total * 100, 2),
        "annual_return": round(annual * 100, 2),
        "max_drawdown": round(drawdown * 100, 2),
        "volatility": round(volatility * 100, 2),
        "sharpe": round(annual / volatility, 2) if volatility else None,
        "annual_turnover": round(result["total_turnover"] / years * 100, 1),
        "cost_drag_bps": round(
            result["total_cost"] / average_nav / years * 10000, 1
        ),
        "rebalances": len(result["rebalances"]),
        "capped_ratio": round(
            sum(1 for item in result["rebalances"] if item["capped"])
            / max(1, len(result["rebalances"]))
            * 100,
            1,
        ),
        "cash_ratio": round(result.get("cash_ratio_mean", 0.0) * 100, 1),
        "final_nav": round(nav[-1], 4),
    }


class RuleScorer:
    """把 analyzer 的规则评分包装成"按日取分"，只复用事实来源，不复制规则。"""

    def __init__(
        self,
        histories: Sequence[dict[str, Any]],
        benchmark: dict[str, Any] | None,
        weights: dict[str, tuple[int, ...]] | None = None,
    ) -> None:
        self.histories = {history["symbol"]: history for history in histories}
        self.index = {
            history["symbol"]: {row["date"]: i for i, row in enumerate(history["rows"])}
            for history in histories
        }
        self.benchmark = benchmark
        self.weights = weights or DEFAULT_WEIGHTS
        self._cache: dict[tuple[str, str], int] = {}

    def score(self, date: str, symbols: Sequence[str]) -> dict[str, float]:
        result: dict[str, float] = {}
        for symbol in symbols:
            index = self.index.get(symbol, {}).get(date)
            if index is None:
                continue
            key = (symbol, date)
            if key not in self._cache:
                history = self.histories[symbol]
                prefix = {**history, "rows": history["rows"][: index + 1]}
                self._cache[key] = score_factors(prefix, self.weights, self.benchmark)[0]
            result[symbol] = float(self._cache[key])
        return result


def top_n(scores: dict[str, float], fraction: float) -> list[str]:
    if not scores:
        return []
    count = max(1, round(len(scores) * fraction))
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [symbol for symbol, _score in ordered[:count]]


def period_breakdown(
    strategy: dict[str, Any],
    benchmark: dict[str, Any],
    *,
    step: int,
) -> list[dict[str, Any]]:
    """把净值曲线切成调仓窗口，逐窗口对比策略与等权基准。

    这是"是否跑赢"的最小证据单位：总收益会被一两段极端行情主导，
    逐窗口看才知道亏损来自信号本身还是某一次押注。
    """
    strategy_nav = {point["date"]: point["nav"] for point in strategy["nav"]}
    benchmark_nav = {point["date"]: point["nav"] for point in benchmark["nav"]}
    dates = [point["date"] for point in strategy["nav"]]
    marks = dates[::step] + [dates[-1]]
    rows: list[dict[str, Any]] = []
    for start, end in zip(marks, marks[1:]):
        if start == end:
            continue
        stock = (strategy_nav[end] / strategy_nav[start] - 1) * 100
        bench = (benchmark_nav[end] / benchmark_nav[start] - 1) * 100
        rows.append(
            {
                "start": start,
                "end": end,
                "strategy": round(stock, 2),
                "benchmark": round(bench, 2),
                "excess": round(stock - bench, 2),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def run(
    symbols: Iterable[str] | None = None,
    *,
    days: int = 600,
    horizon: int = HORIZON,
    split_ratio: float = SPLIT_RATIO,
    rebalance: int = DEFAULT_REBALANCE,
    top_fraction: float = DEFAULT_TOP_FRAC,
    max_turnover: float = DEFAULT_MAX_TURNOVER,
    refresh: bool = False,
    offline: bool = False,
    universe: str = "hs300",
    max_symbols: int = 60,
    variants: Sequence[int] = (5, 10, 20),
) -> dict[str, Any]:
    if symbols is None:
        symbols = load_universe(universe, max_symbols, refresh=refresh, offline=offline)
    histories, failures = load_histories(symbols, days, refresh=refresh, offline=offline)
    if len(histories) < MIN_XSEC:
        raise MarketDataError(f"可用个股只有 {len(histories)} 只，横截面至少需要 {MIN_XSEC} 只")
    benchmark = load_benchmark(days, refresh=refresh, offline=offline)
    if benchmark is None:
        raise MarketDataError("横截面模型需要基准行情计算相对强弱，请先联网运行一次")

    panel = build_panel(histories, benchmark, horizon=horizon)
    split = time_split(panel["dates"], split_ratio=split_ratio, horizon=horizon)
    if not split["valid_dates"]:
        raise MarketDataError("验证集为空，请增加 --days 或降低 --split")

    records = panel["records"]
    train_records = [r for r in records if r["date"] in set(split["train_dates"])]
    valid_records = [r for r in records if r["date"] in set(split["valid_dates"])]

    screening = screen_features(train_records, horizon=horizon)
    model = fit_rank_model(train_records, screening["features"])
    scores = model_scores(records, model)

    # 训练 / 验证两段 IC：验证集只用来报告，不参与任何选择
    ic_train = {key: _daily_ic(train_records, key, horizon=horizon) for key in model["features"]}
    ic_valid = {key: _daily_ic(valid_records, key, horizon=horizon) for key in model["features"]}

    prices = price_maps(histories)
    benchmark_calendar = [row["date"] for row in benchmark["rows"]]
    valid_start = split["valid_start"]
    calendar = [date for date in benchmark_calendar if date >= valid_start]
    train_calendar = [
        date for date in benchmark_calendar
        if split["train_start"] <= date <= split["train_end"]
    ]

    def model_pick(date: str, available: list[str]) -> list[str]:
        day = {
            symbol: scores[(date, symbol)]
            for symbol in available
            if (date, symbol) in scores
        }
        return top_n(day, top_fraction)

    def equal_pick(date: str, available: list[str]) -> list[str]:
        return list(available)

    rule = RuleScorer(histories, benchmark)

    def rule_pick(date: str, available: list[str]) -> list[str]:
        return top_n(rule.score(date, available), top_fraction)

    strategy = simulate(
        calendar, prices, model_pick,
        rebalance_every=rebalance, max_turnover=max_turnover,
    )
    benchmark_run = simulate(
        calendar, prices, equal_pick,
        rebalance_every=rebalance, cost_buy=0.0, cost_sell=0.0, max_turnover=1.0,
    )
    legacy = simulate(
        calendar, prices, rule_pick,
        rebalance_every=rebalance, max_turnover=max_turnover,
    )
    # 训练期同口径对照：说明模型不是只在验证集里"碰巧"有效
    strategy_train = simulate(
        train_calendar, prices, model_pick,
        rebalance_every=rebalance, max_turnover=max_turnover,
    )
    benchmark_train = simulate(
        train_calendar, prices, equal_pick,
        rebalance_every=rebalance, cost_buy=0.0, cost_sell=0.0, max_turnover=1.0,
    )
    # 成本前对照：用来区分"信号没用"与"信号被摩擦吃掉"
    strategy_gross = simulate(
        calendar, prices, model_pick,
        rebalance_every=rebalance, cost_buy=0.0, cost_sell=0.0, max_turnover=max_turnover,
    )

    cost_sensitivity = []
    for every in variants:
        run_v = simulate(calendar, prices, model_pick, rebalance_every=every, max_turnover=max_turnover)
        bench_v = simulate(
            calendar, prices, equal_pick,
            rebalance_every=every, cost_buy=0.0, cost_sell=0.0, max_turnover=1.0,
        )
        stat = metrics(run_v, len(calendar))
        bench_stat = metrics(bench_v, len(calendar))
        cost_sensitivity.append(
            {
                "rebalance": every,
                "strategy_total": stat["total_return"],
                "benchmark_total": bench_stat["total_return"],
                "excess": round(stat["total_return"] - bench_stat["total_return"], 2),
                "annual_turnover": stat["annual_turnover"],
                "cost_drag_bps": stat["cost_drag_bps"],
                "max_drawdown": stat["max_drawdown"],
            }
        )

    cap_sensitivity = []
    for cap in (1.0, 0.6, max_turnover, 0.2):
        run_c = simulate(calendar, prices, model_pick, rebalance_every=rebalance, max_turnover=cap)
        stat = metrics(run_c, len(calendar))
        cap_sensitivity.append(
            {
                "cap": cap,
                "strategy_total": stat["total_return"],
                "annual_turnover": stat["annual_turnover"],
                "cost_drag_bps": stat["cost_drag_bps"],
                "max_drawdown": stat["max_drawdown"],
                "excess": round(stat["total_return"] - metrics(benchmark_run, len(calendar))["total_return"], 2),
            }
        )
    periods = period_breakdown(strategy, benchmark_run, step=max(rebalance, horizon))

    # 标签周期稳健性：同一套流程换一个 horizon，看结论是否只在某个周期成立
    horizon_variants = []
    for variant in sorted({5, 10, horizon}):
        if variant == horizon:
            horizon_variants.append(
                {
                    "horizon": horizon,
                    "strategy": metrics(strategy, len(calendar))["total_return"],
                    "benchmark": metrics(benchmark_run, len(calendar))["total_return"],
                    "excess": round(
                        metrics(strategy, len(calendar))["total_return"]
                        - metrics(benchmark_run, len(calendar))["total_return"],
                        2,
                    ),
                    "annual_turnover": metrics(strategy, len(calendar))["annual_turnover"],
                }
            )
            continue
        panel_v = build_panel(histories, benchmark, horizon=variant)
        split_v = time_split(panel_v["dates"], split_ratio=split_ratio, horizon=variant)
        if not split_v["valid_dates"]:
            continue
        train_v = [r for r in panel_v["records"] if r["date"] in set(split_v["train_dates"])]
        screening_v = screen_features(train_v, horizon=variant)
        model_v = fit_rank_model(train_v, screening_v["features"])
        scores_v = model_scores(panel_v["records"], model_v)
        calendar_v = [date for date in benchmark_calendar if date >= split_v["valid_start"]]

        def pick_v(date: str, available: list[str]) -> list[str]:
            day = {s: scores_v[(date, s)] for s in available if (date, s) in scores_v}
            return top_n(day, top_fraction)

        stat_v = metrics(
            simulate(calendar_v, prices, pick_v, rebalance_every=rebalance, max_turnover=max_turnover),
            len(calendar_v),
        )
        bench_v = metrics(
            simulate(
                calendar_v, prices, equal_pick,
                rebalance_every=rebalance, cost_buy=0.0, cost_sell=0.0, max_turnover=1.0,
            ),
            len(calendar_v),
        )
        horizon_variants.append(
            {
                "horizon": variant,
                "strategy": stat_v["total_return"],
                "benchmark": bench_v["total_return"],
                "excess": round(stat_v["total_return"] - bench_v["total_return"], 2),
                "annual_turnover": stat_v["annual_turnover"],
            }
        )

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "params": {
            "universe": universe,
            "symbols": len(histories),
            "days": days,
            "horizon": horizon,
            "warmup": WARMUP,
            "min_xsec": MIN_XSEC,
            "split_ratio": split_ratio,
            "embargo": EMBARGO,
            "rebalance": rebalance,
            "top_fraction": top_fraction,
            "max_turnover": max_turnover,
            "cost_buy_bps": round(BUY_COST * 10000, 2),
            "cost_sell_bps": round(SELL_COST * 10000, 2),
            "commission_bps": COMMISSION_BPS,
            "stamp_duty_bps": STAMP_DUTY_BPS,
            "slippage_bps": SLIPPAGE_BPS,
            "benchmark": {"symbol": benchmark["symbol"], "name": benchmark["name"]},
        },
        "split": split,
        "coverage": {
            "records_train": len(train_records),
            "records_valid": len(valid_records),
            "dates_valid": len(split["valid_dates"]),
            "failures": failures,
        },
        "screening": screening,
        "model": model,
        "ic": {"train": ic_train, "valid": ic_valid},
        "quintiles_valid": quintile_returns(valid_records, scores),
        "nav": {
            "calendar": calendar,
            "strategy": strategy["nav"],
            "benchmark": benchmark_run["nav"],
            "legacy": legacy["nav"],
            "strategy_gross": strategy_gross["nav"],
        },
        "stats": {
            "valid_days": len(calendar),
            "strategy": metrics(strategy, len(calendar)),
            "benchmark": metrics(benchmark_run, len(calendar)),
            "legacy": metrics(legacy, len(calendar)),
            "strategy_gross": metrics(strategy_gross, len(calendar)),
            "train_strategy": metrics(strategy_train, len(train_calendar)),
            "train_benchmark": metrics(benchmark_train, len(train_calendar)),
        },
        "cost_sensitivity": cost_sensitivity,
        "cap_sensitivity": cap_sensitivity,
        "horizon_variants": horizon_variants,
        "periods": periods,
        "excess_win_rate": (
            round(
                sum(1 for row in periods if row["excess"] > 0) / len(periods) * 100, 1
            )
            if periods
            else None
        ),
        "rebalances": strategy["rebalances"][-8:],
    }
    return report


# --------------------------------------------------------------------------- #
# 呈现
# --------------------------------------------------------------------------- #


def _fmt(value: float | None, suffix: str = "%", digits: int = 2, signed: bool = True) -> str:
    if value is None:
        return ELLIPSIS
    template = f"{{:+.{digits}f}}" if signed else f"{{:.{digits}f}}"
    return template.format(value) + suffix


def _svg_nav(report: dict[str, Any], *, width: int = 900, height: int = 340) -> str:
    """三线净值图（纯 SVG，无第三方依赖）。"""
    series = [
        ("新模型（含成本）", report["nav"]["strategy"], "#b34938", ""),
        ("新模型（成本前）", report["nav"]["strategy_gross"], "#8a5cc4", ' stroke-dasharray="5 3"'),
        ("等权基准", report["nav"]["benchmark"], "#176b4d", ""),
        ("旧规则 v3（同成本）", report["nav"]["legacy"], "#aa7627", ""),
    ]
    left, right, top, bottom = 54, 16, 34, 34
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [point["nav"] for _, points, _, _ in series for point in points]
    if not values:
        return ""
    low, high = min(values), max(values)
    span = max(high - low, 1e-6)
    low -= span * 0.06
    high += span * 0.06
    total = len(series[0][1])

    def x_at(index: int) -> float:
        return left + plot_w * (index / max(total - 1, 1))

    def y_at(value: float) -> float:
        return top + plot_h * (1 - (value - low) / (high - low))

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="策略与等权基准净值曲线">']
    parts.append(
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fffdf7" rx="8"/>'
    )
    for step in range(5):
        value = low + (high - low) * step / 4
        y = y_at(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            f'stroke="#e2e5de" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'fill="#6e7b74" font-size="11" font-family="ui-monospace,Menlo,monospace">'
            f'{value:.3f}</text>'
        )
    dates = [point["date"] for point in series[0][1]]
    ticks = 5
    for step in range(ticks):
        index = round((len(dates) - 1) * step / (ticks - 1))
        anchor = "middle"
        x = x_at(index)
        if index == 0:
            anchor, x = "start", left
        elif index == len(dates) - 1:
            anchor, x = "end", width - right
        parts.append(
            f'<text x="{x:.1f}" y="{height - 12}" text-anchor="{anchor}" '
            f'fill="#6e7b74" font-size="11" font-family="ui-monospace,Menlo,monospace">'
            f'{dates[index]}</text>'
        )
    for name, points, color, dash in series:
        path = " ".join(
            f'{"M" if i == 0 else "L"}{x_at(i):.1f},{y_at(point["nav"]):.1f}'
            for i, point in enumerate(points)
        )
        parts.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.9"{dash}/>'
        )
        parts.append(
            f'<circle cx="{x_at(len(points) - 1):.1f}" cy="{y_at(points[-1]["nav"]):.1f}" '
            f'r="3" fill="{color}"/>'
        )
    for order, (name, _points, color, _dash) in enumerate(series):
        x = left + 4 + order * 158
        parts.append(f'<rect x="{x}" y="{top - 24}" width="10" height="3" fill="{color}"/>')
        parts.append(
            f'<text x="{x + 15}" y="{top - 19}" fill="#16221d" font-size="11.5" '
            f'font-family="-apple-system,Segoe UI,PingFang SC,sans-serif">{name}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _row_cells(report: dict[str, Any], key: str) -> str:
    stat = report["stats"][key]
    bench = report["stats"]["benchmark"]["total_return"]
    excess = round(stat["total_return"] - bench, 2)
    tone = "wf-up" if excess > 0 else "wf-down" if excess < 0 else ""
    return (
        f'<td class="num">{_fmt(stat["total_return"])}</td>'
        f'<td class="num">{_fmt(stat["annual_return"])}</td>'
        f'<td class="num">{_fmt(stat["max_drawdown"])}</td>'
        f'<td class="num">{_fmt(stat["volatility"], "%", 2, signed=False)}</td>'
        f'<td class="num">{stat["sharpe"] if stat["sharpe"] is not None else ELLIPSIS}</td>'
        f'<td class="num">{_fmt(stat["annual_turnover"], "%", 0, signed=False)}</td>'
        f'<td class="num">{stat["cost_drag_bps"]:.0f}</td>'
        f'<td class="num {tone}">{_fmt(excess)}</td>'
    )


def render_html(report: dict[str, Any]) -> str:
    params = report["params"]
    stats = report["stats"]
    split = report["split"]
    strategy = stats["strategy"]
    bench = stats["benchmark"]
    legacy = stats["legacy"]
    excess = round(strategy["total_return"] - bench["total_return"], 2)
    annual_excess = round(strategy["annual_return"] - bench["annual_return"], 2)
    verdict_tone = "wf-up" if excess > 0 else "wf-down"
    verdict = "跑赢" if excess > 0 else "跑输"

    screening_rows = []
    for family in report["screening"]["families"]:
        for row in family["candidates"]:
            mark = "●" if row["selected"] else ""
            screening_rows.append(
                "<tr>"
                f'<td>{family["family"]}</td>'
                f'<td>{row["label"]}<span class="wf-code">{row["key"]}</span></td>'
                f'<td class="num">{_fmt(row["ic"], "", 4)}</td>'
                f'<td class="num">{row["t"] if row["t"] is not None else ELLIPSIS}</td>'
                f'<td class="num">{_fmt(row["positive_rate"], "%", 1, signed=False)}</td>'
                f'<td class="wf-verdict">{mark} {row["note"]}</td>'
                "</tr>"
            )

    coefficient_rows = []
    for key, value in report["model"]["coefficients"].items():
        coefficient_rows.append(
            "<tr>"
            f'<td>{FEATURE_LABELS[key]}<span class="wf-code">{key}</span></td>'
            f'<td class="num">{value:+.4f}</td>'
            f'<td class="num">{_fmt(report["ic"]["train"][key]["ic"], "", 4)}</td>'
            f'<td class="num">{_fmt(report["ic"]["valid"][key]["ic"], "", 4)}</td>'
            "</tr>"
        )

    quintile_rows = []
    values = [row["mean"] for row in report["quintiles_valid"] if row["mean"] is not None]
    for row in report["quintiles_valid"]:
        quintile_rows.append(
            "<tr>"
            f'<td>{row["group"]}</td>'
            f'<td class="num">{row["n"]}</td>'
            f'<td class="num">{_fmt(row["mean"])}</td>'
            f'<td class="num">{_fmt(row["win_rate"], "%", 1, signed=False)}</td>'
            "</tr>"
        )
    spread = round(values[-1] - values[0], 2) if len(values) >= 2 else None

    sensitivity_rows = []
    for row in report["cost_sensitivity"]:
        sensitivity_rows.append(
            "<tr>"
            f'<td class="num">{row["rebalance"]} 日</td>'
            f'<td class="num">{_fmt(row["strategy_total"])}</td>'
            f'<td class="num">{_fmt(row["benchmark_total"])}</td>'
            f'<td class="num {("wf-up" if row["excess"] > 0 else "wf-down")}">{_fmt(row["excess"])}</td>'
            f'<td class="num">{_fmt(row["annual_turnover"], "%", 0, signed=False)}</td>'
            f'<td class="num">{row["cost_drag_bps"]:.0f}</td>'
            "</tr>"
        )

    period_rows = []
    for row in report["periods"]:
        tone = "wf-up" if row["excess"] > 0 else "wf-down"
        period_rows.append(
            "<tr>"
            f'<td>{row["start"]} → {row["end"]}</td>'
            f'<td class="num {("wf-up" if row["strategy"] > 0 else "wf-down")}">{_fmt(row["strategy"])}</td>'
            f'<td class="num {("wf-up" if row["benchmark"] > 0 else "wf-down")}">{_fmt(row["benchmark"])}</td>'
            f'<td class="num {tone}">{_fmt(row["excess"])}</td>'
            "</tr>"
        )

    cap_rows = []
    for row in report["cap_sensitivity"]:
        cap_rows.append(
            "<tr>"
            f'<td class="num">{int(row["cap"] * 100)}%</td>'
            f'<td class="num">{_fmt(row["strategy_total"])}</td>'
            f'<td class="num {("wf-up" if row["excess"] > 0 else "wf-down")}">{_fmt(row["excess"])}</td>'
            f'<td class="num">{_fmt(row["annual_turnover"], "%", 0, signed=False)}</td>'
            f'<td class="num">{row["cost_drag_bps"]:.0f}</td>'
            "</tr>"
        )

    horizon_rows = []
    for row in report["horizon_variants"]:
        horizon_rows.append(
            "<tr>"
            f'<td class="num">{row["horizon"]} 日</td>'
            f'<td class="num">{_fmt(row["strategy"])}</td>'
            f'<td class="num">{_fmt(row["benchmark"])}</td>'
            f'<td class="num {("wf-up" if row["excess"] > 0 else "wf-down")}">{_fmt(row["excess"])}</td>'
            f'<td class="num">{_fmt(row["annual_turnover"], "%", 0, signed=False)}</td>'
            "</tr>"
        )

    train_lead = round(
        stats["train_strategy"]["total_return"] - stats["train_benchmark"]["total_return"], 2
    )
    if train_lead > 0:
        train_note = (
            f"训练集小幅领先（{_fmt(stats['train_strategy']['total_return'])} vs "
            f"{_fmt(stats['train_benchmark']['total_return'])}，{_fmt(train_lead)}），"
            "验证集优势反转，属于样本外失效而非参数没调好"
        )
    else:
        train_note = (
            f"训练集已经跑输（{_fmt(stats['train_strategy']['total_return'])} vs "
            f"{_fmt(stats['train_benchmark']['total_return'])}），"
            "样本内外方向一致，说明因子本身缺少可迁移的 edge"
        )

    return f"""<section class="wf" aria-labelledby="xsecTitle">
  <style>
    .wf {{ margin: 0; padding: 0; color: var(--ink, #16221d); }}
    .wf h3 {{ margin: 0 0 6px; font-size: 17px; }}
    .wf .wf-sub {{ margin: 0 0 14px; color: var(--muted, #6e7b74); font-size: 12.5px; line-height: 1.7; }}
    .wf .wf-scroll {{ overflow-x: auto; border: 1px solid var(--line, #d8ddd6); border-radius: 8px; background: var(--panel, rgba(255,255,252,.84)); }}
    .wf table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; white-space: nowrap; }}
    .wf thead th {{ padding: 8px 10px; color: var(--muted, #6e7b74); font-weight: 600; font-size: 11px; border-bottom: 1px solid var(--line, #d8ddd6); background: rgba(23,107,77,.04); }}
    .wf tbody td {{ padding: 8px 10px; border-bottom: 1px solid rgba(216,221,214,.7); }}
    .wf tbody tr:last-child td {{ border-bottom: 0; }}
    .wf .num {{ text-align: right; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-variant-numeric: tabular-nums; }}
    .wf .wf-up {{ color: var(--red, #b34938); }}
    .wf .wf-down {{ color: var(--green, #176b4d); }}
    .wf .wf-code {{ display: block; margin-top: 1px; color: var(--muted, #6e7b74); font: 700 10px/1.4 ui-monospace, Menlo, monospace; }}
    .wf .wf-chart {{ border: 1px solid var(--line, #d8ddd6); border-radius: 8px; background: #fffdf7; padding: 6px 4px 0; }}
    .wf .wf-verdict {{ font-size: 12px; line-height: 1.6; white-space: normal; }}
    .wf .wf-warn {{ margin: 14px 0 0; padding: 9px 12px; border-left: 3px solid var(--amber, #aa7627); border-radius: 0 6px 6px 0; background: rgba(170,118,39,.09); color: #7a5418; font-size: 12px; line-height: 1.75; }}
    .wf .wf-notes {{ margin: 14px 0 0; padding-left: 18px; color: var(--muted, #6e7b74); font-size: 12px; line-height: 1.85; }}
    .wf .wf-factor {{ margin-top: 20px; }}
  </style>
  <h3 id="xsecTitle">横截面分位模型与净值对比</h3>
  <p class="wf-sub">标签与特征都改成当日横截面分位，按时间切分训练 / 验证；
    验证集 {split['valid_start']} → {split['valid_end']}（{stats['valid_days']} 个交易日）·
    {params['symbols']} 只个股 · 每 {params['rebalance']} 日调仓 · 持有 Top {int(params['top_fraction'] * 100)}% ·
    单次换手上限 {int(params['max_turnover'] * 100)}%</p>

  <div class="wf-chart">{_svg_nav(report)}</div>

  <div class="wf-factor">
    <h3>净值对比（验证集，扣成本）</h3>
    <p class="wf-sub">同区间、同股票池、同一调仓节奏。基准为股票池等权持有且不计成本，
      策略与旧规则使用相同的成本与换手上限。</p>
    <div class="wf-scroll">
      <table>
        <thead>
          <tr><th scope="col">组合</th><th scope="col">总收益</th><th scope="col">年化</th>
          <th scope="col">最大回撤</th><th scope="col">年化波动</th><th scope="col">夏普</th>
          <th scope="col">年换手</th><th scope="col">成本(bps/年)</th><th scope="col">相对基准</th></tr>
        </thead>
        <tbody>
          <tr><td><b>新模型</b><span class="wf-code">3 因子 · 分位标签</span></td>{_row_cells(report, "strategy")}</tr>
          <tr><td>新模型（成本前）<span class="wf-code">诊断用，不扣成本</span></td>{_row_cells(report, "strategy_gross")}</tr>
          <tr><td>等权基准<span class="wf-code">不计成本</span></td>{_row_cells(report, "benchmark")}</tr>
          <tr><td>旧规则 v3<span class="wf-code">同成本同约束</span></td>{_row_cells(report, "legacy")}</tr>
        </tbody>
      </table>
    </div>
    <p class="wf-warn">结论：验证集新模型{verdict}等权基准，
      总收益 {_fmt(strategy['total_return'])} vs {_fmt(bench['total_return'])}（差 {_fmt(excess)}），
      年化差 {_fmt(annual_excess)}；旧规则 v3 同期 {_fmt(legacy['total_return'])}。
      最大回撤 {_fmt(strategy['max_drawdown'])} vs {_fmt(bench['max_drawdown'])}。
      {train_note}；两套评分同时跑输，说明问题在因子本身。
      在净值曲线确认之前，不建议把这套模型接到前端评分。</p>  </div>

  <div class="wf-factor">
    <h3>逐窗口超额（每 {max(params['rebalance'], params['horizon'])} 个交易日）</h3>
    <p class="wf-sub">总收益容易被一两段极端行情主导，逐窗口对比才知道亏损来自信号还是某一次押注。
      超额为正的窗口占 {report['excess_win_rate']}%。</p>
    <div class="wf-scroll">
      <table>
        <thead><tr><th scope="col">区间</th><th scope="col">策略</th>
        <th scope="col">等权</th><th scope="col">超额</th></tr></thead>
        <tbody>{''.join(period_rows)}</tbody>
      </table>
    </div>
  </div>
  <div class="wf-factor">
    <h3>特征筛选（仅用训练集）</h3>
    <p class="wf-sub">上一轮 60 只样本的单因子归因里，只有 MACD、量比、RS 三族的超额 edge 为正。
      族已定，族内的连续化表达由训练集横截面 IC 决定，验证集不参与选择。</p>
    <div class="wf-scroll">
      <table>
        <thead><tr><th scope="col">因子族</th><th scope="col">候选表达</th>
        <th scope="col">训练 IC</th><th scope="col">t</th><th scope="col">正 IC 日</th><th scope="col">选择</th></tr></thead>
        <tbody>{''.join(screening_rows)}</tbody>
      </table>
    </div>
  </div>

  <div class="wf-factor">
    <h3>次级诊断：模型系数与分段 IC</h3>
    <p class="wf-sub">目标为标签分位（0–1），系数表示"特征分位每提高一个横截面档位，标签分位平均提高多少"。
      验证集 IC 仅作诊断，不用于调参。</p>
    <div class="wf-scroll">
      <table>
        <thead><tr><th scope="col">特征</th><th scope="col">系数</th>
        <th scope="col">训练 IC</th><th scope="col">验证 IC</th></tr></thead>
        <tbody>{''.join(coefficient_rows)}</tbody>
      </table>
    </div>
    <ul class="wf-notes">
      <li>训练集 {stats['train_strategy']['total_return']:+.2f}% vs 等权 {stats['train_benchmark']['total_return']:+.2f}%</li>
      <li>验证集五分组平均收益首尾差 {_fmt(spread)}，Q1→Q5 {_fmt(values[0] if values else None)} → {_fmt(values[-1] if values else None)}</li>
    </ul>
  </div>

  <div class="wf-factor">
    <h3>调仓频率与成本敏感性</h3>
    <p class="wf-sub">同样的信号、同样的换手上限，只改调仓间隔。换手越高，成本吃掉的正 edge 越多。</p>
    <div class="wf-scroll">
      <table>
        <thead><tr><th scope="col">调仓间隔</th><th scope="col">策略总收益</th>
        <th scope="col">等权总收益</th><th scope="col">超额</th><th scope="col">年换手</th>
        <th scope="col">成本(bps/年)</th></tr></thead>
        <tbody>{''.join(sensitivity_rows)}</tbody>
      </table>
    </div>
    <div class="wf-scroll" style="margin-top:10px">
      <table>
        <thead><tr><th scope="col">单次换手上限</th><th scope="col">策略总收益</th>
        <th scope="col">超额</th><th scope="col">年换手</th><th scope="col">成本(bps/年)</th></tr></thead>
        <tbody>{''.join(cap_rows)}</tbody>
      </table>
    </div>
  </div>

  <div class="wf-factor">
    <h3>标签周期稳健性</h3>
    <p class="wf-sub">同一流程只改标签与目标周期（其余参数不变），验证集结论是否只在某个周期成立。</p>
    <div class="wf-scroll">
      <table>
        <thead><tr><th scope="col">标签周期</th><th scope="col">策略总收益</th>
        <th scope="col">等权总收益</th><th scope="col">超额</th><th scope="col">年换手</th></tr></thead>
        <tbody>{''.join(horizon_rows)}</tbody>
      </table>
    </div>
  </div>

  <div class="wf-factor">
    <h3>次级诊断：五分组（验证集）</h3>
    <div class="wf-scroll">
      <table>
        <thead><tr><th scope="col">分组</th><th scope="col">样本</th>
        <th scope="col">未来 {params['horizon']} 日均值</th><th scope="col">胜率</th></tr></thead>
        <tbody>{''.join(quintile_rows)}</tbody>
      </table>
    </div>
  </div>

  <p class="wf-note">切分：训练 {split['train_start']} → {split['train_end']}，
    purge {split['purged']} 个交易日（标签跨度 {params['horizon']} 日 + embargo {params['embargo']} 日），
    验证 {split['valid_start']} 起。成本：佣金 {params['commission_bps']}bp、印花税 {params['stamp_duty_bps']}bp（仅卖出）、
    滑点 {params['slippage_bps']}bp，单边买入 {params['cost_buy_bps']}bp、卖出 {params['cost_sell_bps']}bp。
    信号以当日收盘价成交，未计次日开盘跳空；基准不计成本。
    局限：股票池取当前沪深300 成分股，存在存续偏差（已退市标的缺席），
    验证集只有 {stats['valid_days']} 个交易日 / {len(report['periods'])} 个调仓窗口，
    单次结论的统计功效有限，因此同时看了成本前后、换手上限与标签周期的敏感性。
    生成于 {report['generated_at']}。</p>
</section>
"""


def render_preview(report: dict[str, Any]) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>横截面模型净值对比 · 预览</title>
<style>
  :root {{
    --ink: #16221d; --muted: #6e7b74; --paper: #f4f2eb; --panel: rgba(255,255,252,.84);
    --line: #d8ddd6; --green: #176b4d; --red: #b34938; --amber: #aa7627;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--paper); color: var(--ink);
    font: 15px/1.65 -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; }}
  .wrap {{ max-width: 1040px; margin: 0 auto; padding: 36px 24px 56px; }}
  .mock-panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
    padding: 22px 22px 18px; box-shadow: 0 18px 50px rgba(31,48,40,.08); }}
  .mock-label {{ color: var(--green); font: 700 11px/1.4 ui-monospace, Menlo, monospace;
    letter-spacing: .18em; margin-bottom: 4px; }}
  .mock-title {{ margin: 0 0 18px; font-size: 20px; font-weight: 650; }}
</style>
</head>
<body>
  <div class="wrap">
    <div class="mock-panel">
      <div class="mock-label">CROSS-SECTIONAL MODEL</div>
      <h2 class="mock-title">横截面分位模型 · 净值对比</h2>
      {render_html(report)}
    </div>
  </div>
</body>
</html>
"""


def render_markdown(report: dict[str, Any]) -> str:
    params = report["params"]
    stats = report["stats"]
    split = report["split"]
    strategy, bench, legacy = stats["strategy"], stats["benchmark"], stats["legacy"]
    excess = round(strategy["total_return"] - bench["total_return"], 2)
    worst = sorted(report["periods"], key=lambda row: row["excess"])[:2]
    lines = [
        "# 横截面分位模型与净值回测",
        "",
        f"生成时间：{report['generated_at']}　股票池：{params['symbols']} 只 A 股个股　"
        f"验证区间：{split['valid_start']} → {split['valid_end']}（{stats['valid_days']} 个交易日）",
        "",
        "## 1. 结论（先看净值）",
        "",
        f"- 验证集新模型总收益 **{_fmt(strategy['total_return'])}**，等权基准 "
        f"**{_fmt(bench['total_return'])}**，差 **{_fmt(excess)}** → {'跑赢' if excess > 0 else '跑输'}。",
        f"- 年化：策略 {_fmt(strategy['annual_return'])} vs 基准 {_fmt(bench['annual_return'])}；"
        f"最大回撤 {_fmt(strategy['max_drawdown'])} vs {_fmt(bench['max_drawdown'])}。",
        f"- 成本前策略 {_fmt(stats['strategy_gross']['total_return'])}，"
        f"说明{'亏损主要来自信号本身，不只是交易摩擦' if stats['strategy_gross']['total_return'] < bench['total_return'] else '扣掉成本才转为跑输'}。",
        f"- 逐窗口（{len(report['periods'])} 段）超额为正的比例 {report['excess_win_rate']}%，"
        f"最差两段：{worst[0]['start']}→{worst[0]['end']}（{_fmt(worst[0]['excess'])}）、"
        f"{worst[1]['start']}→{worst[1]['end']}（{_fmt(worst[1]['excess'])}）。"
        if len(worst) >= 2
        else f"- 逐窗口超额为正的比例 {report['excess_win_rate']}%。",
        f"- 旧规则 v3 同成本同约束：{_fmt(legacy['total_return'])}（年化 {_fmt(legacy['annual_return'])}，"
        f"回撤 {_fmt(legacy['max_drawdown'])}）。",
        f"- 成本：单边买 {params['cost_buy_bps']}bp / 卖 {params['cost_sell_bps']}bp，"
        f"年换手 {_fmt(strategy['annual_turnover'], '%', 0, signed=False)}，"
        f"成本拖累 {strategy['cost_drag_bps']:.0f}bp/年，"
        f"换手上限触发率 {strategy['capped_ratio']:.0f}%。",
        "",
        "## 2. 净值与绩效对照（验证集）",
        "",
        "| 组合 | 总收益 | 年化 | 最大回撤 | 年化波动 | 夏普 | 年换手 | 成本(bps/年) | 相对基准 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, stat in (("新模型", strategy), ("新模型（成本前）", stats["strategy_gross"]), ("等权基准", bench), ("旧规则 v3", legacy)):
        lines.append(
            f"| {label} | {_fmt(stat['total_return'])} | {_fmt(stat['annual_return'])} | "
            f"{_fmt(stat['max_drawdown'])} | {_fmt(stat['volatility'], '%', 2, signed=False)} | "
            f"{stat['sharpe'] if stat['sharpe'] is not None else ELLIPSIS} | "
            f"{_fmt(stat['annual_turnover'], '%', 0, signed=False)} | "
            f"{stat['cost_drag_bps']:.0f} | "
            f"{_fmt(round(stat['total_return'] - bench['total_return'], 2))} |"
        )

    lines += [
        "",
        f"训练期同口径：策略 {_fmt(stats['train_strategy']['total_return'])} vs "
        f"等权 {_fmt(stats['train_benchmark']['total_return'])}。",
        "",
        "## 3. 切分方式",
        "",
        f"- 训练：{split['train_start']} → {split['train_end']}（{len(split['train_dates'])} 个交易日）",
        f"- purge：切分点前剔除 {split['purged']} 个交易日 = 标签跨度 {params['horizon']} 日 + "
        f"embargo {params['embargo']} 日，确保训练样本的标签终点早于验证集第一天。",
        f"- 验证：{split['valid_start']} → {split['valid_end']}（{len(split['valid_dates'])} 个交易日）。"
        f"验证集不参与特征选择、系数估计与参数调整。",
        "",
        "## 4. 标签",
        "",
        f"- 未来 {params['horizon']} 个交易日收益，按当日横截面排名转为分位 `(rank-1)/(n-1)`，"
        "标签是「当天在股票池里排第几」，而不是收益数值。",
        f"- 当日有效标的数 < {params['min_xsec']} 的横截面直接丢弃，避免小样本分位失真。",
        "",
        "## 5. 特征筛选（仅训练集）",
        "",
        "| 因子族 | 候选表达 | 训练 IC | t | 正 IC 日 | 选择 |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for family in report["screening"]["families"]:
        for row in family["candidates"]:
            lines.append(
                f"| {family['family']} | {row['label']} | {_fmt(row['ic'], '', 4)} | "
                f"{row['t'] if row['t'] is not None else ELLIPSIS} | "
                f"{_fmt(row['positive_rate'], '%', 1, signed=False)} | "
                f"{'● ' if row['selected'] else ''}{row['note']} |"
            )

    lines += [
        "",
        "## 6. 系数与分段 IC",
        "",
        "| 特征 | 系数 | 训练 IC | 验证 IC |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, value in report["model"]["coefficients"].items():
        lines.append(
            f"| {FEATURE_LABELS[key]}（{key}） | {value:+.4f} | "
            f"{_fmt(report['ic']['train'][key]['ic'], '', 4)} | "
            f"{_fmt(report['ic']['valid'][key]['ic'], '', 4)} |"
        )

    lines += ["", f"截距 {report['model']['intercept']:.4f}，训练样本 {report['model']['n_train']} 条。", ""]

    lines += ["## 7. 五分组（验证集）", "", "| 分组 | 样本 | 均值 | 胜率 |", "| --- | ---: | ---: | ---: |"]
    for row in report["quintiles_valid"]:
        lines.append(
            f"| {row['group']} | {row['n']} | {_fmt(row['mean'])} | "
            f"{_fmt(row['win_rate'], '%', 1, signed=False)} |"
        )

    lines += [
        "",
        "## 8. 调仓频率敏感性",
        "",
        "| 调仓间隔 | 策略总收益 | 等权总收益 | 超额 | 年换手 | 成本(bps/年) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["cost_sensitivity"]:
        lines.append(
            f"| {row['rebalance']} 日 | {_fmt(row['strategy_total'])} | "
            f"{_fmt(row['benchmark_total'])} | {_fmt(row['excess'])} | "
            f"{_fmt(row['annual_turnover'], '%', 0, signed=False)} | {row['cost_drag_bps']:.0f} |"
        )

    lines += [
        "",
        "## 8b. 单次换手上限敏感性",
        "",
        f"固定 {params['rebalance']} 日调仓，只改单次单边换手上限。",
        "",
        "| 上限 | 策略总收益 | 超额 | 年换手 | 成本(bps/年) | 最大回撤 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["cap_sensitivity"]:
        lines.append(
            f"| {int(row['cap'] * 100)}% | {_fmt(row['strategy_total'])} | "
            f"{_fmt(row['excess'])} | {_fmt(row['annual_turnover'], '%', 0, signed=False)} | "
            f"{row['cost_drag_bps']:.0f} | {_fmt(row['max_drawdown'])} |"
        )

    lines += [
        "",
        "## 8c. 逐窗口超额",
        "",
        "| 区间 | 策略 | 等权 | 超额 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in report["periods"]:
        lines.append(
            f"| {row['start']} → {row['end']} | {_fmt(row['strategy'])} | "
            f"{_fmt(row['benchmark'])} | {_fmt(row['excess'])} |"
        )

    lines += [
        "",
        "## 8d. 标签周期稳健性",
        "",
        "同一流程只改标签与目标周期，其余参数不变。",
        "",
        "| 标签周期 | 策略总收益 | 等权总收益 | 超额 | 年换手 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["horizon_variants"]:
        lines.append(
            f"| {row['horizon']} 日 | {_fmt(row['strategy'])} | {_fmt(row['benchmark'])} | "
            f"{_fmt(row['excess'])} | {_fmt(row['annual_turnover'], '%', 0, signed=False)} |"
        )

    lines += [
        "",
        "## 9. 成本与换手参数",
        "",
        f"- 佣金 {params['commission_bps']}bp（双边）、过户费 {params['transfer_bps'] if 'transfer_bps' in params else 0.1}bp（双边）、"
        f"印花税 {params['stamp_duty_bps']}bp（仅卖出）、滑点 {params['slippage_bps']}bp（双边）。",
        f"- 单边买入 {params['cost_buy_bps']}bp、卖出 {params['cost_sell_bps']}bp，往返约 "
        f"{params['cost_buy_bps'] + params['cost_sell_bps']:.1f}bp。",
        f"- 单次调仓单边换手上限 {int(params['max_turnover'] * 100)}%，超出按比例缩减；"
        "未成交的旧仓位继续持有，不为了凑目标再交易。",
        f"- 持有 Top {int(params['top_fraction'] * 100)}% 等权，每 {params['rebalance']} 个交易日调仓一次。",
        "",
        "> 说明：信号以当日收盘价成交，未计次日开盘跳空与涨跌停无法成交；"
        "基准为同股票池等权、同调仓日再平衡且不计成本。",
        "",
    ]
    return "\n".join(lines)


def write_reports(report: dict[str, Any], outdir: Path) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": outdir / "xsec.json",
        "markdown": outdir / "xsec.md",
        "html": outdir / "xsec_table.html",
        "preview": outdir / "xsec_preview.html",
    }
    paths["json"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["markdown"].write_text(render_markdown(report), encoding="utf-8")
    paths["html"].write_text(render_html(report), encoding="utf-8")
    paths["preview"].write_text(render_preview(report), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="横截面分位建模与净值回测（离线）")
    parser.add_argument("--symbols", default=None, help="逗号分隔代码；省略时按 --universe 取股票池")
    parser.add_argument("--universe", default="hs300", help="股票池，默认沪深300 抽样")
    parser.add_argument("--max-symbols", type=int, default=60, help="股票池抽样数量")
    parser.add_argument("--days", type=int, default=600, help="日线根数（80–600）")
    parser.add_argument("--horizon", type=int, default=HORIZON, help="标签与目标周期（交易日）")
    parser.add_argument("--split", type=float, default=SPLIT_RATIO, help="训练集占比")
    parser.add_argument("--rebalance", type=int, default=DEFAULT_REBALANCE, help="调仓间隔（交易日）")
    parser.add_argument("--top", type=float, default=DEFAULT_TOP_FRAC, help="持仓比例，默认 0.2")
    parser.add_argument("--turnover", type=float, default=DEFAULT_MAX_TURNOVER, help="单次单边换手上限")
    parser.add_argument("--refresh", action="store_true", help="忽略缓存重新拉取")
    parser.add_argument("--offline", action="store_true", help="只用本地缓存")
    parser.add_argument("--outdir", default=str(REPORT_DIR / "xsec"), help="报告目录")
    args = parser.parse_args(argv)

    symbols = None
    universe = args.universe
    if args.symbols:
        symbols = [item.strip() for item in args.symbols.split(",") if item.strip()]
        universe = "custom"
    try:
        report = run(
            symbols,
            days=max(WARMUP + HORIZON + 40, min(args.days, 600)),
            horizon=args.horizon,
            split_ratio=args.split,
            rebalance=max(1, args.rebalance),
            top_fraction=min(0.9, max(0.05, args.top)),
            max_turnover=min(1.0, max(0.05, args.turnover)),
            refresh=args.refresh,
            offline=args.offline,
            universe=universe,
            max_symbols=args.max_symbols,
        )
    except (MarketDataError, ValueError) as exc:
        print(f"运行失败：{exc}", file=sys.stderr)
        return 2

    paths = write_reports(report, Path(args.outdir))
    stats = report["stats"]
    print(
        f"股票池 {report['params']['symbols']} 只，训练 {len(report['split']['train_dates'])} 天，"
        f"验证 {len(report['split']['valid_dates'])} 天"
    )
    print(f"  训练 {report['split']['train_start']} → {report['split']['train_end']}")
    print(f"  验证 {report['split']['valid_start']} → {report['split']['valid_end']}")
    print(f"  入选特征：{', '.join(report['screening']['features'])}")
    for key in ("strategy", "benchmark", "legacy"):
        stat = stats[key]
        print(
            f"  {key:10s} 总收益 {stat['total_return']:+.2f}%  年化 {stat['annual_return']:+.2f}%  "
            f"回撤 {stat['max_drawdown']:.2f}%  年换手 {stat['annual_turnover']:.0f}%  "
            f"成本 {stat['cost_drag_bps']:.0f}bp/年"
        )
    for key, path in paths.items():
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
