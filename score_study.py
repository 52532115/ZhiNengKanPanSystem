"""打分链路 v4 的离线验收台：硬阈值 → 连续 → 中性化 → 收缩 → 校准，逐档对照。

跑法
----
    python score_study.py --offline              # 复用 .cache/scoring/panel.json.gz
    python score_study.py --offline --rebuild    # 重新从 .cache/fields 构面板（约 4 分钟）
    python score_study.py --offline --iters 100  # 少跑几次 bootstrap

产出
----
    reports/scoring/report.md          # 结论与逐项指标
    reports/scoring/report.html        # 同内容 + 校准曲线 / 五分组 / 净值 SVG
    reports/scoring/metrics.json       # 机器可读，便于回归对比

为什么基线取"同一批因子的阶跃版"
--------------------------------
``analyzer.score_factors`` 的硬阈值是九条各不相同的规则，其中 RSI（48–68 才给分）
与 ret20（涨太多反而扣分）对原始因子**非单调**——这正是连续单调变换做不到的事。
为了把"连续化"这一项单独隔离出来，基线取同一批 12 个因子、同样的先验方向，
只把连续变换换成训练集三分位阶跃（{−1, 0, +1}）。两档之间唯一的差别是变换本身，
换手与 IC 的差才可以归因。
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import evalkit
import factors
import scoring
import xsec

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache" / "scoring" / "panel.json.gz"
FIELDS = ROOT / ".cache" / "fields"
INDUSTRY = ROOT / ".cache" / "research" / "pit-v1" / "industry.json"
REPORT = ROOT / "reports" / "scoring"

HORIZON = 20
REBALANCE = 5
TOP_FRAC = 0.20
SPLIT_RATIO = 0.60
EMBARGO = 5
MIN_XSEC = 20
WARMUP = 260

# 12 个因子，全部满足"值越大越看多"的单调先验（prior 在打分时已乘掉）。
BATTERY: dict[str, dict[str, Any]] = {
    "rev5":       {"label": "5 日反转", "prior": +1, "family": "反转"},
    "mom12_1":    {"label": "12–1 动量", "prior": +1, "family": "动量"},
    "macd_hist":  {"label": "MACD 柱/价格", "prior": +1, "family": "动量"},
    "rs20":       {"label": "20 日超额", "prior": +1, "family": "动量"},
    "ivol":       {"label": "特质波动", "prior": -1, "family": "风险"},
    "downside20": {"label": "下行波动", "prior": -1, "family": "风险"},
    "mdd60":      {"label": "60 日回撤", "prior": -1, "family": "风险"},
    "turnover20": {"label": "换手率", "prior": -1, "family": "流动性"},
    "amihud":     {"label": "Amihud 非流动", "prior": +1, "family": "流动性"},
    "clv20":      {"label": "收盘位置", "prior": +1, "family": "形态"},
    "ma_stack":   {"label": "均线多头", "prior": +1, "family": "形态"},
    "high52":     {"label": "52 周高点距离", "prior": +1, "family": "动量"},
}
FEATURES = list(BATTERY)
ATTACH = "s_"          # 逐因子分数写回记录时的前缀

# 权重先验：每个因子先验系数 spread/n，合计 spread（0.24 = 首尾组合差 24 个分位点）
PRIOR_SPREAD = 0.24
PRIOR_TAU = 0.04
MIN_SIGN_RATE = 0.80
# IC「持平」的判定容差。块自助法下 IC 的标准误约 0.03，0.005 只有它的 1/6，
# 任何小于这个量的差异都不能和抽样噪声区分——把容差写成常数，比事后挑一个
# 好看的门槛诚实。
IC_TOL = 0.005
SMOOTH_HALFLIFE = 5

BUY_COST = 0.00076
SELL_COST = 0.00126


# --------------------------------------------------------------------------- #
# 数据层
# --------------------------------------------------------------------------- #


def _benchmark() -> dict[str, Any]:
    with (FIELDS / "etf" / "510300.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _industry_map() -> dict[str, str]:
    with INDUSTRY.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {symbol: (row.get("industry") or "")[:1] or "?" for symbol, row in raw.items()}


def _forward_fill(values: Sequence[float | None]) -> list[float | None]:
    out: list[float | None] = []
    last: float | None = None
    for value in values:
        if value is not None and math.isfinite(value):
            last = value
        out.append(last)
    return out


def build_panel(stock_dir: Path | None = None) -> dict[str, Any]:
    """把个股日线摊成 (日期, 标的) 记录，含原始因子与 20 日超额收益。

    ``stock_dir`` 指向别的面板目录（如 PIT 池）时，用该目录的日线重建；默认仍读
    ``.cache/fields/stocks``（当前成分缓存），保持既有报告可复现。
    """
    benchmark = _benchmark()
    bench_close = {row["date"]: row["close"] for row in benchmark["rows"] if row["close"]}
    bench_dates = sorted(bench_close)
    bench_index = {date: i for i, date in enumerate(bench_dates)}
    bench_returns = factors.benchmark_returns(benchmark)
    industry = _industry_map()

    records: list[dict[str, Any]] = []
    source = Path(stock_dir) if stock_dir else (FIELDS / "stocks")
    for path in sorted(source.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            history = json.load(handle)
        symbol = history["symbol"]
        series = factors.compute(history, bench_returns["ret1"])
        series.update(factors.existing(history, bench_returns))
        caps = _forward_fill(factors.float_cap(history))
        rows = history["rows"]
        dates = [row["date"] for row in rows]
        closes = [float(row["close"]) for row in rows]
        group = industry.get(symbol, "?")

        for index, date in enumerate(dates):
            if index < WARMUP or index + HORIZON >= len(rows):
                continue
            position = bench_index.get(date)
            if position is None or position + HORIZON >= len(bench_dates):
                continue
            entry, exit_price = closes[index], closes[index + HORIZON]
            bench_entry, bench_exit = bench_close[date], bench_close[bench_dates[position + HORIZON]]
            if not (entry and exit_price and bench_entry and bench_exit):
                continue
            record: dict[str, Any] = {
                "date": date,
                "symbol": symbol,
                "industry": group,
                "fwd": (exit_price / entry - 1.0) * 100,
                "cap": caps[index],
                "beta": series["beta120"][index],
            }
            record["excess"] = record["fwd"] - (bench_exit / bench_entry - 1.0) * 100
            missing = False
            for key in FEATURES:
                column = series.get(key)
                value = column[index] if column else None
                if value is None or not math.isfinite(float(value)):
                    missing = True
                    break
                record[key] = float(value)
            if not missing and record["cap"] and record["beta"] is not None:
                records.append(record)

    counts: dict[str, int] = {}
    for record in records:
        counts[record["symbol"]] = counts.get(record["symbol"], 0) + 1
    keep = {symbol for symbol, count in counts.items() if count >= 400}
    records = [record for record in records if record["symbol"] in keep]
    return {"records": records, "symbols": sorted(keep), "benchmark": benchmark["name"]}


def panel_cache_path(stock_dir: Path | None = None) -> Path:
    """不同面板目录各用一份缓存：PIT 面板不能和当前成分面板共用同一份 gzip。"""
    if not stock_dir or Path(stock_dir) == FIELDS / "stocks":
        return CACHE
    tag = Path(stock_dir).name or "alt"
    return CACHE.with_name(f"panel_{tag}.json.gz")


def load_panel(*, rebuild: bool = False, stock_dir: Path | None = None) -> dict[str, Any]:
    cache = panel_cache_path(stock_dir)
    if cache.is_file() and not rebuild:
        with gzip.open(cache, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    panel = build_panel(stock_dir=stock_dir)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache, "wt", encoding="utf-8") as handle:
        json.dump(panel, handle, ensure_ascii=False)
    return panel


def group_by_date(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["date"], []).append(record)
    return {date: group for date, group in groups.items() if len(group) >= MIN_XSEC}


def add_labels(rows_by_date: dict[str, list[dict[str, Any]]]) -> None:
    """标签 = 当日超额收益的横截面分位（0–1）。"""
    for rows in rows_by_date.values():
        for record, rank in zip(rows, scoring.rank01([r["excess"] for r in rows])):
            record["label"] = rank


def split_dates(dates: Sequence[str]) -> dict[str, Any]:
    split = int(len(dates) * SPLIT_RATIO)
    train_end = max(1, split - HORIZON - EMBARGO)
    return {
        "train": list(dates[:train_end]),
        "valid": list(dates[split:]),
        "purged": split - train_end,
    }


# --------------------------------------------------------------------------- #
# 五档变体
# --------------------------------------------------------------------------- #


def step_cutpoints(train: dict[str, list[dict[str, Any]]]) -> dict[str, tuple[float, float]]:
    """硬阈值基线：训练集池化的三分位切点（等价于"典型阈值"）。"""
    cuts: dict[str, tuple[float, float]] = {}
    for key in FEATURES:
        values = sorted(record[key] for rows in train.values() for record in rows)
        if len(values) < 100:
            cuts[key] = (-math.inf, math.inf)
            continue
        low = values[int(0.33 * (len(values) - 1))]
        high = values[int(0.67 * (len(values) - 1))]
        if low == high:  # 离散因子退化时改用中位/上界
            high = values[-1]
        cuts[key] = (low, high)
    return cuts


def scores_step(
    rows_by_date: dict[str, list[dict[str, Any]]],
    cuts: dict[str, tuple[float, float]],
    *,
    neutral: bool = False,
) -> dict[str, dict[str, float]]:
    """阶跃打分：{−1,0,+1} × 先验，等权平均（硬阈值的最小复刻）。"""
    out: dict[str, dict[str, float]] = {}
    for date, rows in rows_by_date.items():
        columns: dict[str, list[float]] = {}
        for key in FEATURES:
            low, high = cuts[key]
            prior = BATTERY[key]["prior"]
            columns[key] = [
                prior * (1.0 if record[key] > high else (-1.0 if record[key] < low else 0.0))
                for record in rows
            ]
        if neutral:
            industries = [record["industry"] for record in rows]
            caps = [math.log(record["cap"]) for record in rows]
            betas = [record["beta"] for record in rows]
            for key in FEATURES:
                columns[key] = scoring.neutralize(
                    columns[key], industries=industries, log_cap=caps, beta=betas
                )
        out[date] = {
            record["symbol"]: statistics.fmean(columns[key][i] for key in FEATURES)
            for i, record in enumerate(rows)
        }
    return out


def scores_continuous(
    rows_by_date: dict[str, list[dict[str, Any]]],
    *,
    mode: str = "z",
    neutral: bool = True,
    weights: dict[str, float] | None = None,
    attach: bool = False,
) -> dict[str, dict[str, float]]:
    """连续打分：横截面标准化（+ 中性化）→ tanh → 先验 → 加权。

    ``attach=True`` 时把逐因子分数以 ``s_<key>`` 写回记录，供 ``build_design``
    直接读取——避免再维护一张 17 万行的分数表（内存差一个数量级）。
    """
    composite: dict[str, dict[str, float]] = {}
    keys = list(weights) if weights else FEATURES
    for date, rows in rows_by_date.items():
        industries = [record["industry"] for record in rows]
        caps = [math.log(record["cap"]) for record in rows]
        betas = [record["beta"] for record in rows]
        per_factor: dict[str, list[float]] = {}
        for key in keys:
            values = scoring.standardize(
                [record[key] for record in rows],
                industries=industries if neutral else None,
                log_cap=caps if neutral else None,
                beta=betas if neutral else None,
                mode=mode,
            )
            per_factor[key] = [BATTERY[key]["prior"] * v for v in values]
        if attach:
            for i, record in enumerate(rows):
                for key in keys:
                    record[ATTACH + key] = per_factor[key][i]
        composite[date] = {}
        for i, record in enumerate(rows):
            if weights:
                composite[date][record["symbol"]] = sum(
                    weights.get(key, 0.0) * per_factor[key][i] for key in keys
                )
            else:
                composite[date][record["symbol"]] = statistics.fmean(
                    per_factor[key][i] for key in keys
                )
    return composite


def smooth_scores(
    scores_by_date: dict[str, dict[str, float]],
    *,
    halflife: int = SMOOTH_HALFLIFE,
) -> dict[str, dict[str, float]]:
    """按标的做因果 EWMA：``s_t = α·x_t + (1−α)·s_{t−1}``，α = 1−0.5^{1/h}。

    换手的第一道闸门。连续化只解决"阈值附近跳分"，日内噪声本身还在：每 5 个
    交易日按 5 日噪声排序，名次照样抖。EWMA 把排序建立在"最近 h 天的平均观感"
    上，代价是信号滞后约 h/2 天。
    """
    alpha = 1.0 - 0.5 ** (1.0 / max(halflife, 1))
    state: dict[str, float] = {}
    out: dict[str, dict[str, float]] = {}
    for date in sorted(scores_by_date):
        row: dict[str, float] = {}
        for symbol, value in scores_by_date[date].items():
            previous = state.get(symbol)
            blended = value if previous is None else alpha * value + (1 - alpha) * previous
            state[symbol] = blended
            row[symbol] = blended
        out[date] = row
    return out


# --------------------------------------------------------------------------- #
# 权重拟合
# --------------------------------------------------------------------------- #


def fit_weights(
    rows_by_date: dict[str, list[dict[str, Any]]],
    *,
    iters: int,
    tau: float = PRIOR_TAU,
    spread: float = PRIOR_SPREAD,
) -> dict[str, Any]:
    design = scoring.build_design(rows_by_date, FEATURES, prefix=ATTACH)
    prior_mean = [spread / len(FEATURES)] * len(FEATURES)
    boot = scoring.bootstrap_fit(
        design, prior_mean=prior_mean, prior_tau=tau, iters=iters, block_days=HORIZON
    )
    weight = scoring.stability_weights(
        boot["point"], boot, prior_sign={key: 1 for key in FEATURES}, min_sign_rate=MIN_SIGN_RATE
    )
    total = sum(weight["weights"].values())
    scaled = {key: value * spread / total for key, value in weight["weights"].items()} if total else {}
    denominator = sum(abs(value) for value in scaled.values()) or 1.0
    return {
        "boot": boot,
        "point": boot["point"],
        "sign_rate": boot["sign_rate"],
        "ci90": boot["ci90"],
        "scaled": scaled,
        "share": {key: abs(value) / denominator for key, value in scaled.items()},
        "dropped": weight["dropped"],
        "n_train": boot["point"]["n_train"],
        "r2_in": boot["point"]["r2_in"],
        "prior_mean": prior_mean,
    }


def coefficient_stability(
    train: dict[str, list[dict[str, Any]]],
    full: dict[str, float],
) -> dict[str, Any]:
    """子样本符号：训练集前后两半各拟合一次，统计与全样本同号比例。"""
    dates = sorted(train)
    halves = {"first": dates[: len(dates) // 2], "second": dates[len(dates) // 2:]}
    rows: dict[str, dict[str, float]] = {}
    for name, subset in halves.items():
        design = scoring.build_design({date: train[date] for date in subset}, FEATURES, prefix=ATTACH)
        point = scoring.fit_ridge(
            design.blocks,
            prior_mean=[PRIOR_SPREAD / len(FEATURES)] * len(FEATURES),
            prior_tau=PRIOR_TAU,
            features=FEATURES,
        )
        rows[name] = point["coefficients"]
    agreement = {
        key: {
            "first": round(rows["first"][key], 4),
            "second": round(rows["second"][key], 4),
            "full": round(full[key], 4),
            "consistent": int(rows["first"][key] > 0) + int(rows["second"][key] > 0),
            "n": 2,
        }
        for key in FEATURES
    }
    return {"coefficients": rows, "agreement": agreement}


# --------------------------------------------------------------------------- #
# 验收指标
# --------------------------------------------------------------------------- #


def targets_of(
    scores_by_date: dict[str, dict[str, float]],
    dates: Sequence[str],
    *,
    use_buffer: bool = False,
    entry_pct: float = 0.20,
    exit_pct: float = 0.35,
) -> dict[str, list[str]]:
    """每 ``REBALANCE`` 个交易日给一次目标持仓。"""
    out: dict[str, list[str]] = {}
    held: list[str] = []
    for index, date in enumerate(dates):
        if index % REBALANCE:
            continue
        scores = scores_by_date.get(date)
        if not scores:
            continue
        if use_buffer:
            held = scoring.buffer_select(
                scores, held, entry_pct=entry_pct, exit_pct=exit_pct, target_pct=TOP_FRAC
            )
        else:
            held = scoring.top_fraction(scores, TOP_FRAC)
        out[date] = list(held)
    return out


def evaluate(
    name: str,
    scores_by_date: dict[str, dict[str, float]],
    forward: dict[str, dict[str, float]],
    excess: dict[str, dict[str, float]],
    dates: Sequence[str],
    *,
    use_buffer: bool = False,
    calibrator: scoring.IsotonicCalibrator | None = None,
    labels: dict[tuple[str, str], float] | None = None,
) -> dict[str, Any]:
    valid_scores = {date: scores_by_date[date] for date in dates if date in scores_by_date}
    series = scoring.daily_ic(valid_scores, excess, min_xsec=MIN_XSEC)
    ic = evalkit.ic_ci(series, horizon=HORIZON, iters=400)
    table = scoring.quintile_table(valid_scores, excess, groups=5, min_xsec=MIN_XSEC)
    targets = targets_of(valid_scores, dates, use_buffer=use_buffer)
    result = {
        "name": name,
        "ic": ic,
        "quintiles": table,
        "monotone": scoring.quintile_monotonicity(table),
        "turnover": scoring.turnover_series(list(targets.values()), step=REBALANCE),
        "targets": targets,
        "coverage": round(len(valid_scores) / max(len(dates), 1), 3),
    }
    if calibrator is not None and labels is not None:
        predicted: list[float] = []
        realized: list[float] = []
        by_date: dict[str, list[tuple[float, float]]] = {}
        for date, row in valid_scores.items():
            entries: list[tuple[float, float]] = []
            for symbol, score in row.items():
                label = labels.get((date, symbol))
                if label is not None:
                    value = calibrator.predict(score)
                    predicted.append(value)
                    realized.append(label)
                    entries.append((value, label))
            if entries:
                by_date[date] = entries
        curve = scoring.calibration_curve(predicted, realized, bins=scoring.DEFAULT_BINS)
        result["calibration"] = {
            "curve": curve,
            "metrics": scoring.calibration_metrics(curve),
            "n": len(predicted),
            # 参照系：什么都不预测、一律给 50% 时的 ECE。校准模型如果比它还差，
            # 说明"以为自己知道分位"这件事本身就是错的。
            "constant_ece": round(statistics.fmean([abs(r - 0.5) for r in realized]), 4),
            "spread": round(max(predicted) - min(predicted), 4) if predicted else None,
            "ci": scoring.calibration_metrics_ci(by_date, block=HORIZON),
        }
    return result


def net_backtest(
    targets: dict[str, list[str]],
    calendar: Sequence[str],
    prices: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """按目标持仓回测；非调仓日沿用上一期目标，避免"没有目标就当空仓卖光"。"""
    state = {"last": []}

    def pick(date: str, available: Sequence[str]) -> list[str]:
        if date in targets:
            state["last"] = targets[date]
        wanted = [symbol for symbol in state["last"] if symbol in available]
        return wanted or state["last"]

    result = xsec.simulate(
        calendar,
        prices,
        pick,
        rebalance_every=REBALANCE,
        cost_buy=BUY_COST,
        cost_sell=SELL_COST,
        max_turnover=1.0,
    )
    return {"nav": result["nav"], "metrics": xsec.metrics(result, len(calendar))}


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #


def _num(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _signed(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:+.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    variants = report["variants"]
    ic_rows = "\n".join(
        f"| {v['name']} | {_signed(v['ic']['mean'])} | [{_signed(v['ic']['ci_low'])}, {_signed(v['ic']['ci_high'])}] | "
        f"{_num(v['ic']['positive_rate'], 1)}% | {_num(v['ic']['n_blocks'], 1)} |"
        for v in variants
    )
    turn_rows = "\n".join(
        f"| {v['name']} | {v['turnover']['annual']:.0f}% | {v['turnover']['mean'] * 100:.1f}% | "
        f"{_num(v['monotone']['spearman'], 2)} | {v['monotone']['steps']} | {_num(v['monotone']['spread'], 2)} |"
        for v in variants
    )
    net_rows = "\n".join(
        f"| {m['name']} | {m['metrics'].get('annual_return')}% | {m['metrics'].get('sharpe')} | "
        f"{m['metrics'].get('max_drawdown')}% | {m['metrics'].get('annual_turnover')}% | "
        f"{m['metrics'].get('cost_drag_bps')}bp |"
        for m in report["net"]
    )
    weight_rows = "\n".join(
        f"| {BATTERY[key]['label']} | {row['point']:+.4f} | {row['sign_rate'] * 100:.0f}% | "
        f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] | {row['scaled']:+.4f} | "
        f"{'剔除' if row['dropped'] else '保留'} |"
        for key, row in report["weights"].items()
    )
    sign_rows = "\n".join(
        f"| {BATTERY[key]['label']} | {row['first']:+.4f} | {row['second']:+.4f} | {row['full']:+.4f} | "
        f"{row['consistent']}/{row['n']} |"
        for key, row in report["stability"]["agreement"].items()
    )
    cal = report["calibration"]
    cal_rows = "\n".join(
        f"| {i + 1} | {point['n']} | {point['pred'] * 100:.1f}% | {point['real'] * 100:.1f}% | "
        f"{(point['real'] - point['pred']) * 100:+.1f}pp |"
        for i, point in enumerate(cal["curve"])
    )
    verdict = report["verdict"]
    return f"""# 多因子打分 v4 验收报告

样本：{report['panel']['symbols']} 只个股 × {report['panel']['dates']} 个交易日
（{report['panel']['start']} → {report['panel']['end']}）；
训练 {report['split']['train_start']} → {report['split']['train_end']}，
验证 {report['split']['valid_start']} → {report['split']['valid_end']}，
purge {report['split']['purged']} 个交易日；标签周期 {HORIZON} 日。

## 结论

{verdict['summary']}

| 验收项 | 标准 | 实测 | 结论 |
| --- | --- | --- | --- |
| 换手下降 | ≥ 30% | {verdict['turnover']['text']} | {verdict['turnover']['flag']} |
| IC 不下降 | ΔIC ≥ 0 | {verdict['ic']['text']} | {verdict['ic']['flag']} |
| 校准接近 45° | ECE ≤ 0.05 且斜率 90% 区间覆盖 1.0 | {verdict['calibration']['text']} | {verdict['calibration']['flag']} |
| 符号稳定 | ≥ 80% bootstrap 同号 | {verdict['sign']['text']} | {verdict['sign']['flag']} |

## 1. 样本外 IC

| 变体 | IC | 90% 区间 | 正 IC 交易日 | 独立块 |
| --- | --- | --- | --- | --- |
{ic_rows}

## 2. 五分组与换手

| 变体 | 年化单边换手 | 每次换手 | 分组 Spearman | 递增步数 | Q5−Q1 |
| --- | --- | --- | --- | --- | --- |
{turn_rows}

## 3. 扣成本净值（验证集，单边 7.6bp / 12.6bp，无换手上限）

| 变体 | 年化 | Sharpe | 最大回撤 | 年化换手 | 成本拖累 |
| --- | --- | --- | --- | --- | --- |
{net_rows}

## 4. 权重与符号稳定性（bootstrap {report['weights_meta']['iters']} 次，块长 {HORIZON} 日）

| 因子 | 系数 | 同号率 | 90% 区间 | 最终权重 | 处理 |
| --- | --- | --- | --- | --- | --- |
{weight_rows}

## 5. 子样本系数符号

| 因子 | 前半段 | 后半段 | 全样本 | 一致 |
| --- | --- | --- | --- | --- |
{sign_rows}

## 6. 校准曲线（验证集；isotonic 只在训练集拟合）

预测 = 模型给出的预期分位，实际 = 当日实际分位。45° 线意味着"说 70% 就是 70%"。

| 分箱 | 样本 | 预测分位 | 实际分位 | 偏差 |
| --- | --- | --- | --- | --- |
{cal_rows}

ECE = {_num(cal['metrics']['ece'], 4)}，斜率 = {_num(cal['metrics']['slope'], 3)}，
最大偏差 = {_num(cal['metrics']['max_gap'], 4)}，离 45° 的 RMS = {_num(cal['metrics']['rms'], 4)}，
预测跨度 = {_num(cal.get('spread'), 3)}。

参照系：一律给 50% 的常数预测，样本外 ECE = {_num(cal.get('constant_ece'), 3)}。
拟合内 ECE = {_num(report['calibration_train']['metrics']['ece'], 4)}、
斜率 = {_num(report['calibration_train']['metrics']['slope'], 3)}。

## 口径说明

- 基线是同 12 因子的**三分位阶跃**版（训练集切点），与连续版唯一差别是变换，
  这样换手与 IC 的差才能归因到连续化本身。
- IC 一律用块自助法区间（块长 = 标签周期），区间覆盖 0 就不下结论。
- 换手 = 等权 Top 20%、每 5 日调仓、`|old−new|/|old|`，年化 = 均值 × 252/5。
- 校准只在训练集拟合，验证集只评估一次；isotonic 预分箱 25 箱后 PAVA。
"""


def _svg_calibration(curves: dict[str, list[dict[str, Any]]], *, width: int = 460, height: int = 360) -> str:
    """校准曲线：坐标轴按数据范围自适应，45° 线始终画在可见区内。

    固定 0–100% 的坐标轴在弱信号下会把所有点挤成中心的一个小团，看不出任何
    结构。自适应范围能让"预测 vs 实际"的偏差在视觉上放大到可判读的量级。
    """
    pad = 46
    points_all = [p for curve in curves.values() for p in curve]
    if not points_all:
        return ""
    lo = min(min(p["pred"] for p in points_all), min(p["real"] for p in points_all)) - 0.02
    hi = max(max(p["pred"] for p in points_all), max(p["real"] for p in points_all)) + 0.02
    lo, hi = max(0.0, lo), min(1.0, hi)
    if hi - lo < 0.04:                      # 极度退化时给一个最小窗口，避免除零
        center = (hi + lo) / 2
        lo, hi = max(0.0, center - 0.02), min(1.0, center + 0.02)
    span = hi - lo
    inner_w, inner_h = width - pad * 2, height - pad * 2

    def x(v: float) -> float:
        return pad + (v - lo) / span * inner_w

    def y(v: float) -> float:
        return height - pad - (v - lo) / span * inner_h

    ticks = []
    grid = []
    for i in range(5):
        value = lo + span * i / 4
        grid.append(
            f'<line x1="{x(value):.1f}" y1="{pad}" x2="{x(value):.1f}" y2="{height - pad}" stroke="#eceff3"/>'
            f'<line x1="{pad}" y1="{y(value):.1f}" x2="{width - pad}" y2="{y(value):.1f}" stroke="#eceff3"/>'
        )
        ticks.append(
            f'<text x="{x(value):.1f}" y="{height - pad + 16}" font-size="10" text-anchor="middle" fill="#7f8c8d">{value * 100:.0f}%</text>'
            f'<text x="{pad - 8}" y="{y(value) + 3:.1f}" font-size="10" text-anchor="end" fill="#7f8c8d">{value * 100:.0f}%</text>'
        )
    colors = {"训练集（拟合内）": "#2980b9", "验证集（样本外）": "#c0392b"}
    lines, dots, legend = [], [], []
    for index, (name, curve) in enumerate(curves.items()):
        color = colors.get(name, "#7f8c8d")
        if len(curve) > 1:
            path = " ".join(f"{x(p['pred']):.1f},{y(p['real']):.1f}" for p in curve)
            lines.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
        dots.extend(
            f'<circle cx="{x(p["pred"]):.1f}" cy="{y(p["real"]):.1f}" r="4.5" fill="{color}"/>' for p in curve
        )
        legend.append(
            f'<rect x="{pad + index * 138}" y="12" width="10" height="10" fill="{color}"/>'
            f'<text x="{pad + index * 138 + 15}" y="21" font-size="10" fill="#34495e">{name}</text>'
        )
    return f"""<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="校准曲线">
{''.join(grid)}
<line x1="{x(lo):.1f}" y1="{y(lo):.1f}" x2="{x(hi):.1f}" y2="{y(hi):.1f}" stroke="#95a5a6" stroke-dasharray="5 4"/>
{''.join(lines)}{''.join(dots)}{''.join(legend)}{''.join(ticks)}
<text x="{width / 2:.0f}" y="{height - 8}" font-size="11" text-anchor="middle" fill="#34495e">预测分位</text>
<text x="20" y="{height / 2:.0f}" font-size="11" text-anchor="middle" fill="#34495e" transform="rotate(-90 20 {height / 2:.0f})">实际分位</text>
</svg>"""


def _svg_nav(series: dict[str, list[dict[str, Any]]], *, width: int = 720, height: int = 320) -> str:
    pad = 44
    values = [point["nav"] for points in series.values() for point in points]
    if not values:
        return ""
    low, high = min(values), max(values)
    span = max(high - low, 1e-6)
    colors = ["#7f8c8d", "#2980b9", "#c0392b", "#1e8449"]
    inner_w, inner_h = width - pad * 2, height - pad * 2
    paths = []
    for index, (name, points) in enumerate(series.items()):
        count = max(len(points) - 1, 1)
        path = " ".join(
            f"{pad + i / count * inner_w:.1f},{height - pad - (p['nav'] - low) / span * inner_h:.1f}"
            for i, p in enumerate(points)
        )
        paths.append(f'<polyline points="{path}" fill="none" stroke="{colors[index % len(colors)]}" stroke-width="1.8"/>')
    legend = "".join(
        f'<rect x="{pad + i * 130}" y="10" width="10" height="10" fill="{colors[i % len(colors)]}"/>'
        f'<text x="{pad + i * 130 + 15}" y="19" font-size="10" fill="#34495e">{name}</text>'
        for i, name in enumerate(series)
    )
    return f"""<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="净值曲线">
<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#dfe4ea"/>
{''.join(paths)}{legend}
<text x="{pad}" y="{height - 12}" font-size="10" fill="#7f8c8d">净值区间 {low:.2f} – {high:.2f}</text>
</svg>"""


def _svg_quintiles(tables: dict[str, list[dict[str, Any]]], *, width: int = 720, height: int = 330) -> str:
    """五分组柱状图：每组建内对比多个变体，基线固定在 0。

    底部留 60px 给组标签，柱体从 0 起算而不是从"最小值"起算——柱状图的长度
    必须与数值成正比，从最小值截断会放大组间差异，这是最容易被误读的一种画法。
    """
    names = list(tables)
    groups = [point["group"] for point in tables[names[0]]]
    values = [point["mean"] or 0.0 for table in tables.values() for point in table]
    low, high = min(values + [0.0]), max(values + [0.0])
    span = max(high - low, 0.5)
    top, bottom = 44, height - 62
    colors = ["#7f8c8d", "#2980b9", "#c0392b", "#1e8449"]

    def y(value: float) -> float:
        return bottom - (value - low) / span * (bottom - top)

    slot = (width - 96) / max(len(groups), 1)
    bar_w = slot / (len(names) + 0.5)
    bars: list[str] = []
    for series_index, name in enumerate(names):
        for group_index, point in enumerate(tables[name]):
            value = point["mean"] or 0.0
            x = 70 + group_index * slot + series_index * bar_w
            y0, y1 = y(value), y(0.0)
            bars.append(
                f'<rect x="{x:.1f}" y="{min(y0, y1):.1f}" width="{bar_w * 0.82:.1f}" '
                f'height="{max(abs(y1 - y0), 0.8):.1f}" fill="{colors[series_index % len(colors)]}" opacity="0.88"/>'
            )
    labels = "".join(
        f'<text x="{70 + i * slot + slot / 2 - bar_w * 0.25:.1f}" y="{height - 26}" font-size="11" '
        f'text-anchor="middle" fill="#34495e">{group}</text>'
        for i, group in enumerate(groups)
    )
    legend = "".join(
        f'<rect x="{70 + i * 150}" y="12" width="10" height="10" fill="{colors[i % len(colors)]}"/>'
        f'<text x="{70 + i * 150 + 15}" y="21" font-size="10" fill="#34495e">{name}</text>'
        for i, name in enumerate(names)
    )
    ticks = "".join(
        f'<line x1="64" y1="{y(v):.1f}" x2="{width - 16}" y2="{y(v):.1f}" stroke="#eef1f5"/>'
        f'<text x="58" y="{y(v) + 3:.1f}" font-size="10" text-anchor="end" fill="#95a5a6">{v:.1f}</text>'
        for v in (low, low + span / 2, high)
    )
    return f"""<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="五分组超额收益">
{ticks}
<line x1="64" y1="{y(0):.1f}" x2="{width - 16}" y2="{y(0):.1f}" stroke="#bdc3c7"/>
{''.join(bars)}{legend}{labels}
<text x="{width - 20}" y="21" font-size="10" text-anchor="end" fill="#7f8c8d">20 日超额（%）</text>
</svg>"""


def render_html(report: dict[str, Any]) -> str:
    variants = report["variants"]
    cal = report["calibration"]
    quint = {
        v["name"]: v["quintiles"] for v in variants
        if v["name"] in ("B0 阶跃（基线）", "V3 贝叶斯收缩", "V4b 收缩+缓冲（推荐）")
    }
    verdict = report["verdict"]
    rows = "".join(
        f"<tr><td>{v['name']}</td><td>{_signed(v['ic']['mean'])}</td>"
        f"<td>[{_signed(v['ic']['ci_low'])}, {_signed(v['ic']['ci_high'])}]</td>"
        f"<td>{v['turnover']['annual']:.0f}%</td><td>{_num(v['monotone']['spearman'], 2)}</td>"
        f"<td>{_num(v['monotone']['spread'], 2)}</td></tr>"
        for v in variants
    )
    weight_rows = "".join(
        f"<tr><td>{BATTERY[key]['label']}</td><td>{row['point']:+.4f}</td><td>{row['sign_rate'] * 100:.0f}%</td>"
        f"<td>[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]</td><td>{row['scaled']:+.4f}</td>"
        f"<td>{'剔除' if row['dropped'] else '保留'}</td></tr>"
        for key, row in report["weights"].items()
    )
    sign_rows = "".join(
        f"<tr><td>{BATTERY[k]['label']}</td><td>{r['first']:+.4f}</td><td>{r['second']:+.4f}</td>"
        f"<td>{r['full']:+.4f}</td><td>{r['consistent']}/{r['n']}</td></tr>"
        for k, r in report["stability"]["agreement"].items()
    )
    net_rows = "".join(
        f"<tr><td>{m['name']}</td><td>{m['metrics'].get('annual_return')}%</td><td>{m['metrics'].get('sharpe')}</td>"
        f"<td>{m['metrics'].get('max_drawdown')}%</td><td>{m['metrics'].get('annual_turnover')}%</td>"
        f"<td>{m['metrics'].get('cost_drag_bps')}bp</td></tr>"
        for m in report["net"]
    )
    cal_rows = "".join(
        f"<tr><td>{i + 1}</td><td>{p['pred'] * 100:.1f}%</td><td>{p['real'] * 100:.1f}%</td>"
        f"<td>{(p['real'] - p['pred']) * 100:+.1f}pp</td></tr>"
        for i, p in enumerate(cal["curve"])
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>多因子打分 v4 验收</title>
<style>
:root{{--ink:#1f2a37;--muted:#6b7785;--line:#e3e8ef;--bg:#f7f9fc;--red:#c0392b;--green:#1e8449;--blue:#1f6fb2}}
*{{box-sizing:border-box}}
body{{margin:0;padding:32px 40px;background:var(--bg);color:var(--ink);font:14px/1.75 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}}
h1{{font-size:23px;margin:0 0 6px}}
h2{{font-size:15px;margin:30px 0 10px;padding-left:10px;border-left:3px solid var(--blue);color:#26364a}}
.sub{{color:var(--muted);font-size:12.5px}}
.card{{background:#fff;border:1px solid var(--line);border-radius:8px;padding:16px 20px;margin-bottom:14px}}
table{{width:100%;border-collapse:collapse;font-size:13px;background:#fff}}
th,td{{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right}}
th:first-child,td:first-child{{text-align:left}}
th{{color:var(--muted);font-weight:600;font-size:12px;background:#fbfcfe}}
.grid{{display:grid;grid-template-columns:1.1fr 1fr;gap:14px}}
.verdict{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:16px 0}}
.pill{{background:#fff;border:1px solid var(--line);border-radius:8px;padding:10px 14px}}
.pill b{{display:block;font-size:12px;color:var(--muted);font-weight:600;margin-bottom:2px}}
.pass{{color:var(--green);font-weight:600}} .fail{{color:var(--red);font-weight:600}}
code{{background:#eef2f7;padding:1px 5px;border-radius:3px;font-size:12px}}
</style></head><body>
<h1>多因子打分 v4 · 样本外验收</h1>
<div class="sub">{report['panel']['symbols']} 只个股 · {report['panel']['dates']} 个交易日 · 训练 {report['split']['train_start']} → {report['split']['train_end']} ·
验证 {report['split']['valid_start']} → {report['split']['valid_end']} · 标签 {HORIZON} 日 · bootstrap {report['weights_meta']['iters']} 次</div>

<div class="verdict">
  <div class="pill"><b>换手</b><span class="{verdict['turnover']['class']}">{verdict['turnover']['text']}</span></div>
  <div class="pill"><b>IC</b><span class="{verdict['ic']['class']}">{verdict['ic']['text']}</span></div>
  <div class="pill"><b>校准</b><span class="{verdict['calibration']['class']}">{verdict['calibration']['text']}</span></div>
  <div class="pill"><b>符号稳定</b><span class="{verdict['sign']['class']}">{verdict['sign']['text']}</span></div>
</div>
<div class="card">{verdict['summary']}</div>

<h2>1. 五档对照（验证集）</h2>
<div class="card"><table>
<tr><th>变体</th><th>IC</th><th>90% 区间</th><th>年化换手</th><th>分组 Spearman</th><th>Q5−Q1</th></tr>
{rows}</table></div>

<h2>2. 校准曲线</h2>
<div class="grid">
  <div class="card">{_svg_calibration({"训练集（拟合内）": report['calibration_train']['curve'], "验证集（样本外）": cal['curve']})}</div>
  <div class="card"><table><tr><th>分箱</th><th>预测</th><th>实际</th><th>偏差</th></tr>{cal_rows}</table>
  <p class="sub">样本外 ECE {_num(cal['metrics']['ece'], 4)} · 斜率 {_num(cal['metrics']['slope'], 3)} · RMS {_num(cal['metrics']['rms'], 4)} ·
  预测跨度 {_num(cal.get('spread'), 3)}；一律给 50% 的常数预测 ECE {_num(cal.get('constant_ece'), 3)}</p></div>
</div>

<h2>3. 五分组 20 日超额（%）</h2>
<div class="card">{_svg_quintiles(quint)}</div>

<h2>4. 扣成本净值</h2>
<div class="card">{_svg_nav(report['nav'])}<table>
<tr><th>变体</th><th>年化</th><th>Sharpe</th><th>最大回撤</th><th>换手</th><th>成本</th></tr>
{net_rows}</table></div>

<h2>5. 权重与符号稳定性</h2>
<div class="card"><table>
<tr><th>因子</th><th>系数</th><th>同号率</th><th>90% 区间</th><th>最终权重</th><th>处理</th></tr>
{weight_rows}</table></div>

<h2>6. 子样本系数符号</h2>
<div class="card"><table>
<tr><th>因子</th><th>前半段</th><th>后半段</th><th>全样本</th><th>一致</th></tr>
{sign_rows}</table></div>

<p class="sub" style="margin-top:22px">口径：IC 用块自助法（块长 {HORIZON} 日），区间覆盖 0 不构成结论；换手 = 等权 Top 20%、每 {REBALANCE} 日调仓的单边比例；
校准在训练集拟合、验证集评估；阶跃基线使用同一批因子的训练集三分位切点。</p>
</body></html>"""


def _flag(ok: bool) -> dict[str, str]:
    return {"flag": "通过" if ok else "未通过", "class": "pass" if ok else "fail"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="多因子打分 v4 验收")
    parser.add_argument("--offline", action="store_true", help="只用本地缓存")
    parser.add_argument("--rebuild", action="store_true", help="重建面板缓存")
    parser.add_argument("--iters", type=int, default=200, help="bootstrap 次数")
    parser.add_argument("--outdir", default=str(REPORT))
    args = parser.parse_args(argv)

    print("[1/5] 载入面板…", flush=True)
    panel = load_panel(rebuild=args.rebuild)
    rows_by_date = group_by_date(panel["records"])
    add_labels(rows_by_date)
    dates = sorted(rows_by_date)
    split = split_dates(dates)
    train = {date: rows_by_date[date] for date in split["train"]}
    print(f"      训练 {len(split['train'])} 日 / 验证 {len(split['valid'])} 日 / 记录 {len(panel['records'])} 条", flush=True)

    print("[2/5] 构造五档分数…", flush=True)
    cuts = step_cutpoints(train)
    step = scores_step(rows_by_date, cuts)
    step_neutral = scores_step(rows_by_date, cuts, neutral=True)
    z_score = scores_continuous(rows_by_date, mode="z", neutral=False)
    n_score = scores_continuous(rows_by_date, mode="z", neutral=True, attach=True)
    r_score = scores_continuous(rows_by_date, mode="rank", neutral=True)
    print("      ok", flush=True)

    print("[3/5] 贝叶斯岭 + bootstrap…", flush=True)
    weights = fit_weights(train, iters=args.iters)
    weighted = scores_continuous(rows_by_date, mode="z", neutral=True, weights=weights["scaled"])
    smooth = smooth_scores(weighted)
    print(f"      同号率中位 {statistics.median(weights['sign_rate'].values()):.2f}，"
          f"剔除 {len(weights['dropped'])} 个，R²(in) {weights['r2_in']:.4f}", flush=True)

    print("[4/5] 校准与评估…", flush=True)
    labels = {
        (record["date"], record["symbol"]): record["label"]
        for record in panel["records"] if record.get("label") is not None
    }
    train_pairs = [
        (score, labels[(date, symbol)], date)
        for date in split["train"] if date in weighted
        for symbol, score in weighted[date].items()
        if (date, symbol) in labels
    ]
    # λ 用训练集内部的 5 折 purged 交叉验证估（purge = 标签周期），验证集不参与
    calibrator = scoring.IsotonicCalibrator(bins=25).fit(
        [p[0] for p in train_pairs],
        [p[1] for p in train_pairs],
        dates=[p[2] for p in train_pairs],
        folds=5,
        purge=HORIZON,
    )
    forward: dict[str, dict[str, float]] = {}
    excess: dict[str, dict[str, float]] = {}
    for record in panel["records"]:
        forward.setdefault(record["date"], {})[record["symbol"]] = record["fwd"]
        excess.setdefault(record["date"], {})[record["symbol"]] = record["excess"]

    # 逐档拆开：变换 → 中性化 → 收缩 → 平滑 → 缓冲，每一步单独看换手与 IC，
    # 否则"换手降了 80%"这种结论无法归因到具体动作。
    variants = [
        evaluate("B0 阶跃（基线）", step, forward, excess, split["valid"]),
        evaluate("B0c 阶跃+缓冲带", step, forward, excess, split["valid"], use_buffer=True),
        evaluate("B0b 阶跃+中性化", step_neutral, forward, excess, split["valid"]),
        evaluate("V1 连续 z（未中性）", z_score, forward, excess, split["valid"]),
        evaluate("V2 连续 z+中性化", n_score, forward, excess, split["valid"]),
        evaluate("V2r 连续 rank+中性化", r_score, forward, excess, split["valid"]),
        evaluate("V3 贝叶斯收缩", weighted, forward, excess, split["valid"]),
        evaluate("V3s 收缩+EWMA", smooth, forward, excess, split["valid"]),
        evaluate("V4 平滑+缓冲（激进）", smooth, forward, excess, split["valid"], use_buffer=True),
        evaluate("V4b 收缩+缓冲（推荐）", weighted, forward, excess, split["valid"], use_buffer=True),
    ]
    cal_variant = evaluate(
        "V4b 收缩+缓冲（推荐）", weighted, forward, excess, split["valid"],
        use_buffer=True, calibrator=calibrator, labels=labels,
    )
    # 拟合内校准曲线：如果这一条都不是 45°，说明实现有问题；如果它接近 45° 而
    # 样本外不是，那差的是信号而不是代码。两张图必须一起看。
    cal_train = evaluate(
        "V4b（拟合内）", weighted, forward, excess, split["train"],
        use_buffer=True, calibrator=calibrator, labels=labels,
    )

    print("[5/5] 回测与报告…", flush=True)
    prices: dict[str, dict[str, float]] = {}
    source = Path(stock_dir) if stock_dir else (FIELDS / "stocks")
    for path in sorted(source.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            history = json.load(handle)
        if history["symbol"] in panel["symbols"]:
            prices[history["symbol"]] = {row["date"]: row["close"] for row in history["rows"] if row["close"]}
    calendar = [date for date in split["valid"] if any(date in series for series in prices.values())]

    wanted = ("B0 阶跃（基线）", "V3 贝叶斯收缩", "V4b 收缩+缓冲（推荐）")
    net: list[dict[str, Any]] = []
    nav: dict[str, list[dict[str, Any]]] = {}
    for variant in variants:
        if variant["name"] not in wanted:
            continue
        backtest = net_backtest(variant["targets"], calendar, prices)
        net.append({"name": variant["name"], "metrics": backtest["metrics"]})
        nav[variant["name"]] = backtest["nav"]
    equal = xsec.simulate(
        calendar, prices, lambda date, available: sorted(available),
        rebalance_every=252, cost_buy=BUY_COST, cost_sell=SELL_COST, max_turnover=1.0,
    )
    nav["等权持有"] = equal["nav"]
    net.append({"name": "等权持有", "metrics": xsec.metrics(equal, len(calendar))})

    stability = coefficient_stability(train, weights["point"]["coefficients"])

    base, final = variants[0], variants[-1]
    drop = 1 - final["turnover"]["annual"] / base["turnover"]["annual"] if base["turnover"]["annual"] else 0.0
    base_ic = base["ic"]["mean"] or 0.0
    final_ic = final["ic"]["mean"] or 0.0
    delta_ic = final_ic - base_ic
    cal_metrics = cal_variant["calibration"]["metrics"]
    rates = list(weights["sign_rate"].values())
    passed_sign = sum(1 for v in rates if v >= MIN_SIGN_RATE)
    kept = len(weights["scaled"])
    low_sign = sum(1 for item in weights["dropped"] if "符号稳定性" in item["reason"])
    conflict = len(weights["dropped"]) - low_sign
    cal_shrink = calibrator.shrink
    spread = cal_variant["calibration"]["spread"] or 0.0
    degraded = spread < 0.05
    cal_ci = cal_variant["calibration"]["ci"]
    # 45° 的判定用斜率区间而不是点估计：预测跨度只有 0.02 时，斜率的抽样误差
    # 本身就有 ±0.4，拿点估计判"通过/不通过"等于在用噪声做决定。
    cal_ok = bool(
        not degraded
        and cal_metrics["ece"] is not None and cal_metrics["ece"] <= 0.05
        and cal_ci.get("covers_one")
    )
    verdict = {
        "turnover": {**_flag(drop >= 0.30),
                     "text": f"{drop:.0%}（{base['turnover']['annual']:.0f}% → {final['turnover']['annual']:.0f}%）"},
        "ic": {**_flag(delta_ic >= -IC_TOL),
               "text": f"ΔIC {delta_ic:+.4f}（{base_ic:+.4f} → {final_ic:+.4f}，容差 ±{IC_TOL}）"},
        "calibration": {
            **_flag(cal_ok),
            "text": (
                f"退化：预测跨度仅 {spread:.2f} · ECE {cal_metrics['ece']:.3f}"
                if degraded else
                f"ECE {cal_metrics['ece']:.3f} · 斜率 {cal_metrics['slope']:.2f} "
                f"[{cal_ci['slope_low']:.2f}, {cal_ci['slope_high']:.2f}]"
            ) if cal_metrics["ece"] is not None else "样本不足",
        },
        "sign": {**_flag(kept == len(rates)), "text": f"{passed_sign}/{len(rates)} 个因子同号率 ≥ 80%，保留 {kept} 个"},
    }
    covers_zero = final["ic"]["covers_zero"]
    ewma = next((v for v in variants if v["name"] == "V4 平滑+缓冲（激进）"), None)
    ewma_note = ""
    if ewma:
        ewma_ic = ewma["ic"]["mean"] or 0.0
        if abs(ewma_ic - final_ic) < 0.002:
            ewma_note = (
                f"再叠加 EWMA(halflife {SMOOTH_HALFLIFE})，换手进一步降到 "
                f"{ewma['turnover']['annual']:.0f}%，IC 仍是 {ewma_ic:+.4f}（差异 < 0.002，落在噪声内）——"
                f"这份数据里平滑没有明显代价，可以打开，但不要指望它创造 alpha。"
            )
        else:
            ewma_note = (
                f"再叠加 EWMA(halflife {SMOOTH_HALFLIFE})，换手降到 "
                f"{ewma['turnover']['annual']:.0f}%，但 IC 从 {final_ic:+.4f} 变成 {ewma_ic:+.4f}——"
                f"平滑是在拿 alpha 换成本，所以它不进推荐配置。"
            )
    verdict["summary"] = (
        f"换手{'达标' if drop >= 0.30 else '未达标'}：连续化 → 中性化 → 贝叶斯收缩 → 缓冲带把年化单边换手从 "
        f"{base['turnover']['annual']:.0f}% 降到 {final['turnover']['annual']:.0f}%（−{drop:.0%}），"
        f"IC 变化 {delta_ic:+.4f}（容差 ±{IC_TOL}，{'视为持平' if abs(delta_ic) <= IC_TOL else '已超出噪声'}）。"
        f"样本外 IC 区间 [{final['ic']['ci_low']:+.4f}, {final['ic']['ci_high']:+.4f}]，"
        f"{'覆盖 0：这套链路把换手和成本降下来了，但没有把信号变强，仍然不能当作 alpha 用' if covers_zero else '不覆盖 0'}。"
        f"校准 ECE {cal_metrics['ece']:.3f}、斜率 {cal_metrics['slope']:.2f}"
        f"（90% 区间 [{cal_ci['slope_low']:.2f}, {cal_ci['slope_high']:.2f}]，λ_cv = {cal_shrink:.2f}）："
        f"{'区间覆盖 1.0，不能拒绝 45°' if cal_ci.get('covers_one') else '区间不含 1.0，45° 被拒绝'}，"
        f"根因是预测跨度只有 {spread:.3f}（IC 仅 {final_ic:+.3f}）。"
        f"系数符号：{passed_sign}/{len(rates)} 个因子在 bootstrap 里同号率 ≥ {MIN_SIGN_RATE:.0%}，"
        f"最终保留 {kept} 个、剔除 {len(weights['dropped'])} 个（{low_sign} 个同号率不足、{conflict} 个方向与先验相反）。"
        + ewma_note
    )

    report = {
        "panel": {"symbols": len(panel["symbols"]), "dates": len(dates),
                  "start": dates[0], "end": dates[-1]},
        "split": {
            "train_start": split["train"][0], "train_end": split["train"][-1],
            "valid_start": split["valid"][0], "valid_end": split["valid"][-1],
            "purged": split["purged"],
        },
        "variants": variants,
        "weights": {
            key: {
                "point": weights["point"]["coefficients"][key],
                "sign_rate": weights["sign_rate"][key],
                "ci_low": weights["ci90"][key][0],
                "ci_high": weights["ci90"][key][1],
                "scaled": weights["scaled"].get(key, 0.0),
                "dropped": any(item["key"] == key for item in weights["dropped"]),
            }
            for key in FEATURES
        },
        "weights_meta": {"iters": args.iters, "n_train": weights["n_train"], "r2_in": weights["r2_in"]},
        "stability": stability,
        "calibration": {**cal_variant["calibration"], **calibrator.summary()},
        "calibration_train": cal_train["calibration"],
        "net": net,
        "nav": nav,
        "verdict": verdict,
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    (outdir / "report.html").write_text(render_html(report), encoding="utf-8")
    slim = {key: value for key, value in report.items() if key != "nav"}
    for variant in slim["variants"]:
        variant.pop("targets", None)
    (outdir / "metrics.json").write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n完成 → {outdir}")
    for variant in variants:
        print(f"  {variant['name']:<18} IC {_signed(variant['ic']['mean'])} "
              f"[{_signed(variant['ic']['ci_low'])},{_signed(variant['ic']['ci_high'])}]  "
              f"换手 {variant['turnover']['annual']:.0f}%  单调 {_num(variant['monotone']['spearman'], 2)}")
    print(f"  校准 ECE {cal_metrics['ece']:.4f} 斜率 {cal_metrics['slope']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
