"""横截面建模的因果性、切分与回测约束测试。"""

from __future__ import annotations

import random
import unittest

import xsec


def make_history(
    symbol: str,
    *,
    days: int = 120,
    seed: int = 1,
    drift: float = 0.001,
) -> dict:
    """构造一段确定性的合成日线，避免测试依赖网络与缓存。"""
    rng = random.Random(seed)
    rows = []
    close = 10.0
    previous = close
    for index in range(days):
        change = rng.gauss(drift, 0.02)
        close = round(close * (1 + change), 4)
        rows.append(
            {
                "date": f"2024-{1 + index // 28:02d}-{1 + index % 28:02d}",
                "open": round(close * 0.998, 4),
                "close": close,
                "high": round(close * 1.01, 4),
                "low": round(close * 0.99, 4),
                "volume": round(1_000_000 * (1 + rng.random()), 0),
                "amount": 0.0,
                "amplitude": 0.0,
                "pct_change": round((close / previous - 1) * 100, 4),
                "change": 0.0,
                "turnover": 0.0,
            }
        )
        previous = close
    return {"symbol": symbol, "name": symbol, "asset_type": "stock", "rows": rows}


def make_benchmark(days: int = 120) -> dict:
    rows = []
    close = 3500.0
    for index in range(days):
        close = round(close * 1.0005, 2)
        rows.append({"date": f"2024-{1 + index // 28:02d}-{1 + index % 28:02d}", "close": close})
    return {"symbol": "000300", "name": "沪深300", "asset_type": "index", "rows": rows}


class LabelTest(unittest.TestCase):
    def test_label_is_within_day_percentile(self):
        """同一天的标签只取决于当日排名，与收益绝对值无关。"""
        histories = []
        # 5 只标的，构造严格单调的未来收益，便于验证分位
        for order in range(5):
            history = make_history(f"00000{order}", days=40, seed=order + 1)
            # 直接改写收盘价，让未来 1 日收益按 order 递增
            for index, row in enumerate(history["rows"]):
                row["close"] = 10.0 + order * 0.5 + index * (order + 1) * 0.001
            histories.append(history)

        panel = xsec.build_panel(
            histories, make_benchmark(40), horizon=1, warmup=2, min_xsec=3
        )
        date = panel["dates"][-2]
        members = {r["symbol"]: r for r in panel["records"] if r["date"] == date}
        self.assertEqual(len(members), 5)
        labels = sorted(r["label"] for r in members.values())
        self.assertAlmostEqual(labels[0], 0.0)
        self.assertAlmostEqual(labels[-1], 1.0)
        # 分位与未来收益的排序必须完全一致
        ordered = sorted(members.values(), key=lambda r: r["fwd"])
        self.assertEqual(
            [r["label"] for r in ordered],
            sorted(r["label"] for r in ordered),
        )
        # 分位落在 0–1 之间，且不与收益量级挂钩
        for record in members.values():
            self.assertGreaterEqual(record["label"], 0.0)
            self.assertLessEqual(record["label"], 1.0)


class CausalityTest(unittest.TestCase):
    def test_features_do_not_use_future_rows(self):
        """截断未来数据后，同一个评估点的特征必须完全不变。"""
        histories = [make_history(f"60000{index}", days=140, seed=index + 7) for index in range(30)]
        benchmark = make_benchmark(140)
        full = xsec.build_panel(histories, benchmark, horizon=5, warmup=30, min_xsec=10)

        target = full["dates"][60]
        truncated = [
            {**history, "rows": history["rows"][: 61 + 30]}  # warmup 30 + 第 60 个评估点
            for history in histories
        ]
        cut = xsec.build_panel(truncated, benchmark, horizon=5, warmup=30, min_xsec=10)
        if target not in cut["dates"]:
            self.skipTest("截断后该日期未进入有效横截面")

        full_rows = {r["symbol"]: r for r in full["records"] if r["date"] == target}
        cut_rows = {r["symbol"]: r for r in cut["records"] if r["date"] == target}
        self.assertTrue(full_rows)
        self.assertEqual(set(full_rows), set(cut_rows))
        for symbol, record in full_rows.items():
            for key in xsec.FEATURE_LABELS:
                self.assertEqual(record[key], cut_rows[symbol][key], f"{symbol} {key}")
            # 未来收益在截断样本里不存在，特征仍然可用
            self.assertIsNone(cut_rows[symbol]["fwd"])
            self.assertIsNone(cut_rows[symbol]["label"])

    def test_truncation_before_warmup_changes_nothing_for_early_dates(self):
        """只保留更长的历史不会改变早期评估点的特征值。"""
        history = make_history("600519", days=200, seed=3)
        benchmark = make_benchmark(200)
        long_panel = xsec.build_panel([history], benchmark, horizon=5, warmup=40, min_xsec=1)
        short_panel = xsec.build_panel(
            [{**history, "rows": history["rows"][:150]}], benchmark, horizon=5, warmup=40, min_xsec=1
        )
        common = sorted(set(long_panel["dates"]) & set(short_panel["dates"]))
        self.assertTrue(common)
        long_map = {r["date"]: r for r in long_panel["records"]}
        short_map = {r["date"]: r for r in short_panel["records"]}
        for date in common:
            for key in xsec.FEATURE_LABELS:
                self.assertEqual(long_map[date][key], short_map[date][key])


class SplitTest(unittest.TestCase):
    def test_train_labels_end_before_validation(self):
        dates = [f"d{index:04d}" for index in range(100)]
        split = xsec.time_split(dates, split_ratio=0.6, horizon=20, embargo=5)
        last_train = split["train_dates"][-1]
        self.assertEqual(split["valid_dates"][0], dates[60])
        # 最后一个训练样本的标签终点（+20）必须严格早于验证集第一天
        self.assertLess(dates.index(last_train) + 20, dates.index(split["valid_dates"][0]))
        # purge 数量 = horizon + embargo
        self.assertEqual(split["purged"], 25)
        # 训练与验证不重叠，且都按时间递增
        self.assertEqual(split["train_dates"], sorted(split["train_dates"]))
        self.assertEqual(split["valid_dates"], sorted(split["valid_dates"]))
        self.assertFalse(set(split["train_dates"]) & set(split["valid_dates"]))


class BacktestTest(unittest.TestCase):
    def _prices(self, symbols, dates, price=100.0):
        return {symbol: {date: price for date in dates} for symbol in symbols}

    def test_turnover_cap_and_cost(self):
        dates = [f"d{index:02d}" for index in range(21)]
        prices = self._prices(("A", "B"), dates)
        state = {"flip": False}

        def pick(date, available):
            state["flip"] = not state["flip"]
            return ["A"] if state["flip"] else ["B"]

        result = xsec.simulate(
            dates, prices, pick,
            rebalance_every=5, cost_buy=0.001, cost_sell=0.002, max_turnover=0.4,
        )
        for item in result["rebalances"]:
            self.assertLessEqual(item["one_way_turnover"], 0.4 + 1e-9)
            self.assertTrue(item["capped"])
        # 价格不变时净值下降只来自成本
        self.assertLess(result["final_nav"], 1.0)
        self.assertGreater(result["total_cost"], 0.0)
        # 成本 ≈ 买卖金额按各自费率加总，不应出现数量级偏差
        self.assertLess(result["total_cost"], 0.2)

    def test_equal_weight_flat_market_is_flat(self):
        dates = [f"d{index:02d}" for index in range(11)]
        prices = self._prices(("A", "B", "C"), dates)
        result = xsec.simulate(
            dates, prices, lambda date, available: list(available),
            rebalance_every=5, cost_buy=0.0, cost_sell=0.0, max_turnover=1.0,
        )
        self.assertAlmostEqual(result["final_nav"], 1.0, places=6)

    def test_missing_price_is_carried(self):
        """停牌（缺当日行情）时沿用上一笔价格，不应把仓位计成 0。"""
        dates = ["d0", "d1", "d2", "d3"]
        prices = {"A": {"d0": 10.0, "d2": 11.0, "d3": 12.0}, "B": {"d0": 10.0, "d1": 10.0, "d2": 10.0, "d3": 10.0}}
        result = xsec.simulate(
            dates, prices, lambda date, available: list(available),
            rebalance_every=10, cost_buy=0.0, cost_sell=0.0, max_turnover=1.0,
        )
        # A 从 10 涨到 12（+20%），B 不动，等权组合应为 +10%
        self.assertAlmostEqual(result["final_nav"], 1.1, places=6)


class ModelTest(unittest.TestCase):
    def test_fit_recovers_positive_exposure(self):
        """标签与某个特征同向时，拟合系数必须为正。"""
        records = []
        for index in range(400):
            rank = (index % 20) / 19
            records.append(
                {
                    "date": f"d{index // 20:03d}",
                    "symbol": f"S{index % 20:02d}",
                    "label": rank,
                    "fwd": rank,
                    "rank_macd_hist": rank,
                    "rank_vol_ratio": 1 - rank,
                    "rank_rs20": 0.5,
                    "macd_hist": 0.0,
                    "vol_ratio": 1.0,
                    "rs20": 0.0,
                }
            )
        model = xsec.fit_rank_model(records, ["macd_hist", "vol_ratio", "rs20"])
        self.assertGreater(model["coefficients"]["macd_hist"], 0.3)
        self.assertLess(model["coefficients"]["vol_ratio"], -0.3)
        self.assertAlmostEqual(model["intercept"], 0.5, places=2)

    def test_top_n_is_deterministic(self):
        scores = {"A": 0.2, "B": 0.9, "C": 0.5, "D": 0.5, "E": 0.1}
        self.assertEqual(xsec.top_n(scores, 0.4), ["B", "C"])
        self.assertEqual(xsec.top_n({}, 0.2), [])


if __name__ == "__main__":
    unittest.main()
