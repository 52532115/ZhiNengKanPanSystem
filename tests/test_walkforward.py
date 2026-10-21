import math
import unittest

from analyzer import analyze
from walkforward import (
    BUCKETS,
    bucket_key,
    build_report,
    factor_table,
    quantile_groups,
    consistency_check,
    per_symbol_stats,
    render_html,
    render_markdown,
    score_breakdown,
    summarize,
    walk_forward,
)


def make_history(count=300, drift=0.25, wave=0.9, symbol="600000", asset_type="stock"):
    rows = []
    price = 20.0
    for i in range(count):
        price = max(1.5, price + drift + math.sin(i / 7.0) * wave)
        rows.append(
            {
                "date": f"D{i:04d}",
                "open": price - 0.05,
                "close": price,
                "high": price + 0.25,
                "low": price - 0.25,
                "volume": 1_000_000 + (i % 17) * 30_000,
                "amount": 20_000_000,
                "amplitude": 2.0,
                "pct_change": drift / max(price, 1) * 100,
                "change": drift,
                "turnover": 1.2,
            }
        )
    return {"symbol": symbol, "name": "测试标的", "asset_type": asset_type, "rows": rows}


class NoLookaheadTests(unittest.TestCase):
    def test_scores_ignore_rows_after_evaluation_day(self):
        full = walk_forward(make_history(300), stride=1)
        self.assertTrue(full)
        truncated_history = make_history(300)
        truncated_history["rows"] = truncated_history["rows"][:250]
        truncated = walk_forward(truncated_history, stride=1)

        full_by_date = {sample["date"]: sample["score"] for sample in full}
        # 截断后仍存在的评估点，分数必须与完整序列完全一致。
        overlap = [sample for sample in truncated if sample["date"] in full_by_date]
        self.assertGreater(len(overlap), 50)
        for sample in overlap:
            self.assertEqual(sample["score"], full_by_date[sample["date"]])

        # 截断序列不可能评到未来数据。
        self.assertLess(max(s["date"] for s in truncated), max(s["date"] for s in full))

    def test_forward_label_uses_future_rows_only(self):
        history = make_history(300)
        samples = walk_forward(history, stride=1)
        closes = {row["date"]: row["close"] for row in history["rows"]}
        order = [row["date"] for row in history["rows"]]
        sample = samples[10]
        index = order.index(sample["date"])
        expected = (closes[order[index + 20]] / closes[sample["date"]] - 1) * 100
        self.assertAlmostEqual(sample["forward"]["20"]["return"], expected, places=6)
        self.assertLessEqual(sample["forward"]["20"]["mdd"], 0)


class ShadowScoreTests(unittest.TestCase):
    def test_shadow_breakdown_matches_analyze(self):
        for drift, wave in ((0.35, 0.6), (-0.3, 1.2), (0.0, 2.2)):
            for count in (140, 260, 320):
                history = make_history(count, drift=drift, wave=wave)
                score, parts = score_breakdown(history)
                self.assertEqual(score, analyze(history)["score"], f"drift={drift} count={count}")
                self.assertEqual(
                    set(parts),
                    {"trend", "align", "long", "macd", "rsi", "ret20", "volume", "volatility", "drawdown"},
                )

    def test_breakdown_sums_to_raw_score(self):
        history = make_history(300)
        score, parts = score_breakdown(history)
        raw = 50 + sum(parts.values())
        self.assertEqual(score, max(0, min(100, round(raw))))


class BucketTests(unittest.TestCase):
    def test_boundaries_follow_view_thresholds(self):
        self.assertEqual(bucket_key(0), "lt30")
        self.assertEqual(bucket_key(29), "lt30")
        self.assertEqual(bucket_key(30), "30_44")
        self.assertEqual(bucket_key(44), "30_44")
        self.assertEqual(bucket_key(45), "45_59")
        self.assertEqual(bucket_key(59), "45_59")
        self.assertEqual(bucket_key(60), "60_74")
        self.assertEqual(bucket_key(74), "60_74")
        self.assertEqual(bucket_key(75), "ge75")
        self.assertEqual(bucket_key(100), "ge75")
        self.assertEqual(len(BUCKETS), 5)


class SummaryTests(unittest.TestCase):
    def test_bucket_stats_and_win_rate(self):
        samples = []
        for score, forward in ((80, [2.0, 3.0, 1.0]), (20, [-2.0, -1.0, -4.0])):
            for value in forward:
                samples.append(
                    {
                        "symbol": "600000",
                        "name": "测试标的",
                        "date": f"D{len(samples):04d}",
                        "score": score,
                        "view": "偏积极" if score >= 75 else "高风险",
                        "parts": {"trend": 9 if score >= 75 else -9},
                        "forward": {
                            "5": {"return": value, "mdd": -1.0},
                            "20": {"return": value * 2, "mdd": -2.0},
                            "60": {"return": value * 3, "mdd": -3.0},
                        },
                    }
                )
        summary = summarize(samples, (5, 20, 60))
        top = summary["buckets"][-1]
        bottom = summary["buckets"][0]
        self.assertEqual(top["n"], 3)
        self.assertEqual(top["horizons"]["5"]["win_rate"], 100.0)
        self.assertEqual(bottom["horizons"]["5"]["win_rate"], 0.0)
        self.assertAlmostEqual(top["horizons"]["5"]["mean"], 2.0)
        self.assertGreater(summary["spread"]["5"]["top_minus_bottom"], 0)
        self.assertEqual(summary["buckets"][2]["horizons"]["5"]["mean"], None)

    def test_empty_bucket_is_not_an_error(self):
        summary = summarize([], (5, 20, 60))
        self.assertEqual(summary["buckets"][0]["n"], 0)
        self.assertIsNone(summary["buckets"][0]["horizons"]["5"]["mean"])
        self.assertIsNone(summary["spread"]["20"]["top_minus_bottom"])

    def test_factor_edge_direction(self):
        samples = []
        for index in range(40):
            positive = index % 2 == 0
            samples.append(
                {
                    "symbol": "600000",
                    "date": f"D{index:04d}",
                    "score": 60,
                    "parts": {"trend": 9 if positive else -9, "macd": 0},
                    "forward": {
                        "5": {"return": 1.0, "mdd": -1.0},
                        "20": {"return": 3.0 if positive else -1.0, "mdd": -2.0},
                        "60": {"return": 0.0, "mdd": -3.0},
                    },
                }
            )
        rows = factor_table(samples, 20)
        trend = next(row for row in rows if row["key"] == "trend")
        self.assertAlmostEqual(trend["edge"], 4.0)
        self.assertEqual(trend["verdict"], "方向有效")
        self.assertEqual(trend["basis"], "加分 vs 扣分")
        macd = next(row for row in rows if row["key"] == "macd")
        self.assertIsNone(macd["edge"])

    def test_one_sided_factor_compares_against_untriggered(self):
        samples = []
        for index in range(60):
            triggered = index % 2 == 0
            samples.append(
                {
                    "symbol": "600000",
                    "date": f"D{index:04d}",
                    "score": 40,
                    "parts": {"drawdown": -6 if triggered else 0},
                    "forward": {
                        "5": {"return": 0.0, "mdd": -1.0},
                        "20": {"return": -2.0 if triggered else 2.0, "mdd": -2.0},
                        "60": {"return": 0.0, "mdd": -3.0},
                    },
                }
            )
        row = next(item for item in factor_table(samples, 20) if item["key"] == "drawdown")
        self.assertEqual(row["basis"], "未触发 vs 扣分")
        self.assertAlmostEqual(row["edge"], 4.0)
        self.assertEqual(row["verdict"], "方向有效")


class QuantileTests(unittest.TestCase):
    def test_quantiles_are_ordered_by_score(self):
        samples = []
        for index in range(100):
            samples.append(
                {
                    "symbol": "600000",
                    "date": f"D{index:04d}",
                    "score": index,
                    "parts": {},
                    "forward": {
                        "5": {"return": 0.0, "mdd": -1.0},
                        "20": {"return": float(index), "mdd": -2.0},
                        "60": {"return": 0.0, "mdd": -3.0},
                    },
                }
            )
        groups = quantile_groups(samples, (5, 20, 60), 5)
        self.assertEqual([item["group"] for item in groups], ["Q1", "Q2", "Q3", "Q4", "Q5"])
        self.assertEqual(groups[0]["score_low"], 0)
        self.assertEqual(groups[-1]["score_high"], 99)
        self.assertLess(groups[0]["horizons"]["20"]["mean"], groups[-1]["horizons"]["20"]["mean"])
        self.assertEqual(sum(item["n"] for item in groups), 100)


class ConsistencyTests(unittest.TestCase):
    def test_divergence_between_pooled_and_per_symbol(self):
        rows = [
            {"ic20": -0.30, "ic60": -0.20},
            {"ic20": -0.10, "ic60": 0.10},
            {"ic20": 0.20, "ic60": 0.20},
        ]
        positive_pooled = consistency_check(rows, 0.05)
        self.assertEqual(positive_pooled["positive_ic20"], 1)
        self.assertTrue(positive_pooled["divergent"])

        negative_pooled = consistency_check(rows, -0.05)
        self.assertFalse(negative_pooled["divergent"])

    def test_per_symbol_stats_are_split_by_symbol(self):
        history_a = make_history(260, symbol="600001")
        history_b = make_history(260, drift=-0.2, symbol="600002")
        samples = walk_forward(history_a) + walk_forward(history_b)
        rows = per_symbol_stats(samples, [history_a, history_b])
        self.assertEqual([row["symbol"] for row in rows], ["600001", "600002"])
        self.assertEqual(sum(row["n"] for row in rows), len(samples))
        self.assertGreater(rows[0]["score_mean"], rows[1]["score_mean"])


class RenderTests(unittest.TestCase):
    def test_render_smoke(self):
        history = make_history(280)
        samples = walk_forward(history, stride=1)
        report = build_report([history], samples, days=280, stride=1, failures=[])
        markdown = render_markdown(report)
        html = render_html(report)
        self.assertIn("评分分桶", markdown)
        self.assertIn("未来 20 日", markdown)
        self.assertIn("评分有效性验证", html)
        self.assertIn("wf-scroll", html)
        self.assertEqual(html.count("<tr>"), html.count("</tr>"))
        # 五档分桶必须全部出现在表里，包括零样本行。
        for _key, _low, _high, label, _tone in BUCKETS:
            self.assertIn(label, html)


if __name__ == "__main__":
    unittest.main()
