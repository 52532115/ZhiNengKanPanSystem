"""口径一致性回归：文案与实现、校准类别、档位切点、0 分位排序。"""

from __future__ import annotations

import datetime
import math
import unittest
from pathlib import Path

import analyzer
import fields as fd

ROOT = Path(__file__).resolve().parent.parent


def rising_history(count: int = 320, *, start: str = "2022-01-03", wave: float = 0.004) -> dict:
    rows = []
    day = datetime.date.fromisoformat(start)
    price = 1.0
    for index in range(count):
        price = max(0.1, price * (1 + 0.0006 + wave * math.sin(index / 13)))
        rows.append({
            "date": day.isoformat(),
            "open": price * 0.999, "high": price * 1.004, "low": price * 0.996,
            "close": price, "volume": 1_000_000, "amount": price * 1_000_000,
            "pct_change": 0.0, "turnover": 1.0, "change": 0.0,
        })
        day += datetime.timedelta(days=1)
        while day.weekday() >= 5:
            day += datetime.timedelta(days=1)
    return {"symbol": "TEST", "name": "测试", "rows": rows}


class AssetClassTest(unittest.TestCase):
    def test_index_is_not_etf(self):
        self.assertEqual(fd.asset_class_of("index"), "index")
        self.assertEqual(fd.asset_class_of("etf"), "etf")
        self.assertEqual(fd.asset_class_of("fund"), "etf")
        self.assertEqual(fd.asset_class_of("stock"), "stock")
        self.assertEqual(fd.asset_class_of(None), "stock")

    def test_index_calibration_is_explicitly_absent(self):
        self.assertIn("index", fd.DEFAULT_CALIBRATION)
        self.assertFalse(fd.is_calibrated("index"))
        self.assertTrue(fd.is_calibrated("stock"))
        self.assertIsNone(fd.percentile_of_score(60.0, "index"))
        self.assertIsNone(fd.percentile_of_score(60.0, "index", {"index": {"score_knots": []}}))

    def test_calibrate_keeps_index_uncalibrated_source(self):
        table = fd.calibrate({"stock": [50.0] * 600, "etf": [52.0] * 600})
        self.assertEqual(table["index"]["score_knots"], [])
        self.assertIn("不做同类校准", table["index"]["source"])
        self.assertGreaterEqual(len(table["stock"]["score_knots"]), 21)

    def test_index_descriptor_skips_class_percentile(self):
        field = fd.describe(rising_history(), asset_type="index")
        self.assertTrue(field["available"])
        self.assertEqual(field["asset_class"], "index")
        self.assertEqual(field["asset_class_label"], "指数")
        percentile = field["percentile"]
        self.assertIsNone(percentile["value"])
        self.assertFalse(percentile["available"])
        self.assertNotIn("ETF", percentile["label"])          # 不再冒充 ETF 同类分位
        self.assertIn("不做同类分位", percentile["label"])
        self.assertTrue(percentile["reason"])
        # 档位退回"相对自身"，并且明确标注依据
        self.assertEqual(field["band"]["basis"], "self")
        self.assertIsNotNone(field["self_percentile"]["value"])
        self.assertIn("指数无同类校准", field["headline"])

    def test_etf_descriptor_keeps_class_percentile(self):
        field = fd.describe(rising_history(), asset_type="etf")
        self.assertEqual(field["asset_class"], "etf")
        self.assertTrue(field["percentile"]["available"])
        self.assertIsNotNone(field["percentile"]["value"])
        self.assertIn("ETF", field["percentile"]["label"])
        self.assertEqual(field["band"]["basis"], "asset_class")


class BandCutsTest(unittest.TestCase):
    def test_default_cuts(self):
        self.assertEqual(fd.band_of(85)["label"], "强势")
        self.assertEqual(fd.band_of(65)["label"], "偏强")
        self.assertEqual(fd.band_of(45)["label"], "中性")
        self.assertEqual(fd.band_of(25)["label"], "偏弱")
        self.assertEqual(fd.band_of(5)["label"], "弱势")

    def test_custom_cuts_actually_apply(self):
        # 同样的 55 分位：默认切点是中性（≥40），(70,50,30) 下变成偏强（≥50）
        default = fd.band_of(55)
        custom = fd.band_of(55, (70, 50, 30))
        self.assertEqual(default["label"], "中性")
        self.assertEqual(default["cut"], 40)
        self.assertEqual(custom["label"], "偏强")
        self.assertEqual(custom["cut"], 50)
        self.assertEqual(fd.band_of(75, (70, 50, 30))["label"], "强势")
        self.assertEqual(fd.band_of(75, (70, 50, 30))["cut"], 70)
        # 3 个切点 = 4 档：低于最低切点的那一档按位置取第 4 个标签（偏弱），
        # 「弱势」需要 4 个切点（默认），这是"n 个切点切出 n+1 档"的直接结果
        self.assertEqual(fd.band_of(10, (70, 50, 30))["label"], "偏弱")
        self.assertEqual(fd.band_of(10, (70, 50, 30))["cut"], 0)
        self.assertEqual(fd.band_of(10, (80, 60, 40, 20))["label"], "弱势")

    def test_cuts_are_order_insensitive(self):
        self.assertEqual(fd.band_of(55, (30, 70, 50)), fd.band_of(55, (70, 50, 30)))

    def test_describe_honours_table_cuts(self):
        calibration = {
            "stock": {
                "score_knots": fd.DEFAULT_CALIBRATION["stock"]["score_knots"],
                "band_cuts": [70, 50, 30],
                "n": 500,
                "source": "测试",
            }
        }
        field = fd.describe(rising_history(), asset_type="stock", calibration=calibration)
        self.assertIn(field["band"]["cut"], {70, 50, 30, 0})
        self.assertNotIn(field["band"]["cut"], {80, 60, 40, 20})


class ZeroPercentileSortTest(unittest.TestCase):
    def test_zero_percentile_is_not_missing(self):
        import app

        ranked = [
            {"alert": False, "selection": {"percentile": 90}},
            {"alert": False, "selection": {"percentile": 0}},
            {"alert": False, "selection": {"percentile": None}},
            {"alert": True, "selection": {"percentile": 10}},
        ]
        ordered = sorted(ranked, key=app.batch_rank)
        percentiles = [item["selection"]["percentile"] for item in ordered]
        # 异动置顶；其余 90 → 0 → None；0 分位排在"没有分位"之前
        self.assertEqual(percentiles, [10, 90, 0, None])

    def test_old_bug_would_have_put_zero_first(self):
        # 记录旧写法的问题：0 与 None 都变成 -1，取负后 1，排在 90 分位（-90）之前
        old_key = lambda item: (not item["alert"], -((item["selection"]["percentile"]) or -1))
        legacy = sorted(
            [{"alert": False, "selection": {"percentile": 90}},
             {"alert": False, "selection": {"percentile": 0}}],
            key=old_key,
        )
        self.assertEqual(legacy[0]["selection"]["percentile"], 90)
        self.assertEqual(legacy[1]["selection"]["percentile"], 0)
        # 而新写法在 0 与 None 之间不再混淆
        import app

        self.assertLess(app.batch_rank({"alert": False, "selection": {"percentile": 0}}),
                        app.batch_rank({"alert": False, "selection": {"percentile": None}}))


class MethodologyConsistencyTest(unittest.TestCase):
    def test_weights_have_no_momentum(self):
        self.assertNotIn("mom", fd.DESCRIPTOR_WEIGHTS)
        self.assertEqual(set(fd.DESCRIPTOR_WEIGHTS), set(fd.DESCRIPTOR_LABELS))
        self.assertAlmostEqual(sum(fd.DESCRIPTOR_WEIGHTS.values()), 1.0, places=6)

    def test_methodology_text_lists_every_weight(self):
        text = analyzer.descriptor_methodology()
        for key in fd.DESCRIPTOR_WEIGHTS:
            self.assertIn(fd.DESCRIPTOR_LABELS[key], text, f"文案缺少分量 {key}")
        self.assertIn(f"由 {len(fd.DESCRIPTOR_WEIGHTS)} 个分量", text)
        # 必须写明"不含动量"以及为什么，而不是悄悄不提
        self.assertIn("不含 12–1 动量", text)
        self.assertIn("抵消", text)

    def test_etf_methodology_mentions_calibration_scope(self):
        text = analyzer.descriptor_methodology(is_fund=True)
        self.assertIn("ETF", text)
        self.assertIn("不等同于底层指数估值", text)

    def test_component_reasons_has_no_momentum_branch(self):
        history = rising_history()
        descriptor = {
            "available": True,
            "components": [
                {"key": "mom", "label": "12-1 动量", "score": 90},
                {"key": "trend", "label": "价格 vs MA20", "score": 90},
            ],
        }
        reasons = analyzer._component_reasons(descriptor, history)
        joined = "".join(item["text"] for item in reasons)
        self.assertNotIn("动量", joined)          # 死分支已删除
        self.assertIn("20 日均线", joined)         # 真分支仍在

    def test_analysis_methodology_field_uses_same_text(self):
        history = rising_history()
        result = analyzer.analyze(history, quote=None)
        self.assertNotIn("12–1 动量、", result["methodology"])
        self.assertEqual(result["methodology"], analyzer.descriptor_methodology(is_fund=False))


class ReadmeConsistencyTest(unittest.TestCase):
    def test_feature_list_does_not_claim_rs_scoring(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("并把 20 日超额收益作为评分因子（v3）", text)
        self.assertIn("RS 只用于展示与对照，不进入任何评分", text)
        self.assertIn("历史章节", text)


if __name__ == "__main__":
    unittest.main()
