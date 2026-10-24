"""三字段口径的契约测试：分位语义、档位切点、跨口径拒绝。"""

import math
import unittest

import fields


def make_history(direction=1.0, count=420, symbol="600000", asset_type="stock", wave=0.12):
    rows = []
    price = 20.0
    for i in range(count):
        price = max(2.0, price + direction * 0.06 + math.sin(i / 9) * wave)
        rows.append({
            "date": f"2025-{(i // 28) % 12 + 1:02d}-{i % 28 + 1:02d}",
            "open": price - 0.05, "close": price, "high": price + 0.2, "low": price - 0.2,
            "volume": 1_000_000 + i * 1000, "amount": 20_000_000,
            "amplitude": 2.0, "pct_change": direction * 0.3, "change": direction * 0.06,
            "turnover": 1.2,
        })
    return {"symbol": symbol, "name": "测试标的", "asset_type": asset_type, "rows": rows}


class FieldContractTests(unittest.TestCase):
    def test_cross_usage_is_rejected(self):
        for field, bad_purpose in (
            (fields.DESCRIPTOR, "timing"),
            (fields.DESCRIPTOR, "selection"),
            (fields.SELECTION, "description"),
            (fields.REGIME, "selection"),
        ):
            with self.assertRaises(ValueError):
                fields.assert_use(field, bad_purpose)
        fields.assert_use(fields.DESCRIPTOR, "description")
        fields.assert_use(fields.SELECTION, "selection")
        fields.assert_use(fields.REGIME, "timing")

    def test_unknown_field_is_rejected(self):
        with self.assertRaises(ValueError):
            fields.assert_use("score", "description")

    def test_every_field_declares_validation_and_usage(self):
        for key, spec in fields.FIELD_SPECS.items():
            self.assertTrue(spec["validation"])
            self.assertTrue(spec["note"])
            self.assertIn(spec["use"], ("description", "selection", "timing"))
            self.assertNotIn(spec["use"], spec["not_for"])
        self.assertTrue(fields.usage_note(fields.DESCRIPTOR).startswith("趋势状态"))


class PercentileTests(unittest.TestCase):
    def test_percentile_is_monotone_in_score(self):
        table = fields.DEFAULT_CALIBRATION
        for asset_class in ("stock", "etf"):
            values = [
                fields.percentile_of_score(score, asset_class, table)
                for score in range(0, 101, 5)
            ]
            self.assertEqual(values, sorted(values), f"{asset_class} 分位映射不单调")
            self.assertEqual(values[0], 0.0)
            self.assertEqual(values[-1], 100.0)

    def test_band_cutoffs_are_percentiles(self):
        self.assertEqual(fields.band_of(95)["label"], "强势")
        self.assertEqual(fields.band_of(80)["label"], "强势")
        self.assertEqual(fields.band_of(79.9)["label"], "偏强")
        self.assertEqual(fields.band_of(50)["label"], "中性")
        self.assertEqual(fields.band_of(20)["label"], "偏弱")
        self.assertEqual(fields.band_of(19.9)["label"], "弱势")
        self.assertEqual(fields.band_of(0)["label"], "弱势")

    def test_pool_percentiles_are_bounded_and_tie_safe(self):
        ranks = fields.pool_percentiles({"a": 10, "b": 20, "c": 20, "d": 30})
        self.assertEqual(set(ranks), {"a", "b", "c", "d"})
        self.assertEqual(ranks["b"], ranks["c"])          # 同分同名次
        self.assertLess(ranks["a"], ranks["b"])
        self.assertLess(ranks["c"], ranks["d"])
        for value in ranks.values():
            self.assertGreater(value, 0)
            self.assertLessEqual(value, 100)
        self.assertEqual(fields.pool_percentiles({"a": 1}), {})
        self.assertEqual(fields.pool_percentiles({}), {})

    def test_calibration_is_asset_specific(self):
        table = fields.calibrate({
            "stock": [float(v) for v in range(0, 1000)],
            "etf": [float(v) for v in range(100, 400)],
        })
        self.assertEqual(len(table["stock"]["score_knots"]), 21)
        self.assertNotEqual(table["stock"]["score_knots"], table["etf"]["score_knots"])
        self.assertEqual(table["etf"]["band_cuts"], list(fields.BAND_CUTS))


class DescriptorTests(unittest.TestCase):
    def test_uptrend_scores_above_downtrend(self):
        up = fields.describe(make_history(1.0), asset_type="stock")
        down = fields.describe(make_history(-1.0), asset_type="stock")
        self.assertTrue(up["available"])
        self.assertGreater(up["score"], down["score"])

    def test_descriptor_outputs_percentiles_not_verdicts(self):
        result = fields.describe(make_history(1.0), asset_type="stock")
        for key in ("view", "tone", "position", "action"):
            self.assertNotIn(key, result)
        self.assertGreaterEqual(result["percentile"]["value"], 0)
        self.assertLessEqual(result["percentile"]["value"], 100)
        self.assertIn(result["band"]["label"], [label for _cut, label, _tone, _meaning in fields.BAND_LABELS])
        self.assertEqual(result["usage"], fields.usage_note(fields.DESCRIPTOR))
        self.assertNotIn("买入", result["headline"])
        self.assertNotIn("看多", result["headline"])

    def test_short_history_is_unavailable_not_guessed(self):
        result = fields.describe(make_history(1.0, count=120), asset_type="stock")
        self.assertFalse(result["available"])
        self.assertIn("不足", result["reason"])

    def test_self_percentile_uses_recent_window(self):
        result = fields.describe(make_history(1.0, count=420), asset_type="stock")
        self.assertLessEqual(result["self_percentile"]["window"], fields.SELF_WINDOW)
        self.assertGreaterEqual(result["self_percentile"]["value"], 0)

    def test_etf_and_stock_calibrate_separately(self):
        history = make_history(1.0, asset_type="etf", symbol="510300")
        stock = fields.describe(history, asset_type="stock")
        etf = fields.describe(history, asset_type="etf")
        self.assertEqual(stock["asset_class"], "stock")
        self.assertEqual(etf["asset_class"], "etf")
        self.assertNotEqual(stock["percentile"]["label"], etf["percentile"]["label"])

    def test_series_is_causal(self):
        history = make_history(1.0, count=420)
        full = [item["score"] if item else None for item in fields.descriptor_series(history)]
        prefix = {**history, "rows": history["rows"][:-20]}
        short = [item["score"] if item else None for item in fields.descriptor_series(prefix)]
        self.assertEqual(full[: len(short)], short)


class SelectionAndRegimeTests(unittest.TestCase):
    def test_selection_field_requires_pool(self):
        missing = fields.selection_field("600000", None)
        self.assertFalse(missing["available"])
        present = fields.selection_field("600000", {"600000": 75.0, "000001": 25.0}, pool_size=2, as_of="2026-01-05")
        self.assertTrue(present["available"])
        self.assertEqual(present["percentile"], 75.0)
        self.assertEqual(present["evidence"]["status"], "untested")

    def test_regime_states_have_exposure_caps(self):
        for state, spec in fields.REGIME_STATES.items():
            self.assertGreater(spec["exposure"], 0)
            self.assertLessEqual(spec["exposure"], 1.0)
        self.assertGreater(
            fields.REGIME_STATES["risk_on"]["exposure"], fields.REGIME_STATES["risk_off"]["exposure"]
        )

    def test_regime_uses_zscore_not_absolute_score(self):
        history = make_history(1.0, count=900, asset_type="index", symbol="000300")
        result = fields.regime(history)
        self.assertTrue(result["available"])
        self.assertIn(result["state"], fields.REGIME_STATES)
        self.assertAlmostEqual(result["cut"], fields.REGIME_Z_CUT)
        self.assertLessEqual(result["exposure"], 1.0)
        series = fields.regime_series(history)
        self.assertEqual(len(series), len(history["rows"]))
        states = [point["state"] for point in series if point]
        self.assertTrue(states)


if __name__ == "__main__":
    unittest.main()
