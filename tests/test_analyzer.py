import math
import unittest

from analyzer import (
    DEFAULT_WEIGHTS,
    FACTOR_META,
    WEIGHTS_V1,
    WEIGHTS_V2,
    analyze,
    legacy_view,
    ema,
    max_drawdown,
    rsi,
    score_factors,
    sma,
    weight_text,
)


def make_history(direction=1.0, count=260, symbol="600000", asset_type="stock"):
    rows = []
    price = 20.0
    for i in range(count):
        wave = math.sin(i / 9) * 0.12
        price = max(2, price + direction * 0.08 + wave)
        rows.append({
            "date": f"2025-{(i // 28) % 12 + 1:02d}-{i % 28 + 1:02d}",
            "open": price - 0.05, "close": price, "high": price + 0.2, "low": price - 0.2,
            "volume": 1_000_000 + i * 1000, "amount": 20_000_000,
            "amplitude": 2.0, "pct_change": direction * 0.4, "change": direction * 0.08, "turnover": 1.2,
        })
    return {"symbol": symbol, "name": "测试标的", "asset_type": asset_type, "rows": rows}


class IndicatorTests(unittest.TestCase):
    def test_sma_and_ema(self):
        self.assertEqual(sma([1, 2, 3, 4], 3), [None, None, 2.0, 3.0])
        self.assertEqual(len(ema([1, 2, 3], 2)), 3)

    def test_rsi_uptrend(self):
        values = list(range(1, 40))
        self.assertEqual(rsi(values)[-1], 100)

    def test_drawdown(self):
        self.assertAlmostEqual(max_drawdown([10, 12, 9, 11]), -25.0)

    def test_analysis_exposes_three_fields_without_composite_score(self):
        bullish = analyze(make_history(1.0))
        bearish = analyze(make_history(-1.0))
        # 单一分数与"由分数推出的观点"已经下线，接口里不应再出现
        for key in ("score", "view", "tone", "summary", "position_guidance", "factors"):
            self.assertNotIn(key, bullish)
        self.assertEqual(set(bullish["fields"]), {"descriptor", "selection", "regime"})
        self.assertGreater(
            bullish["fields"]["descriptor"]["score"], bearish["fields"]["descriptor"]["score"]
        )
        self.assertEqual(bullish["fields"]["descriptor"]["use"], "description")
        self.assertFalse(bullish["fields"]["selection"]["available"])
        self.assertEqual(len(bullish["chart"]), 180)
        self.assertIn("volume_ma5", bullish["chart"][-1])
        self.assertIn("volume_ma10", bullish["chart"][-1])
        self.assertIn("pct_change", bullish["chart"][-1])
        self.assertIn("risk_reference", bullish["levels"])
        self.assertTrue(bullish["reasons"])

    def test_etf_analysis_uses_fund_metadata_and_precision(self):
        result = analyze(
            make_history(symbol="510300", asset_type="etf"),
            {"name": "沪深300ETF", "price": 4.123, "pct_change": 0.8, "pe_ttm": 12.3, "pb": 1.4},
        )
        self.assertEqual(result["asset_type"], "etf")
        self.assertEqual(result["asset_label"], "ETF")
        self.assertEqual(result["price_precision"], 3)
        self.assertIsNone(result["metrics"]["pe_ttm"])
        self.assertIn("底层指数估值", result["methodology"])

    def test_legacy_score_factors_are_self_consistent(self):
        history = make_history(1.0)
        score, factors = score_factors(history)
        self.assertEqual([factor["key"] for factor in factors], list(FACTOR_META))
        self.assertEqual(50 + sum(factor["delta"] for factor in factors), score)
        self.assertTrue(all(factor["label"] and factor["weight"] for factor in factors))
        self.assertEqual(legacy_view(score), legacy_view(score))

    def test_weight_profiles_are_switchable(self):
        history = make_history(1.0)
        score_v1, factors_v1 = score_factors(history, WEIGHTS_V1)
        score_v2, factors_v2 = score_factors(history, WEIGHTS_V2)
        self.assertNotEqual(score_v1, score_v2)
        self.assertEqual(score_v2, score_factors(history, DEFAULT_WEIGHTS)[0])
        self.assertEqual(score_v1, max(0, min(100, 50 + sum(f["delta"] for f in factors_v1))))
        self.assertEqual(weight_text(WEIGHTS_V2["rsi"]), "+3 / −3 / −2")
        self.assertEqual(factors_v2[0]["weight"], weight_text(WEIGHTS_V2["trend"]))
        self.assertEqual(len(factors_v1), 9)
        self.assertEqual(len(factors_v2), 9)

    def test_relative_strength_factor(self):
        strong = make_history(1.5)
        bench = make_history(-0.5, asset_type="index", symbol="000300")

        _, strong_factors = score_factors(strong, benchmark=bench)
        rs = next(factor for factor in strong_factors if factor["key"] == "rs")
        self.assertEqual(rs["delta"], 6)
        self.assertIn("跑赢", rs["reason"])

        weak = make_history(-0.5)
        _, weak_factors = score_factors(weak, benchmark=make_history(1.5, asset_type="index", symbol="000300"))
        rs_weak = next(factor for factor in weak_factors if factor["key"] == "rs")
        self.assertEqual(rs_weak["delta"], -6)
        self.assertIn("跑输", rs_weak["reason"])

        _, no_bench = score_factors(strong)
        rs_missing = next(factor for factor in no_bench if factor["key"] == "rs")
        self.assertEqual(rs_missing["delta"], 0)
        self.assertIsNone(rs_missing["reason"])


if __name__ == "__main__":
    unittest.main()
