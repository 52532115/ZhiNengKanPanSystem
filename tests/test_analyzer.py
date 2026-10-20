import math
import unittest

from analyzer import analyze, ema, max_drawdown, rsi, sma


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

    def test_analysis_schema_and_score(self):
        bullish = analyze(make_history(1.0))
        bearish = analyze(make_history(-1.0))
        self.assertGreater(bullish["score"], bearish["score"])
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


if __name__ == "__main__":
    unittest.main()
