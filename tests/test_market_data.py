import unittest

from market_data import MarketDataClient, instrument_type, normalize_symbol, security_id


class SymbolTests(unittest.TestCase):
    def test_etf_codes_are_supported(self):
        self.assertEqual(normalize_symbol("SH510300"), "510300")
        self.assertEqual(normalize_symbol("sz159915"), "159915")
        self.assertEqual(instrument_type("510300"), "etf")
        self.assertEqual(instrument_type("159915"), "etf")

    def test_exchange_routing(self):
        self.assertEqual(security_id("510300"), "1.510300")
        self.assertEqual(security_id("159915"), "0.159915")

    def test_000001_routes_to_stock_outside_market_endpoint(self):
        self.assertEqual(security_id("000001"), "0.000001")
        self.assertEqual(instrument_type("000001"), "stock")

    def test_invalid_symbol(self):
        with self.assertRaises(ValueError):
            normalize_symbol("ETF300")

    def test_fallback_history_parser(self):
        client = MarketDataClient()
        lines = []
        for index in range(80):
            close = 4 + index / 100
            lines.append([f"2026-01-{index + 1:02d}", str(close - 0.01), str(close), str(close + 0.02), str(close - 0.02), "10000"])
        client._get_json = lambda *_args, **_kwargs: {
            "data": {"sh510300": {"qfqday": lines, "qt": {"sh510300": ["1", "沪深300ETF"]}}}
        }
        result = client._fallback_history("510300", 80, index=False)
        self.assertEqual(result["name"], "沪深300ETF")
        self.assertEqual(result["data_source"], "fallback")
        self.assertEqual(len(result["rows"]), 80)
        self.assertGreater(result["rows"][-1]["pct_change"], 0)


if __name__ == "__main__":
    unittest.main()
