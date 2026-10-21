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

    def test_bse_920_codes_route_to_market_zero(self):
        # 920 段若按首位 9 处理会落到 1.920002，上游返回空行情。
        self.assertEqual(security_id("920002"), "0.920002")

    def test_bse_legacy_codes_route_to_market_zero(self):
        self.assertEqual(security_id("832566"), "0.832566")
        self.assertEqual(security_id("430047"), "0.430047")

    def test_shanghai_b_share_keeps_shanghai_routing(self):
        self.assertEqual(security_id("900901"), "1.900901")

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

    def test_bse_fallback_uses_bj_prefix(self):
        client = MarketDataClient()
        captured = {}
        lines = []
        for index in range(80):
            close = 40 + index / 10
            lines.append([f"2026-01-{index + 1:02d}", str(close - 0.05), str(close), str(close + 0.1), str(close - 0.1), "1000"])

        def fake(_url, params):
            captured.update(params)
            key = params["param"].split(",")[0]
            return {"data": {key: {"qfqday": lines, "qt": {key: ["1", "万达轴承"]}}}}

        client._get_json = fake
        result = client._fallback_history("920002", 80, market="0")
        self.assertTrue(captured["param"].startswith("bj920002"))
        self.assertEqual(len(result["rows"]), 80)


class QuotePrecisionTests(unittest.TestCase):
    """f59 是上游返回的小数位数：股票 2 位、场内基金 3 位。"""

    def quote_with(self, symbol, payload):
        client = MarketDataClient()
        client._get_json = lambda *_args, **_kwargs: {"data": payload}
        return client.quote(symbol)

    def test_etf_quote_uses_three_decimal_precision(self):
        quote = self.quote_with(
            "510300",
            {
                "f57": "510300",
                "f58": "沪深300ETF",
                "f59": 3,
                "f43": 4204,
                "f44": 4237,
                "f45": 4204,
                "f46": 4234,
                "f60": 4220,
                "f162": 0,
                "f167": 0,
                "f170": -38,
                "f171": 78,
            },
        )
        self.assertEqual(quote["price"], 4.204)
        self.assertEqual(quote["high"], 4.237)
        self.assertEqual(quote["low"], 4.204)
        self.assertEqual(quote["open"], 4.234)
        self.assertEqual(quote["previous_close"], 4.22)
        self.assertEqual(quote["pct_change"], -0.38)
        self.assertEqual(quote["amplitude"], 0.78)

    def test_stock_quote_keeps_two_decimal_precision(self):
        quote = self.quote_with(
            "600519",
            {
                "f57": "600519",
                "f58": "贵州茅台",
                "f59": 2,
                "f43": 131430,
                "f44": 132500,
                "f45": 130000,
                "f46": 131000,
                "f60": 133188,
                "f162": 1845,
                "f167": 654,
                "f170": -132,
                "f171": 250,
            },
        )
        self.assertEqual(quote["price"], 1314.3)
        self.assertEqual(quote["high"], 1325.0)
        self.assertEqual(quote["low"], 1300.0)
        self.assertEqual(quote["open"], 1310.0)
        self.assertEqual(quote["previous_close"], 1331.88)
        self.assertEqual(quote["pe_ttm"], 18.45)
        self.assertEqual(quote["pb"], 6.54)

    def test_missing_precision_falls_back_to_two_digits(self):
        quote = self.quote_with("600519", {"f57": "600519", "f43": 10000, "f60": 9900})
        self.assertEqual(quote["price"], 100.0)
        self.assertEqual(quote["previous_close"], 99.0)


if __name__ == "__main__":
    unittest.main()
