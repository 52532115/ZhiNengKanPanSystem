import math
import unittest

from analyzer import relative_strength
from market_data import MarketDataClient


def make_rows(closes, start_day=1):
    rows = []
    for index, close in enumerate(closes):
        rows.append(
            {
                "date": f"2026-{start_day + index // 28:02d}-{index % 28 + 1:02d}",
                "open": close - 0.05,
                "close": close,
                "high": close + 0.1,
                "low": close - 0.1,
                "volume": 1_000_000,
                "amount": 10_000_000,
                "amplitude": 1.0,
                "pct_change": 0.1,
                "change": 0.1,
                "turnover": 1.0,
            }
        )
    return rows


class RelativeStrengthTests(unittest.TestCase):
    def build(self, stock_curve, bench_curve):
        stock = {"symbol": "600519", "name": "贵州茅台", "asset_type": "stock", "rows": make_rows(stock_curve)}
        bench = {"symbol": "000300", "name": "沪深300", "asset_type": "index", "rows": make_rows(bench_curve)}
        return relative_strength(stock, bench, 40)

    def test_outperformer_is_real_strength(self):
        stock = [100 * (1.01 ** i) for i in range(60)]
        bench = [4000 * (1.001 ** i) for i in range(60)]
        result = self.build(stock, bench)
        self.assertEqual(result["verdict"], "真强")
        self.assertEqual(result["tone"], "positive")
        self.assertGreater(result["excess_return"], 5)
        self.assertGreater(result["rs"], 100)
        self.assertEqual(result["series"][0]["rs"], 100.0)
        self.assertEqual(result["window"], 40)

    def test_synchronized_rally_is_flagged(self):
        stock = [100 * (1.005 ** i) for i in range(60)]
        bench = [4000 * (1.005 ** i) for i in range(60)]
        result = self.build(stock, bench)
        self.assertEqual(result["verdict"], "同步")
        self.assertEqual(result["tone"], "neutral")
        self.assertAlmostEqual(result["rs"], 100.0, places=2)
        self.assertIn("普涨", result["note"])

    def test_laggard(self):
        stock = [100 * (0.998 ** i) for i in range(60)]
        bench = [4000 * (1.004 ** i) for i in range(60)]
        result = self.build(stock, bench)
        self.assertEqual(result["verdict"], "跑输基准")
        self.assertEqual(result["rs_trend"], "down")
        self.assertLess(result["rs"], 100)

    def test_insufficient_overlap(self):
        stock = {"symbol": "600519", "name": "贵州茅台", "rows": make_rows([100] * 10)}
        bench = {"symbol": "000300", "name": "沪深300", "rows": make_rows([4000] * 10)}
        with self.assertRaises(ValueError):
            relative_strength(stock, bench, 40)


class SearchFilterTests(unittest.TestCase):
    def test_only_a_share_and_fund_are_returned(self):
        client = MarketDataClient()
        client._get_json = lambda *_args, **_kwargs: {
            "QuotationCodeTable": {
                "Data": [
                    {"Code": "600519", "Name": "贵州茅台", "Classify": "AStock", "SecurityTypeName": "沪A"},
                    {"Code": "000300", "Name": "沪深300", "Classify": "Index", "SecurityTypeName": "指数"},
                    {"Code": "03147", "Name": "X南方中创业", "Classify": "HK", "SecurityTypeName": "港股"},
                    {"Code": "510300", "Name": "沪深300ETF华泰柏瑞", "Classify": "Fund", "SecurityTypeName": "基金"},
                    {"Code": "600519", "Name": "贵州茅台", "Classify": "AStock", "SecurityTypeName": "沪A"},
                ]
            }
        }
        items = client.search("茅台", 8)
        self.assertEqual([item["symbol"] for item in items], ["600519", "510300"])
        self.assertEqual(items[1]["asset_type"], "etf")

    def test_upstream_failure_returns_empty(self):
        client = MarketDataClient()
        from market_data import MarketDataError

        def boom(*_args, **_kwargs):
            raise MarketDataError("down")

        client._get_json = boom
        self.assertEqual(client.search("茅台"), [])


class BenchmarkFallbackTests(unittest.TestCase):
    def test_index_fallback_uses_shanghai_prefix(self):
        client = MarketDataClient()
        captured = {}

        def fake(url, params):
            captured.update(params)
            key = params["param"].split(",")[0]
            lines = []
            for index in range(80):
                close = 4000 + index
                lines.append([f"2026-01-{index + 1:02d}", str(close - 1), str(close), str(close + 2), str(close - 2), "1000"])
            return {"data": {key: {"qfqday": lines, "qt": {key: ["1", "沪深300"]}}}}

        client._get_json = fake
        result = client._fallback_history("000300", 80, index=True, market="1")
        self.assertTrue(captured["param"].startswith("sh000300"))
        self.assertEqual(result["name"], "沪深300")


if __name__ == "__main__":
    unittest.main()
