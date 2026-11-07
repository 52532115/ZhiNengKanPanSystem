"""降换手组件测试：buffer 选择、最短持有、分批、平滑、容量与买持对照。"""

from __future__ import annotations

import math
import unittest

import rebalance as rb


def ranked(symbols: list[str], scores: list[float] | None = None) -> list[tuple[str, float]]:
    scores = scores or [float(len(symbols) - i) for i in range(len(symbols))]
    return list(zip(symbols, scores))


class BufferSelectTest(unittest.TestCase):
    def _symbols(self, n: int = 20) -> list[str]:
        return [f"s{i:02d}" for i in range(n)]

    def test_fresh_selection_buys_top_names(self) -> None:
        state = rb.BufferState()
        config = rb.BufferConfig(entry_pct=0.20, exit_pct=0.40)
        weights, stats = rb.select_with_buffer(ranked(self._symbols()), step=0, state=state,
                                               config=config)
        self.assertEqual(len(weights), 4)
        self.assertTrue(set(weights) <= set(self._symbols()[:4]))
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_held_name_inside_buffer_is_kept_even_when_rank_slips(self) -> None:
        state = rb.BufferState()
        config = rb.BufferConfig(entry_pct=0.20, exit_pct=0.50)
        symbols = self._symbols()
        rb.select_with_buffer(ranked(symbols), step=0, state=state, config=config)
        held = sorted(state.held_since)[0]
        # 让一只老仓掉到第 7 名（仍在 50% 的保留阈值之内）
        reordered = symbols[:7] + [held] + [s for s in symbols[7:] if s != held]
        weights, _ = rb.select_with_buffer(ranked(reordered), step=1, state=state, config=config)
        self.assertIn(held, weights)

    def test_name_beyond_exit_threshold_is_sold(self) -> None:
        state = rb.BufferState()
        config = rb.BufferConfig(entry_pct=0.20, exit_pct=0.30)
        symbols = self._symbols()
        rb.select_with_buffer(ranked(symbols), step=0, state=state, config=config)
        victim = symbols[0]
        reordered = [s for s in symbols if s != victim] + [victim]      # 掉到最后一名
        weights, stats = rb.select_with_buffer(ranked(reordered), step=1, state=state,
                                               config=config)
        self.assertNotIn(victim, weights)
        self.assertGreaterEqual(stats["exit"], 1)

    def test_min_holding_period_protects_recent_buys(self) -> None:
        state = rb.BufferState()
        config = rb.BufferConfig(entry_pct=0.20, exit_pct=0.30, min_hold_rebalances=3)
        symbols = self._symbols()
        rb.select_with_buffer(ranked(symbols), step=0, state=state, config=config)
        victim = symbols[0]
        # 掉到第 9 名：已超出保留阈值（30%），但仍在硬止损线（60%）之内
        reordered = symbols[1:9] + [victim] + symbols[9:]
        weights, _ = rb.select_with_buffer(ranked(reordered), step=1, state=state, config=config)
        self.assertIn(victim, weights)      # 最短持有期内，轻度掉队不卖

    def test_hard_stop_overrides_min_holding(self) -> None:
        state = rb.BufferState()
        config = rb.BufferConfig(entry_pct=0.20, exit_pct=0.30, min_hold_rebalances=5,
                                 hard_stop_pct=0.60)
        symbols = self._symbols()
        rb.select_with_buffer(ranked(symbols), step=0, state=state, config=config)
        victim = symbols[0]
        reordered = [s for s in symbols if s != victim] + [victim]
        weights, _ = rb.select_with_buffer(ranked(reordered), step=1, state=state, config=config)
        self.assertNotIn(victim, weights)   # 跌出硬止损线，最短持有期不再保护

    def test_entry_threshold_is_respected_when_slots_are_short(self) -> None:
        state = rb.BufferState()
        # 目标 5 只，但 entry 只允许前 1 只进入
        config = rb.BufferConfig(entry_pct=0.05, exit_pct=0.30, target_pct=0.25)
        weights, _ = rb.select_with_buffer(ranked(self._symbols()), step=0, state=state,
                                           config=config)
        # entry 比目标窄时会被抬到目标数，避免参数退化
        self.assertEqual(len(weights), 5)

    def test_score_weights_are_normalised(self) -> None:
        state = rb.BufferState()
        config = rb.BufferConfig(entry_pct=0.20, exit_pct=0.40, weight_mode="score")
        weights, _ = rb.select_with_buffer(ranked(self._symbols()), step=0, state=state,
                                           config=config)
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        ordered = sorted(weights.values(), reverse=True)
        self.assertGreater(ordered[0], ordered[-1])

    def test_inv_vol_weights_favour_low_vol_names(self) -> None:
        state = rb.BufferState()
        config = rb.BufferConfig(entry_pct=0.20, exit_pct=0.40, weight_mode="inv_vol")
        symbols = self._symbols()
        vols = {symbol: (0.1 if symbol == symbols[0] else 0.5) for symbol in symbols}
        weights, _ = rb.select_with_buffer(ranked(symbols), step=0, state=state,
                                           config=config, volatilities=vols)
        self.assertGreater(weights[symbols[0]], weights[symbols[1]])

    def test_empty_universe_clears_state(self) -> None:
        state = rb.BufferState()
        config = rb.BufferConfig()
        rb.select_with_buffer(ranked(self._symbols()), step=0, state=state, config=config)
        weights, stats = rb.select_with_buffer([], step=1, state=state, config=config)
        self.assertEqual(weights, {})
        self.assertEqual(stats["n"], 0)
        self.assertEqual(state.held_since, {})


class SmootherTest(unittest.TestCase):
    def test_zero_halflife_is_identity(self) -> None:
        smoother = rb.RankSmoother(0)
        first = [(f"s{i}", 10 - i) for i in range(5)]
        out = smoother.update(first)
        self.assertEqual([symbol for symbol, _ in out], [symbol for symbol, _ in first])

    def test_smoothing_damps_a_single_period_jump(self) -> None:
        smooth = rb.RankSmoother(10)
        base = [(f"s{i}", 10 - i) for i in range(5)]
        smooth.update(base)
        flipped = list(reversed(base))
        out = dict(smooth.update(flipped))
        # 一期反转后，平滑值只走到一半左右：既没有停在 1.0，也没有瞬间归零
        self.assertGreater(out["s0"], 0.0)
        self.assertLess(out["s0"], 0.5)

    def test_halflife_conversion(self) -> None:
        self.assertAlmostEqual(rb.protocol_rebalances(20, 5), 4.0)


class MetricsTest(unittest.TestCase):
    def test_jaccard(self) -> None:
        self.assertAlmostEqual(rb.jaccard([["a", "b"], ["a", "b"]]), 1.0)
        self.assertAlmostEqual(rb.jaccard([["a", "b"], ["c", "d"]]), 0.0)
        self.assertAlmostEqual(rb.jaccard([["a", "b"], ["b", "c"]]), 1 / 3)

    def test_net_sharpe_of_constant_return_is_large(self) -> None:
        nav = [{"date": f"d{i}", "nav": 1.0 * (1.001 ** i)} for i in range(120)]
        sharpe = rb.net_sharpe(nav)
        self.assertIsNotNone(sharpe)
        self.assertGreater(sharpe, 10)

    def test_net_sharpe_needs_enough_points(self) -> None:
        self.assertIsNone(rb.net_sharpe([{"date": "d1", "nav": 1.0}]))

    def test_annual_turnover_and_cost_drag(self) -> None:
        rebalances = [{"one_way_turnover": 0.1, "cost_bps": 20.0, "capped": False}] * 10
        self.assertAlmostEqual(rb.annual_turnover(rebalances, years=1.0), 100.0)
        self.assertAlmostEqual(rb.cost_drag_bps(rebalances, years=1.0), 200.0)

    def test_cap_trigger_rate(self) -> None:
        rebalances = [{"capped": True}, {"capped": False}, {"capped": False}, {"capped": False}]
        self.assertAlmostEqual(rb.cap_trigger_rate(rebalances), 25.0)

    def test_summary_bundles_metrics(self) -> None:
        nav = [{"date": f"d{i}", "nav": 1.0 + i * 0.001} for i in range(60)]
        result = {
            "nav": nav,
            "rebalances": [{"one_way_turnover": 0.2, "cost_bps": 10.0, "capped": False}],
            "holdings": [{"symbols": ["a", "b"]}, {"symbols": ["a", "c"]}],
            "total_cost": 0.01,
            "cash_ratio_mean": 0.0,
        }
        summary = rb.summary(result, years=1.0)
        self.assertIn("sharpe", summary)
        self.assertAlmostEqual(summary["annual_turnover"], 20.0)
        self.assertAlmostEqual(summary["jaccard"], 1 / 3, places=3)


class CapacityTest(unittest.TestCase):
    def test_impact_grows_with_participation(self) -> None:
        self.assertAlmostEqual(rb.impact_cost_bp(0.01), 10.0, places=6)
        self.assertGreater(rb.impact_cost_bp(0.10), rb.impact_cost_bp(0.01))
        self.assertEqual(rb.impact_cost_bp(0.0), 0.0)

    def test_capacity_scales_with_adv(self) -> None:
        trades = [{"symbol": "a", "fraction": 0.05, "date": "d1"}] * 10
        adv = {"a": 1e8}
        result = rb.capacity(trades, adv, participation=0.10)
        self.assertTrue(result["available"])
        self.assertAlmostEqual(result["aum_binding"], 0.10 * 1e8 / 0.05 / 1e8, places=2)

    def test_capacity_without_adv(self) -> None:
        self.assertFalse(rb.capacity([{"symbol": "x", "fraction": 0.1, "date": "d"}], {})["available"])


class BuyAndHoldTest(unittest.TestCase):
    def _bars(self) -> dict:
        rows = {}
        for index, price_series in enumerate(([10, 11, 12, 13], [20, 20, 20, 22])):
            symbol = f"s{index}"
            rows[symbol] = {
                f"d{i}": {"open": price, "close": price, "amount": 1e7, "turnover": 1.0}
                for i, price in enumerate(price_series)
            }
        return rows

    def test_no_rebalance_keeps_initial_weights(self) -> None:
        result = rb.buy_and_hold(["s0", "s1"], self._bars(), ["d0", "d1", "d2", "d3"],
                                 entry_index=0, cost=0.0)
        # 期末 d3：s0 13 元、s1 22 元，等权 → (13/10 + 22/20)/2 - 1 = 20%
        self.assertAlmostEqual(result["total_return"], ((13 / 10 + 22 / 20) / 2 - 1) * 100, places=2)

    def test_entry_cost_reduces_return(self) -> None:
        clean = rb.buy_and_hold(["s0"], self._bars(), ["d0", "d1", "d2"], entry_index=0, cost=0.0)
        costly = rb.buy_and_hold(["s0"], self._bars(), ["d0", "d1", "d2"], entry_index=0, cost=0.01)
        self.assertLess(costly["total_return"], clean["total_return"])

    def test_empty_symbols(self) -> None:
        self.assertIsNone(rb.buy_and_hold([], self._bars(), ["d0", "d1"], entry_index=0)["total_return"])


class StickyScoreTest(unittest.TestCase):
    def test_held_names_use_entry_score(self) -> None:
        state = rb.BufferState()
        config = rb.BufferConfig(entry_pct=0.20, exit_pct=0.40)
        symbols = [f"s{i}" for i in range(10)]
        rb.select_with_buffer(ranked(symbols), step=0, state=state, config=config)
        held = sorted(state.held_since)[0]
        entry_score = state.entry_score[held]
        new_ranking = [(symbol, 100.0 - i) for i, symbol in enumerate(symbols)]
        sticky = dict(rb.sticky_scores(new_ranking, state, step=1))
        self.assertAlmostEqual(sticky[held], entry_score)


if __name__ == "__main__":
    unittest.main()
