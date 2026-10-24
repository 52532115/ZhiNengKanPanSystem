"""统计口径层的测试：区间、功效、多重检验、记账。"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import evalkit as ek


class DistributionTest(unittest.TestCase):
    def test_norm_ppf_matches_known_quantiles(self) -> None:
        self.assertAlmostEqual(ek.norm_ppf(0.975), 1.959964, places=5)
        self.assertAlmostEqual(ek.norm_ppf(0.5), 0.0, places=6)
        self.assertAlmostEqual(ek.norm_ppf(0.8), 0.841621, places=5)
        self.assertAlmostEqual(ek.norm_ppf(0.001), -3.090232, places=4)

    def test_norm_cdf_inverts_norm_ppf(self) -> None:
        for probability in (0.01, 0.2, 0.5, 0.73, 0.99):
            self.assertAlmostEqual(ek.norm_cdf(ek.norm_ppf(probability)), probability, places=6)

    def test_norm_ppf_rejects_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            ek.norm_ppf(0.0)
        with self.assertRaises(ValueError):
            ek.norm_ppf(1.0)


class RankTest(unittest.TestCase):
    def test_rank_average_handles_ties(self) -> None:
        self.assertEqual(ek.rank_average([10, 20, 20, 30]), [1.0, 2.5, 2.5, 4.0])

    def test_spearman_perfect_and_inverse(self) -> None:
        self.assertAlmostEqual(ek.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertAlmostEqual(ek.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertIsNone(ek.spearman([1, 1, 1, 1], [1, 2, 3, 4]))

    def test_spearman_is_rank_based(self) -> None:
        # 单调变换不改变秩相关
        self.assertAlmostEqual(
            ek.spearman([1, 2, 3, 4], [1, 8, 27, 64]),
            ek.spearman([1, 2, 3, 4], [1, 2, 3, 4]),
        )


class BootstrapTest(unittest.TestCase):
    def test_constant_series_has_zero_width_and_no_pvalue(self) -> None:
        stat = ek.block_bootstrap_mean([0.05] * 40, block=20, iters=200)
        self.assertAlmostEqual(stat["ci_low"], 0.05, places=9)
        self.assertAlmostEqual(stat["ci_high"], 0.05, places=9)
        # 退化分布下自助法无法评估抽样误差，必须拒绝给出显著性
        self.assertIsNone(stat["p"])
        self.assertFalse(stat["significant"])

    def test_strong_signal_excludes_zero(self) -> None:
        series = [0.2 + 0.05 * math.sin(i * 1.7) + 0.03 * math.cos(i * 0.4) for i in range(120)]
        stat = ek.block_bootstrap_mean(series, block=20, iters=400)
        self.assertGreater(stat["ci_low"], 0.0)
        self.assertLess(stat["p"], 0.01)

    def test_zero_mean_noise_usually_covers_zero(self) -> None:
        covered = 0
        for seed in range(12):
            series = [
                0.15 * math.sin(i * 0.7 + seed) + 0.05 * math.cos(i * 1.3 + seed)
                for i in range(80)
            ]
            stat = ek.block_bootstrap_mean(series, block=20, iters=200, seed=seed)
            covered += int(stat["covers_zero"]) if stat.get("p") is not None else 0
        self.assertGreaterEqual(covered, 7)

    def test_ic_ci_flags_significance_consistently(self) -> None:
        series = [0.06 + 0.03 * math.sin(i * 1.3) for i in range(100)]
        stat = ek.ic_ci(series, horizon=20, iters=400)
        self.assertTrue(stat["significant"])
        self.assertFalse(stat["covers_zero"])
        self.assertEqual(stat["n_blocks"], 5.0)

    def test_short_series_returns_empty_stat(self) -> None:
        stat = ek.ic_ci([0.1, 0.2], horizon=20, iters=50)
        self.assertIsNone(stat["ci_low"])
        self.assertIsNone(stat["significant"])


class PowerTest(unittest.TestCase):
    def test_required_blocks_formula(self) -> None:
        # ((1.96 + 0.8416) * 0.13 / 0.03)^2 ≈ 147
        blocks = ek.required_independent_blocks(0.03, 0.13)
        self.assertAlmostEqual(blocks, 147.4, delta=1.0)

    def test_smaller_ic_needs_more_blocks(self) -> None:
        small = ek.required_independent_blocks(0.02, 0.13)
        large = ek.required_independent_blocks(0.06, 0.13)
        self.assertGreater(small, large)

    def test_detectable_ic_is_inverse_of_required_blocks(self) -> None:
        sigma = 0.14
        blocks = ek.required_independent_blocks(0.04, sigma)
        self.assertAlmostEqual(ek.detectable_ic(blocks, sigma), 0.04, places=6)

    def test_power_table_reports_days_per_horizon(self) -> None:
        rows = ek.power_table(0.13, ics=(0.03,), horizons=(5, 20, 60))
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["n_days"]["h20"], rows[0]["n_blocks"] * 20, delta=1.0)
        self.assertGreater(rows[0]["n_years"]["h20"], 1.0)


class MultipleTestingTest(unittest.TestCase):
    def test_bh_fdr_known_example(self) -> None:
        result = ek.bh_fdr([0.01, 0.02, 0.03, 0.5])
        self.assertEqual(result["n_reject"], 3)
        self.assertEqual(result["reject"], [True, True, True, False])
        self.assertLess(result["qvalues"][2], 0.05)

    def test_bh_fdr_all_null_rejects_nothing(self) -> None:
        result = ek.bh_fdr([0.4, 0.6, 0.8, 0.9])
        self.assertEqual(result["n_reject"], 0)

    def test_bh_fdr_tighter_than_raw_threshold(self) -> None:
        pvalues = [0.01, 0.02, 0.03, 0.04, 0.045]
        raw = sum(1 for p in pvalues if p < 0.05)
        self.assertLessEqual(ek.bh_fdr(pvalues, q=0.05)["n_reject"], raw)

    def test_deflated_sharpe_penalises_more_trials(self) -> None:
        returns = [0.0012 if i % 3 else -0.0004 for i in range(400)]
        few = ek.deflated_sharpe(returns, n_trials=1)
        many = ek.deflated_sharpe(returns, n_trials=500)
        self.assertGreater(many["sr0"], few["sr0"])
        self.assertLessEqual(many["dsr"], few["dsr"])
        self.assertAlmostEqual(few["sr0"], 0.0, places=6)

    def test_deflated_sharpe_needs_enough_observations(self) -> None:
        self.assertIsNone(ek.deflated_sharpe([0.01, 0.02], n_trials=5)["dsr"])


class ImportanceTest(unittest.TestCase):
    def _records(self) -> list[dict]:
        records = []
        for index in range(120):
            x1 = (index % 17) / 16.0
            x2 = ((index * 7) % 13) / 12.0
            label = 0.8 * x1 + 0.02 * ((index * 11) % 5) / 4.0
            records.append({"label": label, "rank_a": x1, "rank_b": x2})
        return records

    def test_mdi_ranks_the_informative_factor_first(self) -> None:
        rows = ek.mdi_r2(self._records(), ["a", "b"])
        self.assertEqual(rows[0]["feature"], "a")
        self.assertGreater(rows[0]["drop_r2"], rows[1]["drop_r2"])

    def test_mdi_returns_empty_when_samples_are_thin(self) -> None:
        self.assertEqual(ek.mdi_r2(self._records()[:10], ["a", "b"]), [])


class LedgerTest(unittest.TestCase):
    def test_same_configuration_is_recorded_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = ek.TrialLedger(Path(tmp) / "trials.jsonl")
            first = ledger.record(study="t", target="B", feature="macd_hist", horizon=20,
                                  metric="ic", value=0.01, p=0.3)
            second = ledger.record(study="t", target="B", feature="macd_hist", horizon=20,
                                   metric="ic", value=0.02, p=0.2)
            self.assertEqual(first["runs"], 1)
            self.assertEqual(second["runs"], 2)
            entries = ledger.entries()
            self.assertEqual(len(entries), 1)
            self.assertAlmostEqual(entries[0]["value"], 0.02)

    def test_different_configurations_are_counted_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = ek.TrialLedger(Path(tmp) / "trials.jsonl")
            ledger.record(study="t", target="B", feature="a", horizon=5, metric="ic", value=0.01, p=0.4)
            ledger.record(study="t", target="B", feature="a", horizon=20, metric="ic", value=0.01, p=0.04)
            summary = ledger.summary()
            self.assertEqual(summary["n_trials"], 2)
            self.assertEqual(summary["n_tested"], 2)
            self.assertEqual(summary["by_metric"]["ic"], 2)

    def test_summary_applies_bh_to_ledger_pvalues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trials.jsonl"
            ledger = ek.TrialLedger(path)
            for index, p in enumerate([0.001, 0.002, 0.9, 0.95]):
                ledger.record(study="t", target="B", feature=f"f{index}", metric="ic", value=0.0, p=p)
            summary = ledger.summary(q=0.05)
            self.assertEqual(summary["n_raw_p05"], 2)
            self.assertGreaterEqual(summary["n_bh_reject"], 1)
            self.assertTrue(all("qvalue" in entry for entry in summary["entries"]))

    def test_ledger_ignores_broken_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trials.jsonl"
            path.write_text('{"study": "t", "metric": "ic", "p": 0.1}\nnot json\n', encoding="utf-8")
            self.assertEqual(len(ek.TrialLedger(path).entries()), 1)

    def test_ledger_writes_readable_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trials.jsonl"
            ek.TrialLedger(path).record(study="t", target="A", feature="vol", metric="ic",
                                        value=0.05, p=0.01)
            payload = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(payload["target"], "A")


if __name__ == "__main__":
    unittest.main()
