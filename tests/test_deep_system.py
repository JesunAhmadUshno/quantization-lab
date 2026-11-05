"""Unit tests for Level 3 Deep. Hand-computed expectations where noted."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "level-3-deep"))
sys.path.insert(0, os.path.join(ROOT, "level-2-build"))
sys.path.insert(0, os.path.join(ROOT, "level-1-learn"))
import build_system as bs  # noqa: E402
import deep_system as ds  # noqa: E402


def _row(variant, size, acc, mse, batch_ms):
    return {"variant": variant, "size_bytes": size,
            "test_accuracy": acc, "logit_mse_vs_fp32": mse,
            "timing_batch_ms_median": batch_ms,
            "timing_single_ms_median": batch_ms / 400.0}


class TestPareto(unittest.TestCase):
    def test_hand_computed_flags(self):
        # B dominates A (smaller, same accuracy and time).
        rows = [_row("A", 100, 0.90, 0.1, 1.0),
                _row("B", 50, 0.90, 0.1, 1.0),
                _row("C", 200, 0.95, 0.2, 2.0)]
        flags = ds.pareto_flags(rows)
        self.assertEqual(flags, {"A": False, "B": True, "C": True})

    def test_identical_rows_are_both_pareto(self):
        rows = [_row("A", 100, 0.90, 0.1, 1.0),
                _row("B", 100, 0.90, 0.1, 1.0)]
        flags = ds.pareto_flags(rows)
        self.assertTrue(flags["A"] and flags["B"])


class TestDecisionRule(unittest.TestCase):
    def setUp(self):
        self.rows = [_row("big_accurate", 3000, 0.90, 0.0, 1.0),
                     _row("small_accurate", 1000, 0.90, 0.05, 1.0),
                     _row("small_weak", 900, 0.80, 0.5, 0.5),
                     _row("tiny_weak", 500, 0.70, 1.0, 0.4)]

    def test_unconstrained_picks_lowest_mse_at_top_accuracy(self):
        self.assertEqual(ds.recommend_route(self.rows)["variant"],
                         "big_accurate")

    def test_size_cap_picks_best_survivor(self):
        winner = ds.recommend_route(self.rows, max_size_bytes=1200)
        self.assertEqual(winner["variant"], "small_accurate")

    def test_accuracy_floor_and_size_cap(self):
        winner = ds.recommend_route(
            self.rows, max_size_bytes=950, min_accuracy=0.75)
        self.assertEqual(winner["variant"], "small_weak")

    def test_impossible_constraints_return_none(self):
        self.assertIsNone(ds.recommend_route(
            self.rows, max_size_bytes=100, min_accuracy=0.99))


class TestCombinedRoutes(unittest.TestCase):
    def test_pruned_zeros_survive_quantization(self):
        datasets = bs.build_datasets()
        model = bs.train_model(bs.init_model(), *datasets["train"])
        pruned = bs.pruned_model(model, 0.5)
        quantized, dequant, chosen = ds.combined_quantized(
            pruned, 8, datasets["calibration"][0])
        for key in bs.WEIGHT_KEYS:
            self.assertTrue(
                (dequant[key][pruned[key] == 0] == 0).all())
        self.assertEqual(set(chosen), set(bs.WEIGHT_KEYS))

    def test_combined_route_is_smaller_than_fp32(self):
        datasets = bs.build_datasets()
        model = bs.train_model(bs.init_model(), *datasets["train"])
        pruned = bs.pruned_model(model, 0.5)
        quantized, dequant, _ = ds.combined_quantized(
            pruned, 4, datasets["calibration"][0])
        entries = bs.variant_entries(
            pruned, quantized, "combined_quantized")
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            size = bs.save_qlab(os.path.join(tmp, "c.qlab"), entries)
        self.assertLess(size, 3013)


class TestDerivedAndCharts(unittest.TestCase):
    def test_accuracy_per_kb_hand_computed(self):
        row = _row("x", 2000, 0.80, 0.1, 1.0)
        self.assertAlmostEqual(ds.accuracy_per_kb(row), 0.40)

    def test_accuracy_per_kb_rejects_zero_size(self):
        with self.assertRaises(ds.DeepError):
            ds.accuracy_per_kb(_row("x", 0, 0.8, 0.1, 1.0))

    def test_charts_contain_every_route_and_title(self):
        rows = [_row("alpha", 1000, 0.90, 0.01, 1.0),
                _row("beta", 500, 0.80, 0.20, 0.8)]
        charts = ds.build_charts(rows)
        self.assertEqual(sorted(charts),
                         ["accuracy_vs_size.svg",
                          "batch_time_vs_accuracy.svg",
                          "logit_mse_vs_size.svg"])
        for svg in charts.values():
            self.assertIn("<svg", svg)
            self.assertIn("alpha", svg)
            self.assertIn("beta", svg)
            self.assertEqual(svg.count("<circle"), 2)

    def test_chart_rejects_empty_rows(self):
        with self.assertRaises(ds.DeepError):
            ds.svg_scatter([], "size_bytes", "test_accuracy",
                           "t", "x", "y")


if __name__ == "__main__":
    unittest.main()
