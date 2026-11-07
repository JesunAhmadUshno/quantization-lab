"""Tests for the research upgrades: tie tolerance in the decision rule,
Wilson intervals, the exact McNemar test, and the seed study.
Hand-computed expectations where noted."""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "level-3-deep"))
sys.path.insert(0, os.path.join(ROOT, "level-2-build"))
sys.path.insert(0, os.path.join(ROOT, "level-1-learn"))
import deep_system as ds  # noqa: E402
import seed_study as ss  # noqa: E402

with open(os.path.join(ROOT, "level-3-deep", "results_deep.json"),
          encoding="utf-8") as fh:
    DEEP = json.load(fh)


class TestTieTolerance(unittest.TestCase):
    def test_default_rule_is_unchanged_on_recorded_rows(self):
        winner = ds.recommend_route(DEEP["rows"], max_size_bytes=1200)
        self.assertEqual(winner["variant"],
                         DEEP["verdict_inputs"]["under_1200_bytes_best"])

    def test_one_sample_tie_matches_written_verdict(self):
        # 348 vs 347 of 400 is one sample (0.0025). Treated as a tie,
        # the lower logit MSE wins: plain INT4, as the README verdict says.
        winner = ds.recommend_route(DEEP["rows"], max_size_bytes=1200,
                                    accuracy_tie=1 / 400)
        self.assertEqual(winner["variant"], "int4_per_channel_packed")

    def test_tie_does_not_change_unconstrained_pick(self):
        winner = ds.recommend_route(DEEP["rows"], accuracy_tie=1 / 400)
        self.assertEqual(winner["variant"], "fp32")

    def test_negative_tie_rejected(self):
        with self.assertRaises(ds.DeepError):
            ds.recommend_route(DEEP["rows"], accuracy_tie=-0.1)


class TestWilson(unittest.TestCase):
    def test_hand_computed_half(self):
        # 5 of 10, z 1.96: centre 0.5, half width
        # 1.96 * sqrt(0.025 + 0.009604) / 1.38416 = 0.2366
        lo, hi = ds.wilson_interval(5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=4)
        self.assertAlmostEqual(hi, 0.7634, places=4)

    def test_zero_successes_has_zero_lower_bound(self):
        lo, hi = ds.wilson_interval(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.0)

    def test_rejects_bad_counts(self):
        with self.assertRaises(ds.DeepError):
            ds.wilson_interval(5, 0)
        with self.assertRaises(ds.DeepError):
            ds.wilson_interval(11, 10)


class TestMcNemar(unittest.TestCase):
    def test_no_discordant_pairs(self):
        self.assertEqual(ds.mcnemar_exact(0, 0), 1.0)

    def test_hand_computed_one_sided_run(self):
        # b=5, c=0: 2 * 0.5**5 = 0.0625
        self.assertAlmostEqual(ds.mcnemar_exact(5, 0), 0.0625)

    def test_symmetric_and_capped_at_one(self):
        # b=1, c=2: 2 * P(X <= 1 | n=3) = 2 * 4/8 = 1.0
        self.assertEqual(ds.mcnemar_exact(1, 2), 1.0)
        self.assertEqual(ds.mcnemar_exact(2, 1), 1.0)


class TestSeedStudy(unittest.TestCase):
    def test_canonical_seed_reproduces_recorded_deep_rows(self):
        out = ss.run_seed(0)
        recorded = {r["variant"]: r for r in DEEP["rows"]}
        self.assertEqual(sorted(out["routes"]), sorted(recorded))
        for name, row in out["routes"].items():
            self.assertEqual(row["test_correct"],
                             recorded[name]["test_correct"], name)
            self.assertEqual(row["size_bytes"],
                             recorded[name]["size_bytes"], name)
            self.assertAlmostEqual(row["logit_mse_vs_fp32"],
                                   recorded[name]["logit_mse_vs_fp32"],
                                   places=6, msg=name)

    def test_predictions_and_discordance_are_consistent(self):
        out = ss.run_seed(0)
        fp32 = out["fp32_predictions"]
        labels = out["labels"]
        for name, row in out["routes"].items():
            preds = row["predictions"]
            b = sum(1 for f, p, y in zip(fp32, preds, labels)
                    if f == y and p != y)
            c = sum(1 for f, p, y in zip(fp32, preds, labels)
                    if f != y and p == y)
            self.assertEqual((b, c), (row["mcnemar_b"], row["mcnemar_c"]))
            self.assertEqual(sum(p == y for p, y in zip(preds, labels)),
                             row["test_correct"])


if __name__ == "__main__":
    unittest.main()
