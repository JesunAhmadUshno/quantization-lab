"""Portfolio standard checks: required files exist, no public grade, and the regeneration check catches a change.

No lock test: the repo is not pre-registered (PROPOSAL.md), so there is no locked file to pin.
"""
import os
import re
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from check_results import diff  # noqa: E402

REQUIRED = ["README.md", "PROPOSAL.md", "REPORT.md", "CHANGELOG.md", "CITATION.cff", "SECURITY.md", "LICENSE",
            "scripts/check_results.py", "docs/decisions/0001-results-stay-beside-their-runners.md",
            "level-1-learn/results_learn.json", "level-2-build/results_build.json",
            "level-3-deep/results_deep.json", "level-3-deep/results_seeds.json"]
GRADE = re.compile(r"of 10,000|10,000 scale|/ ?10000|Grade:|grade:|1,000[- ]point|Total /1000", re.I)


class TestLayout(unittest.TestCase):
    def test_required_files_exist(self):
        missing = [p for p in REQUIRED if not os.path.exists(os.path.join(ROOT, p))]
        self.assertEqual(missing, [], "missing: " + ", ".join(missing))

    def test_readme_opens_with_the_abstract(self):
        text = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
        for field in ("**Question.**", "**Headline.**", "**Field.**", "**Job family.**", "**Verdict.**",
                      "## Reproduce", "## Lifecycle", "## Layout"):
            self.assertIn(field, text)

    def test_no_public_grade(self):
        # Every tracked text file, not only Markdown: a grade hid in a .py fixture once.
        files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
                               check=True).stdout.split()
        hits = []
        for rel in files:
            try:
                lines = open(os.path.join(ROOT, rel), encoding="utf-8").readlines()
            except (UnicodeDecodeError, FileNotFoundError):
                continue  # binary or deleted in the working tree
            for n, line in enumerate(lines, 1):
                if GRADE.search(line) and not line.startswith("GRADE = re.compile("):
                    hits.append(f"{rel}:{n}")
        self.assertEqual(hits, [], "grades are internal: " + ", ".join(hits))

    def test_grade_pattern_catches_a_grade(self):
        # Synthetic value, built in pieces so this file does not trip its own scan.
        self.assertTrue(GRADE.search("Gra" + "de: 1,234 of 10" + ",000"))

    def test_regeneration_check_flags_changes_and_skips_timing(self):
        old = {"rows": [{"test_correct": 353, "timing_batch_ms_median": 0.25, "logit_mse": 0.045583}],
               "environment": {"numpy": "1.26.4"}}
        same = {"rows": [{"test_correct": 353, "timing_batch_ms_median": 0.31, "logit_mse": 0.0455830000001}],
                "environment": {"numpy": "2.5.4"}}
        changed = {"rows": [{"test_correct": 354, "timing_batch_ms_median": 0.25, "logit_mse": 0.045583}],
                   "environment": {"numpy": "1.26.4"}}
        self.assertEqual(diff(old, same), [])
        self.assertEqual(len(diff(old, changed)), 1)


if __name__ == "__main__":
    unittest.main()
