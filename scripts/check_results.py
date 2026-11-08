"""Prove the committed results regenerate from the code.

Run: python scripts/check_results.py

Reruns every runner (Learn, Build, Deep, seed study) and compares each
fresh results file with the committed one. Two kinds of field are
machine facts, not results, and are skipped: timing medians (keys
starting "timing_" with a number value) and the recorded environment
(Python and NumPy versions). Floats must agree to a relative 1e-6,
because NumPy versions differ in the last bits; every count, size,
string and flag must match exactly. The committed results files and
charts are restored afterwards, so the working tree is left as found.
Exit code 0 means every file regenerated.
"""
import glob
import json
import math
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = [
    ("level-1-learn/run_learn.py", "level-1-learn/results_learn.json"),
    ("level-2-build/run_build.py", "level-2-build/results_build.json"),
    ("level-3-deep/run_deep.py", "level-3-deep/results_deep.json"),
    ("level-3-deep/seed_study.py", "level-3-deep/results_seeds.json"),
]


def diff(old, new, path="$"):
    """Return a list of differences between two JSON values."""
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for key in sorted(set(old) | set(new)):
            if key == "environment":
                continue
            if key.startswith("timing_") and isinstance(old.get(key, 0.0), (int, float)):
                continue
            if key not in old or key not in new:
                out.append(f"{path}.{key}: present in only one file")
            else:
                out += diff(old[key], new[key], f"{path}.{key}")
        return out
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return [f"{path}: length {len(old)} != {len(new)}"]
        return [d for i, (a, b) in enumerate(zip(old, new)) for d in diff(a, b, f"{path}[{i}]")]
    if isinstance(old, float) and isinstance(new, float):
        return [] if math.isclose(old, new, rel_tol=1e-6, abs_tol=1e-12) else [f"{path}: {old} != {new}"]
    return [] if old == new and type(old) is type(new) else [f"{path}: {old!r} != {new!r}"]


def main():
    keep = [p for _, p in RUNS] + glob.glob(os.path.join(ROOT, "level-3-deep", "charts", "*.svg"))
    saved = {p: open(os.path.join(ROOT, p), "rb").read() for p in keep}
    failures = []
    try:
        for runner, result in RUNS:
            subprocess.run([sys.executable, os.path.join(ROOT, runner)], check=True,
                           stdout=subprocess.DEVNULL, cwd=ROOT)
            with open(os.path.join(ROOT, result), encoding="utf-8") as fh:
                fresh = json.load(fh)
            problems = diff(json.loads(saved[result]), fresh)
            print(f"{result}: {'regenerated' if not problems else 'DIFFERS'}")
            failures += [f"{result} {p}" for p in problems]
    finally:
        for p, data in saved.items():
            with open(os.path.join(ROOT, p), "wb") as fh:
                fh.write(data)
    for line in failures[:20]:
        print("  " + line)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
