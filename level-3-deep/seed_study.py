"""Seed study for the Level 3 Deep routes.

Run: python level-3-deep/seed_study.py
Writes level-3-deep/results_seeds.json only. It does not touch
results_deep.json, artifacts/ or charts/, and it records no
timing, so the file is fully deterministic for a given NumPy.

Why: results_deep.json is one training seed on one draw of the
synthetic data, so accuracy gaps of 1 to 6 samples out of 400
cannot be told apart from noise. This study reruns the same 10
routes (same model, data recipe, calibration and pruning code,
imported from build_system and deep_system) over SEEDS seed
sets. Seed index 0 is the recorded configuration (init 42,
train 101, calibration 202, test 303) and must reproduce
results_deep.json exactly; index k uses init 42 + k and splits
101 + 10k, 202 + 10k, 303 + 10k. The class centers (seed 7)
stay fixed, so the task is the same and only the training run
and the data draw vary.

Per route and seed it records test_correct, logit MSE against
the FP32 model of the same seed, the per-sample predictions
(seed 0 only, to keep the file small), the McNemar discordant
counts against FP32 with the exact p-value, and the decision
rule winner under 1200 bytes with and without a one-sample tie.
"""
import json
import os
import platform
import statistics
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "level-2-build"))
import build_system as bs  # noqa: E402
import deep_system as ds  # noqa: E402

SEEDS = 10
ROUTE_ORDER = ["fp32", "int8_per_tensor", "int8_per_channel",
               "int4_per_channel_packed", "pruned_0.3", "pruned_0.5",
               "pruned_0.7", "pruned_0.3_int8_per_channel",
               "pruned_0.5_int8_per_channel",
               "pruned_0.5_int4_per_channel_packed"]


def seed_config(k):
    return {"init": bs.INIT_SEED + k, "train": bs.TRAIN_SEED + 10 * k,
            "calibration": bs.CALIB_SEED + 10 * k,
            "test": bs.TEST_SEED + 10 * k}


def _size(entries):
    with tempfile.TemporaryDirectory() as tmp:
        return bs.save_qlab(os.path.join(tmp, "route.qlab"), entries)


def build_routes(model, X_calib):
    """(name, runnable dequantized model, QLAB entries) for all 10 routes."""
    out = [("fp32", model, bs.variant_entries(model, None, "fp32"))]
    q, deq = bs.per_tensor_model(model, 8)
    out.append(("int8_per_tensor", deq,
                bs.variant_entries(model, q, "int8_per_tensor")))
    for bits, name in [(8, "int8_per_channel"),
                       (4, "int4_per_channel_packed")]:
        q, deq, _ = bs.calibrated_model(model, bits, "per_channel", X_calib)
        out.append((name, deq, bs.variant_entries(model, q, name)))
    for sparsity in bs.PRUNE_SPARSITIES:
        name = "pruned_" + str(sparsity)
        pruned = bs.pruned_model(model, sparsity)
        out.append((name, pruned, bs.variant_entries(pruned, pruned, name)))
    for spec in ds.COMBINED_ROUTES:
        pruned = bs.pruned_model(model, spec["sparsity"])
        q, deq, _ = ds.combined_quantized(pruned, spec["bits"], X_calib)
        out.append((spec["name"], deq,
                    bs.variant_entries(pruned, q, "combined_quantized")))
    return out


def run_seed(k):
    cfg = seed_config(k)
    X_train, y_train = bs.make_split(cfg["train"], bs.TRAIN_N)
    X_calib, _ = bs.make_split(cfg["calibration"], bs.CALIB_N)
    X_test, y_test = bs.make_split(cfg["test"], bs.TEST_N)
    model = bs.train_model(bs.init_model(cfg["init"]), X_train, y_train)
    fp32_pred = bs.forward(model, X_test).argmax(axis=1)
    fp32_ok = fp32_pred == y_test
    routes = {}
    for name, runnable, entries in build_routes(model, X_calib):
        pred = bs.forward(runnable, X_test).argmax(axis=1)
        ok = pred == y_test
        b = int((fp32_ok & ~ok).sum())
        c = int((~fp32_ok & ok).sum())
        routes[name] = {
            "size_bytes": _size(entries),
            "test_correct": int(ok.sum()),
            "test_total": int(len(y_test)),
            "test_accuracy": float(ok.mean()),
            "logit_mse_vs_fp32": bs.logit_mse(model, runnable, X_test),
            "mcnemar_b": b, "mcnemar_c": c,
            "mcnemar_p": ds.mcnemar_exact(b, c),
            "predictions": [int(v) for v in pred]}
    return {"seed_index": k, "seeds": cfg, "routes": routes,
            "labels": [int(v) for v in y_test],
            "fp32_predictions": [int(v) for v in fp32_pred]}


def _rule_rows(routes):
    # The seed study records no timing, so batch time is set to 0.0
    # for every route; it is the last sort key and never decides here.
    return [dict(r, variant=n, timing_batch_ms_median=0.0)
            for n, r in routes.items()]


def main():
    runs = [run_seed(k) for k in range(SEEDS)]
    per_seed = []
    for run in runs:
        rows = _rule_rows(run["routes"])
        strict = ds.recommend_route(rows, max_size_bytes=1200)
        tied = ds.recommend_route(rows, max_size_bytes=1200,
                                  accuracy_tie=1 / bs.TEST_N)
        flags = ds.pareto_flags(rows)
        per_seed.append({
            "seed_index": run["seed_index"], "seeds": run["seeds"],
            "under_1200_strict": strict["variant"],
            "under_1200_one_sample_tie": tied["variant"],
            "pareto_routes": [n for n in ROUTE_ORDER if flags[n]],
            "routes": {n: {key: v for key, v in r.items()
                           if key != "predictions"}
                       for n, r in run["routes"].items()}})
    summary = {}
    for name in ROUTE_ORDER:
        acc = [s["routes"][name]["test_accuracy"] for s in per_seed]
        delta = [s["routes"][name]["test_correct"]
                 - s["routes"]["fp32"]["test_correct"] for s in per_seed]
        mse = [s["routes"][name]["logit_mse_vs_fp32"] for s in per_seed]
        summary[name] = {
            "size_bytes": per_seed[0]["routes"][name]["size_bytes"],
            "accuracy_mean": statistics.fmean(acc),
            "accuracy_sd": statistics.stdev(acc),
            "accuracy_min": min(acc), "accuracy_max": max(acc),
            "delta_correct_vs_fp32_mean": statistics.fmean(delta),
            "delta_correct_vs_fp32_min": min(delta),
            "delta_correct_vs_fp32_max": max(delta),
            "logit_mse_mean": statistics.fmean(mse),
            "logit_mse_min": min(mse), "logit_mse_max": max(mse),
            "seeds_mcnemar_p_below_0_05": sum(
                1 for s in per_seed
                if s["routes"][name]["mcnemar_p"] < 0.05)}
    canonical = runs[0]
    wilson = {}
    for name in ROUTE_ORDER:
        r = canonical["routes"][name]
        lo, hi = ds.wilson_interval(r["test_correct"], r["test_total"])
        wilson[name] = {"test_correct": r["test_correct"],
                        "test_total": r["test_total"], "low": lo, "high": hi}
    out = {
        "scenario": "Seed study for the Level 3 Deep routes: the same 10 "
                    "routes rerun over " + str(SEEDS) + " seed sets "
                    "(training init and the train, calibration and test "
                    "draws vary; class centers fixed). CPU, no timing.",
        "seed_rule": "index k: init 42+k, train 101+10k, calibration "
                     "202+10k, test 303+10k; index 0 is the recorded "
                     "results_deep.json configuration",
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__},
        "seeds": SEEDS,
        "tie_rule": "under_1200_one_sample_tie uses "
                    "recommend_route(max_size_bytes=1200, "
                    "accuracy_tie=1/400)",
        "summary": summary,
        "decision_counts": {
            "under_1200_strict": _count(per_seed, "under_1200_strict"),
            "under_1200_one_sample_tie": _count(
                per_seed, "under_1200_one_sample_tie")},
        "canonical": {
            "labels": canonical["labels"],
            "fp32_predictions": canonical["fp32_predictions"],
            "wilson95": wilson,
            "predictions": {n: canonical["routes"][n]["predictions"]
                            for n in ROUTE_ORDER},
            "mcnemar": {n: {key: canonical["routes"][n][key]
                            for key in ("mcnemar_b", "mcnemar_c",
                                        "mcnemar_p")}
                        for n in ROUTE_ORDER}},
        "per_seed": per_seed,
    }
    path = os.path.join(HERE, "results_seeds.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print("Wrote", path)
    for name in ROUTE_ORDER:
        s = summary[name]
        print(f"{name}: acc {s['accuracy_mean']:.4f} sd "
              f"{s['accuracy_sd']:.4f} delta {s['delta_correct_vs_fp32_min']}"
              f"..{s['delta_correct_vs_fp32_max']} mcnemar<0.05 "
              f"{s['seeds_mcnemar_p_below_0_05']}/{SEEDS}")
    print("decisions:", out["decision_counts"])
    return 0


def _count(per_seed, key):
    counts = {}
    for s in per_seed:
        counts[s[key]] = counts.get(s[key], 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    raise SystemExit(main())
