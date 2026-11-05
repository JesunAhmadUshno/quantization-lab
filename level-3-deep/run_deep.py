"""Run the Level 3 Deep study end to end and write results.

Run: python level-3-deep/run_deep.py
Writes level-3-deep/results_deep.json, QLAB files under
level-3-deep/artifacts/, and SVG charts under
level-3-deep/charts/. The results file is deterministic
except the timing fields timing_batch_ms_median and
timing_single_ms_median in each row: run it twice and the
files are identical once those fields are removed.

Timing method: batch time is the median of 31 full test-set
(400 sample) forward passes after 5 warmup passes.
Single-sample latency is the median of 101 forward passes on
the first test sample after 5 warmup passes. Both are CPU
timing on the machine running this script, on dequantized
weights for quantized variants (fake quantization timing).
"""
import json
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "level-2-build"))
import build_system as bs  # noqa: E402
import deep_system as ds  # noqa: E402

BATCH_REPEATS = 31
SINGLE_REPEATS = 101
WARMUP = 5


def time_model(model, X_batch, x_single):
    for _ in range(WARMUP):
        bs.forward(model, X_batch)
    batch = []
    for _ in range(BATCH_REPEATS):
        start = time.perf_counter()
        bs.forward(model, X_batch)
        batch.append((time.perf_counter() - start) * 1000.0)
    for _ in range(WARMUP):
        bs.forward(model, x_single)
    single = []
    for _ in range(SINGLE_REPEATS):
        start = time.perf_counter()
        bs.forward(model, x_single)
        single.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(batch), statistics.median(single)


def main():
    datasets = bs.build_datasets()
    X_train, y_train = datasets["train"]
    X_calib, _ = datasets["calibration"]
    X_test, y_test = datasets["test"]
    model = bs.train_model(bs.init_model(), X_train, y_train)
    fp32_correct = int((bs.forward(model, X_test).argmax(axis=1)
                        == y_test).sum())
    artifacts = os.path.join(HERE, "artifacts")
    os.makedirs(artifacts, exist_ok=True)
    rows = []
    calibration = {}

    def serialize_and_check(variant_name, entries, reference_model,
                            entry_variant=None):
        # entry_variant is what build_system branches on. Combined
        # route names start with pruned, which build_system treats
        # as the dense FP32 path, so combined routes pass a
        # neutral name here and keep their real name for the file.
        branch = entry_variant or variant_name
        path = os.path.join(artifacts, variant_name + ".qlab")
        size = bs.save_qlab(path, entries)
        header, data = bs.load_qlab(path)
        loaded = bs.model_from_qlab(header, data, branch)
        for key in bs.ALL_KEYS:
            if not np.allclose(loaded[key], reference_model[key],
                               atol=1e-6):
                raise bs.BuildError(
                    "QLAB round trip mismatch for " + variant_name)
        return size, loaded

    def add_row(name, runnable, size, extra=None):
        errors = bs.weight_error(model, runnable)
        correct = int((bs.forward(runnable, X_test).argmax(axis=1)
                       == y_test).sum())
        batch_ms, single_ms = time_model(runnable, X_test, X_test[:1])
        row = {"variant": name,
               "size_bytes": size,
               "size_ratio_vs_fp32": None,
               "max_abs_weight_error": errors["max_abs_error"],
               "mean_abs_weight_error": errors["mean_abs_error"],
               "test_accuracy": bs.accuracy(runnable, X_test, y_test),
               "test_correct": correct,
               "test_total": int(len(y_test)),
               "accuracy_delta_vs_fp32": None,
               "logit_mse_vs_fp32": bs.logit_mse(model, runnable, X_test),
               "timing_batch_ms_median": batch_ms,
               "timing_single_ms_median": single_ms}
        if extra:
            row.update(extra)
        rows.append(row)
        print(f"{name}: size={size} acc={row['test_accuracy']:.4f} "
              f"mse={row['logit_mse_vs_fp32']:.6f} "
              f"batch_ms={batch_ms:.3f} single_ms={single_ms:.4f}")
        return row

    # The 7 Build routes, rebuilt here so Deep is self-contained.
    size, loaded = serialize_and_check(
        "fp32", bs.variant_entries(model, None, "fp32"), model)
    add_row("fp32", loaded, size,
            {"nonzero_parameters": bs.nonzero_count(model)})

    q_pt, deq_pt = bs.per_tensor_model(model, 8)
    size, loaded = serialize_and_check(
        "int8_per_tensor",
        bs.variant_entries(model, q_pt, "int8_per_tensor"), deq_pt)
    add_row("int8_per_tensor", loaded, size)

    q_pc8, deq_pc8, chosen8 = bs.calibrated_model(
        model, 8, "per_channel", X_calib)
    calibration["int8_per_channel"] = chosen8
    size, loaded = serialize_and_check(
        "int8_per_channel",
        bs.variant_entries(model, q_pc8, "int8_per_channel"), deq_pc8)
    add_row("int8_per_channel", loaded, size)

    q_pc4, deq_pc4, chosen4 = bs.calibrated_model(
        model, 4, "per_channel", X_calib)
    calibration["int4_per_channel_packed"] = chosen4
    size, loaded = serialize_and_check(
        "int4_per_channel_packed",
        bs.variant_entries(model, q_pc4, "int4_per_channel_packed"),
        deq_pc4)
    add_row("int4_per_channel_packed", loaded, size)

    for sparsity in bs.PRUNE_SPARSITIES:
        name = "pruned_" + str(sparsity)
        pruned = bs.pruned_model(model, sparsity)
        size, loaded = serialize_and_check(
            name, bs.variant_entries(pruned, pruned, name), pruned)
        add_row(name, loaded, size,
                {"sparsity": sparsity,
                 "nonzero_parameters": bs.nonzero_count(pruned),
                 "note": "NumPy magnitude pruning, labelled equivalent "
                         "track. TensorFlow unavailable on this "
                         "machine. Stored dense in FP32."})

    # The 3 combined routes, new in Deep.
    for spec in ds.COMBINED_ROUTES:
        pruned = bs.pruned_model(model, spec["sparsity"])
        quantized, dequant, chosen = ds.combined_quantized(
            pruned, spec["bits"], X_calib)
        calibration[spec["name"]] = chosen
        size, loaded = serialize_and_check(
            spec["name"],
            bs.variant_entries(pruned, quantized, "combined_quantized"),
            dequant, entry_variant="combined_quantized")
        add_row(spec["name"], loaded, size,
                {"sparsity": spec["sparsity"], "bits": spec["bits"],
                 "nonzero_parameters": bs.nonzero_count(dequant),
                 "note": "Prune first, then calibrated per-channel "
                         "quantization of the pruned weights. NumPy "
                         "pruning track, not a TensorFlow result."})

    fp32_size = rows[0]["size_bytes"]
    fp32_acc = rows[0]["test_accuracy"]
    for row in rows:
        row["size_ratio_vs_fp32"] = float(fp32_size) / float(
            row["size_bytes"])
        row["accuracy_delta_vs_fp32"] = float(
            row["test_accuracy"] - fp32_acc)
        row["accuracy_per_kb"] = ds.accuracy_per_kb(row)
    flags = ds.pareto_flags(rows)
    for row in rows:
        row["pareto_optimal"] = bool(flags[row["variant"]])

    charts = ds.build_charts(rows)
    charts_dir = os.path.join(HERE, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    for filename, svg in charts.items():
        with open(os.path.join(charts_dir, filename), "w",
                  encoding="utf-8") as fh:
            fh.write(svg)

    at_fp32 = [r for r in rows if r["test_accuracy"] >= fp32_acc]
    verdict_inputs = {
        "best_accuracy_route": ds.recommend_route(rows)["variant"],
        "smallest_route": min(rows, key=lambda r: r["size_bytes"])[
            "variant"],
        "smallest_at_fp32_accuracy": min(
            at_fp32, key=lambda r: (r["size_bytes"],
                                    r["logit_mse_vs_fp32"]))["variant"]
        if at_fp32 else None,
        "lowest_mse_quantized_at_fp32_accuracy": min(
            (r for r in at_fp32 if r["variant"] != "fp32"),
            key=lambda r: r["logit_mse_vs_fp32"])["variant"]
        if len(at_fp32) > 1 else None,
        "under_1200_bytes_best": (
            ds.recommend_route(rows, max_size_bytes=1200)["variant"]
            if ds.recommend_route(rows, max_size_bytes=1200) else None),
        "pareto_routes": [r["variant"] for r in rows
                          if r["pareto_optimal"]],
    }
    out = {
        "scenario": "Level 3 Deep: quality against size against speed "
                    "study on the Build MLP (16-32-4, 676 parameters), "
                    "10 routes (7 Build routes plus 3 combined prune "
                    "plus quantize routes), CPU only",
        "model": {"architecture": "MLP 16-32-4, ReLU hidden, 676 "
                                   "parameters",
                  "initialization_seed": bs.INIT_SEED,
                  "training": "full-batch gradient descent, 60 epochs, "
                              "learning rate 0.1, softmax cross entropy"},
        "dataset": {"kind": "synthetic Gaussian mixture, 4 classes, "
                             "input dimension 16, centers seed 7 "
                             "scale 0.6, noise std 1.0",
                    "train_samples": bs.TRAIN_N,
                    "calibration_samples": bs.CALIB_N,
                    "test_samples": bs.TEST_N},
        "substitutions": [
            "QLAB container format, designed for this project, is "
            "used instead of GGUF. QLAB is not GGUF and no file "
            "here runs in llama.cpp, which is unavailable on this "
            "machine.",
            "The pruning and combined rows use the NumPy "
            "equivalent pruning track. TensorFlow is unavailable "
            "on this machine, verified this cycle.",
            "No GPU was used. BlackBox was unreachable this cycle "
            "(NEEDS-BLACKBOX-ON). All timing is CPU timing on this "
            "Muse VM."],
        "timing_note": "timing_batch_ms_median is the median of 31 "
                       "full test-set (400 sample) forward passes "
                       "after 5 warmup passes. "
                       "timing_single_ms_median is the median of 101 "
                       "single-sample forward passes after 5 warmup "
                       "passes. Both are CPU timing on this machine "
                       "(single machine, varying load), on "
                       "dequantized weights for quantized variants. "
                       "These two fields per row are the only "
                       "nondeterministic fields in this file.",
        "fp32_test_correct": fp32_correct,
        "calibration": calibration,
        "verdict_inputs": verdict_inputs,
        "rows": rows,
    }
    path = os.path.join(HERE, "results_deep.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("Wrote", path)
    print("Pareto routes:", verdict_inputs["pareto_routes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
