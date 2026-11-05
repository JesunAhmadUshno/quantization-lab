"""Run the Level 2 Build study end to end and write results.

Run: python level-2-build/run_build.py
Writes level-2-build/results_build.json and QLAB model files
under level-2-build/artifacts/. The results file is
deterministic except the timing fields named timing_ms_median
in each row: run it twice and the files are identical once
those fields are removed. Timing is the median of 15 full
test-set forward passes on the machine running this script
(CPU, single machine, varying load), measured on dequantized
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
import build_system as bs  # noqa: E402

TIMING_REPEATS = 15


def time_forward(model, X):
    times = []
    for _ in range(TIMING_REPEATS):
        start = time.perf_counter()
        bs.forward(model, X)
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times)


def evaluate_row(name, fp32_model, runnable_model, size_bytes, X_test, y_test,
                 extra=None):
    errors = bs.weight_error(fp32_model, runnable_model)
    row = {"variant": name,
           "size_bytes": size_bytes,
           "max_abs_weight_error": errors["max_abs_error"],
           "mean_abs_weight_error": errors["mean_abs_error"],
           "test_accuracy": bs.accuracy(runnable_model, X_test, y_test),
           "test_correct": int((bs.forward(runnable_model, X_test).argmax(axis=1)
                                 == y_test).sum()),
           "test_total": int(len(y_test)),
           "logit_mse_vs_fp32": bs.logit_mse(fp32_model, runnable_model, X_test),
           "timing_ms_median": time_forward(runnable_model, X_test)}
    if extra:
        row.update(extra)
    return row


def main():
    datasets = bs.build_datasets()
    X_train, y_train = datasets["train"]
    X_calib, _ = datasets["calibration"]
    X_test, y_test = datasets["test"]
    model = bs.train_model(bs.init_model(), X_train, y_train)
    artifacts = os.path.join(HERE, "artifacts")
    os.makedirs(artifacts, exist_ok=True)
    rows = []
    calibration = {}

    def serialize_and_check(variant_name, entries, reference_model):
        path = os.path.join(artifacts, variant_name + ".qlab")
        size = bs.save_qlab(path, entries)
        header, data = bs.load_qlab(path)
        loaded = bs.model_from_qlab(header, data, variant_name)
        # QLAB stores FP32 values and FP32 scales, so a loaded model
        # can differ from the float64 reference by float32 rounding
        # (below 1e-6 here). The check uses that named tolerance,
        # and the rows below evaluate the loaded model itself.
        for key in bs.ALL_KEYS:
            if not np.allclose(loaded[key], reference_model[key], atol=1e-6):
                raise bs.BuildError("QLAB round trip mismatch for " + variant_name)
        return size, loaded

    # FP32 baseline.
    size, loaded = serialize_and_check(
        "fp32", bs.variant_entries(model, None, "fp32"), model)
    rows.append(evaluate_row("fp32", model, loaded, size, X_test, y_test,
                             {"nonzero_parameters": bs.nonzero_count(model)}))

    # INT8 per-tensor, the Learn method, no calibration.
    q_pt, deq_pt = bs.per_tensor_model(model, 8)
    size, loaded = serialize_and_check(
        "int8_per_tensor", bs.variant_entries(model, q_pt, "int8_per_tensor"),
        deq_pt)
    rows.append(evaluate_row("int8_per_tensor", model, loaded, size,
                             X_test, y_test))

    # INT8 per-channel, calibrated.
    q_pc8, deq_pc8, chosen8 = bs.calibrated_model(model, 8, "per_channel", X_calib)
    calibration["int8_per_channel"] = chosen8
    size, loaded = serialize_and_check(
        "int8_per_channel", bs.variant_entries(model, q_pc8, "int8_per_channel"),
        deq_pc8)
    rows.append(evaluate_row("int8_per_channel", model, loaded, size,
                             X_test, y_test))

    # INT4 per-channel, packed, calibrated.
    q_pc4, deq_pc4, chosen4 = bs.calibrated_model(model, 4, "per_channel", X_calib)
    calibration["int4_per_channel_packed"] = chosen4
    size, loaded = serialize_and_check(
        "int4_per_channel_packed",
        bs.variant_entries(model, q_pc4, "int4_per_channel_packed"), deq_pc4)
    rows.append(evaluate_row("int4_per_channel_packed", model, loaded, size,
                             X_test, y_test))

    # Pruning track, NumPy equivalent of the TensorFlow track.
    for sparsity in bs.PRUNE_SPARSITIES:
        name = "pruned_" + str(sparsity)
        pruned = bs.pruned_model(model, sparsity)
        size, loaded = serialize_and_check(
            name, bs.variant_entries(pruned, pruned, name), pruned)
        rows.append(evaluate_row(
            name, model, loaded, size, X_test, y_test,
            {"sparsity": sparsity,
             "nonzero_parameters": bs.nonzero_count(pruned),
             "note": "NumPy magnitude pruning, labelled equivalent track. "
                     "The TensorFlow Model Optimization Toolkit row did "
                     "not run because TensorFlow is unavailable on this "
                     "machine. Stored dense in FP32, so size equals FP32."}))

    for row in rows:
        print(f"{row['variant']}: size={row['size_bytes']} "
              f"max_err={row['max_abs_weight_error']:.6f} "
              f"mean_err={row['mean_abs_weight_error']:.6f} "
              f"accuracy={row['test_accuracy']:.4f} "
              f"timing_ms_median={row['timing_ms_median']:.3f}")
    out = {
        "scenario": "Level 2 Build: 2-layer MLP (16-32-4, 676 parameters) "
                    "trained with NumPy on a synthetic Gaussian mixture, "
                    "quantized, pruned, serialized in QLAB format, and "
                    "benchmarked on CPU",
        "model": {"architecture": "MLP 16-32-4, ReLU hidden, 676 parameters",
                  "initialization_seed": bs.INIT_SEED,
                  "training": "full-batch gradient descent, 60 epochs, "
                              "learning rate 0.1, softmax cross entropy"},
        "dataset": {"kind": "synthetic Gaussian mixture, 4 classes, "
                             "input dimension 16, centers seed 7 scale 0.6, "
                             "noise std 1.0",
                    "train_samples": bs.TRAIN_N,
                    "calibration_samples": bs.CALIB_N,
                    "test_samples": bs.TEST_N},
        "substitutions": [
            "QLAB container format, designed for this project, is used "
            "instead of GGUF. QLAB is not GGUF and no file here runs in "
            "llama.cpp, which is unavailable on this machine.",
            "The pruning rows are a NumPy equivalent track. The "
            "TensorFlow Model Optimization Toolkit row did not run "
            "because TensorFlow is unavailable on this machine.",
            "No GPU was used. BlackBox was unreachable this cycle "
            "(NEEDS-BLACKBOX-ON). All timing is CPU timing on this "
            "machine."],
        "calibration": calibration,
        "timing_note": "timing_ms_median is the median of 15 full test-set "
                       "forward passes on this machine (CPU, single "
                       "machine, varying load), on dequantized weights "
                       "for quantized variants. It is the only "
                       "nondeterministic field in this file.",
        "rows": rows,
    }
    path = os.path.join(HERE, "results_build.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("Wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
