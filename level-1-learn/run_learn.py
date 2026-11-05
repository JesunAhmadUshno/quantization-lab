"""Run the Learn quantization study and write deterministic results.

Run: python level-1-learn/run_learn.py
Writes level-1-learn/results_learn.json. No timestamps, no randomness,
no wall clock values, so two runs produce an identical file.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from quantize import (  # noqa: E402
    build_task_samples, build_weight_matrix, error_metrics, evaluate_task,
    flatten, pack_int4, quantize_symmetric, reshape, score_mae, size_bytes,
)


def fmt_result(name, bits, flat):
    if bits == 32:
        sizes = size_bytes(len(flat), 32)
        return {"format": name, "bits": 32, "scale": None, "codes": None,
                "packed_bytes": None, "dequantized": list(flat),
                "errors": {"max_abs_error": 0.0, "mean_abs_error": 0.0,
                           "mse": 0.0, "total_abs_error": 0.0},
                "sizes": sizes}
    q = quantize_symmetric(flat, bits)
    result = {"format": name, "bits": bits, "scale": q["scale"],
              "codes": q["codes"],
              "packed_bytes": pack_int4(q["codes"]) if bits == 4 else None,
              "dequantized": q["dequantized"],
              "errors": error_metrics(flat, q["dequantized"]),
              "sizes": size_bytes(len(flat), bits)}
    return result


def main():
    matrix = build_weight_matrix()
    flat = flatten(matrix)
    samples = build_task_samples()
    formats = [fmt_result("fp32", 32, flat),
               fmt_result("int8", 8, flat),
               fmt_result("int4", 4, flat)]
    task = {}
    for f in formats:
        deq_matrix = reshape(f["dequantized"], 4, 3)
        ev = evaluate_task(deq_matrix, samples)
        task[f["format"]] = {
            "predictions": ev["predictions"], "correct": ev["correct"],
            "total": ev["total"], "accuracy": ev["accuracy"],
            "score_mae_vs_fp32": score_mae(matrix, deq_matrix, samples)}
    per_sample = []
    for i, sample in enumerate(samples):
        row = {"sample": i + 1, "input": sample["input"], "label": sample["label"]}
        for f in formats:
            deq_matrix = reshape(f["dequantized"], 4, 3)
            from quantize import linear_scores
            row[f["format"] + "_scores"] = linear_scores(deq_matrix, sample["input"])
            row[f["format"] + "_prediction"] = task[f["format"]]["predictions"][i]
        per_sample.append(row)
    summary = {
        "weight_count": len(flat),
        "fp32_total_bytes": formats[0]["sizes"]["total_bytes"],
        "int8_total_bytes": formats[1]["sizes"]["total_bytes"],
        "int4_total_bytes": formats[2]["sizes"]["total_bytes"],
        "int8_max_abs_error": formats[1]["errors"]["max_abs_error"],
        "int8_mean_abs_error": formats[1]["errors"]["mean_abs_error"],
        "int8_mse": formats[1]["errors"]["mse"],
        "int4_max_abs_error": formats[2]["errors"]["max_abs_error"],
        "int4_mean_abs_error": formats[2]["errors"]["mean_abs_error"],
        "int4_mse": formats[2]["errors"]["mse"],
        "fp32_accuracy": task["fp32"]["accuracy"],
        "int8_accuracy": task["int8"]["accuracy"],
        "int4_accuracy": task["int4"]["accuracy"],
        "task_samples": len(samples),
    }
    for f in formats:
        print(f"{f['format']}: scale={f['scale']} total_bytes={f['sizes']['total_bytes']} "
              f"max_abs_error={f['errors']['max_abs_error']:.6f} "
              f"mean_abs_error={f['errors']['mean_abs_error']:.6f} "
              f"mse={f['errors']['mse']:.6f}")
    print(f"Task: fp32 accuracy={task['fp32']['accuracy']} "
          f"int8 accuracy={task['int8']['accuracy']} "
          f"int4 accuracy={task['int4']['accuracy']} "
          f"on {len(samples)} samples")
    out = {"scenario": "symmetric per-tensor quantization of a deterministic "
                       "4x3 linear model, 12 weights, 8 synthetic task samples",
           "weights_matrix": matrix, "weights_flat": flat,
           "formats": formats, "task": task, "per_sample": per_sample,
           "summary": summary}
    path = os.path.join(HERE, "results_learn.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("Wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
