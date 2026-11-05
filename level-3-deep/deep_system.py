"""Level 3 Deep system for quantization-lab (P11).

The quality against size against speed study on the Build MLP.
Reuses level-2-build/build_system.py by import for the model,
data, quantization, pruning, calibration, and the QLAB
container. Adds three combined prune plus quantize routes that
Build did not measure, a Pareto analysis, a decision rule, and
SVG charts generated from measured rows only.

Honest limits, also in DEEP-SPEC.md: QLAB is a format designed
for this project, not GGUF, and no file here runs in
llama.cpp. Pruning is the Build NumPy equivalent track, not a
TensorFlow result. Quality and timing use dequantized weights
in a NumPy forward pass on CPU (fake quantization), so timing
does not show a real INT8 or INT4 kernel speed.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(os.path.dirname(HERE), "level-2-build")
if BUILD_DIR not in sys.path:
    sys.path.insert(0, BUILD_DIR)
import build_system as bs  # noqa: E402


class DeepError(Exception):
    """Named error for invalid Deep inputs."""


COMBINED_ROUTES = [
    {"name": "pruned_0.3_int8_per_channel", "sparsity": 0.3, "bits": 8},
    {"name": "pruned_0.5_int8_per_channel", "sparsity": 0.5, "bits": 8},
    {"name": "pruned_0.5_int4_per_channel_packed", "sparsity": 0.5,
     "bits": 4},
]


def combined_quantized(pruned, bits, X_calib):
    """Calibrate and quantize a pruned model per-channel.

    Pruned zeros stay zero after quantization: a zero weight
    divided by any positive scale rounds to code 0, and code 0
    times the scale is zero. The tests check this property.
    """
    return bs.calibrated_model(pruned, bits, "per_channel", X_calib)


def accuracy_per_kb(row):
    """Test accuracy divided by serialized size in kilobytes."""
    if row["size_bytes"] <= 0:
        raise DeepError("size_bytes must be positive")
    return float(row["test_accuracy"]) / (float(row["size_bytes"]) / 1000.0)


def pareto_flags(rows):
    """Pareto flags on the two deterministic axes (size, accuracy).

    A route is Pareto optimal when no other route has size at
    most its size and accuracy at least its accuracy, with at
    least one of the two strict. Speed is not a Pareto axis
    here, by design: batch timing on dequantized weights is
    noise at this scale (the same routes measured 0.19 to
    0.32 ms across runs in this cycle), so a timing axis made
    the flags flip between runs. Rows need variant,
    size_bytes, and test_accuracy. Hand check: rows A (100
    bytes, 0.90), B (50 bytes, 0.90), C (200 bytes, 0.95):
    B dominates A, so A is False; B and C are True.
    """
    flags = {}
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            at_least_as_good = (
                other["size_bytes"] <= row["size_bytes"]
                and other["test_accuracy"] >= row["test_accuracy"])
            strictly_better = (
                other["size_bytes"] < row["size_bytes"]
                or other["test_accuracy"] > row["test_accuracy"])
            if at_least_as_good and strictly_better:
                dominated = True
                break
        flags[row["variant"]] = not dominated
    return flags


def recommend_route(rows, max_size_bytes=None, min_accuracy=None,
                    max_batch_ms=None):
    """The decision rule, as code. See DEEP-SPEC.md.

    Filters routes meeting every given constraint, then ranks
    survivors by accuracy (highest), logit MSE (lowest), size
    (smallest), batch time (lowest). Returns the winning row,
    or None when no route meets the constraints.
    """
    survivors = []
    for row in rows:
        if max_size_bytes is not None and row["size_bytes"] > max_size_bytes:
            continue
        if min_accuracy is not None and row["test_accuracy"] < min_accuracy:
            continue
        if (max_batch_ms is not None
                and row["timing_batch_ms_median"] > max_batch_ms):
            continue
        survivors.append(row)
    if not survivors:
        return None
    return sorted(
        survivors,
        key=lambda r: (-r["test_accuracy"], r["logit_mse_vs_fp32"],
                       r["size_bytes"], r["timing_batch_ms_median"]))[0]


def _scale(value, low, high, out_low, out_high):
    if high == low:
        return (out_low + out_high) / 2.0
    frac = (float(value) - float(low)) / (float(high) - float(low))
    return out_low + frac * (out_high - out_low)


def svg_scatter(rows, x_key, y_key, title, x_label, y_label, width=720,
                height=460):
    """One labelled SVG scatter chart built from the given rows.

    Every point is one row, labelled with its variant name.
    Axis ranges come from the data with a 10 percent pad. The
    output is deterministic for fixed rows: point order follows
    row order and all coordinates are rounded to one decimal.
    """
    if not rows:
        raise DeepError("charts need at least one row")
    margin_left, margin_right = 70, 170
    margin_top, margin_bottom = 44, 56
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    xs = [float(r[x_key]) for r in rows]
    ys = [float(r[y_key]) for r in rows]
    x_low, x_high = min(xs), max(xs)
    y_low, y_high = min(ys), max(ys)
    x_pad = (x_high - x_low) * 0.10 if x_high > x_low else 1.0
    y_pad = (y_high - y_low) * 0.10 if y_high > y_low else 0.01
    x_low, x_high = x_low - x_pad, x_high + x_pad
    y_low, y_high = y_low - y_pad, y_high + y_pad
    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">')
    parts.append(f"<title>{title}</title>")
    parts.append(
        f'<rect x="0" y="0" width="{width}" height="{height}" '
        f'fill="white"/>')
    parts.append(
        f'<text x="{width // 2}" y="24" text-anchor="middle" '
        f'font-family="monospace" font-size="14">{title}</text>')
    parts.append(
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" '
        f'height="{plot_h}" fill="none" stroke="black"/>')
    parts.append(
        f'<text x="{margin_left + plot_w // 2}" '
        f'y="{height - 12}" text-anchor="middle" '
        f'font-family="monospace" font-size="12">{x_label}</text>')
    parts.append(
        f'<text x="16" y="{margin_top + plot_h // 2}" '
        f'text-anchor="middle" font-family="monospace" font-size="12" '
        f'transform="rotate(-90 16 {margin_top + plot_h // 2})">'
        f'{y_label}</text>')
    parts.append(
        f'<text x="{margin_left}" y="{margin_top + plot_h + 18}" '
        f'font-family="monospace" font-size="10">'
        f'{x_low:.4g}</text>')
    parts.append(
        f'<text x="{margin_left + plot_w}" y="{margin_top + plot_h + 18}" '
        f'text-anchor="end" font-family="monospace" font-size="10">'
        f'{x_high:.4g}</text>')
    parts.append(
        f'<text x="{margin_left - 6}" y="{margin_top + plot_h}" '
        f'text-anchor="end" font-family="monospace" font-size="10">'
        f'{y_low:.4g}</text>')
    parts.append(
        f'<text x="{margin_left - 6}" y="{margin_top + 8}" '
        f'text-anchor="end" font-family="monospace" font-size="10">'
        f'{y_high:.4g}</text>')
    for row in rows:
        cx = _scale(row[x_key], x_low, x_high, margin_left,
                    margin_left + plot_w)
        cy = _scale(row[y_key], y_low, y_high, margin_top + plot_h,
                    margin_top)
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="#b00020"/>')
        parts.append(
            f'<text x="{cx + 8:.1f}" y="{cy + 4:.1f}" '
            f'font-family="monospace" font-size="10">'
            f'{row["variant"]}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def build_charts(rows):
    """The three Deep charts, as a {filename: svg text} dict."""
    return {
        "accuracy_vs_size.svg": svg_scatter(
            rows, "size_bytes", "test_accuracy",
            "Test accuracy vs serialized size (P11 Deep, measured)",
            "serialized QLAB size, bytes", "test accuracy"),
        "logit_mse_vs_size.svg": svg_scatter(
            rows, "size_bytes", "logit_mse_vs_fp32",
            "Logit MSE vs FP32 vs serialized size (P11 Deep, measured)",
            "serialized QLAB size, bytes", "logit MSE vs FP32"),
        "batch_time_vs_accuracy.svg": svg_scatter(
            rows, "timing_batch_ms_median", "test_accuracy",
            "Test accuracy vs batch time (P11 Deep, measured)",
            "batch time, median ms for 400 samples", "test accuracy"),
    }
