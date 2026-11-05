"""Symmetric per-tensor quantization, Level 1 Learn.

Quantizes one small deterministic 4 by 3 linear model to symmetric
INT8 and symmetric INT4, dequantizes it, measures size and error, and
runs a small synthetic labelled task before and after quantization.

This is teaching work only: no real model, no GGUF, no llama.cpp,
and no GPU. Real models and those formats are Build and Deep scope.

Only the Python standard library is used.
"""
import math


class QuantizationError(Exception):
    """Named error for invalid quantization inputs."""


def round_half_away(value):
    """Round to the nearest integer, halves away from zero.

    This rule is used instead of the built in round so the hand
    checks do not depend on bankers rounding: 3.5 goes to 4 and
    -3.5 goes to -4.
    """
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def validate_values(values):
    if not isinstance(values, list) or not values:
        raise QuantizationError("values must be a non-empty list")
    for v in values:
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise QuantizationError("every value must be a number")
        if not math.isfinite(float(v)):
            raise QuantizationError("every value must be finite")
    return values


def quantize_symmetric(values, bits):
    """Symmetric per-tensor quantization for bits in (4, 8).

    qmax is 127 for INT8 and 7 for INT4. The INT4 range is clamped
    to -7 to 7, not -8 to 7, so the scale stays symmetric.
    scale is max absolute value divided by qmax, or 1.0 when every
    value is zero. Codes are round half away from zero of the value
    divided by the scale, clamped to -qmax to qmax.
    """
    validate_values(values)
    if bits not in (4, 8):
        raise QuantizationError("bits must be 4 or 8 at this level")
    qmax = 2 ** (bits - 1) - 1
    max_abs = max(abs(float(v)) for v in values)
    scale = max_abs / qmax if max_abs > 0 else 1.0
    codes = []
    for v in values:
        code = round_half_away(float(v) / scale)
        codes.append(max(-qmax, min(qmax, code)))
    dequantized = [c * scale for c in codes]
    return {"bits": bits, "qmax": qmax, "scale": scale,
            "codes": codes, "dequantized": dequantized}


def error_metrics(original, dequantized):
    """Max and mean absolute error and mean squared error."""
    if len(original) != len(dequantized) or not original:
        raise QuantizationError("error inputs must have equal non-zero length")
    errors = [abs(float(a) - float(b)) for a, b in zip(original, dequantized)]
    squared = [(float(a) - float(b)) ** 2 for a, b in zip(original, dequantized)]
    return {"max_abs_error": max(errors),
            "mean_abs_error": sum(errors) / len(errors),
            "mse": sum(squared) / len(squared),
            "total_abs_error": sum(errors)}


def size_bytes(count, bits):
    """Payload, scale, and total bytes for count values at bits.

    FP32 (bits 32) has no scale. Quantized formats store one FP32
    scale, 4 bytes. INT4 packs two codes per byte; an odd count
    wastes half of the last byte, which the ceiling counts.
    """
    if not isinstance(count, int) or count < 1:
        raise QuantizationError("count must be a positive integer")
    if bits == 32:
        payload = count * 4
        scale = 0
    elif bits == 8:
        payload = count
        scale = 4
    elif bits == 4:
        payload = (count * 4 + 7) // 8
        scale = 4
    else:
        raise QuantizationError("bits must be 4, 8, or 32 for sizes")
    return {"payload_bytes": payload, "scale_bytes": scale,
            "total_bytes": payload + scale}


def pack_int4(codes):
    """Pack signed INT4 codes, two per byte, first code low.

    Each code is stored as its 4 bit two complement pattern.
    Codes must lie in -8 to 7 for packing; the symmetric quantizer
    in this level only produces -7 to 7.
    """
    for c in codes:
        if not isinstance(c, int) or c < -8 or c > 7:
            raise QuantizationError("INT4 codes for packing must be in -8 to 7")
    packed = []
    for i in range(0, len(codes), 2):
        low = codes[i] & 0xF
        high = (codes[i + 1] & 0xF) if i + 1 < len(codes) else 0
        packed.append(low | (high << 4))
    return packed


def unpack_int4(packed, count):
    """Unpack count signed INT4 codes packed by pack_int4."""
    if count < 1:
        raise QuantizationError("count must be a positive integer")
    codes = []
    for byte in packed:
        if not isinstance(byte, int) or byte < 0 or byte > 255:
            raise QuantizationError("packed bytes must be integers in 0 to 255")
        for nibble in (byte & 0xF, (byte >> 4) & 0xF):
            codes.append(nibble - 16 if nibble >= 8 else nibble)
    return codes[:count]


def linear_scores(matrix, inputs):
    """Scores of one linear model: one dot product per matrix row."""
    scores = []
    for row in matrix:
        if len(row) != len(inputs):
            raise QuantizationError("input length must match the matrix row length")
        scores.append(sum(float(w) * float(x) for w, x in zip(row, inputs)))
    return scores


def argmax_first(scores):
    """Index of the largest score; the first index wins a tie."""
    if not scores:
        raise QuantizationError("scores must be non-empty")
    best = 0
    for i in range(1, len(scores)):
        if scores[i] > scores[best]:
            best = i
    return best


def reshape(flat, rows, cols):
    if len(flat) != rows * cols:
        raise QuantizationError("flat length must equal rows times cols")
    return [list(flat[r * cols:(r + 1) * cols]) for r in range(rows)]


def build_weight_matrix():
    """The deterministic model: 4 classes, 3 inputs, 12 weights.

    Each row is a signed direction specialist. The largest absolute
    weight is 1.0, so the INT8 scale is 1/127 and the INT4 scale is
    1/7. Every other weight is a simple fraction of 1.0, which makes
    the codes and errors hand-checkable (see the level README).
    """
    return [
        [1.0, -0.5, 0.25],
        [-0.5, 1.0, -0.25],
        [0.25, -0.75, 1.0],
        [-0.75, -0.25, -1.0],
    ]


def flatten(matrix):
    return [v for row in matrix for v in row]


def build_task_samples():
    """Eight synthetic labelled samples. Labels are the FP32 argmax.

    Hand-computed FP32 scores are in the level README and the tests.
    Sample 7 is the closest call: class 3 scores 1.75 against
    class 1 at 1.50, a margin of 0.25.
    """
    return [
        {"input": [1, 0, 0], "label": 0},
        {"input": [0, 1, 0], "label": 1},
        {"input": [0, 0, 1], "label": 2},
        {"input": [0, 0, -1], "label": 3},
        {"input": [2, 1, 0], "label": 0},
        {"input": [1, 2, 0], "label": 1},
        {"input": [0, 1, -2], "label": 3},
        {"input": [1, 0, -1], "label": 0},
    ]


def evaluate_task(matrix, samples):
    """Accuracy and mean absolute score error against the FP32 model.

    The reference is the FP32 weight matrix passed as ref_matrix
    through evaluate_all; this helper scores one matrix and returns
    its predictions and accuracy against the sample labels.
    """
    predictions = []
    correct = 0
    for sample in samples:
        pred = argmax_first(linear_scores(matrix, sample["input"]))
        predictions.append(pred)
        if pred == sample["label"]:
            correct += 1
    return {"predictions": predictions, "correct": correct,
            "total": len(samples),
            "accuracy": correct / len(samples) if samples else 0.0}


def score_mae(matrix_a, matrix_b, samples):
    """Mean absolute difference between two models scores on samples."""
    total = 0.0
    count = 0
    for sample in samples:
        a = linear_scores(matrix_a, sample["input"])
        b = linear_scores(matrix_b, sample["input"])
        for x, y in zip(a, b):
            total += abs(x - y)
            count += 1
    return total / count if count else 0.0
