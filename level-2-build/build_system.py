"""Level 2 Build system for quantization-lab (P11).

A 2-layer MLP (16 inputs, 32 hidden, 4 outputs, 676 parameters)
trained with NumPy on a synthetic Gaussian mixture dataset, then
quantized (INT8 per-tensor, INT8 per-channel, INT4 per-channel
packed), magnitude pruned, serialized in the QLAB container
format defined here, and benchmarked end to end on CPU.

Reuse of Learn, by import and not by copy: quantize_symmetric,
pack_int4, unpack_int4, and round_half_away come from
level-1-learn/quantize.py. The per-tensor INT8 variant is the
Learn method applied per weight matrix.

Honest limits, also in BUILD-SPEC.md: QLAB is a format designed
for this project. It is not GGUF and no file here runs in
llama.cpp. There is no TensorFlow in this Build: the pruning
track is a NumPy equivalent, labelled as such. Quality numbers
use dequantized weights in a NumPy forward pass (fake
quantization). Timing is CPU timing on the machine that runs
this code.
"""
import json
import math
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LEARN_DIR = os.path.join(os.path.dirname(HERE), "level-1-learn")
if LEARN_DIR not in sys.path:
    sys.path.insert(0, LEARN_DIR)
from quantize import pack_int4, quantize_symmetric, round_half_away, unpack_int4  # noqa: E402

INPUT_DIM = 16
HIDDEN_DIM = 32
NUM_CLASSES = 4
CENTER_SEED = 7
CENTER_SCALE = 0.6
NOISE_STD = 1.0
INIT_SEED = 42
TRAIN_SEED = 101
CALIB_SEED = 202
TEST_SEED = 303
TRAIN_N = 800
CALIB_N = 200
TEST_N = 400
EPOCHS = 60
LEARNING_RATE = 0.1
CALIB_CANDIDATES = [100.0, 99.9, 99.5, 99.0, 98.0]
PRUNE_SPARSITIES = [0.3, 0.5, 0.7]
WEIGHT_KEYS = ["W1", "W2"]
ALL_KEYS = ["W1", "b1", "W2", "b2"]


class BuildError(Exception):
    """Named error for invalid Build inputs."""


# ---------------------------------------------------------------- data

def class_centers():
    return np.random.RandomState(CENTER_SEED).normal(
        0.0, CENTER_SCALE, size=(NUM_CLASSES, INPUT_DIM))


def make_split(seed, n):
    """One balanced synthetic split. Labels are the generating class."""
    rng = np.random.RandomState(seed)
    labels = np.array([i % NUM_CLASSES for i in range(n)])
    rng.shuffle(labels)
    centers = class_centers()
    X = centers[labels] + rng.normal(0.0, NOISE_STD, size=(n, INPUT_DIM))
    return X, labels


def build_datasets():
    return {"train": make_split(TRAIN_SEED, TRAIN_N),
            "calibration": make_split(CALIB_SEED, CALIB_N),
            "test": make_split(TEST_SEED, TEST_N)}


# ---------------------------------------------------------------- model

def init_model():
    rng = np.random.RandomState(INIT_SEED)
    return {"W1": rng.normal(0.0, math.sqrt(2.0 / INPUT_DIM),
                             size=(INPUT_DIM, HIDDEN_DIM)),
            "b1": np.zeros(HIDDEN_DIM),
            "W2": rng.normal(0.0, math.sqrt(2.0 / HIDDEN_DIM),
                             size=(HIDDEN_DIM, NUM_CLASSES)),
            "b2": np.zeros(NUM_CLASSES)}


def forward(model, X):
    hidden = np.maximum(0.0, X @ model["W1"] + model["b1"])
    return hidden @ model["W2"] + model["b2"]


def train_model(model, X, y, epochs=EPOCHS, lr=LEARNING_RATE):
    """Full-batch gradient descent on softmax cross entropy. Deterministic."""
    n = len(X)
    for _ in range(epochs):
        hidden_pre = X @ model["W1"] + model["b1"]
        hidden = np.maximum(0.0, hidden_pre)
        logits = hidden @ model["W2"] + model["b2"]
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        dlogits = probs.copy()
        dlogits[np.arange(n), y] -= 1.0
        dlogits /= n
        dW2 = hidden.T @ dlogits
        db2 = dlogits.sum(axis=0)
        dhidden = dlogits @ model["W2"].T
        dhidden[hidden_pre <= 0] = 0.0
        dW1 = X.T @ dhidden
        db1 = dhidden.sum(axis=0)
        model["W1"] -= lr * dW1
        model["b1"] -= lr * db1
        model["W2"] -= lr * dW2
        model["b2"] -= lr * db2
    return model


def accuracy(model, X, y):
    return float((forward(model, X).argmax(axis=1) == y).mean())


def logit_mse(model_a, model_b, X):
    diff = forward(model_a, X) - forward(model_b, X)
    return float((diff ** 2).mean())


def weight_error(reference, candidate):
    """Max and mean absolute error over both weight matrices."""
    errors = []
    for key in WEIGHT_KEYS:
        errors.append(np.abs(reference[key] - candidate[key]).ravel())
    flat = np.concatenate(errors)
    return {"max_abs_error": float(flat.max()),
            "mean_abs_error": float(flat.mean())}


# ---------------------------------------------------------------- quantization

def _qmax(bits):
    if bits not in (4, 8):
        raise BuildError("bits must be 4 or 8")
    return 2 ** (bits - 1) - 1


def quantize_per_tensor_matrix(matrix, bits):
    """Learn symmetric per-tensor quantization of one matrix."""
    flat = [float(v) for v in np.asarray(matrix, dtype=np.float64).ravel()]
    q = quantize_symmetric(flat, bits)
    codes = np.array(q["codes"], dtype=np.int64).reshape(np.shape(matrix))
    dequant = np.array(q["dequantized"], dtype=np.float64).reshape(np.shape(matrix))
    return {"codes": codes, "scales": np.array([q["scale"]]),
            "dequantized": dequant, "bits": bits, "mode": "per_tensor"}


def quantize_per_channel_matrix(matrix, bits, clip_percentile=100.0):
    """Symmetric per-channel quantization, one scale per output column.

    Hand check: for [[1.0, -0.5], [0.25, 2.0]] at INT8 with no
    clipping, column 0 has max abs 1.0 so its scale is 1/127 and
    its codes are 127 and 32; column 1 has max abs 2.0 so its
    scale is 2/127 and its codes are -32 and 127.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise BuildError("per-channel quantization needs a 2D matrix")
    qmax = _qmax(bits)
    clipped = matrix.copy()
    if clip_percentile < 100.0:
        limit = float(np.percentile(np.abs(matrix), clip_percentile))
        clipped = np.clip(matrix, -limit, limit)
    scales = []
    codes = np.zeros(matrix.shape, dtype=np.int64)
    dequant = np.zeros(matrix.shape, dtype=np.float64)
    for col in range(matrix.shape[1]):
        column = clipped[:, col]
        max_abs = float(np.abs(column).max()) if column.size else 0.0
        scale = max_abs / qmax if max_abs > 0 else 1.0
        scales.append(scale)
        for row in range(matrix.shape[0]):
            code = round_half_away(float(column[row]) / scale)
            code = max(-qmax, min(qmax, code))
            codes[row, col] = code
            dequant[row, col] = code * scale
    return {"codes": codes, "scales": np.array(scales),
            "dequantized": dequant, "bits": bits, "mode": "per_channel",
            "clip_percentile": clip_percentile}


def calibrate_matrix(model, key, bits, mode, X_calib):
    """Pick the clipping percentile for one matrix by calibration MSE.

    Each candidate clips, quantizes, and dequantizes only the
    named matrix; the rest of the model stays FP32. The winner
    has the lowest mean squared logit error against the FP32
    model on the calibration inputs.
    """
    best = None
    for pct in CALIB_CANDIDATES:
        if mode == "per_channel":
            q = quantize_per_channel_matrix(model[key], bits, pct)
        else:
            raise BuildError("calibration is defined for per_channel mode")
        trial = dict(model)
        trial[key] = q["dequantized"]
        mse = logit_mse(model, trial, X_calib)
        if best is None or mse < best["calibration_logit_mse"]:
            best = {"quantized": q, "clip_percentile": pct,
                    "calibration_logit_mse": mse}
    return best


def calibrated_model(model, bits, mode, X_calib):
    """Quantize both weight matrices with per-matrix calibration."""
    quantized = {}
    chosen = {}
    dequant_model = {"b1": model["b1"].copy(), "b2": model["b2"].copy()}
    for key in WEIGHT_KEYS:
        best = calibrate_matrix(model, key, bits, mode, X_calib)
        quantized[key] = best["quantized"]
        chosen[key] = {"clip_percentile": best["clip_percentile"],
                       "calibration_logit_mse": best["calibration_logit_mse"]}
        dequant_model[key] = best["quantized"]["dequantized"]
    return quantized, dequant_model, chosen


def per_tensor_model(model, bits):
    quantized = {}
    dequant_model = {"b1": model["b1"].copy(), "b2": model["b2"].copy()}
    for key in WEIGHT_KEYS:
        q = quantize_per_tensor_matrix(model[key], bits)
        quantized[key] = q
        dequant_model[key] = q["dequantized"]
    return quantized, dequant_model


# ---------------------------------------------------------------- pruning

def pruning_mask(matrix, sparsity):
    """Boolean keep mask: True where the weight survives.

    The smallest absolute weights are pruned. Ties at the cutoff
    are broken by flat index order, so the count pruned is exact:
    floor(sparsity times count) entries are False. Hand check:
    [0.1, -0.5, 0.05, 0.9] at sparsity 0.5 prunes 0.05 and 0.1,
    mask [False, True, False, True].
    """
    if not 0.0 <= sparsity < 1.0:
        raise BuildError("sparsity must be in [0, 1)")
    flat_abs = np.abs(np.asarray(matrix, dtype=np.float64)).ravel()
    count = int(math.floor(sparsity * flat_abs.size))
    order = np.argsort(flat_abs, kind="stable")
    mask = np.ones(flat_abs.shape, dtype=bool)
    mask[order[:count]] = False
    return mask.reshape(np.shape(matrix))


def prune_matrix(matrix, sparsity):
    return np.asarray(matrix, dtype=np.float64) * pruning_mask(matrix, sparsity)


def pruned_model(model, sparsity):
    out = {"b1": model["b1"].copy(), "b2": model["b2"].copy()}
    for key in WEIGHT_KEYS:
        out[key] = prune_matrix(model[key], sparsity)
    return out


def nonzero_count(model):
    return int(sum(int((model[key] != 0).sum()) for key in ALL_KEYS))


# ---------------------------------------------------------------- QLAB container

QLAB_MAGIC = b"QLAB"
QLAB_VERSION = 1


def _entry_bytes(kind, array):
    if kind == "fp32":
        return np.asarray(array, dtype="<f4").tobytes()
    if kind == "scales":
        return np.asarray(array, dtype="<f4").tobytes()
    if kind == "int8":
        return np.asarray(array, dtype=np.int8).tobytes()
    if kind == "int4_packed":
        return bytes(array)
    raise BuildError("unknown QLAB entry kind: " + str(kind))


def save_qlab(path, entries):
    """Write a QLAB file. Entries are (name, kind, shape, array) tuples.

    For int4_packed, array is the packed byte list and shape is
    the shape of the unpacked codes matrix. For per-channel
    matrices, callers pass codes already flattened column by
    column, and load_qlab returns that flat order; the variant
    helpers below own the reshape.
    """
    header_entries = []
    payloads = []
    for name, kind, shape, array in entries:
        raw = _entry_bytes(kind, array)
        header_entries.append({"name": name, "kind": kind,
                               "shape": list(shape), "nbytes": len(raw)})
        payloads.append(raw)
    header = {"format": "QLAB", "version": QLAB_VERSION,
              "entries": header_entries}
    header_bytes = json.dumps(header, sort_keys=True).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(QLAB_MAGIC)
        fh.write(struct.pack("<H", QLAB_VERSION))
        fh.write(struct.pack("<I", len(header_bytes)))
        fh.write(header_bytes)
        for raw in payloads:
            fh.write(raw)
    return os.path.getsize(path)


def load_qlab(path):
    """Read a QLAB file back into (header, {name: array or bytes})."""
    with open(path, "rb") as fh:
        raw = fh.read()
    if raw[:4] != QLAB_MAGIC:
        raise BuildError("not a QLAB file: bad magic")
    version = struct.unpack("<H", raw[4:6])[0]
    if version != QLAB_VERSION:
        raise BuildError("unsupported QLAB version: " + str(version))
    header_len = struct.unpack("<I", raw[6:10])[0]
    header = json.loads(raw[10:10 + header_len].decode("utf-8"))
    offset = 10 + header_len
    out = {}
    for entry in header["entries"]:
        chunk = raw[offset:offset + entry["nbytes"]]
        offset += entry["nbytes"]
        if entry["kind"] in ("fp32", "scales"):
            out[entry["name"]] = np.frombuffer(chunk, dtype="<f4").astype(np.float64)
        elif entry["kind"] == "int8":
            out[entry["name"]] = np.frombuffer(chunk, dtype=np.int8).astype(np.int64)
        elif entry["kind"] == "int4_packed":
            out[entry["name"]] = list(chunk)
        else:
            raise BuildError("unknown QLAB entry kind: " + str(entry["kind"]))
    if offset != len(raw):
        raise BuildError("QLAB file has trailing bytes")
    return header, out


def _column_major(matrix):
    return np.asarray(matrix).T.ravel()


def _from_column_major(flat, shape):
    return np.asarray(flat).reshape((shape[1], shape[0])).T


def variant_entries(model, quantized, variant):
    """QLAB entries for one variant. Biases are FP32 in every variant."""
    entries = []
    if variant == "fp32" or variant.startswith("pruned"):
        source = model if variant == "fp32" else quantized
        for key in ALL_KEYS:
            entries.append((key, "fp32", list(source[key].shape),
                            np.asarray(source[key], dtype=np.float64).ravel()))
        return entries
    for key in WEIGHT_KEYS:
        q = quantized[key]
        shape = list(q["codes"].shape)
        if q["mode"] == "per_tensor":
            flat_codes = q["codes"].ravel()
            if q["bits"] == 8:
                entries.append((key + "_codes", "int8", shape, flat_codes))
            else:
                entries.append((key + "_codes", "int4_packed", shape,
                                pack_int4([int(c) for c in flat_codes])))
        else:
            flat_codes = _column_major(q["codes"])
            if q["bits"] == 8:
                entries.append((key + "_codes", "int8", shape, flat_codes))
            else:
                entries.append((key + "_codes", "int4_packed", shape,
                                pack_int4([int(c) for c in flat_codes])))
        entries.append((key + "_scales", "scales", list(q["scales"].shape),
                        q["scales"]))
    for key in ("b1", "b2"):
        entries.append((key, "fp32", list(model[key].shape),
                        np.asarray(model[key], dtype=np.float64).ravel()))
    return entries


def model_from_qlab(header, data, variant):
    """Rebuild a runnable (dequantized) model from loaded QLAB data."""
    shapes = {e["name"]: e["shape"] for e in header["entries"]}
    model = {}
    if variant == "fp32" or variant.startswith("pruned"):
        for key in ALL_KEYS:
            model[key] = data[key].reshape(shapes[key])
        return model
    for key in ("b1", "b2"):
        model[key] = data[key].reshape(shapes[key])
    for key in WEIGHT_KEYS:
        shape = shapes[key + "_codes"]
        scales = data[key + "_scales"]
        raw = data[key + "_codes"]
        kinds = {e["name"]: e["kind"] for e in header["entries"]}
        count = int(np.prod(shape))
        if kinds[key + "_codes"] == "int4_packed":
            codes_flat = unpack_int4(list(raw), count)
        else:
            codes_flat = [int(c) for c in raw]
        if len(scales) == 1:
            codes = np.array(codes_flat, dtype=np.int64).reshape(shape)
            model[key] = codes * scales[0]
        else:
            codes = _from_column_major(np.array(codes_flat, dtype=np.int64), shape)
            model[key] = codes * scales.reshape((1, shape[1]))
    return model
