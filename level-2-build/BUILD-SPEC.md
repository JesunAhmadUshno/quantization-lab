# BUILD-SPEC: quantization-lab Level 2, Build

Written before the Build code, per the task order for this cycle.
Date: 2026-11-05. Status: this spec governs the Build shipped in
this cycle. Anything this spec names as a substitution is a
substitution, stated plainly, not a silent replacement.

## What SPEC.md asks Build to do

SPEC v1 core: run INT8 and INT4 models, produce GGUF builds and
run them locally with llama.cpp. Added track: prune or quantize
one TensorFlow model with the TensorFlow Model Optimization
Toolkit and add its row to the study table. Build closes when
the system runs end to end, tests pass, and one benchmark table
with method is in the README.

## Verified environment for this cycle (2026-11-05, Muse VM)

- numpy 1.26.4 is available. Python 3.12.
- import tensorflow fails: ModuleNotFoundError. TensorFlow is
  not available on this VM.
- import torch fails: ModuleNotFoundError.
- llama-cli is not on PATH (which llama-cli returns nothing).
  llama.cpp is not available on this VM.
- BlackBox, the assigned GPU fleet node, is unreachable this
  cycle: NEEDS-BLACKBOX-ON (coordinator verified SSH closed, no
  output). No GPU run is claimed.

## Substitutions, named plainly

1. No GGUF, no llama.cpp. The serialized format in this Build is
   QLAB, a small binary container format designed and documented
   in build_system.py for this project only. QLAB is not GGUF,
   is not compatible with GGUF or llama.cpp, and no file produced
   here can be run by llama.cpp. A real GGUF writer was not
   attempted because it could not be validated here: there is no
   llama.cpp binary on this VM to parse or run its output, and an
   unvalidated writer would be a claim without evidence.
2. No real pretrained model. The model is a 2-layer MLP trained
   in this repo with NumPy on a larger synthetic dataset (see
   below). It is a real trained model in the narrow sense that
   its weights come from gradient descent on labelled data and
   its test accuracy is measured, not set by construction. It is
   not a pretrained or real-world model.
3. TensorFlow track substitution. The TensorFlow Model
   Optimization Toolkit row did not run, because TensorFlow is
   unavailable (verified above). In its place, the same model is
   magnitude pruned with NumPy at named sparsity levels, labelled
   in every table and result as the NumPy equivalent pruning
   track, never as a TensorFlow result.
4. CPU only. All timing is CPU inference time measured on this
   Muse VM, single machine, varying load. No GPU timing, no
   llama.cpp timing, and no speed claim beyond this VM is made.
5. Fake quantization for quality numbers. Accuracy and error for
   quantized variants are measured with dequantized weights in a
   NumPy forward pass (fake quantization), which is the standard
   way to measure quantization quality without a quantized
   kernel. Size numbers are real serialized QLAB file bytes.
   Timing for quantized variants is the dequantized forward pass
   timing and is labelled as such; it does not show the speed a
   real INT8 or INT4 kernel would give.

## The Build system

- Model: 2-layer MLP, input 16, hidden 32 with ReLU, output 4.
  Weights W1 (16 by 32), b1 (32), W2 (32 by 4), b2 (4). Total
  676 parameters. Initialization and training are deterministic:
  NumPy RandomState seeds fixed in code, full-batch gradient
  descent, 60 epochs, learning rate 0.1, no shuffling.
- Data: synthetic 4-class Gaussian mixture, input dimension 16.
  Class centers from RandomState(7), scale 0.6; sample noise
  standard deviation 1.0. Splits: train 800, calibration 200,
  held-out test 400, each from its own fixed seed, labels
  balanced by construction then shuffled with the split seed.
  Labels are the generating class. The classes overlap by
  design, so FP32 accuracy is below 1.0 and quantization has
  room to hurt it.
- Reuse of Learn: level-2-build/build_system.py imports
  quantize_symmetric, pack_int4, unpack_int4, and
  round_half_away from level-1-learn/quantize.py. The per-tensor
  INT8 variant is exactly the Learn method applied per weight
  matrix. Reuse is by import, stated here and in the code
  header, not a silent copy. Learn files are unchanged.
- Quantization variants:
  - FP32 baseline.
  - INT8 per-tensor: Learn symmetric method, one scale per
    weight matrix, no calibration.
  - INT8 per-channel: one scale per output column of each
    weight matrix, with calibration (below).
  - INT4 per-channel, packed: one scale per output column,
    codes packed two per byte with the Learn pack_int4, with
    calibration (below).
  - Biases are stored in FP32 in every variant, which is common
    practice and is counted in every size.
- Calibration: the calibration split (200 samples, never used
  for training or for the reported test accuracy) selects the
  clipping percentile for each weight matrix. Candidates are
  100, 99.9, 99.5, 99.0, and 98.0 percent of the absolute
  weights. For each candidate the matrix is clipped, quantized,
  dequantized, and run through the full model on the
  calibration inputs; the candidate with the lowest mean
  squared logit error against the FP32 model wins. The chosen
  percentile per matrix is recorded in results_build.json.
- QLAB container format (version 1): bytes 0 to 3 are the ASCII
  magic QLAB; bytes 4 to 5 are the version as an unsigned 16-bit
  little-endian integer; bytes 6 to 9 are the header length as
  an unsigned 32-bit little-endian integer; then a UTF-8 JSON
  header with sorted keys listing each entry (name, kind, shape,
  dtype, byte length) in payload order; then the payloads in
  header order. Kinds: fp32 (little-endian float32), int8
  (signed bytes), int4_packed (Learn packing, first code in the
  low nibble), scales (little-endian float32). For per-channel
  matrices, codes are stored column by column (column-major), so
  each column has one contiguous block and one scale. Save and
  load are implemented in build_system.py and covered by a
  round-trip test.
- Pruning track (NumPy equivalent, see substitution 3):
  magnitude pruning zeroes the smallest absolute weights in
  each weight matrix at sparsity 0.3, 0.5, and 0.7. Pruned
  models are stored as FP32 in QLAB, so their serialized size
  equals the FP32 size; that fact is reported, not hidden,
  because dense storage does not shrink from pruning alone.
- Benchmark table: one row per variant (FP32, INT8 per-tensor,
  INT8 per-channel, INT4 per-channel packed, pruned 0.3, pruned
  0.5, pruned 0.7) with serialized size in bytes, maximum and
  mean absolute weight error against FP32, held-out test
  accuracy, mean squared logit error against FP32 on the test
  set, and CPU inference time (median of 15 full test-set
  forward passes on this VM, milliseconds).
- Runner: level-2-build/run_build.py runs training, calibration,
  quantization, pruning, serialization, and evaluation end to
  end and writes level-2-build/results_build.json. The results
  file is deterministic except the named timing fields
  (timing_ms_median per row and the timing note); two runs must
  produce identical files once timing fields are removed.
- Tests: tests/test_build_system.py with hand-computed
  expectations for per-channel scales and codes, INT4 packing,
  the QLAB save and load round-trip, the pruning mask, and
  calibration candidate selection on a small case, plus the
  existing Learn tests and the scaffold smoke test. The full
  suite must pass.

## Acceptance for this Build

- Runner runs end to end on this VM and writes results.
- Full test suite passes.
- Benchmark table with measured values and method is in
  level-2-build/README.md and the root README.md.
- Every substitution above is named in the level README.
- Em dash count is 0 in all touched files.
