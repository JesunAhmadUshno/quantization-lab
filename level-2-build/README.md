# Level 2, Build: quantization-lab

Status: DONE (2026-11-05). A 2-layer MLP trained with NumPy,
quantized to INT8 and INT4, magnitude pruned, serialized in a
documented binary container, and benchmarked end to end on CPU.
The governing spec for this level, written before the code, is
BUILD-SPEC.md in this folder, including every substitution
named below.

## Main result

Build result: on a synthetic 4-class task with 400 held-out samples, a NumPy-trained 2-layer MLP with 676 parameters scored 353 of 400 (0.8825) in FP32 at 3013 serialized bytes; INT8 per-tensor kept 353 of 400 at 1248 bytes and calibrated INT8 per-channel kept 353 of 400 at 1388 bytes with lower mean weight error (0.001267 against 0.002292), while calibrated INT4 per-channel packed scored 347 of 400 (0.8675) at 1081 bytes; every size is a QLAB container size (QLAB is a format designed for this project, not GGUF), every quality number was measured with dequantized weights on CPU, and the pruning rows in the benchmark table are a NumPy equivalent track, not a TensorFlow result, because TensorFlow was unavailable on the build machine.

## Honest scope line

- No GGUF and no llama.cpp. The container format is QLAB,
  designed for this project and documented in build_system.py.
  QLAB is not GGUF, is not compatible with GGUF, and no file
  produced here runs in llama.cpp. llama.cpp is not installed
  on the build machine (verified 2026-11-05), so a GGUF writer
  could not have been validated here.
- No TensorFlow result. import tensorflow fails on the build
  machine (verified 2026-11-05). The TensorFlow Model
  Optimization Toolkit row did not run. The pruning rows are a
  NumPy equivalent track on the same model, labelled as such in
  every table.
- No GPU. BlackBox, the assigned fleet node, was unreachable
  this cycle (NEEDS-BLACKBOX-ON, SSH closed). All work ran on
  the Muse VM CPU. No GPU number is claimed.
- No real pretrained model and no real data. The model is
  trained in this repo on a synthetic Gaussian mixture. Its
  weights come from gradient descent and its accuracy is
  measured, but it says nothing about real model quality.
- Fake quantization for quality and timing. Quantized quality
  uses dequantized weights in a NumPy forward pass. Timing is
  that same forward pass on CPU, so it does not show the speed
  of a real INT8 or INT4 kernel. Sizes are real: they are the
  byte counts of the serialized QLAB files.

## Method

1. Model. MLP, input 16, hidden 32 with ReLU, output 4.
   W1 (16 by 32), b1 (32), W2 (32 by 4), b2 (4), 676
   parameters. Seeded initialization (RandomState 42),
   full-batch gradient descent, 60 epochs, learning rate 0.1,
   softmax cross entropy. No shuffling, so training is
   deterministic.
2. Data. Synthetic 4-class Gaussian mixture in 16 dimensions.
   Centers from RandomState(7) at scale 0.6, sample noise
   standard deviation 1.0, labels balanced then shuffled per
   split. Train 800 (seed 101), calibration 200 (seed 202),
   held-out test 400 (seed 303). The classes overlap on
   purpose, so FP32 accuracy is 0.8825, not 1.0, and
   quantization has room to hurt it.
3. Quantization. Weight matrices only; biases stay FP32 and
   are counted in every size. INT8 per-tensor is the Learn
   method (level-1-learn/quantize.py, imported, not copied):
   one symmetric scale per matrix. INT8 per-channel uses one
   symmetric scale per output column. INT4 per-channel uses
   per-column scales with codes packed two per byte by the
   Learn pack_int4. Per-channel code order in the file is
   column by column.
4. Calibration. The calibration split selects a clipping
   percentile per weight matrix for the per-channel variants.
   Candidates 100, 99.9, 99.5, 99.0, 98.0 percent of absolute
   weights are each clipped, quantized, and run through the
   full model on the calibration inputs; the lowest mean
   squared logit error against FP32 wins. Chosen this run:
   INT8 per-channel picked 100.0 for both matrices (no
   clipping); INT4 picked 100.0 for W1 and 99.5 for W2.
   Calibration changed little here, and the grade says so.
5. Pruning (NumPy equivalent track). Magnitude pruning zeroes
   the smallest absolute weights in each matrix at sparsity
   0.3, 0.5, and 0.7, ties broken by flat index order so the
   pruned count is exact. Pruned models are stored dense in
   FP32, so their serialized size equals the FP32 size.
6. QLAB container (version 1). Magic QLAB (4 bytes), version
   (unsigned 16-bit little-endian), header length (unsigned
   32-bit little-endian), a UTF-8 JSON header with sorted keys
   listing each entry (name, kind, shape, byte length) in
   payload order, then the payloads. Kinds: fp32, int8,
   int4_packed, scales. The runner saves every variant,
   loads it back, checks the loaded model against the
   reference within a float32 rounding tolerance of 1e-6,
   and evaluates the loaded model itself.
7. Benchmark. One row per variant: serialized QLAB size in
   bytes, maximum and mean absolute weight error against
   FP32 across both weight matrices, held-out test accuracy,
   mean squared logit error against FP32 on the test set, and
   CPU inference time, the median of 15 full test-set forward
   passes on this VM (single machine, varying load).

## Run it

    python level-2-build/run_build.py
    python -m unittest discover -s tests -v
    python tests/test_smoke.py

The runner writes level-2-build/results_build.json and QLAB
files under level-2-build/artifacts/. The results file is
deterministic except the timing_ms_median fields: two runs on
2026-11-05 produced identical files once those fields were
removed (verified by comparison in this cycle).

## Results (measured 2026-11-05, Muse VM CPU, Python 3.12, NumPy 1.26.4)

| Variant | Size bytes | Size vs FP32 | Max abs weight error | Mean abs weight error | Test accuracy | Logit MSE vs FP32 | CPU time ms, median of 15 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FP32 | 3013 | 1.00x | 0.000000 | 0.000000 | 353 of 400 (0.8825) | 0.000000 | 0.258 |
| INT8 per-tensor (Learn method) | 1248 | 2.41x smaller | 0.005103 | 0.002292 | 353 of 400 (0.8825) | 0.000296 | 0.258 |
| INT8 per-channel, calibrated | 1388 | 2.17x smaller | 0.004610 | 0.001267 | 353 of 400 (0.8825) | 0.000129 | 0.195 |
| INT4 per-channel packed, calibrated | 1081 | 2.79x smaller | 0.083784 | 0.023948 | 347 of 400 (0.8675) | 0.045583 | 0.193 |
| Pruned 0.3 (NumPy equivalent) | 3013 | 1.00x | 0.133321 | 0.018083 | 349 of 400 (0.8725) | 0.099430 | 0.256 |
| Pruned 0.5 (NumPy equivalent) | 3013 | 1.00x | 0.218123 | 0.052308 | 350 of 400 (0.8750) | 0.533220 | 0.260 |
| Pruned 0.7 (NumPy equivalent) | 3013 | 1.00x | 0.341136 | 0.106046 | 319 of 400 (0.7975) | 1.955935 | 0.254 |

Notes on the table. The FP32 row error is not bit zero: it is
2.9e-08 maximum, from storing float64 weights as FP32 in the
container, and it rounds to 0.000000 at six decimals. Per-channel
INT8 is 140 bytes larger than per-tensor INT8 because it stores
36 scales instead of 2, and it buys a 45 percent lower mean
weight error and a 56 percent lower logit MSE for that price at
this small scale. Pruning at 0.5 scores one sample higher than
pruning at 0.3 (350 against 349 of 400): at this size that is
noise, not an improvement claim, and no fine-tuning after
pruning was done. Timing separates nothing here, because every
variant runs the same dequantized NumPy forward pass; the
timing column is reported for completeness and supports no
speed claim. Nonzero parameters: FP32 676, pruned 0.3 has 485,
pruned 0.5 has 356, pruned 0.7 has 229.

## Failure modes and limits

- Outliers and clipping: one large weight sets its column
  scale, and the calibration search found clipping helped
  only one matrix of four (INT4 W2 at 99.5). On a real model
  with heavy outlier channels, per-tensor INT4 without
  calibration would degrade far more than the 0.015 accuracy
  drop seen here.
- INT4 error is structural: max weight error 0.083784 is 16
  times the INT8 per-tensor error, and logit MSE grows by
  two orders of magnitude against INT8 per-channel. Accuracy
  only drops 6 samples because the class margins on this
  task are wide; a harder task would separate INT4 sooner.
- Pruning without recovery: pruned models get no fine-tuning,
  so sparsity 0.7 costs 34 correct samples against FP32.
  Pruning also saves no bytes in dense storage: all three
  pruned files are 3013 bytes, the same as FP32. A sparse
  format or a quantized pruned model is needed before
  pruning pays for its size, and neither is built here.
- Accuracy noise at this scale: 400 test samples mean one
  sample is 0.0025 of accuracy. Differences of one or two
  samples (pruned 0.3 against 0.5) are not evidence.
- Synthetic data ceiling: a Gaussian mixture with known
  centers is far easier than any real task, and the trained
  MLP is 676 parameters. Nothing here transfers numerically
  to a transformer or a real dataset; only the pipeline and
  the trade-off shape transfer.
- No quantized kernel: there is no INT8 or INT4 execution
  path, so the size win is real and the speed win is
  unmeasured. A llama.cpp or ONNX Runtime run is the Deep
  level scope, on a machine that has those tools.
- QLAB is a teaching format: it has no compression, no
  checksums, no streaming, and no other implementation. It
  proves save and load round-trip inside this repo only.

## Tests

43 unit tests pass (23 Learn, 20 Build), plus the scaffold
smoke test. Build tests carry hand-computed expectations for
per-channel scales and codes (including a clipping case),
INT4 packing sizes, QLAB save and load round-trips for FP32,
INT8 per-channel, and INT4 packed variants, the pruning mask,
and training determinism.

## Grade

Grade: 7,600 of 10,000 for this Build level.

Deductions (they sum to 2,400):
- No real pretrained model, no GGUF build, and no llama.cpp
  run, against the SPEC v1 core: minus 800.
- TensorFlow track did not run; the pruning rows are a NumPy
  equivalent substitute: minus 400.
- Synthetic data only, one Gaussian mixture dataset: minus 400.
- CPU timing only, on dequantized weights, with no GPU run
  (BlackBox unreachable) and no quantized kernel speed
  measured: minus 400.
- One small architecture (676 parameters) and one dataset, so
  the study is a single point, not a sweep: minus 250.
- Calibration effect is minimal: three of four calibrated
  matrices chose no clipping (percentile 100.0): minus 150.

## Jev gate

The main result paragraph above was gated with the Jev
claim_check gate on 2026-11-05. Model jev-1.13.0, input
tokens 1836, question id supported, noul 0.79, route REVIEW
(bands: CLEAR at 0.85 or above, BLOCK at 0.10 or below,
REVIEW between). Full record: level-2-build/JEV-GATE.md.

## What Deep adds next

The quality against size against speed study at real scale:
a real small model, a real GGUF build validated by a real
parser or a llama.cpp run, a quantized kernel timing path,
and pruning combined with quantization and fine-tuning.
See SPEC.md and the root README level map.
