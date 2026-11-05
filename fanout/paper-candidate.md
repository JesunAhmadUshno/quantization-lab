# Paper candidate brief: a quality, size, and speed decision rule for quantization routes

Project: P11 quantization-lab
Status: CANDIDATE, brief dated 2026-11-05
Submission note: submission is Jesun's tap. Nothing is submitted by
this brief, and no venue has been contacted.

## Working title

Measure all three axes: a quality against size against speed study
note with a tested decision rule for quantization and pruning routes

## Candidate claim, in one line

On a small, fully reproducible pipeline, measuring serialized
size, held-out accuracy, and logit fidelity together, and encoding
the choice as a tested decision rule, gives a route record that a
size-only or accuracy-only table cannot give. The evidence here is
a 676-parameter MLP on synthetic data, NOT a real pretrained model,
and the speed axis is measured but cannot separate routes under
fake quantization.

## Abstract draft

Based only on the measured v1.0 results recorded in the repo.
Scope limits are stated inside the abstract, not after it.

Model compression is usually reported on one axis at a time: a
size, an accuracy, or a speed. Each single axis can mislead. This
candidate presents a small study that measures all three together
over 10 routes on one model, with the choice between routes
encoded as a decision rule implemented and tested as code. The
model is a 2-layer MLP (16-32-4, 676 parameters) trained with
NumPy on a synthetic Gaussian mixture (train 800, calibration
200, held-out test 400), NOT a real pretrained model. FP32 scored
353 of 400 (0.8825) at 3013 serialized bytes in the QLAB container
format designed for this project (QLAB is not GGUF). INT8
per-tensor kept 353 of 400 at 1248 bytes (2.41x smaller, logit MSE
0.000296); calibrated INT8 per-channel kept 353 of 400 at 1388
bytes with the lowest quantized logit MSE (0.000129); calibrated
INT4 per-channel packed was smallest at 1081 bytes (2.79x smaller)
with 347 of 400 (0.8675) and logit MSE 0.045583. Magnitude pruning
at sparsity 0.3, 0.5, and 0.7 (a NumPy equivalent track, not a
TensorFlow result) kept the dense serialized size at 3013 bytes at
every sparsity while logit MSE rose to 0.533220 at sparsity 0.5
and 1.955935 at 0.7, and the three combined prune plus quantize
routes did not shrink size beyond their plain quantized
counterparts. Timing (batch median of 31 test-set passes, single
median of 101 passes) is the dequantized NumPy forward pass on
CPU, so it cannot separate routes and carries no speed ranking
claim. The pipeline ships with 55 unit tests. The candidate claims
the three-axis measurement discipline and the tested decision
rule on this small pipeline, not production model performance.

## Method outline

1. Model and data. MLP 16-32-4 (676 parameters), seeded
   full-batch gradient descent, synthetic 4-class Gaussian
   mixture, splits 800 train, 200 calibration, 400 held-out test.
   Classes overlap on purpose, so FP32 accuracy is 0.8825, not 1.0.
2. Routes. FP32; INT8 per-tensor (one symmetric scale per
   matrix); INT8 and INT4 per-channel (one scale per output
   column, INT4 codes packed two per byte); magnitude pruning at
   0.3, 0.5, 0.7 in dense FP32 storage; and three combined prune
   plus quantize routes. Biases stay FP32 and are counted in
   every size.
3. Calibration. The calibration split selects a clipping
   percentile per weight matrix per per-channel variant by lowest
   mean squared logit error against FP32. Most matrices chose
   100.0 (no clipping); that weakness is recorded, not hidden.
4. Serialization. Every route is saved in the QLAB container,
   loaded back, checked against its reference within 1e-6, and
   evaluated as the loaded model. Sizes are QLAB byte counts.
5. Analysis in code. Pareto flags on size and accuracy only,
   accuracy per kilobyte, and the decision rule: filter by max
   size, min accuracy, and max batch time, then rank survivors by
   accuracy, logit MSE, size, and batch time. Speed is not a
   Pareto axis, because fake quantization timing flipped flags
   between runs during development.
6. Determinism. Runners write results files that are identical
   across runs once the timing fields are removed, verified by
   comparison in each level cycle.

## Results table

Measured 2026-11-05, Muse VM CPU, Python 3.12, NumPy 1.26.4,
from the level READMEs and results files in the repo.

| Route | Size bytes | Test accuracy | Logit MSE vs FP32 |
| --- | --- | --- | --- |
| FP32 | 3013 | 353 of 400 (0.8825) | 0.000000 |
| INT8 per-tensor | 1248 | 353 of 400 (0.8825) | 0.000296 |
| INT8 per-channel, calibrated | 1388 | 353 of 400 (0.8825) | 0.000129 |
| INT4 per-channel packed, calibrated | 1081 | 347 of 400 (0.8675) | 0.045583 |
| Pruned 0.5 (NumPy equivalent) | 3013 | 350 of 400 (0.8750) | 0.533220 |
| Pruned 0.5 plus INT8 per-channel | 1388 | 351 of 400 (0.8775) | 0.533527 |
| Pruned 0.5 plus INT4 packed | 1081 | 348 of 400 (0.8700) | 0.542604 |

## Limits a reviewer will raise, stated first

- No real model. Every number describes a 676-parameter MLP on
  synthetic data, NOT a real pretrained model or dataset.
- No GGUF and no kernel. Sizes are QLAB bytes, quality uses
  dequantized weights, and no INT8 or INT4 execution path is
  timed, so the speed axis carries no ranking claim.
- No TensorFlow result. Pruning is a NumPy equivalent track.
- No GPU. All runs are on one machine (the Muse VM CPU), not
  the assigned fleet node.
- Small base and one-sample noise. 400 test samples mean one
  sample is 0.0025 of accuracy; the brief names such edges as
  noise wherever they appear.
- Weak calibration. Most calibrated matrices chose no clipping,
  so the calibration contribution measured here is small.

## What would strengthen the candidate

Each item below is v1.1 backlog in ROADMAP.md, not done work: a
real small pretrained model through the same pipeline, a
validated GGUF build and llama.cpp run, a real quantized kernel
timing path, a TensorFlow row, pruning with sparse storage and
fine-tuning, and a scale sweep across model sizes and a harder
task.

## Venue note

No venue is named as submitted or contacted. Venue choice and
any submission click are Jesun's tap only.
