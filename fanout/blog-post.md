# Smaller, Faster, Honest: A Quality Against Size Against Speed Study of Quantization

By Jesun Ahmad Ushno
Date: 2026-11-05
Status: DRAFT, written fan-out for P11 quantization-lab v1.0. Not posted.

## The problem

Making a model smaller is easy to claim and hard to prove. A format
can shrink the bytes and quietly damage the outputs, a pruning run
can remove half the weights and save no space at all in dense
storage, and a timing column can look decisive while every variant
runs the same dequantized code. Quantization needs the three axes
measured together: quality, size, and speed, with the limits named.

This project, P11 in Roadmap v3, builds that study in three
levels: symmetric quantization taught on a 12-weight model, a
NumPy-trained MLP quantized to INT8 and INT4, pruned, and
serialized in a documented container, then a 10-route study with
charts, a Pareto analysis, and a decision rule implemented as
code. All code is in the open and runs on CPU with NumPy.

Repository: https://github.com/JesunAhmadUshno/quantization-lab

## Honest scope, stated first

The model is a small MLP trained in this repo on a synthetic
Gaussian mixture. It is NOT a real pretrained model, and no number
here describes a transformer or a real dataset. Sizes are QLAB
container bytes: QLAB is a format designed for this project, it is
NOT GGUF, and no file here runs in llama.cpp. Pruning rows are a
NumPy equivalent track, NOT a TensorFlow result, because
TensorFlow was unavailable on the build machine. Timing is the
dequantized NumPy forward pass on CPU, so it does NOT show the
speed of a real INT8 or INT4 kernel. All levels ran on the Muse
VM CPU, not on the assigned fleet node BlackBox, which was
unreachable (NEEDS-BLACKBOX-ON).

## Past, present, future (the dossier, condensed)

Past: quantization toolchains such as llama.cpp, GPTQ, and AWQ
tooling already exist. The prior art is recorded in DOSSIER.md and
NOVELTY.md in the repo. The novelty text there stays marked
PROPOSED. This post makes no novelty claim beyond the measured
work below.

Present: v1.0 closes three levels on 2026-11-05. Learn quantized
a deterministic teaching model. Build trained and quantized the
MLP and added calibrated per-channel routes and magnitude
pruning. Compressed Deep ran the quality against size against
speed study over 10 routes. Grades on the house 10,000 scale:
Learn 8,700, Build 7,600, Deep 7,450.

Future: the rest of the v2 Deep scope is named in ROADMAP.md in
the repo as the v1.1 backlog. It includes a real pretrained model,
a validated GGUF build and llama.cpp run, a GPU window on
BlackBox, a real quantized kernel timing path, a TensorFlow row,
pruning with sparse storage and fine-tuning, and a scale sweep.
None of that is done, and none of it is claimed here.

## Main results claim

Deep result: on the Build synthetic task (400 held-out samples, 2-layer MLP with 676 parameters), FP32 scored 353 of 400 (0.8825) at 3013 QLAB bytes; INT8 per-tensor kept 353 of 400 at 1248 bytes (2.41x smaller) and calibrated INT8 per-channel kept 353 of 400 at 1388 bytes with the lowest quantized logit MSE (0.000129); INT4 per-channel packed was smallest at 1081 bytes (2.79x smaller) with 347 of 400 (0.8675); pruning alone kept the size at 3013 bytes at every sparsity, and the combined routes did not shrink size beyond their plain quantized counterparts (pruned 0.5 plus INT8 per-channel is 1388 bytes, the same as INT8 per-channel; pruned 0.5 plus INT4 packed is 1081 bytes, the same as INT4 packed, with 348 of 400 and a logit MSE of 0.542604, about 12 times the plain INT4 value of 0.045583); all sizes are QLAB container bytes (QLAB is a format designed for this project, not GGUF), all quality numbers were measured with dequantized weights on CPU on the Muse VM, and the pruning and combined rows are a NumPy equivalent track, not a TensorFlow result, because TensorFlow was unavailable on this machine.

Every number in that claim comes from a run recorded in the level
READMEs and the results files in the repo. Read the limits section
below before quoting any of it.

## Results by level

### Level 1, Learn (release v0.1)

One deterministic 4 by 3 linear model, 12 weights, quantized with
symmetric per-tensor scales.

| Format | Total bytes | Max abs error | Task |
| --- | --- | --- | --- |
| FP32 | 48 | 0.000000 | 8 of 8 |
| INT8 | 16 (3.0x smaller) | 0.003937 | 8 of 8 |
| INT4 packed | 10 (4.8x smaller) | 0.071429 | 8 of 8 |

Grade: 8,700 of 10,000. Tests: 23 unit tests, plus the scaffold
smoke test. The synthetic task saturates at 8 of 8 for every
format; only the score MAE (INT8 0.002707, INT4 0.049107) shows
the degradation at this level.

### Level 2, Build (release v0.5)

A 2-layer MLP (16-32-4, 676 parameters) trained with NumPy on a
synthetic Gaussian mixture: train 800, calibration 200, held-out
test 400.

| Variant | Size bytes | Test accuracy | Logit MSE vs FP32 |
| --- | --- | --- | --- |
| FP32 | 3013 | 353 of 400 (0.8825) | 0.000000 |
| INT8 per-tensor (Learn method) | 1248 | 353 of 400 (0.8825) | 0.000296 |
| INT8 per-channel, calibrated | 1388 | 353 of 400 (0.8825) | 0.000129 |
| INT4 per-channel packed, calibrated | 1081 | 347 of 400 (0.8675) | 0.045583 |
| Pruned 0.3 (NumPy equivalent) | 3013 | 349 of 400 (0.8725) | 0.099430 |
| Pruned 0.5 (NumPy equivalent) | 3013 | 350 of 400 (0.8750) | 0.533220 |
| Pruned 0.7 (NumPy equivalent) | 3013 | 319 of 400 (0.7975) | 1.955935 |

Grade: 7,600 of 10,000. Tests at this level: 20 new tests, 43 unit
tests in total with Learn.

### Level 3, Deep, compressed (release v1.0)

The 7 Build routes plus 3 combined prune plus quantize routes.
Batch time is the median of 31 full test-set passes; single time
is the median of 101 single-sample passes. Timing separates
nothing here, because every route runs the same dequantized
forward pass.

| Route | Size bytes | Test accuracy | Logit MSE vs FP32 |
| --- | --- | --- | --- |
| FP32 | 3013 | 353 of 400 (0.8825) | 0.000000 |
| INT8 per-tensor (Learn method) | 1248 | 353 of 400 (0.8825) | 0.000296 |
| INT8 per-channel, calibrated | 1388 | 353 of 400 (0.8825) | 0.000129 |
| INT4 per-channel packed, calibrated | 1081 | 347 of 400 (0.8675) | 0.045583 |
| Pruned 0.5 plus INT8 per-channel | 1388 | 351 of 400 (0.8775) | 0.533527 |
| Pruned 0.5 plus INT4 packed | 1081 | 348 of 400 (0.8700) | 0.542604 |

The verdict, in one line: at this scale, INT8 per-channel is the
default route, INT8 per-tensor is the tight-budget route, INT4
packed is the smallest-size route with a named accuracy and
fidelity cost, and pruning in any form is not a size route until
storage or recovery changes.

Grade: 7,450 of 10,000. Tests at this level: 12 new tests, 55
unit tests in total, plus the scaffold smoke test.

## Honest limits

- Synthetic data and one small model. A 676-parameter MLP on a
  Gaussian mixture says nothing numerically about a transformer
  or a real dataset. Only the pipeline and the trade-off shape
  transfer.
- No GGUF, no llama.cpp, no TensorFlow result, no GPU run. Those
  four gaps are named in every level README and are the top of
  the v1.1 backlog.
- Pruning saved no bytes. Pruned models are stored dense, so all
  pruned sizes stay 3013 bytes, the same as FP32.
- Speed cannot separate these routes. Fake quantization timing is
  the dequantized forward pass; a real kernel is v1.1 scope.
- One-sample edges are noise. At 400 test samples, one sample is
  0.0025 of accuracy; the combined INT4 route's one-sample edge
  over plain INT4 comes with about 12 times the logit MSE.
- Accuracy noise and calibration weakness are stated in the level
  READMEs: most calibrated matrices chose no clipping.

## Why this matters for real optimization work

The transferable part is the discipline, not the toy numbers:
measure size, accuracy, and logit fidelity together, state the
container and the execution path behind every number, never let
a timing column from fake quantization carry a speed claim, and
write the decision rule as code that can be tested against the
measured table. That discipline is what the v1.1 backlog carries
to a real pretrained model.

## Links

- Repo: https://github.com/JesunAhmadUshno/quantization-lab
- Release v0.1: https://github.com/JesunAhmadUshno/quantization-lab/releases/tag/v0.1
- Release v0.5: https://github.com/JesunAhmadUshno/quantization-lab/releases/tag/v0.5
- Release v1.0: https://github.com/JesunAhmadUshno/quantization-lab/releases/tag/v1.0
- Tutorial: fanout/tutorial.md in the repo
