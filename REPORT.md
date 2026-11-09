# Report: quantization-lab (P11)

Status: 2026-11-08. Verdict: Functional, not pre-registered. The study
ran end to end on CPU and every number below is regenerated from the
code by `python scripts/check_results.py`, which CI runs on Python 3.11
and 3.12. The hypotheses were written after the results (PROPOSAL.md),
so nothing here is confirmatory.

## Background

Weight quantization trades bits for fidelity. This repo measures that
trade on a model small enough to check by hand: symmetric per-tensor
INT8 and INT4 on a 12-weight linear model (baseline, Appendix A), then
a 676-parameter NumPy MLP quantized per-tensor and per-channel,
calibrated, magnitude pruned and serialized in the QLAB container
designed for this project (Appendix B), then a 10-route study of size,
accuracy, logit fidelity and CPU time (Appendix C), and a 10-seed
uncertainty study (2026-11-07, Appendix E).

## Method

Model, data, quantization, calibration, pruning, the QLAB container and
timing are described in full in Appendix B (Method) and Appendix C
(Method). In short: MLP 16-32-4 (676 parameters), full-batch gradient
descent, 60 epochs, on a synthetic 4-class Gaussian mixture (train 800,
calibration 200, test 400); weight-only symmetric quantization with
biases in FP32; sizes are serialized QLAB bytes; quality uses
dequantized weights (fake quantization) on CPU. Uncertainty: Wilson 95%
intervals on test accuracy, exact McNemar tests against FP32 on the
recorded per-sample predictions, and 10 seed sets (training init and
data draws vary; class centers fixed). Machine of record for the
recorded files: the Muse VM, CPU, Python 3.12, NumPy 1.26.4 for
results_build.json and results_deep.json, NumPy 2.4.5 for
results_seeds.json (each file records its environment or states it).

## Results

Seed 0 is the recorded run (results_deep.json); the seed columns come
from results_seeds.json. Wilson intervals are computed from the counts
by deep_system.wilson_interval.

| Route | Size bytes | Test correct | Wilson 95% | Logit MSE vs FP32 | McNemar p vs FP32, seed 0 | Change vs FP32 over 10 seeds, samples | Seeds with p < 0.05 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FP32 | 3013 | 353 of 400 | 0.847 to 0.910 | 0.000000 | 1 | +0 to +0 | 0 of 10 |
| INT8 per-tensor | 1248 | 353 of 400 | 0.847 to 0.910 | 0.000296 | 1 | -1 to +1 | 0 of 10 |
| INT8 per-channel, calibrated | 1388 | 353 of 400 | 0.847 to 0.910 | 0.000129 | 1 | -2 to +1 | 0 of 10 |
| INT4 per-channel packed, calibrated | 1081 | 347 of 400 | 0.831 to 0.897 | 0.045583 | 0.0703 | -6 to +4 | 0 of 10 |
| Pruned 0.3 | 3013 | 349 of 400 | 0.836 to 0.902 | 0.099430 | 0.344 | -9 to -1 | 1 of 10 |
| Pruned 0.5 | 3013 | 350 of 400 | 0.839 to 0.904 | 0.533220 | 0.678 | -20 to -3 | 4 of 10 |
| Pruned 0.7 | 3013 | 319 of 400 | 0.755 to 0.834 | 1.955935 | 2.44e-05 | -59 to -19 | 10 of 10 |
| Pruned 0.3 plus INT8 per-channel | 1388 | 350 of 400 | 0.839 to 0.904 | 0.099504 | 0.549 | -10 to -1 | 2 of 10 |
| Pruned 0.5 plus INT8 per-channel | 1388 | 351 of 400 | 0.842 to 0.906 | 0.533527 | 0.832 | -20 to -2 | 4 of 10 |
| Pruned 0.5 plus INT4 packed | 1081 | 348 of 400 | 0.833 to 0.899 | 0.542604 | 0.442 | -22 to -2 | 3 of 10 |

FP32 accuracy over the 10 seeds is 0.900 (sd 0.0115), a spread larger
than every quantized gap. Timing per route is in Appendix C; it does
not separate routes (H5).

## Hypotheses scored

| ID | Result | Evidence |
| --- | --- | --- |
| H1 INT8 accuracy-neutral | CONFIRMED (on this model) | Both INT8 routes: p < 0.05 in 0 of 10 seeds; net change in correct samples -2 to +1 against FP32 (at most 3 of 400 flipped) |
| H2 INT4 smallest, with an accuracy cost | PARTLY: smallest CONFIRMED (1081 bytes, 2.79x); accuracy cost NOT MET | Pass line needs McNemar p < 0.05; it held in 0 of 10 seeds (seed 0 p = 0.0703), so this half is a miss. 400 test samples are underpowered to detect a cost this small; the cost shows in logit MSE (0.045583 against 0.000129 for INT8 per-channel) |
| H3 Pruning alone not a size route; heavy pruning hurts | CONFIRMED | 3013 bytes at every sparsity; pruned 0.7 p < 0.05 in 10 of 10 seeds |
| H4 Prune plus quantize does not stack size | CONFIRMED | Combined sizes 1388 and 1081 bytes equal their plain quantized routes |
| H5 Speed separates routes | REFUTED on this setup | Batch medians 0.178 to 0.250 ms, every route runs the same dequantized NumPy pass; run-to-run spread of 0.178 to 0.315 ms for one route is stated in Appendix C but its raw timings are not kept in the repo |
| H6 SPEC v1 core: GGUF in llama.cpp, TensorFlow row | NOT RUN (miss) | No llama.cpp or TensorFlow on the build machine; QLAB is not GGUF; pruning is a NumPy equivalent track |
| H7 Code rule agrees with the verdict | MISSED on the recorded rows, fixed by an option | Strict rule picks pruned 0.5 plus INT4 under 1200 bytes (results_deep.json verdict_inputs); plain INT4 in 9 of 10 seeds; with accuracy_tie = 1/400, plain INT4 in 10 of 10 (results_seeds.json decision_counts) |

## Misses

- The SPEC.md v1 core did not run: no GGUF build, no llama.cpp run, no
  TensorFlow Model Optimization row, no GPU (H6).
- No real pretrained model and no real data: one 676-parameter MLP on a
  synthetic Gaussian mixture.
- The speed axis measures nothing about quantized kernels (H5).
- The decision rule disagreed with the written verdict on the recorded
  seed until the tie option was added (H7).
- The recorded Pareto set (INT8 per-tensor and pruned 0.5 plus INT4)
  appears in only 1 of 10 seeds; plain INT4 is Pareto in 9 of 10.

## Field profile check (ML and AI research)

| Item | State |
| --- | --- |
| Baselines a reviewer would expect | Not met: FP32 and round-to-nearest only; no GPTQ, AWQ or llama.cpp baseline |
| Ablations | Partial: per-tensor against per-channel, with and without pruning, three sparsities |
| Uncertainty on every headline number | Met for accuracy (Wilson intervals, 10 seeds, McNemar); not for timing |
| Compute, data and versions recorded | Met: machine, Python and NumPy versions, seeds and splits in each results file or its write-up |

The profile is not met, and the repo is not pre-registered, so the
verdict stays Functional.

## Threats to validity

- Scale: nothing here transfers numerically to a transformer; only the
  study shape transfers.
- Noise: one test sample is 0.0025 of accuracy; edges of one or two
  samples are not findings.
- Fake quantization: quality uses dequantized weights, so integer
  arithmetic effects are not measured.
- Environment drift: the Build and Deep files were recorded with NumPy
  1.26.4; regeneration on NumPy 2.x matches every count and size
  exactly and every float to a relative 1e-6 (scripts/check_results.py).
- Hypotheses written after the results (PROPOSAL.md).

## Evidence trace

| Number | File | Command |
| --- | --- | --- |
| 353 of 400, 1388 bytes, 3013 bytes, 1081 bytes, logit MSE | level-3-deep/results_deep.json | `python3 -c "import json; r={x['variant']:x for x in json.load(open('level-3-deep/results_deep.json'))['rows']}; print({k:(r[k]['size_bytes'], r[k]['test_correct'], r[k]['logit_mse_vs_fp32']) for k in r})"` |
| Wilson 0.847 to 0.910 | level-3-deep/results_deep.json | `python3 -c "import json,sys; sys.path.insert(0,'level-3-deep'); from deep_system import wilson_interval; r={x['variant']:x for x in json.load(open('level-3-deep/results_deep.json'))['rows']}; print(wilson_interval(r['int8_per_channel']['test_correct'], r['int8_per_channel']['test_total']))"` |
| Seed ranges, McNemar counts | level-3-deep/results_seeds.json | `python3 -c "import json; s=json.load(open('level-3-deep/results_seeds.json'))['summary']; print({k:(v['delta_correct_vs_fp32_min'], v['delta_correct_vs_fp32_max'], v['seeds_mcnemar_p_below_0_05']) for k,v in s.items()})"` |
| All of the above regenerate from code | every results file | `python3 scripts/check_results.py` |

## Conclusion

On this model, INT8 weight quantization is free in accuracy and saves
2.2 to 2.4 times in size; INT4 saves 2.8 times with a clear fidelity
cost that 400 test samples cannot resolve into an accuracy cost; and
pruning without sparse storage or recovery is not a size route. The
study is a working, reproducible harness, not an answer about real
models: the GGUF, llama.cpp, TensorFlow, GPU and real-model parts of
the original scope never ran.

## Future work

The v1.1 backlog (Appendix F, formerly ROADMAP.md) and the gap table in
Appendix E list what each open item needs. The first step is a newly
locked proposal (PROPOSAL.md, "What a confirmatory cycle needs").

## Appendices: records as they were (moved here 2026-11-08)

Old file to section map: level-1-learn/README.md is Appendix A,
level-2-build/README.md is Appendix B, level-3-deep/README.md is
Appendix C, level-2-build/JEV-GATE.md and level-3-deep/JEV-GATE.md are
Appendices D1 and D2, docs/research-review.md is Appendix E, ROADMAP.md is
Appendix F, and the README before standard v2 is Appendix G. References
inside these records to those file names now point to these sections;
SPEC.md, NOVELTY.md, DOSSIER.md, BUILD-SPEC.md and DEEP-SPEC.md are in
PROPOSAL.md, Appendices A to E. Grades were removed (they are kept
internally).


## Appendix A: baseline, the Learn record (was level-1-learn/README.md)

### Level 1, Learn: quantization-lab

Status: DONE (2026-11-05). Symmetric per-tensor INT8 and INT4
quantization of one small deterministic linear model, built with
the Python standard library only.

#### Honest scope line

There is no real model in this level, no GGUF, no llama.cpp, and
no GPU. The model is a hand-built 4 by 3 weight matrix (12 weights)
and the task is a small synthetic labelled set. Those limits are
stated plainly here, not hidden. A real model, GGUF builds, and
llama.cpp runs are Build level scope, and a quality against size
against speed study on real models is Deep level scope. This level
was run on the Muse VM, not on BlackBox, the assigned GPU fleet
node: Learn is CPU teaching work by design, one fleet SSH probe
was attempted and did not connect from this VM.

#### Method

1. Model. One linear model, 4 classes by 3 inputs, 12 weights in
   row-major order: 1.0, -0.5, 0.25, -0.5, 1.0, -0.25, 0.25, -0.75,
   1.0, -0.75, -0.25, -1.0. Each row is a signed direction
   specialist. The largest absolute weight is 1.0.
2. Symmetric per-tensor quantization. One scale covers the whole
   tensor: scale is the maximum absolute value divided by qmax,
   where qmax is 127 for INT8 and 7 for INT4. The INT4 range is
   clamped to -7 to 7, not -8 to 7, so the scale stays symmetric.
   A code is the value divided by the scale, rounded half away
   from zero (3.5 goes to 4, -3.5 goes to -4), clamped to the
   range. Dequantization is code times scale.
3. INT4 packing. Two codes go in one byte, the first code in the
   low nibble, each stored as its 4 bit two complement pattern.
   The six packed bytes for this model are 199, 194, 231, 178,
   183, 158, and unpacking them returns the 12 codes exactly.
4. Sizes. FP32 stores 4 bytes per weight and no scale. Quantized
   formats store their payload plus one FP32 scale (4 bytes).
   INT4 packing for an odd weight count wastes half of the last
   byte, and the size count charges for that whole byte.
5. Errors. For each format, the maximum absolute error, the mean
   absolute error, and the mean squared error (MSE) between the
   original and dequantized weights are reported.
6. Task. Eight synthetic labelled samples are scored by the FP32
   model and by each dequantized model. The prediction is the
   argmax of the four class scores, first index winning a tie.
   The labels are the FP32 argmax labels, checked by hand in the
   tests. Accuracy and the mean absolute score difference against
   the FP32 scores are reported for each format.

#### Run it

    python level-1-learn/run_learn.py
    python -m unittest tests.test_learn_quantization -v
    python tests/test_smoke.py

The runner writes level-1-learn/results_learn.json deterministically:
run it twice and the file is identical (verified 2026-11-05:
sha256 2291d1d3295b83ee8fc204a90f5bf3c4bcca04274a04a664251c8f48b544f120
on both runs).

#### Results (measured 2026-11-05, Muse VM, Python 3.12)

| Format | Scale | Payload bytes | Scale bytes | Total bytes | Total vs FP32 | Max abs error | Mean abs error | MSE | Task accuracy | Score MAE vs FP32 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FP32 | none | 48 | 0 | 48 | 1.0x | 0.000000 | 0.000000 | 0.000000 | 8 of 8 | 0.000000 |
| INT8 | 0.007874 (1/127) | 12 | 4 | 16 | 3.0x smaller | 0.003937 | 0.001640 | 0.000005 | 8 of 8 | 0.002707 |
| INT4 | 0.142857 (1/7) | 6 | 4 | 10 | 4.8x smaller | 0.071429 | 0.029762 | 0.001488 | 8 of 8 | 0.049107 |

Codes, in the flat weight order above:

- INT8: 127, -64, 32, -64, 127, -32, 32, -95, 127, -95, -32, -127.
- INT4: 7, -4, 2, -4, 7, -2, 2, -5, 7, -5, -2, -7.

| Sample | Input | Label | FP32 scores | FP32 pred | INT8 pred | INT4 pred |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1, 0, 0 | 0 | 1.0, -0.5, 0.25, -0.75 | 0 | 0 | 0 |
| 2 | 0, 1, 0 | 1 | -0.5, 1.0, -0.75, -0.25 | 1 | 1 | 1 |
| 3 | 0, 0, 1 | 2 | 0.25, -0.25, 1.0, -1.0 | 2 | 2 | 2 |
| 4 | 0, 0, -1 | 3 | -0.25, 0.25, -1.0, 1.0 | 3 | 3 | 3 |
| 5 | 2, 1, 0 | 0 | 1.5, 0.0, -0.25, -1.75 | 0 | 0 | 0 |
| 6 | 1, 2, 0 | 1 | 0.0, 1.5, -1.25, -1.25 | 1 | 1 | 1 |
| 7 | 0, 1, -2 | 3 | -1.0, 1.5, -2.75, 1.75 | 3 | 3 | 3 |
| 8 | 1, 0, -1 | 0 | 0.75, -0.25, -0.75, 0.25 | 0 | 0 | 0 |

Summary: Learn quantized one deterministic 4 by 3 linear model (12 weights) on the Muse VM: INT8 used 16 total bytes against 48 for FP32 with max absolute error 0.003937, mean absolute error 0.001640, and MSE 0.000005, INT4 used 10 total bytes with max absolute error 0.071429, mean absolute error 0.029762, and MSE 0.001488, and FP32, INT8, and INT4 each scored 8 of 8 on the synthetic task, with 23 unit tests passing plus the scaffold smoke test.

Hand checks: sizes are 12 weights times 4 bytes = 48 for FP32,
12 times 1 byte = 12 payload plus 4 scale = 16 for INT8, and
12 times 4 bits = 48 bits = 6 bytes payload plus 4 scale = 10
for INT4. INT8 codes: -0.5 times 127 is -63.5, which rounds half
away to -64; 0.25 times 127 is 31.75, which rounds to 32; -0.75
times 127 is -95.25, which rounds to -95. INT4 codes: -0.5 times
7 is -3.5, which rounds half away to -4; 0.25 times 7 is 1.75,
which rounds to 2. INT8 absolute errors are 0 for the four
weights with absolute value 1.0, 0.5/127 (0.003937) for the two
-0.5 weights, and 0.25/127 (0.001969) for the six remaining
weights, so the mean is (2 times 0.003937 plus 6 times 0.001969)
divided by 12 = 0.001640. INT4 absolute errors are 0 for the
1.0 weights, 1/14 (0.071429) for the two -0.5 weights, and 1/28
(0.035714) for the six remaining weights. Packed byte 1 is the
pair (7, -4): 7 plus 12 shifted left by 4 = 199. Sample 7 FP32
scores are 0 - 0.5 - 0.5 = -1.0, 0 + 1 + 0.5 = 1.5, 0 - 0.75 - 2
= -2.75, and 0 - 0.25 + 2 = 1.75, so its label is class 3 by a
margin of 0.25 over class 1.

#### Failure modes and limits

- One designed model only: a single 12-weight matrix with simple
  fraction weights chosen so the arithmetic can be checked by
  hand. Real weight distributions have outliers that dominate a
  per-tensor scale; nothing here measures that effect.
- No real model, no GGUF, no llama.cpp, and no speed: no claim is
  made about real model quality, file formats, or inference
  speed. Size is counted in bytes for this tensor only, not for
  a whole model file with metadata.
- The synthetic task saturates: all three formats score 8 of 8,
  because the class margins are wide next to the quantization
  error. Accuracy at this scale does not separate INT4 from FP32;
  only the score MAE (0.049107 for INT4 against 0.002707 for
  INT8) shows the degradation. A harder task is Build scope.
- Per-tensor symmetric only: no per-channel scales, no
  asymmetric quantization with a zero point, and no calibration
  data. One large weight would set the scale for every weight.
- Rounding is one fixed rule (half away from zero) with clamping;
  no rounding search or error compensation is tried.
- INT4 wastes range and space at the edges: the -8 code is never
  used, to keep the scale symmetric, and an odd weight count
  wastes half a byte of packing.
- This level was run on the Muse VM (CPU), not on its fleet node
  BlackBox.

#### What Build adds next

A real small model quantized to INT8 and INT4, GGUF builds run
locally with llama.cpp, per-channel scales beside per-tensor, and
a quality task that does not saturate. See the root README level
map.

## Appendix B: Build record (was level-2-build/README.md)

### Level 2, Build: quantization-lab

Status: DONE (2026-11-05). A 2-layer MLP trained with NumPy,
quantized to INT8 and INT4, magnitude pruned, serialized in a
documented binary container, and benchmarked end to end on CPU.
The governing spec for this level, written before the code, is
BUILD-SPEC.md in this folder, including every substitution
named below.

#### Main result

Build result: on a synthetic 4-class task with 400 held-out samples, a NumPy-trained 2-layer MLP with 676 parameters scored 353 of 400 (0.8825) in FP32 at 3013 serialized bytes; INT8 per-tensor kept 353 of 400 at 1248 bytes and calibrated INT8 per-channel kept 353 of 400 at 1388 bytes with lower mean weight error (0.001267 against 0.002292), while calibrated INT4 per-channel packed scored 347 of 400 (0.8675) at 1081 bytes; every size is a QLAB container size (QLAB is a format designed for this project, not GGUF), every quality number was measured with dequantized weights on CPU, and the pruning rows in the benchmark table are a NumPy equivalent track, not a TensorFlow result, because TensorFlow was unavailable on the build machine.

#### Honest scope line

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

#### Method

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
   Calibration changed little here.
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

#### Run it

    python level-2-build/run_build.py
    python -m unittest discover -s tests -v
    python tests/test_smoke.py

The runner writes level-2-build/results_build.json and QLAB
files under level-2-build/artifacts/. The results file is
deterministic except the timing_ms_median fields: two runs on
2026-11-05 produced identical files once those fields were
removed (verified by comparison in this cycle).

#### Results (measured 2026-11-05, Muse VM CPU, Python 3.12, NumPy 1.26.4)

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

#### Failure modes and limits

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

#### Tests

43 unit tests pass (23 Learn, 20 Build), plus the scaffold
smoke test. Build tests carry hand-computed expectations for
per-channel scales and codes (including a clipping case),
INT4 packing sizes, QLAB save and load round-trips for FP32,
INT8 per-channel, and INT4 packed variants, the pruning mask,
and training determinism.

#### Jev gate

The main result paragraph above was gated with the Jev
claim_check gate on 2026-11-05. Model jev-1.13.0, input
tokens 1836, question id supported, noul 0.79, route REVIEW
(bands: CLEAR at 0.85 or above, BLOCK at 0.10 or below,
REVIEW between). Full record: level-2-build/JEV-GATE.md.

#### What Deep adds next

The quality against size against speed study at real scale:
a real small model, a real GGUF build validated by a real
parser or a llama.cpp run, a quantized kernel timing path,
and pruning combined with quantization and fine-tuning.
See SPEC.md and the root README level map.

## Appendix C: Deep record (was level-3-deep/README.md)

### Level 3, Deep: quantization-lab

Status: DONE (2026-11-05, compressed Deep under the December
Compression Protocol). The quality against size against speed
study on the Build MLP: 10 routes (the 7 Build routes plus 3
combined prune plus quantize routes), measured accuracy,
serialized size, and CPU speed for each, three charts built
by code from the measured rows, a Pareto analysis on size
and accuracy, and a written verdict with a decision rule
that is also implemented and tested as code. The governing
spec for this level, written before the code, is DEEP-SPEC.md
in this folder. The rest of the v2 Deep scope is the v1.1
backlog in the root ROADMAP.md.

#### Main result

Deep result: on the Build synthetic task (400 held-out samples, 2-layer MLP with 676 parameters), FP32 scored 353 of 400 (0.8825) at 3013 QLAB bytes; INT8 per-tensor kept 353 of 400 at 1248 bytes (2.41x smaller) and calibrated INT8 per-channel kept 353 of 400 at 1388 bytes with the lowest quantized logit MSE (0.000129); INT4 per-channel packed was smallest at 1081 bytes (2.79x smaller) with 347 of 400 (0.8675); pruning alone kept the size at 3013 bytes at every sparsity, and the combined routes did not shrink size beyond their plain quantized counterparts (pruned 0.5 plus INT8 per-channel is 1388 bytes, the same as INT8 per-channel; pruned 0.5 plus INT4 packed is 1081 bytes, the same as INT4 packed, with 348 of 400 and a logit MSE of 0.542604, about 12 times the plain INT4 value of 0.045583); all sizes are QLAB container bytes (QLAB is a format designed for this project, not GGUF), all quality numbers were measured with dequantized weights on CPU on the Muse VM, and the pruning and combined rows are a NumPy equivalent track, not a TensorFlow result, because TensorFlow was unavailable on this machine.

#### Honest scope line

- No GGUF and no llama.cpp. Sizes are QLAB container bytes
  from Build. QLAB is not GGUF and no file here runs in
  llama.cpp, which is not installed on this machine (checked
  this cycle).
- No TensorFlow result. import tensorflow fails on this
  machine (checked this cycle). Pruning is the Build NumPy
  equivalent track.
- No GPU. BlackBox, the assigned fleet node, was unreachable
  this cycle (NEEDS-BLACKBOX-ON stands). All work ran on the
  Muse VM CPU. No GPU number is claimed.
- Fake quantization for quality and timing, as in Build.
  Timing is the dequantized NumPy forward pass on this VM.
  It does not show a real INT8 or INT4 kernel speed, and the
  verdict below uses speed only to say that it cannot
  separate these routes here.
- Same model and data as Build, on purpose: the trained MLP
  (16-32-4, 676 parameters) and splits (train 800,
  calibration 200, test 400), so Deep rows compare directly
  with Build rows. The 7 rebuilt Build routes reproduce the
  Build sizes, accuracies, and errors exactly.

#### Method

1. Routes. The 7 Build routes plus 3 combined routes that
   Build did not measure: pruned 0.3 plus INT8 per-channel,
   pruned 0.5 plus INT8 per-channel, and pruned 0.5 plus INT4
   per-channel packed. A combined route prunes first, then
   calibrates and quantizes the pruned weights on the same
   calibration split, using Build calibration unchanged.
2. Serialization check. Every route is saved as QLAB, loaded
   back, and checked against its reference model within the
   Build tolerance of 1e-6. The rows evaluate the loaded
   model itself. One integration quirk is recorded in the
   failure modes: Build branches on route names that start
   with pruned, so combined routes pass a neutral branch
   name to the Build container helpers.
3. Speed, stated method. Batch time is the median of 31 full
   test-set (400 sample) forward passes after 5 warmup
   passes. Single-sample latency is the median of 101 forward
   passes on the first test sample after 5 warmup passes.
   Both are CPU timing on this VM (single machine, varying
   load). These two fields per row are the only
   nondeterministic fields in results_deep.json.
4. Analysis, in code. Pareto flags on size and accuracy
   only (deep_system.pareto_flags), accuracy per kilobyte,
   and the decision rule (deep_system.recommend_route):
   filter by any of max size, min accuracy, and max batch
   time, then rank survivors by accuracy (highest), logit
   MSE (lowest), size (smallest), batch time (lowest).
   Speed is not a Pareto axis, for the reason in the failure
   modes.
5. Charts. Three SVG charts generated by code from the
   measured rows only, axis ranges computed from the data:
   charts/accuracy_vs_size.svg, charts/logit_mse_vs_size.svg,
   and charts/batch_time_vs_accuracy.svg. Every point is one
   measured route, labelled by name.

#### Run it

    python level-3-deep/run_deep.py
    python -m unittest discover -s tests -v
    python tests/test_smoke.py

The runner writes level-3-deep/results_deep.json, QLAB files
under level-3-deep/artifacts/ (not committed, as in Build),
and the three charts under level-3-deep/charts/. The results
file is deterministic except the two timing fields per row:
two runs on 2026-11-05 produced identical files once those
fields were removed (verified by comparison in this cycle).

#### Results (measured 2026-11-05, Muse VM CPU, Python 3.12, NumPy 1.26.4)

| Route | Size bytes | Size vs FP32 | Test accuracy | Logit MSE vs FP32 | Batch ms, median of 31 | Single ms, median of 101 | Pareto (size, accuracy) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FP32 | 3013 | 1.00x | 353 of 400 (0.8825) | 0.000000 | 0.250 | 0.0085 | no |
| INT8 per-tensor (Learn method) | 1248 | 2.41x smaller | 353 of 400 (0.8825) | 0.000296 | 0.250 | 0.0085 | yes |
| INT8 per-channel, calibrated | 1388 | 2.17x smaller | 353 of 400 (0.8825) | 0.000129 | 0.192 | 0.0085 | no |
| INT4 per-channel packed, calibrated | 1081 | 2.79x smaller | 347 of 400 (0.8675) | 0.045583 | 0.188 | 0.0085 | no |
| Pruned 0.3 (NumPy equivalent) | 3013 | 1.00x | 349 of 400 (0.8725) | 0.099430 | 0.248 | 0.0085 | no |
| Pruned 0.5 (NumPy equivalent) | 3013 | 1.00x | 350 of 400 (0.8750) | 0.533220 | 0.248 | 0.0085 | no |
| Pruned 0.7 (NumPy equivalent) | 3013 | 1.00x | 319 of 400 (0.7975) | 1.955935 | 0.249 | 0.0084 | no |
| Pruned 0.3 plus INT8 per-channel | 1388 | 2.17x smaller | 350 of 400 (0.8750) | 0.099504 | 0.189 | 0.0085 | no |
| Pruned 0.5 plus INT8 per-channel | 1388 | 2.17x smaller | 351 of 400 (0.8775) | 0.533527 | 0.178 | 0.0085 | no |
| Pruned 0.5 plus INT4 packed | 1081 | 2.79x smaller | 348 of 400 (0.8700) | 0.542604 | 0.207 | 0.0086 | yes |

Notes on the table. INT8 per-channel is marked not Pareto
only because INT8 per-tensor is smaller at the same accuracy;
on fidelity (logit MSE 0.000129 against 0.000296) per-channel
is the better INT8 route, and the verdict uses both facts.
The second Pareto route, pruned 0.5 plus INT4, holds its flag
on a one-sample accuracy edge over plain INT4 (348 against
347 of 400) at the same size, while its logit MSE is about
12 times worse (0.542604 against 0.045583): at 400 samples
one sample is 0.0025 of accuracy, so that edge is noise, not
an improvement claim, and the verdict does not recommend the
combined INT4 route over plain INT4. Timing separates
nothing: batch medians span 0.178 to 0.250 ms and
single-sample medians are all about 0.0085 ms, because every
route runs the same dequantized forward pass.

#### Verdict and decision rule

The decision rule, in words (the code is
deep_system.recommend_route, and the checks below were run
against the measured table):

1. If accuracy cannot drop at all, use a quantized INT8
   route, not FP32. Both INT8 routes keep 353 of 400. Pick
   INT8 per-channel when logit fidelity matters (lowest
   quantized MSE, 0.000129, at 1388 bytes); pick INT8
   per-tensor when the byte budget is the binding constraint
   (1248 bytes, MSE 0.000296). FP32 buys nothing on accuracy
   here and costs 3013 bytes.
2. If the byte budget is under 1200 bytes, the only routes
   are the two INT4 routes at 1081 bytes. Pick plain INT4
   per-channel packed (347 of 400, MSE 0.045583) over the
   pruned combined INT4 route: the combined route gains one
   test sample and loses about 12 times on logit MSE, which
   is the worse trade at this sample count.
3. Never pick pruning alone for size. At every sparsity the
   serialized size stays 3013 bytes in dense FP32 storage,
   accuracy drops, and logit MSE rises sharply (0.533220 at
   sparsity 0.5, 1.955935 at 0.7). Pruning pays only with a
   sparse format or fine-tuning after pruning, neither of
   which is built here (v1.1 backlog).
4. Do not expect prune plus quantize to stack size gains in
   this pipeline. A combined route has exactly the size of
   its plain quantized counterpart, because pruned zeros are
   still stored as codes at the same bit width. What pruning
   changes is the weight distribution the quantizer sees,
   and here it only raised logit MSE.
5. Speed cannot decide between these routes on this setup.
   Every route runs the same dequantized NumPy forward pass,
   and the measured medians overlap across runs. A speed
   verdict needs a real quantized kernel (v1.1 backlog).

In one line: at this scale, INT8 per-channel is the default
route, INT8 per-tensor is the tight-budget route, INT4
packed is the smallest-size route with a named accuracy and
fidelity cost, and pruning in any form is not a size route
until storage or recovery changes.

#### Failure modes and limits

- Speed axis is noise under fake quantization. An early
  version of the Pareto check used size, accuracy, and batch
  time as axes, and its flags flipped between runs (the same
  route measured 0.178 to 0.315 ms across runs). The shipped
  check uses the two deterministic axes only, and this README
  makes no speed ranking claim. A real INT8 or INT4 kernel
  could separate the routes; none is measured here.
- One-sample edges are not findings. The combined INT4
  Pareto flag and the pruned 0.5 against 0.3 accuracy order
  both rest on one or two test samples out of 400. The
  verdict names them as noise wherever they appear.
- Pruning damage is in the logits, not just accuracy. At
  sparsity 0.5, accuracy drops only 3 samples while logit
  MSE rises to 0.533220, about 4,100 times the INT8
  per-channel value. A study that reported accuracy alone
  would call pruning nearly free; the MSE column shows it
  is not. No fine-tuning after pruning was done, so these
  rows are a lower bound on pruned quality, not a fair test
  of pruning with recovery.
- Calibration stayed weak, as in Build. Most calibrated
  matrices chose percentile 100.0 (no clipping). One
  exception is new: the combined INT4 route chose 98.0 for
  W2, because pruning changed that matrix distribution.
  The full chosen percentiles are in results_deep.json.
- Build integration quirk. build_system.variant_entries and
  model_from_qlab branch on variant names starting with
  pruned and treat them as dense FP32 routes, so a combined
  route name cannot be passed to them directly. The Deep
  runner passes a neutral branch name and keeps the real
  route name for files and tables. The quirk is documented
  here rather than patched in Build, because Build is a
  shipped level and its files are unchanged by Deep.
- Synthetic data and one small model, inherited from Build.
  A 676-parameter MLP on a Gaussian mixture says nothing
  numerically about a transformer, a real dataset, or a
  production kernel. Only the study shape transfers: measure
  size, accuracy, and logit fidelity together, and do not
  let a one-sample edge or a noisy timing column carry a
  verdict.
- No quantized kernel, no GPU, no GGUF, no TensorFlow: the
  four named gaps from Build all remain open in Deep and are
  the top of the v1.1 backlog in the root ROADMAP.md.

#### Tests

55 unit tests pass across the Learn, Build, and Deep suites,
plus the scaffold smoke test. Deep tests carry hand-computed
expectations for the Pareto check and the decision rule,
the combined route property that pruned zeros survive
quantization, a combined INT4 size check against FP32,
accuracy per kilobyte, and SVG chart generation from fixed
rows (titles, route labels, and point counts).

#### Jev gate

The main result paragraph above was gated with the Jev
claim_check gate on 2026-11-05. Model jev-1.13.0, input
tokens 1295, question id supported, noul 0.77, route REVIEW
(bands: CLEAR at 0.85 or above, BLOCK at 0.10 or below,
REVIEW between). Full record: level-3-deep/JEV-GATE.md.

#### What v1.1 adds next

The named backlog is the root ROADMAP.md: a real small
pretrained model, a validated GGUF build and llama.cpp run,
a TensorFlow optimization row, a GPU window on BlackBox, a
real quantized kernel timing path, pruning with sparse
storage and fine-tuning, and a sweep across model sizes.

## Appendix D1: Build claim-check record (was level-2-build/JEV-GATE.md)

### Jev gate record: P11 Build main claim, 2026-11-05

Gate: claim_check, run through the TypeSafe Jev connector
helper (typesafe_jev.py evaluate).

- Model returned: jev-1.13.0
- Input tokens: 1836
- Output tokens: 20
- Question id: supported
- Question: whether the evidence in the spec state supports
  the main result paragraph in level-2-build/README.md
  exactly, with no part of the claim going beyond the evidence.
- Noul for supported: 0.79
- Route under the claim_check bands (CLEAR at 0.85 or above,
  BLOCK at 0.10 or below, REVIEW between): REVIEW.
- Spec file:
  ~/workspace/goals/ml-ai-role-at-a-top-tier-ai-company/hidden_files/jev-p11-build-claim-spec-2026-11-05.json
- Results file:
  ~/workspace/goals/ml-ai-role-at-a-top-tier-ai-company/hidden_files/jev-p11-build-claim-results-2026-11-05.json
- HTTP status: 200.

REVIEW means the claim is neither cleared nor blocked by the
gate. The paragraph stands as written in the README with its
substitutions named inside the claim itself (QLAB not GGUF,
CPU dequantized quality numbers, NumPy pruning track), and
this record is the gate evidence. No wording in the paragraph
was changed after the gate run.

## Appendix D2: Deep claim-check record (was level-3-deep/JEV-GATE.md)

### Jev gate record: P11 Deep main claim, 2026-11-05

Gate: claim_check, run through the TypeSafe Jev connector
helper (typesafe_jev.py evaluate).

- Model returned: jev-1.13.0
- Input tokens: 1295
- Output tokens: 20
- Question id: supported
- Question: whether the evidence in the spec state supports
  the main result paragraph in level-3-deep/README.md
  exactly, with no part of the claim going beyond the evidence.
- Noul for supported: 0.77
- Route under the claim_check bands (CLEAR at 0.85 or above,
  BLOCK at 0.10 or below, REVIEW between): REVIEW.
- Arithmetic pre-computed in code before the gate run:
  3013/1248 = 2.4143 (stated 2.41x), 3013/1081 = 2.7872
  (stated 2.79x), combined INT4 logit MSE ratio
  0.542604/0.045583 = 11.90 (stated about 12 times).
- Spec file:
  ~/workspace/goals/ml-ai-role-at-a-top-tier-ai-company/hidden_files/jev-p11-deep-claim-spec-2026-11-05.json
- Results file:
  ~/workspace/goals/ml-ai-role-at-a-top-tier-ai-company/hidden_files/jev-p11-deep-claim-results-2026-11-05.json
- HTTP status: 200.

REVIEW means the claim is neither cleared nor blocked by the
gate. The paragraph stands as written in the README with its
substitutions named inside the claim itself (QLAB not GGUF,
CPU dequantized quality numbers, NumPy pruning track), and
this record is the gate evidence. No wording in the paragraph
was changed after the gate run.

## Appendix E: research review 2026-11-07 (was docs/research-review.md)

### Research review: quantization-lab

Reviewed 2026-11-07 against a doctoral research checklist. Every number below
comes from a committed results file (`level-1-learn/results_learn.json`,
`level-2-build/results_build.json`, `level-3-deep/results_deep.json`,
`level-3-deep/results_seeds.json`).

#### Summary

The repo is an honest, well-documented teaching study of weight
quantization on a 676-parameter MLP with synthetic data. Its main
weakness was statistical: one seed, one data draw, and accuracy
differences of 1 to 6 samples out of 400 read without intervals. A
second weakness was that the decision rule in code disagreed with the
written verdict. Both are fixed in this change with CPU-only work that
runs in under a second (seed study, Wilson intervals, exact McNemar
test, optional tie in `recommend_route`). What remains needs a real
model, a GPU, real kernels or other toolchains, and each item below
names exactly what it needs.

#### Checklist

| Criterion | Notes |
| --- | --- |
| Research question | Clear: what does INT8, INT4 and pruning cost in size, fidelity and speed on one small model. The scope is narrow by design. |
| Hypotheses | The Deep README states a verdict, not falsifiable hypotheses. The seed study now tests the implicit ones: "INT8 is accuracy-neutral" (0 of 10 seeds with McNemar p < 0.05 for both INT8 routes) and "pruning 0.7 hurts" (10 of 10 seeds). |
| Method | Deterministic pipeline, QLAB round trip checked, calibration by logit MSE, Pareto on deterministic axes only. Fake quantization (dequantized weights) is stated. |
| Baselines | FP32 and round-to-nearest only. No error-compensating methods (GPTQ, AWQ) or toolchain baseline (llama.cpp). Needs a real model, see remaining gaps. |
| Statistics and uncertainty | Added: Wilson 95% intervals per route, exact McNemar test vs FP32 on recorded per-sample predictions, and a 10-seed study with mean, sd and range per route. |
| Reproducibility | Deterministic results except two timing fields; CI reruns every level on Python 3.11 and 3.12. Added: seed index 0 of the seed study reproduces `results_deep.json` counts, sizes and logit MSE, asserted by a test, and the file records the NumPy version used. |
| Threats to validity | Failure modes were written. Added: the decision-rule threat (accuracy-first ranking rewards noise), quantified by the seed study. |
| Related work | `NOVELTY.md` had no measurement and `DOSSIER.md` no citations. A cited related-work section is below. |
| Limitations | Substitutions (QLAB not GGUF, NumPy pruning, CPU only) are stated in every results file and README. |

#### What this change fixed

1. **No uncertainty.** `deep_system.wilson_interval` and
   `deep_system.mcnemar_exact` (both with hand-checked tests). On the
   recorded run, every route's Wilson interval overlaps FP32's (0.847
   to 0.910) except pruned 0.7 (0.755 to 0.834). Exact McNemar vs FP32:
   INT4 packed p = 0.0703 (7 samples lost, 1 gained), pruned 0.5 plus
   INT4 p = 0.442, pruned 0.7 p = 2.4e-05 (49 lost, 15 gained).
2. **One seed.** `level-3-deep/seed_study.py` reruns all 10 routes over
   10 seed sets in about 0.4 s on CPU and writes
   `level-3-deep/results_seeds.json`. FP32 accuracy over seeds is
   0.9000 ± 0.0115, which is larger than every quantized gap. INT8
   per-tensor stays within -1 to +1 samples of FP32, INT8 per-channel
   within -2 to +1, INT4 packed within -6 to +4. Pruning 0.5 loses 3 to
   20 samples.
3. **Per-sample predictions were not recorded.** The seed study stores
   them for seed 0 (all 10 routes, 400 samples each), so paired tests
   can be recomputed from the file.
4. **Code rule vs written verdict.** `recommend_route(max_size_bytes=1200)`
   returns pruned 0.5 plus INT4 on the recorded rows, while the README
   verdict picks plain INT4. The seed study shows the code's pick is a
   one-seed artifact: under 1200 bytes the original rule picks plain
   INT4 in 9 of 10 seeds, and the recorded seed is the only exception.
   The new optional `accuracy_tie` argument (default `None`, so the
   recorded `verdict_inputs` do not change) treats accuracy within a
   tolerance as tied and then ranks by logit MSE. With
   `accuracy_tie=1/400` it picks plain INT4 in 10 of 10 seeds and on
   the recorded rows. Tests cover both.
5. **Pareto flags are seed-dependent.** Reported, not hidden: the
   recorded Pareto set (INT8 per-tensor and pruned 0.5 plus INT4)
   appears in only 1 of 10 seeds; plain INT4 packed is Pareto in 9 of
   10.
6. **README first line promised GGUF builds.** Now states the delivered
   QLAB scope and points to the backlog.
7. **Numbers are inspectable.** The live dashboard reproduces the Learn
   quantizer and the decision rule in the browser, with a build-time
   check that they equal the committed results.

#### Remaining gaps and exactly what each needs

| Gap | Why it cannot be closed here | What closes it |
| --- | --- | --- |
| Toy scale, no outlier channels | A 676-parameter MLP on Gaussian data has no real weight distribution. | A small pretrained open model with a recorded licence (for example a 100M to 1B parameter transformer) and a public evaluation set; quantizing it on CPU is feasible, evaluating perplexity on a standard set needs several hours of CPU or a GPU window. |
| No GGUF or llama.cpp validation | No llama.cpp build on the machine that recorded the results. | A machine with llama.cpp built, producing a GGUF file of the model above and running its perplexity tool. |
| Speed axis is noise | Every route runs the same dequantized NumPy forward pass. | A real INT8 or INT4 execution path (ONNX Runtime integer kernels, or llama.cpp) timed on a quiet machine, ideally the GPU node assigned to the project. |
| No GPTQ or AWQ baseline | Those methods target transformer layers with calibration data; on this MLP they would not be a fair comparison. | The pretrained model above plus the reference implementations, run in a GPU window. |
| No activation or asymmetric quantization | Not built. | CPU-feasible on this MLP, but it is a new method row with its own calibration design, so it is listed as v1.1 work, not a review fix. |
| Pruning without recovery or sparse storage | Pruned rows are a lower bound. | Fine-tuning after pruning (CPU-feasible here) plus a sparse QLAB entry kind, then a re-run; a TensorFlow Model Optimization row needs a machine with TensorFlow. |
| Synthetic data only | The task is a fixed Gaussian mixture. | At least one real dataset, which the pretrained-model item brings. |
| Results recorded with NumPy 1.26.4; local reruns use 2.4.5 | The original environment is not available here. | Re-recording `results_deep.json` on the pinned NumPy, which also rewrites the timing fields; deliberately not done so the README tables stay valid. The seed study shows NumPy 2.4.5 reproduces every recorded count and size exactly. |

#### Related work

- Jacob et al., "Quantization and Training of Neural Networks for
  Efficient Integer-Arithmetic-Only Inference", CVPR 2018. The affine
  (zero-point) integer scheme this repo's symmetric quantizer is a
  special case of.
- Krishnamoorthi, "Quantizing deep convolutional networks for efficient
  inference: A whitepaper", arXiv:1806.08342, 2018. Per-tensor vs
  per-channel weight quantization, the comparison in Build and Deep.
- Nagel et al., "A White Paper on Neural Network Quantization",
  arXiv:2106.08295, 2021. Calibration and range setting, which the
  percentile search here implements in its simplest form.
- Han, Mao and Dally, "Deep Compression", arXiv:1510.00149 (ICLR 2016).
  Pruning plus quantization with sparse storage and retraining, the two
  ingredients this study omits and names as the reason pruning does not
  shrink size here.
- Dettmers et al., "LLM.int8()", arXiv:2208.07339, 2022. Outlier features
  in large transformers, which a 676-parameter MLP cannot exhibit.
- Frantar et al., "GPTQ", arXiv:2210.17323, 2022, and Lin et al., "AWQ",
  arXiv:2306.00978, 2023. Error-compensating weight-only INT4 methods,
  the missing baselines.
- Wilson, "Probable inference, the law of succession, and statistical
  inference", JASA 1927; McNemar, "Note on the sampling error of the
  difference between correlated proportions or percentages",
  Psychometrika 1947. The interval and paired test added here.

The honest novelty is a fully reproducible, measured teaching study that
writes down its failure modes and now its uncertainty. It makes no claim
to a new quantization method.

## Appendix F: v1.1 backlog (was ROADMAP.md)

### ROADMAP: quantization-lab (P11), v1.1 backlog

Written at Deep ship time (2026-11-05) under the December
Compression Protocol. Compressed Deep shipped the quality
against size against speed study on the routes that run on
the Muse VM CPU. Everything below is the named remainder of
the v2 Deep scope. Nothing here is built, measured, or
claimed; each item names what would close it.

#### v1.1 backlog, in priority order

1. Real small pretrained model. Quantize a real pretrained
   model (a small open transformer or classifier with a
   recorded source and license) through the same pipeline,
   so the study speaks about real weight distributions and
   outlier channels. Closes the largest Deep gap.
2. GGUF build and llama.cpp run. Produce a GGUF file and
   validate it with a real parser or a llama.cpp run on a
   machine that has llama.cpp. QLAB stays the teaching
   format; GGUF is the interoperability claim, and it needs
   its own validation evidence.
3. GPU window on BlackBox. Re-run the study timing in a
   verified BlackBox window (the fleet node assigned to this
   project) and add GPU rows beside the CPU rows. Blocked
   this cycle by NEEDS-BLACKBOX-ON.
4. Real quantized kernel timing. Time an actual INT8 or
   INT4 execution path (for example ONNX Runtime or a
   llama.cpp run) so the speed axis can separate routes.
   The Deep verdict is explicit that fake quantization
   timing cannot.
5. TensorFlow optimization row. Run the TensorFlow Model
   Optimization Toolkit track from SPEC.md on a machine with
   TensorFlow, and replace the NumPy equivalent pruning rows
   with a real TensorFlow result, labelled by toolkit
   version.
6. Pruning with recovery. Add sparse storage (so pruned
   size can actually shrink) and fine-tuning after pruning,
   then re-measure the pruning and combined rows. The Deep
   study shows pruning alone, stored dense with no recovery,
   is not a size route and damages logit fidelity.
7. Scale sweep. Repeat the study across model sizes and at
   least one harder task, so the decision rule is tested
   where INT4 and pruning separate earlier than they do on
   the 676-parameter MLP.
8. Activation quantization and asymmetric modes. Deep
   quantizes weights only, symmetric only. Add activation
   quantization and a zero-point (asymmetric) variant, with
   calibration on activations.
9. GPTQ and AWQ comparison rows. Add one error-compensated
   method row beside the round-to-nearest rows in this repo,
   on the real model from item 1.

#### Standing rules for v1.1 work

- Same gates as v1.0: tests, measured benchmark tables,
  failure modes written, Jev claim_check on the main claim,
  no invented results, em dash count 0.
- GPU rows only in a verified GPU window, named in the
  results file. No GPU number is ever carried over from a
  CPU run.

## Appendix G: README before standard v2 (was README.md)

### quantization-lab

Model optimization: INT8 and INT4 quantization and pruning of a small NumPy MLP, serialized in the QLAB container (not GGUF), with a quality against size against speed study. GGUF builds are v1.1 backlog (ROADMAP.md).

Live dashboard: https://jesunahmadushno.github.io/quantization-lab/

Status: Learn DONE, Build DONE, and Deep DONE (2026-11-05, ROADMAP-V3-BATCH). v1.0 shipped. The v1.1 backlog is ROADMAP.md.
Project: P11. Fleet node: BlackBox (GPU). Wave: Wave 6. All levels were run on the Muse VM (CPU); BlackBox was unreachable in the Build and Deep cycles (NEEDS-BLACKBOX-ON). See the level READMEs for the named deductions and substitutions: the project uses the QLAB container format designed for this project (not GGUF, no llama.cpp run), and its pruning rows are a NumPy equivalent track because TensorFlow was unavailable on the build machine.

#### Start here

1. Read SPEC.md for scope and acceptance criteria.
2. Read DOSSIER.md for the past, present, and future of this problem (v1).
3. Read NOVELTY.md for the proposed novelty (DRAFT, pending the Jev claim_check gate).
4. Start at level-1-learn/README.md, then level-2-build, then level-3-deep.

#### Level map

- Level 1, Learn: the core idea, built small and clean, with a teaching path.
- Level 2, Build: a real working system other people can run.
- Level 3, Deep: benchmarks, optimization, evaluation, hard cases.

#### Benchmarks

##### Deep (Level 3)

Measured 2026-11-05 with level-3-deep/run_deep.py on the Muse VM (CPU, Python 3.12, NumPy 1.26.4). The quality against size against speed study: the Build MLP and splits, 10 routes (7 Build routes plus 3 combined prune plus quantize routes). Batch time is the median of 31 full test-set passes after 5 warmups; single time is the median of 101 single-sample passes after 5 warmups. Verdict: INT8 per-channel is the default route (full FP32 accuracy at 1388 bytes, lowest quantized logit MSE), INT4 packed is the smallest route (1081 bytes, 347 of 400), and pruning alone never shrinks the dense size. Full study, charts, and failure modes are in level-3-deep/README.md.

| Route | Size bytes | Test accuracy | Logit MSE vs FP32 | Batch ms, median of 31 | Single ms, median of 101 |
| --- | --- | --- | --- | --- | --- |
| FP32 | 3013 | 353 of 400 (0.8825) | 0.000000 | 0.250 | 0.0085 |
| INT8 per-tensor (Learn method) | 1248 | 353 of 400 (0.8825) | 0.000296 | 0.250 | 0.0085 |
| INT8 per-channel, calibrated | 1388 | 353 of 400 (0.8825) | 0.000129 | 0.192 | 0.0085 |
| INT4 per-channel packed, calibrated | 1081 | 347 of 400 (0.8675) | 0.045583 | 0.188 | 0.0085 |
| Pruned 0.3 (NumPy equivalent) | 3013 | 349 of 400 (0.8725) | 0.099430 | 0.248 | 0.0085 |
| Pruned 0.5 (NumPy equivalent) | 3013 | 350 of 400 (0.8750) | 0.533220 | 0.248 | 0.0085 |
| Pruned 0.7 (NumPy equivalent) | 3013 | 319 of 400 (0.7975) | 1.955935 | 0.249 | 0.0084 |
| Pruned 0.3 plus INT8 per-channel | 1388 | 350 of 400 (0.8750) | 0.099504 | 0.189 | 0.0085 |
| Pruned 0.5 plus INT8 per-channel | 1388 | 351 of 400 (0.8775) | 0.533527 | 0.178 | 0.0085 |
| Pruned 0.5 plus INT4 packed | 1081 | 348 of 400 (0.8700) | 0.542604 | 0.207 | 0.0086 |

##### Build (Level 2)

Measured 2026-11-05 with level-2-build/run_build.py on the Muse VM (CPU, Python 3.12, NumPy 1.26.4). The model is a 2-layer MLP (16-32-4, 676 parameters) trained with NumPy on a synthetic Gaussian mixture: train 800, calibration 200, held-out test 400. Sizes are serialized QLAB container bytes. Quality was measured with dequantized weights on CPU. Pruning rows are a NumPy equivalent track, not a TensorFlow result. Full method and failure modes are in level-2-build/README.md. Learn is unchanged by Build.

| Variant | Size bytes | Max abs weight error | Mean abs weight error | Test accuracy | Logit MSE vs FP32 | CPU time ms, median of 15 |
| --- | --- | --- | --- | --- | --- | --- |
| FP32 | 3013 | 0.000000 | 0.000000 | 353 of 400 (0.8825) | 0.000000 | 0.258 |
| INT8 per-tensor (Learn method) | 1248 | 0.005103 | 0.002292 | 353 of 400 (0.8825) | 0.000296 | 0.258 |
| INT8 per-channel, calibrated | 1388 | 0.004610 | 0.001267 | 353 of 400 (0.8825) | 0.000129 | 0.195 |
| INT4 per-channel packed, calibrated | 1081 | 0.083784 | 0.023948 | 347 of 400 (0.8675) | 0.045583 | 0.193 |
| Pruned 0.3 (NumPy equivalent) | 3013 | 0.133321 | 0.018083 | 349 of 400 (0.8725) | 0.099430 | 0.256 |
| Pruned 0.5 (NumPy equivalent) | 3013 | 0.218123 | 0.052308 | 350 of 400 (0.8750) | 0.533220 | 0.260 |
| Pruned 0.7 (NumPy equivalent) | 3013 | 0.341136 | 0.106046 | 319 of 400 (0.7975) | 1.955935 | 0.254 |

##### Learn (Level 1)

Measured 2026-11-05 with level-1-learn/run_learn.py on the Muse VM, Python 3.12. The model is a deterministic 4 by 3 linear model, not a real model.

| Test | Result |
| --- | --- |
| Weights quantized | 12 weights, symmetric per-tensor, FP32 vs INT8 vs INT4 packed |
| Size, total bytes | FP32 48, INT8 16 (3.0x smaller), INT4 10 (4.8x smaller), each quantized total includes one 4 byte FP32 scale |
| INT8 error | max abs 0.003937, mean abs 0.001640, MSE 0.000005 |
| INT4 error | max abs 0.071429, mean abs 0.029762, MSE 0.001488 |
| Synthetic task (8 samples) | FP32 8 of 8, INT8 8 of 8, INT4 8 of 8; score MAE vs FP32: INT8 0.002707, INT4 0.049107 |
| Unit tests | 23 passed, plus the scaffold smoke test |
| Determinism | results_learn.json identical across two runs (hash verified) |

##### Seed study and uncertainty (Level 3 add-on)

level-3-deep/seed_study.py reruns the same 10 routes over 10 seed sets (training init and the train, calibration and test draws vary; class centers fixed) and writes level-3-deep/results_seeds.json, with no timing. Seed index 0 reproduces results_deep.json exactly (a test asserts it). It records per-sample predictions for seed 0, the exact McNemar test of every route against FP32, and Wilson 95% intervals. Across seeds, INT8 per-tensor and per-channel stay within 2 samples of FP32 and never differ significantly; pruned 0.7 is significantly worse in 10 of 10 seeds. Under a 1200 byte budget, recommend_route picks plain INT4 in 9 of 10 seeds; the recorded seed is the one exception. The new optional accuracy_tie argument (1/400, one sample) makes the code pick plain INT4 on the recorded rows, matching the verdict. The research review is in docs/research-review.md.

#### Fan-out

Written fan-out set for v1.0, in fanout/: blog post, tutorial,
website page copy, LinkedIn post draft (DRAFT NOT POSTED),
paper candidate brief, and release notes covering v0.1, v0.5,
and v1.0. The full channel log is in PUBLICITY.md.

#### Novelty

PROPOSED. See NOVELTY.md. The novelty text is a draft until the Jev claim_check gate passes on it.

#### License

Code: Apache-2.0 (see LICENSE). Tutorials: CC BY 4.0. Ruled in D-V3-11, 2026-11-04.
