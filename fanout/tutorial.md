# Tutorial: P11 quantization-lab, course module

Module: P11-quantization-lab
By Jesun Ahmad Ushno
Date: 2026-11-05
License for this tutorial: CC BY 4.0. Code in the repo: Apache-2.0.

## What you will do

You will clone the quantization-lab repo, run all three levels,
run the test suite for each level, and read your own results
against the recorded v1.0 numbers. Every command below exists in
the repo as shipped at v1.0 and was verified by running it in
this fan-out cycle. Level 1 needs Python 3.11 or 3.12 and the
standard library only. Levels 2 and 3 and their tests need
NumPy (see requirements.txt, pinned to the measured version
1.26.4).

Three scope facts govern every step. First, the model is a small
MLP trained in this repo on synthetic data, NOT a real pretrained
model. Second, sizes are QLAB container bytes: QLAB is a format
designed for this project, NOT GGUF, and nothing here runs in
llama.cpp. Third, the pruning rows are a NumPy equivalent track,
NOT a TensorFlow result.

## Step 0: Clone and enter the repo

```text
git clone https://github.com/JesunAhmadUshno/quantization-lab.git
cd quantization-lab
python -m pip install -r requirements.txt
```

Optional smoke check, before the levels:

```text
python tests/test_smoke.py
```

Expected: a line that says the scaffold smoke test passed, with
8 files present.

## Step 1: Level 1, Learn, symmetric quantization by hand

Read level-1-learn/README.md first. It defines the 12-weight
model, the symmetric per-tensor scale, the rounding rule, and
the INT4 packing.

Run the Learn runner:

```text
python level-1-learn/run_learn.py
```

Run the Learn tests:

```text
python -m unittest tests.test_learn_quantization -v
```

Read your results:

- The runner writes level-1-learn/results_learn.json.
- Expected sizes: FP32 48 total bytes, INT8 16 total bytes
  (3.0x smaller), INT4 10 total bytes (4.8x smaller). Each
  quantized total includes one 4 byte FP32 scale.
- Expected errors: INT8 max abs 0.003937, mean abs 0.001640,
  MSE 0.000005; INT4 max abs 0.071429, mean abs 0.029762,
  MSE 0.001488.
- Expected task: FP32, INT8, and INT4 each score 8 of 8 on the
  synthetic task. The task saturates; the score MAE (INT8
  0.002707, INT4 0.049107) is what separates the formats here.
- Expected tests: 23 tests, all passing.
- The results file is deterministic: run the runner twice and
  the file is identical.

## Step 2: Level 2, Build, quantize and prune a trained MLP

Read level-2-build/README.md and level-2-build/BUILD-SPEC.md
first. They define the MLP, the splits, the calibration search,
the QLAB container, and every substitution named in the level
(no GGUF, no TensorFlow result, no GPU).

Run the Build runner:

```text
python level-2-build/run_build.py
```

Run the Build tests:

```text
python -m unittest tests.test_build_system -v
```

Read your results:

- The runner writes level-2-build/results_build.json and QLAB
  files under level-2-build/artifacts/.
- Expected headline rows: FP32 3013 bytes at 353 of 400
  (0.8825); INT8 per-tensor 1248 bytes at 353 of 400; INT8
  per-channel, calibrated, 1388 bytes at 353 of 400 with mean
  abs weight error 0.001267; INT4 per-channel packed,
  calibrated, 1081 bytes at 347 of 400 (0.8675).
- Expected pruning rows: 3013 bytes at every sparsity (0.3,
  0.5, 0.7), the same size as FP32, because pruned models are
  stored dense. Pruning saved no bytes in this pipeline.
- Expected tests: 20 Build tests, all passing (43 unit tests in
  total with Learn).
- The results file is deterministic except the timing fields:
  timing is the dequantized NumPy forward pass and supports no
  speed claim.

## Step 3: Level 3, Deep, the quality, size, speed study

Read level-3-deep/README.md and level-3-deep/DEEP-SPEC.md
first. They define the 10 routes, the Pareto check, and the
decision rule.

Run the Deep runner:

```text
python level-3-deep/run_deep.py
```

Run the Deep tests:

```text
python -m unittest tests.test_deep_system -v
```

Read your results:

- The runner writes level-3-deep/results_deep.json, QLAB files
  under level-3-deep/artifacts/, and the three charts under
  level-3-deep/charts/.
- Expected study rows: the 7 Build routes reproduce their Build
  sizes and accuracies exactly, plus 3 combined routes. Pruned
  0.5 plus INT8 per-channel is 1388 bytes at 351 of 400; pruned
  0.5 plus INT4 packed is 1081 bytes at 348 of 400 with logit
  MSE 0.542604, about 12 times the plain INT4 value of 0.045583.
- Expected verdict: INT8 per-channel is the default route (full
  FP32 accuracy at 1388 bytes, lowest quantized logit MSE
  0.000129), INT4 packed is the smallest route, and pruning
  alone never shrinks the dense size.
- Expected tests: 12 Deep tests, all passing (55 unit tests in
  total).
- Batch and single timing fields are the only nondeterministic
  fields in the results file. Timing separates nothing here,
  because every route runs the same dequantized forward pass.

## Step 4: Run the full suite the way CI does

Run every test and every runner in one pass:

```text
python -m unittest discover -s tests -v
python level-1-learn/run_learn.py
python level-2-build/run_build.py
python level-3-deep/run_deep.py
```

Expected: 55 unit tests pass, the scaffold smoke test passes,
and the results files are unchanged after the runners apart
from the Build and Deep timing fields.

## What to try next

- In results_learn.json, check the INT4 packed bytes (199, 194,
  231, 178, 183, 158) against the hand check in the Learn
  README.
- In results_build.json, compare the per-tensor and per-channel
  INT8 rows: 140 extra bytes of scales buy a lower mean weight
  error and a lower logit MSE at the same accuracy.
- In results_deep.json, verify that every combined route has
  exactly the size of its plain quantized counterpart, then
  read the decision rule in level-3-deep/deep_system.py.
- The v1.1 backlog in ROADMAP.md names what this tutorial does
  not cover: a real pretrained model, a GGUF build, a GPU
  window, a real quantized kernel, and a TensorFlow row. None
  of that is shipped at v1.0.
