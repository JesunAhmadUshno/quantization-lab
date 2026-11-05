# Level 1, Learn: quantization-lab

Status: DONE (2026-11-05). Symmetric per-tensor INT8 and INT4
quantization of one small deterministic linear model, built with
the Python standard library only.

## Honest scope line

There is no real model in this level, no GGUF, no llama.cpp, and
no GPU. The model is a hand-built 4 by 3 weight matrix (12 weights)
and the task is a small synthetic labelled set. Those limits are
stated plainly here, not hidden. A real model, GGUF builds, and
llama.cpp runs are Build level scope, and a quality against size
against speed study on real models is Deep level scope. This level
was run on the Muse VM, not on BlackBox, the assigned GPU fleet
node: Learn is CPU teaching work by design, one fleet SSH probe
was attempted and did not connect from this VM, and the grade
below carries a named deduction for the Muse VM run.

## Method

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

## Run it

    python level-1-learn/run_learn.py
    python -m unittest tests.test_learn_quantization -v
    python tests/test_smoke.py

The runner writes level-1-learn/results_learn.json deterministically:
run it twice and the file is identical (verified 2026-11-05:
sha256 2291d1d3295b83ee8fc204a90f5bf3c4bcca04274a04a664251c8f48b544f120
on both runs).

## Results (measured 2026-11-05, Muse VM, Python 3.12)

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

## Failure modes and limits

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
  BlackBox. That costs a grade deduction, named in the grade.

## Grade

Grade: 8,700 of 10,000 for this Learn level.

Deductions (they sum to 1,300):
- Small synthetic designed set, one 12-weight model and 8 task
  samples: minus 500.
- Deterministic toy model, not a real model, with no GGUF, no
  llama.cpp, and no inference speed measured: minus 400.
- Run on the Muse VM, not the assigned BlackBox fleet node:
  minus 200.
- Per-tensor symmetric quantization only, with no per-channel
  scales, no asymmetric mode, and no calibration: minus 100.
- Synthetic task saturates at 8 of 8 for every format, so the
  accuracy number does not separate INT4 from FP32 at this
  level: minus 100.

## What Build adds next

A real small model quantized to INT8 and INT4, GGUF builds run
locally with llama.cpp, per-channel scales beside per-tensor, and
a quality task that does not saturate. See the root README level
map.
