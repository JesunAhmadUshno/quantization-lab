# quantization-lab

Model optimization: INT8 and INT4, GGUF builds, a quality against size against speed study.

Status: Learn DONE, Build DONE, and Deep DONE (2026-11-05, ROADMAP-V3-BATCH). v1.0 shipped. The v1.1 backlog is ROADMAP.md.
Project: P11. Fleet node: BlackBox (GPU). Wave: Wave 6. All levels were run on the Muse VM (CPU); BlackBox was unreachable in the Build and Deep cycles (NEEDS-BLACKBOX-ON). See the level READMEs for the named deductions and substitutions: the project uses the QLAB container format designed for this project (not GGUF, no llama.cpp run), and its pruning rows are a NumPy equivalent track because TensorFlow was unavailable on the build machine.

## Start here

1. Read SPEC.md for scope and acceptance criteria.
2. Read DOSSIER.md for the past, present, and future of this problem (v1).
3. Read NOVELTY.md for the proposed novelty (DRAFT, pending the Jev claim_check gate).
4. Start at level-1-learn/README.md, then level-2-build, then level-3-deep.

## Level map

- Level 1, Learn: the core idea, built small and clean, with a teaching path.
- Level 2, Build: a real working system other people can run.
- Level 3, Deep: benchmarks, optimization, evaluation, hard cases.

## Benchmarks

### Deep (Level 3)

Measured 2026-11-05 with level-3-deep/run_deep.py on the Muse VM (CPU, Python 3.12, NumPy 1.26.4). The quality against size against speed study: the Build MLP and splits, 10 routes (7 Build routes plus 3 combined prune plus quantize routes). Batch time is the median of 31 full test-set passes after 5 warmups; single time is the median of 101 single-sample passes after 5 warmups. Verdict: INT8 per-channel is the default route (full FP32 accuracy at 1388 bytes, lowest quantized logit MSE), INT4 packed is the smallest route (1081 bytes, 347 of 400), and pruning alone never shrinks the dense size. Full study, charts, failure modes, and grade (7,450 of 10,000) are in level-3-deep/README.md.

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

### Build (Level 2)

Measured 2026-11-05 with level-2-build/run_build.py on the Muse VM (CPU, Python 3.12, NumPy 1.26.4). The model is a 2-layer MLP (16-32-4, 676 parameters) trained with NumPy on a synthetic Gaussian mixture: train 800, calibration 200, held-out test 400. Sizes are serialized QLAB container bytes. Quality was measured with dequantized weights on CPU. Pruning rows are a NumPy equivalent track, not a TensorFlow result. Full method, failure modes, and grade (7,600 of 10,000) are in level-2-build/README.md. Learn is unchanged by Build.

| Variant | Size bytes | Max abs weight error | Mean abs weight error | Test accuracy | Logit MSE vs FP32 | CPU time ms, median of 15 |
| --- | --- | --- | --- | --- | --- | --- |
| FP32 | 3013 | 0.000000 | 0.000000 | 353 of 400 (0.8825) | 0.000000 | 0.258 |
| INT8 per-tensor (Learn method) | 1248 | 0.005103 | 0.002292 | 353 of 400 (0.8825) | 0.000296 | 0.258 |
| INT8 per-channel, calibrated | 1388 | 0.004610 | 0.001267 | 353 of 400 (0.8825) | 0.000129 | 0.195 |
| INT4 per-channel packed, calibrated | 1081 | 0.083784 | 0.023948 | 347 of 400 (0.8675) | 0.045583 | 0.193 |
| Pruned 0.3 (NumPy equivalent) | 3013 | 0.133321 | 0.018083 | 349 of 400 (0.8725) | 0.099430 | 0.256 |
| Pruned 0.5 (NumPy equivalent) | 3013 | 0.218123 | 0.052308 | 350 of 400 (0.8750) | 0.533220 | 0.260 |
| Pruned 0.7 (NumPy equivalent) | 3013 | 0.341136 | 0.106046 | 319 of 400 (0.7975) | 1.955935 | 0.254 |

### Learn (Level 1)

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

## Novelty

PROPOSED. See NOVELTY.md. The novelty text is a draft until the Jev claim_check gate passes on it.

## License

Code: Apache-2.0 (see LICENSE). Tutorials: CC BY 4.0. Ruled in D-V3-11, 2026-11-04.
