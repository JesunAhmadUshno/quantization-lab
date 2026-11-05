# quantization-lab

Model optimization: INT8 and INT4, GGUF builds, a quality against size against speed study.

Status: Learn DONE (2026-11-05, ROADMAP-V3-BATCH). Level 1 is built, tested, and measured below. Build and Deep are not started.
Project: P11. Fleet node: BlackBox (GPU). Wave: Wave 6. Learn was run on the Muse VM (CPU); see the level README for the named deduction.

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
