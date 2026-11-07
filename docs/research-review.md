# Research review: quantization-lab

Reviewed 2026-11-07 against a doctoral research rubric. Every number below
comes from a committed results file (`level-1-learn/results_learn.json`,
`level-2-build/results_build.json`, `level-3-deep/results_deep.json`,
`level-3-deep/results_seeds.json`). Grades: A strong, B adequate, C weak,
D missing.

## Summary

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

## Rubric

| Criterion | Before | After | Notes |
| --- | --- | --- | --- |
| Research question | B | B | Clear: what does INT8, INT4 and pruning cost in size, fidelity and speed on one small model. The scope is narrow by design. |
| Hypotheses | C | B | The Deep README states a verdict, not falsifiable hypotheses. The seed study now tests the implicit ones: "INT8 is accuracy-neutral" (0 of 10 seeds with McNemar p < 0.05 for both INT8 routes) and "pruning 0.7 hurts" (10 of 10 seeds). |
| Method | B | B | Deterministic pipeline, QLAB round trip checked, calibration by logit MSE, Pareto on deterministic axes only. Fake quantization (dequantized weights) is stated. |
| Baselines | C | C | FP32 and round-to-nearest only. No error-compensating methods (GPTQ, AWQ) or toolchain baseline (llama.cpp). Needs a real model, see remaining gaps. |
| Statistics and uncertainty | D | B | Added: Wilson 95% intervals per route, exact McNemar test vs FP32 on recorded per-sample predictions, and a 10-seed study with mean, sd and range per route. |
| Reproducibility | A | A | Deterministic results except two timing fields; CI reruns every level on Python 3.11 and 3.12. Added: seed index 0 of the seed study reproduces `results_deep.json` counts, sizes and logit MSE, asserted by a test, and the file records the NumPy version used. |
| Threats to validity | B | A | Failure modes were written. Added: the decision-rule threat (accuracy-first ranking rewards noise), quantified by the seed study. |
| Related work | D | B | `NOVELTY.md` had no measurement and `DOSSIER.md` no citations. A cited related-work section is below. |
| Limitations | A | A | Substitutions (QLAB not GGUF, NumPy pruning, CPU only) are stated in every results file and README. |

## What this change fixed

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

## Remaining gaps and exactly what each needs

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

## Related work

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
