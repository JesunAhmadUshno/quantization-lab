# ROADMAP: quantization-lab (P11), v1.1 backlog

Written at Deep ship time (2026-11-05) under the December
Compression Protocol. Compressed Deep shipped the quality
against size against speed study on the routes that run on
the Muse VM CPU. Everything below is the named remainder of
the v2 Deep scope. Nothing here is built, measured, or
claimed; each item names what would close it.

## v1.1 backlog, in priority order

1. Real small pretrained model. Quantize a real pretrained
   model (a small open transformer or classifier with a
   recorded source and license) through the same pipeline,
   so the study speaks about real weight distributions and
   outlier channels. Closes the largest Deep deduction.
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

## Standing rules for v1.1 work

- Same gates as v1.0: tests, measured benchmark tables,
  failure modes written, Jev claim_check on the main claim,
  no invented results, em dash count 0.
- GPU rows only in a verified GPU window, named in the
  results file. No GPU number is ever carried over from a
  CPU run.
