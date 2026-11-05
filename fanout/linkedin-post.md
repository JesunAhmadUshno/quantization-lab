# LinkedIn post draft: P11 quantization-lab v1.0

Status: DRAFT NOT POSTED
Date drafted: 2026-11-05
Author: Jesun Ahmad Ushno
Rule for this platform: the link goes in the first comment, not in the
post body. The first comment line is at the end of this file.

## Post body

Shrinking a model is easy to claim. Proving what the shrink cost
is the work. So I measured quality, size, and speed together, and
wrote down where each route fails.

quantization-lab v1.0 is done: Learn, Build, and a compressed
Deep, all open, all measured, on CPU with NumPy.

The scope, up front: the model is a small MLP (676 parameters)
trained in the repo on synthetic data, NOT a real pretrained
model. Sizes are QLAB container bytes: QLAB is a format designed
for this project, NOT GGUF. Pruning is a NumPy equivalent track,
NOT a TensorFlow result. Timing is the dequantized forward pass,
NOT a real quantized kernel.

Learn: symmetric INT8 and INT4 on a 12-weight teaching model.
FP32 48 bytes, INT8 16 bytes, INT4 10 bytes, all scoring 8 of 8,
with the errors hand-checkable.

Build: the MLP quantized and pruned. FP32 scored 353 of 400
(0.8825) at 3013 bytes. INT8 per-tensor kept 353 of 400 at 1248
bytes. INT8 per-channel kept 353 of 400 at 1388 bytes with the
lowest quantized logit MSE. INT4 packed scored 347 of 400
(0.8675) at 1081 bytes.

Deep: a 10-route study with a decision rule in code. INT8
per-channel is the default route. INT4 packed is the smallest
route with a named cost. And pruning alone saved no bytes at any
sparsity: stored dense, every pruned model stays 3013 bytes, and
prune plus quantize does not stack size gains.

The honest limits, in the post and not in a footnote: synthetic
data only, no GGUF or llama.cpp run, no TensorFlow result, no
GPU run, and no quantized kernel, so the speed axis cannot
separate these routes. 55 tests in total. The rest of the Deep
scope is the v1.1 backlog in the repo, not claimed as done.

Grades, house scale: Learn 8,700, Build 7,600, Deep 7,450 of
10,000.

If you optimize models: which do you trust first when a format
gets smaller, the accuracy, the logit fidelity, or the byte
count?

## First comment (post this as the first comment, per the standing rule)

Repo, releases v0.1, v0.5, and v1.0, and the full tutorial: https://github.com/JesunAhmadUshno/quantization-lab
