# Release notes: quantization-lab (P11)

One section per release line. Every number below comes from the
level README of that release and the release text on GitHub,
verified with gh release list on 2026-11-05. The model in every
release is a small MLP trained in this repo on synthetic data,
NOT a real pretrained model. Sizes are QLAB container bytes
(QLAB is a format designed for this project, not GGUF). The full
release text and artifacts live on the GitHub releases page:
https://github.com/JesunAhmadUshno/quantization-lab/releases

## v0.1, Learn (2026-11-05)

- What shipped: symmetric per-tensor INT8 and INT4 quantization
  of one deterministic 4 by 3 linear model (12 weights), built
  with the Python standard library only, with hand-checked codes,
  packing, sizes, and errors, plus 23 unit tests and the scaffold
  smoke test.
- Measured result: FP32 48 total bytes, INT8 16 total bytes
  (3.0x smaller, max abs error 0.003937), INT4 packed 10 total
  bytes (4.8x smaller, max abs error 0.071429), and FP32, INT8,
  and INT4 each scoring 8 of 8 on the synthetic task.
- Release: https://github.com/JesunAhmadUshno/quantization-lab/releases/tag/v0.1

## v0.5, Build (2026-11-05)

- What shipped: a 2-layer MLP (16-32-4, 676 parameters) trained
  with NumPy on a synthetic Gaussian mixture, quantized to INT8
  per-tensor and calibrated INT8 and INT4 per-channel, magnitude
  pruned on a NumPy equivalent track, and serialized in the QLAB
  container, plus 20 new tests (43 unit tests in total with
  Learn).
- Measured result: FP32 scored 353 of 400 (0.8825) at 3013 bytes;
  INT8 per-tensor kept 353 of 400 at 1248 bytes; INT8 per-channel
  kept 353 of 400 at 1388 bytes; INT4 packed scored 347 of 400
  (0.8675) at 1081 bytes; pruned variants stayed at 3013 bytes at
  every sparsity in dense storage.
- Release: https://github.com/JesunAhmadUshno/quantization-lab/releases/tag/v0.5

## v1.0, Learn, Build, and compressed Deep (2026-11-05)

- What shipped: the quality against size against speed study
  over 10 routes (the 7 Build routes plus 3 combined prune plus
  quantize routes), three code-generated charts, a Pareto
  analysis on size and accuracy, and a decision rule implemented
  and tested as code, plus 12 new tests (55 unit tests in total,
  plus the scaffold smoke test). The rest of the v2 Deep scope
  is named in ROADMAP.md as the v1.1 backlog.
- Measured result: INT8 per-channel is the default route (353 of
  400 at 1388 bytes, lowest quantized logit MSE 0.000129), INT8
  per-tensor is the tight-budget route (353 of 400 at 1248
  bytes), INT4 packed is the smallest route (1081 bytes, 347 of
  400), pruning alone never shrinks the dense size, and combined
  routes do not shrink size beyond their plain quantized
  counterparts.
- Release: https://github.com/JesunAhmadUshno/quantization-lab/releases/tag/v1.0

## Held channels

- Social posts for every release above: HELD under the standing
  platform rules. The loop prepares drafts and never posts;
  Threads needs Jesun's one tap per post.
- The v1.1 backlog lives in ROADMAP.md: a real small pretrained
  model, a validated GGUF build and llama.cpp run, a GPU window
  on BlackBox, a real quantized kernel timing path, a TensorFlow
  row, pruning with sparse storage and fine-tuning, a scale
  sweep, activation and asymmetric quantization, and GPTQ and
  AWQ comparison rows.
