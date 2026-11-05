# quantization-lab

Model optimization: INT8 and INT4 quantization, magnitude pruning, and a quality against size against speed study, built in three levels with measured results at each level. The model is a small NumPy-trained MLP on synthetic data, not a real pretrained model, and that is stated plainly in every level README.

- Learn: symmetric per-tensor INT8 and INT4 on a deterministic 12-weight model. FP32 48 bytes, INT8 16 bytes (3.0x smaller), INT4 10 bytes (4.8x smaller), all three scoring 8 of 8 on the synthetic task.
- Build: a 2-layer MLP (16-32-4, 676 parameters) quantized and pruned, serialized in the QLAB container format designed for this project (not GGUF). FP32 scored 353 of 400 (0.8825) at 3013 bytes; INT8 per-tensor kept 353 of 400 at 1248 bytes; INT8 per-channel kept 353 of 400 at 1388 bytes; INT4 packed scored 347 of 400 (0.8675) at 1081 bytes.
- Deep (compressed): the quality against size against speed study over 10 routes. INT8 per-channel is the default route (full FP32 accuracy at 1388 bytes, lowest quantized logit MSE 0.000129), INT4 packed is the smallest route (1081 bytes, 347 of 400), and pruning alone never shrinks the dense serialized size (3013 bytes at every sparsity).

Grades on record (of 10,000): Learn 8,700, Build 7,600, Deep 7,450.

Every number above is already published in this repo (README.md and the level READMEs) and reproduced by the runners in this repo, checked in CI. The full write-ups are in the level folders: level-1-learn, level-2-build, level-3-deep.

Releases: https://github.com/JesunAhmadUshno/quantization-lab/releases
