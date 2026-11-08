# Website project page copy: quantization-lab

Site: jesunahmadushno.com
Status: COPY ONLY, draft for the project page. Dated 2026-11-05.
Nothing on any website was changed by this file.

## Headline

quantization-lab: smaller, faster, honest - a quality against size against speed study

## Summary

quantization-lab is an open model optimization project built by
Jesun Ahmad Ushno as project P11 in Roadmap v3. It teaches
symmetric quantization on a 12-weight model, trains a small MLP
with NumPy and quantizes it to INT8 and INT4, prunes it, stores
it in the QLAB container format designed for this project, and
then runs a 10-route study of quality against size against
speed, ending in a decision rule implemented and tested as code.

Scope facts, stated plainly. The model is a small MLP trained in
this repo on synthetic data, NOT a real pretrained model. QLAB
is NOT GGUF, and no file here runs in llama.cpp. Pruning rows
are a NumPy equivalent track, NOT a TensorFlow result. Timing
is the dequantized NumPy forward pass on CPU, NOT a real
quantized kernel.

Status: v1.0, 2026-11-05. Learn, Build, and compressed Deep are
done. Total tests: 55 unit tests, plus the scaffold
smoke test, all passing at v1.0.

## Results

All results below are measured runs recorded in the level
READMEs and the results files in the GitHub repo. They describe
a 676-parameter MLP on synthetic data, NOT a real model, and
they prove the study discipline, not production model quality.

| Level | Set | Key result | Notes |
| --- | --- | --- | --- |
| Learn | 12 weights, 8 task samples | FP32 48 bytes, INT8 16 bytes, INT4 10 bytes, all 8 of 8 | INT8 max abs error 0.003937, INT4 0.071429; 23 tests |
| Build | 400 held-out samples | FP32 353 of 400 at 3013 bytes; INT8 per-tensor 353 of 400 at 1248 bytes; INT8 per-channel 353 of 400 at 1388 bytes; INT4 packed 347 of 400 at 1081 bytes | Pruned rows stay 3013 bytes in dense storage; 20 tests, 43 total |
| Deep (compressed) | 10 routes on the Build model | INT8 per-channel is the default route: 353 of 400 at 1388 bytes, logit MSE 0.000129; INT4 packed is smallest at 1081 bytes | Combined routes do not shrink size beyond their plain quantized counterparts; 12 tests, 55 total |

## Honest limits

- Synthetic data only, one small architecture, one dataset.
- No real pretrained model, no GGUF build, no llama.cpp run.
- No TensorFlow result: pruning is a NumPy equivalent track.
- No GPU run: all levels ran on the Muse VM CPU; BlackBox was
  unreachable (NEEDS-BLACKBOX-ON).
- No quantized kernel: the speed axis is measured but cannot
  separate routes under fake quantization.
- Compressed Deep is not the full v2 Deep scope. The remainder
  is the v1.1 backlog in the repo ROADMAP.md.

## Links

- GitHub repo: https://github.com/JesunAhmadUshno/quantization-lab
- Release v0.1 (Learn): https://github.com/JesunAhmadUshno/quantization-lab/releases/tag/v0.1
- Release v0.5 (Build): https://github.com/JesunAhmadUshno/quantization-lab/releases/tag/v0.5
- Release v1.0 (Learn, Build, compressed Deep): https://github.com/JesunAhmadUshno/quantization-lab/releases/tag/v1.0
- Tutorial: fanout/tutorial.md in the GitHub repo
- Blog post draft: fanout/blog-post.md in the GitHub repo

## License note

Code: Apache-2.0. Tutorials: CC BY 4.0.
