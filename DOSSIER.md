# DOSSIER: quantization-lab (P11)

Written under the documentation and publishing standard
(past, present, future). Updated at Deep ship time,
2026-11-05, to ship state for v1.0.

## Past

Prior art for this project is named in NOVELTY.md
(quantization toolchains (llama.cpp, GPTQ and AWQ tooling)).
The history of the problem, the key papers and systems, and
what was tried before are recorded in the Roadmap v2 section
text in SPEC.md and will be expanded with dated sources at
the project's wave, before Build opens.

## Present

Current demand evidence: hiring value 3 of 5 (1 named posting family, xAI preference) [confirmed, fleet and scoring pass 2026-11-04; for P23 to P29, the decision pack 2026-11-04]. The Round 3 research (2026-11-04) found the market centre at post-training, evals, environments, and inference depth [confirmed as the research finding; the report is at ~/workspace/research_notes/roadmap-v3-revaluation-research-20261104-0949/report.md]. Tools and versions are pinned at Build time, not in this v1.

Shipped state (2026-11-05, v1.0): Learn, Build, and
compressed Deep are DONE. Learn: symmetric per-tensor INT8
and INT4 on a 12-weight teaching model (grade 8,700 of
10,000). Build: a NumPy-trained MLP (16-32-4, 676
parameters) quantized to INT8 and INT4, magnitude pruned,
serialized in the QLAB container format designed for this
project, and benchmarked on CPU (grade 7,600 of 10,000).
Deep: the quality against size against speed study over 10
routes, adding 3 combined prune plus quantize routes, three
code-generated charts, a Pareto analysis on size and
accuracy, and a decision rule implemented and tested as
code (grade 7,450 of 10,000). Headline Deep finding: both
INT8 routes keep full FP32 accuracy (353 of 400) at 2.41x
and 2.17x smaller size; INT4 packed is smallest (1081
bytes, 2.79x) at 347 of 400; pruning alone never shrinks
the dense serialized size, and prune plus quantize does not
stack size gains in this pipeline. All three levels ran on
the Muse VM CPU. Named gaps that remain: no real
pretrained model, no GGUF build or llama.cpp run, no
TensorFlow result (NumPy equivalent pruning track instead),
no GPU run (BlackBox unreachable, NEEDS-BLACKBOX-ON), and
no quantized kernel timing. Full method, tables, failure
modes, and Jev gate records are in the level READMEs.

## Future

Open problems and the project's next steps sit in the Deep level scope in SPEC.md. Any forecast about this field is [unverified] unless a dated primary source is added here at the wave update.

The v1.1 future for this project is the root ROADMAP.md:
a real pretrained model, a validated GGUF build, a GPU
window, a real quantized kernel timing path, the TensorFlow
row, pruning with sparse storage and fine-tuning, a scale
sweep, activation and asymmetric quantization, and GPTQ and
AWQ comparison rows. None of it is claimed as built.
