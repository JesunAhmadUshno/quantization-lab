# Proposal: quantization-lab (P11)

Status: NOT PRE-REGISTERED. Written up 2026-11-08 under the portfolio
standard v2 from the plan files that existed before. No proposal tag
exists and none is added, because no plan with hypotheses and pass
lines was committed before the first result.

What git history shows:

- a7ff5b6 (2026-11-04), before any code: SPEC.md, the scope and
  acceptance criteria (Appendix A). It names what to build (INT8 and
  INT4 models, GGUF builds run in llama.cpp, a TensorFlow optimization
  row, and a quality against size against speed study with a written
  verdict). It has no hypotheses and no numeric pass lines, so it is a
  scope, not a pre-registration.
- f97cd57 (2026-11-05): BUILD-SPEC.md (Appendix D) says it was written
  before the Build code, but it was committed in the same commit as the
  code and results_build.json, so git cannot prove the order.
- 35eaf6f (2026-11-05): DEEP-SPEC.md (Appendix E), same situation as
  BUILD-SPEC.md, committed with results_deep.json.

So the repo carries the "not pre-registered" label and is capped at
Functional (METHOD section 7) until a confirmatory cycle runs under a
newly locked proposal.

## Question

At small-model scale, what do symmetric INT8 and INT4 weight
quantization and magnitude pruning cost in serialized size, accuracy
and logit fidelity, and which route should a practitioner pick under a
given byte budget?

## Prior work

Named in Appendix B (NOVELTY.md) and Appendix C (DOSSIER.md); the
cited related-work list (Jacob et al. 2018, Krishnamoorthi 2018, Nagel
et al. 2021, Han et al. 2016, Dettmers et al. 2022, GPTQ, AWQ, Wilson
1927, McNemar 1947) is in REPORT.md, Appendix E (research review).
The honest novelty is a reproducible, measured teaching study with its
failure modes and uncertainty written down; it claims no new method.

## Hypotheses (stated after the results; not a lock)

These are the claims the Build and Deep write-ups and the 2026-11-07
research review actually tested. They are written down here so
REPORT.md can score each one, not to pretend they came first.

| ID | Hypothesis | Pass line used | Source |
| --- | --- | --- | --- |
| H1 | INT8 weight quantization (per-tensor and per-channel) is accuracy-neutral against FP32 | exact McNemar p >= 0.05 against FP32 in every one of 10 seeds | Deep verdict rule 1, research review |
| H2 | INT4 packed is the smallest route, at a measurable accuracy cost | smallest serialized size; accuracy cost significant (McNemar p < 0.05) | Deep verdict rule 2 |
| H3 | Pruning alone is not a size route, and heavy pruning hurts accuracy | dense size equals FP32 at every sparsity; pruned 0.7 McNemar p < 0.05 in at least 9 of 10 seeds | Deep verdict rule 3, research review |
| H4 | Prune plus quantize does not stack size gains | combined route size equals its plain quantized route | Deep verdict rule 4 |
| H5 | The speed axis separates routes | batch-time medians separate routes consistently across runs | Deep scope (SPEC.md "speed") |
| H6 | The SPEC.md v1 core runs: GGUF builds in llama.cpp, a TensorFlow optimization row | a GGUF file runs in llama.cpp; a TensorFlow Model Optimization row in the study table | SPEC.md |
| H7 | The decision rule in code agrees with the written verdict | recommend_route(max_size_bytes=1200) returns plain INT4 packed on the recorded rows | research review, fix 4 |

## Design

The design as built is in Appendix D (Build) and Appendix E (Deep):
a 2-layer NumPy MLP (16-32-4, 676 parameters) on a synthetic 4-class
Gaussian mixture (train 800, calibration 200, test 400), weight-only
symmetric quantization, the QLAB container for sizes, 10 routes,
Pareto on size and accuracy, and a decision rule in code. The seed
study (level-3-deep/seed_study.py, 2026-11-07) adds 10 seed sets,
Wilson 95% intervals and exact McNemar tests.

## What a confirmatory cycle needs

A new PROPOSAL.md section locked by tag before any run, naming a real
pretrained model with a recorded licence, the baselines a reviewer
would expect (GPTQ or AWQ, a llama.cpp GGUF run), a real quantized
kernel for the speed axis, and numeric pass lines. That needs a GPU
window and an Owner decision (see REPORT.md, Future work).

## Appendices: plan files as they were (moved here 2026-11-08)


## Appendix A: scope and acceptance criteria (was SPEC.md)

### SPEC: quantization-lab (P11)

Phase skill: spec-driven-development (house engineering pipeline).
Status: Wave 0 scaffold spec, from Roadmap v2 text and the v3 decision pack. Scope and acceptance criteria below are the v2 section text, kept as the source of truth until the project's wave opens its own spec pass.

Fleet node: BlackBox (GPU). Wave: Wave 6.
Hiring evidence: 3 of 5 (1 named posting family, xAI preference) [confirmed, fleet and scoring pass 2026-11-04, or the decision pack for P23 to P29].

#### Project section text (Roadmap v2, or the v3 candidate text for P23 to P29)

quantization-lab: EXPAND

Subject: model optimization.
v1 core [confirmed, v1]: run INT8 and INT4 models; produce GGUF builds and run them locally with llama.cpp; Deep is a quality against size against speed study with charts and a written verdict.
Added track, TensorFlow optimization (register item 232):
- Build step inside this project: prune or quantize one TensorFlow model with the TF model optimization toolkit and add its row to the study table, so the study speaks for both stacks he works in.
Fan-out: Book chapter "Smaller, Faster, Honest". Paper angle: the quality, size, speed decision rule as a citable study (the CEO next-15 list, step 15, names this artifact) [confirmed, cross-match Part 5]. Course module: P11-quantization-lab.

---

### 5. The projects, P12 to P22

#### Acceptance criteria (scaffold level)

- Learn closes when the level-1-learn code runs, its README teaches the idea, and a fresh-eyes reader can follow it.
- Build closes when the system runs end to end, tests pass, and one benchmark or result table with method is in the README.
- Deep closes when the Deep scope in the section text is done, the failure modes are written, and the dossier is updated to ship state.
- No level closes on a claim. Evidence only: code, tests, numbers, links.

## Appendix B: novelty draft (was NOVELTY.md)

### NOVELTY: quantization-lab (P11)

Status: DRAFT, PROPOSED. Not verified. This text does not move into any public post, paper, or profile until the Jev claim_check gate passes on the exact text.

#### Proposed claim

Against quantization toolchains (llama.cpp, GPTQ and AWQ tooling): this project aims to differ in the way its SPEC.md section text defines (a from-zero or self-hosted build, measured in public, on a named home fleet, with the failure modes written down). The exact difference is stated at the project's wave, after the prior art is re-read.

#### Prior art table

| Prior art | What it does | What this project does differently | Measured difference |
| --- | --- | --- | --- |
| quantization toolchains (llama.cpp, GPTQ and AWQ tooling) | See DOSSIER.md, Past | PROPOSED, see the claim above | NO MEASUREMENT YET |

#### Honest size

If the novelty proves small, this file will say so and name the real value instead (a cleaner reproduction, a better benchmark, a first public dataset). No novelty is invented to fill this section.

## Appendix C: dossier (was DOSSIER.md)

### DOSSIER: quantization-lab (P11)

Written under the documentation and publishing standard
(past, present, future). Updated at Deep ship time,
2026-11-05, to ship state for v1.0.

#### Past

Prior art for this project is named in NOVELTY.md
(quantization toolchains (llama.cpp, GPTQ and AWQ tooling)).
The history of the problem, the key papers and systems, and
what was tried before are recorded in the Roadmap v2 section
text in SPEC.md and will be expanded with dated sources at
the project's wave, before Build opens.

#### Present

Current demand evidence: hiring value 3 of 5 (1 named posting family, xAI preference) [confirmed, fleet and scoring pass 2026-11-04; for P23 to P29, the decision pack 2026-11-04]. The Round 3 research (2026-11-04) found the market centre at post-training, evals, environments, and inference depth [confirmed as the research finding; the report is at ~/workspace/research_notes/roadmap-v3-revaluation-research-20261104-0949/report.md]. Tools and versions are pinned at Build time, not in this v1.

Shipped state (2026-11-05, v1.0): Learn, Build, and
compressed Deep are DONE. Learn: symmetric per-tensor INT8
and INT4 on a 12-weight teaching model. Build: a NumPy-trained MLP (16-32-4, 676
parameters) quantized to INT8 and INT4, magnitude pruned,
serialized in the QLAB container format designed for this
project, and benchmarked on CPU.
Deep: the quality against size against speed study over 10
routes, adding 3 combined prune plus quantize routes, three
code-generated charts, a Pareto analysis on size and
accuracy, and a decision rule implemented and tested as
code. Headline Deep finding: both
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

#### Future

Open problems and the project's next steps sit in the Deep level scope in SPEC.md. Any forecast about this field is [unverified] unless a dated primary source is added here at the wave update.

The v1.1 future for this project is the root ROADMAP.md:
a real pretrained model, a validated GGUF build, a GPU
window, a real quantized kernel timing path, the TensorFlow
row, pruning with sparse storage and fine-tuning, a scale
sweep, activation and asymmetric quantization, and GPTQ and
AWQ comparison rows. None of it is claimed as built.

## Appendix D: Build design (was level-2-build/BUILD-SPEC.md)

### BUILD-SPEC: quantization-lab Level 2, Build

Written before the Build code, per the task order for this cycle.
Date: 2026-11-05. Status: this spec governs the Build shipped in
this cycle. Anything this spec names as a substitution is a
substitution, stated plainly, not a silent replacement.

#### What SPEC.md asks Build to do

SPEC v1 core: run INT8 and INT4 models, produce GGUF builds and
run them locally with llama.cpp. Added track: prune or quantize
one TensorFlow model with the TensorFlow Model Optimization
Toolkit and add its row to the study table. Build closes when
the system runs end to end, tests pass, and one benchmark table
with method is in the README.

#### Verified environment for this cycle (2026-11-05, Muse VM)

- numpy 1.26.4 is available. Python 3.12.
- import tensorflow fails: ModuleNotFoundError. TensorFlow is
  not available on this VM.
- import torch fails: ModuleNotFoundError.
- llama-cli is not on PATH (which llama-cli returns nothing).
  llama.cpp is not available on this VM.
- BlackBox, the assigned GPU fleet node, is unreachable this
  cycle: NEEDS-BLACKBOX-ON (coordinator verified SSH closed, no
  output). No GPU run is claimed.

#### Substitutions, named plainly

1. No GGUF, no llama.cpp. The serialized format in this Build is
   QLAB, a small binary container format designed and documented
   in build_system.py for this project only. QLAB is not GGUF,
   is not compatible with GGUF or llama.cpp, and no file produced
   here can be run by llama.cpp. A real GGUF writer was not
   attempted because it could not be validated here: there is no
   llama.cpp binary on this VM to parse or run its output, and an
   unvalidated writer would be a claim without evidence.
2. No real pretrained model. The model is a 2-layer MLP trained
   in this repo with NumPy on a larger synthetic dataset (see
   below). It is a real trained model in the narrow sense that
   its weights come from gradient descent on labelled data and
   its test accuracy is measured, not set by construction. It is
   not a pretrained or real-world model.
3. TensorFlow track substitution. The TensorFlow Model
   Optimization Toolkit row did not run, because TensorFlow is
   unavailable (verified above). In its place, the same model is
   magnitude pruned with NumPy at named sparsity levels, labelled
   in every table and result as the NumPy equivalent pruning
   track, never as a TensorFlow result.
4. CPU only. All timing is CPU inference time measured on this
   Muse VM, single machine, varying load. No GPU timing, no
   llama.cpp timing, and no speed claim beyond this VM is made.
5. Fake quantization for quality numbers. Accuracy and error for
   quantized variants are measured with dequantized weights in a
   NumPy forward pass (fake quantization), which is the standard
   way to measure quantization quality without a quantized
   kernel. Size numbers are real serialized QLAB file bytes.
   Timing for quantized variants is the dequantized forward pass
   timing and is labelled as such; it does not show the speed a
   real INT8 or INT4 kernel would give.

#### The Build system

- Model: 2-layer MLP, input 16, hidden 32 with ReLU, output 4.
  Weights W1 (16 by 32), b1 (32), W2 (32 by 4), b2 (4). Total
  676 parameters. Initialization and training are deterministic:
  NumPy RandomState seeds fixed in code, full-batch gradient
  descent, 60 epochs, learning rate 0.1, no shuffling.
- Data: synthetic 4-class Gaussian mixture, input dimension 16.
  Class centers from RandomState(7), scale 0.6; sample noise
  standard deviation 1.0. Splits: train 800, calibration 200,
  held-out test 400, each from its own fixed seed, labels
  balanced by construction then shuffled with the split seed.
  Labels are the generating class. The classes overlap by
  design, so FP32 accuracy is below 1.0 and quantization has
  room to hurt it.
- Reuse of Learn: level-2-build/build_system.py imports
  quantize_symmetric, pack_int4, unpack_int4, and
  round_half_away from level-1-learn/quantize.py. The per-tensor
  INT8 variant is exactly the Learn method applied per weight
  matrix. Reuse is by import, stated here and in the code
  header, not a silent copy. Learn files are unchanged.
- Quantization variants:
  - FP32 baseline.
  - INT8 per-tensor: Learn symmetric method, one scale per
    weight matrix, no calibration.
  - INT8 per-channel: one scale per output column of each
    weight matrix, with calibration (below).
  - INT4 per-channel, packed: one scale per output column,
    codes packed two per byte with the Learn pack_int4, with
    calibration (below).
  - Biases are stored in FP32 in every variant, which is common
    practice and is counted in every size.
- Calibration: the calibration split (200 samples, never used
  for training or for the reported test accuracy) selects the
  clipping percentile for each weight matrix. Candidates are
  100, 99.9, 99.5, 99.0, and 98.0 percent of the absolute
  weights. For each candidate the matrix is clipped, quantized,
  dequantized, and run through the full model on the
  calibration inputs; the candidate with the lowest mean
  squared logit error against the FP32 model wins. The chosen
  percentile per matrix is recorded in results_build.json.
- QLAB container format (version 1): bytes 0 to 3 are the ASCII
  magic QLAB; bytes 4 to 5 are the version as an unsigned 16-bit
  little-endian integer; bytes 6 to 9 are the header length as
  an unsigned 32-bit little-endian integer; then a UTF-8 JSON
  header with sorted keys listing each entry (name, kind, shape,
  dtype, byte length) in payload order; then the payloads in
  header order. Kinds: fp32 (little-endian float32), int8
  (signed bytes), int4_packed (Learn packing, first code in the
  low nibble), scales (little-endian float32). For per-channel
  matrices, codes are stored column by column (column-major), so
  each column has one contiguous block and one scale. Save and
  load are implemented in build_system.py and covered by a
  round-trip test.
- Pruning track (NumPy equivalent, see substitution 3):
  magnitude pruning zeroes the smallest absolute weights in
  each weight matrix at sparsity 0.3, 0.5, and 0.7. Pruned
  models are stored as FP32 in QLAB, so their serialized size
  equals the FP32 size; that fact is reported, not hidden,
  because dense storage does not shrink from pruning alone.
- Benchmark table: one row per variant (FP32, INT8 per-tensor,
  INT8 per-channel, INT4 per-channel packed, pruned 0.3, pruned
  0.5, pruned 0.7) with serialized size in bytes, maximum and
  mean absolute weight error against FP32, held-out test
  accuracy, mean squared logit error against FP32 on the test
  set, and CPU inference time (median of 15 full test-set
  forward passes on this VM, milliseconds).
- Runner: level-2-build/run_build.py runs training, calibration,
  quantization, pruning, serialization, and evaluation end to
  end and writes level-2-build/results_build.json. The results
  file is deterministic except the named timing fields
  (timing_ms_median per row and the timing note); two runs must
  produce identical files once timing fields are removed.
- Tests: tests/test_build_system.py with hand-computed
  expectations for per-channel scales and codes, INT4 packing,
  the QLAB save and load round-trip, the pruning mask, and
  calibration candidate selection on a small case, plus the
  existing Learn tests and the scaffold smoke test. The full
  suite must pass.

#### Acceptance for this Build

- Runner runs end to end on this VM and writes results.
- Full test suite passes.
- Benchmark table with measured values and method is in
  level-2-build/README.md and the root README.md.
- Every substitution above is named in the level README.
- Em dash count is 0 in all touched files.

## Appendix E: Deep design (was level-3-deep/DEEP-SPEC.md)

### DEEP-SPEC: quantization-lab Level 3, Deep (compressed Deep)

Written before the Deep code, per the December Compression
Protocol. Date: 2026-11-05. Compressed Deep is the single
hardest capability in SPEC.md: the quality against size
against speed study with charts and a written verdict. The
rest of the v2 Deep scope is the v1.1 backlog in the root
ROADMAP.md, not claimed here.

#### What SPEC.md asks Deep to do

SPEC v1 core: run INT8 and INT4 models, produce GGUF builds
and run them locally with llama.cpp. Added track: a
TensorFlow optimization row. Deep closes when the Deep scope
is done, failure modes are written, and the dossier is
updated to ship state. Under compression, Deep is the study
itself (quality, size, speed, charts, verdict, decision
rule), built on the routes that actually run on this machine.

#### Verified environment for this cycle (2026-11-05, Muse VM)

- Python 3.12.3, NumPy 1.26.4. Matplotlib is present on the
  VM but the Deep code uses only NumPy and the standard
  library, so the study runs anywhere Build runs.
- import tensorflow fails: ModuleNotFoundError (checked this
  cycle). import torch fails the same way.
- llama-cli is not on PATH (checked this cycle). No GGUF
  validation route exists on this machine.
- BlackBox, the assigned GPU fleet node, is unreachable this
  cycle: NEEDS-BLACKBOX-ON stands (host key verification
  failed, per the task brief). No GPU run is claimed. All
  work runs on the Muse VM CPU. GPU rows appear only in a verified GPU window,
  and this cycle has none.

#### Substitutions and limits, named plainly

1. No GGUF and no llama.cpp run. Sizes are QLAB container
   bytes from Build (level-2-build/build_system.py), a format
   designed for this project. QLAB is not GGUF. No file here
   runs in llama.cpp.
2. No TensorFlow result. The pruning rows and the combined
   prune plus quantize rows are the Build NumPy equivalent
   track, labelled as such in every table.
3. CPU only, fake quantization timing. Quality uses
   dequantized weights in a NumPy forward pass. Speed is that
   forward pass timed on this VM. It does not show the speed
   of a real INT8 or INT4 kernel, and the verdict says so and
   uses speed only where the measurement supports it.
4. Synthetic data and one small model, inherited from Build:
   the same trained MLP (16-32-4, 676 parameters) and the
   same splits (train 800, calibration 200, test 400), so
   Deep numbers compare directly with Build numbers.

#### The Deep system

- Reuse by import, not by copy: level-3-deep/deep_system.py
  imports level-2-build/build_system.py for the model, data,
  quantization, pruning, calibration, and QLAB container.
  No Build or Learn file is changed.
- Routes (10): the 7 Build routes (fp32, int8_per_tensor,
  int8_per_channel, int4_per_channel_packed, pruned_0.3,
  pruned_0.5, pruned_0.7) plus 3 combined routes that Build
  did not measure: pruned_0.3 plus INT8 per-channel,
  pruned_0.5 plus INT8 per-channel, and pruned_0.5 plus INT4
  per-channel packed. Combined routes prune first, then
  calibrate and quantize the pruned weights on the same
  calibration split.
- Per route, measured: serialized QLAB size in bytes, size
  ratio against FP32, max and mean absolute weight error,
  held-out test accuracy (correct of 400), accuracy change
  against FP32, logit MSE against FP32, batch timing (median
  of 31 full test-set forward passes after 5 warmup passes),
  and single-sample latency (median of 101 forward passes on
  one test sample after 5 warmup passes). Timing method is
  stated here and in the results file. Timing fields are the
  only nondeterministic fields in results_deep.json.
- Derived analysis, computed in code from the measured rows
  only: Pareto flags on the two deterministic axes, size and
  accuracy (a route is Pareto optimal when no other route is
  at most its size and at least its accuracy, with at least
  one strict), accuracy per kilobyte, and the decision rule
  below. Speed is reported per route but is not a Pareto
  axis: batch timing on dequantized weights is noise at this
  scale, and an early three-axis version of this check
  flipped its flags between runs in this cycle, which is
  recorded in the Deep README failure modes.
- Decision rule, implemented as code and tested:
  recommend_route(rows, max_size_bytes, min_accuracy,
  max_batch_ms) filters routes that meet every given
  constraint, then ranks survivors by accuracy (highest),
  then logit MSE (lowest), then size (smallest), then batch
  time (lowest). With no constraints it returns the highest
  accuracy route by that ranking. When no route meets the
  constraints it returns None with the binding reason named
  by the caller. The Deep README verdict is this rule stated
  in words, checked against the measured table.
- Charts, generated by code from the measured rows only, as
  hand-built SVG files (no invented values, axis ranges
  computed from the data): charts/accuracy_vs_size.svg,
  charts/logit_mse_vs_size.svg, charts/batch_time_vs_accuracy.svg.
  Each chart title names its axes and the run, and each data
  point is one measured route, labelled by route name.
- Runner: level-3-deep/run_deep.py trains the Build model
  with the Build seeds, builds every route, serializes and
  round-trip checks each QLAB file (tolerance 1e-6, as in
  Build), evaluates the loaded model itself, times it,
  computes the analysis, writes the charts, and writes
  level-3-deep/results_deep.json. Two runs must produce
  identical results files once the timing fields
  (timing_batch_ms_median and timing_single_ms_median per
  row) are removed.
- Tests: tests/test_deep_system.py with hand-computed
  expectations for the Pareto check, the decision rule, the
  combined route construction (pruned zeros survive
  quantization), accuracy per kilobyte, and SVG chart
  generation from fixed rows (titles, labels, and point
  count). The full suite (Learn, Build, Deep, smoke) must
  pass.

#### Acceptance for this Deep

- DEEP-SPEC.md written before the code (this file).
- Runner runs end to end on this VM and writes results and
  charts. Determinism proven by two runs with timing fields
  removed.
- Study table with measured values and method, three charts,
  a written verdict with the decision rule, and a failure
  modes section are in level-3-deep/README.md.
- Root README.md and DOSSIER.md updated to ship state for
  Deep. Root ROADMAP.md holds the v1.1 backlog.
- Jev claim_check run on the exact main results claim in
  the Deep README, with arithmetic pre-computed in code.
- Em dash count is 0 in all touched files.
