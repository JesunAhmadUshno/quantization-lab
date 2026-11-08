# quantization-lab

**Question.** At small-model scale, what do symmetric INT8 and INT4
weight quantization and magnitude pruning cost in serialized size,
accuracy and logit fidelity, and which route should be picked under a
byte budget?

**Headline.** On a 676-parameter NumPy MLP with synthetic data (not a real model), INT8 per-channel weight quantization kept FP32 test accuracy, 353 of 400 (Wilson 95% interval 0.847 to 0.910), at 1388 bytes, 2.17 times smaller than FP32; across 10 seeds neither INT8 route moved more than 2 of 400 samples from FP32 and none differed significantly (exact McNemar p < 0.05 in 0 of 10 seeds).

**Field.** ML and AI research. **Job family.** ML engineering, model
optimization and inference.

**Verdict.** Functional, not pre-registered. Tests pass in CI on Python
3.11 and 3.12, the whole study reruns with one command, and CI proves
every results file regenerates from the code. No plan with hypotheses
and pass lines was committed before the results, so the repo is capped
at Functional; the field profile is also not met (no GPTQ, AWQ or
llama.cpp baseline, no real model). See REPORT.md.

Live dashboard: https://jesunahmadushno.github.io/quantization-lab/

## Reproduce

```text
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py" -v
python scripts/check_results.py
```

The last command reruns Learn, Build, Deep and the 10-seed study (CPU,
a few seconds) and fails if any count, size or float in the committed
results files changes; timing medians are machine facts and are not
compared. Python 3.11 or 3.12 and NumPy only; Learn uses the standard
library only.

## Lifecycle

- Question: DONE 2026-11-04
- Proposal: NOT PRE-REGISTERED 2026-11-08
- Baseline: DONE 2026-11-05
- Build: DONE 2026-11-05
- Evaluate: PARTIAL 2026-11-07
- Report: DONE 2026-11-08
- Review: OPEN
- Release: OPEN

Proposal: SPEC.md (2026-11-04) set scope only; the Build and Deep
designs were committed together with their results, so nothing is
locked (PROPOSAL.md). Evaluate is partial: uncertainty and seeds are
in, the baselines a reviewer would expect are not. Release: the tags
v0.1, v0.5 and v1.0 (2026-11-05) predate this standard and are kept as
they are; they are not a v1.0 Released verdict.

## Layout

| Path | Role in the standard | What it holds |
| --- | --- | --- |
| PROPOSAL.md | Proposal (not locked) | Question, prior work, hypotheses written after the results, design; the old SPEC, NOVELTY, DOSSIER, BUILD-SPEC and DEEP-SPEC files as appendices |
| REPORT.md | Report | Results with intervals, every hypothesis scored, misses, threats; the old level READMEs, claim-check records, research review and roadmap as appendices |
| level-1-learn/ | Code and results (baseline) | quantize.py, run_learn.py, results_learn.json: INT8 and INT4 on a 12-weight model, standard library only |
| level-2-build/ | Code and results (build) | build_system.py (MLP, quantizers, calibration, pruning, QLAB container), run_build.py, results_build.json |
| level-3-deep/ | Code and results (evaluation) | deep_system.py, run_deep.py, results_deep.json, seed_study.py, results_seeds.json (per-sample predictions), charts/ |
| scripts/check_results.py | Results reproduced | Reruns every runner and compares with the committed results |
| tests/ | Tests | Unit tests per level, the research tests, and the layout test (required files, no public grade) |
| site/ | Dashboard | Reads the level results files at build time; a build-time check re-derives them in JavaScript |
| docs/decisions/ | Decisions | Records of hard-to-reverse choices |
| docs/index.md, fanout/, PUBLICITY.md | Outreach | The legacy summary page, the written fan-out drafts and the channel log |
| CHANGELOG.md, CITATION.cff, SECURITY.md, LICENSE | Project files | |

There is no separate experiments/ or results/ folder: every run is
seeded and deterministic apart from timing, so each runner's results
file is both the raw evidence and the scored result, and
scripts/check_results.py proves it regenerates (docs/decisions/0001).

## License

Code: Apache-2.0 (see LICENSE). Tutorials: CC BY 4.0. Ruled in D-V3-11, 2026-11-04.
