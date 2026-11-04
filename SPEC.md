# SPEC: quantization-lab (P11)

Phase skill: spec-driven-development (house engineering pipeline).
Status: Wave 0 scaffold spec, from Roadmap v2 text and the v3 decision pack. Scope and acceptance criteria below are the v2 section text, kept as the source of truth until the project's wave opens its own spec pass.

Fleet node: BlackBox (GPU). Wave: Wave 6.
Hiring evidence: 3 of 5 (1 named posting family, xAI preference) [confirmed, fleet and scoring pass 2026-11-04, or the decision pack for P23 to P29].

## Project section text (Roadmap v2, or the v3 candidate text for P23 to P29)

quantization-lab: EXPAND

Subject: model optimization.
v1 core [confirmed, v1]: run INT8 and INT4 models; produce GGUF builds and run them locally with llama.cpp; Deep is a quality against size against speed study with charts and a written verdict.
Added track, TensorFlow optimization (register item 232):
- Build step inside this project: prune or quantize one TensorFlow model with the TF model optimization toolkit and add its row to the study table, so the study speaks for both stacks he works in.
Fan-out: Book chapter "Smaller, Faster, Honest". Paper angle: the quality, size, speed decision rule as a citable study (the CEO next-15 list, step 15, names this artifact) [confirmed, cross-match Part 5]. Course module: P11-quantization-lab.

---

# 5. The projects, P12 to P22

## Acceptance criteria (scaffold level)

- Learn closes when the level-1-learn code runs, its README teaches the idea, and a fresh-eyes reader can follow it.
- Build closes when the system runs end to end, tests pass, and one benchmark or result table with method is in the README.
- Deep closes when the Deep scope in the section text is done, the failure modes are written, and the dossier is updated to ship state.
- No level closes on a claim. Evidence only: code, tests, numbers, links.
