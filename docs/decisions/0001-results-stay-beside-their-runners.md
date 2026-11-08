# 0001: Results stay beside their runners; no experiments/ or results/ folder

Date: 2026-11-08. Status: accepted.

## Context

The portfolio standard v2 puts raw evidence in experiments/ and scored
results in results/. In this repo every run is seeded and
deterministic apart from two timing fields, and each runner writes one
JSON file next to itself (level-*/results_*.json). The dashboard,
the build-time JavaScript check, the tests and the fan-out drafts all
read those paths.

## Decision

Keep the three level folders as domain-named code folders, each
holding its runner and its results file. The results file is both the
evidence and the scored result. scripts/check_results.py, run in CI,
reruns every runner and fails if any count, size or float (relative
1e-6) changes, which is the standard's "results regenerated from
experiments/ by one command" check.

## Consequences

No duplicate copies of the results and no broken links in the
dashboard or the dated fan-out drafts. Timing medians are not
reproducible and are not compared; the raw per-pass timings behind
them are not kept, which REPORT.md states (H5). A future study with
non-deterministic or externally captured evidence (a GPU run, a
llama.cpp run) must add an append-only experiments/ folder.
