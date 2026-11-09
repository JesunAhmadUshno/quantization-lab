# Changelog

All notable changes to this project. Format: Keep a Changelog 1.1.
Versions follow Semantic Versioning. The tags v0.1, v0.5 and v1.0 were
cut under the earlier Learn / Build / Deep ladder; under the portfolio
standard v2 the verdict is Functional, not pre-registered, and no
standard v2 release is tagged yet.

## [Unreleased]

### Changed
- 2026-11-08: Adopted the portfolio standard v2. README opens with the
  question, headline (with its interval), field, job family and
  verdict (Functional, not pre-registered). SPEC.md, NOVELTY.md,
  DOSSIER.md, BUILD-SPEC.md and DEEP-SPEC.md moved into PROPOSAL.md;
  the level READMEs, claim-check records, docs/research-review.md and
  ROADMAP.md moved into REPORT.md, which now scores seven hypotheses
  with the misses shown. Grades left the public repo (grades are
  internal). The dashboard links point to the new files.
- 2026-11-07: Upgraded the dashboard's vite to 6.4.5 (npm audit clean
  of high and critical advisories).

### Added
- 2026-11-08: scripts/check_results.py and a CI step proving every
  results file regenerates from the code; a layout test (required
  files, no public grade); CHANGELOG, CITATION, SECURITY; decision
  record 0001; CI least-privilege permissions; npm in Dependabot and
  JavaScript in CodeQL for site/.
- 2026-11-07: Seed study over 10 seed sets with per-sample
  predictions, Wilson intervals and exact McNemar tests; an optional
  accuracy tie in recommend_route; the live dashboard and its Pages
  workflow; the research review.

### Fixed
- 2026-11-06: numpy pinned per Python version (1.26.4 on 3.11, 2.5.4
  on 3.12 and later).

## [1.0] - 2026-11-05

### Added
- Deep: the quality against size against speed study over 10 routes,
  three charts, Pareto flags on size and accuracy, and the decision
  rule in code; CI, CodeQL, Dependabot and the release workflow; the
  written fan-out set.

## [0.5] - 2026-11-05

### Added
- Build: the NumPy MLP (16-32-4, 676 parameters), INT8 per-tensor and
  calibrated INT8 and INT4 per-channel quantization, the NumPy
  pruning track and the QLAB container.

## [0.1] - 2026-11-05

### Added
- Learn: symmetric per-tensor INT8 and INT4 on a 12-weight model,
  standard library only.

[Unreleased]: https://github.com/JesunAhmadUshno/quantization-lab/compare/v1.0...HEAD
[1.0]: https://github.com/JesunAhmadUshno/quantization-lab/compare/v0.5...v1.0
[0.5]: https://github.com/JesunAhmadUshno/quantization-lab/compare/v0.1...v0.5
[0.1]: https://github.com/JesunAhmadUshno/quantization-lab/releases/tag/v0.1
