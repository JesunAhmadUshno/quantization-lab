# PUBLICITY: quantization-lab (P11)

Publicity log under the documentation and publishing standard,
Section 7. Every finished level produces its written fan-out set,
each piece graded on the 1,000-point publicity rubric (accuracy 300,
research 200, gap fit 200, clarity 150, reach format 150), ship line
900. Drafts are prepared by the loop. Actual social posting follows
the standing platform rules and gates. The loop itself never posts
to social.

Project state: v1.0, Learn, Build, and compressed Deep done
2026-11-05, 55 tests passing plus the scaffold smoke test. Level
grades on record: Learn 8,700 of 10,000, Build 7,600 of 10,000,
Deep 7,450 of 10,000.

## Actions polish pack (2026-11-05)

Applied through a pull request on 2026-11-05, following the Actions
polish pack standard. The pack adds: CI on a Python 3.11 and 3.12
matrix with a ruff lint step (the pre-existing CI step preserved:
scaffold smoke test; plus per-level test and runner steps for Learn,
Build, and Deep, following the proven pack pattern, with the repo
dependencies installed from requirements.txt), CodeQL scanning,
Dependabot for pip and GitHub Actions, a release workflow that
attaches a source tarball named quantization-lab-<tag>.tar.gz to
v* tag releases, Labeler, Greetings (snake_case inputs for
actions/first-interaction@v3: repo_token, issue_message,
pr_message), Stale in comment-only mode (days-before-close: -1,
never closes), and a docs summary page at docs/index.md for GitHub
Pages. Ruff scope is recorded in ruff.toml (E, F, I, line-length
120). Ruff fixes in this pull request were style only: the first
ruff run found 5 errors (import formatting in
level-1-learn/run_learn.py, tests/test_learn_quantization.py, and
tests/test_smoke.py, and unused numpy imports in
level-3-deep/deep_system.py and tests/test_deep_system.py), fixed
by ruff fix output, with no change to behaviour, imported names in
use, or any string value. All three results files stayed unchanged
across the runners apart from the timing fields in the Build and
Deep results, which were restored. requirements.txt records the
true dependency fact for this repo: Level 1 uses the Python
standard library only, and Level 2, Level 3, and the Build and
Deep tests use NumPy, pinned to the measured version 1.26.4.

| Channel | Outcome on 2026-11-05 |
| --- | --- |
| GitHub Pages | Docs page added at docs/index.md in this pull request. Pages enablement and live build status are verified after merge and recorded in the batch ledger; no live claim is made in this file before that check. |
| PyPI publishing | Held: needs a PyPI token, a new credential. Jesun's tap only. Not set up. |
| Social channels (LinkedIn, X, Facebook, Instagram, Threads) | Held: no posts made. The loop never posts to social; Threads waits for Jesun's tap per post under the standing gates. |

Publicity grade for this polish pack log piece, on the 1,000
point rubric (accuracy 300, research 200, gap fit 200, clarity
150, reach format 150), ship line 900:

| Piece | Accuracy /300 | Research /200 | Gap fit /200 | Clarity /150 | Reach /150 | Total /1000 |
| --- | --- | --- | --- | --- | --- | --- |
| PUBLICITY.md polish pack section (2026-11-05) | 295 | 180 | 195 | 145 | 95 | 910 |

At or above the 900 ship line. Named deductions: research minus
20, because the pack is a standard applied across repos and this
section records its application rather than new research; reach
minus 55, because the docs page is copy in this pull request with
no live Pages claim yet, PyPI publishing is held, and no social
channel carries this piece by the standing gates; accuracy minus
5, gap fit minus 5, and clarity minus 5 for summary compression
of the full pack standard, which lives in the batch files.
