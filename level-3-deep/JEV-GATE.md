# Jev gate record: P11 Deep main claim, 2026-11-05

Gate: claim_check, run through the TypeSafe Jev connector
helper (typesafe_jev.py evaluate).

- Model returned: jev-1.13.0
- Input tokens: 1295
- Output tokens: 20
- Question id: supported
- Question: whether the evidence in the spec state supports
  the main result paragraph in level-3-deep/README.md
  exactly, with no part of the claim going beyond the evidence.
- Noul for supported: 0.77
- Route under the claim_check bands (CLEAR at 0.85 or above,
  BLOCK at 0.10 or below, REVIEW between): REVIEW.
- Arithmetic pre-computed in code before the gate run:
  3013/1248 = 2.4143 (stated 2.41x), 3013/1081 = 2.7872
  (stated 2.79x), combined INT4 logit MSE ratio
  0.542604/0.045583 = 11.90 (stated about 12 times).
- Spec file:
  ~/workspace/goals/ml-ai-role-at-a-top-tier-ai-company/hidden_files/jev-p11-deep-claim-spec-2026-11-05.json
- Results file:
  ~/workspace/goals/ml-ai-role-at-a-top-tier-ai-company/hidden_files/jev-p11-deep-claim-results-2026-11-05.json
- HTTP status: 200.

REVIEW means the claim is neither cleared nor blocked by the
gate. The paragraph stands as written in the README with its
substitutions named inside the claim itself (QLAB not GGUF,
CPU dequantized quality numbers, NumPy pruning track), and
this record is the gate evidence. No wording in the paragraph
was changed after the gate run.
