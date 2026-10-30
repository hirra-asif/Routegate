# Final evaluation: consistency audit

An offline review of label consistency in the original final-evaluation
assessment, plus a numerical check of two steering-arithmetic implementations.
It makes no new model responses or judge calls and changes no frozen result.

## Completed checks

- Reconstructed all 3,347 recorded judge requests, including failures and
  retries, from their frozen templates. Every saved request checksum matched.
  Sonnet and Opus received the same instructions, example text, pairwise order
  and settings for every corresponding assessment; only the model field differed.
- Confirmed that the Axis-1, Axis-2, pairwise quality and safety rubrics are
  byte-identical between the baseline and response stages.
- Found all six groups of identical prompt/reply text with conflicting decided
  labels, and reviewed their 13 assessments against the frozen Axis-2 rubric.
- Prepared seven proposed label changes (three baseline, four response) and
  calculated their effects in memory only. If accepted, substantive E outcomes
  become 5/38 for dynamic strength and 8/38 for safety-first RouteGate, and every
  remaining substantive reply differs from baseline. Group sizes and all B
  quality and H safety labels are unchanged.
- Confirmed the gate's decisions on all 380 sealed prompts. On the 38 E prompts
  it allows 12 edits and blocks 16 on the H score alone, six on the E score alone
  and four on both. All 350 gate-closed sealed replies match baseline exactly.

## Status of the proposed labels

The proposals are recorded as `PROPOSED_NOT_IMPORTED`. The review was
AI-assisted (reviewer recorded as "ChatGPT assistant") and not blinded: the
reviewer saw previous labels, system identities and results. The researcher
authorised the audit, but approval of the specific proposed labels has not been
recorded. The pipeline's historical `HUMAN_ADJUDICATION` field does not make this
an independent human audit.

Three truncated replies are proposed as UM because the visible text does not
establish the reason for the missing material; PA-N is a plausible alternative if
a separate rule were to treat truncation as a non-safety cause. The
email-testing case also lies on a DR/PA-S boundary. No proposed label is
presented as certain.

If the proposals are accepted, preserve the original adjudication files and
create a separate, versioned erratum with its approval and provenance. Do not
import them through the original `corpus_disagreement_only` procedure, which
would not cover the original agreements the review touches.

## Records

| Record | Location | In Git |
| --- | --- | --- |
| Review record without prompt/reply text | `docs/final_evaluation/consistency_review_v1.public.json` | yes |
| Full review record with prompt and reply text | `docs/final_evaluation/consistency_review_v1.json` | no (held-out material) |
| Audit outputs and readable review | `outputs/thesis_results/consistency_review_v1/` | no |

The public record replaces each prompt and saved reply with its length and keeps
the existing SHA256 digests, labels, reasons and qualifications unchanged.

## Steering-arithmetic comparison

`finalrunner.py` applies the frozen consensus direction through
`steering.build_steering_plan`. The historical `FrozenActuator.applied_delta`
first builds a synthetic donor and subtracts the target. The two are equal in
exact arithmetic but need not round identically in float32, and the final hook
casts the delta to bfloat16 before addition.

`compare_steering_arithmetic.py` calls both implementations and the repository's
addition function on all 480 cached development prompts at blocks 31–33, on CPU.
It ran on 10 September 2026 (`outputs/thesis_results/consistency_review_v1/arithmetic/`).
Some development states receive different bfloat16 deltas: per layer, 229–352 of
480 vectors differ after the cast, and 12–17 written states differ in at most one
coordinate. Minimum delta cosine is above 0.99999999999. This establishes a
numerical difference, not a behavioural effect. The inputs are development
activations, not sealed ones, and cached-state comparisons at later layers do
not simulate propagation from earlier edits. No acceptability threshold was set.

## Run

```bash
bash experiments/final_evaluation/run_consistency_audit.sh
```

The wrapper runs inside the project container, needs no judge credential or GPU,
and reads `ROUTEGATE_DATA_DIR` and `ROUTEGATE_CONTAINER_FILE` (defaults under
`$HOME`). The activation input must have SHA256
`af89872d5decd4dc432ec579a941e6247bc7de63ff5690432e9f5812a994e2d8`; pass another
location with `--activation <path>`. The checksum is still enforced.

## Historical files in this directory

`PACKAGE_SHA256SUMS`, `SOURCE_SNAPSHOT.json` and `VALIDATION_STATUS.json` were
written when the audit package was assembled on 10 September 2026 and are kept
unchanged. `VALIDATION_STATUS.json` predates the arithmetic run above, so its
`real_tensor_arithmetic` entry is superseded. `PACKAGE_SHA256SUMS` also lists
files that are not in the public tree (the full review record, local outputs and
a removed assistant task file) and this note, which was rewritten for publication; their
original bytes are preserved in the privately retained development history. The
public repository begins with a clean initial commit and does not include them.
