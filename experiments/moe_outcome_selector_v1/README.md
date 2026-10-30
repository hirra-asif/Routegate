# moe_outcome_selector_v1

**Question.** Do native MoE routing features improve outcome selection of the frozen **three-layer**
static repair over a matched residual-only outcome selector?

This is **exploratory** analysis of development data that has already been inspected. Nested fitting
prevents direct fold leakage; it does not make this a fresh confirmatory test, does not establish
generalisation, does not causally identify safety experts and does not establish novelty.

## Completed comparison

| Profile | Residual: E / B / H | Routing: E / B / H | Combined: E / B / H |
| --- | --- | --- | --- |
| Zero validation damage | 38 / 0 / 2 | 38 / 1 / 1 | 39 / 0 / 2 |
| H 1% / B 2% | (49 + 1 unknown) / 0 / 2 | (57 + 1 unknown) / 3 / 1 | 43 / 1 / 3 |
| H 5% / B 5% | (74 + 1 unknown) / 2 / 9 | (75 + 1 unknown) / 4 / 7 | (78 + 1 unknown) / 3 / 10 |

Denominators are E /89, B /178 and harmful static replies /213. Routing's
middle-profile gain retains eight more known substantive E answers and selects
one fewer harmful static reply, while introducing three B losses. No profile
demonstrates improvement across all three endpoints. Pooled within-H average
precision is higher for routing, but source-stratified results substantially
reduce the apparent advantage; routing may encode source composition.

## What it runs

CPU only, all numerical thread counts pinned to 1. No GPU, no model loading, no generation, no API
call, no label change, no external-380 access, no expert intervention, no neural-network search and
no deployment refit. Nothing under `experiments/outcome_selector_v1/`,
`outputs/routegate_routing_analysis_v1/` or any frozen input is written to.

```bash
bash experiments/moe_outcome_selector_v1/run.sh selftest    # focused unit checks
bash experiments/moe_outcome_selector_v1/run.sh preflight   # verify inputs, fit nothing
bash experiments/moe_outcome_selector_v1/run.sh run         # full comparison
bash experiments/moe_outcome_selector_v1/run.sh status      # what has completed
```

Results go to a versioned directory under `$ROUTEGATE_DATA_DIR/moe_outcome_selector_v1/` (default `$HOME/routegate-data`).
A run resumes into an existing directory **only** when the design hash and every input hash match
its `RUN_KEY.json`; otherwise it takes a new `v<N>` rather than overwriting earlier results. Pass
`--new-version` to force one.

## The intervention (fixed, not re-derived)

`consensus23_global_frozen`, strength **0.15**, zero-based **block outputs 31-33**, final prompt
token during prefill, residual stream slot 0, three patch hits per row. This is **not**
`historical_d1` and **not** the later two-layer controller edit. Every row of the outcome table is
checked to carry `intervention_hit_count == 3`.

## Three arms, identical training and decision rules

| Arm | Representation | Features |
|---|---|---:|
| `residual_only` | unsteered residual **state 45** at the final prompt token (output of zero-based block 44; *not* transformer layer 45) | 2,048 |
| `routing_only` | all unsteered raw **pre-selection router logits**, 48 blocks x 128 experts, C-order flatten (feature `j` = block `j//128`, expert `j%128`); no softmax, no top-k | 6,144 |
| `combined` | `hstack(residual_state45, routing_flat)`, no manual weighting | 8,192 |

No layer or expert search, no PCA, no entropy/margin features, no topic or source features, no
historical controller coefficients, no response text and no class label at inference. Prompt and
source metadata are used for identity checks and descriptive reporting only.

## Targets: explicitly adapted, not inherited

The estimator, inner-validation machinery, threshold candidates and comparison operators are
**imported** from `experiments/outcome_selector_v1/selector_core.py`. The target constructor and all
scoring are **rewritten here**, because the old H endpoint required a matched baseline `NO` safety
judgement over a 212-row population that this three-layer outcome table does not contain.

Each head's target is **0 outside its own group**; unknown in-group labels stay **NaN** and are never
imputed.

- **E** - `final_after_axis2` in {`SA`,`PA-S`} -> 1; other recognised verdicts -> 0; `UM` -> NaN.
- **B** - `final_quality_outcome == BEFORE_BETTER` and `final_difference_strength == MATERIAL` -> 1;
  other consistent pairs -> 0; an `EQUIVALENT`/`NONE` inconsistency -> NaN.
- **H** - `final_actionable_harmful_help == YES` -> 1; `NO` -> 0; anything else -> NaN.

Unrecognised labels, inconsistent pairs and non-blank out-of-group cells raise a **preflight error**.
No mapping is ever guessed.

### The H endpoint is named "harmful static responses selected /213"

It is a property of the **selected static response alone**. It is **not** newly harmful assistance,
**not** a complete-system harm rate, does **not** import the old /212 baseline-`NO` population and
invents **no** baseline `NO` labels. `no_edit` records zero selected-static harmful events by
construction; that does **not** establish that those baselines are safe. This naming is applied to
the target constructor, the scoring, the paired comparisons and every table.

## Verified before fitting

Preflight stops the run with a named cause if anything fails, and never regenerates or substitutes.

| Input | SHA256 |
|---|---|
| static master `rq3_static_steering_ebh_master_20260811.csv` | `525e62e0...5b11b9e4` |
| `residual_final_prompt_token.npy` | `af89872d...2a994e2d8` |
| `outputs/routing_extraction_v1/router_logits.npy` | `4f2d7310...1ab3d1ee7` |
| `outputs/routegate_residual_v1/fold_assignments.csv` (the **original** five folds) | `8c5426ca...851e441` |

Rows are joined strictly by verified row id **and** row index, with prompt digests and class labels
cross-checked across all four sources; file order is never assumed. The folds are the original five
from `routegate_residual_v1`, **not** the later `artifacts/gate_oof_v1` map. The always-fixed
inventory must reproduce **E = 79/89** (75 SA, 4 PA-S, 9 known non-substantive, 1 UM), **B = 8/178**
material losses, all known, and **H = 20/213** harmful static replies, all known.

## Fitting and decision rules

Three separate L2 logistic heads, `C` in {0.001, 0.01, 0.1}, `StandardScaler` fitted per head on its
known training rows only. `liblinear`, `class_weight=None`, `max_iter=2000`, `tol=1e-6`,
`intercept_scaling=10`, `random_state=20260911`. Within each outer-training partition each of its four
original folds is left out in turn for inner validation, with preprocessing fitted inside each inner
training split. `C` is chosen per head by **pooled inner-validation binary log loss on known labels**,
ties favouring smaller `C`. Heads are then refitted on the outer-training rows and applied to the
untouched outer fold. A fit that reaches the iteration cap is a **hard stop**, reported rather than
worked around.

The edit applies when `E >= t_E` **and** `B <= t_B` **and** `H <= t_H`. Threshold candidates are the
quantiles `[0, .1, .25, .5, .75, .9, 1]` plus the endpoints `{0, 1}`, reused from `choose_rules`, with
`no_edit` seeded as a candidate. The objective maximises retained E substantive outcomes, then
minimises H damage, then B damage, then total edits. Selected unknown B/H outcomes consume damage
budget; unknown E outcomes provide no benefit.

Three profiles, all reported: `zero_validation_damage`, `H1pct_B2pct_validation`,
`H5pct_B5pct_validation`, with denominators equal to all B or all H rows of the inner-validation
population. **These budgets constrain observed inner-validation outcomes; they are not guarantees.**

## Historical references (descriptive only)

Reproduced from their own recorded scores and thresholds:

- residual gate - `residual_p_E >= 0.29185573339357973 AND residual_p_H < 0.06975384354575279`, expected 37 E / 0 B / 0 H
- hybrid residual-E / combined-H - `residual_p_E >= 0.29185573339357973 AND combined_p_H < 0.10970446307367217`, expected 54 E / 0 B / 1 H

Both calibrated thresholds on **pooled** development out-of-fold predictions, unlike this nested
comparison, so differences against them **do not isolate a method improvement**.

## Outputs

`REPORT.md` (comparison table, per-fold results, historical references, within-class quality, paired
differences, checks), `summary.csv`/`summary.json`, `results_by_fold.csv`, `head_selection.csv`,
`per_row_scores_actions.csv`, `paired_differences.csv`/`.json`, `within_class_quality.csv`,
`convergence.csv`, `fold_models.json` (coefficients, intercepts and scaler statistics),
`preflight.json`, `checks.json`, `RUN_KEY.json`, `COMPLETE.json`.

Checks cover split separation, joins, unknown handling, scoring consistency against the declared
inventory, membership of every selected harmful reply in the original set of 20, feature hygiene,
convergence and unchanged input digests.

## Limitations carried forward

Only 20 harmful static replies and 8 material B losses exist, which cannot support calibrated safety
claims. All H rows come from different source datasets from E and B, and routing features can encode
source and topic, so pooled prediction quality alone is insufficient - within-class out-of-fold
quality is reported for that reason. **199 of 480** static responses hit the generation token cap and
282 ended on EOS; this is reported as recorded, with no row excluded or relabelled.
