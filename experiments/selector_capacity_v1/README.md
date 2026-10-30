# selector_capacity_v1 — is linear prediction too restrictive?

One focused question: on the frozen 480 development prompts, does a **small nonlinear
selector** choose the fixed repair edit better than the two **existing linear selectors**?

Nothing about the intervention changes. All three selectors choose between the same two
already-generated replies: the saved baseline and the saved fixed-edit reply
(direction `B_d1 − E`, strength **0.15**, zero-based layers **31–32**). The only input at
prediction time is the frozen **unsteered state 32** — the 2,048-dimensional output of
zero-based layer 31 at the final prompt token. Topic, task frame, dataset source,
historical controller coefficients and response text are never features.

## The three methods

| Method | Family | Trained to predict |
|---|---|---|
| `class_selector` | existing linear (logistic, `C ∈ {0.001, 0.01, 0.1}`) | E / B / H class membership |
| `outcome_selector` | existing linear (same family) | the three joint outcomes |
| `nn_outcome_selector` | **new** — one hidden layer, 16 ReLU units | the *same* three joint outcomes |

The three prediction targets are reused verbatim from
`experiments/outcome_selector_v1/selector_core.py::outcome_arrays`:

1. **E opportunity** — the prompt is E and the fixed reply is SA or PA-S.
2. **B material loss** — the prompt is B and the fixed reply loses material quality against its own baseline.
3. **H new harm** — the prompt is H, its baseline is NO harmful assistance, and its fixed reply is YES.

Outside a target's class the label is 0 by definition; inside it, a missing judgement stays
missing and is dropped from fitting. No label is changed, added or imputed anywhere.

## The network, as registered before the run

Everything below is fixed in `design.json`, which also pins the SHA-256 of every reused
module so the comparison cannot silently drift.

- **Architecture** — 2,048 standardised inputs → one hidden layer of **16 ReLU** units →
  one logistic output. Three **separate** binary networks, one per target (32,801 parameters each, including biases).
- **Regularisation search** — L2 `alpha ∈ {10, 100, 1000}`, deliberately small. The grid
  mirrors the linear arm's effective penalty `1/C` for `C ∈ {0.1, 0.01, 0.001}`. Chosen per head
  by pooled inner-validation binary log loss; ties favour **stronger** regularisation.
- **Stopping** — full-batch L-BFGS, stop at `tol = 1e-5` on the training objective or
  `max_iter = 1000`, whichever comes first. No internal validation split and no best-epoch
  selection, so early stopping never consumes inner rows. Iterations, final training loss and a
  converged flag are recorded for every fit and reported; no fit is discarded.
- **Seeds** — three fixed seeds `20260912, 20260913, 20260914`. Every score, at inner
  validation *and* at outer evaluation, is the plain mean of the three seeds' sigmoid outputs.
  No seed is ever selected, ranked or dropped.

## Training and evaluation, unchanged

The original five outer folds from `artifacts/gate_oof_v1/fold_assignments.json` are preserved.
For outer fold *k*: set fold *k* aside; inside the other four, leave each out in turn; fit each
head and its scaler on the remaining three; pick `alpha` (or `C`) from pooled inner log loss;
pick thresholds from the same predeclared quantile grid under each validation budget; refit on
all four training folds; apply the already-chosen rule to fold *k*. The decision rule is
identical for all three methods:

    select fixed iff opportunity ≥ t_R and benign_loss ≤ t_B and new_harm ≤ t_H

All three existing validation-budget profiles are reused with their original
threshold-selection rules, and **every profile is reported**:
`zero_validation_damage` (H 0 % / B 0 %), `H1pct_B2pct_validation`, `H5pct_B5pct_validation`.
They constrain inner-validation observed-plus-unknown damage only — they are exploratory
settings, not deployment tolerances or safety guarantees. No profile is picked from outer results.

## Completed comparison

| Profile | Linear class: E / B / H | Linear outcome: E / B / H | Neural outcome: E / B / H |
| --- | --- | --- | --- |
| Zero validation damage | 26 / 1 / 0 | 17 / 1 / 0 | 14 / 1 / 0 |
| H 1% / B 2% | 42 / 3 / 1 | 46 / 4 / 0 | 45 / 3 / 0 |
| H 5% / B 5% | 58 / 7 / 4 | 60 / (10 + 1 unknown) / 5 | 58 / (8 + 2 unknown) / 5 |

Denominators are E /89, B /178 and newly harmful H /212. At the middle
profile, the neural network retains one fewer useful E answer and incurs one
fewer B loss than the linear outcome selector. It is worse at the strict profile.
Its within-H ranking improves (AUROC 0.941 versus 0.846), without a consistent
joint improvement in selected outcomes. Of 585 network fits, 66 reach the
iteration cap; four capped fits contribute to final fold predictions. Seeds and
capped fits remain included as specified. These results do not identify whether
data size, regularisation or threshold selection caused the remaining limitations.

## Run it

```bash
bash experiments/selector_capacity_v1/run.sh selftest    # 17 CPU checks, synthetic data, seconds
bash experiments/selector_capacity_v1/run.sh preflight   # read-only input and inventory verification
bash experiments/selector_capacity_v1/run.sh run         # the comparison, ~10 min, 1 thread
bash experiments/selector_capacity_v1/run.sh status
```

Everything runs inside `$HOME/containers/vllm-openai-v0.10.1.1.sif` with all BLAS thread
counts pinned to 1. No GPU, no network, no judge API calls, no new generations.
Results go to `$HOME/routegate-data/selector_capacity_v1/` plus a `.tar.gz` beside it.
The runner refuses to overwrite an existing output directory or archive; for a separate run
pass `--output <new path>`.

Required inputs, all hash-pinned and verified before and after the run:
`routegate-data/controller_error_audit_v1/cases.jsonl`, the extraction directory named in
`configs/paths.local.json` (`manifest.csv` and `residual_final_prompt_token.npy`),
`artifacts/gate_oof_v1/fold_assignments.json`, and
`experiments/outcome_selector_v1/{design.json,selector_core.py,inputs.py,reporting.py}`.

## What comes out

| File | Contents |
|---|---|
| `REPORT.md` / `REPORT.html` | headline table, per-fold results, gained/lost prompts, H overlap, ranking quality, convergence, checks |
| `summary.csv` / `summary.json` | every policy: edits, E of 89, B of 178 with unknowns, H of 212 with unknowns |
| `results_by_fold.csv` | the same counts per outer fold |
| `gained_lost_prompts.json` / `.csv` | the **exact** row IDs gained and lost against each linear selector |
| `h_regression_prompts.json` | which H prompts regress under each policy, and whether they are the same ones |
| `within_class_quality.csv` | out-of-fold AUC / average precision / log loss inside each head's class, plus the reversed control |
| `class_membership_quality.csv` | each head's AUC against E/B/H membership, descriptive |
| `prediction_scores.csv`, `selected_actions.csv` | every out-of-fold score and every selected response identity |
| `split_separation.csv`, `checks.json`, `convergence.json` | the four required checks and their evidence |
| `fold_models/` | per-fold linear coefficients, network metadata, network weights (`.npz`), inner-validation scores |

## Reading the endpoints correctly

- **E** counts **retained fixed-edit substantive answers** out of the whole 89-prompt E
  inventory. Baseline-selected E behaviour carries no behavioural label, so this is **not** a
  complete-system repair rate and must not be quoted as one.
- **B** counts material loss against the row's own baseline, out of 178, with unknowns kept
  visible and never counted as favourable.
- **H new harm** is counted over the **212 known non-harmful H baselines**. The single H row
  with an unknown baseline is reported separately and is never folded into that denominator.
- The **reversed H-class ranking** figure is a descriptive control showing whether the new-harm
  head orders harm at all. Every method and profile keeps the gate `new_harm_score ≤ t_H`.
  No gate was reversed after seeing outcomes.

## Limits

The prompts that motivated `outcome_selector_v1` are reused, so nested fitting prevents fold
leakage but this is exploratory, not a fresh confirmatory test. Only ten known fixed-edit
new-harm positives exist, which cannot support a calibrated safety claim for any model class.
All H rows come from different source datasets from E/B, so source-related shortcuts remain
possible. A null result here is evidence about *this* feature set at *this* sample size — not
about nonlinearity in general. There is no final all-development refit, no external-380
evaluation and no deployment decision in this experiment.
