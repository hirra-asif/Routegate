# Same-region dual-direction controller

> **Superseded on 2026-08-26 by the phase-2 closure.** Phase 2 ran and returned
> **`NO_STRENGTH_QUALIFIES`**: no `alpha_S` both restored ≥ 9 of the 18 damaged
> H rows and caused 0 of 18 on its own. `d_S = H − E` is **retired as an
> actuator**; the H-versus-(E+B) probe is **retained as a sensor** — a brake or
> a veto, never a steering direction — and the continuation is the
> **one-direction controller**: one sigmoid gate `g_R`, applied strength
> `alpha_R = 0.15 · g_R`.
>
> The decision is recorded in
> `manifests/focused_direction_validation_v1_phase2_closure.json`; the
> continuation is specified in `configs/single_direction_controller.json`.
> **Everything below this banner is the design as it stood before phase 2 was
> decided**, kept as the historical record and not rewritten. Where it describes
> two coefficients, that is what was proposed then, not what continues.

**Status: the controller is not implemented.** Its *method* is now declared —
see below and `configs/dual_direction_controller.json → proposed_controller` —
but nothing in this directory trains a coefficient, edits an activation,
generates text or judges a response. What is implemented is phase 1 — the two
frozen probe heads the controller would read — and phase 2, the two direction
vectors themselves. Nothing beyond that.

---

## Research question

Can two prompt-level readings taken at **one** representation region — how much a
prompt looks like an unnecessary refusal, and how much it looks like a harmful
request whose refusal must survive — support one shared, single-pass controller
that emits, per prompt, two sigmoid coefficients — one scaling a fixed repair
direction (**B − E**), one scaling a fixed safety direction (**H − E**), applied
in that same region — without a separate safety gate somewhere else in the
network?

The question is about *sufficiency of one region*. The frozen RQ2 actuator
already edits this region; if the two readings the controller needs are legible
at the same depth, the controller reads and acts in one place. If they are not,
that is the finding, and it is the reason both readings are measured before
anything is built on them.

## Frozen decisions

These were fixed before any head was fitted and are recorded in
`configs/dual_direction_controller.json`.

| | |
| --- | --- |
| **Corpus** | the 480-row development frame, E = 89, B = 178, H = 213 — hash-gated through `manifests/development_inputs.json` |
| **Fold division** | the cross-fitted direction bank's own frozen 480-row `fold_assignment.csv` |
| **Read region** | historical layers 31/32/33 = states 32/33/34 |
| **Primary read state** | state 32, predeclared |
| **Token position** | the final prompt token |
| **Probe** | `StandardScaler` + `LogisticRegression`, the frozen specification in `src/routegate/probes.py`, unchanged |
| **Calibration** | none is fitted |

**The repair head.** Positive E, negative B. **H is never in its training
labels** — not as a positive, not as a negative, and not among the rows its
scaler is estimated from. Every held-out row is scored regardless, so a
cross-fitted `p(E)` exists for all 480 prompts, H included.

The asymmetry is deliberate. What the controller needs from this head is the
E-versus-B over-refusal signal the completed probe analysis measured. Admitting H
rows to its training set or its scaler would change what `p(E)` *means* while
leaving its name unchanged. Scoring H rows with it afterwards is a different act:
applying a fixed function to new rows does not alter the function.

**The safety head.** Positive H, negative E and B **as one class**. The question
it answers is whether a prompt should be refused at all — not why a safe prompt
was or was not answered.

**Cross-fitting.** Within fold *k*: the repair head is fitted on training-fold
E/B rows only; the safety head is fitted on training-fold E/B/H rows; both score
every row fold *k* holds out; each head's scaler is fitted on that head's own
training rows, inside the pipeline. Every one of the 480 rows receives exactly
one cross-fitted `p(E)` and exactly one cross-fitted `p(H)`.

**No calibration model is fitted.** The Brier score in `metrics.json` measures
the probability quality of the logistic outputs. The probabilities themselves are
reported exactly as the frozen pipeline emits them.

## Layer and state convention

The historical index is a **zero-based decoder block**; this package numbers
**stored representations**, where state 0 is the embedding output and state *s*
(1–48) is the output of transformer block *s−1*. They differ by one, always:

| historical layer | state | role |
| --- | --- | --- |
| 31 | **32** | **primary controller read state** |
| 32 | 33 | sensitivity check only |
| 33 | 34 | sensitivity check only |

State 32 is primary because it is the frozen actuator's own first layer — a
declared choice, fixed before any head was fitted, not a measured optimum. States
33 and 34 exist so the primary reading can be shown not to hang on one state.

**No state is selected here.** States 33 and 34 are reported *beside* state 32
and never instead of it. No "best" state is chosen from their outcomes, and the
package deliberately offers no function that would rank them. The completed probe
analysis reported its headline at state 45; that state receives no preference
here, and no probe score selects an intervention site anywhere in this study.

## Fold source, and why it is not rebuilt

The division of the rows is inherited, not constructed. It is the assignment the
cross-fitted direction bank froze:

```
outputs/causal_repair/orthogonalized_population_scan/direction_bank/
    manifest.json          manifest_sha256 6e945d7349cb18c31a2d59e1984a309b72de05e601dc1493cb17a741329d570c
    fold_assignment.csv    sha256          3692fd99cca4521e88176c5fa3e09253b27e9ae80251d92f4d8af8b65739fc6f
```

480 rows, `array_row_index` 0–479 in file order, `row_id` identical position for
position to the frozen extraction manifest and therefore to the activation array.
Its E/B half is `outputs/residual_geometry/corrected/cv_assignments.csv`
**repeat 0** (SHA256 `be3961a2…cd8d1d85`); its H half is that file's deterministic
withheld deal — sorted by `(source_family, row_id)`, dealt round-robin from one
running counter, no seed.

The runner **rebuilds** the division from that stated rule and refuses to continue
unless it reproduces the bank's fold for every one of the 480 rows. Of the twenty
repeats in `cv_assignments.csv`, repeat 0 is the only one that matches. Two
conflicting historical assignments are never silently reconciled, and the pilot's
own assignment is kept for the reproduction gate and for nothing else.

A read state and a direction that were cross-fitted on different divisions would
not be comparable, which is the whole reason the bank's assignment is inherited
rather than a fresh split being drawn. The bank contributes its fold assignment
and nothing else at this stage: its population directions are **not**
automatically adopted as the final controller directions.

## The proposed controller, declared in advance

Fixed now, before any controller exists — recorded in
`configs/dual_direction_controller.json → proposed_controller`:

- **Method.** One **shared coefficient controller**: a single learned function,
  evaluated once per prompt (**single-pass**), shared across the intervention
  states rather than one controller per state.
- **Outputs.** **Two sigmoid values in [0, 1]**: one coefficient controls the
  repair direction, one controls the safety direction.
- **Direction signs.** Repair = **B − E** — towards benign compliance, the
  frozen actuator's own convention. Safety = **H − E** — towards justified
  refusal.
- **Direction source.** The **retrospectively validated triplets**, contingent
  on their passing the frozen validation protocol; if they do not pass, the
  source decision reopens rather than falling back to a default. The existing
  480-row direction bank binds the folds in this phase and nothing more — its
  population directions are not automatically adopted.
- **Where it acts.** The same predicted coefficient pair is reused, unchanged,
  across historical layers 31–33 = states 32–34 — the frozen actuator's own
  region. The primary read position stays the final prompt token at state 32;
  states 33 and 34 remain sensitivity checks on the reading, not candidates for
  a new best layer.
- **Continuous, not a gate.** The proposed controller scales both directions
  continuously. A binary on/off gate is a **comparison baseline**, not the
  proposed controller.
- **Frozen while it trains.** The base model, the two probe heads and the final
  direction artefacts do not change during controller training.

Still unresolved — and each of these **must be fixed before the controller is
trained, not after inspecting generated-response outcomes**:

- the controller input: the state-32 activation, the two probe scores, or both;
- hidden size, depth and activation;
- training loss and regularisation;
- optimiser and training schedule;
- the mapping from a sigmoid output to a maximum physical steering strength;
- *(settled 2026-08-22 — direction normalisation and orthogonalisation; see
  `configs/dual_direction_controller.json → resolved`)*
- exact successful-strength targets, or whatever other supervisory signal;
- the controller-selection criterion and operating point.

A choice taken after seeing generation results would tune the design on the very
endpoint it is supposed to be evaluated on — the same selection bias the
corrected residual-geometry analysis exists to price, reintroduced by hand.

## Phases

| | | |
| --- | --- | --- |
| **1** | **Probe heads** | **specified and implemented.** Two frozen heads, cross-fitted, `p(E)` and `p(H)` per prompt at states 32/33/34. CPU only, no model. |
| **2** | **Direction pair** | **specified and implemented.** repair = B_d1 − E, safety = H − E, from the ten validated triplets; per-case unit, averaged, renormalised. Primary method: no orthogonalisation. No strength. Orthogonalised safety is a predeclared ablation. |
| **2c** | **Causal screen** | **specified and implemented, not run.** 3,214 generations: does repair repair E, does safety preserve or restore refusal on H, and is any restoration more than weakened repair. Generation only; no judging. |
| **2d** | **Phase 1: `focused_direction_validation_v1`** | **specified and implemented, not run.** 871 generations: does the current d_R repair E, and which H rows does it damage? Repair direction only. |
| 3 | Coefficient controller | ~~partially specified: one shared, single-pass controller, two sigmoid coefficients, the pair reused across states 32–34~~ — **superseded 2026-08-26.** Phase 2 returned `NO_STRENGTH_QUALIFIES`; the continuation is one sigmoid gate `g_R` with `alpha_R = 0.15 · g_R`, read at state 32 on the final prompt token. See `configs/single_direction_controller.json`. Input, capacity, loss, optimiser, supervision and selection remain open. |
| 4 | Generation and judging | **not specified.** |

Every choice still open in phases 2–4 must be fixed **before** the controller is
trained — not after inspecting generated-response outcomes — and none has a
default anywhere in this repository. See
`docs/open_questions_pre_controller.md`, which states each open question and what
would settle it.

## What phase 1 produces

`experiments/dual_direction_controller/fit_probe_heads.py` is a thin runner over
`src/routegate/probeheads.py`. It requires an explicit `--output-dir`, refuses one
that already holds anything, reads development data only, exposes no option that
reaches material outside the development corpus, and loads no model.

| file | |
| --- | --- |
| `oof_predictions.csv` | `row_id, final_ebh, fold, state, p_E, p_H` |
| `metrics.json` | ROC-AUC, PR-AUC and Brier per head, per state, as separate records for 32, 33 and 34 |
| `fold_assignments.csv` | the division actually used, one row per prompt |
| `validation.json` | every gate that was evaluated, and its result |
| `run_manifest.json` | input, configuration, code and output hashes |

No prompt text and no response text is written.

`outputs/` is gitignored, so the direction bank lives in whichever checkout it
was built in. If this is a worktree and the bank is not at the default
repo-relative path, pass `--direction-bank` with the path to the bank directory
and, optionally, `--expect-bank-sha256` with the `manifest_sha256` above so the
run refuses any other bank.

## Phase 2 — the validated direction bank

`build_validated_direction_bank.py` builds the two direction vectors from the
frozen RQ2 geometry capture. It loads no model, touches no GPU, generates
nothing and judges nothing. `manifests/rq1_triplet_inclusion.json` is the
authoritative eligibility record and the only one consulted.

| direction | arm | eligibility rule | cases |
| --- | --- | --- | --- |
| `repair` | `B_d1` — the historical d1 donor | `comparator_actuator_source` | 10 validated |
| `safety` | `H` — the harmful contrast | `h_e_direction_source` | 10 validated |

**The recipe, at each layer independently.** For every eligible case,
`delta = arm − E` and `case_direction = delta / ‖delta‖₂`; then average the ten
case directions and renormalise to unit norm. Identical for both. The
subtraction is **arm − E**, never E − arm. Per-case normalisation happens
*before* averaging, so every case carries one vote and a large delta cannot
dominate the bank. A zero, near-zero or non-finite delta is refused rather than
normalised.

**The historical B_d2/B_d3 primary actuator does not enter this bank.**
`primary_actuator_source` is validated for 3 of 10 cases — the other seven were
judged and failed. It remains a comparison baseline;
`src/routegate/directionbank.py` names none of its arms and refuses that rule as
an eligibility source outright.

**No orthogonalisation and no strength — frozen, not merely current.** The
primary method is the two direct directions; their per-layer cosine is recorded
as a diagnostic and nothing reads it to alter either one. No coefficient is
selected, embedded or defaulted. Both are fixed in
`configs/dual_direction_controller.json → direction_construction`, and an
**orthogonalised safety direction** — the safety direction with its component
along repair removed, then renormalised — is predeclared there as an optional
ablation under `predeclared_ablations`. It is not implemented, not run, and
does not replace the primary method.

**Parity is a gate.** The rebuilt repair direction must reproduce
`historical_d1_global_frozen` to float32 rounding — both are `B_d1 − E` over the
same ten cases by the same recipe — or the build stops.

| file | |
| --- | --- |
| `validated_direction_bank.npz` | `repair_direction [3, 2048]`, `safety_direction [3, 2048]`, float32, pickle-free, byte-deterministic |
| `direction_bank_manifest.json` | every input/output digest, case ids and their eligibility-record digests, subtraction order, normalisation recipe, layer/state mapping, shapes, norms, repair/safety cosine, historical-d1 parity, and the explicit no-orthogonalisation / no-strength statements |

The capture is a frozen input that is **not migrated yet**; it is pinned by
digest in `manifests/rq2_geometry_capture.json`. Copy it under the data root
with its digests preserved, add `rq2_geometry_capture_dir` to
`configs/paths.local.json`, then run the builder with explicit paths:

```bash
python experiments/dual_direction_controller/build_validated_direction_bank.py \
  --capture-dir       "$TV_CAPTURE" \
  --inclusion-manifest manifests/rq1_triplet_inclusion.json \
  --source-case-manifest "$RQ2_ACTUATOR/source_case_manifest.csv" \
  --actuator-dir      "$RQ2_ACTUATOR" \
  --output-dir        "$BANK_OUT" [--dry-run]
```

A copy whose digest differs from its source is a failed copy, not a new input:
the builder re-hashes every file against the pinned record and refuses on any
mismatch. `--dry-run` writes nothing and still reports the digest the real run
would produce.

```python
from routegate.directionbank import read_npz
bank = read_npz("<output-dir>/validated_direction_bank.npz")
bank["repair_direction"].shape      # (3, 2048), layers 31/32/33
```

## Phase 2c — the preregistered causal screen

`run_causal_screen.py` puts the two built directions through the development
corpus and records what the model says. It loads a model and needs a GPU; it
calls no API, and it **judges nothing** — generation and judging are separate
stages, and the judging stage is not specified yet.

**The three questions.** Does `d_R = B_d1 − E` causally repair development E
prompts? Does `d_S = H − E` preserve or restore safe refusal on development H
prompts? And is any apparent H restoration more than simply weakening the
repair steering?

The third is why the combined arms exist and why they are shaped as they are.
The four combined arms hold `alpha_R` at 0.15 and vary `alpha_S` alone, so a
restoration that moves with `alpha_S` while the repair coefficient does not
move is not explicable as less repair.

| class | rows | arms | generations |
| --- | --- | --- | --- |
| E | 89 | baseline + repair 0.05/0.10/0.15/0.20 | 445 |
| H | 213 | the same, + safety 0.05/0.10/0.15/0.20, + repair 0.15 × safety 0.05/0.10/0.15/0.20 | 2,769 |
| | | | **3,214** |

B rows built the historical directions and are not steered.

**The edit**, at historical layers 31/32/33 = states 32/33/34, at the final
prompt token, prefill only:

```
delta = ‖block_input‖ * (alpha_R * d_R + alpha_S * d_S)
```

added to the block output. The combination is **not renormalised** — that would
make the two coefficients compete for one budget, so raising `alpha_S` would
lower the repair component even at constant `alpha_R`, and question three would
be unanswerable by construction. The same two coefficients are used at all
three layers; only the directions differ, because the bank stores one per
layer. Neither direction is orthogonalised.

`‖block_input‖` is the target's own residual norm at the *preceding* state —
the budget reference RQ1 used and RQ2 inherited, and not the tensor being
edited.

**The convention gate.** With `alpha_S = 0` this must be the edit
`routegate.steering.build_steering_plan` already performs, and the run checks
exactly that before loading a model. The two are not bit-identical by
construction — `build_steering_plan` divides its direction by its own norm and
this formula does not — so the residual is the bank's unit-norm error. Measured
on the real bank: **5.96e-08**. A larger disagreement stops the run.

**Order of the gates.** Everything that can fail without a GPU does: the bank's
SHA256, the corpus digest, the row counts and ids, the prompt hashes, the class
labels, the full 3,214-item inventory, the convention check and the resume
file's compatibility are all settled before a model is loaded.

**Resume.** A generation is skipped only if all seven of `row_id`, `row_class`,
`arm_id`, `alpha_R`, `alpha_S`, the direction-bank digest and the
generation-configuration digest agree. Records already in an output directory
must also agree on the model, revision, tokenizer and chat-template digests,
the intervention type, the budget reference, the states, the layers and the
dataset digest; a mismatch stops the run rather than mixing two experiments in
one file.

| file | |
| --- | --- |
| `screen_records.jsonl` | one record per generation: row, class, arm, both coefficients, the response and its digest, token counts, per-state plan diagnostics and direction digests, hook counts, generation settings, model revision, tokenizer and chat-template digests, timestamps and environment |
| `run_manifest.json` | the inventory, the shard, every input digest, the convention measurement and the checklist |

Results are written under the data root, outside git.

## Phase 1 — `focused_direction_validation_v1`

A **named, reportable protocol** over the same generation path, added
2026-08-23. It asks two questions and deliberately not a third:

1. does the **current** validated repair direction `d_R = B_d1 − E` repair E
   prompts?
2. which H prompts, if any, does the **current** `d_R` damage?

It does **not** test `d_S = H − E`. That is phase 2, and it is applied only to
the `d_R` failures this phase finds. `alpha_S` is zero in every phase-1 arm.

| class | rows | arms | generations |
| --- | --- | --- | --- |
| E | 89 | baseline + repair 0.05/0.10/0.15/0.20 | 445 |
| H | 213 | baseline + repair 0.15 | 426 |
| | | | **871** |

The E ladder measures the dose-response; H is asked one question at the
mid-ladder setting, which is where the exploratory repair arms and the legacy
static actuator both sat.

Everything else is the frozen configuration already in use: the validated
direction bank at `d759574b…3516bb`, historical layers 31/32/33 = states
32/33/34, final prompt token, prefill only, no orthogonalisation, and the
existing model, revision, tokenizer and generation settings reused verbatim.

**Reportable only when complete.** A protocol is 871 records or it is not that
protocol. Completeness is established by `consolidate_causal_screen.py`, never
by a single shard — a shard is a part, and a short run is not a smaller result.
`--only-row-id`, `--only-arm` and `--limit` runs remain smoke tests: every
record they write is marked `smoke`, which is run-locked, so a smoke record can
never join a protocol's records in one directory.

**The population is the whole development corpus** — all 89 E rows and all 213
H rows. It is *not* defined or selected by the 20 legacy regressions in
`manifests/targeted_h_legacy_regressions_v1.json`: those came from a different
actuator (`consensus23_global_frozen`, donors d2/d3) and remain a separate
secondary stress set. Nothing in the phase-1 code path reads that manifest.

**Running it.** `--protocol focused_direction_validation_v1` on both the runner
and the consolidator. The default stays `exploratory_dual_direction_v1`, so the
original 3,214-generation screen is unchanged in every respect.

## Phase 2 — `focused_direction_validation_v1_phase2`

Frozen 2026-08-25, **not run**. Phase 1 finished and its frozen regression rule
returned 18 H rows that `d_R` damaged. Phase 2 applies `d_S = H − E` to those 18
rows and to nothing else.

| | |
| --- | --- |
| rows | the 18 ids pinned in `dualsteer.PHASE2_SELECTED_ROWS`, digest `a54483a4…49ff1` |
| arms to generate | `safety_005/010/015/020` (`alpha_R` = 0) and `repair015_safety005/010/015/020` (`alpha_R` = 0.15) |
| generations to run | 18 × 8 = **144** — none of which exist yet |
| inherited arms | `baseline` and `repair_015`, already recorded in phase 1 and **never regenerated** — 18 × 2 = 36 |
| analysis table | 144 + 36 = **180**, formed by joining the 36 inherited records to the consolidated 144 |

The consolidated **generation** output is **144 records**, not 180: the 36
references are not expected inside the phase-2 shards, and a `baseline` or
`repair_015` record appearing in one fails `no_reference_arm_was_regenerated`.
The 18 row identities are checked too, not merely their count —
`H_rows_match_the_bound_selection` refuses a run on a different 18.

Two gates run before the inventory is built and long before a model is loaded:
`verify_phase2_selection` and `verify_phase2_predeclaration`, both recorded on
the run manifest's checklist. The predeclaration is written by
`freeze_phase2.py`, which writes atomically, refuses to overwrite a
non-identical file, and re-verifies what landed on disk.

**The decision rule is frozen** in `routegate.phase2.PHASE2_DECISION_RULE` and
mirrored in the predeclaration. A strength qualifies when its combined arm
restores at least 9/18 and its safety-only arm causes 0/18; among qualifiers
the greatest recovery wins, ties break to the smallest `alpha_S`; if none
qualifies, `d_S` is removed as a steering actuator and the H-versus-E+B probe
is retained as a brake or veto. An undecided row leaves the decision
`UNDETERMINED` rather than passing it.

These 18 rows are a selected conditional failure set: `population_reportable`
is `false` and no phase-2 count is a population rate.

## Explicitly not implemented

- the controller itself — declared, not built: no architecture instantiated,
  no parameter trained;
- any **orthogonalised** direction, and any application strength — the two
  direct directions are built (phase 2 above); nothing scales them;
- **judging inside this directory** — the causal screen generates; nothing
  here scores, labels or classifies a response. Phase-1 judging is a
  separate stage: `experiments/focused_direction_judging/`, which reuses the
  frozen judging protocol and has been run and adjudicated; phase-2 judging is
  not specified;
- **any phase-2 generation** — phase 2 is frozen and its plan is verifiable by
  dry run, but `phase2_generations_run` is 0 and no phase-2 response exists;
- any calibration model;
- any use of material outside the development corpus.

Phase 1 produces two readings. It does not decide anything with them.
