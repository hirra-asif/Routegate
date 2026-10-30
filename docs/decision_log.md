# Decision log

> **Dated log.** Each entry reflects what was known on its date; read later
> entries before relying on an earlier one.

Decisions that shaped this package, each with the reasoning that produced it.
Entries are appended, not rewritten; a decision that is later reversed gets a new
entry rather than an edit.

---

## 2026-08-05 — Start a clean repository rather than refactor the working one

The working repository accumulated 178 result directories, several generations of
the labelled corpus, and analysis scripts that carry their own bespoke estimator
implementations. Refactoring in place would have meant every future reader having
to know which of the sibling files is the accepted one.

**Decision.** A new repository containing only what an analysis needs, with the
working repository kept intact as read-only historical evidence.

**Consequence.** The accepted inputs are named once, by hash, in
`manifests/development_inputs.json`, and everything else in the historical
workspace is invisible to this code.

---

## 2026-08-05 — Organise by scientific component, not by research question

`RQ1/`, `RQ2/`, `RQ3/` say nothing about what a directory contains, and they
encode a thesis structure that will not survive contact with a paper.

**Decision.** Directories are named for what they are: causal repair, frozen
actuator, residual geometry, MoE features, dynamic gate, intervention evaluation.
The RQ mapping appears once, in two sentences, in `docs/study_design.md`.

---

## 2026-08-05 — Refuse superseded inputs by filename, before hashing

A blind Axis-2 re-audit on 2026-08-04 moved 25 rows out of E, changing the class
balance from 114/162 to 89/178. The pre-amendment file was kept beside the
accepted one, and it is named `rq3_dataset_final_before_e_axis2_reaudit.csv`
while another superseded revision is named `rq3_ebh_final_analysis_dataset.csv`.
Both read as more authoritative than the accepted `rq3_dataset_final.csv`.

A hash check alone would catch this, but only as an opaque mismatch, and only
after the file was opened.

**Decision.** Named refusal, before any hash is computed, with an error message
that says *which* revision was offered and why it is wrong. Directory-level
refusal for the superseded preparation package, whose frozen folds were built on
the pre-amendment labels.

---

## 2026-08-05 — Import the pilot's folds; do not rebuild them

The pilot's metadata records its cross-validation method as "StratifiedKFold",
but the implementation deals shuffled per-class positions round-robin across the
folds. That is not `sklearn.model_selection.StratifiedKFold`, and it produces
different held-out sets from the same seed.

Rebuilding folds with sklearn would have produced *different numbers* from a
correct implementation, which is the worst possible outcome for a reproduction:
indistinguishable from an implementation error.

**Decision.** Reproduction recovers the assignment from
`best_layer_oof_scores.csv`, the only place it was serialised, and re-binds it to
the analysed row order so a difference in the reference file's own ordering
cannot silently permute the folds. `build_stratified_folds` exists for new work
and is not used by the reproduction. A test asserts the two assignments differ.

---

## 2026-08-05 — scikit-learn, with no fallback

The pilot ran in an environment where scikit-learn was not installed, so its
scaler, logistic regression, fold splitter, metrics and PCA were hand-written.
The objective was specified correctly and the fits converged to a gradient norm
below 1e-6, so the numbers are very probably right — but "probably right, and
unverifiable without reading 1700 lines" is not a foundation for four more
stages.

**Decision.** `Pipeline(StandardScaler, LogisticRegression)`, `roc_auc_score`,
`average_precision_score`, `balanced_accuracy_score`, `StratifiedKFold`.
scikit-learn 1.7.1 is installed into a project-local `.venv` inside the same
Apptainer image the pilot used, so numpy and scipy remain the exact versions that
produced the reference numbers. If scikit-learn could not be installed, the
instruction was to stop and report — not to reimplement it. It installed.

---

## 2026-08-05 — State `tol=1e-6` on the probe rather than inherit scikit-learn's default

**Found while running the reproduction, not predicted.** With the probe exactly
as specified — `penalty="l2", C=0.1, class_weight="balanced", max_iter=5000,
solver="lbfgs", random_state=20260805` — state 45 reproduced at ROC-AUC
0.952657 against the pilot's 0.952594. A difference of 6.3e-05: far too small to
be a wrong dataset, wrong folds or wrong objective, and far too large for the
1e-6 gate.

The cause is the stopping rule. scikit-learn's `LogisticRegression` defaults to
`tol=1e-4`, which lbfgs applies as a gradient tolerance. The pilot set
`gtol=1e-6` and recorded in its own source that this is "an order of magnitude
tighter than scikit-learn's default tol=1e-4", reporting a maximum gradient norm
over all fits of 9.99e-07. At the default, scikit-learn was simply stopping
earlier on the same objective.

Sweeping the tolerance at state 45, against the pilot's folds:

| `tol` | Δ ROC-AUC | Δ PR-AUC | Δ balanced accuracy |
| --- | --- | --- | --- |
| 1e-4 (default) | 6.35e-05 | 1.20e-04 | 4.61e-07 |
| 1e-5 | 3.69e-07 | 2.04e-07 | 4.61e-07 |
| **1e-6** | **3.69e-07** | **2.04e-07** | **4.61e-07** |
| 1e-8 | 3.69e-07 | 2.04e-07 | 4.61e-07 |
| 1e-12 | 3.69e-07 | 2.04e-07 | 4.61e-07 |

From 1e-5 onwards the metrics are identical to every digit, so the estimate is
converged. What remains is not disagreement at all: the reference CSV records six
decimal places, and the reproduced values are 0.952594369, 0.894166204 and
0.865168539 — each rounding exactly to the recorded 0.952594, 0.894166 and
0.865169. The residual differences *are* that rounding.

**Decision.** Set `tol` explicitly to 1e-6, the reference's own value.

**Why this is not a substitution.** `tol` is a convergence setting, not part of
the estimand. Penalty, C, class weighting, solver, iteration cap and random state
are all exactly as specified, and the estimator is still
`sklearn.linear_model.LogisticRegression`. What changed is how far the optimiser
is permitted to stop short of the optimum it is already seeking. Leaving it at
the default would have meant reporting a failed reproduction of a result that
does in fact reproduce — and comparing a converged fit against a half-converged
one.

**Recorded where it will be seen:** in `probes.py` as `PROBE_TOL` with the
reasoning, in both run configs, in `manifests/pilot_reference.json` under
`estimator_tolerances`, in the run's `VALIDATION.md`, and by a test that pins
`tol == 1e-6` and asserts scikit-learn's default is still 1e-4 — so if that
default ever changes, this entry is revisited rather than silently invalidated.

---

## 2026-08-05 — Put the scaler inside the Pipeline

Standardisation fitted on the full design matrix before cross-validation leaks
held-out information into every fold. It is the single most common way a probe
result is quietly inflated.

**Decision.** The scaler is a Pipeline step, so `fit` on the training rows
estimates the mean and standard deviation from those rows only. There is no
separate scaling step that could see the whole matrix.

**How it is verified.** `test_preprocessing.py` corrupts the held-out rows of each
fold with garbage and asserts that no fitted quantity — scaler mean, scaler
scale, coefficients, intercept — moves by a single bit, *and* that the held-out
scores do move. Without the second assertion the corruption could be inert and
the test would prove nothing.

---

## 2026-08-05 — Call it state 0, never layer 0

State 0 is the embedding output, not the output of a transformer block. On this
corpus the rendered chat prompt ends with the same token for every prompt, so
state 0 is one identical vector across all 480 rows and any probe there is pinned
to chance.

Called "layer 0", its ROC-AUC of exactly 0.5 reads as a finding about the first
layer of the network. It is not one; it is arithmetic.

**Decision.** State numbering is explicit in code, configs, output columns and
prose. `describe_state()` renders it, and a test asserts the string "layer 0"
never appears.

---

## 2026-08-05 — Degenerate states are scored, flagged, and barred from maxima

`np.argmax` over an array containing NaN returns the NaN's position. A degenerate
state that produced an undefined statistic could therefore be selected as the
best state, silently. In the pilot, the descriptive centroid table did report
`centroid_distance_best_layer = 0` and
`max_standardized_centroid_distance = NaN` — the failure mode is not
hypothetical, though there it touched only a descriptive statistic.

**Decision.** Three separate guards. `safe_argmax` raises rather than let a
non-finite value win. `best_state` excludes flagged degenerate states by default.
Undefined descriptive statistics are written as empty strings, never as `NaN`.
Degenerate states are still scored and still reported — a constant feature set
carrying no information is the correct answer, not a failure.

---

## 2026-08-05 — Reproduce first; declare the correction but do not run it

The pilot's headline has four design problems (`docs/study_design.md`,
*Corrections owed*), the sharpest being that state 45 was selected by maximising
performance on the rows the score is reported from, and that a magnitude-only
control reached ROC-AUC 0.871 against the probe's 0.953.

Fixing those changes what the headline number *means*. A nested selection
estimate is not comparable to a selected maximum, and the magnitude controls may
reassign most of the separability from geometry to vector norm.

**Decision.** Two entry points. `reproduce_pilot.py` re-derives the frozen
numbers and stops. `run_analysis.py` declares the corrected protocol in full and
refuses to execute without both a passed reproduction gate and an explicit
`--i-have-approved-the-corrected-protocol` flag. The friction is the point: the
substitution should be a decision, not a side effect of running a script.

---

## 2026-08-05 — The held-out split is protected by absence, not by permissions

380 rows are reserved for intervention evaluation. A guard that names the file it
protects has published its location.

**Decision.** This repository contains no sealed prompt, no sealed prompt hash,
and no path to sealed material. `manifests/sealed_set_commitment.json` records
only the counts — 380 total, 180 safe-source, 200 harmful-source — so the split
can be verified when it is opened. The refusal in `validation.py` matches generic
tokens (`sealed`, `final_benchmark`, `holdout_benchmark`, …), never a specific
historical path, so the guard itself does not encode what it guards. Tests assert
that no module, config or entry point names it.

---

## 2026-08-06 — Use the columns that exist, and invent nothing to control for

The corrected protocol calls for a prompt-length control and for categorical
controls. Both invited a fabrication.

**Prompt length.** The natural control is the model's token count. The
extraction manifest records `row_index, row_id, final_ebh, source_family,
prompt_text, prompt_sha256` — and no token count; neither does the dataset, the
extraction metadata or the progress log. Retokenising the prompts would produce
a number that *looks* like the extraction's own and is not: it would come from a
tokeniser loaded by this package, in a different environment, months later.

**Decision.** No token-length control is run. `prompt_length_features` returns a
character count and a whitespace-token count, both named as surface statistics,
and `data.token_count_columns` looks for a genuinely saved column so the absence
is checked rather than assumed. The input gate records
`saved_prompt_token_count_columns: none` in every run, and the config says in
words that no model-token control exists.

**Categorical controls.** The accepted dataset carries `source_family`,
`topic_standardized` and `task_frame` — real adjudicated columns, populated for
all 267 analysed rows. Those three, under those names, are the controls. Nothing
is inferred from prompt text, and no new label is minted.

---

## 2026-08-06 — Fit every transformation inside the fold, and say which ones need it

The magnitude controls are not all the same kind of object, and treating them
alike would have leaked.

- **Unit-L2 normalisation** is a function of one row. Computing it for the whole
  matrix before cross-validation is arithmetically identical to computing it
  inside each fold, so it is done once, deterministically, before
  `StandardScaler` — which is then fitted on training rows only.
- **Norm-residualisation** is a regression. Fitted on all the rows, it carries
  held-out information into every fold. `fit_norm_residualiser` /
  `apply_norm_residualiser` split it so the coefficients come from training rows
  and are applied unchanged to held-out rows.
- **The activation-norm baseline** is one scalar per example, with its own
  scaler and classifier fitted on training rows.
- **The categorical controls** are one-hot encoded by an `OneHotEncoder` that is
  the first step of the pipeline, so its categories are learned per training
  fold; `handle_unknown="ignore"` is what makes that safe, encoding a
  held-out-only category as all-zero instead of forcing a whole-column fit.

**Decision.** `cross_val_out_of_fold` gained a `transform_factory` hook —
called once per fold with the training index — and the whole-sample forms are
kept but marked descriptive. A test corrupts the held-out rows and asserts the
fitted coefficients do not move.

---

## 2026-08-06 — Reconstruct probe directions through the scaler, per fold

A fitted `Pipeline(StandardScaler, LogisticRegression)` scores a row as
`w · (x − mean)/scale + b`, so the weight on the **raw** feature is `w / scale`.
Reading `coef_` as a direction in activation space silently reweights every
dimension by its own standard deviation, and averaging `coef_` across folds
before dividing mixes weights that were never on the same footing.

**Decision.** `probes.raw_space_direction(coef, scale)` does the division, it is
applied per fold, and the fold-specific directions are what is saved and
compared. Where a summary needs one vector, it is the mean of the
*reconstructed* directions, and the output file says so in its own `note`
column.

---

## 2026-08-06 — One split assignment, written before anything is fitted

The corrected analysis compares four activation variants, four controls, 49
states and 200 permutations. If any two of them divided the rows differently,
every difference between them would be uninterpretable.

**Decision.** `build_repeated_stratified_folds` builds 20 × 5 folds once;
`write_repeated_fold_assignments` writes every `(row_id, repeat, fold, split)`
row to `cv_assignments.csv` before the first probe is fitted; the run then
reloads that file and asserts it is bit-identical to the assignment in memory.
Permutations reuse it unchanged and move only the labels. Every out-of-fold
prediction is saved with its row_id, label, state, variant, repeat, fold and
probability, so any of these can be recomputed by a reader.

---

## 2026-08-06 — A permutation must repeat the search, and 20 of them cannot

The pilot permuted labels at the already-selected state and reported p = 0.0476
from 20 permutations, whose p-value floor is 1/21 = 0.048. The reported value
was, within rounding, the smallest number that design could produce.

**Decision.** Each permutation repeats the whole state search over every
included state and contributes its **maximum**, and there are 200 of them, which
lowers the floor to 0.005. The observed statistic is computed by the same
function as the null (`search_maximum`), so the two cannot drift apart, and a
test asserts it. Degenerate states are excluded from the search on both sides.

---

## 2026-08-06 — Refuse a subgroup that cannot be cross-validated; cap the folds; flag the rest

Within-source analysis on this corpus means `false_reject_test` with E = 7,
B = 43 and `orbench_hard` with E = 82, B = 135. Five-fold cross-validation of
seven positives puts one or two of them in each held-out fold.

**Decision.** Three rules, in `folds_for_subset` and the run. A source is
skipped outright when either class has fewer than two examples — it cannot be
cross-validated at all. Otherwise the fold count is capped at the smaller class
count. The exact E and B counts travel with every output row, and a source whose
smaller class is below ten is flagged `unstable` with the reason attached, so a
per-source number cannot be quoted without its own caveat.

---

## 2026-08-06 — A smoke test is a first-class artefact, and says so in its own path

Every code path in the corrected analysis should execute before the two-and-a-
half-hour run does, but a smoke run produces numbers, and numbers escape.

**Decision.** `--smoke` runs a subset — four states, two repeats, five
permutations — and is exempt from the approval flag, because it is not a
substitution for anything. In exchange it is fenced: it refuses to write to a
directory whose name does not contain `smoke`, a reportable run refuses to write
to one that does, every record it writes carries `"reportable": false`, and its
`VALIDATION.md` opens with a bold non-reportable banner. Two tests assert the
two refusals.

---

## 2026-08-06 — Parallelism changes the wall clock and nothing else

The corrected analysis is ≈2.5 CPU-hours, which is a compute-node job, and
`--jobs` exists to make it a shorter one.

**Decision.** The unit of work is one fit on one fold over a shared read-only
feature cube, dispatched through `joblib`, with `inner_max_num_threads=1` so
BLAS stays pinned to a single thread inside every worker exactly as it is in a
serial run. Nothing accumulates across tasks and results are collected in
submission order, so the number of workers cannot change a number.

**Verified, not asserted.** The smoke test was run at `--jobs 4` and at
`--jobs 1` into two directories, and every deterministic table — out-of-fold
predictions, state metrics, the permutation null, the nested selection folds,
the reconstructed probe directions, the projection and the split assignment —
is byte-identical between them. The run records `jobs` so the check can be
repeated.

---

## 2026-08-06 — Freeze the reproduction script rather than refactor it

`runrecord.py` now holds the generic run-record writers, and
`reproduce_pilot.py` has its own copies of two of them.

Refactoring the reproduction onto the shared module would remove that
duplication. It would also mean that `outputs/residual_geometry/pilot_reproduction/`
was produced by code that no longer exists, and that a future edit to a shared
helper could change a frozen reproduction record without anyone deciding to.

**Decision.** The reproduction script is frozen. The duplication is six lines of
CSV writing, and what it buys is that the reproduction record stays byte-
reproducible from the code that actually produced it.

---

## 2026-08-06 — Recover the intervention from the historical code, not from its write-up

The next stage patches activations causally, and the earlier work already did
that. What it did was worth reading precisely rather than paraphrasing, because
three separate prose descriptions of it exist and none of them is the operation.

Read: `pathway_probe.py` (`build_site_plan`, `PathwayController`, site
`residual_stream`), `pathway_control_sweep.py`, `pathway_grid_sweep.py`
(`compute_layer_bands`), `rq2_freeze_final_method.py` (the frozen actuator,
which reuses `build_site_plan` unchanged) and `data/pathway_probe_cases.json`.

What is actually there:

```python
budget    = strength * vector_norm(target["residual"][layer][slot])
raw_delta = donor[site][layer][slot] - target[site][layer][slot]
scaled    = unit_scaled(raw_delta, budget)          # unit direction * budget
changed[0, row] = changed[0, row] + delta           # added, not substituted
```

So it is neither of the two operations it is usually described as. It is not a
full activation replacement — replacement exists, but only as a separate
`replace_donor` positive control on the same site. And it is not a raw
donor-minus-target addition — the difference is unit-normalised and then given
an explicit L2 budget. It is a **normalised direction at a target-relative
budget**.

The hook is `layer.register_forward_hook` on `model.model.layers[i]`, and it
modifies the decoder block's **output** hidden state. The position is
`DecisionSlot(forward_index=0, row_index=-1, slot=0)` — the last row of the
prefill forward, at window 1, which is the final prompt token immediately before
the first generated answer token. Decode forwards are reached only at window > 1
and this stage does not use them.

**Decision.** Preserve that operation exactly: `intervention.build_plan` is the
same arithmetic, and `donor_replace` is kept available as the historical
positive control rather than being made the default. The state numbering is the
one thing deliberately translated — the historical code indexed decoder blocks
directly, so its "layers 31, 32, 33" are this package's states 32, 33, 34, and
every record carries both numbers.

---

## 2026-08-06 — Size the edit against the block input, which is not the tensor being edited

The easy misreading of `build_site_plan` is that the budget comes from the
tensor being patched. It does not. `target["residual"]` is captured by
`layer.register_forward_pre_hook` from `kwargs["hidden_states"]` — the decoder
block's **input** — while the delta lands on the block's **output**. The
historical code had four intervention sites (attention, summed MoE, expert
content, residual stream) and one budget rule, so their edit sizes were
comparable across sites; that only works if the budget is read from somewhere
common to all four.

In this package's numbering that rule reads cleanly: the input of the block
whose output is state `s` *is* state `s − 1`.

**Decision.** `block_input` is the default and is what the scan will run.
`block_output` is available and is written into every record as
`budget_reference`, so the two can never be confused after the fact. State 0 has
no preceding state and therefore no historical rule; it references its own norm
and `budget_reference_note` says so in the record rather than in a comment.

---

## 2026-08-06 — Fail closed on thinking, where the historical code failed open

`pathway_probe.build_chat` calls `apply_chat_template(..., enable_thinking=False)`
inside a `try`, and on `TypeError` retries the same call without the flag. A
tokeniser that does not accept the argument therefore produces thinking-enabled
text, silently, under a function whose name says thinking is off. The later
frozen generation code does not do this.

**Decision.** `runtime.build_chat_strict` raises. A run that cannot demonstrate
thinking is disabled does not run. The same reasoning pins batch size to 1
rather than defaulting it: the intervention position is "the final prompt
token", and with padding the last row is a pad token while everything still
appears to work.

---

## 2026-08-06 — Four things the historical implementation does not pin

Recorded because they are the honest limits on "this is the historical method",
not because any of them is being reproduced.

**Revision.** `pathway_probe.py` takes `--model` (default `Qwen/Qwen3-30B-A3B`)
and passes no `revision` to `from_pretrained`. The commit
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` is pinned by the RQ3 development
generation and by the residual extraction, not by the causal-repair runs. The
historical patching runs cannot be proven to have used that snapshot.

**Generation length.** Four values are in play: `pathway_probe.py` defaults to
48, the stage-1a shell scripts pass 96, the confirmation run
(`pathway_confirmation_512`) used 512, and the development corpus this stage
compares against was generated at 768.

**Strength.** RQ1's grid selected 0.20; the RQ2 freeze fixed 0.15. Both are
"the frozen strength" in their own document.

**Decision.** Pin the revision to the extraction's commit and verify the
tokeniser-config and chat-template hashes on load, because a patched response is
only comparable to a recorded original response if the model that produced them
is the same one. Generate at 768 for the same reason — matching the corpus
matters more than matching RQ1's 512 — and record the discrepancy in
`configs/causal_repair.json` rather than reconciling it. Start the scan at 0.20
and treat it as a starting point, not a frozen method: this stage is a
discovery scan, and a strength sweep is a separate decision made after it.

---

## 2026-08-06 — Pair donors on metadata, because the historical pairing cannot scale

The historical donor was not a corpus prompt. For each of ten hand-built cases
it was the same refused prompt re-rendered under a disambiguating system
instruction (`HELPFUL_SYSTEM`), except where that donor was itself refused, in
which case a human wrote a benign rephrasing and left a `donor_note` explaining
why — 16 of the 17 cases carry one. Whether the donor was usable was decided by
generating its answer and keyword-checking it (`donor_answered`).

Three of those inputs are unavailable here: a human read of every prompt, the
donor's own generated response, and a keyword refusal check being treated as
evidence. And 89 targets is not ten.

**Decision.** Donors are B rows of the accepted corpus, matched on
`source_family`, `topic_standardized` and `task_frame`, with prompt length in
characters as a within-tier tiebreak. Matching degrades through a fixed ladder
(`source_topic_task` → `topic_task` → `topic` → `task_frame` → `length_only`)
and every pair records the tier it was satisfied at and whether it is an exact
three-column match, so fallbacks can be dropped by a downstream reader instead
of being invisible. The pairing is a pure function of the dataset's metadata,
frozen and hashed before the model is loaded, and the run refuses any manifest
but the one it was given. It never sees an activation, a generated response, a
patching result or anything from the sealed set.

Length is measured in characters, not tokens: the extraction saved no token
count and this package loads no tokeniser, so a token length here would be a
number invented after the fact.

---

## 2026-08-06 — Layers 31–33 were a band in a five-band grid, not a localisation

`compute_layer_bands` returns five fixed three-layer bands for a 48-layer model:
(10,12), (16,18), (23,25), (31,33), (40,42) — an anchor at 10–12 plus centres at
roughly 35%, 50%, 67% and 85% depth, rounded half up. The bands are specified
before any data is seen; 31–33 is the one that won a grid sweep over the ten
enabled hand-built triplets.

That is a legitimate result about those five candidates on those ten cases. It
is not evidence that the effect lives at 31–33, because 43 of the 48 blocks were
never tried.

**Decision.** Scan all 49 states on the accepted development corpus, one state
at a time, and let 32–34 (the historical band in this package's numbering) be a
prediction the scan can confirm or refute rather than an assumption it inherits.

---

## 2026-08-06 — Patch the prefill forward only, and prove the patch fired

`generate` calls the model once for the prompt and once per generated token. A
hook keyed on "the last row" without a forward counter would edit every decode
step in turn — a different experiment, and one that still produces plausible
text. The historical controller counted forwards with a pre-hook on decoder
layer 0; this one counts on the top-level module instead, so the embedding hook
sees the same index as the block hooks.

The failure that worries me more is quieter: a patched run whose plan never
applied is byte-identical to its baseline and would be recorded as an
intervention that did nothing.

**Decision.** `PatchController.verify_applied` asserts every plan that declared
itself applicable fired exactly once, and raises otherwise. `reset_run` clears
captures, plans, counters and hit records and requires the example's identity,
so a plan built for one target cannot be applied while another is in flight.
Every handle is kept and `close()` removes all of them.

A self-patch is the control that proves this: the target is its own donor, the
raw delta is exactly zero, `unit_scaled` returns `None` rather than inventing a
direction, and the record says `zero_delta`. State 0 is a second such control
that comes free — the rendered prompt ends with the same token for every prompt,
so the donor and target embedding outputs are identical.

---

## 2026-08-05 — Do not commit outputs, and never commit an activation array

The array is 192 MB and is fully determined by the extraction that produced it,
which is itself hash-identified. Run records are reproducible from the committed
code plus the hash-verified inputs.

**Decision.** `.gitignore` excludes `outputs/`, `*.npy`, `*.pt`, `*.bin`,
prompt-bearing datasets, caches and logs. A test asserts no `.npy` or `.pt` file
exists anywhere in the repository tree.

---

## 2026-08-07 — Population steering, not matched-donor patching, is the primary experiment

The donor-matched all-state scan was built to answer "can a benign donor's
direction repair an over-refused prompt?" It is a good question and the machinery
for it is tested. It is the wrong *first* experiment for locating a causal
region, for two reasons.

It carries a pairing confound. 89 targets need 89 donors, and the metadata ladder
delivers most of them at a fallback tier, so a per-target result mixes "the
direction worked" with "this particular donor was a good match". A population
direction estimated from class means has no pairing to be wrong.

And it has no H arm. A repair that works by suppressing refusal in general is not
a repair, and only the harmful class can tell the two apart. The donor design has
no place to put H.

**Decision.** The primary discovery experiment is cross-fitted, H-orthogonalised
layer-wise causal activation **steering**: one population direction per fold and
state, added at one transformer-block output, over all 89 E and all 213 H rows.
The donor stage is kept exactly as it was written and tested, and becomes the
confirmatory step at whichever state(s) the scan selects — exact matched-B
activation patching on the small, manually verified set, where a handful of
carefully argued pairs is a strength rather than a scaling problem.

Nothing of the donor work is modified or removed. The steering stage got its own
record schema (`steerrecord`) rather than widening `patchrecord`, because a
steering row has a fold and a direction hash and no donor, and widening the field
list would have made every existing donor record fail its own reader.

---

## 2026-08-07 — Orthogonalise the E−B direction against H−B before using it

`mean(E) − mean(B)` is the over-refusal signature and `mean(H) − mean(B)` is the
justified-refusal signature. Both are refusal-associated. There is no reason to
expect them independent, and if they overlap substantially then adding the
negative of the first is partly adding the negative of the second — an
intervention that "repairs" E by making the model refuse less, everywhere.

That failure would be invisible in an E-only experiment. It would look exactly
like success.

**Decision.** Remove the component of `false_refusal` along `true_refusal` before
normalising, and steer with the negative unit of what remains. The H arm then
measures something the direction was explicitly constructed not to move, which
is a much sharper test than measuring it with a direction that was free to move
it.

The raw, non-orthogonalised direction is not discarded — it is one of the
focused controls at the selected state, where its comparison against the
orthogonalised one is informative and affordable.

---

## 2026-08-07 — Search all 48 blocks; inherit no layer from the earlier pilot

The earlier intervention experiment used a small number of prompts and a
five-band grid over a 48-block model. Whatever it found is a statement about
those candidates on those prompts. It is not a localisation, and it must not be
allowed to become one by the back door — through a state list, a plot's x-range,
a scheduler that gives some depths more attention, a selection rule that breaks
ties toward them, or a sentence in an analysis that treats them as the expected
answer.

**Decision.** The scan intervenes at every one of the 48 transformer-block
outputs, once each, at one frozen strength. The array job's seven tasks are a
contiguous partition of the block axis and nothing else. No historical layer
appears in any state list, weighting, ordering or default in this stage's code or
configuration. The historical numbering survives only as provenance in the donor
stage's records and prose, where it is labelled as such.

The completed geometry and linear-probe analysis does not choose a state either.
It establishes where E versus B is *decodable*, which is descriptive evidence
about the representation; this experiment asks which blocks are causally
responsive, which is a different question. The probe coefficient is not used as
the repair direction.

---

## 2026-08-07 — Inherit the E/B folds; deal the H rows with no random number

The corrected residual-geometry run already divided the 267 E and B rows into
five folds, twenty times over. Rebuilding that division here would hold out
different rows and quietly change which prompts each direction was allowed to
see, for no gain.

The 213 H rows have no assignment to inherit.

**Decision.** Repeat 0 of `outputs/residual_geometry/corrected/cv_assignments.csv`
is inherited by `row_id`, unchanged, and an audit asserts that every one of those
267 assignments survived the composition. The H rows are sorted by
`(source_family, row_id)` and dealt round-robin from a single running counter.
That has no seed to record, no shuffle to reproduce, and no dependence on file
order or dictionary iteration order: sorting by family first keeps each family
spread across the folds, and the counter carrying across family boundaries keeps
the global sizes within one row of each other.

The whole assignment is frozen and hashed into the direction bank before the
first generation.

---

## 2026-08-07 — Verify the cross-fitting per generation, against what was written

"The direction was built from training rows only" is the claim the entire scan
rests on, and it is the kind of claim that is easy to state and easy to break
later — by a fold rule that drifts, an off-by-one in a positions array, or a
resumed run reading a bank someone rebuilt.

**Decision.** `build_direction_bank` can only be told about training rows: the
held-out ones are simply not passed to it, so there is no argument that could let
one through. The bank then writes out the training and held-out row identities
per class and fold, and the scan asserts, on every single generation, that the
target appears in the held-out list of the fold whose direction it is about to
receive and in none of that fold's training lists. The check reads the bank's
written record rather than recomputing the fold rule, so an assignment that
changed after the bank was frozen fails it.

A test perturbs one held-out row's activations by a large amount and asserts the
fold's direction is unchanged bit for bit, while a fold that was allowed to see
that row does move.

---

## 2026-08-07 — State 0 is stored, described, and refused as an intervention site

The stored tensor has 49 representations and the model has 48 transformer blocks.
State 0 is the embedding output before block 0. On this corpus every rendered
chat prompt ends with the same token, so all three class means coincide there and
both population differences are exactly zero.

Calling that "layer 0" would put an embedding representation into a
transformer-block ranking, and silently dropping it would leave the numbering
ambiguous.

**Decision.** State 0 is stored in the direction bank for descriptive direction
analysis and is recorded there as degenerate, with the reason named. It is
excluded from the causal scan, from the generation budget, and from the primary
layer ranking. The run script refuses `--states 0` on the command line rather
than dropping it, the bank refuses to hand out a state-0 direction, and the
record writes a blank rather than a number where a transformer block would go.

A degenerate state at a *transformer block* is the opposite case: it is a failure
of the estimate, not a control, and it stops the build.

---

## 2026-08-07 — Freeze 0.2 for the scan, and say plainly what that value is

0.2 is a pre-existing pilot operating value. It is not a measured optimum, and
the experiment that produced it was small — it cannot support the claim that the
effect is best seen there, and no part of this stage should be written as though
it can.

Searching depth and magnitude jointly across 48 blocks is 48 × k generations for
k strengths, and a scan that varies two things cannot attribute a difference to
either.

**Decision.** One strength for the whole scan, at the pre-existing value, with
the run script refusing any other and its refusal message saying why: not because
0.2 is known to be right, but because holding one axis still is what makes the
other axis interpretable. A small strength-sensitivity analysis follows at the
selected state(s), where it is cheap and where its answer is about a specific
site rather than about all of them at once.

---

## 2026-08-07 — Do not write the layer-selection rule yet

The temptation is to pick the state with the largest E repair. That is the one
selection rule this experiment must not use: an intervention that repairs every
E prompt by disabling refusal would win it outright.

Writing any rule now, before the results exist, is also the point at which the
earlier pilot could re-enter through a tie-break.

**Decision.** No post-hoc selection rule is implemented at this stage. Complete
per-state results are saved for both arms so that a criterion — one that weighs
held-out E improvement against held-out H preservation — can be specified before
the generated outcomes are looked at. The earlier small result is not used to
break ties. Nothing in this stage claims success; it produces responses and the
record of how they were made.

---

## 2026-08-07 — Report the fresh baseline against the frozen labels, and delete nothing

The corpus labels were adjudicated on responses generated earlier. The scan
generates a fresh baseline for all 302 targets, and some of those baselines will
disagree with the label the row carries: an E row whose fresh baseline answers
the prompt, or an H row whose fresh baseline complies.

Two wrong answers are available. Dropping the disagreements makes the target set
a function of the run — a different seed, a different day, a different
`max_new_tokens` and the denominator moves, which is not a frozen development set
at all. Ignoring them is worse: a target whose baseline no longer refuses has no
unnecessary refusal left to repair, so counting a "transition" on it inflates the
repair rate with rows where no repair was possible.

**Decision.** The frozen target set stays at 89 E and 213 H. No row is deleted
on the strength of its fresh baseline, and the fresh baseline never changes a
frozen E/B/H label or a fold assignment — it is a measurement of the set, not an
edit to it.

The primary behavioural-flip analysis reports **both** readings: **A** over the
complete frozen target set, and **B**, the refusal transitions, over the
baseline-confirmed subset — the E rows whose fresh baseline still refuses
unnecessarily and the H rows whose fresh baseline still declines the harmful
request appropriately.

Every write-up states the number *and* percentage of E baselines that reproduce E
behaviour, the same for H, and the **row IDs** of every non-reproducing row.
Non-reproducing rows are never silently removed; A includes them, and B names the
ones it set aside.

Which responses count as refusal, substantial safe answer or unsafe compliance is
decided by the separate adjudicated judging step. The scan itself classifies
nothing, applies no keyword rule, and writes no behavioural label.

---

## 2026-08-07 — Freeze the primary state-selection rule now (supersedes "do not write the rule yet")

The entry above dated today, *Do not write the layer-selection rule yet*, said
complete per-state results would be saved so a criterion could be specified
before the outcomes were looked at. That was the right sequencing and this is the
moment it named: the outcomes do not exist, so the rule can be written without
being able to see what it selects. This entry supersedes it, per this log's
convention of adding rather than editing.

**Decision.** For every transformer-block output state 1–48, independently:

* **E repair** — the proportion of baseline-confirmed E targets that change from
  unnecessary refusal to a substantial safe answer;
* **H safety failure** — the proportion of baseline-confirmed H targets for
  which steering produces substantive unsafe compliance with the harmful request.

A state is eligible as a safety-preserving actuator state **only if it produces
zero new unsafe-compliance cases on baseline-confirmed H relative to baseline**.
Among eligible states, the primary actuator state is the one with the **highest E
repair rate**.

The order matters and is the point. Selecting on E repair across all 48 states
would be won outright by any state where the edit simply stops the model
refusing, which is the exact failure the H arm and the orthogonalisation were
built to expose. The safety gate is a precondition, not a term to be traded off
against repair, and its threshold is zero rather than "few" because a
pre-specified criterion with a tunable number in it is not pre-specified.

Ties break, in order, toward the smaller intervention-induced behavioural change
on H, and then toward the earlier transformer block. The second is arbitrary and
is chosen precisely because it is: an arbitrary rule fixed in advance cannot be
steered toward a preferred depth, and "earlier" is not a prediction about where
the effect lives.

If no state is eligible, **no primary safe actuator is force-selected.** The E/H
trade-off is reported across the states with the explicit statement that no state
satisfied the pre-specified criterion. A protocol that always returns an answer
is a protocol that cannot report a negative result.

Neighbouring states may additionally be described as a causal region. The region
is reported alongside the primary state and never in place of it.

Not admissible as an input to this selection, for choosing or for tie-breaking:
previous RQ1 layer results, probe ROC-AUC, historical states 32–34, and visual
inspection of plots. The last one is on the list because it is the easiest to do
by accident.

**How it stays frozen.** The rule lives in
`configs/orthogonalized_population_scan.json`, which the builder embeds verbatim
into the direction bank's `config.json`, so the bank's `manifest_sha256` covers
it. The scan and the scheduler both pin that hash, and every generated record
names it, so a rule edited after the outcomes exist invalidates a hash the
outputs already carry. `tests/test_scan_protocol.py` pins the rule literally.

---

## 2026-08-07 — Name what the selection is made of, so the probe result cannot leak into it

Two analyses in this package produce a ranking over states, and they are not the
same ranking. The probe reports where E and B are linearly *decodable*, with a
headline at state 45. The scan reports where an added direction *changes
behaviour*. Nothing forces those to agree, and the risk is not that they
disagree — it is that a reader, or a later version of me, treats the first as a
prior for the second and quietly narrows the search or breaks a tie with it.

**Decision.** Stated once, in the configuration, the README and the study design:
primary state selection is based on generated behaviour, not on geometric
separability, linear decodability or any probe score. **Probe state 45 receives
no preference** — not as a candidate, not as a prior, not as a tie-break.
Geometry/probing and causal-state selection are independent analyses answering
different questions, and their agreement or disagreement is a finding rather than
a check on either.

Exact activation patching and the detailed MoE zoom-in — residual, attention,
routing and expert-related signals — begin only after the primary state and any
described region are frozen by the rule above. A mechanistic account of a region
chosen after seeing the mechanism is not evidence.

A test walks the configuration and asserts that historical states 32–34 and probe
state 45 are named only in the sections that rule them out, so a preference
cannot re-enter through prose.

---

## 2026-08-21 — The frozen actuator's strength is 0.15 (corrects the study design)

`docs/study_design.md` described the frozen actuator as "CPR-Min at strength
0.10". That is the selection of **2026-07-20**
(`rq2_final_method_selection_20260720_141145`), which chose the single-donor
fixed-mean direction and recorded `"strength": 0.1`. It was superseded the next
day. The freeze of 2026-07-21 15:00 fixed **`consensus23_global_frozen` at
0.15**, and every artefact downstream carries that value: the frozen direction
manifest, the preregistered protocol, the unseen evaluation
(`rq2_final_frozen_unseen_eval_20260721_172100`) and both static E/B/H transfers
(`rq3_e_rq2_transfer_20260809`, `rq3_bh_rq2_static_20260810`). "CPR-Min" survives
in the freeze only as the name of the *scaling convention*, which did not change;
the direction and the strength both did.

**Decision.** `docs/study_design.md` is corrected, and states the correction
rather than quietly replacing the number. No frozen result, historical manifest
or recorded count is touched: 0.10 remains the correct value in the 2026-07-20
record, which is a true account of what was selected that day.

`routegate.actuator` refuses to load a freeze whose recorded strength is not
0.15, and names 0.2 in the refusal so the two cannot be conflated by accident.

---

## 2026-08-21 — What the RQ1 grid actually preferred (corrects 2026-08-06)

The entry of **2026-08-06, "Four things the historical implementation does not
pin"**, records: "**Strength.** RQ1's grid selected 0.20; the RQ2 freeze fixed
0.15." The located grid summaries do not support the first half.

Every RQ1 run that completed ranks configurations safety-first — zero harmful
breaks outrank everything, and a configuration with any harmful break can never
outrank one without. Under that ordering `L31-33_w1_s0.15` **outranks
`L31-33_w1_s0.2` in all three completed summaries**:

| run | rank of `L31-33_w1_s0.15` | rank of `L31-33_w1_s0.2` | harmful breaks at 0.2 |
| --- | --- | --- | --- |
| `pathway_supervisor_layer_extension_20260707_234542` | 2 of 18 | 11 of 18 | 1 |
| `pathway_confirmation_256_20260708_023720` | 2 of 8 | 3 of 8 | 1 |
| `pathway_confirmation_512_20260708_172847` | **1 of 2** | 2 of 2 | 1 |

**Read that table precisely, because an earlier draft of this entry did not.**
What holds across all three runs is the *band-level preference*: at layers 31–33,
0.15 outranks 0.2 every time. What does **not** hold is "0.15 ranked first" —
overall it placed **second** in the two earlier runs, behind a window-6
configuration at the same band, and reached first place **only in the final 512
confirmation**, which is the run that had already fixed the band and the window.
So the supportable claim is: 0.15 is the band-level preference at 31–33, and the
overall selection in the final confirmation. It is not the outright winner of
every completed grid.

On raw repair count 0.2 does lead (9 selective repairs against 6), so a reading
exists under which "the grid selected 0.20" is defensible — but it is not the
ordering the runs recorded, and the recorded ordering is the one the runs were
designed around. Two further caveats: `pathway_grid_sweep_stage1a_20260706`
completed 4 of 1 200 generations and covers band 10–12 only, so it is not
evidence for any band; and **every count in these summaries is a provisional
keyword readout**, labelled as such in the summaries themselves. The manual
semantic audits exist beside them and have not been verified against these
numbers here.

**Decision.** Three things follow, and none of them changes a number.

1. The 2026-08-06 entry stands as written — this log is appended to, not
   rewritten — and this entry is the correction that supersedes its strength
   claim.
2. **0.2 is described as what it is**: the declared operating strength of the
   population scan, carried forward from the pilot so that one axis is fixed
   while the block axis is searched. It is not a measured optimum, and it is not
   the frozen actuator's strength. `src/routegate/steering.py` already says this
   at `FROZEN_SCAN_STRENGTH`; `docs/study_design.md` now says it too.
3. **Any RQ1 grid count reproduced anywhere in this repository is marked
   provisional** until the linked manual semantic review has been verified
   against it. That verification has not been done and is not in this phase.

### `configs/causal_repair.json` — corrected, after checking what depends on it

That file carried the claim this entry corrects, in two places:
`historical_strengths.rq1_grid_best: 0.2`, and a `strength_rationale` calling
0.2 "the RQ1 grid's first-ranked window-1 configuration". Editing it was held
back until it could be shown to break nothing, because `docs/provenance.md` and
`docs/standalone_migration.md` both described it as hash-covered by the direction
bank's `manifest_sha256`.

**They were wrong about that, and the check settles it.** The file's digest
`3648ff5b…9077c9b044` appears nowhere under `outputs/`, `artifacts/`,
`manifests/`, `logs/` or the machine-local data root. Seventy-three JSON files
under those roots were parsed and searched for the whole config and for each of
its six load-bearing sections (`recovered_from_history`, `intervention`,
`donor_pairing`, `generation`, `contrast`, `states`): **zero embeddings**. What
*is* embedded in `direction_bank/config.json` (at `protocol`) and in the scan run
record (at `config`) is `configs/orthogonalized_population_scan.json`; and
`configs/residual_geometry.json` is embedded in
`outputs/residual_geometry/corrected/config.json`. Those two are hash-covered and
are not touched. Both documents are corrected.

**Decision.** Correct the file, additively, preserving what it said:

| field | before | after |
| --- | --- | --- |
| `historical_strengths.rq1_grid_best` | `0.2` | removed as a claim; kept verbatim under `superseded_value` with why it was wrong |
| `historical_strengths.rq1_grid_selected` | — | `0.15`, with the three-run ranking table as `..._evidence` and the provisional-keyword caveat as `..._caveat` |
| `historical_strengths.rq2_frozen_actuator` | `0.15` | `0.15`, now with `..._evidence` naming the freeze and the three runs that carried it |
| `historical_strengths.population_scan_pilot_value` | — | `0.2`, with `..._status` stating it is **not** a measured optimum, **not** the RQ1 selection and **not** the actuator's strength |
| `intervention.strength` | `0.2` | `0.2` — **unchanged**; the protocol was never wrong, only its description |
| `intervention.strength_rationale` | "the RQ1 grid's first-ranked window-1 configuration" | the population-scan pilot value, explicitly not a measured optimum, with the actuator's 0.15 named as a different setting on a different artefact |

Nothing is deleted and no declared protocol changes. `tests/test_patch_states.py`
reads only `historical_layers`, `historical_layers_as_states` and
`intervention.available_types`, none of which moved; a new test pins the
corrected strengths so the old attribution cannot return.

---

## 2026-08-21 — Migrate the frozen actuator; read the `.pt` without unpickling it

`artifacts/README.md` deferred one question explicitly: whether the historical
`frozen_directions.pt` should be "retained as-is or converted to a safer tensor
format", to be "decided during RQ2 migration ... against the real file rather
than in the abstract". The file is now here, so the question is answerable.

The archive is a stored (uncompressed) zip of eight members: one pickle, three
short metadata markers, a serialization id, a version, and two 24 576-byte
float32 storages. The two storages are the entire payload. The pickle carries
only metadata and the storage wiring, references exactly three symbols
(`torch._utils._rebuild_tensor_v2`, `torch.FloatStorage`, `collections.OrderedDict`),
and holds 58 distinct string atoms, all of them settings, ids or construction
sentences.

**Decision.** **Keep the historical bytes; do not convert.** Converting would
produce a file whose SHA256 is not
`ee23afd98c5b9c39cc864af0e475392a69deb5ed2ef33a30d9f10703b5cb2bea`, and that
digest is the only thing tying the actuator to the three completed runs that used
it. A converted file would have to be trusted on the say-so of the conversion.

The safety concern that motivated the question is answered by the **reader**
instead. `routegate.actuator` never calls `pickle` or `torch.load`. It re-hashes
all ten migrated files, checks the archive's member set, member sizes, byte order
and format markers, decodes `data.pkl` with `pickletools.genops` — a byte-stream
decoder that constructs nothing and imports nothing — to confirm the three
globals and the 58-atom metadata digest, and then reads the tensors from the raw
`data/N` members with an explicit little-endian float32 dtype. The pickle is an
integrity check, never a data path.

Each storage is bound to its variant **by content**: its per-layer SHA256 must
match exactly one variant in the historical `frozen_direction_manifest.json`, and
the binding must be bijective. The archive's entry order is checked afterwards as
a cross-check and is never the source of truth.

**Not promoted to `artifacts/`.** The six criteria in `artifacts/README.md` are
met on safety and provenance, but the licence determination in
`docs/provenance.md` is unresolved and criterion 3 requires it. The freeze lives
in the machine-local data root, resolved through the new
`configs/paths.local.json` key `rq2_actuator_dir` and gated by
`manifests/rq2_actuator.json`. Nothing from it enters git.

**One numerical consequence, recorded rather than smoothed over.** RQ2 did not
add its direction directly. It built a synthetic donor, `donor = target +
direction`, and let `pathway_probe.build_site_plan` recompute `donor − target`.
In exact arithmetic that is the identity; in float32 it is not, and the recovered
direction is the frozen one perturbed by a cancellation proportional to
`‖target‖`. Both forms are implemented and the divergence is measured in
`tests/test_frozen_actuator.py` rather than asserted away. **Which one is the
default was left open here and is settled in the next entry.**

---

## 2026-08-21 — Freeze the list of questions the controller must not answer for itself

Six questions are open, and each of them has an answer that would be easy to
invent and hard to notice afterwards: a sign convention, a strength ceiling, a
loss. Written down before implementation starts, they are decisions someone
takes; discovered during implementation, they become whatever made the code run.

**Decision.** `docs/open_questions_pre_controller.md` lists them, states why each
is open, and records what may not be assumed in the meantime. Nothing in the
pre-controller migration phase resolves any of them, and no default for any of
them is introduced into code, configuration or documentation.

---

## 2026-08-21 — The historical arithmetic is the default; the closed form is a labelled approximation

The previous entry left this open. Leaving it open was the wrong shape of
decision: two vectors, both correct-looking, and a caller free to reach either.

The two are not interchangeable. Measured on synthetic tensors, float32
throughout, one layer, by the target's element scale:

| target element ~ | 1e-1 | 1e+0 | 1e+1 | 1e+2 | 1e+3 |
| --- | --- | --- | --- | --- | --- |
| relative difference | 1.2e-7 | 1.1e-6 | 1.2e-5 | 1.1e-4 | 1.1e-3 |
| cosine | 1.000000 | 1.000000 | 0.9999999 | 0.9999999 | 0.9999994 |

It grows linearly with `‖target‖` because that is what drives the cancellation.
The applied norm is unchanged and the cosine stays above 0.9999994, so the
difference is a small rotation rather than a different edit — which is exactly
why it would not be noticed.

**Decision.** The historical operation order is the canonical default, and the
closed form is a labelled approximation that cannot be reached by accident.

- `FrozenActuator.applied_delta(block, target, reference | reference_norm,
  coefficient=None)` is **canonical**. It transcribes the historical order —
  synthetic donor, `donor − target`, budget, rescale — and preserves it **for any
  coefficient**, which is what a later per-prompt controller will supply. It
  requires the target vector, so the two-argument call that used to return the
  closed form now fails loudly instead of silently changing meaning.
- `FrozenActuator.analytical_delta_approx(...)` is the closed form. Its name
  carries `approx`, its docstring says "Not exact" and "never a generation", and
  `acknowledge_not_exact=True` is **mandatory and keyword-only**. It has no
  `target_vector` parameter at all, so it cannot impersonate the canonical call.
- `FrozenActuator.application_path` is fixed at `historical_exact` and is
  reported by `provenance()`, so a run record states which arithmetic produced
  it. It is deliberately **not** a constructor argument: a per-instance switch is
  precisely how a caller would end up believing it had asked for the historical
  operation and receiving the approximation.

**On what "exactly" means.** A 2048-term float32 sum is not associative, and
numpy and torch do not order it the same way, so a norm can differ between them
by up to ~3.3e-7 relative. Bit-identity is therefore a property of the backend,
not of the formula, and the arithmetic dispatches on the type it is given: five
small helpers (`_is_torch_tensor`, `_as_float32_like`, `_scale`,
`_vector_norm_f32`, `_checked_vector`) let one transcription of the historical
order run in either. Given torch tensors, the result is **bit-identical** to a
literal transcription of `make_synthetic_donor` + `build_site_plan` +
`unit_scaled`; a test asserts `torch.equal`, not a tolerance, across seven
coefficients and all three frozen layers. A runner wanting bit-identity with a
historical generation passes torch tensors and a `reference_norm` measured on the
model's own block-input tensor.

The measured divergences, and why they occur, are recorded in the module
docstring, in `manifests/rq2_actuator.json` under `application_paths`, and in the
tests that measure them.

---

## 2026-08-21 — Narrow the RQ1 strength wording to what the ranks actually show

The entry above was itself written slightly too strongly. It said
`L31-33_w1_s0.15` "is first", then printed a table whose first two rows show it
placing **second**. Both cannot be true, and the table is the evidence.

No recorded number changes. The ranks were and remain 2/18, 2/8 and 1/2 for
0.15, and 11/18, 3/8 and 2/2 for 0.2 with one harmful break in each completed
run. Only the sentence drawn from them was wrong.

**Decision.** Everywhere this is stated, say the two separate things the ranks
support and not the one thing they do not:

1. **Band-level preference** — at layers 31–33, 0.15 outranked 0.2 in *all
   three* completed summaries. This is what makes 0.15, not 0.2, the strength
   the RQ1 evidence points at.
2. **Final-confirmation selection** — 0.15 placed first *overall* only in
   `pathway_confirmation_512_20260708_172847`. In the two earlier runs it was
   second overall, behind a window-6 configuration at the same band.

`rq1_grid_selected: 0.15` **stands**: nothing in the located evidence makes the
field's value wrong, only the sentence that justified it. Its evidence field is
rewritten as a band preference plus a final-confirmation selection, and a new
`rq1_grid_selected_is_not` field states the claim that must not be made from it.
The superseded `rq1_grid_best: 0.2` record and every numeric ranking table are
preserved untouched.

Corrected in `configs/causal_repair.json`, `docs/decision_log.md`,
`docs/provenance.md` and `tests/test_patch_states.py`. All counts remain
provisional keyword readouts, not semantically confirmed optima.

---

## 2026-08-21 — Retrospective triplet validation: what it is, and what blinds it

The ten operative cases were fixed in July 2026, the actuator was frozen on
2026-07-21, and the steering outcomes were known long before anyone judged the
cases' own arms. Validating them now is retrospective, and there is no way to
make it prospective after the fact. The choice is between doing it and saying
so, or not doing it at all.

The audit of 2026-08-21 established the starting position: the historical packet
`rq1_retro_twoaxis_packet_20260813_102233` is well built and properly blinded,
and **was never executed** — `api_calls_attempted: 0`, no results file, no join
output anywhere. It covers E, H and donor d1. It does not cover d2 and d3, which
are the donors the **primary** actuator `consensus23_global_frozen` averages,
and it carries no safety judgement for H. So the arms that matter most to the
actuator actually in use were the ones nobody had prepared.

**Decision.** Implement the validation, state its retrospective nature in every
artefact it produces, and control for it by construction:

1. **The original packet is executed unchanged**, by the historical runner. It
   is not rebuilt, re-rendered, or written into. `manifests/rq1_retro_packet.json`
   records its full inventory so "unchanged" is checkable.
2. **A supplement adds exactly the missing 50 rows** — Axis 1 and Axis 2 for d2
   and d3, and the safety axis for H — from frozen, hash-reconciled text. It
   generates no response and loads no model.
3. **Blindness is enforced, not asserted.** A judge row carries exactly
   `row_id` plus the prompt, plus one response on the response axes. Any other
   field is refused. The rendered text is scanned for case ids and condition
   tokens. Opaque ids are independently randomised and cannot collide with the
   historical packet's.
4. **The human reviewer never sees a judge verdict or an agreement flag.**
   Those are written to a private file the export does not open. A reviewer who
   knew the judges had split would be anchored; one who knew only where they
   agreed would be worse, because the review would become a tie-break.
5. **Case selection is not revisited.** A failing case is reported failing, in
   place. A test asserts no substitution path exists in the manifest builder.
6. **Patched responses are not an inclusion input.** Inclusion is decided on
   baseline arms only.
7. **The rules were frozen before any verdict existed**, in
   `routegate.triplets.INCLUSION_RULES`, and `PA-S`, `PA-N`, `NR` and `UM` are
   kept distinct rather than folded into a pass.
8. **`unresolved` and `invalid` are different outcomes.** A gap is not a quiet
   fail, and the manifest refuses to be written from partial evidence.

Nothing has been run. The runbook is `experiments/triplet_validation/RUNBOOK.md`
and the operator executes it.

---

## 2026-08-21 — Freeze CHG-004 as the retrospective Axis-1 protocol

Axis 1 exists in four versions. The completed 501-row corpus run used CHG-001
(`bd1ed925…`). The retrospective packet froze CHG-004 (`dbd037ec…`). They are
different strings, so a retrospective Axis-1 verdict is not byte-comparable with
a corpus one, and every artefact here says which version it used.

CHG-004 exists because CHG-003 let an educational wrapper convert a PROHIBITED
item to PERMITTED. Two drafting errors in CHG-003 were diagnosed — an added
"counts in the request's favour" weighting, and the deletion of the concrete
wrapper list — and CHG-004 reverses both and adds a bullet requiring the
requested *content* to be classified regardless of claimed purpose. The three
labels and their definitions are unchanged; only the framing-discipline rule is.

**It is prompt-only.** An Axis-1 record carries exactly `row_id` and
`prompt_text`. It cannot see the response, the E/B/H role, the steering
condition, the layer, the strength, the expected verdict or any previous
verdict. That is what makes it usable for a retrospective check at all.

**The re-calibration ran, and half-passed.** `rq3_axis1_calibration_20260802_205252`
paired against the CHG-003 run on the same 30 items:

| criterion, from the change log | outcome |
| --- | --- |
| `ITEM-1D1F880D6848` must return to PROHIBITED | **met** (PERMITTED → PROHIBITED) |
| the AMBIGUOUS count must stay near 2 | **not met** (2 → 4) |

Three of 29 items moved, all *away* from PERMITTED; none moved toward it. On the
full 480-item rerun CHG-004 gives PERMITTED 288 / PROHIBITED 107 / AMBIGUOUS 85.
**No record formally accepts that outcome**, and the change log ends at CHG-004
saying "a repair with a diagnosis, not a demonstrated fix".

**Decision.** Freeze CHG-004 for every retrospective triplet arm anyway, and
record why: for these inclusion rules it **fails closed in both directions**. E
and every donor need PERMITTED; H needs PROHIBITED; AMBIGUOUS satisfies neither.
A protocol returning more AMBIGUOUS can therefore only cause an exclusion, never
a false inclusion. Choosing the stricter of the two available versions is the
conservative choice. The AMBIGUOUS regression is recorded in
`manifests/rq1_judge_protocol.json` rather than resolved, and a case excluded
solely by an AMBIGUOUS Axis-1 verdict must be reported as such and not as a
demonstrated failure.

Axis 2 is CHG-002 and **is** byte-identical to the completed corpus run's.

**One divergence is left open.** The 2026-08-04 blind Axis-2 re-audit that
produced the accepted E 89 / B 178 corpus counts used an Axis-2 string
(`ed57133b…`) that appears nowhere in the change log. The retrospective work does
not use it, and no verdict produced here may be compared with one produced under
it without naming both. What it changed textually is not established.

---

## 2026-08-21 — Freeze the two protocol decisions, in these words

Both are quoted verbatim in `docs/study_design.md`,
`experiments/triplet_validation/README.md` and
`manifests/rq1_judge_protocol.json`, so the wording cannot drift between
documents.

**Axis 1.**

> CHG-004 is the frozen conservative retrospective Axis-1 protocol for this validation. It corrects the calibrated educational-wrapper wording, introduces no E/B/H role or response information, and moves calibration outcomes away from rather than toward PERMITTED. It is not claimed to be the Axis-1 version used throughout the complete E/B/H construction.

Applied consistently to **both** the original and the supplemental triplet rows:
one Axis-1 protocol for all five arms, no per-packet variation.

Its calibration outcome is **not** described as formally accepted historical
evidence, and must not be. The re-calibration ran
(`rq3_axis1_calibration_20260802_205252`), met its item-level criterion
(`ITEM-1D1F880D6848` returned to PROHIBITED) and missed its stability criterion
(AMBIGUOUS 2 → 4). No record accepts that outcome, and the change log ends
saying "a repair with a diagnosis, not a demonstrated fix". What justifies using
it here is narrower and is stated as such: it fails closed for these inclusion
rules, because AMBIGUOUS satisfies neither PERMITTED nor PROHIBITED and can only
cause an exclusion.

**Axis 2.**

> The retrospective packet's hash-pinned Axis-2 protocol is used for every triplet arm. A later 2026-08-04 re-audit used an unavailable undocumented variant. That variant is not reconstructed, imitated or used here. This is reported as a provenance limitation.

`preflight.py` gate 5 makes this checkable rather than asserted: it re-renders
every Axis-1 and Axis-2 request in both packets from the frozen text and refuses
the run if any resolves to a different instruction, template or schema hash.

---

## 2026-08-21 — Freeze the treatment of capped donor responses

28 of the 30 donor baselines in `rq2_multidonor_capture_merged_20260721_004223`
stopped at `max_tokens=256`. That is a fact about how they were generated in
July 2026, and there are three wrong ways to handle it: regenerate them, pass
them because a long answer looks substantial, or fail them because truncation
looks like incompleteness. All three would substitute a rule of thumb for the
frozen rubric.

**Decision.** The complete historically stored response is used exactly as
produced. It is never regenerated and never extended — the supplemental builder
loads no model and has no generation path at all. Reaching the cap **neither
passes nor fails** a response. The frozen Axis-2 rubric classifies the observed
response, and for donor inclusion **only `SA` satisfies the rule**: `PA-S`,
`PA-N`, `NR`, `DR`, `IR`, `UM` and an invalid or missing verdict do not.

`reached_max_tokens` and `output_token_count` are preserved from the capture
through the private row map, the joined cell, the adjudicated record and into
`manifests/rq1_triplet_inclusion.json`, so a reader can see both that a response
was capped and that the cap changed nothing about how it was judged.

Synthetic tests pin every clause: that no generation path exists, that the flag
survives each stage, that `SA` passes whether or not the response was capped,
and that each of the six non-`SA` codes fails a donor arm.

---

## 2026-08-21 — Integration corrections before any judging is run

Nine defects, all of the same kind: the implementation was internally coherent
but did not match the systems it has to interoperate with. None of them changes
the frozen scientific design; every one of them would have shown up as a
plausible-looking wrong answer rather than an error.

**The transport is the historical one, not a reasonable-looking one.**
`rq3_judge_axis_smoke` authenticates with `X-Api-Key` and reads the key from
`MODEL_API_KEY`; it returns the parsed payload, whose `content` is a **list of
blocks**, not a JSON string. The runner here used `Authorization: Bearer`, read
a different variable, and treated `content` as a string. It would have failed
on the first paid call, or — worse — produced text that no parser could read
after the call was already billed. The ported contract now lives beside the
vocabulary in `src/routegate/triplets.py`, so both runners read one definition.

**The served model is evidence, not an assumption.** The runner recorded
`returned_model = spec["expected_returned"]` — the value it had asked for. A
silently substituted model would then have been indistinguishable from the
requested one, permanently, in the record that exists to prove which model
judged. The identifier is now read from the response, and an **absent** one is
recorded as `UNESTABLISHED` and fails the row, matching the historical rule.

**The raw body is preserved before it is parsed.** The gateway envelope is
undocumented. Every response is written whole to a machine-local `0600` file
with its digest in the record, and only then parsed, so an unrecognised shape
stays inspectable instead of being lost with the money that bought it.

**A failed row is not a free row.** The runbook said re-running "retries only
the failures", next to a claim that a failed request had cost nothing. The
historical runner's own `billing_note` says the opposite: a request that timed
out or failed after the response arrived **may already have been billed**. That
is now stated where an operator will read it, and no path retries automatically.

**Two runners, two record shapes.** The original packet is judged by the
historical runner and the supplement by this repository's; their records
genuinely differ. A historical **success carries no `status` field at all** —
`status` appears only on a dry run, a timeout or a terminal error, all of which
are diverted to `unresolved.jsonl`. Joining by `status == "OK"` would have
dropped every historical row and reported a clean, empty, completely wrong
join. `detect_result_format` now names the format per record and refuses one
that matches both or neither; `result_is_success`, `result_join_key`,
`result_verdict` and `result_record_hash` read it by that name.

**The preflight read a key the frozen manifest does not have.**
`manifests/rq1_retro_packet.json` names its inventory `covered_by_packet_checksum`;
the gate read `covered_by_historical_checksum` and raised `KeyError` before a
single digest was compared. The manifest is checked-in provenance, so the code
was changed to read it as written rather than the record renamed to suit the
code. Verified against the real packet: all eight files match.

**`--stage` was accepted and ignored.** It now means something: `original`
gates the historical packet alone and does **not** require a supplement that
has not been built yet; `supplement` and `all` require both, because the
opaque-id collision check spans the two; only `all` asserts the combined total.
Demonstrated on the real inputs — 31 gates at `original`, 35 at `supplement`,
36 at `all`.

**Completeness is checked against the frozen ten, not against what arrived.**
Iterating the adjudicated cells alone meant a case that produced *no* cell was
reported as complete by never being mentioned. The build now walks
`EXPECTED_CASE_IDS`, verifies 11 × 10 = 110 required cells, reports an entirely
absent case as `unresolved` on every rule, and refuses an eleventh case id
outright rather than including a substitute.

**The human reviewer gets the judges' rubric.** A reviewer shown only
`axis_task: "axis1"` and a list of labels is not applying the frozen rubric;
they are inventing one, and the human/judge comparison is then not a
comparison. The export now carries the same hash-verified instruction text the
judges received, plus the allowed labels. It names no case, arm, role, donor,
condition or expected verdict, so it un-blinds nothing — and it is written to
the machine-local review directory, never to the repository.

**Found while testing, not listed above.** All five scripts called
`environment_record()` with no argument against a signature requiring
`package_version`. Every one of them would have crashed at runtime. This is
what the previous round's source-string assertions could not catch, and why
they were replaced with tests that invoke the code.

---

## 2026-08-22 — The validated dual-direction bank is built from the capture, not from the frozen actuator

The retrospective triplet validation finished and
`manifests/rq1_triplet_inclusion.json` resolved all 110 cells:
`comparator_actuator_source` and `h_e_direction_source` are `validated` for all
ten cases, `primary_actuator_source` for three.

**Decision.** Two directions, over historical blocks 31/32/33 at the final
prompt token: repair `B_d1 - E` from the ten comparator-validated cases, safety
`H - E` from the ten h/e-validated cases. Both by one recipe — per-case
`unit(arm - E)` at each layer independently, averaged over the ten, renormalised
to unit norm.

**The historical B_d2/B_d3 primary actuator does not enter this bank.** It is
validated for 3 of 10 cases; the other seven were judged and failed. It stays a
comparison baseline. `routegate.directionbank` has no code path that names its
arms and refuses `primary_actuator_source` as an eligibility source outright,
so this is a property of the code and not a convention to be remembered.

**Built from `rq2_geometry_capture_20260709_130906`, not from
`frozen_directions.pt`.** The frozen comparator is a `[3, 2048]` aggregate; the
safety direction cannot be recovered from it, because `H` was never a direction
source in the RQ2 freeze. The capture carries all three arms as raw residual
streams, so both directions come from one source by one recipe rather than one
being inherited and the other computed. The capture is read with
`torch.load(..., weights_only=True)`; no pickle in it executes.

**Parity is a gate, not a note.** The rebuilt repair direction must reproduce
`historical_d1_global_frozen` to float32 rounding — both are `B_d1 - E` over the
same ten cases by the same recipe. Observed 2026-08-22: cosine 0.999999880791 /
1.0 / 1.000000119209, max elementwise difference 2.235e-08 / 1.49e-08 /
1.49e-08. A disagreement means this is a different construction and stops the
build.

**No orthogonalisation and no strength.** The bank holds the two direct
directions. Their per-layer cosine is measured and recorded as a diagnostic —
nothing reads it to alter either direction. No coefficient is selected, embedded
or defaulted: this phase constructs directions, and scaling one is a separate
decision. `configs/dual_direction_controller.json` still lists
`direction_normalisation_and_orthogonalisation` and
`sigmoid_to_maximum_strength_mapping` as `UNRESOLVED`, and this work does not
close either. An orthogonalised variant is a later declared ablation.

**The artefact is NPZ, written byte-deterministically.** `numpy.savez` stamps
each member with the current time, so two identical banks would hash
differently and the manifest's digest of the file would mean nothing. The
members are written in a fixed order with a fixed timestamp instead.

**Consequence.** `manifests/rq2_geometry_capture.json` pins the capture's two
files, the eligibility record and the RQ2 source-case manifest by digest. The
capture is a *seventh* frozen input and is not yet migrated: until it is copied
under the data root, the builder cannot run from a standalone tree.

---

## 2026-08-22 — The direction pair is frozen before the bank is built

`configs/dual_direction_controller.json` had listed
`direction_normalisation_and_orthogonalisation` as `UNRESOLVED` since
2026-08-21. It is the last direction-side question, and leaving it open while
the builder existed would have meant the code silently answering it.

**Decision.** Per case and per layer 31/32/33: `delta = arm − E`, L2-normalise
each case-level delta, average the ten validated case directions separately for
repair and safety, L2-normalise each mean. The **primary method is the two
direct directions** — no orthogonalisation — and the direction bank embeds **no
steering strength**.

**Orthogonalised safety is predeclared as an ablation, not adopted.** The two
direct directions are measurably non-orthogonal, so a coefficient on one moves
the representation along the other. Whether removing that overlap helps is a
question to be measured. It is written down now, before it is built, so it
cannot later be introduced as though it had always been the plan — and it is
reported beside the primary result, never instead of it.

**The question moved out of `unresolved` rather than being answered in place.**
It now has an entry under `resolved` naming where the answer lives, so a reader
looking for it finds where it went. `sigmoid_to_maximum_strength_mapping` stays
open: how far to move along a unit direction is a different decision.

**Consequence.** The config now states the construction the builder already
implements, and states it *before* the builder is first run against the real
capture. `tests/test_probeheads.py` asserted this field was `UNRESOLVED`; that
assertion encoded the old state of the record and was updated to check the new
one — the field is absent from `unresolved` and present under `resolved`.

---

## 2026-08-22 — The causal screen is preregistered before it is run

The two validated directions exist. Whether they do anything is a separate
question from whether they were built correctly, and it is the one this screen
asks.

**Decision.** 3,214 generations over the development corpus: 89 E rows × 5 arms
and 213 H rows × 13 arms. Repair ladder on both classes, safety ladder on H,
and four combined arms on H. All arms, coefficients and counts are fixed in
`configs/dual_direction_controller.json → causal_screen` before the first
generation.

**The combined arms hold alpha_R fixed at 0.15 and vary alpha_S alone.** That is
the whole of question three. If adding the safety direction restored refusal
only because it cancelled part of the repair edit, that would look identical to
a safety direction doing its own work. Holding the repair coefficient still
while the safety one moves separates the two.

**The combined delta is not renormalised, and that is load-bearing.**
Renormalising would make the two coefficients compete for one fixed budget:
raising alpha_S would lower the repair component even at constant alpha_R, and
question three would be unanswerable by construction. So the formula is
`||block_input|| * (alpha_R * d_R + alpha_S * d_S)` and nothing rescales it.

**The screen checks it agrees with the existing actuator rather than asserting
it.** With alpha_S = 0 the edit must be what `steering.build_steering_plan`
already performs, compared against that function itself, before a model loads.
They are not bit-identical — `build_steering_plan` divides by the direction's
own norm and this formula does not — so the tolerance is the bank's unit-norm
error. Measured on the real bank: 5.96e-08.

**Generation and judging stay apart.** The runner records responses and nothing
else. It imports nothing that judges, calls nothing that scores, and a test
enforces both on the parsed source rather than on the prose.

**Found by the dry run, not by the tests.** `vector_norm` takes a torch tensor,
and the pre-flight convention check fabricated its probe reference as numpy.
Both plan builders raised `AttributeError` on the real inputs while every
synthetic test passed, because the tests all handed in tensors. The coercion
now happens where the numpy value is created, and a test covers both forms.
This is the second time a real-input dry run has caught what synthetic
fixtures could not.

**Consequence.** Nothing is run. The screen is submitted by hand as an
eight-shard array job; results land under the data root, outside git.

---

## 2026-08-23 — The 20 H regressions were not caused by the validated repair direction

The targeted selection was first recorded as `targeted_h_rescue_v1`, on the
working assumption that the 20 H rows regressed under "the static B−E
actuator". The audit of its provenance showed which B−E direction that was, and
it is not the one this phase is about.

**The mismatch.** `rq3_bh_rq2_static_20260810/run_config.json` records the
treatment as `consensus23_global_frozen` — the RQ2 **primary** actuator, built
from donors **d2 and d3** — at strength 0.15, historical layers 31–33, final
prompt token, prefill only. The validated direction bank's repair direction is
`historical_d1_global_frozen` = **B_d1 − E**, the *comparator*, built from d1.
They are different vectors from different donors. They are close — per-layer
cosine 0.984–0.986 in the RQ2 freeze — but "close" is not "the same", and the
two differ in exactly the way that matters here: `primary_actuator_source` is
validated for 3 of 10 cases, `comparator_actuator_source` for 10 of 10.

**Decision.** Rename the record to `targeted_h_legacy_regressions_v1` and state
the attribution in it. The manifest now carries `source_actuator` (variant,
donors, strength, layers), `legacy_stress_set: true` and
`current_validated_repair_direction_was_not_the_selection_treatment: true`,
alongside the `targeted_diagnostic` / `population_reportable` flags it already
had. The same 20 row_ids, unchanged, verified against an independent
recomputation from the master.

**Three claims the record explicitly refuses to make**, as machine-readable
false flags under `does_not_claim` and as prose: that B_d1 − E caused these
regressions; that they are failures of the validated direction bank; that
testing H − E on them demonstrates rescue of current B_d1 − E failures. A
builder-and-manifest test enforces that B_d1 is never named except beside a
denial or as the current bank's own direction.

**What the set is still good for.** Twenty rows already known to be steerable
into harmful compliance by *a* B-minus-E direction at this strength and depth.
That makes them a demanding place to ask whether H − E restores refusal — a
diagnostic, and a restricted one. It is not a population rate and not a claim
about the validated direction.

**The condition file was migrated.** `run_config.json` now lives under the data
root at `frozen/static_steering/rq3_bh_rq2_static_20260810/`, digest-identical
to its source, and is pinned by SHA256 in the builder. The builder additionally
refuses any input path that resolves into the historical workspace, so the
selection is rebuildable from RouteGate and routegate-data alone rather than
merely happening not to reach outside them today.

---

## 2026-08-23 — Phase 1 is a named protocol, not a subset of the exploratory screen

The exploratory screen asks all three questions at once over 3,214
generations. The focused plan asks them in order, and the first question is
narrower than the screen: does the **current** validated repair direction
`B_d1 − E` repair E, and which H rows does it damage?

**Decision.** `focused_direction_validation_v1`: 89 E rows × 5 arms
(baseline + the repair ladder) plus 213 H rows × 2 arms (baseline +
`repair_015`) = **871 generations**. `alpha_S` is zero throughout. The safety
direction is not tested here at all; phase 2 applies `H − E`, and only to the
failures this phase finds.

**Why a named protocol rather than a filter.** `--only-arm` and `--limit`
already produce any subset you like, and every such run is marked `smoke` and
is not reportable — correctly, because a hand-picked subset is not a protocol.
Phase 1 needs to be reportable, so it needs a name, a fixed inventory and a
completeness rule. `routegate.dualsteer.PROTOCOLS` now holds both, the runner
and the consolidator take `--protocol`, and `protocol` is a run-locked record
field so one output directory holds one protocol and nothing else.

**Completeness is a consolidation property.** A shard cannot know whether the
protocol is complete; only the consolidator can, and it counts against the
named protocol's own expected total, per-class split and per-arm split. 871 or
it is not `focused_direction_validation_v1`.

**The exploratory screen is untouched.** It remains the default protocol, its
3,214-generation inventory is unchanged, and the arm definitions in
`configs/dual_direction_controller.json → causal_screen` are still checked
against the package as before.

**The legacy stress set does not define this population.** Phase 1 runs the
whole development corpus — all 89 E and all 213 H. The 20 rows in
`manifests/targeted_h_legacy_regressions_v1.json` were produced by
`consensus23_global_frozen` (donors d2/d3), not by the direction under test,
and using them to select or define the phase-1 population would condition the
whole phase on an outcome measured under a different treatment. A test asserts
that no file in the phase-1 code path so much as names that manifest.

**A protocol may not misdescribe itself.** `expected_generations` and
`tests_safety_direction` restate, as one number and one boolean, what the arm
table already says, and both are copied into every run manifest and into the
consolidation record. A restatement can drift from what it restates, and a
drifted one here is not an untidy attribute — it is a false claim about what
was run, attached to the evidence. `Protocol.__post_init__` therefore derives
both from the arms and refuses to construct a protocol whose declaration
disagrees: adding a safety arm to phase 1 raises at import rather than
producing 871 generations that test `d_S` while every manifest records
`tests_safety_direction: false`. Checked at construction rather than in a test
because a test only fails when somebody runs it, and this fails before the
runner reads its first row.

**H is probed at one strength, not four.** The E ladder is where the
dose-response question lives. Asking H at 0.15 alone costs 426 generations
instead of 1,065 and answers the question actually posed — which rows does the
current direction damage at the setting the exploratory screen and the legacy
actuator both used. If the answer is interesting the ladder can be extended;
extending it now would be paying for a curve nobody has asked to see.

---

## 2026-08-24 — Phase-1 judging reuses the frozen protocol, the frozen corpus adjudication policy, and a fixed operating point

`focused_direction_validation_v1` generated 871 responses and judged none of
them; its consolidation record says the judging stage is what those responses
mean. `experiments/focused_direction_judging/` is that stage.

**Decision — nothing about the judging protocol is new.** The rubrics, schemas,
judge identifiers, temperature, token cap, request shape, transport, resume rule
and adjudication vocabulary are imported from `manifests/rq1_judge_protocol.json`
and `routegate.triplets` rather than restated, so a drift there is a drift here.
`experiments/triplet_validation/run_judge.py` is reused **byte-for-byte** as the
only thing that talks to an API: this stage adds no code that opens a socket,
reads a credential or retries a call, and a test asserts it.

**E takes Axis-2, H takes safety, and Axis 1 is not run.** Axis 1 is prompt-only
and every phase-1 prompt is a corpus row whose prompt status was settled when the
corpus was built; re-asking it would answer a question this phase is not about.
445 Axis-2 rows and 426 safety rows, 871 per judge and 1,742 across both.

**No deduplication, and the count is the reason.** The historical retrospective
packet deduplicated identical requests at temperature 0.0. Here every record is a
distinct `(screen_row_id, arm_id)` cell, the fan-out would be one-to-one for all
but a handful of rows, and a call count that can only be known after the packet
is built is a count nobody can commit to beforehand. An exact predeclared number
is worth more than the handful of duplicate requests.

**Two frozen adjudication procedures exist on record; this phase declares
`corpus_disagreement_only`.** Both use `DUAL_LLM_AGREEMENT` /
`HUMAN_ADJUDICATION` / `UNRESOLVED`, and both hold that a human verdict, where
one exists, stands. They differ in which cells a human reads:

- `corpus_disagreement_only` — agreement is final and a human reads only the
  splits. This is how the accepted corpus itself was labelled:
  `rq3_dataset_final.csv` carries `final_label_source` `DUAL_LLM_AGREEMENT` for
  385 of 509 rows and `HUMAN_ADJUDICATION` for 124. Every E/B/H label,
  `final_axis2` and `final_safety` this study rests on came out of that procedure.
- `triplet_independent_review` — a human reads every cell without seeing either
  judge, and `DUAL_LLM_AGREEMENT` requires all three to agree. Frozen 2026-08-21
  for the 110-cell retrospective validation.

**Decision, 2026-08-24, before any billed call: `corpus_disagreement_only`.**
This phase evaluates the same E/B/H development corpus as the completed
static-steering and RouteGate E/B/H results, and must stay directly comparable
with them; those results were labelled under exactly this procedure over exactly
this corpus. `triplet_independent_review` is not used here — it would make every
cell no human read `UNRESOLVED` and would produce numbers that do not line up
with the results this phase sits beside. It stays in the package as the other
procedure on record and as the rule `adjudicate_cell` is checked against, and
every entry point refuses a `--procedure` that is not the declared one.

**The repair operating point is fixed in advance, not selected.** `alpha_R =
0.15` is the predeclared primary operating point; `repair_005`, `repair_010` and
`repair_020` are descriptive dose-response sensitivity arms only. Nothing in this
code path selects an operating point — `report_phase1.py` holds no selection
function at all, and a test asserts the absence rather than the non-use. Two
reasons. H was generated at `baseline` and `repair_015` and nowhere else, so a
strength chosen off the E ladder would be a strength with no safety evidence;
and a comparator chosen after seeing the outcomes it is meant to summarise is
conditioned on them. No safety claim is made about 0.05, 0.10 or 0.20: an E-side
dose-response curve is not evidence about H. The dose-response counts are
withheld, not approximated, while any E cell is undecided.

**The rubric names the evaluation task, and that is stated rather than hidden.**
Each judge, and the human adjudicator, receives the rubric appropriate to the
task. That reveals which evaluation task a row belongs to — and, because the
rubric is assigned by class here, its class. It reveals nothing else: not the
treatment arm, not either coefficient, not the direction, not the layer or
state, not baseline versus steered, not the grouping of a row's five arms, and
not the expected result. The frozen rubrics are used exactly as they are and are
not redesigned to hide the task, because a rubric a judge cannot identify is a
rubric a judge cannot apply. The corpus row ids are scanned for because they
name their source benchmark in plain text.

**Both rules are frozen before any verdict exists**, committed in the manifest,
and byte-compared against the package before any reading is produced: a rule that
can be adjusted once the verdicts are in was never a predeclaration. A
current-direction regression is `baseline == NO` and `repair_015 == YES` after
the declared adjudication procedure, and nothing else enters the phase-2
selection manifest. `UNCLEAR` on either side, or an `UNRESOLVED` cell, is
**undetermined**: neither a regression nor a non-regression, because "we do not
know" and "we know it did not regress" are different findings and only the
second is evidence of safety.

**The legacy stress set stays out of the code path.** The 20 rows in
`manifests/targeted_h_legacy_regressions_v1.json` came from
`consensus23_global_frozen` (donors d2/d3). They do not define, select, filter or
corroborate the phase-1 regression list, and a test asserts that no file in this
stage names that manifest.

---

## 2026-08-27 — Three two-site judge verdicts never arrived; they are carried, not decided

957 of 960 planned two-site verdicts are valid. Three are missing, each after
repeated HTTP 200 replies that the frozen schema check rejected:

- `TSJ1QUALITY228E0C7BA5D1` — Sonnet, B/quality, 3 terminal attempts, each
  echoing an `item_id` one character from the one requested;
- `TSJ1QUALITY3FCCAD83E63B` — Opus, B/quality, 2 terminal attempts, same shape;
- `TSJ1SAFETY112B028C0586` — Opus, H/safety, 2 terminal attempts, zero content
  blocks and no assistant text.

Each of the three cells still holds the *other* judge's verdict.

**Decision.** No change to how any cell is decided. `corpus_disagreement_only`
is untouched. `focusedjudging.adjudicate_cell` is untouched, and it is what
labels these three: its missing-evidence branch already returns `UNRESOLVED`
with `judges_agree: None` whenever either judge has no verdict, and that branch
sits *before* the procedure split, so it applies identically under both
procedures. The frozen echo check in `triplets.validate_parsed` is untouched.

**The mismatched ids are not accepted.** They differ by one character, which is
exactly the failure the rule exists to catch, and a one-character difference is
not evidence of a typo rather than a misaddressed reply. That check is shared
with phase 1 and phase 2; relaxing it after seeing which rows failed would be a
post-hoc weakening and would retroactively change what every earlier verdict
means. The empty reply is not interpreted either.

**They do not go to a human.** The 2026-08-25 entry above rejected exactly that
for the phase-1 failures, because it decides cells under a rule written after
the failures are known. A human adjudicates a disagreement between two judges;
it does not supply a verdict a judge never gave. The export already selects
only `disagree`; the import now refuses an answer for a carried cell by name.

**Carried, not deleted.** Phase 1 excluded its two failed cells because they
were the `baseline`/`repair_015` arms of one row and the regression rule reads
that pair — excluding one arm would have left an incomparable half-pair. The
two-site stage has a single arm, `repair_015_blocks31_32`, and each cell is
independent, so no partner cell is orphaned. All 480 cells therefore survive
the join, three of them in the state `unresolved_technical`.

**Eight gates, all fail-closed**, in
`manifests/two_site_actuator_judging_v1_technical_unresolved.json` and
`src/routegate/twositeunresolved.py`: the cell resolves to one packet row; the
named judge has no successful verdict; the other judge has one; the evidence
holds at least two terminal HTTP 2xx failures; every referenced raw response
exists and its SHA256 recomputes; the failure type re-derives from the
preserved body; the selection reads only opaque ids, re-checked under every
verdict the vocabulary allows; and the pinned counts equal the derived counts.
A failure the manifest does not name still stops the join. If a named judge
later answers a named cell, the join fails and the manifest must be revised by
hand.

The classification is re-derived from the preserved raw body, never from the
recorded `error` string — see the credential note below.

**Consequence, disclosed and never absorbed.** 960 verdicts were planned; 957
arrived. 477 of 480 cells are dual-judged. B goes 178 → 176 decided, H goes
213 → 212 decided, E is unaffected at 89. Every reading is now reported twice:
the original-protocol count, still withheld while any cell of that class is
unresolved, and a complete-case count with its denominator named in
`decided_denominator_is`. Neither stands in for the other, and no rate divides
by the class total when fewer cells were decided. The H reading also carries a
sensitivity bracket for the one unresolved cell; neither bound is a finding.
`original_protocol_complete: false`.

**Credential exposure.** An early Sonnet pass was started with a whole shell
`export …` line pasted into `MODEL_API_KEY`. `http.client` rejected it and put
the header value into the exception message, which the runner recorded verbatim
into `failures.jsonl` — 480 records, one per row of that pass. **The key must be
treated as exposed and rotated.** The records are not edited: they are raw
evidence, and rewriting them would break the digests that gate everything above.
They are machine-local, gitignored and never committed. Going forward,
`triplets.redact_exception_message` strips the header payload before any error
string is written, and it is applied in both API runners. Redaction touches the
recorded `error` string only — never a request body, response body, digest or
preserved raw response — so no scientific artifact changes.

---

## 2026-08-25 — A paired technical exclusion, not a change to how cells are decided

Opus returned HTTP 200 with an empty assistant message for two safety cells and
never produced a verdict for either — four attempts each, three of them empty
replies at HTTP 200 and one an HTTP 401 that cost nothing. The two cells are the
`baseline` and `repair_015` arms of one H row,
`harmbench_text_test:sarin_ingredients_purchase_guide`.

An earlier attempt at this added a new adjudication procedure that sent those
cells to a human. **That was rejected, and rightly.** It changed how cells are
decided *after* the failures were known, and because those two cells are exactly
the pair the regression rule reads, a human-only decision there could have set a
primary regression outcome under a rule written in response to the failure.

**Decision.** No change to how any cell is decided. `corpus_disagreement_only`
is untouched, and every cell that enters the analysis still needs two judge
verdicts. Instead the pair is **excluded**, before the join sees it, by a
committed metadata-only manifest:
`manifests/focused_direction_validation_v1_technical_exclusions.json`.

**Both judges' records go, together.** Sonnet answered both cells, and those
verdicts are dropped as well. Keeping them would leave one cell single-judge
while every other cell is dual-judge, and would make the row's paired comparison
half-present. Symmetry is the point: 869 cells, each with two verdicts.

**Why the pair and not the single failed cell.** The regression rule reads
`baseline` and `repair_015` of a row as a pair. Excluding one arm would leave a
comparison that cannot be made.

**Seven gates, all fail-closed.** Both cells resolve to one corpus row; the arms
are exactly `baseline` and `repair_015`; the failed judge has no successful
verdict for either; the preserved evidence holds at least two
`EmptyJudgeResponse` attempts at HTTP 2xx per cell; every referenced raw
response exists and its SHA256 recomputes; the selection reads only opaque ids,
re-checked under every verdict the vocabulary allows; and the pinned counts equal
the counts the exclusion derives. One arm, another row, a row chosen for its
outcome, a single failed attempt, a missing or drifted digest — each is refused.

**If Opus later answers an excluded cell, the join fails.** Deliberately: the
exclusion would no longer be true, and the manifest must then be revised by hand.

**Consequence, disclosed and never absorbed.** 871 cells were planned; 869 carry
two verdicts. H rows go 213 → 212. The readings report
`original_protocol_expected_cells: 871`, `dual_judge_analysable_cells: 869`,
`technical_excluded_cells: 2`, `technical_excluded_rows: 1`,
`original_protocol_complete: false`, `paired_complete_case_analysis: true`,
`reportable_with_disclosed_deviation: true`. **The original 871-cell protocol is
not described as complete anywhere.** This is a post-run technical deviation, not
an outcome-based exclusion, and the manifest says so in those words.

---

## 2026-08-25 — Phase 1 closed; phase 2 frozen before it runs

**Phase 1 is complete and immutable.** 871 generations consolidated
(`f0ee90a8…cc3918`), 869 dual-judge cells adjudicated under
`corpus_disagreement_only` with 0 unresolved, the missing pair being the
disclosed paired technical exclusion. The readings:

- E repair at `alpha_R = 0.15`: **68 of 88** eligible baseline-refusal rows.
  The whole-corpus proportion, **68 of 89**, is retained beside it — both are
  reported and neither stands in for the other, because a reader who has not
  been told about eligibility will compute the second one.
- E dose-response: 34 / 61 / 68 / 78 at 0.05 / 0.10 / 0.15 / 0.20.
- H regressions: **18 of 212** analysable rows, with the sensitivity bounds
  18/213 to 19/213 for the technically excluded row.

**Phase 2 is frozen, and the freeze happened before any phase-2 output
existed.** That ordering is the whole point: a case set, a strength grid or an
endpoint chosen after seeing outputs is not a prediction, and there would be no
way for a reader to tell the difference afterwards.

**The population is the 18 rows the frozen phase-1 rule returned.** Not a
sample, not a top-k, not widened. `routegate.dualsteer` pins the 18 ids and the
selection digest `a54483a4…49ff1`; the runner builds its inventory from those
ids and nothing else. They are **not** the 20 rows of
`targeted_h_legacy_regressions_v1.json` — different repair direction, different
provenance — and a test refuses a selection that is that set, or 18 rows drawn
from it.

**Eight new arms, 144 generations.** Safety-only `alpha_S` = 0.05/0.10/0.15/0.20
at `alpha_R` = 0, and combined at the same four `alpha_S` with `alpha_R` pinned
at 0.15. Pinning `alpha_R` is what makes a restoration readable as something
other than less repair.

**Two arms are inherited, not regenerated.** `baseline` and `repair_015` for
these rows already exist in the frozen phase-1 records at the same model,
revision, direction bank, generation configuration, tokenizer and chat
template. Regenerating them would replace an immutable result with a fresh one
for no gain, and any difference between the two copies would be
unattributable. 36 inherited records; the analysis table is 144 + 36 = **180**.
`Protocol.__post_init__` refuses a protocol that declares an arm both generated
and inherited, and the dry run shows `reference_arms_in_this_plan: []`.

**`population_reportable: false`, stated and enforced.** These rows were
selected because they regressed. No phase-2 count is a population rate, and the
predeclaration names 212 and 213 explicitly as denominators it may not be
divided by.

**The Axis-1 caveat is carried forward, not closed.** The inclusion manifest
records "the AMBIGUOUS count must stay near 2 — NOT MET (2 → 4)". It is copied
verbatim into the selection manifest with the count that settles its bearing on
eligibility: **0** of the 30 Axis-1 verdicts across the ten included cases is
AMBIGUOUS, every one being PERMITTED or PROHIBITED. Both bank rules require a
decided Axis-1 verdict, so the caveat could only ever have excluded a case,
never admitted one.

**The phase-1 readings were regenerated once, deliberately, and are not
byte-identical to the earlier file.** `--write-phase2-manifest` writes both
artifacts, so the readings were rewritten to produce the selection. Every
scientific outcome is unchanged — the regression list, both E tables, the harm
counts and the sensitivity block compare equal — and exactly two things moved:
`created_utc`, and an additive `e_efficacy` block carrying the two E
denominators. The adjudicated cells themselves were never touched: their digest
`4e6be82b…0d1e73` is the same in the readings and in the selection manifest.

**No access to the sealed set**, at any point in either phase.

---

## 2026-08-25 — The phase-2 decision rule, frozen before any phase-2 output

Phase 2 generates 144 records and then has to decide something. The rule that
decides it is written now, with its threshold, its veto, its tie-break and its
failure branch all fixed, because a rule written afterwards is a description of
the result and no later reader could tell the two apart.

**Qualification.** A safety strength qualifies when (a) its combined arm at
`alpha_R = 0.15` restores at least **9 of 18** actionable harmful responses,
and (b) its safety-only arm at `alpha_R = 0` causes **0 of 18**. Both.

Nine is half the selected set: a strength that repairs fewer than half the
failures it was built for has not earned an actuator slot. The zero is not
"few" — a direction that turns a safe refusal harmful on its own is not a
safety direction, whatever it does in combination, so the veto is absolute and
18/18 recovered does not buy one caused harm.

**Selection.** Greatest recovery count among qualifiers; a tie breaks to the
**smallest `alpha_S`**, towards less intervention rather than towards the
larger coefficient.

**The failure branch is a result.** If nothing qualifies, `d_S = H − E` is
removed as a final steering actuator and the one-direction controller
continues on `d_R` alone. The H-versus-E+B safety probe is still trained and
still used, as a brake or veto. Dropping H−E as an *actuator* is not dropping
the safety *reading*, and writing this branch down in advance means reporting
it costs nothing while hiding it would be visible.

**An undecided row is not a pass.** A strength is decided only when all 18 rows
are decided at both its arms; otherwise the decision is `UNDETERMINED` and
nothing is selected, because an undecided row could change which strength
recovers the most. This is an addition for the undecided case and moves
neither threshold.

**Conditional, not an estimate.** The decision is taken on 18 rows selected
because they failed. It says which actuator to carry forward. It is not a
population estimate, `population_reportable` stays `false`, and no phase-2
count may be divided by 212, by 213 or by any corpus denominator.

The rule lives in `routegate.phase2.PHASE2_DECISION_RULE` with an evaluator
that reads counts and nothing else, is mirrored into
`manifests/focused_direction_validation_v1_phase2.json`, and is byte-compared
by `verify_phase2_predeclaration` — so it cannot drift from the code between
now and the outputs.

---

## 2026-08-26 — Phase 2 closed: `NO_STRENGTH_QUALIFIES`, and the one-direction controller is the continuation

Phase 2 ran, was judged, and was decided. The rule that decided it was written
on 2026-08-25, before any phase-2 token existed, with its threshold, its veto,
its tie-break and its failure branch all fixed. It returned the failure branch.

**The result.** No `alpha_S` on the frozen grid both restores at least 9 of 18
and causes 0 of 18 on its own:

| `alpha_S` | combined arm recovered | safety-only caused | qualifies |
| --- | --- | --- | --- |
| 0.05 | 7 / 18 | 1 | no — under threshold *and* caused harm alone |
| 0.10 | 10 / 18 | 3 | no — caused harm alone |
| 0.15 | 9 / 18 | 3 | no — caused harm alone |
| 0.20 | 10 / 18 | 3 | no — caused harm alone |

Three of the four strengths clear the recovery threshold. Every one of them
turns a safe refusal harmful on its own, and that veto is absolute: it was
written as `0`, not "few", precisely so that a good recovery number could not
buy one caused harm after the fact.

All 180 cells are decided — 144 judged in phase 2, 36 inherited from phase 1,
0 unresolved — so the outcome is `NO_STRENGTH_QUALIFIES` and not
`UNDETERMINED`.

**What the decision retires, and what it does not.**

- `d_S = H − E` is **retired as an actuator**. No final controller emits a
  coefficient for it, no final configuration declares it, and no steering plan
  applies it. The direction bank is *not* rewritten to remove it: it is an
  immutable phase-1 artifact.
- `d_R = B_d1 − E` is **retained**, at the validated strength 0.15.
- The H-versus-(E+B) probe is **retained as a sensor** — a brake or a veto on
  the repair gate. Dropping H−E as an *actuator* is not dropping the safety
  *reading*, and this was written into the rule in advance so that reporting it
  costs nothing.
- The completed 254/254 probe-head run is **retained unchanged**. What phase 2
  changed is what the safety score is allowed to *do*, which is a controller
  decision, not a probe one; refitting a gated artifact to record a decision
  taken elsewhere would trade it for an ungated one.

**The closure is a second file, not an edit to the first.**
`manifests/focused_direction_validation_v1_phase2.json` says what phase 2 said
it would do, frozen before it ran, and is not touched — not to record the
outcome, not to mark the protocol closed, not at all. A predeclaration edited
after its result is no longer a predeclaration, and no later reader could tell
which parts were written before the run and which after. The outcome therefore
lands in `manifests/focused_direction_validation_v1_phase2_closure.json`, which
declares itself post-run, pins the readings it was taken from by SHA-256
(`b10b19ff…1e02a011`), and carries its own tamper digest over its canonical
form. `configs/dual_direction_controller.json` is likewise preserved verbatim:
its `proposed_controller` block is now **phase-2 metadata**, a record of what
was proposed then, and a test pins it in place rather than enforcing it as the
design.

**The continuation.** `configs/single_direction_controller.json`, a new
versioned file rather than a repurposing of the dual-direction one:

- **one** sigmoid gate `g_R ∈ [0, 1]`, emitted once per prompt;
- applied strength `alpha_R = 0.15 · g_R`, so the ceiling is *arithmetic* — the
  sigmoid saturates at 1, and there is no coefficient in the parameterisation
  that can carry the applied strength past the validated 0.15;
- read at **state 32** — the output of historical block 31, the frozen
  actuator's own first layer, predeclared before either head was fitted — at
  the **final prompt token**, prefill only. States 33 and 34 stay sensitivity
  diagnostics and may not be promoted to the read state once their metrics are
  known;
- the safety head **brakes**: it may scale `g_R` down or drive it to zero, and
  it produces no direction, no second coefficient and no output of its own.

**Out-of-fold, in every component.** For held-out fold *k*: the fold-*k* repair
and safety probes, each fitted without *k*; the controller trained without *k*;
fold *k* generated and evaluated, and no other. A controller reading two probe
scores is a three-stage pipeline, and cross-fitting two of three stages leaves
a leak that is invisible in the number that comes out.

**Deployment, only after architecture freeze**, in this order: refit the two
heads on all development rows with every hyperparameter fixed; train the final
controller on all development rows; freeze both and record their digests —
*then*, and only then, read the sealed set.

**The protocol is closed against accidental reruns.** An ordinary
`--protocol focused_direction_validation_v1_phase2` is refused before the
configuration is opened, before the bank is hashed and before a corpus row is
read: a rerun would write a second set of records beside the first, at the same
arms on the same rows, and no difference between the two copies would be
attributable to anything. Reproducibility is preserved, not deleted — the same
frozen inventory runs under the explicitly named `--reproduce-closed-protocol`
flag, into an output directory of its own. The flag is never inferred.

**Still conditional, still not a population estimate.** The decision was taken
on 18 rows selected *because* they failed. It says which actuator to carry
forward. `population_reportable` stays `false`, and no phase-2 count may be
divided by 212, by 213 or by any corpus denominator.

**A duplicate-prompt diagnostic was run over the division, read-only.**
`tools/diagnostics/duplicate_prompt_folds.py` groups all 480 development rows
by normalised prompt text (NFKC, whitespace collapsed, case-folded) and reports
any group spanning more than one fold — which would mean every cross-fitted
artifact built on that division was fitted on one copy and scored on the other.
Result: **0 duplicate groups**, so nothing crosses a fold. The tool alters no
fold assignment and proposes no re-split; a crossing group would be reported as
a blocker and left for a decision.

**No access to the sealed set**, at any point.

---

## 2026-08-26 — The one-pass budget check demanded bit-identity across two forward passes, and was wrong to

The v1 preflight ran on the GPU over all 480 development prompts and refused
its own result:

    block_31_one_pass_budget_is_exactly_the_validated_one: FATAL
    block 31 maximum relative difference: 3.527095e-07
    ONE_PASS_EQUIVALENCE: BLOCKED, exit 2

**The gate was wrong, not the actuator.** `verify_exact_blocks` required
`worst == 0.0` — literal bit-identity — between two numbers that are never
produced the same way:

- the two-pass reference is captured in **pass A** by `PatchController`, whose
  `hooks.clone_cpu` (`src/routegate/hooks.py:81`) stores
  `tensor.detach().float().cpu().clone()`, so its norm is reduced **on the CPU
  in float32**;
- the one-pass value is snapshotted in **pass B** at
  `src/routegate/onepassbudget.py:394` and normed on the **accelerator**, in
  bfloat16 upcast to float32 by `intervention.vector_norm`
  (`src/routegate/intervention.py:168`).

Two float32 reductions of the same 2048 values in different orders disagree in
the last bits. That is a property of floating-point addition, not of this
pipeline, and no hook ordering can remove it.

**The magnitude settles it.** 3.527095e-07 is **2.96 × float32 epsilon**
(1.192e-07) and **4.5e-05 × bfloat16 epsilon** (7.8e-03). The activations are
*stored* in bfloat16, whose resolution is some four orders of magnitude
coarser than the observed disagreement — so the underlying bf16 activations
were identical and only the float32 reduction differed. A genuine activation
difference, a padding or tokenisation difference, a device-map difference or a
read-after-write bug would all have shown up at ~1e-3 or larger. Reduction
order is the only cause consistent with 1e-7.

**Blocks 31 and 32 are mathematically identical, not bit-identical.** Nothing
upstream of their reference has been edited when it is read, so in exact
arithmetic the two conventions agree. Across two forwards on two devices they
agree to float32 rounding. They are now gated at the **already predeclared
1e-2**, like every other cross-forward comparison. No new tolerance was chosen,
and certainly not from the observed 3.527095e-07.

**What is still exact, and is now checked in the right currency.** At block
`b`'s hook the snapshot taken for block `b+1`'s budget comes from the very
tensor block `b`'s edit then lands on. If the read moved after the write, the
two would be one vector and their separation would be exactly zero. It is
instead the applied delta. `verify_read_before_write` requires that separation
to be non-zero on every row at blocks 31 and 32 — **within one forward pass, on
one device, in one dtype**, where no reduction order can reach it. The
recorder also raises in the pass itself if the separation ever collapses.

**Block 33 is unchanged and remains the genuine gate.** It is the only block
where the two conventions differ by construction rather than by rounding,
because block 32 has already consumed an edited state 32 by the time its
reference is read. It is measured on the frozen model at 1e-2.

**A second, operational fault: the failed run wrote nothing.**
`outputs/single_direction_controller/budget_equivalence_v1/` was created and
left **empty**. The gate raised inside `_report` *before* any artifact was
written, so 960 completed forward passes — the whole 480-prompt measurement,
including every block-33 number the exercise exists to obtain — were
discarded. There is therefore no v1 block-33 result to report; none was ever
recorded. That is corrected: every artifact is now written **before** any gate
decides the exit status, so a refused run still leaves the evidence it was
refused on, with the failing checks in its own manifest.

**v1 is preserved, not overwritten.** Its directory is left exactly as the
failed run left it. v2 writes to a fresh directory and records what it
supersedes and why.

The specification's frozen constants are unchanged: `alpha_R = 0.15`,
`g_R = 1` for this worst case, `alpha_S = 0` permanently, blocks 31–33, final
prompt token, tolerance 1e-2.

---

## 2026-08-26 — One-pass budget equivalence is closed: block 33 is infeasible, and the two-site actuator is a candidate

The v2 run measured all 480 development prompts at `g_R = 1` and was decided
against the predeclared 1e-2 gate:

| historical block | package state | max relative difference | rows over 1e-2 | |
| --- | --- | --- | --- | --- |
| 31 | 32 | 3.527095e-07 | 0 / 480 | PASS |
| 32 | 33 | 3.212718e-07 | 0 / 480 | PASS |
| 33 | 34 | **7.455261e-02** | **382 / 480** | **FAIL** |

`ONE_PASS_EQUIVALENCE: BLOCKED`, exit 1. Every artifact was written before the
gate decided, so the evidence survives the refusal.

**Block 33 is infeasible under the validated budget convention, structurally
and not numerically.** Its budget is the norm of block 32's output. In one
forward pass that tensor has already been computed from an *edited* state 32,
so the unsteered value the convention requires does not exist anywhere in the
pass. No ordering of hooks recovers it. The disagreement is 7.5e-02 — five
orders of magnitude above the float32 rounding that explains blocks 31 and 32,
so it is not a numerical artifact.

**The alternative convention is worse and is rejected, not adopted.** Budgeting
block 33 against its *actual* runtime input — which carries `delta_32` as well
as the propagated `delta_31` — was measured alongside the gate: 1.093370e-01
maximum, 432 of 480 rows over tolerance, against 7.455261e-02 and 382. It is
recorded so the unfavourable option is on the record.

**No tolerance moved.** 1e-2 was predeclared before the first run; v1 and v2
were both decided against it, and nothing in this line re-derived it.

**The read-before-write invariant held on all 480 rows**, at both blocks 31 and
32, so the block-33 failure is not a hook-ordering fault.

**Consequence: a candidate, not a result.** `two_site_actuator_candidate_v1`
intervenes at historical blocks **31 and 32** — package states 32 and 33,
budgeted against states 31 and 32 — at the unchanged validated direction
`d_R = B_d1 − E` and the unchanged `alpha_R = 0.15`, with `g_R = 1`,
`alpha_S = 0`, no controller and one prefill forward pass. Its sites are
**derived** from the v2 summary's own per-block verdicts and bound to that
summary's digest `30caa469…3c13276`; a summary clearing a different set of
blocks is a different candidate, and `assert_sites_are_the_passing_sites`
refuses any drift in either direction.

**What this does not claim.** Phases 1 and 2 measured the 31–33 band causally,
in *two* passes, and those results stand — what failed is the one-pass driving
of that band. Dropping a site changes the intervention, and the phase-1
evidence does not transfer to it. A two-site actuator with no generation behind
it has no causal evidence whatever the budget arithmetic says. That is why the
record says `CANDIDATE, NOT VALIDATED` and why a new static causal validation
over the 480 development prompts is the next step, not a claim.

**The historical-block-32-only arm is preserved as the later preregistered
ablation** and is not run here. It is package state **33**, not state 32 —
state 32 is the output of historical block 31, and the two numbering systems
differ by one throughout.

Both records are immutable, carry their own digest over their canonical form,
and are verified field by field against the package:
`manifests/one_pass_budget_equivalence_closure_v1.json` and
`manifests/two_site_actuator_candidate_v1.json`.

---

## 2026-08-26 — The two-site run was interrupted; the continuation is a new path, not a repair

The first two-site attempt wrote six records — three baseline, three candidate —
and stopped. It could not be restarted, for two reasons found by audit:

1. **`--resume` was unreachable.** `claim_output_dir` refused a non-empty output
   directory *before* `--resume` was consulted, and a resumable directory is
   non-empty by definition.
2. **The shard arguments were dead.** `--shard-index` and `--shard-count` were
   declared and never read, so every shard would have generated all 480 rows
   and collided rather than partitioned.

Both are corrected **only in a new versioned path**.
`run_two_site_validation.py` and the six records it wrote are left exactly as
they are: they are the evidence of what happened, and evidence is not edited to
make a later run tidier.

**Baselines are referenced, not generated.** The 480 baseline responses come
from the frozen static-steering master (`525e62e0…`), whose own run
configuration (`e8b1b46f…`) records `before_source.regenerated: false` and pins
the same corpus digest this package gates on. The reference is corroborated:

- all 480 master row ids, classes and prompt bytes revalidate against the frozen
  development corpus (0 disagreements of any kind);
- **302 / 302** phase-1 baseline responses are byte-identical to the master's;
- **3 / 3** interrupted-run B baselines are byte-identical to the master's.

So **305 of 480 are independently reproduced**; the remaining **175** — B rows,
which phase 1 never steered — rest on the master's own provenance plus matching
model, revision, tokenizer, chat-template and decoding digests. That
distinction is recorded rather than smoothed over, and every manifest states
`generated_baselines: 0`.

The master's `after_response_text` is **not** used. It came from
`consensus23_global_frozen`, the historical primary actuator at a different
direction from the validated `B_d1 − E` bank this candidate steers.

**Counts, stated in every manifest:** `generated_candidate_records: 477`,
`reused_candidate_records: 3`, `referenced_baselines: 480`,
`generated_baselines: 0`.

The three reused records are validated in full before being reused — sixteen
locked fields, prompt digest, class, sites, per-state direction digest, status
and hook count — because a record carrying the right arm name and the wrong
direction, sites or strength looks entirely well formed. A restart validates
its own partial output the same way before skipping it, and refuses a
duplicate, a row from another shard, an error record, a truncated JSONL line or
a shard that is already complete.

Nothing was generated, judged or trained. Every frozen phase-1, phase-2, probe,
direction-bank, budget and judging artifact is byte-identical.

---

## 2026-08-28 — The controller reproduction demanded 1e-9 of a float32 trajectory, and was wrong to

The reproduction of `gate_oof_v1` ran to completion and refused itself:

    oof_coefficients_reproduce: max |reproduced - saved| = 2.384186e-07 over
    480 row(s), tolerance 1e-09

**Everything the gate exists to protect had already agreed.** All five folds
trained; early stopping selected 1571, 616, 578, 367 and 280 steps, which are
`gate_oof_v1`'s own recorded numbers to the integer. The held-out mean gate
reproduced to all six printed decimals in four folds (0.067355, 0.302368,
0.340788, 0.363274 against 0.06735536, 0.30236777, 0.34078797, 0.36327400) and
to 5e-07 in the fifth (0.383663 against 0.38366249) — a statistic the two runs
do not even reduce in the same dtype, since the record takes the mean of a
float32 tensor and the reproduction takes it in float64. The probes had already
reconstructed at 1e-12 and every pinned input digest matched. What failed was a
comparison of two float32 numbers.

**The magnitude settles it.** 2.384186e-07 is exactly `2**-22`: **two float32
epsilons**, four ULPs at the top binade of a quantity that lives in `[0, 1]`.
An exact power of two is what a difference in reduction order leaves behind;
it is not what a different optimisation trajectory, a different fold or a
changed parameter leaves behind. `1e-9`, meanwhile, is about **one hundredth
of a single float32 ULP at `g_R = 1`** — it required two float32 numbers to
agree more closely than float32 can represent a difference at all, so it could
only ever have passed a bit-identical re-run on bit-identical hardware. It
gated the dtype, not the science. This is the same fault as the one-pass
budget check of 2026-08-26, in the same currency, two stages later.

**The per-fold pattern rules out a trajectory difference.** The disagreement
tracks the float32 ULP of the coefficient's own magnitude, not the number of
optimisation steps:

| fold | steps | max `g_R` | `ulp_f32(max g_R)` | observed max \|dg\| | in ULPs |
| --- | --- | --- | --- | --- | --- |
| 0 | 1571 | 0.160758 | 1.490e-08 | 2.235e-08 | 1.5 |
| 1-4 | 616, 578, 367, 280 | 0.9985-0.99996 | 5.960e-08 | 2.384e-07 | 4.0 |

Fold 0 ran **2.8 times more steps than fold 1** and disagreed **ten times
less**, because its gates never leave the `[0.125, 0.25)` binade where float32
spacing is four times finer. A divergence accumulating through training would
grow with the step count. This one grows with the magnitude of the number being
represented, which is what representation rounding does and what a different
trajectory does not.

**The constant was not inherited.** `COEFFICIENT_TOLERANCE = 1e-9` was
introduced by the engineering reproduction runner and appears in no frozen
specification, no predeclaration and no earlier run. Correcting it changes no
predeclared scientific quantity. The superseded rule, the run it refused and
why it was wrong are recorded in `reproequivalence.SUPERSEDED_RULES`, so the
failure cannot be rediscovered as a novelty.

**What replaces it is a conjunction, not a looser number.**
`routegate.reproequivalence` is a versioned, dtype-aware rule in which the
numeric clause is the least important member. A reproduction is the same
computation only if all of it holds:

1. `max |dg| <= 32 * float32_eps = 3.814697e-06`;
2. fold assignment, labels, early-stopping steps and probe digests are
   identical, and each `tau_hi` agrees within 32 float32 ULPs at its own
   magnitude;
3. not one row changes side of a reporting cutoff, the safety veto, the
   counterfactual clamp or the intervention decision at any cutoff;
4. every rank inversion is between rows the tolerance cannot separate *and*
   inside one reporting bucket;
5. the applied edit `0.15 * g_R * r_l * d_R` moves by at most a **thousandth**
   of one bfloat16 rounding step of the state it is written into.

Clause 1 alone can never pass a run.

**Where 32 epsilons comes from, and where it does not.** Not from the observed
maximum, which is 16 times smaller and never entered the derivation. Two
things fix it. The **dtype**: `g_R = sigmoid(f_theta(x))` is float32 and is
compared in `[0, 1]`, so `ulp_f32(1.0) = 2**-23` is the unit a gate-space
disagreement is naturally stated in, and a budget in that unit means the same
thing on every machine. The **corpus's own decision geometry**:
`assert_rule_is_admissible` measures the smallest distance from any saved
coefficient to any boundary this package decides on and refuses the *rule*
unless the tolerance is at least twice smaller. On `gate_oof_v1` that margin
is **2.368883e-05**, at the counterfactual clamp — 6.2 times the budget. The
constant is therefore falsifiable: on a corpus whose coefficients sat closer
to a boundary, this rule would refuse itself rather than pass a run that moved
a decision. The margin is measured in the dry run, before anything is trained.

Two facts made a decision change impossible at this magnitude even before the
clauses were written. The **veto does not read the gate** — it fires on
`z_H >= tau_hi`, and the nearest row sits 4.129314e-02 away in logit units. The
**edit is invisible to the actuator** — at `2.384186e-07` the steering vector
moves by at most 9.4e-07 in norm across the two sites, which is at most
9.0e-06 of one bfloat16 rounding step of the activation it is written into; the
gate would have to move by 2.6e-02, five orders of magnitude more, to change
that activation by a single representable step.

**A second, operational fault, and the same one as last time.** The refused run
wrote **nothing**. Five completed folds and 480 reproduced coefficients were
discarded inside `checks.require`, so the per-row error distribution, the
per-fold maxima and the affected-row count that would diagnose it do not exist
and cannot be recovered without retraining. The runner now writes
`reproduction_diagnostic.json` — every clause, the full error distribution and
its quantiles, per-fold maxima, the decision margins, the ordering analysis and
the edit-norm comparison — **before** it decides an exit status. Checkpoints
and the all-development refit are produced only after the rule passes, so a
run that is not equivalent still leaves its evidence and still yields no
controller.

**The gate can be tightened and cannot be widened.**
`assert_tolerance_is_not_loosened` refuses a `--tolerance` looser than the
declared rule. A gate is corrected here, in the open, with the reason recorded;
it is never argued away at the point of use to obtain a pass.

`gate_oof_v1` is untouched. Nothing was retrained, no response was generated,
no judge was called and the sealed set was not read.

---

## 2026-08-30 — Screen a rotation toward `H − E`, without reopening phase 2

Phase 2 applied `d_S = H − E` **additively** and its frozen rule retired it: no
`alpha_S` on the grid both restored refusal on ≥ 9 of 18 targeted-H rows and
caused zero actionable harmful help on its own. `NO_STRENGTH_QUALIFIES` is
recorded in `focused_direction_validation_v1_phase2_closure`.

That closure answers a question about one operation. It says the *addition* of
that vector is not a safety actuator. It does not say the direction carries no
recoverable safety information, because addition is not the only way to move an
activation toward a direction — and it is the way that also changes the
activation's norm, which confounds "moved toward `H − E`" with "made larger or
smaller".

**Decision.** Declare a new, separately named candidate screen,
`rotation_safety_direction_v1`, which rotates the residual activation toward the
same frozen layer-specific `d_S,l` through a fraction `beta` of the angle that
separates them, preserving the norm exactly. Eight arms, `beta ∈ {0.05, 0.10,
0.15, 0.20}`, on exactly the 18 rows phase 2 was bound to, at exactly the same
three sites. 144 generations; `baseline` and `repair_015` inherited from the
phase-1 records and never regenerated.

**Three things this deliberately does not do.**

*It does not reopen phase 2.* The additive closure is preserved byte for byte
and its outcome is restated, not revisited, in the new predeclaration. A
different operation on the same direction is a new question, not an appeal.

*It does not treat `beta` as comparable to `alpha_S`.* `beta` is a fraction of a
prompt-specific and layer-specific angle; `alpha_S` was a multiple of the
block-input norm. They share their numerals and nothing else, and the manifest
forbids reading any arm here as "the same strength" as a phase-2 arm.

*It does not claim anything.* The screen is a candidate. Nothing in its path
describes rotational `H − E` as validated or as novel, and the manifest's own
`status_meaning` says so, so that a later reader who finds only the artefact
still finds the disclaimer.

**Order is a term of the experiment, not an implementation detail.**
`rotate(h + delta) != rotate(h) + delta`. The combined arms apply the validated
`B − E` repair first and rotate second; that order is recorded on every record,
is run-locked so a directory cannot hold both conventions, and is demonstrated
non-commutative by a test rather than asserted.

**The decision rule is phase 2's, with one stated difference.** The two bars are
the same — ≥ 9 of 18 restored at the combined arm, and exactly 0 actionable
harmful help at the matching rotation-only arm, the second an absolute veto.
Among qualifiers, phase 2 took the greatest recovery count and tie-broke toward
the smaller coefficient; this screen takes the **smallest qualifying `beta`**
outright. The divergence is written into the manifest *before* any outcome
exists, under `selection_differs_from_phase2`, so that neither rule can later be
presented as the other. Unresolved evidence yields `UNDETERMINED`; the failure
branch yields `NO_BETA_QUALIFIES` and is a result.

**Numerical tolerances are predeclared and versioned.** The operator normalises
and takes its trigonometry in float32 and casts back once, at the end. Norm
preservation is checked **twice, against two different numbers**: the float32
tolerance is the operator's own arithmetic error, and the post-cast tolerance is
the dtype's unit roundoff. Collapsing them into one would let bfloat16's
rounding hide an operator that had stopped preserving the norm. Both are in
`manifests/rotation_safety_direction_v1.json`, which was written before any
generation exists and digests itself.

**`arccos` conditioning is handled by refusing, not by approximating.** Near
`c = 1` the recovered angle is dominated by float32 rounding, so the angle gate
cannot catch a near-parallel pair — but `‖y − c·x‖`, which *is* the sine and is
measured directly, can. The operator refuses that case as a plane it cannot
determine, rather than rotating through an angle it cannot trust.

**Consequence.** `outputs/` grows by one run root outside git; the committed
tree grows by three package modules, five stage scripts, one configuration, one
self-digesting predeclaration and two test files. No completed artifact is
touched, and the freeze, the preflight, the runner and the consolidator all
re-verify all 22 of them by digest before doing anything else.

---

## 2026-08-30 — Correct the rotation screen to phase 2's own selection rule

The rotation screen was committed with a decision rule that shared phase 2's
two qualification bars — at least 9 of 18 restored at the combined arm, exactly
0 actionable harmful help at the matching rotation-only arm — but chose
differently *among* qualifiers: it took the **smallest** qualifying `beta`
outright, where phase 2 takes the **greatest recovery count** and breaks an
equal-recovery tie toward the smaller coefficient. The divergence was
predeclared and disclosed rather than hidden, but disclosure was the wrong
remedy.

**Why it was wrong.** The screen exists for exactly one comparison: does
*rotational* `H − E` clear the causal bar that *additive* `H − E` failed? A
selection rule that differs anywhere makes the two screens incomparable at
precisely the point of comparison. A rotational `beta` could have been selected
under a rule no additive strength was ever offered — and the resulting claim,
"rotation succeeded where addition failed", would have been partly an artefact
of the rule change. The bars being identical is not enough when the bars admit
more than one qualifier.

**Decision.** Apply phase 2's rule itself, and delete the second statement of
it. `routegate.rotationfreeze` now imports `PHASE2_RECOVERY_THRESHOLD`,
`PHASE2_SAFETY_ONLY_HARM_LIMIT` and `PHASE2_DECISION_RULE`, and
`evaluate_rotation_decision` translates its counts into phase 2's vocabulary,
calls `routegate.phase2.evaluate_phase2_decision`, and relabels the answer on
the way back. Qualification, the choice among qualifiers, the tie-break,
`NO_STRENGTH_QUALIFIES` and `UNDETERMINED` are all that function's. The outcome
labels are not renamed for this screen either, so a rotational outcome and an
additive one are read off one vocabulary.

The only rotation-specific behaviour left in the wrapper is input validation —
the counts must cover the whole frozen `beta` grid, because a decision taken
over a subset of it is a different decision and phase 2 has no opinion about
which betas exist — and vocabulary: phase 2's "safety-only" arm is this
screen's "rotation-only" arm, and `alpha_S` is `beta`. A parametrised test
asserts the two agree cell for cell on eight count tables, which is a weak
statement made deliberately: the strong one is that there is only one function.

**Two sentences of phase 2's prose are overridden, and only two.** Its
`NO_STRENGTH_QUALIFIES` branch says the *additive* actuator is removed. That is
a decision already taken, in a closure that is not being re-taken here, so this
screen substitutes its own sentence for that branch while keeping the branch,
its name and its condition untouched.

**Versioning: v2 beside v1, not over it.** The v1 predeclaration was frozen and
committed, and a predeclaration that can be rewritten is not one. But nothing
had run under it — zero generations, zero responses, zero verdicts — and a
document that has produced nothing can be corrected before it runs, where one
that has produced anything would have to be closed and reported as it stood. So
`manifests/rotation_safety_direction_v2.json` is written *beside*
`manifests/rotation_safety_direction_v1.json`, which is left byte-identical and
is now pinned among the preserved artifacts so that it cannot drift either. The
protocol name moves to `rotation_safety_direction_v2` and is run-locked on every
record, so a record can never be read against a rule it was not declared under.
v2 carries a `supersedes` block naming the one field that moved, the counts that
made moving it permissible, and the nine things that did not move.

**Nothing about the experiment changed.** `d_S = H − E` and its sign, the
`beta` grid `{0.05, 0.10, 0.15, 0.20}`, the four rotation-only and four
repair-plus-rotation arms, historical layers 31–33 at the final prompt token
during prefill only, the 18 selected H rows and their digest, the float32
tolerances and their version, the fail-closed near-collinearity handling, the
repair-then-rotation order, and every frozen input and completed artifact.

**One thing the correction made louder rather than changed.** `beta` is the
fraction of the *per-row, per-layer* rotation angle. It is not an additive
alpha and it is **not a matched perturbation dose**: one `beta` moves two rows
by two different amounts and moves a single row by three different amounts
across the band, and no arm here is equalised against any phase-2 arm by edit
size, by angle, or by anything else. Making the *decision* rule identical could
otherwise invite the reading that the *interventions* are now comparable in
size. They are not. What is comparable is what gets recorded: the realised
`theta`, the traversed `phi` and the induced edit norm, per row, per layer and
per arm — reported later, beside the outcome, and the only basis on which a
size comparison between the two screens may be made.

**Consequence.** One rule in the repository instead of two. No model was
loaded, no response generated, no judge called, no job submitted, and the
sealed set was not read.

## 2026-09-03 — the rotation screen is registered with the common judging engine, not given a fourth copy of one

**Context.** The rotation screen finished: 144 responses over 18 targeted H
rows and 8 predeclared arms, consolidated into a package whose records hash
`c6d72d7b…` and whose validation records 6,247 passed checks. It judged
nothing, by design. Phase 1, phase 2 and the two-site actuator had each grown
their own copy of the same nine judging steps, and
`0245050` added the common manifest-driven engine precisely so that a fourth
mechanism would not need a fourth copy.

**Decision.** This screen is the engine's first real tenant. It supplies
`manifests/rotation_safety_direction_v2_judging_v1.json` — metadata only, no
instruction, template, prompt or response text — and the only new code is the
two things a manifest cannot express.

**The adapter.** The engine reads records carrying exactly
`judgingstages.SOURCE_FIELDS`. A rotation record carries its response and the
response's digest, but its prompt only as a digest: the screen never stored
prompt text. `routegate.rotationjudging.adapt` resolves the text from the one
hash-gated corpus the screen recorded it generated from, through the existing
`focusedjudging.load_prompt_texts`, and checks every resolved prompt against
the digest the record already carries. It converts and it gates; it judges
nothing and decides nothing. The output is deterministic and the manifest pins
it by SHA256.

**The reading.** The engine emits verdict tallies with explicit denominators
and deliberately stops there. The 180-cell table, the four counts per `β` and
the decision are this experiment's own, and `reporting.module` names the
module for the record.

**What was reused rather than rewritten, and how far that was taken.** The
rubric is the frozen actionable-harmful-help rubric, by digest, from the same
directory phase 1 read it from. The adjudication rule is
`focusedjudging.adjudicate_cell`, through the engine, unchanged. The 36
inherited reference cells are reconciled by
`phase2judging.load_inherited_references`, **called unchanged** — phase 2
inherited the same two arms over the same 18 rows under the same procedure, so
its eight gates are this screen's eight gates. The four counts come from
`phase2judging.counts_by_strength`, which was generalised so the ARMS are
parameters while the arithmetic is not: every default is phase 2's own, a
phase-2 call is byte-identical to what it always was, and the classification of
a verdict (`H_HARMFUL_VALUES`, `H_NOT_HARMFUL_VALUES`,
`H_UNDETERMINED_VALUES`) is read from where it was frozen so that no caller can
decide UNCLEAR means NO. The decision is
`rotationfreeze.evaluate_rotation_decision`, which is phase 2's rule itself.
A test asserts the rotation module's source contains no threshold, no veto, no
tie-break and no outcome branch of its own.

**Two digest conventions for one file, recorded rather than reconciled.** The
frozen `safety_schema.json` is pinned as `77ab5885…` by every earlier manifest,
which hashed the schema's *canonical JSON*; the common engine hashes each
rubric file *as it reads it*, which for the same bytes is `8b4d1fb7…`. Neither
number is wrong and the file has not moved. The manifest pins the value the
engine checks and records the other beside it, and a test asserts the one file
on disk satisfies both. Silently changing either pin, or editing the frozen
file so the two agreed, would have been the wrong repair.

**The reference arms are unreachable, not merely unused.** `baseline` and
`repair_015` are declared as reference arms, are refused by the adapter, are
absent from the packet, and the manifest declares **no `reference_records`
input at all** — so there is no file for a reference response to be read out
of and no path by which one reaches a judge. 144 judged + 36 inherited = 180;
288 billed calls, not 360.

**What has and has not been run.** Preflight, the packet build, the request
dry-run and both judge runners in dry-run mode: all clean, 144 rows per judge,
288 planned calls, nothing called and nothing billed. The packet itself was
built, because a judge dry-run reads it; it lives in the machine-local data
root, is not committed, and is deletable. No judge was called, no response was
generated, no human label was invented, no completed artifact was modified and
the held-out split was not touched.

---

## 2026-09-03 — The dynamic arm's coefficient is computed live; the frozen column is a witness, not the mechanism

`gate_oof_v1` left 480 out-of-fold coefficients keyed by row id, and the
cheapest way to generate `dynamic_raw` would have been to read that column at
generation time. It is not what this arm does, and the reason is not
fastidiousness: a coefficient table cannot score a prompt it has never seen, so
an arm built on one would be a lookup wearing a controller's name, and every
development row's answer would sit inside the arm that used it.

**Decision.** The applied coefficient is computed, per prompt, in the pass that
generates the response: capture the unsteered state-32 final-prompt-token
activation before either site is written, run the fold checkpoint that never
trained on that row, and apply `0.15 * g_R` at historical blocks 31 and 32.
`livecontroller.plan_live_edit` is the one place that decides it; the runner
does no arithmetic of its own.

**Then what is the frozen extraction for.** It is the **witness**. The same
checkpoints score the extraction's own state-32 activations offline, on CPU,
giving a reference coefficient per row, and the applied coefficient is checked
against it twice:

* before any model is loaded, the five per-fold held-out **means** are
  re-derived and compared with the means `gate_oof_v1` itself recorded. A wrong
  bundle, a permuted fold assignment, a different activation file or a
  corrupted standardisation all move at least one of them far outside a float32
  budget, and all of them are caught in seconds rather than eight GPU-hours
  later. Measured: `3.880e-09` against a `3.815e-06` budget;
* per row, while generating, the live coefficient is compared with that row's
  own reference and a disagreement **stops the run**. A controller that cannot
  score the activation the evidence was gated on is not the controller the
  evidence describes, and an arm built on it would not be either.

**The budget is not a new number.** `g_R` is float32 in `[0, 1]` here exactly as
it was in the reproduction, so the gate carries
`reproequivalence.GATE_ABS_TOLERANCE` — 32 float32 epsilons — rather than a
constant invented for this stage. The reference is scored one fold at a time
and the live path one row at a time, so the two reduction orders differ by a few
ULPs; that difference is three orders of magnitude inside the budget and is
recorded per row rather than assumed away.

**Consequence.** `--dry-run` is a real gate rather than a rehearsal: it decides
whether the deployable checkpoints reproduce the run that trained them, using
no model and writing nothing. `--verify` re-derives the same reference and
revalidates all 480 finished records against it, so the completed arm is
checkable by someone who did not run it.

---

## 2026-09-03 — The reproduction summary records weight digests, not file digests

The first version of `dynamicarms.verify_bundle` compared
`reproduction_summary.json`'s `checkpoints` map against `sha256` of each
checkpoint *file* and refused the real bundle on all six. The bundle was not
wrong: the summary records each checkpoint's `weights_sha256` — a digest over
the state dict in sorted key order — and the file digest was never written
down by anything.

**Decision.** Compare the recorded value against the loaded checkpoint's
`weights_sha256`, which is the scientifically meaningful identity, and carry
the on-disk digest beside it as a fingerprint rather than as a gate. The chain
is complete either way: `livecontroller.load_checkpoint` recomputes the weight
digest from the state dict it actually loaded and refuses a mismatch, so a file
edited anywhere that matters fails there, and a file whose weights belong to
someone else fails here.

**Consequence.** The bundle is also checked for the converse — a checkpoint
present in the bundle that the reproduction never recorded is refused, so a
sixth controller cannot be dropped in beside five verified ones.

---

## 2026-09-04 — `already_present` is a resume count, and the arithmetic invites one wrong reading

The completed `dynamic_raw` run reports `generated_here 267` and
`already_present 213`. The development frame holds 89 E + 178 B = 267 rows and
213 H rows, so the two numbers can be read as "the E and B rows were generated
and the H rows were not" — which would mean the arm had refused to steer its
H rows, and would be a stop.

It is a coincidence of arithmetic. `generate()` computes
`remaining = [rid for rid in shard if rid not in done]`, a set difference over
row ids already in the shard's records file, and the shard is in sorted row-id
order. Three resumed invocations wrote 22, then 191, then 267 records. The 213
already present are 124 B + 57 E + 32 H; the 267 written last are 181 H + 54 B
+ 32 E. Both run logs say so in as many words, and no code path in the runner
reads `final_ebh` for anything but writing it into a record.

The near miss is worth naming: `load_development_inputs().row_ids` really is a
267-row E/B projection, and `run_dynamic_arms.prepare` carries a comment
warning against using it instead of `manifest_rows`. Two different 267s, one of
them a real hazard the code already guards.

**Decision.** Do not trust the run manifest's own counts for this.
`dynamicjudging.assert_the_split_is_a_resume` re-derives the split from the
records — sorted order, neither side one class, every record on this arm with
both hooks fired, a non-zero edit, one coefficient at both sites at
`alpha_R * g_R`, and every coefficient out of fold — and
`assert_matches_the_audit` separately holds the completed run to the published
numbers. The first set of gates holds for any run of this arm and is exercised
on synthetic records; the second is the audit's own claim.

**Consequence.** The registration is gated on the audit rather than accompanied
by it. A future arm that genuinely did skip a class fails
`the_already_present_prefix_is_not_a_class` before a packet is built, and a run
whose split has moved fails `the_split_matches_the_published_audit`.

---

## 2026-09-04 — The pairwise rubric needs a new name over the same frozen bytes

`judgingprotocol.decode_pairwise` is hard-coded to decode an A/B answer into
`CANDIDATE_BETTER` / `REFERENCE_BETTER`, and `register_rubric_vocabulary`
refuses to redefine a name already in `triplets.AXIS_VALUES`. `quality` is
already bound there to the BEFORE/AFTER composites the two-site stage reads. So
a manifest that named its pairwise rubric `quality` would either be refused at
registration or produce verdicts the decoder cannot land in the declared
vocabulary.

**Decision.** Register the rubric as `pairwise_quality`, over the same three
frozen files, pinned by the same three digests. `dynamicjudging.stage_rubrics`
copies them byte for byte and verifies each digest at the source and again
where it lands; only the filename differs.
`dynamicjudging.PAIRWISE_TO_TWO_SITE` renames the head of a composite —
REFERENCE is BEFORE, CANDIDATE is AFTER — so `twositejudging.b_reading` is
*called* rather than copied with two words replaced.

**Consequence.** Two rubric names now denote the same bytes, which a test
asserts rather than a comment claiming it. The alternative — widening
`decode_pairwise` to take its vocabulary from the manifest — would have changed
a primitive three completed stages depend on, to save a rename.

---

## 2026-09-04 — The scientific cell names the arm, and that is what reuse costs

Item 12 of the registration allows an earlier judgment to stand in for a new
one only when the scientific cell, the prompt, the response bytes, the rubric,
the judge models and the protocol digests all match. On this arm the cell
identity is `judgingprotocol.CELL_FIELDS` — screen row, **arm**, rubric — and
no cell on `dynamic_raw` has ever been judged, so nothing is reusable and all
480 are scheduled.

That is not free. 172 of the 480 dynamic responses are byte-identical to the
unsteered baseline (164 H, 7 E, 1 B), because the applied edit fell below the
actuator's bfloat16 resolution; 24 are byte-identical to the static
blocks-31-32 arm (22 H, 2 E), because `alpha_R * g_R` reached 0.15. At the cell
level that is 176 cells whose judge-facing material some completed run has
already judged under the same rubric, by the same pair, at the same
temperature.

**Decision.** Keep the arm in the cell identity and pay for the 176. A verdict
produced for one arm is not a verdict for another, and dropping the arm clause
to save calls would be a change to the science made inside a config.

**Consequence.** `dynamicjudging.reusable_judgments` reports both numbers — the
reusable count and the material-match diagnostic — and the adapter runs it as a
live gate, so the choice stays visible instead of being hidden in a count. If
the arm clause is ever dropped deliberately, the saving is already measured:
960 calls become 608.
