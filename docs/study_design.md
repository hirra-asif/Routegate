# Study design

> **Historical design record (August 2026).** Written before the controller,
> gate and final evaluation were built. Status lines such as "Not started" and
> "the split has not been opened" describe that time: the dynamic gate and the
> held-out evaluation were later completed. For current status see the
> [experiment index](experiment_index.md); for results see the [README](../README.md#main-results).

## The problem

A safety-tuned language model can fail in two opposite directions. It can comply
with a request it should refuse, and it can refuse a request it should have
answered. The second failure — **over-refusal** — is the subject of this study.
It is not a minor annoyance: a model that refuses benign requests about
medicine, security or law is unusable for exactly the people who most need it,
and the usual remedy, weakening refusal behaviour, trades one failure for the
other.

The study asks whether over-refusal can be *repaired selectively*: corrected on
prompts that were wrongly refused while refusal is preserved on prompts that
should be refused. The two halves are one outcome, not two. A method that
repairs everything by weakening refusal has failed, and so has one that
preserves refusal by never acting.

## The model

Qwen3-30B-A3B, a mixture-of-experts model: 48 transformer blocks, hidden size
2048. Read with `attn_implementation="eager"`, `enable_thinking=False`, no system
prompt, batch size 1. The MoE architecture matters to the study — routing
decisions are a candidate signal that a dense model does not offer — but no
routing signal has been extracted yet.

## Components

The work is organised by what each part scientifically *is*, not by the order it
was done in.

### Causal repair

Establishes that the failure is manipulable at all, and **where**.

The earlier work showed that activation-level edits derived from a matched donor
prompt repair a refused generation, with control conditions — random directions,
wrong-donor directions — asking what the repair is attributable to. Its important
negative result: wrong-donor directions also repair, so the effect is not
narrowly localised and "matched donor" is not doing the work the name suggests.
That work is **pilot evidence**: ten hand-built triplets and a five-band grid
over a 48-block model. It does not locate the effect, and it does not choose a
layer for anything downstream.

**Every count in the RQ1 grid summaries is a provisional keyword readout**, and
the summaries say so themselves. Manual semantic audits exist beside them and
have not been verified against them here. So any RQ1 repair, selectivity or
harm-break figure quoted anywhere — including the negative result above — is
provisional until that verification is done, and none of it may be reported as a
semantic result. The 2026-08-21 entry in `docs/decision_log.md` records what the
grid rankings actually contain, including that they rank `L31-33_w1_s0.15` above
`L31-33_w1_s0.2` on the safety-first ordering the runs were designed around.

**The current stage** is the experiment that does: cross-fitted, H-orthogonalised
layer-wise causal activation *steering*, searching all 48 transformer blocks on
the frozen development corpus, with an H arm so that a repair achieved by
suppressing refusal in general can be told apart from one that is not.
**Implemented; started, and a long way from finished.** As of 2026-08-21 one scan
task exists — `outputs/causal_repair/orthogonalized_population_scan/scan/task_1`,
113 generations of the 2 416 that task requested, covering states 1–7 of 48 — so
"not yet run" is no longer accurate and "run" would be worse. No state has been
selected and no result is reportable. Exact matched-donor activation patching
follows as confirmation at the state(s) the completed scan selects. See
`experiments/causal_repair/README.md`.

This scan is an **ablation**, and specifically it is not the frozen actuator. Its
direction is a difference of class means over the 480-row corpus, H-orthogonalised
and cross-fitted; the frozen actuator's is a mean of per-pair B−E deltas over ten
matched triplets, with no H term. They operate at different depths and different
strengths and answer different questions. `docs/artifact_taxonomy.md` keeps the
five objects in this study apart, because four of them get called "the direction"
somewhere.

Two analysis decisions were frozen on 2026-08-07, before any intervention output
existed, and are pinned by the direction bank's hash:

*Baseline reproducibility.* The frozen target set stays at 89 E and 213 H; no row
is deleted, relabelled or re-folded on the strength of its fresh runtime
baseline. Results are reported both over the complete frozen set and, for refusal
transitions, over the **baseline-confirmed** subset — the E rows whose fresh
baseline still refuses unnecessarily and the H rows whose fresh baseline still
declines the harmful request. The counts, percentages and row IDs of every
non-reproducing row are reported explicitly; none is silently removed.

*Primary state selection.* A state is eligible only if it produces **zero** new
unsafe-compliance cases on baseline-confirmed H relative to baseline; among
eligible states, the primary actuator state is the one with the highest E repair
rate. Ties break toward smaller behavioural change on H, then toward the earlier
transformer block. If no state is eligible, none is force-selected and the E/H
trade-off is reported as a negative result. Selection is made on **generated
behaviour**, never on probe scores, previous RQ1 layer results, historical states
32–34, or visual inspection of plots.

### Frozen actuator

Removes the donor. A single direction is derived from the development set alone
and applied without a per-prompt donor, so the edit becomes a deployable
intervention rather than an oracle. **Complete and frozen**: one method was
selected and its definition fixed, so that everything downstream evaluates a
fixed actuator rather than searching over actuators.

The frozen method is **`consensus23_global_frozen` at strength 0.15**, at
decoder blocks 31/32/33 — states 32/33/34 — added at the final prompt token
during prefill, with a target-relative L2 budget read from each block's *input*
hidden state. It is a mean of per-pair B−E deltas: each benign donor's residual
minus its matched over-refused target's, unit-normalised per pair, averaged
across the two independent donors within a case and then across the ten matched
triplets, renormalising at each step. There is **no H term and no
orthogonalisation** in it.

Two corrections to what this section said before 2026-08-21, both confirmed
against the freeze itself:

- It gave the strength as **0.10**. That was the selection of 2026-07-20
  (`rq2_final_method_selection_20260720_141145`), which chose the single-donor
  fixed-mean direction. The multi-donor work superseded it the next day, and the
  freeze of 2026-07-21 15:00 fixed **0.15**. Every result that rests on the
  actuator — the unseen evaluation and both static E/B/H transfers — was run at
  0.15. "CPR-Min" survives only as the name of the scaling convention, which is
  unchanged.
- It let the number stand without saying which artefact it belonged to. **0.2 is
  not this actuator's strength.** It is the declared operating strength of the
  separate population scan, carried forward from the RQ1 pilot as a fixed value
  while the block axis is searched; it is not a measured optimum for either
  stage, and the RQ1 grid itself did not rank it first. `configs/causal_repair.json`
  carried the same misattribution and is corrected too — see the entries of
  2026-08-21 in `docs/decision_log.md`.

The artefact and its loader now live here: `manifests/rq2_actuator.json` gates
ten migrated files by SHA256, and `routegate.actuator` reads the direction
tensors without `pickle` or `torch.load`. The tensor archive is not committed;
it is resolved from the machine-local data root.

**Applying it is the historical arithmetic, not the closed form.** RQ2 routed its
direction through a synthetic donor, which in float32 is not the identity, so the
two differ by a rotation that grows with the target's magnitude — 1.2e-7 at small
scale to 1.1e-3 at large. `FrozenActuator.applied_delta` reproduces the historical
operation order and is the default; the closed form survives only as
`analytical_delta_approx`, which requires an explicit acknowledgement that it is
not exact and may never produce a generation.

### Triplet validation

Retrospective validation of the ten operative E/B/H cases that the frozen
actuator was derived from. **Implemented, nothing run.**

It is called retrospective because that is what it is: the steering outcomes it
validates the inputs of were produced in July 2026 and were already known when
the judging was prepared in August 2026. Three things make that defensible
rather than merely admitted.

*The judges are blind.* Each sees a prompt, or a prompt and one response, and an
opaque id. Not the case, not the arm, not the E/B/H role, not the donor number,
not the grouping, not the steering condition, not the layer, not the strength,
not the repair outcome, not the expected verdict, and not each other's answers.
The same holds for the human reviewer, who additionally never sees either
judge's verdict or whether they agreed — the review is an independent reading,
not a tie-break.

*Case selection is not revisited.* The ten cases were fixed in July 2026 and are
not replaced on the strength of their own validation. A case that fails is
reported as failing, in place. No code path in this stage can substitute one.

*A patched response is never an inclusion input.* Inclusion is decided on
baseline arms only, so a steering outcome cannot feed back into the decision
about whether its own inputs were valid.

The historical packet covers E, H and the original donor d1, and **was never
executed** — its own record shows zero API calls. It does not cover d2 and d3,
which are the donors the *primary* frozen actuator averages, and it carries no
safety judgement for H. A supplemental packet adds exactly those 50 rows from
frozen text, generating nothing. See `experiments/triplet_validation/`.


**The frozen protocol, in the words this study uses everywhere.**

> CHG-004 is the frozen conservative retrospective Axis-1 protocol for this validation. It corrects the calibrated educational-wrapper wording, introduces no E/B/H role or response information, and moves calibration outcomes away from rather than toward PERMITTED. It is not claimed to be the Axis-1 version used throughout the complete E/B/H construction.

Applied consistently to both the original and the supplemental triplet rows.
Its calibration outcome is **not** formally accepted historical evidence: the
re-calibration ran, met its item-level criterion and missed its
AMBIGUOUS-stability criterion, and no record accepts it. What justifies using it
here is only that it fails closed for these inclusion rules — AMBIGUOUS
satisfies neither PERMITTED nor PROHIBITED, so a stricter protocol can only
cause an exclusion, never a false inclusion.

> The retrospective packet's hash-pinned Axis-2 protocol is used for every triplet arm. A later 2026-08-04 re-audit used an unavailable undocumented variant. That variant is not reconstructed, imitated or used here. This is reported as a provenance limitation.

A preflight gate re-renders every Axis-1 and Axis-2 request in both packets and
refuses the run if any resolves to different protocol hashes.

**Capped donor responses.** 28 of the 30 donor baselines reached
`max_tokens=256`. The complete historically stored response is used exactly as
produced; it is never regenerated or extended, and the cap neither passes nor
fails it. The frozen Axis-2 rubric classifies the observed response, and only
`SA` satisfies donor inclusion. `reached_max_tokens` is preserved through to the
final evidence record.

Until every required judgement exists, the honest statement is that all ten
cases are **unresolved** on every arm, and `manifests/rq1_triplet_inclusion.json`
refuses to be written.

### Residual geometry

Can over-refusal be *detected* from the residual stream at the final prompt token
— before a single token is generated? A linear probe is fitted per hidden state
and scored out-of-fold. The pilot reported ROC-AUC 0.9526 at state 45; this
package reproduces that with scikit-learn and then corrects it (see *Corrections
owed*, below).

This is **decoding** evidence, and it is descriptive: it says where E and B are
linearly separable in the stored representation, not where an edit changes
behaviour. It does not select a state for the causal scan, and its probe
coefficient is not used as a repair direction. **State 45 — the pilot's headline
— receives no preference in the causal scan**, as a candidate, as a prior or as
a tie-break. The two analyses are independent, and whether they agree is a
finding rather than a check on either.

### MoE features

Do router logits and expert selections carry information about over-refusal
*beyond* the residual stream? The read-out is deliberately incremental — ΔAUC
against the residual-geometry baseline on the same rows and folds — never a
standalone router number. **Not started.**

### Dynamic gate

A prompt-level detector that decides, before generation, whether to apply the
frozen actuator. Dual by construction: an edit-benefit arm predicting whether a
prompt is an over-refusal worth repairing, and a harmful-veto arm blocking the
edit when the prompt should be refused. **Not started.**

### Intervention evaluation

The end-to-end test on the held-out benchmark split, with the joint outcome —
repair **and** preserved refusal, per case — as the primary endpoint. **Not
started**, and the split has not been opened.

Mapping to the thesis research questions, for cross-reference only: causal repair
is RQ1, the frozen actuator is RQ2, and residual geometry, MoE features, the
dynamic gate and intervention evaluation together are RQ3. The rest of this
repository uses the scientific names.

## The corpus

509 prompt–response pairs, adjudicated into three classes plus an exclusion:

| class | n | meaning |
| --- | --- | --- |
| **E** | 89 | over-refusal — a benign prompt that was refused |
| **B** | 178 | benign compliance — a benign prompt that was answered |
| **H** | 213 | correct refusal of a harmful prompt |
| excluded | 29 | partial or unclear; not analysable |

480 rows are analysis-eligible (E + B + H) and carry activations. The residual
geometry stage analyses the **E vs B** contrast only — 267 rows, positive
prevalence 0.3333, which is the PR-AUC chance level, not 0.5. H is present in the
corpus and deliberately not analysed at this stage.

## Design commitments

**The held-out split stays sealed.** 380 rows, committed to by count and
composition in `manifests/sealed_set_commitment.json` and by nothing else. It is
opened once, for intervention evaluation, as an explicit logged action outside
every development entry point.

**One accepted version of every input.** Enforced by SHA256, not by convention.
Superseded label revisions are refused by filename before a hash is computed,
because the superseded ones carry the more authoritative-sounding names.

**Preregistration before evaluation.** Thresholds, endpoints and analyses are
written down before the run that tests them. `configs/residual_geometry.json`
declares the corrected protocol and its entry point refuses to execute without
explicit approval, precisely so the protocol is reviewable before it produces a
number.

**Reproduction before extension.** The pilot's estimator, scaler, splitter and
metrics were hand-written. They are re-derived with scikit-learn, against the
pilot's own folds, before anything is built on top.

## Corrections owed to the residual-geometry pilot

Reproduction confirms the arithmetic; it does not repair the design. Four things
are wrong with the pilot as reported, and the corrected analysis is defined by
fixing them:

1. **The selected state is a selected maximum.** State 45 was chosen by
   maximising out-of-fold performance over all 49 states on the same rows the
   score is then reported from. The correction is nested selection: choose the
   state inside each outer training fold, score it on the outer held-out fold.
2. **The permutation null does not price the selection.** It was computed at the
   already-selected state, so it answers a narrower question than the one the
   headline implies. Each permutation must repeat the whole 49-state search and
   contribute its maximum. The pilot's 20 permutations also floor the p-value at
   1/21 ≈ 0.048, which is uncomfortably close to the 0.0476 it reported.
3. **Magnitude is unaccounted for.** An activation-norm control — one scalar per
   row, no direction at all — reached ROC-AUC 0.871 against the probe's 0.953.
   Most of the separability may be vector magnitude rather than geometry. The
   correction runs unit-L2-normalised and norm-residualised probes beside the raw
   one.
4. **One split is one draw.** A single five-fold assignment gives a point
   estimate with no sense of how much of the state ranking is split noise.
   Repeated stratified CV replaces it.

A fifth limitation cannot be corrected with the present corpus and is recorded
rather than fixed: no genuine prompt-family identifier exists, so folds are
label-stratified only and near-duplicate prompts may straddle a boundary. One
such family was found among the 267 E/B rows, and it does straddle.

A sixth is of the same kind. The natural surface-form control is prompt token
length, and no token count was saved by the extraction. Retokenising the prompts
here would produce a number that resembles the extraction's own without being
it, so the control is run on character and whitespace-token counts, named as
such, and the absence is checked by an input gate rather than assumed.

All four corrections are implemented in
`experiments/residual_geometry/run_analysis.py` and exercised end to end by its
smoke test. Two further things are reported that the pilot did not have: the
same probe run within each source family, with folds capped by the rarer class
and small analyses flagged unstable, and controls on the adjudicated
`topic_standardized` and `task_frame` columns beside `source_family`.
