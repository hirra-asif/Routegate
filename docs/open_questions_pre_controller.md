# Deliberately unresolved before the controller

> **Historical record (21 August 2026).** Questions as they stood before the
> controller was implemented. Their later resolution is recorded in the
> [decision log](decision_log.md).

Recorded 2026-08-21, at the close of the pre-controller migration and provenance
phase.

Six questions are open. Each has an answer that would be easy to invent and hard
to notice afterwards — a sign convention, a ceiling, a loss function. Written
down before implementation, they are decisions someone takes. Discovered during
implementation, they quietly become whatever made the code run, and the write-up
then describes a choice nobody made.

**Nothing in this phase resolves any of them.** No default for any of them exists
in code, configuration, manifest or prose in this repository, and none is to be
introduced until the question below it is answered on its own terms.

> **Two of the six have since been answered, and this file is not rewritten to
> hide that they were once open** (2026-08-26). Question 2 — *the construction
> and sign of the H−E direction* — and question 3 — *its maximum strength* —
> were settled empirically by phase 2 of the focused direction validation, which
> returned **`NO_STRENGTH_QUALIFIES`**: `d_S = H − E` is retired as an actuator,
> so it has no sign convention to get wrong and no maximum strength, and the
> H-versus-(E+B) probe continues as a sensor rather than as a steering
> direction. The answers live in
> `manifests/focused_direction_validation_v1_phase2_closure.json` and
> `configs/single_direction_controller.json`; the reasoning is in
> `docs/decision_log.md → 2026-08-26`. The remaining four are still open.

---

## 1. Retrospective Axis validation and the final inclusion list for the ten triplets

**Open.** The ten source cases carry `human_checked: true` in
`data/pathway_probe_cases.json`, but that is a self-declared boolean with no
reviewer identity, no timestamp and no adjudication record behind it. The five
OR-Bench-derived cases carry `human_checked: false` in the discovery packet and
`true` in the merged bank, with nothing recorded in between. The RQ2 development
counts the freeze was taken on *are* backed by a named manual semantic audit with
user confirmation (2026-07-21); the triplet-level check is not backed the same
way.

Separately, `rq1_build_retrospective_judge_packet.py`,
`rq1_retrospective_judge.py` and `rq1_retrospective_join.py` exist in the
historical workspace and produced
`results/rq1_retro_twoaxis_packet_20260813_102233`. Whether that retrospective
two-axis validation covers these ten cases, and what it concluded, has not been
established here.

**Not to be assumed:** that all ten are validated; that the inclusion list is
final; that `human_checked: true` is equivalent to the adjudicated Axis-2 labels
used for the 480-row corpus.

**What settles it:** locating the retrospective packet's coverage and verdicts,
and producing an inclusion list with a named basis per case.

## 2. The construction and sign of the H−E direction

**Open, and the sign is the sharp end.** The frozen actuator is a mean of
per-pair **B − E** deltas: from the over-refused target towards the benign
answered donor. The future H−E direction is to be built from the same validated
triplets, but *which* difference, in *which* order, and *for what purpose* are
three separate unanswered questions:

- `H − E` reads as "what distinguishes a justified refusal from an unjustified
  one" and is a plausible **veto** signal;
- `E − H` is its negation and would push a prompt *towards* the over-refusal
  region;
- neither is `H − B`, which is what the population bank's `true_refusal` term is,
  and which is a different contrast against a different reference class.

Getting the sign wrong produces an intervention that is confidently backwards and
whose outputs still look like outputs.

**Not to be assumed:** that H−E is built the same way as B−E; that per-pair
unit-normalisation before averaging carries over; that it applies at the same
layers; that it is used additively at all rather than as a gate input.

**What settles it:** an explicit statement of the contrast, the sign convention,
the reference class and the role, written before the vector is computed.

## 3. Its maximum strength

**Open.** 0.15 is the frozen actuator's strength and belongs to B−E at blocks
31/32/33. It is not evidence about any bound on a different direction, and the
population scan's 0.2 is a declared scan value rather than a measured optimum
(see `docs/decision_log.md`, 2026-08-21). There is no measured safe ceiling for
an H−E edit, because no H−E edit has been run.

**Not to be assumed:** that 0.15 transfers; that 0.2 is an upper bound; that any
strength is safe on H rows because 0.15 preserved refusal on them with a
different direction.

**What settles it:** a dose–response with a harmful-preservation arm, at whatever
strength grid is preregistered, before any operating point is fixed.

## 4. Raw versus orthogonalised direction treatment

**Open.** The two existing directions differ on exactly this axis and neither
outcome generalises to the third:

- the frozen actuator is **raw** — no orthogonalisation, no H term;
- the population bank is **orthogonalised** — the over-refusal signature with its
  component along the justified-refusal signature removed, then negated.

Whether a dual-direction controller should carry B−E and H−E raw, orthogonalise
one against the other, or orthogonalise both against something else, is a design
question with a testable answer and no default.

**Not to be assumed:** that orthogonalisation is always the safer choice; that
the population bank's Gram–Schmidt form is the one to reuse; that raw and
orthogonalised variants can be swapped after an operating point is chosen.

**What settles it:** running both forms as declared variants, with the H arm as
the discriminating measurement.

## 5. Coefficient-network architecture and loss

**Open, and entirely.** No architecture, no input representation, no output
parameterisation, no loss, no regulariser, no optimiser, no capacity bound and no
selection criterion has been chosen. Whether the per-prompt coefficients are
produced by a network at all, rather than by a rule over probe scores, is itself
open.

**Not to be assumed:** that it is a network; that it takes the residual stream at
one state; that it emits two scalars; that the loss is the joint repair-and-
preserve endpoint; that it is trained on the 480 rows.

**What settles it:** a written specification, reviewed before it is fitted, with
its selection rule fixed in advance — the same discipline
`configs/residual_geometry.json` applies to the corrected analysis.

## 6. Probe refitting at the layer-31 read state

**Open, and the numbering is part of the question.** "Layer 31" is a historical
zero-based decoder-block index and is **state 32** in this package's numbering;
state 31 is decoder block 30. Which of the two is meant has to be said, not
inferred, and the existing probe evidence is at **state 45**, which is neither.

Whether the controller reads at the actuator's own depth, at the probe's headline
state, or somewhere else is unresolved, and so is whether the probes are refitted
at that state or reused. Refitting changes what the probe scores mean; reusing
state 45 scores to gate an edit at states 32–34 asserts a transfer nobody has
measured.

**Not to be assumed:** that "layer 31" means state 31; that state 45's ROC-AUC
carries to the actuator's states; that a probe fitted at one state can gate an
edit at another; that refitting is free of the selection bias the corrected
analysis exists to price.

**Fixed already, and not open:** when the probes are refitted, they use corrected
fold assignment **B, repeat 0**
(`outputs/residual_geometry/corrected/cv_assignments.csv`, SHA256
`be3961a2…cd8d1d85`), which is the assignment the direction bank already
inherits. The pilot's assignment is kept for the reproduction gate and for
nothing else.

---

## What this phase did fix, so the list stays short

- the frozen actuator's strength is 0.15, its layers are 31/32/33 = states
  32/33/34, its budget reference is the block input, and its edit is additive at
  the final prompt token during prefill — all gated in code
  (`manifests/rq2_actuator.json`, `src/routegate/actuator.py`);
- the 480-row corpus stays unpaired;
- the population bank at 0.2 is an ablation and is named as one;
- the five artefacts are distinguished in `docs/artifact_taxonomy.md`.

Everything above this line is still someone's decision to make.
