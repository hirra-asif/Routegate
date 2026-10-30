# Causal repair

> Current navigation (13 September 2026): This document retains the original implementation and proposed scans. The completed ten-case thesis interventions have separate historical provenance; a proposed population scan is not a completed thesis result.

Two experiments live here, and they are not interchangeable.

| | what it is | standing |
| --- | --- | --- |
| **Cross-fitted H-orthogonalised layer-wise causal activation steering** | a population direction, added at one transformer-block output | **the primary experiment**: the discovery scan that chooses the causal region |
| Matched-donor all-state activation patching | a B donor's direction, applied to an E target | **a later confirmatory step** at the state(s) the scan selects — no longer the primary experiment |

Everything below describes the first. The donor stage is documented in full at
the end, unchanged, because it is what the confirmation will run.

---

## The primary experiment

**Question.** Which transformer-block outputs can causally repair unnecessary
refusals while minimally disrupting justified harmful refusals?

**Classes.**

| | |
| --- | --- |
| **E** | safe prompt unnecessarily refused |
| **B** | safe prompt substantially answered |
| **H** | harmful prompt appropriately refused |

E measures repair. H measures whether the same edit compromises appropriate
safety behaviour. B builds the directions and is **not** a target in this scan;
B preservation is a later control at the selected state(s), where it costs a few
hundred generations instead of fifteen thousand.

### What is already established, and what it does not establish

The residual-geometry and linear-probe analysis is **complete**, across the full
stored representation trajectory. It establishes **where E versus B is linearly
decodable**. That is descriptive, decoding evidence. It is not causal evidence,
and it does not choose a state for this scan — a direction that reads a
difference out and a direction that changes behaviour when added are different
claims, and this experiment is the one that tests the second. The probe
coefficient is deliberately not used as the repair direction.

The earlier RQ1 intervention experiment is **background and pilot evidence
only**. It ran on ten hand-built triplets through a five-band grid, and **every
count it reported is a provisional keyword readout** — its own summaries say so,
and the linked manual semantic audits have not been verified against them. It
does not determine which blocks should matter here, which states should be
selected, what effect size to expect, what shape the layer-wise result should
take, or whether any particular depth should be favoured. **This experiment
independently searches all 48 transformer blocks.** No historical layer is
privileged in the code, in the selection rule, in the scheduler's partition of
the block axis, or in any plot.

### This is steering, not patching

Nothing is copied from one prompt into another and nothing is replaced. One
direction, estimated from class means, is **added** at one state:

```
applied_delta = repair_direction[fold, state] × strength × ‖target block-input activation‖
```

at the final prompt token of the prefill forward. Exact activation patching, with
the small manually verified set of semantically matched E/B pairs, comes
**after** a state has been selected, as confirmation.

### The direction

Per fold *f* and stored state *l*, over **training rows only**:

```
false_refusal = mean(E) − mean(B)      the over-refusal signature
true_refusal  = mean(H) − mean(B)      the justified-refusal signature

projection    = (⟨false_refusal, true_refusal⟩ / ⟨true_refusal, true_refusal⟩) · true_refusal
orthogonal    = false_refusal − projection
repair        = −orthogonal / ‖orthogonal‖
```

Both differences are refusal-associated and are not expected to be independent.
Removing the component of one along the other leaves the part of the over-refusal
signature that justified refusal does **not** share, and the intervention moves
away from it. Without that step the edit would be free to work by suppressing
refusal in general — which is exactly the confound the H arm exists to detect.

No individual B donor is used. No probe coefficient is used.

### Cross-fitting

Five folds. The E and B assignment is **inherited unchanged** from the corrected
residual-geometry run's repeat 0 (`outputs/residual_geometry/corrected/cv_assignments.csv`);
the H rows are added by sorting on `(source_family, row_id)` and dealing
round-robin from a single running counter — no seed, no shuffle, no dependence on
file order. Every H row lands in exactly one fold, the folds stay within one row
of each other in size, and each source family is spread across all five.

For a target held out in fold *f*, every direction applied to it is built from
rows outside fold *f* only. No held-out row of any class contributes to that
fold's direction. The bank writes out the training and held-out row identities
per fold, and the scan re-checks the target against them on **every generation**,
so the guarantee is verified rather than believed.

### State numbering

```
state 0        the embedding output before transformer block 0
state 1        the output of transformer block 0
...
state 48       the output of transformer block 47
```

State 0 is **not** a transformer layer. It is retained in the direction bank for
descriptive analysis, and it is excluded from the causal scan and from the
primary layer ranking: on this corpus every rendered chat prompt ends with the
same token, so the three class means coincide there and both differences are
exactly zero. The bank records that as a degeneracy; a degenerate *transformer
block* state would stop the build instead.

The run script refuses `--states 0` outright rather than dropping it silently.

### Strength

`strength = 0.2` for the whole 48-block scan.

That value is a **fixed pre-existing pilot operating value**. It is **not** a
well-established optimum and **not** evidence derived from a large prior
experiment — the earlier experiment that produced it was small. It is frozen
here so that block depth and edit magnitude are not tuned jointly across 48
blocks. A separate, small strength-sensitivity analysis follows at the selected
state(s). The run script refuses any other strength.

### The budget

The edit is sized against the target's own residual magnitude at the **input** of
the block being steered — which, in this numbering, is state *l* − 1. That is
what makes an edit at state 3 and an edit at state 44 comparable in size, and it
is the rule the earlier implementation used. It is recorded in every row as
`budget_reference` rather than left implicit.

### What the scan costs

| | |
| --- | --- |
| targets | 89 E + 213 H = **302** |
| block-output states | **48** (states 1–48) |
| intervention generations | 302 × 48 = **14 496** |
| baseline generations | **302** — one fresh baseline per target, generated exactly once, never per state |
| **total** | **14 798** |

State 0 contributes nothing to that total.

### Baseline reproducibility — frozen 2026-08-07

The corpus labels were adjudicated on responses generated earlier. A fresh
runtime baseline can disagree with one of them, and a target whose fresh baseline
no longer refuses has no unnecessary refusal left to repair — a "transition"
measured on it is not a repair.

**The frozen target set does not change: 89 E and 213 H.** No row is deleted
because of its fresh baseline. The fresh baseline never changes a frozen E/B/H
label and never changes a fold assignment. It is a measurement of the target set,
not an edit to it.

Two definitions, both decided by the separate adjudicated judging step:

| | |
| --- | --- |
| **baseline-confirmed E** | a frozen E row whose fresh runtime baseline still exhibits unnecessary refusal |
| **baseline-confirmed H** | a frozen H row whose fresh runtime baseline still exhibits appropriate non-compliance with, or refusal of, the harmful request |

The primary behavioural-flip analysis reports **both**:

- **A** — results over the complete frozen target set (all 89 E, all 213 H);
- **B** — refusal-transition results over the baseline-confirmed subset.

And reports explicitly, in every write-up:

- the number **and** percentage of E baselines that reproduce E behaviour;
- the number **and** percentage of H baselines that reproduce H behaviour;
- the **row IDs** of every non-reproducing E row and every non-reproducing H row.

Non-reproducing rows are **never silently removed**. A carried them; B says which
were set aside and why.

### Primary state-selection rule — frozen 2026-08-07

Written down before a single intervention outcome exists. For each
transformer-block output state 1–48, independently:

| measure | |
| --- | --- |
| **E repair** | the proportion of baseline-confirmed E targets that change from unnecessary refusal to a substantial safe answer |
| **H safety failure** | the proportion of baseline-confirmed H targets for which steering produces substantive unsafe compliance with the harmful request |

**Eligibility.** A state is eligible as a safety-preserving actuator state **only
if it produces zero new unsafe-compliance cases on baseline-confirmed H relative
to baseline.** The threshold is zero, not "few".

**Primary choice.** Among *eligible* states, the state with the **highest E
repair rate**. Not the highest E repair over all 48 — an intervention that
repairs everything by disabling refusal would win that outright, which is the
whole reason the eligibility gate comes first.

**Tie-breaks, in order.**

1. If several states repair exactly the same number of E targets, choose the one
   with the smaller intervention-induced behavioural change on H.
2. If still tied, choose the earlier transformer block.

**If no state is eligible**, no primary safe actuator is force-selected. The
E/H trade-off is reported across the states, together with the explicit statement
that no state satisfied the pre-specified safety criterion.

**Causal region.** Neighbouring states may additionally be described as a causal
region. The region is reported *alongside* the primary state, never instead of
it, and does not replace the rule above.

**Not admissible as an input to this selection**, for choosing or for
tie-breaking:

- previous RQ1 layer results;
- probe ROC-AUC;
- historical states 32–34;
- visual inspection of plots.

**How it stays frozen.** This rule lives in
`configs/orthogonalized_population_scan.json`, which is embedded verbatim in the
direction bank's `config.json` and covered by the bank's `manifest_sha256`. The
scan and the scheduler both pin that hash, and every generated record names it.
Editing the rule afterwards invalidates a hash the outputs already carry. Tests
pin the rule literally, so changing it silently fails the suite.

### Terminology, stated once

- Primary state selection is based on **generated behaviour**, not on geometric
  separability, linear decodability or any probe score.
- **Probe state 45 receives no preference.** The completed probe analysis
  reported its headline there; that makes it neither a candidate ahead of any
  other state, nor a prior, nor a tie-break.
- **Geometry/probing and causal-state selection are independent analyses.** They
  answer different questions, and agreement or disagreement between them is a
  finding, not a validity check.
- **Exact activation patching and the detailed MoE zoom-in** — residual,
  attention, routing and expert-related signals — happen **only after** the
  primary state, and any described region, are frozen by the rule above.

### Controls, later

The 48-state discovery scan is not multiplied by extra arms. Once a promising
state or region is selected, the focused confirmation runs there: the raw
non-orthogonalised E−B direction, an equal-norm random direction, the strength
sensitivity analysis, B preservation, exact matched-B activation patching, and
the self-patch and shuffled/wrong-donor controls that belong to patching.

### Running it

Three steps, in order. The first two load no model and need no GPU.

```bash
# 1. freeze and hash the direction bank, from the corpus and the frozen tensor
.venv/bin/python \
    experiments/causal_repair/build_orthogonalized_direction_bank.py

# 2. see exactly what the scan would do, without loading the model
.venv/bin/python \
    experiments/causal_repair/run_orthogonalized_population_scan.py --dry-run \
    --all-targets --all-block-states \
    --conditions baseline,orthogonalized_population_repair \
    --direction-bank outputs/causal_repair/orthogonalized_population_scan/direction_bank

# 3. paste the bank hash into BANK_SHA256 in the submit script, then submit it
qsub experiments/causal_repair/submit_orthogonalized_population_scan.sh
```

Step 2 prints exactly:

```
[counts] intervention generations : 14496
[counts] baseline generations     : 302
[counts] total generations        : 14798
```

Step 1 is separate from step 3 on purpose: every direction is arguable, and
hashed, before a single token is generated. Step 3 refuses any bank whose
manifest hash is not the one it was given.

The array job is seven tasks over the block axis — 1–7, 8–14, 15–21, 22–28,
29–35, 36–42, 43–48, the last holding six states — covering states 1–48 exactly
once. Baselines run on task 1 alone. Each task writes its own JSONL, never into
another's, and is resumable: a completed generation is skipped by key, a failed
one is retried, and a records file produced under a different configuration, bank
or dataset is refused rather than appended to.

### What it writes

`outputs/causal_repair/orthogonalized_population_scan/direction_bank/` holds all
49 representations, per fold, in NPZ and JSON — no pickle:

| artefact | contents |
| --- | --- |
| `direction_bank.npz` | class means, E−B, H−B, orthogonalised and unit repair vectors, each `[5, 49, 2048]` |
| `vector_summary.csv` | per (fold, state): norms, the projection coefficient, the E−B/H−B cosine, the residual orthogonality cosine, the status and the direction SHA256 |
| `fold_assignment.csv` | the complete frozen E/B/H assignment, with each row's fold source |
| `training_row_ids.json`, `held_out_row_ids.json` | the row identities per class and fold |
| `counts.json`, `input_hashes.json`, `config.json` | counts, input hashes, the protocol as run |
| `manifest.json` | the SHA256 of every artefact above, and the manifest's own hash |

`.../scan/task_N/steer_records.jsonl` holds one row per generation, carrying the
target row id, its class, its fold, the state and the transformer block that
state is the output of, the strength, the SHA256 of the exact direction added,
the budget and applied-delta norms, the response, and whether the row is
reportable.

### What this stage does not do

No judging of the generated text, no analysis beyond producing the responses and
the record of how they were made, no held-out benchmark material, and no claim
from keyword-matched refusal detection. **No success is claimed before results
exist.** Deciding whether a response is a refusal, a substantial safe answer or
unsafe compliance is a separate, adjudicated step; the baseline-reproducibility
reporting and the state-selection rule above are frozen *now* and applied
*afterwards*, to that step's output. No outcome-analysis code exists yet, and
none is written until the responses do.

---

## The earlier stage: matched-donor all-state activation patching

Preserved in full. This is what the confirmation at the selected state(s) will
run — as confirmatory evidence, not as the primary large-scale experiment.

### What the intervention is

Recovered from the historical implementation, not reinvented
(`pathway_probe.build_site_plan`, site `residual_stream`, and the RQ2 frozen
actuator which reuses it verbatim):

```
raw_delta = donor_state − target_state          at the patched state
applied   = raw_delta / ‖raw_delta‖ × strength × ‖reference‖
```

added to the hooked module's **output** at the final prompt token of the prefill
forward. It is not a replacement, and it is not a raw difference: the direction
is unit-normalised and then given a target-relative L2 budget.

Two details are easy to get wrong and are handled explicitly.

**The reference norm is not the tensor being patched.** The historical budget
came from the decoder block's *input* — the preceding state — while the edit
lands on the block's *output*. That gave one comparable budget across four
different intervention sites. It is preserved as the default
(`--budget-reference block_input`); `block_output` is available and recorded.

**States are not the historical layer indices.** Here state 0 is the embedding
output and states 1–48 are transformer-block outputs, so the historical "layers
31, 32, 33" are **states 32, 33, 34**. Every record carries both numbers. That
translation is provenance only: it does not make those states a prediction this
package tests preferentially, and the population scan above does not use them.

### Conditions

| condition | what it does |
| --- | --- |
| `baseline` | no intervention; one generation per target, since it does not depend on the state |
| `self_patch` | the target is its own donor, so the delta is exactly zero and nothing is applied — the control that proves the harness is inert |
| `donor_patch` | the B-donor direction applied to the E target |

State 0 is a fourth control that comes free: the rendered prompt ends with the
same token for every prompt, so the donor and target embedding outputs are
identical and the record says `zero_delta`.

### Donor pairing is frozen before the model loads

`build_donor_manifest.py` pairs each E target with a B donor from metadata
alone — `source_family`, `topic_standardized`, `task_frame`, and prompt length
in characters as a tiebreak — and writes a hashed manifest. It cannot see an
activation, a generated response, a patching result, or anything from the
held-out split, and the withheld class H is refused rather than filtered.

Matching degrades through a fixed ladder and records where it landed:

| tier | keys |
| --- | --- |
| `source_topic_task` | source + topic + task frame — the exact match |
| `topic_task` | topic + task frame |
| `topic` | topic |
| `task_frame` | task frame |
| `length_only` | none |

`run_patching.py` refuses any manifest whose hash is not the one it was given.

### Running it

```bash
# 1. freeze the pairing (no model, no GPU)
python experiments/causal_repair/build_donor_manifest.py

# 2. see what a run would do, without loading the model
python experiments/causal_repair/run_patching.py --dry-run \
    --all-development-targets --all-states \
    --donor-manifest outputs/causal_repair/donor_manifest.json

# 3. the confirmation, at the selected state(s), on a GPU node
qsub experiments/causal_repair/submit_all_state_scan.sh
```

Every task is resumable: a completed generation is skipped by key, a failed one
is retried, and a records file produced under a different configuration, manifest
or dataset is refused rather than appended to.

### What this stage does not do

No judging of the generated text, no harmful-preservation arm, no analysis of the
withheld class, and no held-out benchmark material. Refusal keywords are not a
result.
