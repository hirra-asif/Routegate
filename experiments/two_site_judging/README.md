# `two_site_actuator_judging_v1`

Blind judging of the completed **two-site static actuator candidate**,
`two_site_actuator_validation_v1`, arm `repair_015_blocks31_32` — the repair
direction applied additively at package states 32 and 33 (historical decoder
blocks 31 and 32), at `alpha_R = 0.15`, `alpha_S = 0`, with no controller.

The generation stage produced 480 candidate responses and judged none of them.
This stage is that judging.

**Nothing here has been run.** No API call has been made, no model loaded, no
GPU used and no sealed material read. Every command that costs money is in
`RUNBOOK.md` and is run by the operator.

---

## What it judges, and with which frozen rubric

| class | rows | rubric | what is judged | answer |
|-------|-----:|--------|----------------|--------|
| E | 89 | frozen **Axis-2 response-behaviour** (`rq3_axis2_response_behaviour_v1`, CHG-002) | the candidate response | one of SA / PA-S / PA-N / DR / IR / NR / UM |
| B | 178 | frozen **blinded pairwise answer-quality** (`rq3_b_quality_pairwise_preference_v1`) | the referenced baseline **against** the candidate, at a randomised A/B presentation | a preference **and** a difference strength |
| H | 213 | frozen **actionable-harmful-help safety** (`rq3_safety_actionable_harmful_help_v1`) | the candidate response | YES / NO / UNCLEAR |

**480 rows per judge. Sonnet and Opus both see every row. 960 billed calls.**

Axis 1 is not run. It is prompt-only, every prompt here is a row of the
accepted corpus whose prompt status was settled when that corpus was built, and
re-asking it would answer a question this stage is not about.

### The three rubrics are located, not written

None of them is new, and none was redesigned for this stage.

* **Axis-2** and **safety** come from `manifests/rq1_judge_protocol.json`,
  which already settled — in writing, before this stage existed — that CHG-002
  Axis-2 and the corpus safety axis are the frozen protocol and that the
  undocumented 2026-08-04 `ed57133b` Axis-2 generation is not used. Their text
  lives at `<triplet_validation_dir>/frozen/rq1_judge_protocol/` and is loaded
  by `routegate.triplets.load_protocol`.

* **Quality** is the rubric that produced `final_quality_outcome` and
  `final_difference_strength` for the 178 B rows of
  `rq3_static_steering_ebh_master_20260811.csv`. It was frozen 2026-08-10 as
  `rq3_bh_rq2_static_b_quality_freeze` and its three digests are pinned in
  `manifests/two_site_actuator_judging_v1.json`. Its text is staged read-only
  at `<two_site_judging_dir>/frozen/b_quality_rubric/` and is loaded by
  `routegate.twositejudging.load_quality_protocol`, which refuses it unless
  every digest and every length matches byte for byte.

  Using it unchanged is the whole point: it is what makes this candidate's B
  result directly comparable with the static actuator's over exactly the same
  178 rows. **No replacement rubric was written**, and had it not been locatable
  by digest, `preflight.py` gate 5 is where this stage would have stopped.

---

## The B comparison

The quality rubric shows a judge one request and two replies, labelled A and B,
and asks which is the better answer. Which position holds the candidate is
drawn **per row from `secrets`** — not from a salt, not from the row id, not
from a seed anyone could re-run — and the only record of it is
`<two_site_judging_dir>/judging/packet/private/row_map.jsonl`.

That is deliberately stronger than the historical B run, which derived
orientation deterministically from a recorded salt and the row id: anyone
holding that freeze manifest can recover which side was which. Here there is
nothing to recover it from.

One quality decision is a **pair** — an outcome and a strength — and the two are
adjudicated together. Two judges agree about a B cell only when they name the
same winner **and** the same strength; naming the same winner at different
strengths is not agreement, and that cell goes to a human. The pair travels as
one composite label (`BEFORE_BETTER|MATERIAL`, `EQUIVALENT|NONE`, …) so that
the frozen adjudication rule, which compares two verdict strings, decides it
without being changed.

`BEFORE` is the referenced baseline; `AFTER` is the candidate. That is the
static master's own vocabulary, kept so a count here means the same thing a
count there means.

The frozen **pairing rule** — `EQUIVALENT` ↔ `NONE` is an iff, a named winner
takes `MINOR` or `MATERIAL` — is applied exactly as frozen: a judgement that
breaks it is recorded verbatim and **flagged, never rewritten**.

---

## Adjudication

`corpus_disagreement_only`, declared in
`manifests/two_site_actuator_judging_v1.json` on 2026-08-27, **before any
billed call**. Sonnet and Opus judge every cell, their agreement is final, and
a human reads only the cells they split on. That is the procedure the E/B/H
development corpus itself was labelled under, which is what keeps these
readings comparable with the existing static-steering and RouteGate E/B/H
results.

`triplet_independent_review` is **not** used here. Every entry point refuses a
`--procedure` that is not the declared one.

The rule is `routegate.focusedjudging.adjudicate_cell` — **imported and called
unchanged**, never copied and never reimplemented. It is reached through
`routegate.twositejudging.adjudicate`, which validates a human verdict against
this stage's vocabulary and then delegates. It never invents a verdict, never
breaks a tie by majority over two judges, and never converts a recorded failure
into a label. `UNRESOLVED` is a finding: not a quiet pass and not a quiet fail.

---

## Blindness

A judge-facing row carries **exactly** its rubric's whitelist:

```
axis2, safety : row_id, prompt_text, response_text
quality       : row_id, prompt_text, response_a, response_b
```

The guarantee is structural rather than a scan. Every row is **constructed**
from named values — a screen record is never copied, filtered or stripped — and
every rendered request is re-derived from the frozen instructions and template
filled with exactly those values and required to match byte for byte. A
treatment field cannot reach a judge by omission, and authentic prompt and
response text is never inspected for what it happens to say.

Withheld: the class, the system, the arm, `alpha_R`, `alpha_S`, the direction,
the layers/blocks/package states, baseline-versus-candidate, **which of A and B
is which**, the coefficients, the expected outcome, the corpus row id (which
names its source benchmark), row grouping, the other judge's verdict, judge
agreement, token counts and cap status.

Opaque ids are cryptographically random (`secrets`), prefixed `TSJ1RESPONSE`,
`TSJ1QUALITY`, `TSJ1SAFETY` and `TSJ1HUMAN`, and are checked against every
other packet's prefixes so two experiments' rows can never be joined.

**Stated limitation.** Each judge, and the human adjudicator, is told which
rubric to apply. Because the rubric is assigned by class here, that implies the
class. It implies nothing else. The frozen rubrics are used exactly as they
are and are not redesigned to hide the task, because a rubric a judge cannot
identify is a rubric a judge cannot apply.

---

## The readings, frozen before any verdict exists

* **E** — the full Axis-2 distribution over 89 rows; **substantive repair**
  (SA **or** PA-S) and **full repair** (SA), over the same denominator. PA-S is
  never quietly counted as SA.
* **B** — the distribution over composite decisions; **material degradation**
  (`BEFORE_BETTER` at `MATERIAL`) and **non-material degradation**
  (`BEFORE_BETTER` at `MINOR`), with improvements reported beside them.
* **H** — YES / NO / UNCLEAR, kept separate. `UNCLEAR` is not `NO`.

A class with any `UNRESOLVED` cell has its counts **withheld** and the gap
reported instead. A count over an unknown denominator is not a count.

Nothing is selected from these verdicts. The arm was fixed in advance by
`manifests/two_site_actuator_candidate_v1.json`.

---

## What this stage is not

* It does **not** reuse `experiments/focused_direction_judging/build_judge_packet.py`.
  That is a different packet over a different frozen input, and none of its
  Phase-1 checks is relaxed, widened or reused with a looser argument.
  `preflight.py` proves the code path does not even import it.
* It does **not** generate a response, load a model, or touch a GPU.
* It does **not** read, name or resolve any held-out or sealed material.
* It does **not** read, write or alter any frozen result — the static-steering
  master and the phase-1/phase-2 adjudications are opened read-only or not at
  all.
* It does **not** make a paired baseline-to-candidate claim on E or H. Only the
  candidate response is judged on those two rubrics; the baseline's Axis-2 and
  safety labels came from a different judging run and are not re-derived or
  differenced here.

---

## Files

| file | what it does | costs money |
|------|--------------|-------------|
| `preflight.py` | every gate that must hold before a billed call | no |
| `build_two_site_judge_packet.py` | the blind 480-row packet and the private map | no |
| `run_two_site_judge.py` | one judge over all three rubric files | **yes** |
| `join_two_site_verdicts.py` | two judges onto 480 cells; decodes the B orientation | no |
| `export_two_site_human_review.py` | the blinded disagreement-only review packet | no |
| `import_two_site_human_review.py` | the adjudication, under the frozen rule | no |
| `report_two_site.py` | the deterministic E/B/H report | no |

Every one of them takes `--dry-run`.

Package vocabulary: `src/routegate/twositejudging.py`.
Predeclaration: `manifests/two_site_actuator_judging_v1.json`.
Tests: `tests/test_two_site_judging.py`.

Nothing under `<two_site_judging_dir>` is ever committed: it holds the frozen
rubric text, prompt and response text, the private maps that de-anonymise the
opaque ids and record the A/B orientation, and every raw API body.


## Technically unresolved cells (2026-08-27)

Three of the 960 planned verdicts never arrived, each after repeated HTTP 200
replies the frozen schema check rejected: two quality cells whose reply echoed
an `item_id` one character from the one requested, and one safety cell that
returned zero content blocks. 957 verdicts are valid; 477 of 480 cells are
dual-judged.

Those three cells are **carried, not decided**. They keep the other judge's
verdict, they hold the join state `unresolved_technical`, and
`focusedjudging.adjudicate_cell` labels them `UNRESOLVED` on its own
missing-evidence branch. Nothing about how a cell is decided changed:
`corpus_disagreement_only` and the frozen echo check are untouched, the
mismatched ids are never accepted, the empty reply is never interpreted, and a
human is never asked to supply a verdict a judge did not give.

The carry-forward is pinned in
`manifests/two_site_actuator_judging_v1_technical_unresolved.json` and gated by
`src/routegate/twositeunresolved.py` — eight fail-closed gates, evidence
re-hashed from disk, failure type re-derived from the preserved body, selection
proved outcome-blind. A failure the manifest does not name still stops the join.

Every reading is reported twice: the original-protocol count, withheld while any
cell of its class is unresolved, and a complete-case count with its denominator
named. No rate divides by the class total when fewer cells were decided.

See the 2026-08-27 entry in `docs/decision_log.md`.
