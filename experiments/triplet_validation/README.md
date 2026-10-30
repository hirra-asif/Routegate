# Retrospective validation of the ten operative E/B/H triplets

**Status: implemented, nothing run.** No API call has been made, no model
loaded, no GPU used. Every command that costs money or touches data is in
`RUNBOOK.md` and is run by the operator, not by this repository.

## What this is, and what it is not

This is **retrospective** validation. The steering outcomes it validates the
inputs of were produced in July 2026 and were already known when these
judgements were prepared, in August 2026. Saying so plainly is the point:
a retrospective check that presents itself as prospective is worth less than
one that admits what it is and controls for it.

The control is blindness. Every judge and the human reviewer see a prompt, or a
prompt and one response, and an opaque id. They do not see the case, the arm,
the E/B/H role, the donor number, the grouping, the steering condition, the
layer, the strength, the repair outcome, the expected verdict, or each other's
answers. And **no case is replaced on the strength of its own validation
result** — a failing case is reported as failing, in place.

## Why a supplement was needed

The historical packet `rq1_retro_twoaxis_packet_20260813_102233` is well built,
properly blinded, and **was never executed** — its own `validation.json` records
`api_calls_attempted: 0`. It covers three of the five inclusion-critical arms:

| arm | in the historical packet | why it matters |
| --- | --- | --- |
| `E` | yes | the over-refused target, needed by every rule |
| `H` | Axis 1 and Axis 2 only | needs a **safety** judgement the packet does not carry |
| `B_d1` | yes | the **comparator** actuator's donor |
| `B_d2` | **no** | one of the two donors the **primary** actuator averages |
| `B_d3` | **no** | the other |

The primary frozen actuator `consensus23_global_frozen` is built from d2 and d3.
The packet's only donor is d1. So the arms that matter most to the actuator in
use were the ones nobody had prepared.

`build_supplemental_packet.py` adds exactly the missing rows — 50 of them — and
nothing else. It generates no response: every prompt and every response already
exists, frozen, and is reconciled by SHA256 before it is written into a row.

## The files

| file | what it does | touches an API |
| --- | --- | --- |
| `build_supplemental_packet.py` | the 50 missing rows, from frozen text only | no |
| `run_judge.py` | one judge over one packet, resumably | **yes** |
| `join_verdicts.py` | two judges' verdicts back onto (case, arm, axis) cells | no |
| `export_human_review.py` | the blinded human packet, no judge verdicts in it | no |
| `import_human_review.py` | the completed review, and the adjudication | no |
| `build_inclusion_manifest.py` | `manifests/rq1_triplet_inclusion.json`, metadata only | no |
| `RUNBOOK.md` | the exact terminal commands, in order | — |

Shared vocabulary, protocol loading, blindness guards, the transport contract
and the frozen inclusion rules live in `src/routegate/triplets.py`.

## Two runners, two record shapes

The original packet is judged by the **historical** runner in the SteerMoE
workspace; the supplement is judged by `run_judge.py` here. Their result
records genuinely differ, and reading one by the other's conventions does not
raise — it silently yields an empty join, which looks exactly like "nothing has
been run yet".

| | historical | this repository |
| --- | --- | --- |
| blinded id | `item_id` (one-way from `row_id`) | none |
| per-record digest | none | `record_sha256` |
| a **successful** record | carries **no** `status` | `status: "OK"` |
| failures go to | `unresolved.jsonl` | `failures.jsonl` |
| join key | `row_id` | `row_id` |

`detect_result_format` names the format per record and refuses one that matches
both or neither. `result_is_success`, `result_join_key`, `result_verdict` and
`result_record_hash` then read it by that name. The historical record also
stores the rendered request, the raw payload and the judge's raw text; none of
those fields is read or copied — only labels, digests and opaque ids cross into
a joined cell.

## The transport

`X-Api-Key`, not `Authorization: Bearer`. The key is read from `MODEL_API_KEY`
(the name the historical runner uses, so one export serves both), falling back
to `ROUTEGATE_JUDGE_API_KEY`. Under `apptainer --cleanenv` the host environment
is dropped, so the runbook propagates it as `APPTAINERENV_MODEL_API_KEY` and
proves it arrived by presence and length only. It is never put on a command
line: on a shared login node that is visible in `ps` to everyone.

The response envelope is undocumented, so the exact raw body of every call is
preserved first, mode `0600`, outside the repository, with its digest in the
record — and only then parsed. `extract_text` handles the Anthropic content
**block list** and the gateway wrappers; there is no keyword rescue.

The served model is read from the response. An **absent** model identifier is
not accepted: it is recorded as `UNESTABLISHED` and the row fails. The expected
identifier is never written in as though the server had confirmed it.

**A failed row is not a free row.** Nothing retries automatically. A request
that timed out or failed after the response arrived may already have been
billed, and is recorded — with that note — rather than re-sent.

## What the human reviewer gets

The same frozen instruction text the judges received, hash-verified from the
same files, plus the allowed labels for that axis task and its `axis_task`
marker. A reviewer given only a label list would be inventing a rubric, and the
human/judge comparison would not be one. The instructions name no case, arm,
role, donor, condition or expected verdict, so this un-blinds nothing. They are
written to the machine-local review directory, never to the repository.

## The frozen protocol

Nine digests in `manifests/rq1_judge_protocol.json` gate the instruction,
template and schema text. The text itself is **not committed** — it is protocol
text from the historical workspace, and committing it would both vendor a file
whose licence this repository has not settled and put judge-facing prompt text
into git. It is copied once into the machine-local working root, and every read
re-hashes it. One differing byte is refused.

> CHG-004 is the frozen conservative retrospective Axis-1 protocol for this validation. It corrects the calibrated educational-wrapper wording, introduces no E/B/H role or response information, and moves calibration outcomes away from rather than toward PERMITTED. It is not claimed to be the Axis-1 version used throughout the complete E/B/H construction.

Applied consistently to the original **and** the supplemental triplet rows — one
Axis-1 protocol for all five arms. Its calibration outcome is **not** formally
accepted historical evidence: the re-calibration ran, met its item-level
criterion (`ITEM-1D1F880D6848` returned to PROHIBITED) and missed its
AMBIGUOUS-stability criterion (2 → 4), and no record accepts it. It is used
because it fails closed for these rules: AMBIGUOUS satisfies neither PERMITTED
nor PROHIBITED, so a stricter protocol can only exclude, never falsely include.

> The retrospective packet's hash-pinned Axis-2 protocol is used for every triplet arm. A later 2026-08-04 re-audit used an unavailable undocumented variant. That variant is not reconstructed, imitated or used here. This is reported as a provenance limitation.

`preflight.py` gate 5 re-renders every Axis-1 and Axis-2 request in both packets
and **fails closed** if any resolves to a different instruction, template or
schema hash.

## Capped donor responses

28 of the 30 donor baselines reached `max_tokens=256`. The complete historically
stored response is used exactly as produced. It is never regenerated and never
extended; there is no generation path in this stage at all. Reaching the cap
neither passes nor fails a response — the frozen Axis-2 rubric classifies what
is actually there, and for donor inclusion **only `SA` satisfies the rule**.
`PA-S`, `PA-N`, `NR`, `DR`, `IR`, `UM` and an invalid or missing verdict do not.
`reached_max_tokens` and `output_token_count` are preserved from the capture
through to the final evidence record.

## The inclusion rules, frozen before any verdict exists

| rule | needs |
| --- | --- |
| `primary_actuator_source` | E: PERMITTED + (DR\|IR) · d2: PERMITTED + SA · d3: PERMITTED + SA |
| `comparator_actuator_source` | E: PERMITTED + (DR\|IR) · d1: PERMITTED + SA |
| `h_e_direction_source` | E: PERMITTED + (DR\|IR) · H: PROHIBITED + (DR\|IR) + safety NO |

`PA-S`, `PA-N`, `NR` and `UM` satisfy no rule and are never coerced into a
passing label. A patched response is never an inclusion input.

Three statuses, kept apart: `validated`, `invalid`, `unresolved`. "We do not
know" and "we know it fails" are different findings, and only the second is a
reason to exclude a case.

## What is still open

The manifest will not be written until every required judgement exists — all
110 cells, 11 per case across the ten frozen case ids. A case that produces no
cell at all is detected and reported as `unresolved` on every rule, not omitted;
an eleventh case id is refused outright.

Until those judgements exist the honest statement is the audit's: **all ten
cases are unresolved on every arm.**
