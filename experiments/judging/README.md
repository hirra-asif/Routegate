# The common judging framework

One engine, nine stages, one command. A future steering experiment supplies a
**manifest**; it does not receive newly written judging logic.

```
python experiments/judging/run_protocol.py --manifest <manifest> --stage <stage>
```

## Why this exists

Phase 1, phase 2 and the two-site actuator each grew their own copy of the same
nine steps. They agree on the science and differ only in bookkeeping — which is
the worst way for code to differ, because every copy is a place a gate can
quietly go missing. A fourth steering mechanism should not need a fourth copy.

## What it does not do

It does not re-implement the rules, and a manifest cannot loosen them:

| rule | lives in | reused how |
|---|---|---|
| exact item-id echo | `triplets.validate_parsed` | same check, re-expressed for multi-field rubrics in `judgingprotocol.validate_reply` |
| adjudication | `focusedjudging.adjudicate_cell` | imported and called unchanged |
| sealed-set refusal | `validation.assert_not_sealed` | applied to every manifest and path file |
| credential redaction | `triplets.redact_exception_message` | applied to every recorded error |
| verdict vocabulary | `triplets.AXIS_VALUES` | rubrics register additively; redefinition is refused |

**It does not touch the completed runs.** Phase 1, phase 2 and the two-site
stage keep their own scripts and their own artifacts, byte for byte. This engine
is for the experiments that come next, which is why some duplication remains on
purpose: retrofitting it onto a finished run would put completed results behind
new code.

One deliberate divergence: `triplets.assess_failed_row` (used by
`triplet_validation/recover_failed_rows.py`) can, under ten gates, turn a
malformed-echo response into a verdict when the request-digest chain proves
identity independently. **This engine does not wire that in.** `malformed` is
constrained to `unresolved_technical` and the manifest rejects any other value.

## The stages

| stage | cost | what it does |
|---|---|---|
| `preflight` | free | every gate checkable before anything is built |
| `build` | free | mint opaque ids, write the blind packet and private row map |
| `dryrun` | free | render and hash every request; calls nothing |
| `judge` | **billed** | one judge over every row; `--judge sonnet`, then `--judge opus` |
| `join` | free | bind both judges onto cells; decode pairwise orientation |
| `export` | free | the blinded review packet — genuine disagreements only |
| `import` | free | read the reviewer's answers and adjudicate |
| `adjudicate` | free | alias for `import` |
| `report` | free | verdict tallies with explicit denominators |

Every stage accepts `--dry-run`, which runs the gates and writes nothing.

## What a manifest declares

Identity (`experiment_id`, `packet_id`, row prefixes, foreign prefixes),
candidate and reference arms, hash-gated inputs, class→rubric assignment,
per-rubric frozen digests and vocabulary, judge model ids, sampling,
adjudication policy, expected counts, and output locations. See
`make_demo_fixture.py` for a complete worked example.

Two things a manifest **cannot** say:

* that a missing, malformed or transport-failed reply becomes a verdict —
  `FAILURE_POLICY` admits exactly one value;
* that a genuine disagreement is resolved by anything other than blinded human
  review.

Widening either is a change to the science, not to a config.

## Single-response and pairwise rubrics

A single-response rubric judges one answer. A pairwise rubric compares two, and
needs **two vocabularies**: `judge_values` in A/B presentation space (what the
judge may answer, knowing nothing about which arm sits where) and `values` in
candidate/reference space (what the analysis sees, after the private row map
decodes the orientation). Conflating them is how an undecoded verdict reaches a
reading, so the manifest requires both and the decoder refuses a result outside
`values`.

The A/B orientation is drawn per row from `secrets` — not from a salt and not
from the row id, because an orientation recoverable from the manifest is not
blind to anyone holding the manifest. It survives only in the private row map.

## Several steering arms

`arms.candidate` is a list, so one framework covers no steering, static repair
steering, binary/CAST-style gating, a dynamic coefficient controller, a
safety-projected controller and layer/site ablations. Each is an arm id in a
manifest; none is new code.

## Experiment-specific readings stay outside

`report` emits per-class verdict tallies, decided denominators and the
unresolved list. A repair rate, a regression rule or a dose-response curve is
the experiment's own calculation, computed from those tallies in its own module
(`reporting.module` names it, for the record). The common evidence and
adjudication logic never learns what any particular experiment is trying to
show.

## Files

```
experiments/judging/run_protocol.py        the one operator command
experiments/judging/make_demo_fixture.py   a synthetic experiment, for rehearsal
src/routegate/judgingprotocol.py           manifest schema, gates, primitives
src/routegate/judgingstages.py             the nine stages
tests/test_judging_protocol.py             37 tests
```
