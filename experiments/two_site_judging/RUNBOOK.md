# Runbook — blind judging of the two-site static actuator candidate

Copy-paste terminal commands, in execution order. Steps 1-5 **have been run**:
957 of the 960 planned verdicts arrived and three are technically unresolved
(step 6). Steps 6-9 are free and offline. Read `README.md` first; read step 0
before running anything at all.

Two commands cost money: step 4 runs Sonnet over all 480 rows and step 5 runs
Opus over the same 480. **960 billed calls in total.** Everything else is free
and offline.

Shell variables used throughout:

```bash
export RG=$HOME/routegate
export TV=$HOME/routegate-data/triplet_validation
export TSJ=$HOME/routegate-data/two_site_judging_v1
export CONS=$HOME/routegate-data/two_site_validation_v2_consolidated
export CONTAINER=$HOME/containers/vllm-openai-v0.10.1.1.sif
export APP="apptainer exec --cleanenv --bind $HOME:$HOME $CONTAINER /bin/bash --noprofile --norc -lc"

# The reviewed commit this run is pinned to.
export EXPECT_HEAD=<the reviewed commit that carries this stage>
```

`--nv` is deliberately absent from every command below. Nothing in this
workflow uses a GPU.

---

## 0. Before anything

- These are **billed API calls**. Steps 1 to 3 are free; run them first.
- Every judging step is **resumable**. If one dies, re-run the identical
  command: rows already recorded as *successful* are skipped and are not
  re-sent, so they are not billed twice.
- **A failed row is not a free row.** `run_two_site_judge.py` retries nothing
  automatically and says why: a request that timed out, returned an
  unparseable body, returned an unexpected model, or failed after the response
  arrived **may already have been billed**. It goes to `failures.jsonl` with
  that note. Re-running it is a second billable call and a deliberate decision
  by you.
- Every attempt is preserved — successes, failures, and the raw body of
  anything that could not be parsed.

### 0a. The declared adjudication procedure

Already declared, on 2026-08-27, before any billed call:

```bash
export PROCEDURE=$(python3 -c "
import json;print(json.load(open('$RG/manifests/two_site_actuator_judging_v1.json'))['adjudication_procedure'])")
echo "  procedure = $PROCEDURE"      # expect corpus_disagreement_only
```

`corpus_disagreement_only` — Sonnet and Opus judge every cell, their agreement
is final, and human adjudication is required only where they disagree. This is
the procedure the E/B/H development corpus itself was labelled under, which is
what keeps these readings comparable with the existing static-steering and
RouteGate E/B/H results.

Every stage refuses a `--procedure` that is not the declared one.

### 0b. The credential

Set it in the shell, never in a file.

```bash
# MODEL_API_KEY is the name the frozen runner reads.
read -rs -p "judge API key: " MODEL_API_KEY && export MODEL_API_KEY && echo

# $APP uses --cleanenv, which drops the entire host environment. Apptainer
# forwards APPTAINERENV_X into the container as X, which is the only
# propagation that survives --cleanenv and keeps the value off every command
# line and out of every process listing.
export APPTAINERENV_MODEL_API_KEY="$MODEL_API_KEY"
export ROUTEGATE_JUDGE_ENDPOINT="<the gateway endpoint>"
export APPTAINERENV_ROUTEGATE_JUDGE_ENDPOINT="$ROUTEGATE_JUDGE_ENDPOINT"
```

Neither the key nor the endpoint is ever written into a record, printed,
logged, committed, or placed on a command line.

---

## 1. Stage the frozen B quality rubric (once)

The Axis-2 and safety rubric text is already at
`$TV/frozen/rq1_judge_protocol/`. The B quality rubric needs staging once, from
the historical workspace, **read-only and byte-for-byte**:

```bash
SRC=$HOME/SteerMoE-main/results/rq3_bh_rq2_static_20260810/b_quality
mkdir -p $TSJ/frozen/b_quality_rubric
for f in quality_instructions.txt quality_user_template.txt quality_schema.json; do
  cp -n "$SRC/$f" "$TSJ/frozen/b_quality_rubric/$f"
done
chmod 0444 $TSJ/frozen/b_quality_rubric/*

sha256sum $TSJ/frozen/b_quality_rubric/*
# expect
#   96157444a9e1dac0967e83633fdae9f6e4d6ebbb58e60c816886ec42752e3b40  quality_instructions.txt
#   56f1d7b0b13a086033087e272d4c239977a2142fb7aa5ecedfa8b657bd6e6bcb  quality_schema.json
#   b6acc4c6ef49d73972b7fe316540f606d0b641b953869fa4daa4122d0060d6a8  quality_user_template.txt
```

The three digests are pinned in `manifests/two_site_actuator_judging_v1.json`
and are checked on **every** read. The text itself is never committed.

Point `configs/paths.local.json` at both roots (already done on this machine):

```json
"two_site_validation_v2_consolidated_dir": "/home/<user>/routegate-data/two_site_validation_v2_consolidated",
"two_site_judging_dir": "/home/<user>/routegate-data/two_site_judging_v1"
```

---

## 2. Preflight, before the packet exists

```bash
$APP "cd $RG && .venv/bin/python experiments/two_site_judging/preflight.py \
  --stage input --expect-head $EXPECT_HEAD"

$APP "cd $RG && .venv/bin/python experiments/two_site_judging/preflight.py \
  --stage protocol --expect-head $EXPECT_HEAD"
```

Both must end `preflight passed`. A failure means **stop** — not "adapt the
check".

---

## 3. Build the blind packet

Dry-run first. It validates every gate and writes nothing:

```bash
$APP "cd $RG && .venv/bin/python \
  experiments/two_site_judging/build_two_site_judge_packet.py --dry-run"
```

Then build:

```bash
$APP "cd $RG && .venv/bin/python \
  experiments/two_site_judging/build_two_site_judge_packet.py"
```

Expect `axis2 89`, `quality 178`, `safety 213`, `480 call(s) per judge, 960
across both`, and a `B orientation` line whose two numbers add to 178.

Then the full preflight, which now checks all 480 packet rows:

```bash
$APP "cd $RG && .venv/bin/python experiments/two_site_judging/preflight.py \
  --expect-head $EXPECT_HEAD"
```

The packet writes to `$TSJ/judging/packet`. **The private map
`$TSJ/judging/packet/private/row_map.jsonl` is the only record of the A/B
orientation.** If it is lost, the B verdicts cannot be decoded and the 178 B
calls are wasted. Do not delete it, and do not commit it.

---

## 4. Sonnet — 480 billed calls

Dry-run first (renders and hashes, calls nothing):

```bash
$APP "cd $RG && .venv/bin/python experiments/two_site_judging/run_two_site_judge.py \
  --judge sonnet --dry-run"
```

Then run:

```bash
$APP "cd $RG && .venv/bin/python experiments/two_site_judging/run_two_site_judge.py \
  --judge sonnet"
```

One command, three rubric files, 480 calls. It exits non-zero if any row
failed; re-run the identical command to retry only those rows.

---

## 5. Opus — 480 billed calls

Back-to-back with step 4, same shape:

```bash
$APP "cd $RG && .venv/bin/python experiments/two_site_judging/run_two_site_judge.py \
  --judge opus --dry-run"

$APP "cd $RG && .venv/bin/python experiments/two_site_judging/run_two_site_judge.py \
  --judge opus"
```

**960 billed calls are now complete.** Everything below is free.

To run both judges back-to-back in one line, once each dry-run is clean:

```bash
$APP "cd $RG && .venv/bin/python experiments/two_site_judging/run_two_site_judge.py --judge sonnet \
   && .venv/bin/python experiments/two_site_judging/run_two_site_judge.py --judge opus"
```

---

## 6. Join

```bash
$APP "cd $RG && .venv/bin/python \
  experiments/two_site_judging/join_two_site_verdicts.py --dry-run"

$APP "cd $RG && .venv/bin/python \
  experiments/two_site_judging/join_two_site_verdicts.py"
```

**This run:** three of the 960 planned verdicts never arrived, so expect
`sonnet records 479`, `opus records 478`, `cells 480`, and the three states
`agree`, `disagree` and `unresolved_technical 3`. All 480 cells survive; the
three carried cells are decided by nobody. See
`manifests/two_site_actuator_judging_v1_technical_unresolved.json` and the
2026-08-27 entry in `docs/decision_log.md`.

A failure that manifest does not name still stops the join. That is deliberate:
the gate is what keeps a *new* failure from riding in on an old manifest. If it
fires, stop and read the message — do not widen the manifest to fit. The B verdicts are decoded here, through the packet's
private map. Per-cell verdicts and agreement go to
`$TSJ/judging/joined/private/joined_cells.jsonl` and are **not** read by the
review export except to select rows.

---

## 7. Export the blinded human review

```bash
$APP "cd $RG && .venv/bin/python \
  experiments/two_site_judging/export_two_site_human_review.py \
  --procedure $PROCEDURE --dry-run"

$APP "cd $RG && .venv/bin/python \
  experiments/two_site_judging/export_two_site_human_review.py \
  --procedure $PROCEDURE --seed 20260827"
```

Under `corpus_disagreement_only` this covers the cells the two judges split on,
and the count is not knowable before step 6 finishes.

Give the reviewer `$TSJ/judging/human_review/human_review_input.jsonl`,
`reviewer_instructions.json` and `human_review_TEMPLATE.jsonl`. They fill the
template and save it as `human_review_COMPLETED.jsonl`.

Axis-2 and safety rows take one `verdict`. Quality rows take a `preference`
(`A_BETTER` / `B_BETTER` / `EQUIVALENT`) and a `difference` (`MATERIAL` /
`MINOR` / `NONE`) — in the **same A/B space the judges answered in**. The
reviewer is not told, and cannot recover, which reply came from where; the
answer is decoded at step 8.

A blank row is recorded as `UNRESOLVED`, which is a finding. A guess is not.

---

## 8. Adjudicate

```bash
$APP "cd $RG && .venv/bin/python \
  experiments/two_site_judging/import_two_site_human_review.py \
  --procedure $PROCEDURE --dry-run"

$APP "cd $RG && .venv/bin/python \
  experiments/two_site_judging/import_two_site_human_review.py \
  --procedure $PROCEDURE \
  --review-dir $TSJ/judging/human_review \
  --completed $TSJ/judging/human_review/human_review_COMPLETED.jsonl"
```

`--packet` defaults to `$TSJ/judging/packet` and is **required**: the review
packet deliberately carries no orientation, so the reviewer's A/B answers are
decoded through the packet's private map.

Expect `cells 480` and `0 UNRESOLVED`. Any `UNRESOLVED` cell withholds its
class's reading at step 9 — that is the designed behaviour, not a failure to
work around.

---

## 9. The report

```bash
$APP "cd $RG && .venv/bin/python \
  experiments/two_site_judging/report_two_site.py --dry-run"

$APP "cd $RG && .venv/bin/python \
  experiments/two_site_judging/report_two_site.py"
```

Writes `$TSJ/judging/report/two_site_judging_report.json` and prints:

* **E** — substantive repair (SA or PA-S) and full repair (SA), over 89 rows;
* **B** — material and non-material degradation, over 178 rows;
* **H** — YES / NO / UNCLEAR, over 213 rows.

Deterministic: two runs over the same adjudicated file produce identical
readings.

---

## What is committed, and what is not

**Committed:** `manifests/two_site_actuator_judging_v1.json`,
`src/routegate/twositejudging.py`, `experiments/two_site_judging/`,
`tests/test_two_site_judging.py`.

**Never committed:** anything under `$TSJ` or `$CONS` — the frozen rubric text,
the packet, the private maps, every raw API body, the joined cells, the review
packet, the adjudicated cells and the report. All of it is prompt-bearing,
response-bearing or de-anonymising.
