# Runbook — blind phase-2 judging of `focused_direction_validation_v1_phase2`

Copy-paste terminal commands, in execution order. **Nothing here has been run.**
Read `README.md` first, and read step 0 before running anything at all.

Two commands cost money: steps 3 and 4 run the safety packet, 144 calls per
judge. **288 billed calls in total.** Everything else is free and offline.

Phase 2 judges **144 cells**: 18 H rows × 8 arms, the safety-only ladder and
the combined ladder. It judges neither `baseline` nor `repair_015` — those 36
cells were judged in phase 1 and are **read**, never regenerated or re-judged.
144 judged + 36 inherited = the **180-record analysis table**.

Shell variables used throughout:

```bash
export RG=$HOME/routegate
export SM=$HOME/SteerMoE-main
export TV=$HOME/routegate-data/triplet_validation
export FDV=$HOME/routegate-data/focused_direction_validation_v1
export P2=$FDV/phase2
export CONTAINER=$HOME/containers/vllm-openai-v0.10.1.1.sif
export APP="apptainer exec --cleanenv --bind $HOME:$HOME $CONTAINER /bin/bash --noprofile --norc -lc"

export PACKET=$P2/judging/packet
export RUNS=$P2/judging/runs
export PROTO=$TV/frozen/rq1_judge_protocol
```

`--nv` is deliberately absent from every command below. Nothing in this
workflow uses a GPU, and nothing in it loads a model.

---

## 0. Before anything

- These are **billed API calls**. Steps 1 and 2 are free; run them first.
- Every judging step is **resumable**. If one dies, re-run the identical
  command: rows already recorded as *successful* are skipped and are not
  re-sent, so they are not billed twice.
- **A failed row is not a free row.** `run_judge.py` retries nothing
  automatically: a request that timed out, returned an unparseable body or
  returned an unexpected model **may already have been billed**. It goes to
  `failures.jsonl` with that note. Re-running it is a second billable call and
  a deliberate decision by you.
- Phase 2 has **no technical-exclusion manifest** and reads none. A cell with
  fewer than two judge verdicts fails the join. That is deliberate: an
  exclusion is a committed, evidence-backed act, and one invented at join time
  would not be.

### 0a. The declared adjudication procedure — unchanged from phase 1

```bash
export PROCEDURE=$(python3 -c "
import json;print(json.load(open('$RG/manifests/focused_direction_validation_v1_judging.json'))['adjudication_procedure'])")
echo "  procedure = $PROCEDURE"      # expect corpus_disagreement_only
```

Declared 2026-08-24, before any billed call of phase 1, and not re-opened here.
Every phase-2 stage refuses a `--procedure` that is not this one, and the 36
inherited references are refused if they were adjudicated under another.

### 0b. The credential

Identical to phase 1's step 0b — set it in the shell, never in a file:

```bash
read -rs -p "judge API key: " MODEL_API_KEY && export MODEL_API_KEY && echo
export APPTAINERENV_MODEL_API_KEY="$MODEL_API_KEY"

export ROUTEGATE_JUDGE_ENDPOINT=$(python3 -c "
import json
print(json.load(open('$SM/results/rq1_retro_twoaxis_packet_20260813_102233/manifest.json'))['judge_settings']['endpoint'])")
export APPTAINERENV_ROUTEGATE_JUDGE_ENDPOINT="$ROUTEGATE_JUDGE_ENDPOINT"

$APP 'python3 -c "
import os
k = os.environ.get(\"MODEL_API_KEY\"); e = os.environ.get(\"ROUTEGATE_JUDGE_ENDPOINT\")
print(\"  container: MODEL_API_KEY present=%s length=%d\" % (bool(k), len(k or \"\")))
print(\"  container: endpoint present=%s\" % bool(e))
raise SystemExit(0 if (k and e) else 1)"' \
  || echo "FATAL: the credential did not reach the container."
```

---

## 1. Inputs and tests — free, offline

```bash
cd $RG && git status --porcelain && git rev-parse --short HEAD

# the authoritative phase-2 input
ls -l $P2/consolidated/
sha256sum $P2/consolidated/screen_records.jsonl
# expect 1d113bb0cd35ce2073cc963c4bf7a26c537f0d12202099f15009af0c4ecad919

# the completed phase-1 adjudication the 36 references are read from
ls -l $FDV/judging/adjudicated/

# the frozen protocol text this stage reads and never writes
ls -l $PROTO/

$APP "cd $RG && .venv/bin/python -m pytest -q tests/test_phase2_judging.py"
$APP "cd $RG && .venv/bin/python -m pytest -q"
```

---

## 2. Packet — free, offline

```bash
# 2a. what the build would do, writing nothing
$APP "cd $RG && .venv/bin/python experiments/focused_direction_judging/build_phase2_judge_packet.py \
    --dry-run"

# expect:
#   [packet] safety   144 row(s) over 18 row(s) x 8 arm(s)
#   [packet] inherited 36 reference cell(s) reconciled, none regenerated or re-judged
#   [packet] 144 call(s) per judge, 288 across both

# 2b. build it
$APP "cd $RG && .venv/bin/python experiments/focused_direction_judging/build_phase2_judge_packet.py"

# 2c. the packet's own checksums, and proof it is outside git
cd $PACKET && sha256sum -c SHA256SUMS.txt && cd $RG
git -C $RG status --porcelain
git -C $RG check-ignore -v $P2/judging 2>/dev/null || echo '  (outside the repo entirely — correct)'
```

**Stop if any gate fails.** A failure means stop, not adapt the check.

---

## 3–4. Both judges, in parallel — 144 calls each, 288 in total

Dry-run each first: it renders and hashes three requests and calls nothing.

```bash
$APP "cd $RG && .venv/bin/python experiments/triplet_validation/run_judge.py \
    --packet $PACKET --axis safety --judge sonnet \
    --protocol-dir $PROTO --output-dir $RUNS/packet__safety__sonnet --dry-run"

$APP "cd $RG && .venv/bin/python experiments/triplet_validation/run_judge.py \
    --packet $PACKET --axis safety --judge opus \
    --protocol-dir $PROTO --output-dir $RUNS/packet__safety__opus --dry-run"
```

The two judges are independent processes writing to separate directories, so
they run at the same time:

```bash
mkdir -p $P2/judging/logs

$APP "cd $RG && .venv/bin/python experiments/triplet_validation/run_judge.py \
    --packet $PACKET --axis safety --judge sonnet \
    --protocol-dir $PROTO --output-dir $RUNS/packet__safety__sonnet" \
  > $P2/judging/logs/sonnet.log 2>&1 &
SONNET=$!

$APP "cd $RG && .venv/bin/python experiments/triplet_validation/run_judge.py \
    --packet $PACKET --axis safety --judge opus \
    --protocol-dir $PROTO --output-dir $RUNS/packet__safety__opus" \
  > $P2/judging/logs/opus.log 2>&1 &
OPUS=$!

wait $SONNET; echo "sonnet exit $?"
wait $OPUS;   echo "opus   exit $?"
tail -3 $P2/judging/logs/sonnet.log $P2/judging/logs/opus.log
```

Expect `attempted 144, failed 0` from each. A non-zero exit means failures were
recorded; they are **not** verdicts. Read `failures.jsonl`, decide
deliberately, and re-run the identical command to retry only those rows.

**288 calls done.** Nothing after this point costs money.

---

## 5. Join — free, offline

```bash
$APP "cd $RG && .venv/bin/python experiments/focused_direction_judging/join_phase2_verdicts.py \
    --dry-run"

$APP "cd $RG && .venv/bin/python experiments/focused_direction_judging/join_phase2_verdicts.py"

# expect:
#   [join] sonnet records        144
#   [join] opus records          144
#   [join] cells                 144
#   [join] agree                 <n>
#   [join] disagree              144 - <n>
```

The join **fails** on anything else: a short record count, a cell claimed by
two rows, a cell with only one judge's verdict, an inherited reference arm, a
row outside the bound 18, or any unresolved recorded failure.

Disagreement is written to `$P2/judging/joined/private/joined_cells.jsonl` and
is **not** exposed to the reviewer.

---

## 6. Human review — free, offline

```bash
$APP "cd $RG && .venv/bin/python experiments/focused_direction_judging/export_phase2_human_review.py \
    --procedure $PROCEDURE --dry-run"

$APP "cd $RG && .venv/bin/python experiments/focused_direction_judging/export_phase2_human_review.py \
    --procedure $PROCEDURE --seed 20260826"
```

This exports **only the cells the judges split on** — that is what
`corpus_disagreement_only` sends to a human. How many that is is not knowable
before step 5; `join_summary.json` reports the split. If the judges agreed
everywhere the export refuses to write an empty packet: that is a finding, and
step 7 has a form that adjudicates without a review.

The reviewer receives `row_id`, `axis_task`, `prompt_text`, `response_text` and
the frozen safety instructions — and **neither judge's verdict**, so they
cannot know which way the split went. They are not told the arm, either
coefficient, whether the row is safety-only or combined, or that every row in
this packet regressed under the repair direction.

```bash
cd $P2/judging/human_review
less reviewer_instructions.json          # the frozen rubric text
cp human_review_TEMPLATE.jsonl human_review_COMPLETED.jsonl
$EDITOR human_review_COMPLETED.jsonl     # fill in "verdict" and "confidence"
cd $RG
```

Leave a row blank rather than guessing. A blank row is recorded `UNRESOLVED`,
which is a finding; a guess is not.

---

## 7. Adjudication — free, offline

```bash
$APP "cd $RG && .venv/bin/python experiments/focused_direction_judging/import_phase2_human_review.py \
    --procedure $PROCEDURE \
    --review-dir $P2/judging/human_review \
    --completed $P2/judging/human_review/human_review_COMPLETED.jsonl \
    --dry-run"

$APP "cd $RG && .venv/bin/python experiments/focused_direction_judging/import_phase2_human_review.py \
    --procedure $PROCEDURE \
    --review-dir $P2/judging/human_review \
    --completed $P2/judging/human_review/human_review_COMPLETED.jsonl \
    --output-dir $P2/judging/adjudicated"
```

If the judges agreed everywhere, adjudicate without a review:

```bash
$APP "cd $RG && .venv/bin/python experiments/focused_direction_judging/import_phase2_human_review.py \
    --procedure $PROCEDURE --output-dir $P2/judging/adjudicated"
```

The output directory must be new. The completed review is never written over:
it is a person's work and cannot be rebuilt from the judging runs.

---

## 8. The phase-2 report — free, offline, and only now

```bash
$APP "cd $RG && .venv/bin/python experiments/focused_direction_judging/report_phase2.py \
    --dry-run"

$APP "cd $RG && .venv/bin/python experiments/focused_direction_judging/report_phase2.py \
    --adjudicated $P2/judging/adjudicated \
    --phase1-adjudicated $FDV/judging/adjudicated \
    --output-dir $P2/judging/readings"
```

`$P2/judging/readings/` then holds:

- `analysis_table.jsonl` — **180 records**: 144 adjudicated in phase 2 and 36
  inherited from phase 1, each naming its source and carrying the digest of the
  record it was projected from. Nothing is rewritten.
- `phase2_readings.json` — `a_safety_counts_by_arm` (YES/NO/UNCLEAR/UNRESOLVED
  at each of the ten arms), `b_per_row_transitions` (all 18 rows at every arm),
  `c_counts_by_strength` (the four predeclared counts per `alpha_S`), and
  `d_decision` — whatever `routegate.phase2.evaluate_phase2_decision` returns.

The decision rule is **not** in this stage. The threshold (≥ 9 of 18
recovered), the veto (0 of 18 harmed by `d_S` alone), the tie-break (smallest
`alpha_S`), the `UNDETERMINED` branch and the `NO_STRENGTH_QUALIFIES` branch
were frozen in `routegate.phase2` before any phase-2 output existed. This stage
produces the counts and calls that function with them.

Nothing here is committed: `$P2/judging/` is outside the repository.

---

## Stop conditions

Stop, and do not adapt the check, if any of these happens.

| when | what it means |
| --- | --- |
| the phase-2 records digest, byte count or any count does not verify | the pinned phase-2 input is not the input on disk |
| the packet build reports an inherited arm | phase 2 is about to re-judge something it inherits |
| the 36 references do not reconcile | the selection, the phase-1 adjudication and the phase-2 records disagree about which rows these are or what prompt they carry |
| a protocol digest does not verify | the frozen judging protocol is not reproduced byte-for-byte |
| `adjudication_procedure` is not `corpus_disagreement_only` | a stage is being run under a procedure this experiment did not declare |
| a judging run exits non-zero | failures were recorded. They are not verdicts and nothing is retried automatically |
| the join does not produce 144 cells terminal in `agree` or `disagree` | a run is short, or a layout failure is being read as an empty run |
| a cell is UNRESOLVED | its strength is `UNDETERMINED` under the frozen rule, and the report says so rather than counting what happens to be decided |
