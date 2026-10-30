# Runbook — `rotation_safety_direction_v2`

> **Historical runbook.** Written before the screen ran; it was later run and judged.
> See the [README](README.md) status note.

Every command below is written for the Apptainer image the rest of this study
runs in. `.venv` only resolves inside it.

```bash
CONTAINER="$HOME/containers/vllm-openai-v0.10.1.1.sif"
BANK="$HOME/routegate-data/frozen/validated_direction_bank"
ROOT="$HOME/routegate-data/rotation_safety_direction_v2"
IN() { apptainer exec --cleanenv --bind "$HOME:$HOME" "$CONTAINER" \
         /bin/bash --noprofile --norc -lc "cd $HOME/routegate && $*"; }
```

## 0. State

The predeclaration is written and committed. `generations_run` is 0. Nothing
below step 3 has been run, and step 3 is a GPU job that is **not submitted from
here**.

## 1. Freeze — already done, and re-runnable

```bash
IN ".venv/bin/python experiments/rotation_safety_direction/freeze_rotation_screen.py --dry-run"
IN ".venv/bin/python experiments/rotation_safety_direction/freeze_rotation_screen.py"
```

Writes `manifests/rotation_safety_direction_v2.json`. (v1 is superseded and preserved unedited; see the README.) Re-running on an
unchanged tree writes nothing and re-verifies what is there. If the package has
moved since the freeze the script **stops** rather than refreshing the file:
that difference is either a decision to record or a drift, and both need a
human. Re-freezing means removing the file by hand, with a reason.

## 2. Preflight — everything that can fail without a GPU

```bash
IN ".venv/bin/python experiments/rotation_safety_direction/preflight.py \
      --direction-bank '$BANK'"
```

39 checks: the 22 completed artifacts still hash to what they did; the
predeclaration digests itself and matches the package; the phase-2 selection is
the one this screen is bound to; the configuration and the package declare the
same eight arms; the bank is the pinned one and its own manifest records
`d_S = H − E` with no orthogonalisation; the three per-layer `H − E` digests are
the pinned ones and are distinct; the inventory is 18 × 8; the shards partition
it exactly at every count tried; the operator holds its identities over eight
magnitude decades and refuses all five degenerate inputs; and no path this
stage can reach names held-out material.

Add `--write-report <path.json>` to keep the report. It carries no prompt or
response text.

## 3. Generate — 144 responses, on a GPU, not from here

```bash
# what it would do, loading nothing and creating no output directory
IN ".venv/bin/python experiments/rotation_safety_direction/run_rotation_screen.py \
      --direction-bank '$BANK' --output-dir '$ROOT/shard_1' --dry-run"

# the real thing, as a six-task array
qsub experiments/rotation_safety_direction/submit_rotation_screen.sh
```

Before `qsub`: paste the commit that carries this stage into `EXPECT_HEAD` in
the submit script. It is empty on purpose, and the array refuses to start until
it is set. All six tasks share one working tree, so they must also share one
commit, and the tree must be clean.

Six tasks × 24 generations = 144. Six divides the 18 rows exactly, so each task
gets 3 whole rows and no row's prompt states are captured twice.

Re-running a task resumes it. The seven-field key — row, class, arm, `β`,
`α_R`, the bank digest and the generation-configuration digest — skips what is
already recorded, and a mismatched bank, corpus, model, tokenizer, chat
template, predeclaration digest or tolerance version stops the job rather than
appending rows that are not comparable to the ones already there.

Do **not** pass `--only-arm`, `--only-row-id` or `--limit` for the real run.
Each marks every record it writes as a smoke test, and the consolidator refuses
to treat such a shard as the screen.

### 3b. All 144 in one unsharded interactive session

One process, one output directory, `--shard-count 1`. Slower than the array —
144 greedy generations at `max_new_tokens=768` back to back — but it produces
the complete screen in a single directory, and its run manifest records
`complete_screen: true` because it is unfiltered, unsharded and unlimited.

Request the session and run the screen in it, in one command:

```bash
qrsh -pty y -now no -wd $HOME/routegate   -l gpu=4,mem=96G,h_rt=24:00:00,tmpfs=30G -ac allow=L   apptainer exec --nv --cleanenv --bind "$HOME:$HOME"     "$HOME/containers/vllm-openai-v0.10.1.1.sif"     /bin/bash --noprofile --norc -lc     'cd $HOME/routegate && .venv/bin/python        experiments/rotation_safety_direction/run_rotation_screen.py        --direction-bank "$HOME/routegate-data/frozen/validated_direction_bank"        --expect-bank-sha256 d759574b5be630ea5170fb440f8663d8c4bfaa32b479ce5cf641b6dbd73516bb        --output-dir "$HOME/routegate-data/rotation_safety_direction_v2/unsharded"        --shard-index 1 --shard-count 1 --device auto --resume'
```

`--resume` is safe on a first run — the directory is absent, nothing is
skipped — and it is what lets the same command be re-issued if the session
drops partway. Consolidate an unsharded run by pointing the consolidator at a
root holding it as `shard_1` with `--shard-count 1`.

## 4. Consolidate

```bash
IN ".venv/bin/python experiments/rotation_safety_direction/consolidate_rotation_screen.py \
      --shards-root '$ROOT' --output-dir '$ROOT/consolidated' \
      --shard-count 6 --dry-run"

IN ".venv/bin/python experiments/rotation_safety_direction/consolidate_rotation_screen.py \
      --shards-root '$ROOT' --output-dir '$ROOT/consolidated' --shard-count 6"
```

Refuses: a missing or unplanned shard, shards that disagree on any run-locked
field, a dry-run or smoke shard, a missing or duplicated `(row, arm)` cell, a
record carrying `baseline` or `repair_015`, a response whose own digest does not
recompute, a per-state block that is short a field or at the wrong state, a
record whose rotation broke one of the predeclared invariants, and a record
that used one layer's `H − E` vector at more than one layer.

The output order is deterministic — row id, then the arm's declared position —
so two consolidations of the same responses produce the same bytes and the
recorded `records_sha256` identifies the data rather than the scheduler.

## 5. Judge — the common engine, driven by a manifest

No judge runs in steps 1–4. The screen is judged by the **common judging
framework** (`experiments/judging/`, `README.md` there first), which is driven
by `manifests/rotation_safety_direction_v2_judging_v1.json`. The frozen
`rq3_safety_actionable_harmful_help_v1` rubric and the
`corpus_disagreement_only` procedure are reused by digest and by name, blind to
arm, `β`, direction, layer and to whether a row is baseline or steered.

144 cells are judged here. `baseline` and `repair_015` for these same 18 rows
were judged in phase 1 and are **inherited**: 36 cells that no judge sees
again. 144 + 36 = 180 in the analysis table, and 288 billed calls, not 360.

```bash
MF="$HOME/routegate/manifests/rotation_safety_direction_v2_judging_v1.json"
```

### 5a. Adapt the screen into the engine's input shape (free)

```bash
IN ".venv/bin/python experiments/rotation_safety_direction/build_judging_inputs.py --dry-run"
IN ".venv/bin/python experiments/rotation_safety_direction/build_judging_inputs.py"
```

Reads the consolidated package by digest and writes one file,
`<rotation_judging_dir>/inputs/candidate_records.jsonl`. The prompts are
resolved from the accepted corpus and checked against the digest each record
already carries; the consolidation itself is never written to. The output is
deterministic and the manifest pins it, so if the printed SHA256 differs from
`inputs.candidate_records.sha256` something upstream has moved — **stop**.

### 5b. Preflight, packet, dry runs (free)

```bash
IN ".venv/bin/python experiments/judging/run_protocol.py --manifest $MF --stage preflight"
IN ".venv/bin/python experiments/judging/run_protocol.py --manifest $MF --stage build --dry-run"
IN ".venv/bin/python experiments/judging/run_protocol.py --manifest $MF --stage build"
IN ".venv/bin/python experiments/judging/run_protocol.py --manifest $MF --stage dryrun"
IN ".venv/bin/python experiments/judging/run_protocol.py --manifest $MF --stage judge --judge sonnet --dry-run"
IN ".venv/bin/python experiments/judging/run_protocol.py --manifest $MF --stage judge --judge opus --dry-run"
```

The packet is built **once**. After a judge has run, a rebuild is refused.

### 5c. The two billed runs

Set the credential in the shell, never in a file:

```bash
read -rs -p "judge API key: " MODEL_API_KEY && export MODEL_API_KEY && echo
export APPTAINERENV_MODEL_API_KEY="$MODEL_API_KEY"
export ROUTEGATE_JUDGE_ENDPOINT="<the gateway endpoint>"
export APPTAINERENV_ROUTEGATE_JUDGE_ENDPOINT="$ROUTEGATE_JUDGE_ENDPOINT"
```

Back to back, 144 calls each:

```bash
IN ".venv/bin/python experiments/judging/run_protocol.py --manifest $MF --stage judge --judge sonnet \
    && .venv/bin/python experiments/judging/run_protocol.py --manifest $MF --stage judge --judge opus"
```

Resumable: re-running the identical command skips rows already recorded as
successful and never re-sends them. A failed row is **not** retried
automatically — it may already have been billed.

### 5d. Join, review, adjudicate (free)

```bash
IN ".venv/bin/python experiments/judging/run_protocol.py --manifest $MF --stage join --dry-run"
IN ".venv/bin/python experiments/judging/run_protocol.py --manifest $MF --stage join"
IN ".venv/bin/python experiments/judging/run_protocol.py --manifest $MF --stage export --dry-run"
IN ".venv/bin/python experiments/judging/run_protocol.py --manifest $MF --stage export"
IN ".venv/bin/python experiments/judging/run_protocol.py --manifest $MF --stage import --dry-run"
IN ".venv/bin/python experiments/judging/run_protocol.py --manifest $MF --stage import"
```

`export` carries genuine Sonnet–Opus disagreements only. A technically
unresolved cell is never exported: a human adjudicates a disagreement between
two judges, and does not supply a verdict a judge never gave. A blank answer
stays `UNRESOLVED`, which is a finding; a guess is not.

If a technical failure occurs, the join **stops** until it is named in a
technical-unresolved manifest passed with `--carried`. That gate is the point;
widening it is not the fix.

### 5e. The report, and the decision

```bash
IN ".venv/bin/python experiments/judging/run_protocol.py --manifest $MF --stage report"
IN ".venv/bin/python experiments/rotation_safety_direction/report_rotation_judging.py --dry-run"
IN ".venv/bin/python experiments/rotation_safety_direction/report_rotation_judging.py"
```

The engine's `report` emits verdict tallies with explicit denominators. The
second command is this experiment's own reading: it assembles the 180-cell
table (144 adjudicated here + 36 inherited, reconciled against phase 1's own
file by digest), derives the four counts per `β`, and hands them to
`routegate.rotationfreeze.evaluate_rotation_decision`. That function decides
nothing itself: it translates into phase 2's vocabulary, calls
`routegate.phase2.evaluate_phase2_decision`, and relabels the answer. The
outcome is one of `STRENGTH_SELECTED`, `NO_STRENGTH_QUALIFIES` or
`UNDETERMINED` — phase 2's own branches, applied to this screen.

Report the realised `θ`, the traversed `φ` and the induced edit norm beside
whatever the rule says. `β` is a fraction of a per-row, per-layer angle and not
a dose; without those recorded quantities there is no basis for comparing the
size of this intervention with the additive one.

## Tests

```bash
IN ".venv/bin/python -m pytest tests/test_rotation_operator.py tests/test_rotation_screen.py tests/test_rotation_judging.py -q"
IN ".venv/bin/python -m pytest -q"
```

## What outputs go where

Everything the run produces lands under `$ROOT`, outside git, per the
repository's output policy. Nothing under it is committed. What is committed is
the code, the configuration, the predeclaration, and — later — a closure
recording what the decision rule said.
