# Runbook — retrospective validation of the ten operative triplets

Copy-paste terminal commands, in execution order. **Nothing here has been run.**
Read `README.md` first; read step 0 before running anything at all.

Two commands cost money: step 3 and step 4 run the original packet (194 calls),
step 6 and step 7 run the supplement (100 calls). Worst case at the packet's own
estimate is roughly **USD 19**. Everything else is free and offline.

Shell variables used throughout:

```bash
export RG=$HOME/routegate
export SM=$HOME/SteerMoE-main
export TV=$HOME/routegate-data/triplet_validation
export CONTAINER=$HOME/containers/vllm-openai-v0.10.1.1.sif
export APP="apptainer exec --cleanenv --bind $HOME:$HOME $CONTAINER /bin/bash --noprofile --norc -lc"

# The reviewed commit this run is pinned to. Set it to the commit that
# carries the reviewed triplet-validation implementation; until that commit
# exists, use the base commit df93dd7 and expect the clean-tree gate to fail
# while the work is still uncommitted.
export EXPECT_HEAD=df93dd7
```

`--nv` is deliberately absent from every command below. Nothing in this workflow
uses a GPU.

---

## 0. Before anything

- These are **billed API calls**. Run step 1 and step 2 first; they are free.
- Every judging step is **resumable**. If one dies, re-run the identical
  command: rows already recorded as *successful* are skipped and are not
  re-sent, so they are not billed twice.
- **A failed row is not a free row.** Neither runner retries automatically,
  and both say why: a request that timed out, returned an unparseable body,
  returned an unexpected model, or failed after the response arrived **may
  already have been billed**. The historical runner writes it to
  `unresolved.jsonl` with `billing_note: "this request may already have been
  billed; it is recorded unresolved rather than re-sent"`; `run_judge.py`
  writes the same note to `failures.jsonl`. Re-running a failed row is a
  second billable call, and it is a deliberate decision by the operator, not
  something the tooling does on its own.
- Every attempt is preserved — the successes, the failures, and the raw body
  of anything that could not be parsed. Nothing is dropped to make a run look
  clean, and the cost estimate below is a floor, not a cap: retried failures
  add to it.
- Set the credential in the shell, never in a file:

```bash
# The key. MODEL_API_KEY is the name the historical runner reads
# (rq3_judge_axis_smoke.API_KEY_ENV), so it is the name that must be set --
# $SM/rq1_retrospective_judge.py will not find it under any other name.
read -rs -p "judge API key: " MODEL_API_KEY && export MODEL_API_KEY && echo

# $APP uses --cleanenv, which drops the entire host environment. A variable
# exported here is therefore NOT visible inside the container. Apptainer
# forwards APPTAINERENV_X into the container as X, which is the only
# propagation that both survives --cleanenv and keeps the value off every
# command line and out of every process listing.
export APPTAINERENV_MODEL_API_KEY="$MODEL_API_KEY"

# the endpoint: taken from the frozen packet's own manifest, not typed by hand
export ROUTEGATE_JUDGE_ENDPOINT=$(python3 -c "
import json,sys
print(json.load(open('$SM/results/rq1_retro_twoaxis_packet_20260813_102233/manifest.json'))['judge_settings']['endpoint'])")
export APPTAINERENV_ROUTEGATE_JUDGE_ENDPOINT="$ROUTEGATE_JUDGE_ENDPOINT"

# validate both on the host before going further
[ -n "$MODEL_API_KEY" ] || { echo 'FATAL: no key set'; exit 1; }
case "$ROUTEGATE_JUDGE_ENDPOINT" in
  https://*) echo "  endpoint = $ROUTEGATE_JUDGE_ENDPOINT" ;;
  *) echo "FATAL: endpoint did not resolve from the packet manifest"; exit 1 ;;
esac
echo "  host: key present, length ${#MODEL_API_KEY} (value never printed)"
```

Then prove it actually arrived **inside** the container, because a key that is
set on the host and absent in the container fails at the first paid call, not
before it:

```bash
# presence and length only. The value is never printed, never written to a
# file, and never passed as an argument.
$APP 'python3 -c "
import os
k = os.environ.get(\"MODEL_API_KEY\")
e = os.environ.get(\"ROUTEGATE_JUDGE_ENDPOINT\")
print(\"  container: MODEL_API_KEY present=%s length=%d\" % (bool(k), len(k or \"\")))
print(\"  container: endpoint present=%s\" % bool(e))
raise SystemExit(0 if (k and e) else 1)"' \
  || { echo "FATAL: the credential did not reach the container. Check that"; \
       echo "       APPTAINERENV_MODEL_API_KEY is exported in THIS shell."; }
```

If that check fails, stop. Do not work around it by dropping `--cleanenv`
(which would import the whole host environment into a paid run) and do not
place the value on a command line — a command line is visible in `ps` to every
user on this shared login node.

The key is never written to a file, never passed on a command line, never
recorded in a run manifest, and never committed. `run_judge.py` records only
whether one was present, which variable it came from, and its length.

---

## 1. Environment verification

```bash
cd $RG && git status --porcelain && git rev-parse --short HEAD

$APP "cd $RG && .venv/bin/python -c \"
import sys, numpy, sklearn; print(sys.version.split()[0], numpy.__version__, sklearn.__version__)\""

# no GPU is needed, and none should be claimed
ls /dev/nvidia* 2>/dev/null || echo 'no GPU on this host — correct for this workflow'

# the historical workspace must be untouched and must stay that way
ls -d $SM/.git 2>/dev/null || echo 'SteerMoE-main is not a git repo — expected'
```

---

## 2. Frozen-input validation, and the one-time copy

Verify the original packet still matches its own checksums, **then** copy the
protocol text and the frozen sources into the working root. The copy reads
`$SM`; it never writes there.

```bash
# 2a. the original packet, unchanged
cd $SM/results/rq1_retro_twoaxis_packet_20260813_102233 && sha256sum -c SHA256SUMS.txt

# 2b. the working root
mkdir -p $TV/frozen/rq1_judge_protocol $TV/historical_packet

# 2c. the protocol text, extracted from the frozen module by AST, never executed
$APP "cd $SM && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
import ast, json, hashlib, pathlib
src = pathlib.Path('rq3_judge_axis_smoke.py').read_text(encoding='utf-8')
vals = {}
for node in ast.parse(src).body:
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
            if isinstance(t, ast.Name):
                try: vals[t.id] = ast.literal_eval(node.value)
                except Exception: pass
out = pathlib.Path('$TV/frozen/rq1_judge_protocol')
for axis, ins, tpl, sch in (
        ('axis1','AXIS1_INSTRUCTIONS','AXIS1_USER_TEMPLATE','AXIS1_SCHEMA'),
        ('axis2','AXIS2_INSTRUCTIONS','AXIS2_USER_TEMPLATE','AXIS2_SCHEMA'),
        ('safety','SAFETY_INSTRUCTIONS','SAFETY_USER_TEMPLATE','SAFETY_SCHEMA')):
    (out/f'{axis}_instructions.txt').write_text(vals[ins], encoding='utf-8')
    (out/f'{axis}_user_template.txt').write_text(vals[tpl], encoding='utf-8')
    (out/f'{axis}_schema.json').write_text(json.dumps(vals[sch]), encoding='utf-8')
print('protocol text written')
PY"

# 2d. the four frozen sources the supplement reads
cp $SM/results/rq2_final_method_freeze_20260721_150015/source_case_manifest.csv $TV/frozen/
cp $SM/results/rq2_multidonor_capture_merged_20260721_004223/outputs.jsonl \
   $TV/frozen/multidonor_outputs.jsonl
cp $SM/data/pathway_probe_cases.json $SM/data/rq2_multidonor_probe_cases.json $TV/frozen/
chmod 0444 $TV/frozen/*.csv $TV/frozen/*.json $TV/frozen/*.jsonl \
           $TV/frozen/rq1_judge_protocol/*

# 2e. the H baseline responses, taken from the packet's own occurrence map
$APP "cd $RG && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
import json, hashlib, pathlib, csv
occ = [json.loads(l) for l in open('$SM/results/rq1_retro_twoaxis_packet_20260813_102233/private/occurrences.jsonl') if l.strip()]
scm = {r['case_id']: r for r in csv.DictReader(open('$TV/frozen/source_case_manifest.csv'))}
sh = lambda t: hashlib.sha256(t.encode('utf-8')).hexdigest()
out = {}
for o in occ:
    if o['condition'] == 'baseline' and o['role'] == 'harmful_base':
        assert o['prompt_sha256'] == scm[o['case_id']]['harmful_contrast_sha256']
        out[o['case_id']] = {'prompt_sha256': o['prompt_sha256'],
                             'response_text': o['response_text'],
                             'response_sha256': sh(o['response_text'])}
assert len(out) == 10, len(out)
pathlib.Path('$TV/frozen/harmful_baselines.json').write_text(json.dumps(out), encoding='utf-8')
print('10 harmful baselines reconciled')
PY"

# 2f. the protocol gate must now pass, offline
$APP "cd $RG && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c \"
import sys; sys.path.insert(0,'src')
from routegate.triplets import load_protocol
for a in ('axis1','axis2','safety'):
    p = load_protocol(a)
    print(a, 'OK', p.schema['name'])\""
```

---

### 2g. Names, defined once and validated here

Every later step refers to these. There is no placeholder anywhere below that is
not defined and checked in this block.

```bash
# the frozen original packet — byte-identical, never rebuilt, never written into
export PACKET_ORIG=$SM/results/rq1_retro_twoaxis_packet_20260813_102233

# the exact historical runner
export RUNNER=$SM/rq1_retrospective_judge.py

# where every judging run is written. NOT $SM: the runner's --out-base defaults
# to ./results, and letting it default would write into SteerMoE-main.
export OUT_BASE=$TV/judging

# the four original-packet run names. Fixed, not timestamped, so the paths below
# are exact and the joiner finds them by name.
export RUN_A1_SONNET=historical_packet__axis1__sonnet
export RUN_A2_SONNET=historical_packet__axis2__sonnet
export RUN_A1_OPUS=historical_packet__axis1__opus
export RUN_A2_OPUS=historical_packet__axis2__opus

# validate the block before using it
test -d "$PACKET_ORIG" || { echo "FATAL: packet missing"; return 1 2>/dev/null || exit 1; }
test -f "$RUNNER"      || { echo "FATAL: runner missing"; return 1 2>/dev/null || exit 1; }
mkdir -p "$OUT_BASE"
test -w "$OUT_BASE"    || { echo "FATAL: $OUT_BASE not writable"; return 1 2>/dev/null || exit 1; }
for V in PACKET_ORIG RUNNER OUT_BASE RUN_A1_SONNET RUN_A2_SONNET RUN_A1_OPUS RUN_A2_OPUS; do
  [ -n "${!V}" ] || { echo "FATAL: $V is empty"; return 1 2>/dev/null || exit 1; }
  echo "  $V = ${!V}"
done
```

For each `$N` in the four names above, the runner creates
`$OUT_BASE/$N.inprogress` while running and finalises it to `$OUT_BASE/$N`.
`rq1_retro_common.open_run_dir` **refuses** to start if either already exists — completed runs are immutable and a staged run is never
silently reused. That is the collision refusal; you do not have to add one.

---

## 3. Run the ORIGINAL packet, unchanged, with Sonnet

The historical runner, the historical packet, RouteGate's output directory.
Nothing is rebuilt and nothing is written into `$SM`.

```bash
# 3a. preflight — free, and it must pass before any billed call.
#     --original points at the REAL frozen packet in $SM, which is what the
#     runner will read. $TV/historical_packet holds only the verified local
#     copy of private/ (step 5); gating that copy would gate the wrong bytes.
$APP "cd $RG && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/triplet_validation/preflight.py --expect-head $EXPECT_HEAD \
  --original $PACKET_ORIG --stage original"

# 3b. the packet is still byte-identical
cd $PACKET_ORIG && sha256sum -c SHA256SUMS.txt && cd - >/dev/null

# 3c. validate and dry-run — both free, neither writes a result
$APP "cd $SM && .venv/bin/python $RUNNER --packet $PACKET_ORIG \
  --axis axis1 --judge sonnet --validate"
$APP "cd $SM && .venv/bin/python $RUNNER --packet $PACKET_ORIG \
  --axis axis1 --judge sonnet --dry-run"

# 3d. the real calls: 30 Axis-1 rows, then 67 Axis-2 rows
$APP "cd $SM && .venv/bin/python $RUNNER --packet $PACKET_ORIG \
  --axis axis1 --judge sonnet --run --out-base $OUT_BASE --name $RUN_A1_SONNET"

$APP "cd $SM && .venv/bin/python $RUNNER --packet $PACKET_ORIG \
  --axis axis2 --judge sonnet --run --out-base $OUT_BASE --name $RUN_A2_SONNET"
```

**Resumability.** If a pass dies, `$OUT_BASE/$N.inprogress` survives and the
finalised `$OUT_BASE/$N` does not exist. Detect and resume with:

```bash
for N in $RUN_A1_SONNET $RUN_A2_SONNET; do
  if   [ -d "$OUT_BASE/$N" ];             then echo "$N: COMPLETE"
  elif [ -d "$OUT_BASE/$N.inprogress" ];  then echo "$N: RESUMABLE — rerun with --resume $OUT_BASE/$N.inprogress"
  else                                         echo "$N: NOT STARTED"; fi
done
```

Resume by repeating the identical command with `--resume "$OUT_BASE/$N.inprogress"`
added. Rows already recorded as successful are skipped and are not re-sent.

Rows in `unresolved.jsonl` are **not** retried by `--resume`, by design: each
one may already have been billed, and the payload of anything that failed
*after* the response arrived is preserved in the record. Read them before
deciding. Re-sending one is a new billable call — to do it, remove that row
from `unresolved.jsonl` deliberately, and record that you did.

---

## 4. Run the ORIGINAL packet, unchanged, with Opus

Same packet, same runner, same output base, `--judge opus`. Both judges see
every row; there is no sampling.

```bash
# 4a. preflight again — the tree may have changed since step 3
$APP "cd $RG && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/triplet_validation/preflight.py --expect-head $EXPECT_HEAD \
  --original $PACKET_ORIG --stage original"

# 4b. the calls
$APP "cd $SM && .venv/bin/python $RUNNER --packet $PACKET_ORIG \
  --axis axis1 --judge opus --run --out-base $OUT_BASE --name $RUN_A1_OPUS"

$APP "cd $SM && .venv/bin/python $RUNNER --packet $PACKET_ORIG \
  --axis axis2 --judge opus --run --out-base $OUT_BASE --name $RUN_A2_OPUS"
```

### 4c. Exact output locations

| run | directory while running | directory when complete |
| --- | --- | --- |
| Axis 1 / Sonnet | `$OUT_BASE/historical_packet__axis1__sonnet.inprogress` | `$OUT_BASE/historical_packet__axis1__sonnet` |
| Axis 2 / Sonnet | `$OUT_BASE/historical_packet__axis2__sonnet.inprogress` | `$OUT_BASE/historical_packet__axis2__sonnet` |
| Axis 1 / Opus | `$OUT_BASE/historical_packet__axis1__opus.inprogress` | `$OUT_BASE/historical_packet__axis1__opus` |
| Axis 2 / Opus | `$OUT_BASE/historical_packet__axis2__opus.inprogress` | `$OUT_BASE/historical_packet__axis2__opus` |

### 4d. Copy versus move

**Neither.** `--out-base $OUT_BASE` writes the runs directly where the joiner
reads them, so no result directory is ever copied or moved, and nothing lands in
`$SM`. The single copy in this whole step is the packet's private row maps, and
it is a **copy**, never a move:

```bash
mkdir -p $TV/historical_packet
# refuse a collision rather than overwrite
if [ -e $TV/historical_packet/private ]; then
  echo "FATAL: $TV/historical_packet/private already exists — inspect it, do not overwrite"
else
  cp -r --no-clobber $PACKET_ORIG/private $TV/historical_packet/private
  chmod -R a-w $TV/historical_packet/private
fi
```

### 4e. Source and destination verification

The source packet must be unchanged **after** the runs, and the copy must match
it byte-for-byte:

```bash
# the source is still exactly the frozen packet
cd $PACKET_ORIG && sha256sum -c SHA256SUMS.txt && cd - >/dev/null

# the copy matches the source, file by file
for F in axis1_row_map.jsonl axis2_row_map.jsonl occurrences.jsonl; do
  A=$(sha256sum "$PACKET_ORIG/private/$F" | cut -d" " -f1)
  B=$(sha256sum "$TV/historical_packet/private/$F" | cut -d" " -f1)
  [ "$A" = "$B" ] && echo "  OK   $F  $A" || { echo "  MISMATCH $F"; exit 1; }
done
```

### 4f. Expected inventory and counts

Each of the four run directories must contain:

```bash
for N in $RUN_A1_SONNET $RUN_A2_SONNET $RUN_A1_OPUS $RUN_A2_OPUS; do
  D=$OUT_BASE/$N
  echo "== $N"
  ls -1 "$D"
  for F in results.jsonl config.json validation.json SHA256SUMS.txt; do
    test -f "$D/$F" || echo "  MISSING $F"
  done
  echo "  successful rows: $(grep -c '\"status\": \"OK\"' "$D/results.jsonl" 2>/dev/null || echo 0)"
  (cd "$D" && sha256sum -c SHA256SUMS.txt >/dev/null && echo "  checksums OK")
done
```

Expected successful-row counts: **30** for either `axis1` run, **67** for either
`axis2` run. **97 per judge, 194 across both.** Anything less means an
unfinished pass — resume it; do not proceed.

### 4g. Refusing the wrong model or a drifted protocol

Each run writes its own `config.json`. Verify both before trusting any verdict:

```bash
for N in $RUN_A1_SONNET $RUN_A2_SONNET $RUN_A1_OPUS $RUN_A2_OPUS; do
  $APP "cd $RG && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<PYEOF
import json, sys, pathlib
cfg = json.loads(pathlib.Path('$OUT_BASE/$N/config.json').read_text())
frozen = json.loads(pathlib.Path('manifests/rq1_judge_protocol.json').read_text())
want_model = {'sonnet': frozen['judges']['primary']['model'],
              'opus':   frozen['judges']['secondary']['model']}['$N'.rsplit('__',1)[1]]
got = cfg.get('requested_model')
assert got == want_model, f'REFUSED: {got!r} is not the frozen {want_model!r}'
axis = '$N'.rsplit('__',2)[1]
for k, v in (cfg.get('prompt_hashes') or cfg.get('frozen_judge_hashes') or {}).items():
    if k in frozen['protocol_hashes']:
        assert v == frozen['protocol_hashes'][k], f'REFUSED: {k} drifted'
print('  $N: model and protocol hashes match the frozen configuration')
PYEOF"
done
```

A mismatch is a **stop**: the verdicts in that directory were produced under a
different model or a different protocol and are not the evidence this validation
is defined on.

---

## 5. Build the supplemental packet

Free, offline, generates nothing.

```bash
$APP "cd $RG && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/triplet_validation/build_supplemental_packet.py \
  --harmful-baselines $TV/frozen/harmful_baselines.json --dry-run"

$APP "cd $RG && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/triplet_validation/build_supplemental_packet.py \
  --harmful-baselines $TV/frozen/harmful_baselines.json"

cd $TV/supplemental_packet && sha256sum -c SHA256SUMS.txt
```

Expect 20 Axis-1 rows, 20 Axis-2 rows, 10 safety rows: 50 calls per judge.

---

## 6. Run the supplement with Sonnet

```bash
# 6a. preflight at the supplement stage. Both packets are required now: the
#     supplement only means anything against the original it completes, and
#     the opaque-id collision check spans the two.
$APP "cd $RG && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/triplet_validation/preflight.py --expect-head $EXPECT_HEAD \
  --original $PACKET_ORIG --supplement $TV/supplemental_packet \
  --stage supplement"

for AXIS in axis1 axis2 safety; do
  $APP "cd $RG && .venv/bin/python experiments/triplet_validation/run_judge.py \
    --packet $TV/supplemental_packet --axis \$AXIS --judge sonnet --dry-run"
done

for AXIS in axis1 axis2 safety; do
  # MODEL_API_KEY and ROUTEGATE_JUDGE_ENDPOINT arrive through
  # APPTAINERENV_*; nothing is re-assigned on the command line.
  $APP "cd $RG && \
    .venv/bin/python experiments/triplet_validation/run_judge.py \
    --packet $TV/supplemental_packet --axis \$AXIS --judge sonnet"
done
```

---

## 7. Run the supplement with Opus

```bash
for AXIS in axis1 axis2 safety; do
  # MODEL_API_KEY and ROUTEGATE_JUDGE_ENDPOINT arrive through
  # APPTAINERENV_*; nothing is re-assigned on the command line.
  $APP "cd $RG && \
    .venv/bin/python experiments/triplet_validation/run_judge.py \
    --packet $TV/supplemental_packet --axis \$AXIS --judge opus"
done
```

---

## 8. Join

```bash
$APP "cd $RG && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/triplet_validation/join_verdicts.py --dry-run"

$APP "cd $RG && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/triplet_validation/join_verdicts.py"

cat $TV/joined/join_summary.json
```

Expect **110 cells** — 60 from the historical packet's separate-occurrences
layout and 50 from the supplement's inline one — over **10 cases**, joined from
**147** successful records per judge, with every cell in `agree` or
`disagree`. The joiner checks all of that itself and fails the run on anything
else; a short join is refused rather than reported as a smaller result. Any
`missing` or `incomplete` cell means a judging pass is unfinished — go back to
the step that owns it. Do not proceed with gaps.

The historical `rq1_retrospective_join.py` also supports `--also-packet` and can
be run as an independent cross-check; it is not required.

---

## 9. Export the blinded human review

```bash
$APP "cd $RG && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/triplet_validation/export_human_review.py --seed 20260821"

wc -l $TV/human_review/human_review_input.jsonl
cat $TV/human_review/human_review_manifest.json
```

The reviewer fills in `human_review_TEMPLATE.jsonl`: one `verdict` per row from
that axis's vocabulary, an optional `confidence`, an optional `note`. They see
no judge verdict, no agreement flag, no case, no arm and no condition. Do not
show them `$TV/joined/private/`.

### 9b. The second pass: review only what is left

The supplement's 50 cells were reviewed and adjudicated before the joiner read
the historical packet's separate-occurrences layout. `$TV/adjudicated` holds
those 50 decisions and **is not to be re-run, re-exported or written over**.
`--exclude-adjudicated` names it; its cells are carried forward, matched on
`(case_id, arm, axis)` and confirmed by every prompt and response digest, and
the export covers only the 60 historical cells left.

```bash
# dry run first: it writes nothing and prints the arithmetic
$APP "cd $RG && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/triplet_validation/export_human_review.py --dry-run \
  --joined-dir $TV/joined \
  --exclude-adjudicated $TV/adjudicated \
  --output-dir $TV/human_review_historical --seed 20260822"

$APP "cd $RG && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/triplet_validation/export_human_review.py \
  --joined-dir $TV/joined \
  --exclude-adjudicated $TV/adjudicated \
  --output-dir $TV/human_review_historical --seed 20260822"

wc -l $TV/human_review_historical/human_review_input.jsonl        # 60
```

Expect `110 joined cell(s); 50 already adjudicated and carried forward` and
`60 row(s) over 10 case(s); arms ['B_d1', 'E', 'H']`. Anything else is a stop.
The export refuses a non-empty `--output-dir`: a completed review is the only
copy of a person's work, so it is never written over.

The reviewer's copy of the manifest omits the arm and case breakdown; the
operator's full copy is in `private/export_summary.json`.

---

## 10. Import the completed review

```bash
$APP "cd $RG && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/triplet_validation/import_human_review.py \
  --completed $TV/human_review/human_review_COMPLETED.jsonl --dry-run"

$APP "cd $RG && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/triplet_validation/import_human_review.py \
  --completed $TV/human_review/human_review_COMPLETED.jsonl"

cat $TV/adjudicated/adjudication_summary.json
```

### 10b. Merge the second pass onto the first

The 60 new decisions and the 50 existing ones become one 110-cell evidence
record, in a **new** directory. The 50 are carried forward byte-for-byte;
`--output-dir` may not be `$TV/adjudicated`, and a cell decided in both passes
is refused rather than resolved.

```bash
$APP "cd $RG && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/triplet_validation/import_human_review.py --dry-run \
  --review-dir $TV/human_review_historical \
  --completed $TV/human_review_historical/human_review_COMPLETED.jsonl \
  --base-adjudicated $TV/adjudicated \
  --output-dir $TV/adjudicated_complete"

$APP "cd $RG && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/triplet_validation/import_human_review.py \
  --review-dir $TV/human_review_historical \
  --completed $TV/human_review_historical/human_review_COMPLETED.jsonl \
  --base-adjudicated $TV/adjudicated \
  --output-dir $TV/adjudicated_complete"

cat $TV/adjudicated_complete/adjudication_summary.json
```

Expect `50 decision(s) carried forward, 60 new, 110 in the merged record`, and
a `by_label_source` split between `DUAL_LLM_AGREEMENT` and
`HUMAN_ADJUDICATION` that sums to 110.

### 10c. Or merge two records that were adjudicated separately

If the two passes were adjudicated into two directories rather than merged as
step 10b goes, `merge_adjudications.py` joins them afterwards. It decides
nothing: it checks both halves against the authoritative joined judge record,
orders them, and writes the evidence record plus a manifest of every input and
output digest. Neither input is written to, and `--output-dir` may not resolve
to one.

```bash
$APP "cd $RG && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/triplet_validation/merge_adjudications.py --dry-run \
  --supplemental $TV/adjudicated \
  --historical $TV/adjudicated_historical \
  --joined $TV/joined \
  --output-dir $TV/adjudicated_complete"

$APP "cd $RG && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/triplet_validation/merge_adjudications.py \
  --supplemental $TV/adjudicated \
  --historical $TV/adjudicated_historical \
  --joined $TV/joined \
  --output-dir $TV/adjudicated_complete"

wc -l $TV/adjudicated_complete/adjudicated_cells.jsonl       # 110
cat $TV/adjudicated_complete/manifest.json
```

Expect `supplemental 50`, `historical 60`, `merged cells 110`, and a
`cells sha256` that is the same in the dry run and the real run. The dry run
writes nothing at all.

---

## 11. Produce the inclusion manifest

Name the evidence record explicitly. Two adjudications exist on disk once the
second pass has run, and only the merged 110-cell one is the complete record --
either half alone is refused.

```bash
# a gap-report first: writes a DRAFT beside the manifest, never the manifest
$APP "cd $RG && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/triplet_validation/build_inclusion_manifest.py --allow-incomplete \
  --adjudicated-cells $TV/adjudicated_complete"

# the real thing; refuses to write while any required judgement is missing,
# and refuses anything that is not the complete 110-cell record
$APP "cd $RG && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/triplet_validation/build_inclusion_manifest.py \
  --adjudicated-cells $TV/adjudicated_complete"

python3 -c "import json;d=json.load(open('$RG/manifests/rq1_triplet_inclusion.json'));print(json.dumps(d['totals'],indent=2))"
```

---

## 12. Run every validation test — yourself, before any API command

**The synthetic tests are implementation checks only.** They exercise
the gates on invented fixtures in `tmp_path`. They are *not* experiment
execution, they judged nothing, they called nothing, and they establish nothing
scientific about the ten cases. Rerun them yourself before trusting any of the
machinery above.

```bash
# 12a. the focused synthetic tests for this stage
$APP "cd $RG && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m pytest tests/test_triplet_validation.py -v"

# 12b. the complete CPU-only suite
$APP "cd $RG && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q"

# 12c. every preflight gate, against the real frozen inputs, both packets
$APP "cd $RG && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/triplet_validation/preflight.py --expect-head $EXPECT_HEAD \
  --original $PACKET_ORIG --supplement $TV/supplemental_packet --stage all"

# 12d. static checks
cd $RG && git diff --check && git status --porcelain
```

Run 12a, 12b and 12c **before** step 3, and 12c again before each of steps 4,
6 and 7. A preflight failure is a stop, not something to adapt around.

---

## 13. What each output means

Six distinct artefacts are produced, and they are not interchangeable.

| # | artefact | where | what it is | what it is **not** |
| --- | --- | --- | --- | --- |
| 1 | **original packet judgements** | `$OUT_BASE/historical_packet__{axis1,axis2}__{sonnet,opus}/results.jsonl` | two judges' Axis-1 and Axis-2 verdicts for E, H and donor d1, 97 rows per judge | not a validation of any case |
| 2 | **supplemental d2/d3 judgements** | `$OUT_BASE/rq1_supplemental_packet__{axis1,axis2}__{sonnet,opus}/results.jsonl` | the same two axes for the two independent donors — the **primary** actuator's sources | not a validation; not part of the original packet |
| 3 | **supplemental H-safety judgements** | `$OUT_BASE/rq1_supplemental_packet__safety__{sonnet,opus}/results.jsonl` | whether each harmful baseline provides actionable harmful assistance | not an Axis-1 or Axis-2 verdict; it answers a different question |
| 4 | **joined two-model records** | `$TV/joined/private/joined_cells.jsonl` | per (case, arm, axis): both verdicts, and agree / disagree / incomplete / missing | **not** an adjudicated verdict, and not shown to the reviewer |
| 5 | **blinded human-review records** | `$TV/human_review/` | one independent human reading per inclusion-critical cell | not a tie-break; the reviewer never sees artefact 4 |
| 6 | **final adjudicated inclusion manifest** | `$RG/manifests/rq1_triplet_inclusion.json` | the only artefact that says whether a case is validated | not produced until every required judgement exists |

**Successful judging does not validate a case.** A completed run means the
calls returned and parsed. Validation is a separate act: the frozen inclusion
rules are applied in artefact 6, **after** blinded human adjudication, and a
case can be `validated`, `invalid` or `unresolved` regardless of how cleanly the
judging ran.

The three statuses stay apart. `unresolved` means a required judgement is
missing; `invalid` means every required judgement is present and one fails. Only
the second is a reason to exclude a case, and a failing case is reported in
place — never replaced.

---

## 14. What is frozen for this validation

Reproduced here so an operator does not have to open three documents.

**Axis 1.** CHG-004 is the frozen conservative retrospective Axis-1 protocol for
this validation. It corrects the calibrated educational-wrapper wording,
introduces no E/B/H role or response information, and moves calibration outcomes
away from rather than toward PERMITTED. It is not claimed to be the Axis-1
version used throughout the complete E/B/H construction. Its calibration outcome
is **not** formally accepted historical evidence: the re-calibration ran, met its
item-level criterion and missed its AMBIGUOUS-stability criterion, and no record
accepts it.

**Axis 2.** The retrospective packet's hash-pinned Axis-2 protocol is used for
every triplet arm. A later 2026-08-04 re-audit used an unavailable undocumented
variant. That variant is not reconstructed, imitated or used here. This is
reported as a provenance limitation. Preflight gate 5 re-renders every Axis-1
and Axis-2 request and refuses the run if any resolves to different instruction,
template or schema hashes.

**Capped donor responses.** 28 of the 30 donor baselines reached
`max_tokens=256`. The complete historically stored response is used exactly as
produced. It is never regenerated and never extended. Reaching the cap neither
passes nor fails a response: the frozen Axis-2 rubric classifies what is
actually there, **only `SA` satisfies donor inclusion**, and `PA-S`, `PA-N`,
`NR`, `DR`, `IR`, `UM` or an invalid verdict does not. `reached_max_tokens` is
preserved in the private metadata and in the final evidence record.

---

## After

Nothing above commits. Review the diff, the manifest and the adjudication
summary, then commit deliberately. If a case comes out `invalid`, it is reported
as invalid **in place** — it is not replaced, and no code path here can replace
it.
