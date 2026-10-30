# Runbook — the common judging framework

Copy-paste commands, in execution order. Read `README.md` first.

Only **step 4 costs money**. Everything else is free and offline, and every
stage accepts `--dry-run`, which runs the gates and writes nothing.

```bash
export RG=$HOME/routegate
export CONTAINER=$HOME/containers/vllm-openai-v0.10.1.1.sif
export APP="apptainer exec --cleanenv --bind $HOME:$HOME $CONTAINER /bin/bash --noprofile --norc -lc"
export MF=$RG/manifests/<your_experiment>_judging_v1.json
```

`--nv` is absent from every command below. Nothing here uses a GPU.

---

## Rehearsal on a synthetic fixture (no credential, no data root, no GPU)

Run this first, on any machine. Every string it writes is invented; nothing it
touches is real. It is the fastest way to see all nine stages behave.

```bash
export DEMO=/tmp/judging_demo && rm -rf $DEMO

$APP "cd $RG && .venv/bin/python experiments/judging/make_demo_fixture.py --out $DEMO"

# 11 cells, 22 planned calls
D() { $APP "cd $RG && .venv/bin/python experiments/judging/run_protocol.py \
  --manifest $DEMO/demo_manifest.json --root $DEMO $*"; }

D --stage preflight
D --stage build --dry-run          # writes nothing
D --stage build
D --stage dryrun                   # renders and hashes 22 requests, calls nothing
```

Steps 4 to 9 of the rehearsal need judge records. Rather than call an API, the
test suite drives the same code path with an injected transport — see
`tests/test_judging_protocol.py::test_the_report_states_explicit_denominators`,
which takes a fixture from build through report and asserts the denominators.

---

## 0. Before anything

- Step 4 makes **billed API calls**. Steps 1 to 3 are free; run them first.
- Every judging step is **resumable**. If one dies, re-run the identical
  command: rows already recorded as successful are skipped and never re-sent.
- **A failed row is not a free row.** Nothing is retried automatically. A
  request that timed out, returned an unparseable body, returned an unexpected
  model, or failed after the response arrived may already have been billed. It
  goes to `failures.jsonl` with that note, and re-running it is a second
  billable call and a deliberate decision by you.
- Every attempt is preserved — successes, failures and raw bodies. Raw evidence
  is immutable: an existing body is never written over.

### The credential

Set it in the shell, never in a file. No manifest, record, log or command line
ever carries it, and the manifest loader refuses a credential-shaped value.

```bash
read -rs -p "judge API key: " MODEL_API_KEY && export MODEL_API_KEY && echo
export APPTAINERENV_MODEL_API_KEY="$MODEL_API_KEY"
export ROUTEGATE_JUDGE_ENDPOINT="<the gateway endpoint>"
export APPTAINERENV_ROUTEGATE_JUDGE_ENDPOINT="$ROUTEGATE_JUDGE_ENDPOINT"
```

`$APP` uses `--cleanenv`, which drops the host environment; `APPTAINERENV_X` is
the only propagation that survives it and keeps the value off every command line
and out of every process listing.

---

## 1. Preflight

```bash
$APP "cd $RG && .venv/bin/python experiments/judging/run_protocol.py \
  --manifest $MF --stage preflight"
```

Re-hashes every declared input and every frozen rubric file, checks the count
arithmetic, and reports whether the packet is built and whether judging has
started. A failure means **stop** — not "adapt the check".

## 2. Build the blind packet

```bash
$APP "cd $RG && .venv/bin/python experiments/judging/run_protocol.py \
  --manifest $MF --stage build --dry-run"

$APP "cd $RG && .venv/bin/python experiments/judging/run_protocol.py \
  --manifest $MF --stage build"
```

**The packet is built once.** After any judge has run, a rebuild is refused:
new opaque ids and a new A/B orientation would orphan every verdict already
paid for. `packet/private/row_map.jsonl` is the only record of the orientation
— do not delete it, and do not commit it.

## 3. Dry-run the judges

```bash
$APP "cd $RG && .venv/bin/python experiments/judging/run_protocol.py \
  --manifest $MF --stage dryrun"
```

Renders every request from the frozen text and hashes it. Calls nothing.

## 4. The two billed runs

```bash
$APP "cd $RG && .venv/bin/python experiments/judging/run_protocol.py \
  --manifest $MF --stage judge --judge sonnet --dry-run"

$APP "cd $RG && .venv/bin/python experiments/judging/run_protocol.py \
  --manifest $MF --stage judge --judge sonnet"

$APP "cd $RG && .venv/bin/python experiments/judging/run_protocol.py \
  --manifest $MF --stage judge --judge opus"
```

Back-to-back, once each dry-run is clean:

```bash
$APP "cd $RG && .venv/bin/python experiments/judging/run_protocol.py --manifest $MF --stage judge --judge sonnet \
   && .venv/bin/python experiments/judging/run_protocol.py --manifest $MF --stage judge --judge opus"
```

## 5. Join

```bash
$APP "cd $RG && .venv/bin/python experiments/judging/run_protocol.py \
  --manifest $MF --stage join --dry-run"

$APP "cd $RG && .venv/bin/python experiments/judging/run_protocol.py \
  --manifest $MF --stage join"
```

Every cell must end `agree`, `disagree` or `unresolved_technical`. A failed row
with no verdict **stops the join** unless a technical-unresolved manifest names
it — that gate is what keeps a new failure from riding in on an old manifest.
If it fires, stop and read the message; do not widen the manifest to fit.

To carry known technical failures:

```bash
... --stage join --carried $RG/manifests/<experiment>_technical_unresolved.json
```

## 6. Export the blinded review

```bash
$APP "cd $RG && .venv/bin/python experiments/judging/run_protocol.py \
  --manifest $MF --stage export --dry-run"

$APP "cd $RG && .venv/bin/python experiments/judging/run_protocol.py \
  --manifest $MF --stage export"
```

Genuine disagreements only. A technically unresolved cell is never exported: a
human adjudicates a disagreement between two judges; it does not supply a
verdict a judge never gave.

Give the reviewer `human_review_input.jsonl` and `human_review_TEMPLATE.jsonl`;
they fill the template and save it as `human_review_COMPLETED.jsonl`. A blank
row stays `UNRESOLVED`, which is a finding. A guess is not.

## 7. Adjudicate

```bash
$APP "cd $RG && .venv/bin/python experiments/judging/run_protocol.py \
  --manifest $MF --stage import --dry-run"

$APP "cd $RG && .venv/bin/python experiments/judging/run_protocol.py \
  --manifest $MF --stage import"
```

## 8. Report

```bash
$APP "cd $RG && .venv/bin/python experiments/judging/run_protocol.py \
  --manifest $MF --stage report"
```

Every planned cell is retained. Every rate divides by `decided_denominator` —
the cells actually adjudicated for that class — never by the class total when
fewer were decided. Unresolved cells are listed by opaque id and rubric, with
no prompt or response text.

---

## Registering a new steering mechanism

A new mechanism is a **new manifest**, not new code.

```bash
# 1. copy the worked example as a starting point
$APP "cd $RG && .venv/bin/python experiments/judging/make_demo_fixture.py --out /tmp/shape"
cp /tmp/shape/demo_manifest.json $RG/manifests/<experiment>_judging_v1.json

# 2. edit it: experiment_id, packet_id, row prefixes (add every existing
#    prefix to foreign_row_prefixes), arms.candidate / arms.reference,
#    inputs (path + sha256 per file), classes, rubrics (digests + both
#    vocabularies for a pairwise rubric), coverage, outputs.

# 3. hash the inputs and the frozen rubric text
$APP "cd $RG && sha256sum <data-root>/<candidate_records>.jsonl \
  <protocol-dir>/<rubric>_instructions.txt \
  <protocol-dir>/<rubric>_user_template.txt \
  <protocol-dir>/<rubric>_schema.json"

# 4. add the data roots to configs/paths.local.json under the keys the
#    manifest names in inputs.paths_key, outputs.root_key and
#    rubrics.<name>.protocol_dir_key

# 5. the manifest is now self-checking
$APP "cd $RG && .venv/bin/python experiments/judging/run_protocol.py \
  --manifest $RG/manifests/<experiment>_judging_v1.json --stage preflight"

# 6. commit the manifest; never commit a packet, a private map, a raw body or
#    any prompt or response text
git add manifests/<experiment>_judging_v1.json && git commit
```

Arm ids for the mechanisms this framework is meant to cover — all one manifest
field, `arms.candidate`:

```
no_steering                 (usually the reference arm)
static_repair_<alpha>
cast_gate_<threshold>
dynamic_controller_<id>
safety_projected_<id>
layer_ablation_<sites>
```

---

## What is committed, and what is not

**Committed:** the manifest, `src/routegate/judgingprotocol.py`,
`src/routegate/judgingstages.py`, `experiments/judging/`, the tests.

**Never committed:** the packet, the private row map, every raw API body, the
joined cells, the review packet, the adjudicated cells and the report. All of
it is prompt-bearing, response-bearing or de-anonymising.
