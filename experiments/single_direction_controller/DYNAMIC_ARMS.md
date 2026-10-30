# The dynamic controller arms, and the OOF development comparison

Status: **`dynamic_raw` is generated, audited and registered for judging.**
480/480 records, `--verify` clean at 94 checks, the audit below passed, and the
blind packet is built. **No judge has been called.**

## What the static result established

The blocks 31-32 static B-E actuator at `alpha_R = 0.15` is closed and recorded
in `manifests/two_site_static_closure_v1.json`:

| class | denominator | reading |
|---|---|---|
| E | 89 decided | 64 substantive repairs, 61 full repairs |
| B | 176 decided (2 technically unresolved) | 16 material, 16 non-material degradations |
| H | 212 decided (1 technically unresolved) | 10 YES, 202 NO |

A trade-off, not a selection: against historical 31-33, blocks 31-32 reduce H
regressions and lose E repairs and B quality. It remains the only validated
feasible one-forward-pass actuator candidate.

## The controller weights problem, and what was done about it

`gate_oof_v1` produced 480 out-of-fold coefficients and **no weights**. Its own
manifest says `final_refit_performed: false`, and its `output_hashes.json`
lists seven files, none of them a checkpoint.

A column of per-row coefficients **is not a controller**. Using it at
generation time would be a row-id lookup: it cannot score a prompt the table
has never seen, and it would put every development row's answer into the arm
that used it. `reproduce_gate_controller.py` therefore re-runs the same
training and keeps what the first run threw away — five fold checkpoints and
the all-development refit — comparing every reproduced coefficient against the
saved one before it writes anything.

The refit step count is the one quantity v1 left open, because v1 performed no
refit. It is declared rather than tuned: **the median of the five
fold-selected steps**, a function of five integers early stopping already
chose.

### Determinism

BLAS threading changes reduction order, and the probe reconstruction is gated
at `1e-12`. Unpinned on a 36-core node the reduction drifts to `1.63e-12` and
the gate fails — correctly. `frozenprobe.reconstruct_fold_probe` now pins the
fit to one thread, and the reproduction runner pins the five thread variables
before importing numpy or torch and records what it pinned.

## The live path

Read before write, one generation per prompt:

1. read the **unsteered** state-32 final-prompt-token activation (the output of
   historical block 31), captured in a prefill-only forward that emits no
   token;
2. compute `g_R` once, from the fold controller that never saw the row;
3. apply `0.15 * g_R * ||h|| * d_R` at historical blocks 31 **and** 32 — the
   same `g_R` at both sites.

Historical block 33 is not a site: an edit at 31 changes the activation
entering 33, so 33's unsteered budget does not exist in the same pass.

Nothing is differentiable. `gate_from_activation` runs the frozen module under
`torch.no_grad()`, so no graph exists for anything to backpropagate through,
and `assert_no_parameter_requires_grad` proves the language model and both
probes cannot accumulate a gradient.

### The coefficient is computed, never looked up

`gate_oof_v1`'s coefficient column is not opened at generation time. It cannot
score a prompt it has not seen, and reading it would put every development
row's answer inside the arm that used it. What the arm applies is `g_R`
computed from *this pass's own* capture.

The frozen extraction is still used — as a **witness**. The same checkpoints
score the extraction's own state-32 activations, offline and on CPU, giving a
reference coefficient per row. That reference is what the applied one is
checked against:

| when | what is checked | budget |
|---|---|---|
| before any model loads | the five per-fold **held-out mean** coefficients, re-derived, against the means `gate_oof_v1` itself recorded | 32 × float32 eps |
| per row, while generating | the live coefficient against that row's own reference | 32 × float32 eps |

The budget is not a new number: `g_R` is float32 in `[0, 1]` here exactly as it
was in the reproduction, so it carries `reproequivalence.GATE_ABS_TOLERANCE`.
A row over it **stops the run** — it is not recorded, not warned about and not
averaged away. Measured on the real bundle, the five means reproduce to
`3.880e-09` against a `3.815e-06` budget.

Both captures are float32 widenings of the same bfloat16 forward at the same
site (`hooks.clone_cpu` stores them exactly as the historical extraction did),
so bit-identity is the expected outcome; the runner records per row whether it
held rather than assuming it.

## The two arms share one controller

| arm | coefficient |
|---|---|
| `dynamic_raw` | `0.15 * g_R` |
| `dynamic_hard_veto` | the same `g_R`, zeroed only where the frozen safety veto fires |

No second controller is trained, and none may be: the veto is an inference-time
brake around the same gate and can only lower a coefficient.

## How many responses are genuinely new: **480**

From `gate_oof_v1`'s own coefficients, the veto fires on **192** of 480 rows
(187 H, 3 B, 2 E) and on every one of them `g_R_after_veto == 0.0` exactly.

* `dynamic_raw` — **480 new**. Nothing existing was generated at a per-row
  coefficient, so none of it can be reused.
* `dynamic_hard_veto` — **0 new**. On the 192 vetoed rows the applied
  coefficient is exactly `0.0`, so the edit is the addition of a zero vector
  and the response is the unsteered one: the referenced baseline, already
  generated. On the other 288 rows the veto changes nothing, so the response is
  `dynamic_raw`'s.

The second arm is a deterministic recombination, and `reuse_arithmetic` refuses
to reuse a row whose coefficient is not *exactly* zero rather than merely
small.

## The comparison, and one blocker

| arm | status |
|---|---|
| referenced no-steering baseline | available, 480 responses, `c21e5b09…` |
| static B-E 0.15 blocks 31-32 | available, 480 responses, `65f0f91d…` |
| binary / CAST-style gate | **BLOCKER — does not exist** |
| `dynamic_raw` | to generate, 480 |
| `dynamic_hard_veto` | derived, 0 |

**The binary arm cannot be reused, because it was never produced.**
`experiments/dynamic_gate/README.md` states: *"Status: not started. Nothing in
this directory has been run"* and *"No gate has been trained."* There is no
model, no threshold, no response set and therefore no compatibility question to
answer about model, corpus, tokenizer, generation configuration or response
hashes. Producing a replacement is a separate decision and is **not** taken
here; the comparison runs with four arms unless and until that decision is
made.

## Commands

Reproduce the controller and persist checkpoints (CPU, no model, no API):

```bash
python experiments/single_direction_controller/reproduce_gate_controller.py \
  --direction-bank $HOME/routegate-data/frozen/validated_direction_bank \
  --source-run $HOME/routegate-controller-oof/outputs/single_direction_controller/gate_oof_v1 \
  --output-dir $HOME/routegate-data/gate_oof_v1_repro \
  --dry-run          # drop --dry-run to write the checkpoints
```

### Dry run — loads no model, generates nothing, writes nothing

```bash
apptainer exec --cleanenv --bind $HOME:$HOME \
  $HOME/containers/vllm-openai-v0.10.1.1.sif \
  /bin/bash --noprofile --norc -lc \
  "cd $HOME/routegate && .venv/bin/python \
     experiments/single_direction_controller/run_dynamic_arms.py \
     --direction-bank $HOME/routegate-data/frozen/validated_direction_bank \
     --controllers $HOME/routegate-data/gate_oof_v1_repro \
     --output-dir $HOME/routegate-data/dynamic_arms_v1 \
     --arm dynamic_raw --shard-index 1 --shard-count 1 --dry-run"
```

It validates the two-site declarations, the six checkpoint weight digests, the
equivalence clauses the bundle passed, the frozen fold assignment, the 480-row
order, the shard partition, the direction bank's digest and its `B_d1 - E`
subtraction order, and re-derives the five per-fold held-out means. The output
directory is deliberately **not** created.

### The real run — one complete unsharded execution, four GPUs

From a clean tree at the commit that carries the arm:

```bash
qsub -v ROUTEGATE_EXPECT_HEAD=$(git rev-parse HEAD) \
     experiments/single_direction_controller/submit_dynamic_raw.sh
```

The pin comes from the submission, not from a line in the script: editing a
constant would be circular, because setting it dirties the tree the script
refuses and committing the edit moves the hash it was set to.

which runs, on `-l gpu=4 -l mem=96G -l h_rt=24:00:00`:

```bash
apptainer exec --nv --cleanenv --bind $HOME:$HOME \
  $HOME/containers/vllm-openai-v0.10.1.1.sif \
  /bin/bash --noprofile --norc -lc \
  "cd $HOME/routegate && .venv/bin/python \
     experiments/single_direction_controller/run_dynamic_arms.py \
     --direction-bank $HOME/routegate-data/frozen/validated_direction_bank \
     --controllers $HOME/routegate-data/gate_oof_v1_repro \
     --output-dir $HOME/routegate-data/dynamic_arms_v1 \
     --arm dynamic_raw --shard-index 1 --shard-count 1 --device auto --resume"
```

`--shard-count 1` is the sequential form and is what the result is defined by.
The runner is also shardable (`--shard-index k --shard-count n`); each shard
writes its own records file and `--verify` merges them.

### Resume

The identical command. `--resume` is already on it, so an interrupted job is
restarted by re-submitting:

```bash
qsub -v ROUTEGATE_EXPECT_HEAD=$(git rev-parse HEAD) \
     experiments/single_direction_controller/submit_dynamic_raw.sh
```

Records already present are revalidated — locked fields, prompt digest, class,
sites, direction digest, fold, one controller block at both sites, and the
coefficient against its own reference — before being skipped. A row that
errored was never written, so a restart retries it. A shard that is already
complete refuses to be re-entered.

### Completion check — loads no model, writes nothing

```bash
apptainer exec --cleanenv --bind $HOME:$HOME \
  $HOME/containers/vllm-openai-v0.10.1.1.sif \
  /bin/bash --noprofile --norc -lc \
  "cd $HOME/routegate && .venv/bin/python \
     experiments/single_direction_controller/run_dynamic_arms.py \
     --direction-bank $HOME/routegate-data/frozen/validated_direction_bank \
     --controllers $HOME/routegate-data/gate_oof_v1_repro \
     --output-dir $HOME/routegate-data/dynamic_arms_v1 \
     --arm dynamic_raw --shard-index 1 --shard-count 1 --verify"
```

It re-runs every gate above, merges every shard file, refuses a row that
appears twice, validates all 480 records in full and requires exactly 480 — and
requires the applied coefficient to vary, because a single value would mean the
gate was never read.

---

# The audit of the completed run

`--verify` passes 94 checks over the finished directory, and the audit below is
the part `--verify` does not answer, because it is about how the run was
*driven* rather than what it recorded.

## The 267 / 213 split, which is not what it looks like

The run manifest reports `generated_here 267` and `already_present 213`, and
the development frame holds 89 E + 178 B = 267 rows and 213 H rows. The two
267s are **not the same 267**, and reading them as the same one would mean the
arm had refused to steer its H rows.

`already_present` is the **resume** count. `generate()` computes
`remaining = [rid for rid in shard if rid not in done]` — a set difference over
row ids already in the shard's records file — and the shard is in sorted
row-id order. Three invocations wrote the file:

| invocation | records | B | E | H | window (UTC) |
|---|---|---|---|---|---|
| 1 | 22 | 19 | 3 | 0 | 2026-09-03 17:29 → 18:13 |
| 2 | 191 | 105 | 54 | 32 | 2026-09-03 20:05 → 09-04 01:35 |
| 3 | 267 | 54 | 32 | **181** | 2026-09-04 08:22 → 12:42 |

So the 213 already present were **this same arm's own records**, revalidated on
resume against the locked fields, the corpus, the frozen fold and their own
reference coefficient before being skipped. Their classes are 124 B + 57 E +
32 H. The 267 written last are 181 H + 54 B + 32 E. Both logs say so directly:
`logs/dynamic_raw.live.20260903_210126.log` reads *"already written in this
shard: 22 / to generate now: 458"* and
`logs/dynamic_raw.live.20260904_091841.log` reads *"already written in this
shard: 213 / to generate now: 267"*.

Sorted row-id order explains the split exactly: `false_reject_test` (43 B +
7 E) < `harmbench_text_test` (32 H) < `orbench_hard` (135 B + 82 E) <
`orbench_toxic` (62 H) < `strongreject` (119 H). The first 213 in that order
are 50 + 32 + the first 131 `orbench_hard` rows.

## Generated, inherited and reused

| class | generated by this arm | inherited or reused |
|---|---|---|
| E | 89 | 0 |
| B | 178 | 0 |
| H | 213 | 0 |

Nothing was reused. `run_dynamic_arms.reuse_arithmetic` returns `reused = 0`
for `dynamic_raw` — *"no existing arm was generated at a per-row coefficient,
so nothing can be reused"* — and every one of the 480 records carries two fired
hooks, an out-of-fold coefficient and a **strictly non-zero** applied edit. The
smallest applied coefficient in the run is `2.36e-16`; not one row is exactly
zero, which is the only thing a carried-over baseline response could have been.

## What the coefficient is, and what it is not

| claim | evidence |
|---|---|
| one `g_R` per prompt, reused at historical blocks 31 and 32 | every record's two `per_state` blocks carry the same `g_R`, `fold`, `checkpoint` and `applied_alpha_R` |
| read from the unsteered state-32 final-prompt-token activation | `read_state = 32` on every block; the capture is a prefill-only forward before either site is written |
| applied strength is `alpha_R * g_R`, `alpha_R = 0.15` | `record["alpha_R"] == 0.15 * g_R` on all 480; `alpha_S = 0.0` on all 480 |
| out of fold | `fold` matches the frozen assignment (`3692fd99…`) on all 480, the checkpoint is `controller_fold_{fold}.json`, and `is_out_of_fold` is true |
| no class label decides anything | the only reference to `final_ebh` in `run_dynamic_arms.py` is the `row_class=` argument of `make_record`; no branch, threshold, skip or coefficient reads it |
| no stored OOF answer is read | the bundle carries weights and calibration only (`contains_per_row_coefficients = False`); `gate_oof_v1`'s coefficient column is the *witness* the live value is checked against, never the source |
| the language model and both probes are frozen | `assert_no_parameter_requires_grad`; the training specification's `frozen` list is *language model (never loaded), repair probe, safety probe, d_R, activations, folds* and `forward_passes = 0` |
| only the controller trained | `assert_optimiser_is_controller_only`, compared by `id` |

## Which arm this is

**Raw dynamic `g_R`.** The frozen safety veto is *not* applied: it sets `g_R`
to exactly `0.0` on the 192 rows it fires on, and no row in this run carries a
zero coefficient. The arm id, the run manifest, the records and this file all
say `dynamic_raw`.

---

# Judging: registered with the common engine

`manifests/dynamic_arms_v1_judging_v1.json`. The engine is
`experiments/judging/run_protocol.py`; only two things a manifest cannot
express are new code — the adapter and this experiment's own reading — and both
live in `src/routegate/dynamicjudging.py`.

| class | rubric | cells |
|---|---|---|
| E | Axis-2 response behaviour (CHG-002) | 89 |
| B | blinded pairwise quality against the matching unsteered baseline | 178 |
| H | actionable-harmful-help safety | 213 |

**480 cells, 960 billed calls, 0 reused.** Item 12's reuse rule runs as a live
gate in the adapter: an earlier verdict is reused only when the scientific
cell — screen row, **arm** and rubric — the prompt, the response bytes, the
rubric digests, both judge models and the protocol digests all match. No cell
on `dynamic_raw` has ever been judged, so nothing matches. The adapter also
reports the diagnostic that clause costs: **176 cells** (12 E, 164 H, no B)
carry judge-facing material byte-identical to a cell some completed run already
judged, because the applied edit fell below the actuator's bfloat16 resolution
or `alpha_R * g_R` reached 0.15. Reported, never reused.

The pairwise rubric is registered under the new name `pairwise_quality` over
the **same frozen bytes** as the two-site `quality` rubric. It has to be:
`triplets.AXIS_VALUES` already binds `quality` to the BEFORE/AFTER vocabulary,
and this engine decodes a pairwise answer into CANDIDATE/REFERENCE. Redefining
a registered name is refused by the engine, and
`dynamicjudging.PAIRWISE_TO_TWO_SITE` is the one rename that lets
`twositejudging.b_reading` be *called* rather than copied.

## Setup

```bash
export RG=$HOME/routegate
export CONTAINER=$HOME/containers/vllm-openai-v0.10.1.1.sif
export APP="apptainer exec --cleanenv --bind $HOME:$HOME $CONTAINER /bin/bash --noprofile --norc -lc"
export MF=$RG/manifests/dynamic_arms_v1_judging_v1.json
```

`--nv` appears nowhere below. Nothing here uses a GPU.

## 0. Adapt the completed run (free, offline)

```bash
$APP "cd $RG && .venv/bin/python experiments/single_direction_controller/build_judging_inputs.py --dry-run"
$APP "cd $RG && .venv/bin/python experiments/single_direction_controller/build_judging_inputs.py"
```

Re-derives the audit, stages the nine frozen rubric files byte for byte, runs
the reuse rule and writes `inputs/candidate_records.jsonl` (480) and
`inputs/reference_records.jsonl` (178 B baselines). Deterministic: the manifest
pins both by digest.

## 1. Preflight (free)

```bash
$APP "cd $RG && .venv/bin/python experiments/judging/run_protocol.py --manifest $MF --stage preflight"
```

## 2. Build the blind packet (free, once)

```bash
$APP "cd $RG && .venv/bin/python experiments/judging/run_protocol.py --manifest $MF --stage build --dry-run"
$APP "cd $RG && .venv/bin/python experiments/judging/run_protocol.py --manifest $MF --stage build"
```

Built once. After any judge has run, a rebuild is refused: new opaque ids and a
new A/B orientation would orphan every verdict already paid for.

## 3. Dry-run both judges (free)

```bash
$APP "cd $RG && .venv/bin/python experiments/judging/run_protocol.py --manifest $MF --stage dryrun --judge sonnet"
$APP "cd $RG && .venv/bin/python experiments/judging/run_protocol.py --manifest $MF --stage dryrun --judge opus"
$APP "cd $RG && .venv/bin/python experiments/judging/run_protocol.py --manifest $MF --stage judge --judge sonnet --dry-run"
$APP "cd $RG && .venv/bin/python experiments/judging/run_protocol.py --manifest $MF --stage judge --judge opus --dry-run"
```

## 4. The two billed runs — 480 calls each, 960 total

```bash
read -rs -p "judge API key: " MODEL_API_KEY && export MODEL_API_KEY && echo
export APPTAINERENV_MODEL_API_KEY="$MODEL_API_KEY"
export ROUTEGATE_JUDGE_ENDPOINT="<the gateway endpoint>"
export APPTAINERENV_ROUTEGATE_JUDGE_ENDPOINT="$ROUTEGATE_JUDGE_ENDPOINT"

$APP "cd $RG && .venv/bin/python experiments/judging/run_protocol.py --manifest $MF --stage judge --judge sonnet"
$APP "cd $RG && .venv/bin/python experiments/judging/run_protocol.py --manifest $MF --stage judge --judge opus"
```

Resumable by the **identical** command: a row already recorded as successful is
skipped and never re-sent. A failed row is not retried automatically — it may
already have been billed — and re-running it is a deliberate second call.

## 5. Join, review, adjudicate, report

```bash
$APP "cd $RG && .venv/bin/python experiments/judging/run_protocol.py --manifest $MF --stage join --dry-run"
$APP "cd $RG && .venv/bin/python experiments/judging/run_protocol.py --manifest $MF --stage join"

$APP "cd $RG && .venv/bin/python experiments/judging/run_protocol.py --manifest $MF --stage export --dry-run"
$APP "cd $RG && .venv/bin/python experiments/judging/run_protocol.py --manifest $MF --stage export"

# the reviewer fills human_review_TEMPLATE.jsonl and saves it as
# human_review_COMPLETED.jsonl in the same directory
$APP "cd $RG && .venv/bin/python experiments/judging/run_protocol.py --manifest $MF --stage import --dry-run"
$APP "cd $RG && .venv/bin/python experiments/judging/run_protocol.py --manifest $MF --stage import"

$APP "cd $RG && .venv/bin/python experiments/judging/run_protocol.py --manifest $MF --stage report"
```

The review directory is `$HOME/routegate-data/dynamic_arms_v1_judging/judging/review/`,
holding `human_review_input.jsonl`, `human_review_TEMPLATE.jsonl`,
`human_review_COMPLETED.jsonl` (the reviewer writes this one) and
`private/review_row_map.jsonl`. A technically unresolved cell is **never**
exported: a human adjudicates a disagreement between two judges, and does not
supply a verdict a judge never gave. A blank row stays UNRESOLVED, which is a
finding; a guess is not.

## 6. The three-arm reading

```bash
$APP "cd $RG && .venv/bin/python experiments/single_direction_controller/report_dynamic_judging.py --dry-run"
$APP "cd $RG && .venv/bin/python experiments/single_direction_controller/report_dynamic_judging.py"
```

Compares the unsteered baseline, the static blocks-31-32 arm at
`alpha_R = 0.15` and the audited dynamic controller, over the same 480 rows.
The E, B and H readings are `twositejudging.e_reading` / `b_reading` /
`h_reading`, called unchanged; the H regression rule is baseline NO → arm YES,
over the rows decided at both ends, and UNCLEAR is not NO. It also reports the
intervention rate and the `g_R` distribution overall and by class. It fails
closed with a diagnosis when an adjudication is absent, because a comparison
that quietly loses an arm reads as that arm having no verdicts.

The static arm's own closed reading, for reference:

| class | denominator | reading |
|---|---|---|
| E | 89 decided | 64 substantive, 61 full repairs |
| B | 176 decided (2 unresolved) | 16 material, 16 non-material degradations |
| H | 212 decided (1 unresolved) | 10 YES, 202 NO |

## What is committed, and what is not

**Committed:** the closure manifest, the judging manifest,
`src/routegate/twositestaticclosure.py`, `src/routegate/livecontroller.py`,
`src/routegate/dynamicarms.py`, `src/routegate/dynamicjudging.py`, the runners,
`build_judging_inputs.py`, `report_dynamic_judging.py`,
`submit_dynamic_raw.sh`, the tests, this file.

**Never committed:** checkpoints, per-row coefficients, prompts, responses, the
staged rubric text, the packet, the private row map, every raw API body, the
joined cells, the review packet, the adjudicated cells, both reports, and
anything else under the machine-local data root.
