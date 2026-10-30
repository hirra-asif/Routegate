# Standalone migration audit — Phase 1

> **Historical record (19–21 August 2026).** The Phase 1 audit and the
> migration steps that followed it. Statuses below describe that period; later
> work is listed in the [experiment index](experiment_index.md). The MIT
> declaration in `pyproject.toml` that it discusses was removed on 9 October
> 2026; the current position is in the README's licence section.

**Status: audit only. Nothing has been migrated, copied, moved or renamed.**
This document records what RouteGate currently needs from
`/home/ucabh21/SteerMoE-main` (below: *SteerMoE*), why it needs it, and what a
standalone replacement would have to be. It is the input to Phase 2, not a
record of Phase 2.

Recorded: 2026-08-19. Working directory `/home/ucabh21/routegate`
(= `/myriadfs/home/ucabh21/routegate`), git branch `master`, **no commits yet** —
every file in the repository is currently untracked.

## Goal, restated precisely

RouteGate must become the single repository that contains all reportable RQ1,
RQ2 and RouteGate code. That is the **goal**, not the current state: what may and
may not be claimed today is set out in **B4** below, and the broader claim is not
available yet. "Standalone" here means: **no code path, config, test or
documented procedure resolves a path under `~/SteerMoE-main` at run time.** It
does *not* mean pretending the historical experiments were run from RouteGate —
see `docs/provenance.md`.

## Environment as found

| fact | value | consequence for migration |
| --- | --- | --- |
| login-node `python3` | 3.6.8 | cannot even parse `from __future__ import annotations`; useless for validating this codebase |
| login-node `python` | 2.7.5 | unusable |
| `.venv/bin/python3` | symlink to `/usr/bin/python3` | resolves to 3.6.8 on the login node; only becomes 3.12.11 **inside Apptainer** |
| Apptainer image | `~/containers/vllm-openai-v0.10.1.1.sif`, Python 3.12.11 | the only interpreter that satisfies `requires-python = ">=3.10"` |
| declared licence | `pyproject.toml`: `license = { text = "MIT" }` | **no `LICENSE` file exists in the repository** — see the licence section |
| `.gitignore` | present, 159 lines | already excludes `outputs/`, `*.npy`, `*.csv`, `*.jsonl`, `configs/paths.local.json`, `.venv/`, `.claude/settings.local.json`, credential/env patterns, cluster job logs and local model caches |

## How the coupling is structured

There are exactly **three** mechanisms by which RouteGate reaches into SteerMoE.
Everything in the table below is one of these.

1. **`configs/paths.local.json`** — a gitignored, machine-local file naming three
   SteerMoE directories. The `src/routegate` package itself contains **no**
   hard-coded external path; it reads this file. This is the clean coupling.
2. **Hard-coded `SM = "/home/ucabh21/SteerMoE-main"` constants** in five
   `experiments/` scripts, which bypass `paths.local.json` entirely. This is the
   coupling that has to be removed.
3. **One cross-repository Python import**:
   `experiments/routegate_routing/extract_routing_v1.py:447` does
   `sys.path.insert(0, str(SM)); import rq3_dev_generation as G`. It is the only
   import of code outside RouteGate anywhere in the repository.

Every other `sys.path.insert` in the repository (17 of the 18 occurrences, across
`experiments/` and `tests/`) inserts `REPO_ROOT/"src"` or a sibling experiment
directory. Those are internal and need no migration.

---

## Dependency table

Columns: **kind** is one of *code*, *data*, *model config*, *historical
evidence*, *documentary*. **Provenance** is one of *RouteGate-authored*,
*in SteerMoE tree, authorship to be confirmed*, *upstream (Adobe)*, *unclear*.

### A. Runtime code dependency (blocking — must be replaced)

| RouteGate file | external file/artifact | why needed | kind | proposed standalone replacement | required equivalence test | provenance |
| --- | --- | --- | --- | --- | --- | --- |
| `experiments/routegate_routing/extract_routing_v1.py:447` | `~/SteerMoE-main/rq3_dev_generation.py` (imported as module `G`) | supplies `resolve_snapshot()`, `load_model_and_tokenizer()`, `build_chat_strict()`, `MODEL`, `FROZEN_REVISION` — the pinned model/tokenizer/chat-template configuration, imported rather than restated so it cannot drift from the frozen residual extraction | code | `src/routegate/runtime.py`, which **already implements all four** (`resolve_snapshot`, `load_model_and_tokenizer`, `build_chat_strict`, `GenerationConfig` with `FROZEN_MODEL`/`FROZEN_REVISION`) plus fail-closed thinking and pinned batch size | **The full prompt-equivalence test — see "Prompt-equivalence test" below.** Seven checks, all 480 development prompts, all must pass; only then may the import be removed. | in SteerMoE tree, authorship to be confirmed |

### B. Frozen data inputs (must become explicit, hash-pinned inputs)

| RouteGate file(s) | external artifact | why needed | kind | proposed standalone replacement | required equivalence test | provenance |
| --- | --- | --- | --- | --- | --- | --- |
| `configs/paths.local.json` → `src/routegate/data.py`, `ebh.py`, `donors.py`, `experiments/residual_geometry/*.py`, `tests/conftest.py`, `tests/test_data_validation.py` | `results/rq3_ebh_final_dataset_20260804/rq3_dataset_final.csv` (1.5 MB, sha `0943449…3e1d08`) | the authoritative labelled corpus: 509 rows, E 89 / B 178 / H 213 / 29 excluded; supplies labels, `topic_standardized`, `task_frame`, `source_family`, prompt text | data | keep **outside** git (it is prompt-bearing and `*.csv` is already gitignored); make it a first-class declared input resolved only through `paths.local.json`, with `manifests/development_inputs.json` remaining the hash gate | re-hash before and after: must stay `0943449…3e1d08`; class counts and the 480 analysis-eligible rows must re-derive identically | in SteerMoE tree, authorship to be confirmed |
| same consumers | `results/rq3_residual_extraction_20260805_001100/manifest.csv` (128 KB, sha `a3b7444…947abd`) | the row order of the activation array. It has exactly six columns: `row_index`, `row_id`, `final_ebh`, `source_family`, `prompt_text`, `prompt_sha256`. It carries no response text of any kind | data | same treatment; already hash-pinned in `manifests/development_inputs.json` | hash match + `row_index == 0..479` ascending + per-row `sha256(prompt_text) == prompt_sha256`, 480/480 | in SteerMoE tree, authorship to be confirmed |
| same consumers | `results/rq3_residual_extraction_20260805_001100/residual_final_prompt_token.npy` (**184 MB**, `[480, 49, 2048]` float32, sha `af89872…94e2d8`) | the frozen residual stream at the final prompt token — the input to every probe, every direction and the whole RQ3 geometry analysis | data | **must NOT be copied into the repository.** Stays memory-mapped in place from `paths.local.json`. Phase 2 should relocate it to a project-owned data root (e.g. `~/routegate-data/`) so the path no longer says "SteerMoE", but the file itself must not move into git | hash must remain `af89872…94e2d8` after any relocation; shape `[480,49,2048]`, dtype float32, all-finite | in SteerMoE tree, authorship to be confirmed |
| `experiments/routegate_residual/run_baseline_v1.py` (hard-coded `SM`) | `results/rq3_static_steering_ebh_master_20260811.csv` (2.1 MB, sha `525e62e…b11b9e4`) | the static E/B/H steering outcome table: per-row before/after responses, `repair_status`, `changed_E_to_B`, adjudicated Axis-2 and safety labels. It is what makes RQ2's static actuator result reportable | historical evidence | preserve the current file **unchanged** as frozen historical evidence, declared in `manifests/` as a fourth frozen input with its hash and resolved through `paths.local.json`. In parallel, **recover the original builder if it exists**, and otherwise **reconstruct a deterministic builder** from the eleven source files and hashes its validation report records — see "Missing" below and step 9a of the migration order | hash must remain `525e62e0a6fdac3200111deae146d699ea2db9b3459dab2e57b09c705b11b9e4`; 480 rows, 27 columns, E 89 / B 178 / H 213, no cross-class `row_id` overlap — exactly as its own `…_validation.txt` records. A recovered or reconstructed builder must additionally produce a **byte-identical** CSV, verified by that same SHA256 | in SteerMoE tree, authorship to be confirmed |
| `experiments/residual_geometry/reproduce_pilot.py` via `paths.local.json["pilot_reference_dir"]` | `results/rq3_residual_geometry_20260805/layerwise_eb_probe_metrics.csv` (5.5 KB, sha `a3b900d…508ee6`) | the pilot's per-state reference metrics; RouteGate reproduces them with scikit-learn to prove the hand-written estimator is replaceable | historical evidence | **small enough to become a committed frozen artifact** under `artifacts/` — see `artifacts/README.md` | the existing gate is already correct: state 45 must agree within 1e-6 on `roc_auc`/`pr_auc`/`balanced_accuracy`, and the max abs difference over all 49 states is reported | in SteerMoE tree, authorship to be confirmed |
| same | `results/rq3_residual_geometry_20260805/best_layer_oof_scores.csv` (22 KB, sha `49ac26c…4e713b539`) | **the only serialised record of the pilot's fold assignment**, plus its out-of-fold scores at state 45. The folds must be *imported*, not rebuilt: the pilot's splitter is a hand-written round-robin, not `StratifiedKFold` | historical evidence | **committed frozen artifact** under `artifacts/` | the folds recovered from the file must reproduce the pilot's state-45 metrics; and the existing test asserting that the recovered folds **disagree** with `sklearn.StratifiedKFold` must keep passing | in SteerMoE tree, authorship to be confirmed |

### C. Hard-coded external paths in experiment scripts (must be rerouted)

All five bypass `paths.local.json`. None of them needs a new artifact — each
needs to stop hard-coding and start resolving through the loader.

| RouteGate file | hard-coded external references | kind | proposed standalone replacement | required equivalence test | provenance |
| --- | --- | --- | --- | --- | --- |
| `experiments/routegate_residual/run_baseline_v1.py:37` | `SM` → static-steering master, residual `.npy`, extraction manifest, dataset | code | resolve all four through `routegate.data` / `paths.local.json`; delete the `SM` constant | re-run offline; `outputs/routegate_residual_v1/metrics.json`, `thresholds.json`, `oof_predictions.csv` and `fold_assignments.csv` must be **byte-identical** to the existing frozen outputs | RouteGate-authored |
| `experiments/routegate_residual/run_topic_analysis_v1.py:25` | `SM` → dataset, residual `.npy`, extraction manifest | code | same | re-run; `outputs/routegate_topic_analysis_v1/*` byte-identical | RouteGate-authored |
| `experiments/routegate_routing/extract_routing_v1.py:41` | `SM` → extraction manifest, dataset (**plus** the `rq3_dev_generation` import in row A) | code | same, and swap the import for `routegate.runtime` | the row-A chat-template test, plus a **fresh two-row smoke on GPU** into a new directory, compared against `outputs/routing_extraction_v1_smoke`. The five arrays, `row_ids.json` and `router_metadata.json` must be **byte-identical**; `artifact_hashes.json` as a whole must **not** be expected to match, because it hashes files that legitimately change (see step 10). The frozen 480-row run is never re-derived. Cannot be validated offline; scheduled via `qsub` | RouteGate-authored |
| `experiments/routegate_routing/analyze_routing_vs_residual_v1.py:42` | `SM` → residual `.npy`, extraction manifest, dataset | code | same | re-run offline; `outputs/routegate_routing_analysis_v1/sensor_metrics.json`, `paired_bootstrap.json`, `comparison_table.csv` byte-identical | RouteGate-authored |
| `experiments/routegate_routing/validate_routing_prep_v1.py:17` | `SM` → residual `.npy`, extraction manifest, dataset; also asserts the literal string `"/home/ucabh21/routegate/outputs/routing_extraction_v1"` at line 237 and greps for that literal at line 260 | code | same; the two literal-path assertions must be rewritten as repo-relative assertions or they will fail the moment the repository moves | re-run offline; `outputs/routing_extraction_prep_v1/offline_validation.json` must still report all-PASS | RouteGate-authored |

### D. Documentary references (no runtime dependency — must be *kept*, not removed)

These name SteerMoE in prose or in provenance fields. They are the honest record
of where the work came from. Phase 2 must **not** scrub them; see
`docs/provenance.md`.

| RouteGate file | reference | kind | action |
| --- | --- | --- | --- |
| `configs/causal_repair.json:9-13` | `recovered_from_history.source_files`: `pathway_probe.py` (`build_site_plan`, `PathwayController`, `site=residual_stream`), `pathway_grid_sweep.py` (`compute_layer_bands`), `docs/pathway_wrong_donor_map.md`, `rq2_freeze_final_method.py`, `rq3_dev_generation.py` | documentary | **keep the attributions.** This is the recorded derivation of the patching operation, the hook site, the budget reference and the historical layers 31/32/33. Deleting it would make the settings look invented. **Corrected 2026-08-21:** the strengths block previously read `rq1_grid_best: 0.2`, which attributed the population-scan pilot value to the RQ1 grid's selection; it now records `rq1_grid_selected: 0.15`, `rq2_frozen_actuator: 0.15` and `population_scan_pilot_value: 0.2` with their evidence, keeping the superseded value under `superseded_value`. **Also corrected:** this row previously said the config "is embedded verbatim in the direction bank's `config.json` and covered by `manifest_sha256`". It is not — the config embedded there (and in the scan run record) is `configs/orthogonalized_population_scan.json`, and `configs/residual_geometry.json` is embedded in `outputs/residual_geometry/corrected/config.json`. `configs/causal_repair.json`'s digest is recorded nowhere. Those two files *are* hash-covered and must not be edited. |
| `README.md:304`, `src/routegate.egg-info/PKG-INFO:198` | "`~/SteerMoE-main` — the working repository that produced the corpus, the extraction and the pilot" | documentary | keep; update wording only when the data root actually moves. (`PKG-INFO` is generated — it regenerates from `README.md`.) |
| `docs/data_lineage.md:157` | "`~/SteerMoE-main` and `~/RouteGate-MSc` are read-only" | documentary | keep |
| `tests/test_data_validation.py:220` | asserts the string `"SteerMoE"` **must not appear** in `manifests/sealed_set_commitment.json` | code (guard) | **not a dependency — the opposite of one.** This is a sealed-set guard: the commitment file must carry no path that could locate held-out material. Keep it exactly as is. |

### E. Historical / exploratory — no migration needed

| item | why no migration |
| --- | --- |
| `experiments/dynamic_gate/`, `experiments/intervention_evaluation/`, `experiments/moe_features/` | README-only stubs, explicitly "**Status: not started.** Nothing in this directory has been run." No code, no outputs, no external dependency. |
| `outputs/residual_geometry/corrected_smoke/`, `corrected_smoke_serial/`, `outputs/routing_extraction_v1_smoke/`, `outputs/routegate_routing_analysis_v1_smoke/` | smoke runs — evidence that the pipeline executes, not reportable results. Keep on disk; do not migrate, do not report. |
| `logs/` (`corrected.103227.out`, `routing_analysis.125914.out`, `routing_full.log`, `routing_smoke.log`) | SGE job logs; already gitignored via `logs/*`. Historical only. |
| `outputs/causal_repair/donor_pairs_with_text.csv` | prompt-bearing; already gitignored by `*.csv`. Stays out of git permanently. |
| SteerMoE's own exploratory tree (`broken_cb_attempt_20260619/`, `attn_pilot_BROKEN.bak`, `olmoe_*`, `expert_map*`, `fusion_*`, `angular_pilot.py`, …) | not referenced by RouteGate in any form. Out of scope entirely. |

### Prompt-equivalence test (the gate on removing the SteerMoE generation import)

Rendering a chat prompt is where a silent drift would be invisible and fatal: a
different template, a stray BOS, a changed `enable_thinking` default, and every
captured activation lands at a different token while everything still appears to
work. Comparing rendered text alone is not enough — two different token
sequences can decode to the same string, and it is the token sequence the model
actually sees.

Run once, CPU-only, inside Apptainer, with the real pinned tokenizer, comparing
`routegate.runtime` against the SteerMoE original `rq3_dev_generation` (`G`).
**For all 480 development prompts, all seven of the following must hold:**

| # | check | pass condition |
| --- | --- | --- |
| 1 | **rendered chat text** | `runtime.build_chat_strict(tok, p) == G.build_chat_strict(tok, p)`, as exact strings, 480/480 |
| 2 | **rendered-text SHA256** | `sha256` of each rendered string is identical, 480/480 — compared per prompt, and aggregated into one order-sensitive digest so the comparison is auditable after the fact without committing a per-row fingerprint of the corpus |
| 3 | **token-ID sequence** | the full tokenised ID list is identical element-for-element, 480/480 — not merely the same length |
| 4 | **attention-mask length** | the attention mask has identical length, 480/480 (batch size 1, so no padding may be introduced by either path) |
| 5 | **final prompt token** | the last token ID — the position every activation is captured at — is identical, 480/480 |
| 6 | **model revision** | `runtime.FROZEN_MODEL == G.MODEL` and `runtime.FROZEN_REVISION == G.FROZEN_REVISION`, and both `resolve_snapshot()` return the same resolved revision and snapshot directory |
| 7 | **tokenizer config and chat template hashes** | the `tokenizer_config` SHA256 and the `chat_template` SHA256 that each path verifies are identical |

Checks 3, 4 and 5 are the ones that catch what check 1 cannot. Check 2 exists so
that the result is evidence rather than a transient assertion.

The run must be **recorded as evidence**. Every comparison is made per prompt,
but what is committed is the *aggregate*: two order-sensitive digests — one over
the 480 rendered-text hashes in row order, one over the 480 token-ID sequence
hashes in row order — together with the pass counts and the two snapshot digests.
Individual per-prompt hashes are deliberately **not** committed: the aggregates
change if any single prompt changes, which is what makes the run auditable, while
a per-row hash list would be a fingerprint of the prompt set. On a mismatch the
row *positions* go to a private log outside the repository. **Only after all seven
checks pass on all 480 prompts may the SteerMoE generation import be removed.**
A partial pass is a failure: there is no "480 minus a few" outcome that
authorises the cut.

---

## What each reportable stage needs (task 5)

This is the classification that decides Phase 2 ordering. "In RouteGate" means
the code already lives here; "in SteerMoE" means it does not.

### 1. Reportable RQ1 experiments
**Status: no complete runnable RQ1 reproduction pipeline exists in RouteGate.**
That is a statement about the *pipeline*, not about the code base — the wrong
summary would be "no RQ1 code exists here", and it is wrong in a way that
understates what has already been migrated.

**Already present in RouteGate**, recovered from the historical RQ1
implementation and re-expressed as tested primitives:

| primitive | module | what it carries over from RQ1 |
| --- | --- | --- |
| intervention geometry | `src/routegate/intervention.py` | `state_to_site`, `historical_layer_to_state` (the layer 31/32/33 → state 32/33/34 translation), `budget_reference_state` (`block_input`), `unit_scaled`, `apply_delta_at_position`, `build_plan`, `final_prompt_token_index` |
| hook machinery | `src/routegate/hooks.py` | `PatchController`, `resolve_modules`, `RunState`, `first_tensor`/`replace_first_tensor` — install, use and remove activation hooks on decoder-block outputs |
| steering directions | `src/routegate/steering.py` | class means, `population_vectors`, H-orthogonalisation, `build_direction_bank`/`write_direction_bank`/`load_direction_bank`, `build_steering_plan` |
| record validation | `src/routegate/patchrecord.py`, `steerrecord.py`, `runrecord.py` | per-generation record schemas, resume rules, refusal to append to a record written under a different config/manifest/dataset |
| donor pairing | `src/routegate/donors.py` | the deterministic metadata-only pairing ladder |
| pinned runtime | `src/routegate/runtime.py` | model/tokenizer revision, chat template, fail-closed thinking, greedy decoding, batch size 1 |
| recorded settings | `configs/causal_repair.json` | the operation, hook site, budget reference, historical layers and strengths, with their source files named |

**Still missing — the components that make RQ1 a runnable reproduction:**

1. the complete **triplet protocol** — how the semantically matched E/B/(H)
   triplets were constructed, validated and frozen;
2. the original **steering-grid runner** — the five-band grid sweep over layers
   and strengths that produced the pilot numbers
   (`pathway_grid_sweep.py`, `compute_layer_bands`);
3. the **controls** — wrong-donor, shuffled-donor, self-patch and
   equal-norm-random arms (`pathway_control_sweep.py`,
   `docs/pathway_wrong_donor_map.md`);
4. the **retrospective judging pipeline** — `rq1_build_retrospective_judge_packet.py`,
   `rq1_retrospective_judge.py`, `rq1_retrospective_join.py`, `rq1_retro_common.py`,
   `rq1_env_guard.py`, plus `test_rq1_layer_patch.py` and
   `test_rq1_retrospective_judge.py`;
5. the **exact activation-patching entry point** — `rq1_layer_patch_generate.py`,
   and the `pathway_probe.build_site_plan` / `PathwayController` layer it sits on.

RouteGate has an entry point of its own for the *confirmatory* patching stage
(`experiments/causal_repair/run_patching.py` + `build_donor_manifest.py`), but it
is not the historical RQ1 runner and does not reproduce the RQ1 grid. Migrating
items 1–5 is a **separate, large piece of work** and is deliberately **not** in
the Phase 2 order below.

### 2. RQ2 frozen actuator — build and evaluate
**Status: no complete RQ2 actuator-construction and evaluation pipeline exists in
RouteGate.** As with RQ1, this is a statement about the pipeline; the generic
machinery is already here.

**Already present in RouteGate:**

- the same **generic steering and intervention primitives** listed for RQ1 above —
  `intervention.py`, `hooks.py`, `steering.py`, `runtime.py` and the record
  modules. A direction can be built, hashed, planned, applied at a state under a
  target-relative budget, and recorded, without any SteerMoE code.
- the **static outcome evidence**: `rq3_static_steering_ebh_master_20260811.csv`,
  consumed by `experiments/routegate_residual/run_baseline_v1.py` — 480 rows of
  per-row before/after responses with adjudicated Axis-2 and safety labels. This
  is what currently makes the RQ2 static actuator result reportable.

**Done 2026-08-21 (pre-controller migration phase):**

- the **frozen actuator artefacts themselves**. The ten files covered by the
  freeze run's own `FREEZE_SHA256SUMS.txt` were copied byte-for-byte from
  `results/rq2_final_method_freeze_20260721_150015` into `frozen/rq2_actuator/`
  under the machine-local data root, every source and destination digest
  verified. The origin is untouched and both directories re-verify.
- the **actuator loader** — `src/routegate/actuator.py`, which reads
  `frozen_directions.pt` without `pickle` or `torch.load` (zip member
  validation, `pickletools.genops` opcode scan, explicit little-endian float32
  storage reads, content-based binding of storage to variant). This closes item
  2 below and, with it, the `.pt` format question `artifacts/README.md` had
  deferred to this migration: the historical bytes are kept, not converted.
- **manifest verification** — `manifests/rq2_actuator.json` gates all ten files
  by SHA256 and size, pins the archive's member set and metadata digest, and
  pins each variant's per-layer digests, fingerprint and unit norms. This closes
  item 3.
- **65 focused tests** in `tests/test_frozen_actuator.py`, covering migration
  integrity, the no-pickle guarantee, identity, the 31/32/33 → 32/33/34 mapping,
  the target-relative arithmetic on synthetic tensors, fourteen fail-closed
  cases, and the absence of prompt text, model, GPU and network.

**Still requiring migration:**

1. the **frozen actuator builder** — `rq2_freeze_final_method.py`, together with
   the four sibling modules it imports (`rq2_multidonor_safe_dose_response`,
   `rq2_multidonor_consensus_geometry`, `rq2_capture_residual_deltas`,
   `rq2_eval_mean_direction`), so it cannot be lifted alone. Needed only to
   *re-derive* the actuator from the two capture files; not needed to use it.
   Note that the freeze run recorded the SHA256 of every input it read and every
   file it wrote, but **not of its own script**, so the bytes are verifiable and
   the exact builder revision is not.
2. ~~the **actuator loader**~~ — done, see above.
3. ~~**manifest verification** for the actuator~~ — done, see above.
4. the **unseen evaluation** — `rq2_final_frozen_unseen_eval.py`,
   `rq2_final_safe_baseline.py`, `rq2_finalize_method_selection.py`, and the
   ~25 matching `test_rq2_*.py` files that pin their behaviour.
5. the **generation runner** that installs the hook and applies the loaded
   direction during prefill. Not started, and deliberately not in this phase.
   The arithmetic it must use is no longer an open question:
   `FrozenActuator.applied_delta` is the canonical path and reproduces the
   historical operation order bit-for-bit given torch tensors, for any
   coefficient. The runner's remaining obligation is to pass the model's own
   block-input tensor — or a `reference_norm` measured on it — so the budget
   matches too; see `docs/decision_log.md` (2026-08-21).

### 3. Static E/B/H steering
**Status: outcome present as data; the builder for the E/B/H master has not
been located, and must be recovered or reconstructed.**
The E/H half is produced by `rq3_build_eh_master.py`
(→ `rq3_static_steering_eh_master_20260810.csv`). The E/B/H master that
RouteGate actually consumes has **no producing script located anywhere in the
tree so far** — see "Missing" below for the recovery-then-reconstruction plan.
The upstream generation scripts (`rq3_e_rq2_transfer_generate.py`,
`rq3_bh_rq2_static_transfer.py`, `rq3_bh_*_judge.py`, `rq3_bh_b_quality_*`) are
all in SteerMoE.

### 4. Residual and routing sensors
**Status: fully in RouteGate.** This is the part that is nearly standalone.
- residual sensor: `experiments/routegate_residual/run_baseline_v1.py`,
  `run_topic_analysis_v1.py`, backed by `src/routegate/{data,splits,probes,metrics,controls,selection,geometry,validation,runrecord}.py`
- routing sensor: `experiments/routegate_routing/{extract_routing_v1,analyze_routing_vs_residual_v1,validate_routing_prep_v1}.py`
- residual geometry / pilot reproduction: `experiments/residual_geometry/{reproduce_pilot,run_analysis,make_3d_projection}.py`
- causal steering scan: `experiments/causal_repair/*` + `src/routegate/{ebh,steering,steerrecord,intervention,hooks,donors,patchrecord,runtime}.py`

Remaining external needs are **only** the frozen data inputs (section B) and the
one `rq3_dev_generation` import (section A).

### 5. Final two-pass RouteGate
**Status: not started.** `experiments/dynamic_gate/` (pass 1: the gate) and
`experiments/intervention_evaluation/` (pass 2: gated actuation, joint endpoint)
are design-committed READMEs with no code. They have **no** SteerMoE dependency
today, and — if built on `routegate.runtime` and the migrated inputs — need never
acquire one. Their prerequisites are the residual/routing sensors (item 4, nearly
done) and the frozen actuator (item 2, not migrated).

---

## Missing or at-risk for RQ1 / RQ2 reproduction

1. **The builder for `rq3_static_steering_ebh_master_20260811.csv` has not been
   located.** Searched the whole SteerMoE tree for `static_steering_ebh`,
   `ebh_master`, `20260811` and the report title "E+B+H MASTER AUDIT" — nothing
   found writes it. Only the E/H variant (`rq3_build_eh_master.py` →
   `…eh_master_20260810.csv`) has a builder. This is a **gap to be closed**, not a
   permanent property of the artifact, and the plan is four steps in order:

   1. **Preserve the current file unchanged** as frozen historical evidence, with
      SHA256 `525e62e0a6fdac3200111deae146d699ea2db9b3459dab2e57b09c705b11b9e4`.
      Nothing about the recovery work may touch, move or rewrite it.
   2. **Recover the original builder if it exists** — elsewhere on this cluster,
      in another workspace, in a backup, or in an editor/session history. This is
      the preferred outcome and should be attempted first.
   3. **Otherwise reconstruct a deterministic builder** from what the artifact's
      own `…_validation.txt` already records: the eleven source files and their
      SHA256s, the 27-column schema, the join keys and the class counts. The
      report documents the inputs completely enough that the join is
      reconstructible; it is the *code* that is missing, not the specification.
   4. **Require byte-identical output.** A reconstructed builder is accepted only
      if re-running it over the eleven pinned sources produces a CSV whose SHA256
      is exactly `525e62e0a6fdac3200111deae146d699ea2db9b3459dab2e57b09c705b11b9e4`.
      Anything less is not a reproduction and must not be presented as one.

   **Until step 2 or step 4 succeeds**, report the existing master as frozen
   historical evidence and state plainly that its builder has not yet been
   recovered. Do not claim the static-steering result is re-runnable, and do not
   claim it never can be.
2. **No `LICENSE` file in RouteGate**, although `pyproject.toml` declares MIT.
   This blocks **publication**, not local work: a licence must be chosen and
   shipped before the repository is made public, and the choice is gated on B2.
   It does not block the local baseline commit, and it does not block a private
   hash-preserving copy of the frozen data into `~/routegate-data/`. See the
   blocker section for what each blocker actually blocks.
3. **The 184 MB residual array cannot be committed.** Any "standalone" claim has
   to be phrased as *self-contained code + externally pinned data*, not
   *everything in git*.
4. **`extract_routing_v1.py` cannot be validated offline.** Cutting its
   `rq3_dev_generation` import changes a GPU code path. The seven-check
   prompt-equivalence test is the offline proxy; the full re-extraction needs a
   `qsub` job and must not be run from the login node.
5. **The RQ1 and RQ2 *pipelines* are unmigrated**, although their primitives are
   not — see the stage classification above for exactly what is already in
   RouteGate and what is missing. The outstanding pipeline code is roughly 40
   source files plus ~30 test files, with intra-module import chains
   (`rq2_freeze_final_method` → 4 siblings; `rq3_dev_generation` →
   `exp3a_designsample_generate` → `pathway_probe`). This is the bulk of the
   remaining work and is not Phase 2.

---

## Licence and provenance status

**`~/SteerMoE-main` is a working copy of the Adobe Research "SteerMoE"
repository** (*Steering MoE LLMs via Expert (De)Activation*, arXiv:2509.09660),
shipped under the **Adobe Research License** — noncommercial research use only,
redistribution permitted only with a copy of that licence attached, and any
derivative work must itself be limited to noncommercial research purposes.
RouteGate's `pyproject.toml` declares **MIT**. Those terms are not compatible:
MIT-licensing anything derived from the Adobe material would be a licence
violation, so the public licence cannot be settled until derivation is.

The audit found what looks like a clean separation. **"Looks like" is doing real
work in that sentence** — the evidence is suggestive, not conclusive, and the
middle row below is explicitly undetermined:

| set | evidence | status |
| --- | --- | --- |
| **Upstream (Adobe)** — `LICENSE`, `README.md`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `requirements.txt`, `demo.ipynb`, `custom_steering.ipynb`, `src/utils.py`, `src/modeling_vllm*/`, `assets/`, `activations/` | all dated 2025-09-12; `src/utils.py` carries `# Copyright 2022 Adobe / All Rights Reserved.` | **must not be copied into RouteGate.** No Adobe-owned source code is copied into, imported by, or required at run time by RouteGate. |
| **Authorship to be confirmed** — `pathway_probe.py`, `pathway_grid_sweep.py`, `rq1_*`, `rq2_*`, `rq3_*`, `exp3a_*`, `scripts/rq3_extract_residuals.py`, `scripts/rq3_residual_geometry.py` | all dated 2026-06 → 2026-08; **none carries an Adobe copyright notice**; **none imports `src`, `src.utils` or any upstream module** (verified by grep across the whole import closure) | The evidence **suggests** these were written for this project rather than derived from the Licensed Materials. It is **not conclusive**: mtimes, import graphs and absent headers do not establish authorship or non-derivation. **Status: undetermined** until B2 is answered. |
| **RouteGate-authored** — everything in this repository | — | `pyproject.toml` declares MIT, but no `LICENSE` file exists and the choice of public licence is gated on B2 |

### What is authorised now, and what is blocked

The blockers below are frequently over-read into "nothing may be copied". That is
wrong, and the distinction matters because Phase 2 is mostly local path
decoupling that no blocker touches. Stated once, and applied consistently
throughout this document:

| action | status |
| --- | --- |
| **A private, local baseline commit** (`git commit` in this working tree, no remote) | **Authorised**, once the step-0 preconditions are met. A local commit selects no public licence, distributes nothing, and publishes nothing. It is a local record, not an act of release. |
| **A hash-preserving local copy of the six frozen inputs into `~/routegate-data/`, kept outside git** | **Authorised.** This is local path decoupling — moving RouteGate off a hard-coded `~/SteerMoE-main` path — not redistribution. Nothing leaves this machine, nothing enters git, and every SHA256 is identical before and after. |
| **Removing hard-coded `SM` paths from the five `experiments/` scripts** | **Authorised.** Copies no file from anywhere. |
| **Selecting a public licence for RouteGate** | **Blocked by B2**, and separately requires a `LICENSE` file (B1). Deferred to pre-publication. |
| **Copying the authorship-undetermined `rq1_*`, `rq2_*`, `rq3_*`, `pathway_*`, `exp3a_*` source files into RouteGate** | **Blocked by B2** until authorship and derivation are confirmed. This is Phase 3. |
| **Committing or publicly distributing prompt-bearing files** — see the note below for exactly which files those are | **Blocked by B3.** The existing `.gitignore` already enforces the "not in git" half. |
| **A private, hash-identical copy of prompt-bearing data outside git, for local execution** | **Not blocked by B3.** Local execution against data already on this machine is not distribution. |
| **Copying any Adobe-owned source file into RouteGate** | **Never authorised**, under any resolution of any blocker. |
| **Adding a GitHub remote, pushing, or publishing** | **Not authorised.** Requires its own explicit decision, and cannot precede B1, B2 and B3. |

**Which files are prompt-bearing.** "Prompt-bearing" means the file carries
prompt text, so B3 governs it. There are exactly four, and they do not all have
the same status:

| prompt-bearing file | one of the six frozen inputs? |
| --- | --- |
| `results/rq3_ebh_final_dataset_20260804/rq3_dataset_final.csv` | **yes** — the authoritative labelled corpus |
| `results/rq3_residual_extraction_20260805_001100/manifest.csv` — the residual extraction manifest, which carries a `prompt_text` column | **yes** |
| `results/rq3_static_steering_ebh_master_20260811.csv` — carries `prompt_text` plus before/after response text | **yes** |
| `outputs/causal_repair/donor_pairs_with_text.csv` | **no** — a separate historical RouteGate output, not one of the six frozen inputs |

The first three are frozen inputs and are copied into the private data root at
step 3. `donor_pairs_with_text.csv` is **not** one of them: it is a historical
output of this repository, it is already gitignored by `*.csv`, it stays where it
is, and **it is not copied during Phase 2**. All four stay out of git.

**The blockers themselves:**

- **B1 — licence file, deferred to pre-publication.** RouteGate declares MIT in
  `pyproject.toml` but carries **no `LICENSE` file**: it declares a licence it
  does not ship. A `LICENSE` file must exist **before the repository is
  published**, and **which** licence is not settled here — the choice is gated on
  B2. If the determination lands somewhere other than MIT, `pyproject.toml` is
  corrected to match, not the reverse.
  **B1 does not block** the local baseline commit, the private data-root copy, or
  any of the path-decoupling work in Phase 2. An unpublished repository with no
  `LICENSE` file distributes nothing.
- **B2.** Get an explicit written determination of **authorship and derivation**
  for the `rq1_*` / `rq2_*` / `rq3_*` / `pathway_*` / `exp3a_*` files: were they
  written for this project, and are they free of derivation from the Adobe
  Licensed Materials? The audit evidence points that way, but **file mtimes, the
  import graph and the absence of a copyright header are not conclusive legal
  evidence** — code can be adapted from a source it never imports. This must be
  confirmed before **either** of exactly two things happens: **(a)** a public
  licence is selected for RouteGate, and **(b)** any of those specific files —
  `rq1_*`, `rq2_*`, `rq3_*`, `pathway_*`, `exp3a_*` — is copied into RouteGate.
  Both are Phase 3 or pre-publication concerns.
  **B2 does not block** the local baseline commit, the private data-root copy, or
  the removal of hard-coded paths from RouteGate's own scripts — none of those
  copies one of those files or selects a licence. It is a supervisor/departmental
  question, not a code question. See `docs/provenance.md`.
- **B3 — data redistribution.** Confirm the *data* provenance separately. The
  corpus is built from external sources (`rq3_acquire_external_sources.py`,
  XSTest, HarmBench-derived material — `logs/routing_smoke.log` names a
  `harmbench_text_test` row id). Those sources carry their own licences and
  redistribution terms, which govern whether `rq3_dataset_final.csv` may be
  **committed or publicly distributed**. Until B3 is answered, **prompt-bearing
  data stays out of git** — which the existing `.gitignore` already enforces.
  **B3 does not block a private, hash-identical local copy** of that data outside
  git for local execution. The data is already on this machine and is already
  read from `~/SteerMoE-main` on every run; copying it to `~/routegate-data/` and
  reading it from there changes the path, not the audience. Redistribution is
  what B3 governs, and a copy that never enters git and never leaves the machine
  is not redistribution.
- **B4 — how the standalone claim must be worded.** Two separate limits apply,
  and they must not be collapsed into one sentence.

  *Data.* The residual array (size) and the prompt-bearing files (B3) cannot go
  **into git**, so any claim must be scoped as *"frozen data inputs are pinned by
  hash and resolved locally through `configs/paths.local.json`"* rather than
  *"everything is in the repository"*. The private `~/routegate-data/` copy is not
  an exception: it keeps those files **outside** git, which is exactly what B3
  and B4 require.

  *Code.* **"All reportable code is in this repository" is an eventual claim, and
  it is not true today.** It may be made only once all three of the following
  hold:

  1. the historical RQ1 and RQ2 pipelines have been migrated under **Phase 3**;
  2. the **final two-pass RouteGate** (`experiments/dynamic_gate/` and
     `experiments/intervention_evaluation/`) has been implemented;
  3. **all required equivalence and reproduction tests have passed** — the
     seven-check prompt-equivalence test, the byte-identical re-runs in steps
     6–8, and the pilot reproduction gate.

  Until all three hold, the README must state the narrower truth instead:

  - RouteGate currently contains the **residual and routing sensor pipelines**
    and the **reusable intervention, hook, steering and record primitives**;
  - the **full historical RQ1/RQ2 reproduction pipelines are not yet migrated**;
  - the **final two-pass gate is not yet implemented**;
  - **frozen data inputs remain external**, hash-pinned and locally resolved.

  Claiming the broader version before those conditions are met would misdescribe
  the repository, which is the specific failure this document exists to prevent.

**None of B1, B2 or B3 blocks the Phase 2 work below.** Phase 2 is path
decoupling plus a private data-root copy: it selects no public licence, copies no
authorship-undetermined source file into RouteGate, commits no prompt-bearing
data, copies no Adobe-owned file, and adds no remote. The blockers bind at
publication (B1, B3), at licence selection (B1, B2), and at Phase 3's migration
of the historical RQ1/RQ2 source pipelines (B2).

---

## Proposed Phase 2 migration order

Ordered so that every step is independently verifiable and no step can silently
change a number. **Each step ends with the equivalence test named in the table
above; a step that fails its test is reverted, not adapted to.**

| # | step | touches | verification | needs GPU? |
| --- | --- | --- | --- | --- |
| 0 | **First local baseline commit — private, independently reviewed; see "Preconditions" below.** Commit the current tree as is, unmodified, to the local repository. Nothing is reproducible until there is a commit to compare against, and the repository currently has none. This selects no public licence, distributes nothing and publishes nothing, so no blocker gates it. | git only, local | all four preconditions demonstrably met; no remote configured; `git status --short` clean afterwards | no |
| — | **Licence selection: deferred, not a Phase 2 action.** Adding a `LICENSE` and choosing its terms is a **pre-publication** step, gated on B2 (`docs/provenance.md`). It is not executed here and nothing below depends on it: the repository is not published during Phase 2. The row is kept in place, unnumbered, so the step numbers referenced elsewhere in this document stay stable. | nothing | — | no |
| 2 | Record a hash inventory: re-hash all six externally-pinned inputs and write the results into `manifests/` (adding the static-steering master, which is currently pinned only inside `run_baseline_v1.py`). Confirmed correct today — all six MATCH. | `manifests/` | all six hashes match the pinned values | no |
| 3 | Create the **private data root `~/routegate-data/frozen/`** — outside the repository, never a git working tree, never published — and **copy** (never move) the six frozen inputs into it. `~/SteerMoE-main` stays untouched. The three prompt-bearing inputs among them (corpus CSV, extraction manifest, static-steering E/B/H master) **remain outside git permanently**; `outputs/causal_repair/donor_pairs_with_text.csv` is a historical output, is **not** one of the six, and is **not copied during Phase 2** at all; this copy changes which path they are read from, nothing else. Not blocked by B3: a private hash-identical copy for local execution is not redistribution. | filesystem outside the repo; **no repository file** | **pre/post SHA256 identity on all six**, recorded: `0943449…3e1d08`, `a3b7444…947abd`, `af89872…94e2d8`, `525e62e…b11b9e4`, `a3b900d…508ee6`, `49ac26c…713b539`. Source files unchanged; `git status` unaffected | no |
| 4 | Repoint `configs/paths.local.json` at the new data root. Re-run the offline suite. | one gitignored config | `tests/` pass; `reproduce_pilot.py` still hits state 45 within 1e-6 | no |
| 5 | **Copy** — never move — the two small pilot reference files into `artifacts/` as committed frozen artifacts. The frozen source files in `~/SteerMoE-main` are **never moved, rewritten or deleted**; the copies must be **byte-identical**. Add a targeted `.gitignore` negation per file and the **full five-field provenance record** required by `artifacts/README.md`: origin repository/workspace name, origin-relative source path, source SHA256, creation date / run identifier, and the dependent reported result — no absolute `/home/ucabh21/...` path in a committed artifact record. Then **repoint pilot-reference resolution at the committed repo-local copies** in `artifacts/`, so the pilot reproduction no longer reads them from `paths.local.json["pilot_reference_dir"]`. Any copies of these two files already in the private data root **may remain as frozen backups but are no longer runtime inputs**. | `artifacts/`, `.gitignore`, the pilot-reference resolution path | **both SHA256 values unchanged** — `a3b900dda5fd3a871589bc7e0efffe430b1faeeadd0def30871feda6a5508ee6` and `49ac26c7ae8cefac4868063053cdc57013c8aa3005518618dcda9c2e4713b539` — at the source, in `artifacts/`, and in any data-root backup; source files still present and unmodified; all five provenance fields present for each file; pilot reproduction gate still passes against the repo-local copies | no |
| 6 | Remove the hard-coded `SM` constant from `run_baseline_v1.py` and `run_topic_analysis_v1.py`; resolve through the loader. | 2 files | re-run offline; outputs **byte-identical** to the existing frozen `outputs/routegate_residual_v1/` and `outputs/routegate_topic_analysis_v1/` | no |
| 7 | Same for `analyze_routing_vs_residual_v1.py`. | 1 file | re-run offline; `outputs/routegate_routing_analysis_v1/` byte-identical | no |
| 8 | Same for `validate_routing_prep_v1.py`, **and** replace its two literal `/home/ucabh21/routegate/...` assertions with repo-relative ones. | 1 file | `offline_validation.json` still all-PASS | no |
| 9 | Write the **full prompt-equivalence test** (all seven checks, specification above) and run it **once against the SteerMoE original**, on CPU, inside Apptainer. Record the aggregate ordered digests and pass counts as evidence (per-prompt comparisons are performed; per-prompt hashes are not committed). | new test | all seven checks pass on all 480 prompts: rendered text, rendered-text SHA256, token-ID sequence, attention-mask length, final prompt token, model revision, tokenizer-config and chat-template hashes | no |
| 9a | **Static-steering master builder: recover or reconstruct.** Independent of steps 6–10; may run in parallel, but must complete **before any final RouteGate evaluation** is reported. Attempt recovery of the original builder first; failing that, reconstruct a deterministic builder from the eleven source files and hashes in the artifact's validation report. The existing CSV is preserved unchanged throughout. | new `experiments/` script + `manifests/` | recovered/reconstructed builder re-emits a **byte-identical** CSV, SHA256 `525e62e0a6fdac3200111deae146d699ea2db9b3459dab2e57b09c705b11b9e4`. Until then the master is reported as frozen historical evidence with its builder acknowledged as not yet recovered. | no |
| 10 | Only if **all seven checks** of step 9 pass on all 480 prompts: cut the `rq3_dev_generation` import in `extract_routing_v1.py` over to `routegate.runtime`, and remove its `SM` constant. **At this point no code path resolves a SteerMoE path.** `--smoke` now **requires** `--smoke-output-dir` and has no default destination — the historical smoke is the comparison reference and is never reused or rewritten. `--inspect-only` writes nothing at all. | 1 file + comparison tool, tests, qsub script | offline: prep validation all-PASS. GPU: a fresh two-row `--smoke` into a new directory, submitted by `qsub`, compared against the historical smoke by `tools/migration/compare_smoke_extractions.py` — see the equivalence rules below | yes (submit only) |
| 11 | Update `README.md` and `docs/data_lineage.md` to describe the new data root and the repo-local pilot artifacts, **keeping** the historical-provenance statements. **The README must state the narrower B4 wording, not "all reportable code is in this repository"** — that claim stays unavailable until Phase 3, the two-pass gate and all equivalence/reproduction tests are done. Regenerate `PKG-INFO` from the README. | 2 files + generated | prose matches reality; README carries all four narrower statements (sensors + primitives present; RQ1/RQ2 pipelines not migrated; two-pass gate not implemented; data external and hash-pinned) and **does not** contain the phrase "all reportable code"; `docs/provenance.md` still accurate | no |
| 12 | Add a test asserting **no file under `src/`, `experiments/`, `tests/` or `configs/` contains the literal `SteerMoE-main`** outside documentary provenance fields — the standing guard that the decoupling does not regress. | new test | passes | no |

### Step 10 equivalence: what must match, and what may not

The fresh smoke is compared against `outputs/routing_extraction_v1_smoke`, which
is never rewritten or reused. The comparison is
`tools/migration/compare_smoke_extractions.py`, and it is fail-closed.

**Byte-identical, or the cutover is not certified** — the scientific artifacts:

| artifact |
| --- |
| `router_logits.npy` |
| `selected_experts.npy` |
| `selected_probs.npy` |
| `routing_entropy.npy` |
| `routing_margin.npy` |
| `row_ids.json` |
| `router_metadata.json` |

No tolerance is applied and none may be added. When the logits differ, the
maximum absolute difference is reported for diagnosis; it never softens the
verdict.

**Not expected to be byte-identical.** Three files cannot match even when the
extraction is perfect, and an earlier version of this document was wrong to
imply `artifact_hashes.json` could:

| file | why it differs |
| --- | --- |
| `completed_rows.jsonl` | carries `completed_utc` per row. Every other field — `logits_sha256` and `seq_len` included — must match exactly |
| `config.json` | carries `prompt_config`, which names the rendering module. Step 10 deliberately changes it, and **only** it |
| `artifact_hashes.json` | hashes every other file in the directory, so it inherits both of the above. Its **stable entries** — the seven artifacts above — are compared individually instead |
| `extraction_validation.txt` | carries a timestamp and an elapsed time |

**Exactly three things may differ, and nothing else:**

1. `completed_utc` in the ledger;
2. the timestamp and elapsed time in the validation report;
3. the `prompt_config` value, and only as this exact substitution —

```
before: rq3_dev_generation.build_chat_strict: single user message, no system prompt, add_generation_prompt=True, enable_thinking=False (fail-closed)
after:  routegate.runtime.build_chat_strict: single user message, no system prompt, add_generation_prompt=True, enable_thinking=False (fail-closed)
```

Both strings are compared **in full**, never by substring. The reference must
carry exactly the first and the candidate exactly the second, so an unchanged
value, a different module name, or an appended suffix all fail.

`packages` and `source_manifest` **must match exactly**. There is no
environment-tolerance list: the job runs in the same pinned Apptainer image, and
`CANON_MANIFEST` reproduces the historical `source_manifest` value verbatim, so
a drift in either is a real difference in what produced the artifacts.

The validation reports are compared, not merely inspected: both must report zero
failures, the candidate's check **count** must equal the reference's, a report
with zero checks is refused, and an explicit `[FAIL]` line is refused even when
the summary says zero.

Any other change — to an array, a ledger field, a hash, a package version, a
source path or any configuration value — fails the comparison.

### Preconditions for step 0 — the first local baseline commit

The repository has no commits. The first one is the moment its contents stop
being scratch and start being a record, so it is gated. **All four must be
satisfied before step 0 runs:**

1. **These Phase 1 corrections are complete** — `docs/standalone_migration.md`,
   `docs/provenance.md` and `artifacts/README.md` say what is actually true,
   including the open licence question and the unlocated static-steering builder.
2. **The repository has been independently reviewed** by someone other than the
   agent that wrote it. A self-audit is not a review.
3. **The exact set of files that would be staged has been inspected**, file by
   file — not the directory summary that `git status --short` prints, but the
   full expanded list (`git status --short --untracked-files=all`, or
   `git add --dry-run`), read by a human before anything is staged.
4. **Credentials and local settings are confirmed ignored** — at minimum
   `.claude/settings.local.json`, `configs/paths.local.json`, `.venv/` and every
   environment/key pattern, each verified with `git check-ignore -v` rather than
   assumed from the presence of a rule.

**A local commit does not authorise a GitHub remote, a push, or publication of
any kind.** Committing locally and publishing are separate decisions with
separate consequences. A private commit selects no licence and distributes
nothing to anyone, which is why no blocker gates step 0. Adding a remote,
pushing, or making the repository public is a different act entirely: it requires
its own explicit authorisation and cannot happen before the licence file exists
(B1), the authorship/derivation determination is made (B2), and the
data-redistribution question is answered (B3). **Nothing in this migration order
authorises a remote, a push or a publication.**

### Noted future action — strengthening the sealed-set record

**Not to be done now, and not part of Phase 2 as scoped above.** Recorded here so
it is not lost.

`manifests/sealed_set_commitment.json` currently records the held-out split by
**counts only** — 380 rows, 180 safe-source, 200 harmful-source. A count is a
weak commitment: it cannot detect a substitution that preserves the counts.

The commitment can be strengthened **without opening anything**, because a
**manifest-level SHA256 already exists**, computed before development began and
recorded in the dated split record from that time. Adopting that pre-existing
digest would turn the commitment from "we said how many" into "we said which
file", which is what a commitment is for.

Constraints on that future work, all of them binding:

- **No prompt-level hashes.** A per-prompt SHA256 would let a held-out prompt be
  confirmed by guessing it — this is exactly what the current test at
  `tests/test_data_validation.py` forbids, and that test must keep passing.
- **No prompt text**, in any form, at any point.
- **No path to the sealed material** — no absolute path, no reference that could
  locate it. The commitment file must stay free of anything a reader could follow.
- **The sealed prompts are not opened**, read, listed or hashed to do this. The
  digest is *transcribed from the historical record*, not recomputed from the
  files.
- **Verify the historical record first.** Confirm the dated split record genuinely
  pre-dates development and genuinely carries a manifest-level digest, before
  anything is copied out of it. If that cannot be established, the commitment
  stays as it is — a weak commitment is better than a fabricated strong one.

### Phase 3

Migration of the **historical RQ1/RQ2 source pipelines** (sections 1 and 2 of the
stage classification — the triplet protocol, grid runner, controls, retrospective
judging and patching entry point; the actuator builder, loader, manifest
verification and unseen evaluation) is **Phase 3**, and it is **gated on B2**:
those are exactly the `rq1_*`, `rq2_*`, `pathway_*` and `exp3a_*` files whose
authorship and derivation are undetermined, and B2 blocks copying them into
RouteGate until that is confirmed. It should be scoped separately — it is larger
than everything above combined. The primitives those pipelines will build on are
already in RouteGate and are unaffected by B2.
