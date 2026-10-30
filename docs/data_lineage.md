# Data lineage

Every number this package reports traces back through this chain. Each link is
identified by SHA256, and the two accepted inputs at the end of the chain are the
only files the package will open.

```
 external benchmarks
        │
        ▼
 prompt split package ──────────── 2475 development / 380 held-out
        │                          (the held-out split is sealed)
        ▼
 development generation ────────── one unsteered response per prompt
        │
        ▼
 two-axis judging ──────────────── two independent model judges
        │
        ▼
 human adjudication ────────────── disagreements resolved by a human reviewer
        │
        ▼
 rq3_dataset_final.csv ─────────── 509 rows  ◄── ACCEPTED INPUT 1
        │                          E 89 / B 178 / H 213 / excluded 29
        ▼
 residual extraction ───────────── 480 eligible rows, one forward pass each
        │
        ├── manifest.csv ────────── row order of the array
        └── residual_final_prompt_token.npy  ◄── ACCEPTED INPUT 2
                                   [480, 49, 2048] float32
        │
        ▼
 residual-geometry pilot ───────── frozen reference outputs, read only
        │
        ▼
 THIS PACKAGE ──────────────────── reproduction, then correction
```

## Accepted inputs

| | dataset | activation array |
| --- | --- | --- |
| file | `rq3_dataset_final.csv` | `residual_final_prompt_token.npy` |
| SHA256 | `09434496…353e1d08` | `af89872d…2a994e2d8` |
| shape | 509 rows | `[480, 49, 2048]`, `float32` |
| copied here | no | **no** — memory-mapped in place |

Plus `manifest.csv` (SHA256 `a3b7444b…14947abd`), which fixes the row order: its
line *i* is array row *i*, and it is sorted by `row_id`.

The full absolute paths live only in `configs/paths.local.json`, which is
gitignored. `manifests/development_inputs.json` records the hashes, counts, shape
and dtype — the facts — without the machine-specific paths.

## Class composition

| label | n | in the array | analysed here |
| --- | --- | --- | --- |
| E — over-refusal | 89 | yes | yes, `y = 1` |
| B — benign compliance | 178 | yes | yes, `y = 0` |
| H — correct refusal | 213 | yes | **no** |
| EXCLUDED_PARTIAL_OR_UNCLEAR | 29 | no | no |
| **total** | **509** | **480** | **267** |

Source composition of the analysed contrast, which is not balanced across classes
and is therefore a known confound rather than a nuisance:

| source family | E | B |
| --- | --- | --- |
| `orbench_hard` | 82 | 135 |
| `false_reject_test` | 7 | 43 |

A source-only one-hot probe reached ROC-AUC 0.555 in the pilot — above chance,
well below the residual probe, and enough to justify keeping the control.

## How the labels were reached

1. **Judging.** Every development response was scored on two axes — a refusal
   taxonomy and a safety judgement — by two independent model judges.
2. **Adjudication.** Where the judges disagreed, or where a judgement was
   flagged, a human reviewer decided. 113 rows carry `adjudicator = "Human
   reviewer"`.
3. **Amendment (2026-08-04).** A blind Axis-2 re-audit of all 114 rows then
   labelled E surfaced 25 disagreements, all 25 researcher-confirmed: 16 rows
   moved E → B and 9 moved E → EXCLUDED_PARTIAL_OR_UNCLEAR. This produced the
   accepted counts, E 89 / B 178.

That amendment is why input discipline matters here more than usual. It changed
the class balance late, and the pre-amendment file was kept beside the accepted
one:

| refused file | why |
| --- | --- |
| `rq3_dataset_final_before_e_axis2_reaudit.csv` | E 114 / B 162 / excluded 20 — pre-amendment |
| `rq3_ebh_final_analysis_dataset.csv` | E 114 / B 162 / excluded 20 — pre-amendment |
| anything under `rq3_geometry_prepare_20260803_215218` | E 116 / B 180, and frozen folds built on those labels |

All three are refused **by filename, before any hash is computed**. Two of them
read as more authoritative than the accepted `rq3_dataset_final.csv`, which is
exactly the failure mode the by-name refusal is there to prevent.

## How the activations were produced

One forward pass per eligible prompt, no generation, no steering, no gradients.

| | |
| --- | --- |
| model | Qwen/Qwen3-30B-A3B, revision `ad44e777…ebb85d39` |
| execution | HuggingFace eager attention, `bfloat16`, `device_map="auto"` |
| chat template | Qwen `apply_chat_template`, single user message, no system prompt |
| thinking mode | disabled, fail-closed |
| token position | `-1` — the final token of the rendered chat prompt, immediately before generation would begin |
| captured | all 49 hidden states, saved as `float32` |
| completed | 2026-08-04T23:28:49Z, 480 of 480 rows, 15/15 validation checks passed |

The extraction's own validation confirmed: no NaN or infinity, no all-zero row
left from the memory-mapped fill, every prompt identical to the dataset's, every
prompt hash recomputing, and the dataset still hashing to its recorded value at
the end of the run.

### State numbering

49 hidden states, and the off-by-one is the whole reason this package numbers
them explicitly:

- **state 0** — the embedding output at the final prompt token;
- **states 1–48** — the outputs of transformer blocks 1 to 48.

Because the rendered chat prompt ends with the same token for every prompt,
state 0 is one identical vector across all 480 rows. It is a degenerate control
by construction, and calling it "layer 0" invites reading its chance-level score
as a finding about the first layer of the network. It is not one.

## The pilot reference

`results/rq3_residual_geometry_20260805`, read only. Two files are used, both
hash-checked:

| file | role |
| --- | --- |
| `layerwise_eb_probe_metrics.csv` | per-state reference metrics; its `layer` column is this package's `state` |
| `best_layer_oof_scores.csv` | **the only serialised record of the pilot's fold assignment**, plus its out-of-fold scores at state 45 |

The fold assignment has to be *imported* from that second file rather than
rebuilt. The pilot's metadata calls its splitter "StratifiedKFold", but the
implementation is a hand-written round-robin deal of shuffled per-class
positions, which is a different algorithm from
`sklearn.model_selection.StratifiedKFold` and produces different held-out sets. A
test asserts the two disagree; if they ever coincided, the reproduction gate
would stop being informative.

## What this package writes

Only into `outputs/`, and only run records: config, environment, input hashes,
fold assignments, out-of-fold predictions, layerwise metrics, the comparison
against the reference, and a validation record. No input is modified. The two
historical source directories — `~/SteerMoE-main` and `~/RouteGate-MSc` — are
read-only, and a test re-hashes the inputs before and after a run to prove it.
