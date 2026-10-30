# artifacts/

It holds the small, frozen, hash-verified files that a reader needs in order to
reproduce a reported number — and nothing else.

## Promoted artifacts

### `pilot_reference/` — the residual-geometry pilot reference

Promoted 2026-08-19 (Phase 2 step 5 of the standalone migration). **Copied, never
moved**: the origin files are untouched, and a third copy remains in the private
data root as a frozen backup. Both files are read from *here* at run time,
resolved by `routegate.data.pilot_reference_dir()`; their hashes are still
checked against `manifests/pilot_reference.json` before use, because being
committed makes a file present, not trustworthy.

| provenance field | `layerwise_eb_probe_metrics.csv` | `best_layer_oof_scores.csv` |
| --- | --- | --- |
| **origin** | workspace `SteerMoE-main` | workspace `SteerMoE-main` |
| **origin-relative source path** | `results/rq3_residual_geometry_20260805/layerwise_eb_probe_metrics.csv` | `results/rq3_residual_geometry_20260805/best_layer_oof_scores.csv` |
| **source SHA256** | `a3b900dda5fd3a871589bc7e0efffe430b1faeeadd0def30871feda6a5508ee6` | `49ac26c7ae8cefac4868063053cdc57013c8aa3005518618dcda9c2e4713b539` |
| **creation date / run identifier** | 2026-08-05, run `rq3_residual_geometry_20260805` | 2026-08-05, run `rq3_residual_geometry_20260805` |
| **dependent reported result** | the pilot reproduction gate: state 45 must agree within 1e-6 on `roc_auc`, `pr_auc` and `balanced_accuracy`, and the max abs difference over all 49 states is reported | the frozen fold assignment. It is the **only** serialised record of the pilot's splits, so the folds are imported from it rather than rebuilt; without it no pilot number is reproducible |

Size 5 546 and 22 131 bytes. Neither contains prompt or response text. Both
digests are unchanged from the origin — verified at the origin, in the data-root
backup, and here.

## What belongs here

A file qualifies only if **all four** hold:

1. **Small.** Kilobytes as a rule. The single bounded exception is a frozen
   actuator — see *Model artifacts* below. Everything else that is large enough
   to make you hesitate before committing it does not belong here.
2. **Frozen.** It will never be regenerated in place. A file that a run rewrites
   is a run output and belongs in `outputs/`.
3. **Hash-verified.** Its SHA256 is recorded in a manifest and checked before
   use. A file whose hash differs is refused, not adapted to.
4. **Required for reproduction.** A reported result depends on it. Convenience
   copies, intermediate scratch and "might be useful later" files do not qualify.

The two files from the residual-geometry pilot, now promoted (see above):

| candidate | size | SHA256 | why it is required |
| --- | --- | --- | --- |
| `layerwise_eb_probe_metrics.csv` | 5.5 KB | `a3b900dda5fd3a871589bc7e0efffe430b1faeeadd0def30871feda6a5508ee6` | the per-state reference metrics the scikit-learn reproduction is checked against, to 1e-6 |
| `best_layer_oof_scores.csv` | 22 KB | `49ac26c7ae8cefac4868063053cdc57013c8aa3005518618dcda9c2e4713b539` | the **only** serialised record of the pilot's fold assignment; the folds must be imported from it, not rebuilt |

## What does not belong here

- **Large activation arrays, model checkpoints and weights.** These do not
  belong in git. The residual activation array is the standing example: 184 MB,
  `[480, 49, 2048]` float32, memory-mapped in place from the path in
  `configs/paths.local.json` and never copied into this repository. What decides
  the question is what a file *is* — its size and its kind — not the extension on
  the end of its name. See *Model artifacts* below for the one bounded
  exception.
- **Anything prompt-bearing.** Four files carry prompt or response text and none
  of them may enter git, regardless of size:

  | file | what it carries |
  | --- | --- |
  | `rq3_dataset_final.csv` — the development corpus | prompt text |
  | `manifest.csv` — the residual extraction manifest | prompt text (a `prompt_text` column) |
  | `rq3_static_steering_ebh_master_20260811.csv` — the static E/B/H master | prompt text **and** before/after response text |
  | `outputs/causal_repair/donor_pairs_with_text.csv` | prompt text; a historical RouteGate output, not one of the six frozen inputs |

  The first three are frozen inputs, resolved from the machine-local data root.
  Prompt text is study material, not source code.
- **Anything held-out.** No prompt, no prompt hash, no path to the sealed set,
  ever. See `manifests/sealed_set_commitment.json`. This was the policy before
  the external evaluation, and it held until the evaluation and its assessment
  were complete. Afterwards, two committed records added per-prompt metadata,
  including prompt SHA256 digests, for a small number of external prompts. They
  are listed under "Held-out metadata published after evaluation" in
  [Benchmarks and results](../docs/benchmarks_and_results.md#external-evaluation-380).
  No held-out prompt or reply text has been committed.
- **Run outputs.** Those live in `outputs/`, which is gitignored because run
  records are reproducible from the inputs and the committed code.
- **Base model weights and tokenizer caches.** Pinned by revision, downloaded
  to the local HuggingFace cache, never vendored.

## Model artifacts: the frozen actuator

There is no blanket prohibition on `.pt` files here. The question a reviewer has
to answer is what the file *is*, not what it is called.

A **small, model-specific frozen actuator** may qualify as a committed artifact.
It qualifies only if **all six** hold:

1. **Required for inference.** A reported result cannot be reproduced without
   it — it is not a convenience copy of something derivable.
2. **No prompt text.** Not one prompt, not one fragment, in any field or
   attribute.
3. **Legally redistributable** under whatever licence RouteGate settles on, with
   that determination recorded rather than assumed.
4. **SHA256 recorded**, in a manifest, and checked before use.
5. **Fingerprint recorded** — shape, dtype, and the identifying digest the
   actuator's own construction code emits — so that a file carrying the right
   hash but the wrong contents is still caught, and so that the artifact can be
   tied back to the run that built it.
6. **Loaded through a safe, verified loader** that checks the hash and the
   fingerprint before the tensor is used, and that does not execute arbitrary
   pickled code on load.

Point 6 is why the file format was an open question. A raw `.pt` is a pickle, and
loading a pickle executes whatever the pickle contains.

**Decided 2026-08-21, against the real file.** The historical
`frozen_directions.pt` is **kept exactly as it is and not converted**. Converting
it would produce a file whose SHA256 is not
`ee23afd98c5b9c39cc864af0e475392a69deb5ed2ef33a30d9f10703b5cb2bea`, and that
digest is the only thing tying the actuator to the three completed runs that used
it — a converted file would have to be trusted on the say-so of the conversion.

The safety concern is answered by the reader instead. `routegate.actuator` calls
neither `pickle` nor `torch.load`: it re-hashes every migrated file, checks the
archive's member set, sizes, byte order and format markers, decodes `data.pkl`
with `pickletools.genops` — a byte-stream decoder that constructs nothing and
imports nothing — to confirm the three expected globals and the metadata digest,
and then reads the tensors from the raw storage members with an explicit
little-endian float32 dtype. Each storage is bound to its variant by content, not
by position. See `manifests/rq2_actuator.json` and
`docs/decision_log.md` (2026-08-21).

**Still no actuator has been promoted to this directory.** The freeze meets
criteria 1, 2, 4, 5 and 6, but **criterion 3 is unresolved**: the licence
determination in `docs/provenance.md` has not been made, and an artifact cannot
be committed as redistributable before it is known to be. The ten migrated files
therefore live in the machine-local data root — outside git, resolved through the
`rq2_actuator_dir` key in `configs/paths.local.json` — and the tensor archive
would in any case need its own per-file `.gitignore` negation before `git add`
could see it. None has been added. The two pilot-reference CSVs described at the
top of this file remain the only promoted artifacts in this directory.

## A gitignore interaction you must handle before adding a file

The repository-wide data rules in `.gitignore` — `*.csv`, `*.jsonl`, `*.npy`,
`*.parquet` and friends — are deliberately broad and apply here too. **The two
pilot files are `.csv`, so they were ignored by `*.csv` when they first landed
here; they are admitted now only because each has its own exact per-file
negation at the end of `.gitignore`.** Nothing about `artifacts/` exempts a file
from the broad rules.

That is intentional: it means no data file lands in git by accident. When a file
is genuinely promoted to a frozen artifact, add a **targeted, per-file**
negation, next to the others, naming the file explicitly:

```gitignore
# Frozen artifacts -- one line per file, each justified in artifacts/README.md
!artifacts/pilot_reference/layerwise_eb_probe_metrics.csv
```

Both promoted files carry exactly such a line at the end of `.gitignore`.

Never un-ignore `artifacts/` as a whole, and never un-ignore a pattern. One line
per file, added at the moment that file is promoted, so that `git add` is always
a deliberate act.

## Recording an artifact

When a file is promoted, record **all five** of the following in the same commit.
A file here without that record is indistinguishable from a stray copy.

| field | what it must say |
| --- | --- |
| **origin** | the repository or workspace the file came from, by name — e.g. `SteerMoE-main` |
| **source path** | the path **relative to that origin's root** — e.g. `results/rq3_residual_geometry_20260805/best_layer_oof_scores.csv` |
| **source SHA256** | the digest of the file as it stood at the origin, before any copy |
| **creation date / run identifier** | when it was produced, and the run id if it has one |
| **dependent reported result** | which reported result stops being reproducible without it |

**Do not record a permanent absolute path.** `/home/ucabh21/...` is a fact about
one home directory on one cluster: it is meaningless to any other reader, it
breaks the moment anything moves, and it leaks a directory layout into
documentation that may become public. Origin name plus origin-relative path is
both more portable and more informative, and it is what a reader actually needs
to locate the file in the origin repository.

The exception is **frozen run records**. Where a completed run wrote an absolute
path into its own record, that path stays. The record is evidence of what
happened and where the run actually read from; rewriting it would falsify that
account and invalidate the hashes the record already carries.
