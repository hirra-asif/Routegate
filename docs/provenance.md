# Provenance

This file records **where the work in this repository was actually done**. It
exists so that consolidating the code into RouteGate cannot be mistaken for a
claim that the experiments were originally run from RouteGate. They were not,
and the record must say so.

Recorded: 2026-08-19.

## The two repositories

**`~/SteerMoE-main` — the experimental working repository.**
Early RQ1 and RQ2 development, the pathway/patching pilots, the frozen RQ2
actuator, the static E/B/H steering runs, the corpus construction and judging,
the residual extraction and the residual-geometry pilot were all developed and
executed there, between June and August 2026. It is a working tree: it holds
superseded label revisions, abandoned attempts and exploratory scripts alongside
the accepted ones. It is treated here as **read-only historical evidence** and is
never modified, renamed, initialised as a Git repository, or written into.

That directory is also a working copy of the Adobe Research **SteerMoE**
repository (*Steering MoE LLMs via Expert (De)Activation*, arXiv:2509.09660),
under the Adobe Research License, which permits noncommercial research use and
requires the licence to travel with any redistribution. The upstream files —
`LICENSE`, `README.md`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`,
`requirements.txt`, the two notebooks, `src/utils.py`, `src/modeling_vllm*/`,
`assets/`, `activations/` — are Adobe's.

The `rq1_*`, `rq2_*`, `rq3_*`, `pathway_*` and `exp3a_*` files sit in that same
directory. **Their status is not settled, and this document does not settle it.**

What the audit found is *evidence*, and it points one way:

- they post-date the upstream checkout — the upstream files are dated
  2025-09-12, these are dated 2026-06 through 2026-08;
- none of them carries an Adobe copyright notice, whereas `src/utils.py` does
  (`# Copyright 2022 Adobe / All Rights Reserved.`);
- none of them imports the Adobe package: no `import src`, no `from src.utils`,
  no upstream module anywhere in the transitive import closure.

That evidence **suggests these files were written for this project** rather than
derived from the Licensed Materials. It is not a legal determination.
**Timestamps, import graphs and absent copyright headers are not conclusive
evidence of authorship or non-derivation** — a file can be adapted from another
without importing it and without inheriting its header, and mtimes record when
bytes were last written, not who wrote them or what they were based on.

Three rules follow, and they hold regardless of how the question is eventually
answered:

1. **Authorship and derivation must be confirmed** — by the author, and where
   necessary by the supervisor or department — **before a public licence is
   selected for RouteGate, and before any of those files is copied into it.**
   Both decisions are downstream of that determination, not independent of it.
2. **No Adobe-owned source file is to be copied into RouteGate.** Not the
   package, not the notebooks, not the modelling code, not `src/utils.py`.
3. **RouteGate may refer to and cite SteerMoE historically** — that is exactly
   what this document does, and the citation is required for honest provenance —
   **but it must not import, vendor or redistribute Adobe source code.**

To state the current position precisely rather than broadly: **no Adobe-owned
source code is intended to be copied into RouteGate, imported by it, or required
by it at run time.** Historical citation and reference are a separate matter and
are expected to continue.

**`~/RouteGate-MSc`** — an earlier supervisor-facing package. Also read-only
historical evidence; not a dependency of anything here.

**`~/routegate` (this repository) — the standalone final implementation.**
Started 2026-08-05. It is being consolidated into the single repository that
holds all reportable RQ1, RQ2 and RouteGate code, so that the final artefact can
be read, reviewed and re-run without needing the experimental tree. The
consolidation was carried out from August to September 2026;
`docs/standalone_migration.md` records its steps, and the
[experiment index](experiment_index.md) lists what remains historical.

## What consolidation does and does not change

Consolidation moves **code** into one place. It does not re-run history.

- **Historical results keep their original hashes and their original
  provenance.** The frozen corpus (`0943449697327aa47912ef80337462414ab98255c3ad99b717baa716353e1d08`),
  the extraction manifest (`a3b7444b1a2782b5bc04142f3545ac5d71ad57a5f546edc315d7d51d14947abd`),
  the residual array (`af89872d5decd4dc432ec579a941e6247bc7de63ff5690432e9f5812a994e2d8`),
  the static-steering E/B/H master (`525e62e0a6fdac3200111deae146d699ea2db9b3459dab2e57b09c705b11b9e4`)
  and the two pilot reference files (`a3b900dda5fd3a871589bc7e0efffe430b1faeeadd0def30871feda6a5508ee6`,
  `49ac26c7ae8cefac4868063053cdc57013c8aa3005518618dcda9c2e4713b539`) were all
  produced in `~/SteerMoE-main` and carry those hashes there. Relocating a file
  does not change its SHA256, and a relocated file whose hash differs is refused,
  not adapted to.
- **Settings recovered from the historical implementation stay attributed.**
  `configs/causal_repair.json` names the exact source files the patching
  operation, the hook site, the budget reference, the historical layer indices
  31/32/33 and the strengths 0.2 / 0.15 were recovered from. Those attributions
  are load-bearing: without them the settings would look invented. They are kept
  verbatim.

  **Corrected 2026-08-21.** This paragraph previously added that
  `configs/causal_repair.json` "is embedded in the direction bank's
  `config.json` and covered by its `manifest_sha256`". It is not. The config
  embedded verbatim in `direction_bank/config.json` — at its `protocol` key, and
  again at the scan run record's `config` key — is
  `configs/orthogonalized_population_scan.json`, and that is the one an edit
  would invalidate. `configs/residual_geometry.json` is likewise embedded in
  `outputs/residual_geometry/corrected/config.json`.
  `configs/causal_repair.json`'s own digest is recorded nowhere under
  `outputs/`, `artifacts/` or `manifests/`.

  Being editable is not the same as being casually editable. It is still a
  frozen record of settings recovered from history, `tests/test_patch_states.py`
  reads it, and the attributions in it are the reason the settings are not
  arbitrary. Two of its strings were contradicted by the located grid summaries:
  `historical_strengths.rq1_grid_best: 0.2`, and a `strength_rationale` calling
  0.2 "the RQ1 grid's first-ranked window-1 configuration" when in every
  completed summary 0.15 outranks 0.2 at those layers — 0.2 placing 11 of 18,
  3 of 8 and 2 of 2, with one harmful break in each completed run.

  Stated precisely, because the first version of this correction overstated it:
  0.15 is the **band-level preference** at layers 31–33, ahead of 0.2 in all
  three completed summaries, and it is the **overall first place only in the
  final 512 confirmation** — its overall ranks were 2 of 18, 2 of 8 and 1 of 2.
  It was not the outright winner of every completed grid, and all of these counts
  are provisional keyword readouts rather than semantically confirmed optima.

  **Both are corrected as of 2026-08-21**, after an exhaustive check that no
  digest and no embedded copy of the file or of any of its load-bearing sections
  exists under `outputs/`, `artifacts/`, `manifests/`, `logs/` or the data root.
  The correction is additive: `rq1_grid_selected: 0.15` and
  `rq2_frozen_actuator: 0.15` are now stated with their evidence,
  `population_scan_pilot_value: 0.2` is named separately as a pilot value and
  explicitly not a measured optimum, and the superseded `rq1_grid_best: 0.2` is
  **kept inside the file** under `superseded_value`, with why it was wrong and a
  pointer to `docs/decision_log.md`. Nothing was deleted, and the scan's own
  declared `intervention.strength` of 0.2 is unchanged — only its description
  was wrong, not the protocol.
- **`~/SteerMoE-main` is not rewritten to look like RouteGate**, and RouteGate is
  not backdated to look like it produced work it did not.

## The statement that must not be made

The final repository must **not** present historical experiments as having been
originally run from RouteGate. Specifically:

- Run records, manifests and figures produced in the experimental repository are
  labelled as such. A copied artifact is a copy, and carries **all five**
  provenance fields required by `artifacts/README.md`: the **origin repository or
  workspace name**, the **source path relative to that origin's root**, the
  **source SHA256**, the **creation date or run identifier**, and the **dependent
  reported result**.

  Provenance records use origin-relative paths. **A permanent absolute
  `/home/ucabh21/...` path is not required and should not be written into a
  committed artifact record or into public documentation** — it is a fact about
  one home directory on one cluster, it breaks as soon as anything moves, and it
  leaks a directory layout into material that may be published. The origin name
  plus the origin-relative path is more portable and tells a reader more.

  The exception is **frozen run records**: where a completed run wrote an
  absolute path into its own output, that path stays. Those records are evidence
  of where the run actually read from, and rewriting them would falsify the
  account and invalidate hashes the records already carry.
- Where RouteGate *re-derives* a historical number (the residual-geometry pilot
  is the standing example — RouteGate re-computes it with scikit-learn because
  the pilot hand-wrote its estimator, scaler, splitter and metrics), the write-up
  says "reproduced", names the tolerance, and reports the reference alongside the
  reproduction. It does not present the reproduction as the original.
- Where the code that produced a historical artifact **has not been located**,
  that is stated rather than implied away — and stated as a gap, not as a
  permanent property. `rq3_static_steering_ebh_master_20260811.csv` is the known
  case: it exists with a full validation report naming its eleven source files
  and their hashes, but no script found so far writes it. The file is preserved
  unchanged as frozen historical evidence, SHA256
  `525e62e0a6fdac3200111deae146d699ea2db9b3459dab2e57b09c705b11b9e4`. The
  original builder is to be recovered if it exists; failing that, a deterministic
  builder is to be reconstructed from those eleven pinned sources and must
  produce a byte-identical CSV before it is accepted. Until one of those
  succeeds, the master is reported as frozen historical evidence with its missing
  builder acknowledged — neither claimed re-runnable nor written off as
  irreproducible. See `docs/standalone_migration.md`.
- Code that is moved into RouteGate unchanged is described as moved, not as
  newly written here.

## Held-out material

Unchanged by any of this, and stated here because provenance and blinding are
easy to conflate: the held-out benchmark split is not present in this repository.
`manifests/sealed_set_commitment.json` records its size and composition only —
380 rows, 180 safe-source, 200 harmful-source. No prompt, no prompt hash and no
path to it exists here; no module can infer its location; the input gates refuse
any path that looks like held-out material; and a test asserts that the
commitment file itself carries no absolute path, no reference to the historical
workspace, and no 64-hex digest. Consolidation does not open it, and Phase 2
must not create a path that could.

**Post-evaluation note (2026-10-13).** The paragraph above describes the
repository before the split was opened for the final evaluation (generation
from 8 September 2026). It remains true that no held-out prompt or reply text,
and no path to it, is committed. Since the evaluation was completed, however,
the public tree does include held-out-derived metadata: the sealed-judging
release manifest pins the generated external-set records by digest, and
`docs/final_evaluation/consistency_review_v1.public.json` and
`experiments/thesis_figures/posthoc_update_v1/data/static_audit.json` identify
individual external prompts by row identifier and prompt SHA256, with their
labels. See
[Benchmarks and results](benchmarks_and_results.md#external-evaluation-380).
