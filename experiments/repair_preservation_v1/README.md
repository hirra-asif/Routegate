# Repair-preservation follow-up (prepared, not executed)

Preparation and scoring code for an E/B/H development comparison of a selected
edit. It was written to follow the [cancellation diagnostic](../cancellation_controls_v1/README.md).
No candidate was bound and no generation was run for this stage; the later
[selector comparisons](../outcome_selector_v1/README.md) reuse completed
development outputs instead. It is kept because its input preparation and
paired scoring are reusable.

Implemented: verified development inputs, the existing five folds, assessment
preparation with the original rubrics, paired-results scoring and a status
command. Not implemented: a generation runner for a chosen candidate.

## Prepare inputs

```bash
bash experiments/repair_preservation_v1/run.sh prepare-inputs
bash experiments/repair_preservation_v1/run.sh selftest
bash experiments/repair_preservation_v1/run.sh status
```

Preparation should report **480 rows: E89, B178, H213**, five existing folds,
zero new generations and zero API calls. Data are written to
`$ROUTEGATE_DATA_DIR/repair_preservation_v1/` (default `$HOME/routegate-data`).

## What the assessment code will measure

| Population | Main outcome |
|---|---|
| E: previously over-refused permitted requests | Substantive answer, using the existing SA or PA-S definition |
| B: previously answered benign requests | Material loss of answer quality relative to the fresh baseline response |
| H: prohibited requests | Newly harmful assistance among prompts whose fresh baseline gives no harmful assistance |

The frozen E/B/H populations remain visible. Fresh baseline behaviour is checked separately. Unknown judgements, unanswered assessments, capped responses and changes in the baseline are reported explicitly. Comparisons between conditions pair the same prompts.

The input folds support development fitting and evaluation. They are not proof of transfer to new sources. No model or selector has been fitted by these preparation scripts.

## Later stages

A concrete comparison specification would be supplied to `bind-plan`. That freezes the chosen conditions, generation settings and supporting diagnostic report before fresh generation. The number of generations and judge calls depends on those conditions.

The later assessment stages are `prepare`, `build`, `dryrun`, `judge`, `join`, `export`, `adjudicate`, `report` and `score`. They use the established judging engine and require complete, matching future generation records. Exact-material duplicate handling is specified before judge calls; old labels are not imported as new results. Review provenance records whether AI assisted.
