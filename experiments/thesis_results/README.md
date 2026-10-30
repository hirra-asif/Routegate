# thesis_results

One stage, and it is a reporting stage: it derives the numbers the thesis
results section shows, from result files that are already complete, and draws
them. It loads no model, generates nothing, judges nothing, fits nothing,
calls no service and never reads the sealed set.

```
python experiments/thesis_results/build_frozen_development_figures.py --dry-run
python experiments/thesis_results/build_frozen_development_figures.py
```

The derivation and its gates live in `src/routegate/thesisresults.py`; this
directory holds orchestration and the plotting only.

## Where the numbers come from

| reading | file | resolved through |
| --- | --- | --- |
| static B-E outcomes, per row, at `alpha_R = 0.15` | `rq3_static_steering_ebh_master_20260811.csv` | `configs/paths.local.json → static_steering_master` |
| residual sensor, out-of-fold, and its frozen thresholds | `outputs/routegate_residual_v1/{oof_predictions.csv,thresholds.json}` | the repository |
| all three sensor families and their frozen thresholds | `outputs/routegate_routing_analysis_v1/{oof_predictions.csv,thresholds.json}` | the repository |
| phase-2 causal screen | `phase2/judging/readings/phase2_readings.json` | `configs/paths.local.json → focused_direction_validation_dir` |
| the 180 per-cell adjudicated safety verdicts | `phase2/judging/readings/analysis_table.jsonl` | the same |
| probe-head stored metrics, out-of-fold scores, the run's own checklist and its manifest | `outputs/dual_direction_controller/probe_heads_v1/{metrics,validation,run_manifest}.json`, `oof_predictions.csv` | the repository |

Every one is SHA-256'd, checked against `EXPECTED_INPUT_DIGESTS`, and recorded.

## The three systems

All three read the **same** per-row outcome table. They differ only in which
rows were steered, so a row a gate did not fire on is a denominator row that
keeps its baseline outcome — an E row stays refused, a B row is not degraded,
an H row is not made harmful.

| system | steered rows |
| --- | --- |
| `static_be_actuator_alpha_r_015` | every development row; no sensor, no gate |
| `routegate_safety_first` | `residual p_E >= threshold_E[residual] AND residual p_H < threshold_H[residual]` |
| `routegate_higher_recall` | `residual p_E >= threshold_E[residual] AND residual+routing p_H < threshold_H[residual+routing]` |

The gate rule and both thresholds are the frozen ones; nothing is fitted here.

`routegate_safety_first` is the frozen RouteGate v1 gate, and the derived gate
is checked against the `gate_full` column both run records already carry.

### Both RouteGate names are reconstructed labels

Neither `routegate_higher_recall` nor `routegate_safety_first` is quoted from a
run-name field. **No original run artifact records either phrase**, and none is
claimed to. Each label was assigned here to a frozen sensor/threshold
composition, identified by the development counts that composition reproduces:

| label | denotes | standing |
| --- | --- | --- |
| **RouteGate higher-recall** | the unique frozen composition reproducing **54/89** E substantive repairs, **0/178** material B degradations and **1/213** actionable H regressions — the residual E sensor at its frozen `threshold_E`, braked by the residual+routing H veto at its frozen `threshold_H`. Every pairing of a frozen E sensor with a frozen H veto was enumerated; this is the only one yielding those three counts. | reconstructed |
| **RouteGate safety-first** | the frozen RouteGate-v1 gate, reproducing **37/89**, **0/178** and **0/213** — the residual E sensor braked by the residual H veto. The *gate* is recorded: it is the `gate_full` column of `outputs/routegate_residual_v1` and the `residual_gate_full` column of `outputs/routegate_routing_analysis_v1`, and the derivation is checked against both row for row. It is the *phrase* that is assigned here. | reconstructed |
| **Static B-E actuator, alpha_R = 0.15** | the actuator applied to every row; no sensor composition to identify. | recorded |

This is a reconstruction of a **composition**, not a re-derivation of a
**result**. The sensor families, the thresholds and the per-row outcomes are
frozen and are used exactly as they stand; every count is unchanged. What was
assigned is the English label.

The standing travels with the system rather than sitting in a footnote:
`SystemSpecification.name_standing` carries it, `table_1_system_comparison.csv`
has a `name_standing` column, the LaTeX table marks reconstructed names with
`ᴿ`, `routegate_results_so_far.md` sets out the mapping in full, and
`provenance_manifest.json` records it under `system_name_provenance`. A test
asserts each reconstructed label names the counts it denotes.

**Confirm the intended configuration before this goes into the thesis.**

## What it refuses

- a sealed-looking input, output directory or command-line path, and a sealed
  key or sealed path inside any record it reads. A run record whose *prose*
  says the sealed set was not used is recorded as a disclaimer, not refused;
- a superseded dataset revision, by filename, before opening it;
- a raw Sonnet, Opus or human judge column where an adjudicated `final_*`
  verdict belongs, and any prompt or response text column at all;
- a verdict outside the declared vocabulary, and an undecided verdict counted
  as though it were decided;
- an output directory that already holds a run;
- a derivation that has moved off `FROZEN_CHECKPOINTS` — the one place a known
  count appears, read only after the derivation and never fed into one.

## The probe heads (Figure 4, Tables 4 and 5)

Read from one frozen run, `outputs/dual_direction_controller/probe_heads_v1`.
Nothing is fitted, refitted or trained here, and no controller is touched.

The **reported metrics are the ones the run stored**. The out-of-fold
predictions are read to *check* them — that the orientation is what it says,
that the eligible rows are the ones the task defines, that each was scored once
— and to bin the calibration, which the run did not store. A stored metric that
no longer follows from the stored predictions stops the run.

| head | task | eligible rows |
| --- | --- | --- |
| repair | E versus B | E and B only. H is excluded from its training labels and its scaler; H rows are still scored but have no target and enter no metric. |
| safety | H versus (E + B) | every row. E and B are one negative class, not two. |

**State 32** — the output of historical decoder block 31, at the final prompt
token — is the principal figure and the primary table row. It was predeclared
by the frozen run before either head was fitted; this stage checks that claim
and refuses a run whose primary state has moved, that selected a state from its
own metrics, or that promoted a sensitivity state. States **33 and 34** appear
as sensitivity rows in Table 4 only: they are not drawn, and they carry no
checkpoint, because a brake on them would be a second place the run could be
judged from.

**No calibration model is fitted and none is applied.** The probabilities are
the logistic outputs exactly as the run emitted them, labelled *uncalibrated
OOF probabilities* everywhere they appear. Brier is the stored
calibration-sensitive score, read from `metrics.json`. ECE is derived here and
labelled as such — a newly derived descriptive statistic, never a substitute
for a stored one — and the fixed-bin definition is stated wherever the number
is. The bins are ten fixed equal-width intervals over [0, 1], half-open below
and closed at the top, identical for both heads and every state. Every bin,
**including any that is empty**, is exported to
`table_5_oof_calibration_bins.csv` with a count of zero and no statistics; the
figure's lower panel keeps empty bins visible on the axis rather than closing
the gap.

## What it writes

Four figures as PNG (400 dpi) and SVG, `table_1..5` as CSV, a booktabs
`routegate_results_tables.tex`, `routegate_results_so_far.md` with the
captions, `provenance_manifest.json` (git state, every input and output digest,
every derived count, the full checklist, the system-name provenance, and the
H-XX to source-row-id mapping) and `output_hashes.json`.

Figures and tables are byte-identical across runs over the same inputs;
`provenance_manifest.json`, `routegate_results_so_far.md` and
`output_hashes.json` carry `created_utc` and are run records, so they are not.
