# Final system evaluation

The completed evaluation compares baseline, dynamic-strength steering
(`dynamic_raw`) and gated static steering (`safety_first_routegate`) on 380
external prompts, with separate 100-question capability checks.

The frozen run is specified by
[`routegate_final_evaluation_v5.json`](../../manifests/routegate_final_evaluation_v5.json).
Both edited systems make an unsteered capture pass before the edited prompt pass.
Dynamic strength uses `d^(1)` at blocks 31–32; gated static steering uses
`d^(23)` at blocks 31–33. Their comparison measures complete systems.

## Read and rebuild the report

See [the original assessment report](../../docs/final_evaluation/README_FINAL_ANALYSIS.md)
and [consistency audit](../../docs/final_evaluation/CONSISTENCY_AUDIT_README.md).
With the completed records available under `$HOME/routegate-data`, rebuild the
report on Myriad:

```bash
bash experiments/final_evaluation/run_final_thesis_results.sh
```

This exports existing results; it does not run generation or call judges.
Outputs normally appear under `outputs/thesis_results/final_evaluation/`.

## Assessment versions

The original response assessment reports E substantive outcomes of 6/38 for
dynamic strength and 10/38 for gated static steering. Identical-text label
conflicts limit their interpretation as newly repaired refusals.

The subsequent [static gate ablation](../static_ungated_followup_v1/README.md)
jointly assesses baseline, gated and ungated static replies. It isolates
selection by holding the static edit fixed. Use its own denominators and labels;
do not mix its revised static counts into the original dynamic comparison.

## Source and records

Generation, rubric preparation, judge calls, review and export are separate
stages. The entry points remain available for documented reproduction, but
completed generations and decisions should not be overwritten. Full reproduction
needs the frozen inputs, model snapshot, container and assessment records.
