# Experiment index

Current navigation for the thesis study, reviewed against the 13 September 2026
repository export and updated for the public release in October 2026. Results
and evaluation sets are summarised in [benchmarks and results](benchmarks_and_results.md);
the software environment is in [environment](environment.md). Some directories retain their original prospective protocols.
Their presence does not establish that the proposed run was completed.

## Causal pathway localisation

| Directory or record | Role |
| --- | --- |
| [provenance.md](provenance.md) | Historical RQ1/RQ2 work undertaken in `SteerMoE-main` |
| [triplet_validation](../experiments/triplet_validation/README.md) | Retrospective assessment of the ten operative matched cases |
| [causal_repair](../experiments/causal_repair/README.md) | General patching and population-scan implementation; its proposed scans are separate from the completed ten-case thesis evidence |
| [two_site_judging](../experiments/two_site_judging/README.md) | Assessment of the two-site steering comparison |

The historical source pipelines are not all included in this repository.
Their absence here is a reproduction boundary, not evidence that the experiments
were never run.

## Reusable and adaptive steering

| Directory | Role and status |
| --- | --- |
| `residual_geometry/` | Completed pilot reproduction and corrected residual-geometry analysis; local records under `outputs/residual_geometry/` |
| [focused_direction_judging](../experiments/focused_direction_judging/README.md) | Completed repair-direction assessment and additive H−E protective-direction screen; phase 2 closed with `NO_STRENGTH_QUALIFIES` |
| [single_direction_controller](../experiments/single_direction_controller/README.md) | Completed learned-strength development comparison and supporting diagnostics |
| [controller_error_audit_v1](../experiments/controller_error_audit_v1/README.md) | Completed paired audit of recorded baseline, fixed and dynamic replies |
| [cancellation_controls_v1](../experiments/cancellation_controls_v1/README.md) | Completed nine-condition diagnostic on 18 selected H regressions |
| [rotation_safety_direction](../experiments/rotation_safety_direction/README.md) | Separate H−E rotation diagnostic; consult versioned generation and adjudication records for executed status, rather than its original declaration banner |

The 18-prompt diagnostics concern selected harmful regressions. They do not
estimate population safety or permitted-request repair.

## Selective application and generalisation

| Directory | Role and status |
| --- | --- |
| `routegate_residual/` | Residual E/B/H predictors and historical gate analysis |
| `routegate_routing/` | Native router extraction and representation comparisons |
| `routegate_deployment/` | Frozen deployment artifacts and checks |
| [final_evaluation](../experiments/final_evaluation/README.md) | Completed baseline, dynamic-strength and gated-static external evaluation plus capability checks |
| [static_ungated_followup_v1](../experiments/static_ungated_followup_v1/README.md) | Completed fixed-edit gate ablation with joint assessment |
| [outcome_selector_v1](../experiments/outcome_selector_v1/README.md) | Completed linear class-versus-outcome selection comparison, two-layer edit |
| [selector_capacity_v1](../experiments/selector_capacity_v1/README.md) | Completed linear-versus-neural selector comparison, same two-layer edit |
| [moe_outcome_selector_v1](../experiments/moe_outcome_selector_v1/README.md) | Completed residual/routing/combined outcome selectors, three-layer edit |

The original external assessment and later joint static assessment are distinct
versions. The latter resolves shared-material judgements for its own comparison;
it does not silently replace the original dynamic-system results.

## Figures and reporting

- `thesis_results/`: development-result export code.
- `final_evaluation/`: original external-result export and consistency audit.
- `thesis_figures/`: probe and geometry figures.
- [thesis_figures/posthoc_update_v1](../experiments/thesis_figures/posthoc_update_v1/README.md): ten diagrams and post-hoc result figures.

## Historical designs and unreported work

- `dual_direction_controller/` preserves the superseded two-coefficient proposal.
- The guarded/angular/SIGR pilot was archived outside the active repository on
  14 September 2026. See [the retirement record](retired_experiments.md).
- `repair_preservation_v1/` contains a prepared follow-up pipeline; a completed
  result package is not supplied in this export.
- `moe_features/`, `dynamic_gate/` and `intervention_evaluation/` are original
  placeholder directories. The implemented work is listed above.

Frozen manifests and dated decision records should retain their original wording
and hashes. Remove or archive unused source through its import dependencies,
rather than deleting every file that mentions an abandoned method.
