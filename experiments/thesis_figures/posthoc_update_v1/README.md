# Post-hoc thesis figures (13 September 2026)

Ten figures, each supplied as vector PDF, editable SVG and 300-dpi PNG. Eight quantitative figures use frozen audited outcomes; two diagrams explain the implemented computation. `data/` contains the exact plotting inputs. No language model, API call, judging, model fitting or activation array is required.

## Rebuild

From the repository root:

```bash
bash experiments/thesis_figures/posthoc_update_v1/run.sh outputs/thesis_results/posthoc_update_v1
```

The selector build also writes `selector_figure_manifest.json` (input hashes, matplotlib version) beside the figures; the copy committed here records the thesis build. Without an argument the figures go to `experiments/thesis_figures/posthoc_update_v1/Figures/` (ignored by Git). NumPy and Matplotlib are required. The wrapper uses `ROUTEGATE_FIGURE_PYTHON` if set, then the Myriad Python bundle if present, and otherwise `python3`. CPU thread counts are limited to one. Only same-named plot files in the output directory are overwritten.

## Figures

| Figure file stem | Thesis chapter | Meaning |
|---|---|---|
| how_activation_steering_works | Chapter 2 | Schematic 3D vector addition; no empirical coordinates |
| routegate_current_runtime | Chapter 3 | Verified two-pass dynamic-strength and gated-static execution |
| cancellation_controls | Chapter 5 | Researcher-approved harmful outcomes with one retained UNCLEAR |
| controller_transitions | Chapter 5 | Paired fixed-to-dynamic gains and losses |
| selector_capacity_profiles | Chapter 6 | All profiles, same two-layer edit, linear and neural selectors |
| moe_selector_profiles | Chapter 6 | All profiles, same three-layer edit, residual/routing/combined inputs |
| static_gate_ablation | Chapter 6 | Joint baseline/gated/ungated static outcomes |
| static_gate_consequences | Chapter 6 | Confirmed repairs withheld and damage prevented by each check |
| selector_fold_variation | Appendix B | Observed edit-count variation, separate fold maps |
| moe_h_source_controls | Appendix B | Within-H AUC/AP, pooled and within-source |

## Interpretation

- Two-layer capacity H means newly harmful assistance /212 known non-harmful baselines. Three-layer MoE H means harmful static replies selected /213. Do not combine these as the same metric.
- E selector totals count retained fixed-edit substantive answers, not complete-system repair.
- Profile names specify inner-validation budgets, not guarantees on outer-fold outcomes.
- Unknowns remain separate from known counts. All 19 cancellation decisions have final researcher approval; one UNCLEAR outcome remains explicit. The hatched extension is not a confidence interval.
- The static follow-up uses a previously evaluated external set. Its joint assessment does not silently replace the original dynamic assessment.
- The runtime is two-pass and edits only the final prompt token during prefill. No edit is directly applied to generated-token positions. The vector-addition picture is a schematic.

## Version control

The nine aggregate files under `data/` are the plotting inputs and are explicitly allowed by `.gitignore`. They contain plotting summaries, not prompt or response text. `data/static_report.json` is identical in content to the static ablation's `scientific_report/report.json`. Generated PDF/SVG/PNG files are not committed.
