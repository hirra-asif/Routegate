# Final evaluation: original assessment report

This report covers the offline analysis of the completed final evaluation
(`routegate_final_evaluation_v5`): 1,740 generations across nine arms and the
sealed-judging release `routegate_sealed_judging_v1`. It is tied to the baseline
and response adjudication snapshots imported on 9–10 September 2026. The
original generations, rubric files, verdicts and group assignments are
preserved; the report changes no label.

## Main observed results

| Outcome | Dynamic strength | Safety-first RouteGate |
| --- | --- | --- |
| E: recorded substantive verdicts (SA/PA-S) | 6/38 | 10/38 |
| E: substantive verdict and reply changed from baseline | 5/38 | 8/38 |
| B: material answer-quality degradation | 2/78 | 1/78 |
| H: newly harmful assistance among baseline-NO prompts | 4/161 | 1/161 |
| tinyMMLU raw accuracy | 79/100 | 79/100 |
| tinyGSM8K raw accuracy | 95/100 | 91/100 |

Baseline raw accuracy is 79/100 (tinyMMLU) and 91/100 (tinyGSM8K). Baseline
harmful assistance is already YES on 39/201 H prompts, so harmful edited answers
are not all new regressions. One baseline safety verdict and one RouteGate
verdict are UNCLEAR. RouteGate opens its gate on 30/380 sealed prompts and on
none of the 200 capability prompts; all 350 gate-closed sealed replies and all
200 capability replies match baseline byte for byte.

### Qualifications

- Six prompts have conflicting labels for byte-identical replies across the
  baseline and response phases. Two are E prompts, so three candidate cells
  appear repaired without their reply changing. The conflicts involve the
  earlier AI-assisted review as well as original judge decisions. The report
  preserves them, names the exact rows and adds a changed-text sensitivity
  check. It does not decide which conflicting label is correct; see the
  [consistency audit](CONSISTENCY_AUDIT_README.md).
- Of the 380 sealed prompts, 35 have unresolved permission and 15 ambiguous
  permission. They are excluded from E/B/H rates and retained in the coverage
  tables. Another 13 other-permitted prompts are assessed separately.
- The frozen final runner makes a preliminary activation-capture pass for both
  edited systems. These generations do not demonstrate a single-pass runtime
  advantage for dynamic strength.
- This is a complete-system comparison: directions, intervention layers and
  application rules differ. It does not isolate the gate's effect; the
  [static gate ablation](../../experiments/static_ungated_followup_v1/README.md)
  does that for the fixed edit.
- The legacy scorer's `delta_confidence_interval` is an interval for the share
  of gains among discordant pairs. Its prediction rules are preserved; the
  exporter separately calculates correctly labelled paired-difference intervals.
- No tinyBenchmarks IRT/pIRT/gpIRT estimate of full-benchmark accuracy was
  computed. Only raw 100-item subset accuracy is reported.

## Rebuild

With the completed records under the data root, rebuild inside the project
container (see [environment](../environment.md)):

```bash
bash experiments/final_evaluation/run_final_thesis_results.sh
```

or call the exporter directly:

```bash
python experiments/final_evaluation/build_final_thesis_results.py \
  --repo "$PWD" \
  --data-root "$ROUTEGATE_DATA_DIR" \
  --output-dir outputs/thesis_results/final_evaluation
```

No GPU, judge credential, model download or generation is needed. NumPy, pandas,
a Parquet engine (pyarrow), SciPy and Matplotlib are required. If an integrity
check fails, report the named file; do not change frozen scoring code or hashes.

Outputs under `outputs/thesis_results/final_evaluation/`:

| File | Contents |
| --- | --- |
| `RESULTS_REPORT.md`, `RESULTS_REPORT.html` | Readable report with figures |
| `analysis.json` | Numerical results, definitions and limitations |
| `validation.json` | Source-integrity checks, separate from annotation conflicts |
| `input_hashes.json` | Every source file checked |
| `tables/`, `pdf/`, `svg/` | LaTeX tables and editable figures |

## Statistics and provenance

Capability predictions use the existing `routegate.capabilityscoring.score_arm`.
Single-rate intervals are 95% Wilson intervals; paired-difference intervals
combine two 97.5% Clopper–Pearson intervals with a union bound. These intervals
were added at reporting time and are not preregistered. They assume independent
prompts within each eligible group and do not cover judgement uncertainty or
non-random exclusions.

The pipeline's `HUMAN_ADJUDICATION` label is retained verbatim. The response
adjudication was AI-assisted: the researcher accepted 94 displayed proposals and
delegated the final three labels. It is not an independent human annotation
study.

If a documented correction later changes the pinned adjudication snapshots,
create a new reporting version that names both snapshots; do not remove a hash
check to make an altered snapshot pass.
