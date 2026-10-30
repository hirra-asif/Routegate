# Static steering gate ablation

This completed follow-up asks which repairs the final gate withheld and which
harmful or damaging edits it prevented. It applies the same frozen `d^(23)`
edit used by `safety_first_routegate` to all 380 external prompts: strength
0.15, zero-based blocks 31–33, final prompt token during prefill.

Baseline, gated-static and ungated-static replies are assessed jointly, with
shared judgements for identical material. Disagreements pass through the existing
review workflow; for this study the 105 decided review rows are AI-assisted
proposals (ChatGPT / Codex) approved by the researcher, and one quality comparison
was left unresolved (`review_provenance.json`). The earlier final-system records
remain a separate assessment version, and the consistency review's proposed
labels are not used here.

## Results

| Joint-assessment outcome | Gated static | Ungated static |
| --- | ---: | ---: |
| Substantive answers following 37 confirmed baseline refusals | 9 | 22 |
| Newly harmful assistance /161 known non-harmful H baselines | 1 | 29 |
| Material benign-answer losses /78 B prompts | 0 | 4, plus 1 unresolved comparison |

The fixed E group contains 38 prompts; one baseline refusal judgement remains
unresolved. The confirmed-refusal repair denominator is therefore 37. In the
paired comparison, the gate withholds 13 confirmed repairs while preventing 28
newly harmful responses and all four known benign losses.

This is a follow-up on a previously evaluated external set. It identifies the
consequences of the frozen selection rule; it does not validate a newly tuned gate.

## Inspect existing results

```bash
bash experiments/static_ungated_followup_v1/run.sh status
bash experiments/static_ungated_followup_v1/run.sh progress
```

Read `$HOME/routegate-data/static_ungated_followup_v1/assessment_v1/scientific_report/READ_RESULTS.md`
and `report.json`. The expanded cell tables preserve row-level outcomes and
unknowns. `run.sh report` rebuilds the report from completed assessment records;
`run.sh export-results` archives the study at
`$HOME/routegate_static_ungated_results.tar.gz`.

## Pipeline

`preflight` checks inputs; `init` prepares the study; `generate` runs the frozen
ungated condition. `prepare-judging` builds assessment material; `judge sonnet`
and `judge opus` make judge API calls. `review` joins assessments and exports
disagreements. `import --completed <file> --provenance <file>` validates and
imports the completed human review before reporting. Existing failures are
retained; inspect them before using `--retry-failures`.

Generation is complete. Repeating generation or judging is unnecessary for
reading the results or rebuilding the thesis figures.
