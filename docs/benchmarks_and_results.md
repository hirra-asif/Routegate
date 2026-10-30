# Benchmarks and results

Every count on this page is taken from a saved record named in the
**Record** column. Nothing is recomputed or relabelled here. Development,
external and follow-up results are separate assessment versions and are not
combined.

## Evaluation sets

| Evaluation | Size | Type | Role |
| --- | ---: | --- | --- |
| Development set | 480 prompts | Constructed, labelled cohort | Method development, probe and gate fitting, exploratory comparisons |
| External evaluation | 380 prompts | Constructed held-out split | Final system evaluation; later reused for the static gate ablation |
| tinyMMLU | 100 questions | Fixed published subset of MMLU | Capability check |
| tinyGSM8K | 100 questions | Fixed published subset of GSM8K | Capability check |

None of these is a complete public benchmark. The two safety sets are
constructed from several sources; the two capability sets are the fixed
100-item subsets released by tinyBenchmarks.

### Source datasets

The development and external sets come from one prompt-only split built on
29 July 2026 (seed `20260721`; near-duplicate families by Jaccard similarity
≥ 0.6 over normalised tokens kept within one split; previously exposed families
quarantined from the held-out side). Development and held-out canonical prompt
hashes are disjoint (2,475 development-side and 380 held-out prompts).

| Source | Upstream | Split or file used |
| --- | --- | --- |
| OR-Bench-Hard-1K | OR-Bench (`or-bench-hard-1k.csv`, 1,319 rows) | over-refusal candidates |
| OR-Bench-Toxic | OR-Bench (`or-bench-toxic.csv`, 655 rows) | harmful prompts |
| StrongREJECT | official `strongreject_dataset.csv` (313 rows) | harmful prompts |
| FalseReject | `AmazonScience/FalseReject`, revision `493ba967…`, **test** split | over-refusal candidates |
| HarmBench | `centerforaisafety/HarmBench`, revision `8e1604d1…`, `harmbench_behaviors_text_test.csv` | harmful prompts; standard and contextual behaviours; the 80 copyright behaviours excluded |

Source identity is provenance, not a label. E/B/H membership is decided by
assessment of the model's own baseline behaviour (below).

### Development set (480)

A 509-row labelled cohort drawn from the development side of the split. Each
prompt has one unsteered Qwen3-30B-A3B reply, judged on a refusal-behaviour axis
(Axis 2) and a safety axis by Claude Sonnet 4.5 and Claude Opus 4.6
(`eu.anthropic.claude-opus-4-6-v1`); disagreements were resolved by a human
reviewer (113 rows). A later blind Axis-2 re-audit of the E rows moved 16 to B
and 9 to excluded. 29 partial or unclear rows are excluded, leaving 480.

| Class | Meaning | n |
| --- | --- | ---: |
| E | benign prompt refused at baseline (over-refusal) | 89 |
| B | benign prompt answered at baseline | 178 |
| H | harmful prompt correctly refused at baseline | 213 |

By source family (from the extraction manifest): OR-Bench-Hard 217 (E 82,
B 135), FalseReject 50 (E 7, B 43), StrongREJECT 119 (H), OR-Bench-Toxic 62 (H),
HarmBench 32 (H). The E/B contrast is therefore confounded with source; a
source-only probe reached ROC-AUC 0.555. Details: [data lineage](data_lineage.md).

### External evaluation (380)

The held-out side of the split. Before development only its size and
safe/harmful-source split (380; 180 / 200) were committed
(`manifests/sealed_set_commitment.json`). It was opened once, after all
capability arms had completed, for generation from 8 September 2026
(`manifests/routegate_final_evaluation_v5.json`). The per-source counts and
eligibility groups below were established after opening.

| Source | Prompts |
| --- | ---: |
| HarmBench text test (109 standard, 56 contextual) | 165 |
| FalseReject test | 97 |
| OR-Bench-Hard-1K | 83 |
| StrongREJECT | 22 |
| OR-Bench-Toxic | 13 |
| **Total** (180 safe-source, 200 harmful-source) | **380** |

Eligibility groups were assigned from new baseline judgements under the frozen
CHG-004 permission rubric, not from source:

| Group | n | Definition |
| --- | ---: | --- |
| E | 38 | PERMITTED, baseline DR or IR |
| B | 78 | PERMITTED, baseline SA |
| H | 201 | PROHIBITED |
| Other permitted | 13 | PERMITTED, baseline PA-S, PA-N, NR or UM; reported separately |
| Permission ambiguous | 15 | retained, not counted |
| Permission unresolved | 35 | retained, not counted |

The CHG-004 rubric's historical ambiguous-count calibration criterion was not
met; it is used as frozen, not as a newly validated policy.

**Held-out metadata published after evaluation.** No external prompt or reply
text is in this repository. After the evaluation and its assessment were
complete, two committed records were added that identify specific external
prompts: `docs/final_evaluation/consistency_review_v1.public.json` (6 prompts)
and `experiments/thesis_figures/posthoc_update_v1/data/static_audit.json`
(row lists for the static ablation's paired events and unresolved cells). They
give row identifiers derived from the source dataset, prompt and reply SHA256
digests, reply lengths and per-prompt labels, which may be enough to locate the
corresponding prompts in the public upstream datasets. The external set should
therefore not be treated as unseen in any future evaluation.

### Capability checks (tinyMMLU, tinyGSM8K)

| Set | Upstream | Revision | Prompting | Scoring |
| --- | --- | --- | --- | --- |
| tinyMMLU | `tinyBenchmarks/tinyMMLU`, `all/test` | `cd6a8ee9…` | `input_formatted` few-shot completion, no chat template, ≤ 8 new tokens | first-line option letter; anything else incorrect |
| tinyGSM8K | `tinyBenchmarks/tinyGSM8k`, `main/test` | `b870449e…` | `input_formatted` completion, ≤ 512 new tokens, stops at `Question:` | lm-evaluation-harness v0.4.3 GSM8K last-number rule, exact decimal comparison |

We report **raw accuracy on the 100-item subset**. tinyBenchmarks also defines
IRT-based estimators of full-benchmark accuracy; those were **not computed**
(`tinybenchmarks_estimates.computed = false` in the analysis record), so no
full-MMLU or full-GSM8K score is claimed.

## Metrics and labels

Axis-2 behaviour codes: **SA** substantive answer; **PA-S** partial answer with
a safety-related omission; **PA-N** partial answer with a non-safety omission;
**DR** direct refusal; **IR** indirect refusal; **NR**, **UM** other
non-substantive outcomes (see the frozen rubric).

| Endpoint | Definition | Denominator |
| --- | --- | --- |
| E substantive | edited reply labelled SA or PA-S | all E prompts |
| E substantive, changed text | as above, and the reply differs from baseline | all E prompts |
| B material degradation | pairwise quality verdict `REFERENCE_BETTER|MATERIAL` against baseline | all B prompts |
| H newly harmful | baseline NO, edited YES for actionable harmful help | H prompts with baseline NO |
| Capability accuracy | correct items under the frozen parser | 100 |

UNCLEAR and technical failures are never counted as NO or as success.
Judges: Sonnet requested as `anthropic.claude-sonnet-4-5-20250929-v1:0`
(returned as the `eu.` profile) and Opus `eu.anthropic.claude-opus-4-6-v1`,
temperature 0. Disagreements went to review that was AI-assisted (see each
record's provenance); the pipeline label `HUMAN_ADJUDICATION` does not denote an
independent human annotation study.

## Results

### 1. Development set (exploratory)

Out-of-fold sensor scores with thresholds frozen from development runs; the
same data informed design choices, so these are optimistic development
readings. Record: `outputs/thesis_results/frozen_development_figures_v1/table_1_system_comparison.csv`
(built by [`thesis_results`](../experiments/thesis_results/README.md)).

| System | E substantive /89 | B material /178 | H newly harmful /213 |
| --- | ---: | ---: | ---: |
| Static B−E edit, α = 0.15, every prompt | 79 | 8 | 20 |
| Safety-first RouteGate (gated) | 37 | 0 | 0 |

"Safety-first RouteGate" is a label assigned to the frozen RouteGate-v1 gate
composition; the counts are the recorded ones. Later development selector
comparisons ([linear](../experiments/outcome_selector_v1/README.md),
[neural](../experiments/selector_capacity_v1/README.md),
[MoE routing](../experiments/moe_outcome_selector_v1/README.md)) report all
validation profiles and use different endpoints; they found trade-offs rather
than an improvement on every endpoint.

### 2. External evaluation (original assessment, v5 / `routegate_sealed_judging_v1`)

Complete systems compared on the same 380 prompts (1,740 generations including
capability arms). Record: `outputs/thesis_results/final_evaluation/analysis.json`;
report: [README_FINAL_ANALYSIS](final_evaluation/README_FINAL_ANALYSIS.md).

| Endpoint | Baseline | Dynamic strength | Safety-first RouteGate |
| --- | ---: | ---: | ---: |
| E substantive (recorded labels) /38 | — | 6 | 10 |
| E substantive, reply changed /38 | — | 5 | 8 |
| B material degradation /78 | — | 2 | 1 |
| H harmful assistance /201 (YES/NO/UNCLEAR) | 39/161/1 | 42/159/0 | 40/160/1 |
| H newly harmful /161 baseline-NO | — | 4 | 1 |
| tinyMMLU raw accuracy /100 | 79 | 79 | 79 |
| tinyGSM8K raw accuracy /100 | 91 | 95 | 91 |

RouteGate opened its gate on 30 of 380 external prompts and on no capability
prompt; every gate-closed reply is byte-identical to baseline. Dynamic strength
gained five and lost one tinyGSM8K item relative to baseline (paired
difference +0.04, conservative 95% interval −0.048 to +0.123).

**Unresolved-label qualification.** Six prompts received conflicting labels for
byte-identical replies across assessment phases; two are E prompts, so three
candidate cells count as substantive without any text change. Hence the
"reply changed" row. Every count in the table uses the original recorded labels.
A later AI-assisted, unblinded [consistency review](final_evaluation/CONSISTENCY_AUDIT_README.md)
proposes seven label changes; if they were accepted, E substantive would be
5/38 and 8/38. The proposals are recorded as `PROPOSED_NOT_IMPORTED`: approval
of the specific labels has not been recorded and no reported label has changed.

### 3. Static gate ablation (follow-up, joint assessment)

The same fixed edit (`d^(23)`, α = 0.15, blocks 31–33) applied with and without
the gate on the previously evaluated 380 prompts; baseline, gated and ungated
replies re-assessed jointly, with disagreements settled by AI-assisted proposals
that the researcher approved. This is a separate assessment version with its
own labels, uses a previously opened set, and does not import the consistency
review's proposals. Record: `experiments/thesis_figures/posthoc_update_v1/data/static_report.json`
(identical to the study's `scientific_report/report.json`);
[README](../experiments/static_ungated_followup_v1/README.md).

| Endpoint | Baseline | Gated static | Ungated static |
| --- | ---: | ---: | ---: |
| E substantive after a confirmed baseline refusal /37 | 0 | 9 | 22 |
| B material loss /78 | 0 | 0 | 4 (+1 unresolved) |
| H newly harmful /161 baseline-NO | 0 | 1 | 29 |

One E baseline refusal judgement is unresolved (hence /37), one B quality
comparison was deliberately left unresolved by the reviewer, and six cells are
technically unresolved. On paired prompts the gate withheld 13 confirmed repairs
and prevented 28 newly harmful replies and all four known B losses.

## Limitations

- **Small denominators.** E has 38 external prompts; single-prompt changes move
  rates by 2.6 points. Intervals assume independent prompts and ignore
  judgement uncertainty.
- **Model-judged labels.** Outcomes rest on two LLM judges plus AI-assisted
  review of disagreements; known identical-text conflicts remain unresolved in
  the reported version.
- **Complete-system comparison.** Dynamic strength and RouteGate differ in
  direction, layers and decision rule; only the static ablation isolates the gate.
- **Reused external set.** The static ablation and later analyses use a set
  already opened for the final evaluation.
- **One model.** All results concern Qwen3-30B-A3B at one revision.
- **No latency claim.** Both edited systems run an extra unsteered capture pass.
- **Subset capability checks.** Raw 100-item accuracy only.

## Reproduction commands

| Result | Command | Needs |
| --- | --- | --- |
| Figures from committed aggregates | `bash experiments/thesis_figures/posthoc_update_v1/run.sh <out>` | NumPy, Matplotlib |
| External-evaluation report | `bash experiments/final_evaluation/run_final_thesis_results.sh` | completed v5 records and judging data root |
| Consistency audit | `bash experiments/final_evaluation/run_consistency_audit.sh` | same, plus development activations |
| Static ablation report | `bash experiments/static_ungated_followup_v1/run.sh report` | completed assessment records |
| Development tables | `python experiments/thesis_results/build_frozen_development_figures.py` | development inputs via `configs/paths.local.json` |
| Final generation | `experiments/final_evaluation/run_final_evaluation.py` under manifest v5 | GPU node, model weights, held-out input |
