# Cancellation controls: does H−E add anything beyond weakening repair?

A completed diagnostic on selected development H prompts. It does not train Qwen or change earlier results. The 380 external prompts are not used in this diagnostic.

**Nine conditions on the same 18 previously selected development H prompts: 162 fresh generations.** Baseline, repair-only and the original combination are generated afresh too. The comparison tests whether the part of H−E perpendicular to the repair direction contributes anything beyond cancellation of repair.

This selected H diagnostic cannot establish successful over-refusal repair or population safety. It was designed to precede any broader E/B/H development comparison; any method derived from it would need fixing on development data before use on the external set, with that set's earlier use disclosed.

## 1. Initialise

From the repository root (run in this order):

```bash
bash experiments/cancellation_controls_v1/run.sh init
bash experiments/cancellation_controls_v1/run.sh preflight
```

Expected after initialisation: **18 rows, 9 arms, 162 planned generations, 0 API calls.** The study folder holds the fixed row selection, vectors, comparisons, generation settings and copies of the original safety rubric. Initialisation does not load Qwen.

The wrappers read `ROUTEGATE_DATA_DIR` (default `$HOME/routegate-data`) and `ROUTEGATE_CONTAINER_FILE` (default `$HOME/containers/vllm-openai-v0.10.1.1.sif`). The study folder is `$ROUTEGATE_DATA_DIR/cancellation_controls_v1/`.

## 2. Generate on an allocated GPU node

The study used this Myriad allocation:

```bash
qrsh -l mem=96G,h_rt=15:0:0,gpu=4,tmpfs=30G -ac allow=L -now no
```

On the allocated node, from the repository root:

```bash
bash experiments/cancellation_controls_v1/run.sh preflight --check-model
bash experiments/cancellation_controls_v1/run.sh generate
```

The wrapper first runs the CPU hook and integrity tests inside the project container. A failed or skipped required test stops the launch. Generation then verifies the local model revision and uses the existing greedy, thinking-disabled generation procedure. No judge API key is needed.

Progress prints `[generation] 1/162 complete` and continues to 162. The output folder is:

```text
$ROUTEGATE_DATA_DIR/cancellation_controls_v1/
```

To inspect progress later:

```bash
bash experiments/cancellation_controls_v1/run.sh progress
```

To resume after a time limit or interruption, obtain a GPU allocation and run the same `generate` command. Successful generations are preserved. A recorded technical failure requires inspecting `generations/attempts.jsonl` before deliberately using `generate --retry-failures`. Do not change the prompt selection, coefficients or code within a started study.

For an optional technical check, `generate --limit 9` generates the nine conditions for the first fixed prompt. These count towards the 162; a later `generate` continues the same study. This is not a new selection stage.

## 3. Prepare judging after all 162 generations

These commands can run on the login node and make no API calls:

```bash
bash experiments/cancellation_controls_v1/judge.sh --stage prepare
bash experiments/cancellation_controls_v1/judge.sh --stage build
bash experiments/cancellation_controls_v1/judge.sh --stage dryrun
```

The assessor uses the existing safety rubric, Sonnet/Opus settings, reply validation and raw-response preservation. Within this new study, identical prompt-and-response material receives one assessment from each judge and the verdict is mapped back to every corresponding condition. Therefore the planned number of calls is **at most 324**, and may be lower. This rule is fixed before generation. No old judgements are imported or changed.

With judge credentials supplied through the environment (never committed), run the two judges sequentially:

```bash
bash experiments/cancellation_controls_v1/judge.sh --stage judge --judge sonnet
bash experiments/cancellation_controls_v1/judge.sh --stage judge --judge opus
bash experiments/cancellation_controls_v1/judge.sh --stage progress
```

Do not start both writers together. If a judge reports failures, inspect the saved failures before a deliberate retry of that judge with `--retry-failures`. A missing assessment is not a `NO` safety verdict.

After both have answered all planned materials:

```bash
bash experiments/cancellation_controls_v1/judge.sh --stage join
bash experiments/cancellation_controls_v1/judge.sh --stage export
```

This exports blinded disagreements for review. Reviewer metadata must accurately state who reviewed them and whether AI assisted. The protocol retains `YES`, `NO`, `UNCLEAR` and technical failures separately. Human decisions are imported separately with their provenance.

## 4. Results

The completed study is under `$ROUTEGATE_DATA_DIR/cancellation_controls_v1/`. Its aggregate plotting input is committed as `experiments/thesis_figures/posthoc_update_v1/data/cancellation_plotdata.csv` and drawn by the [figure package](../thesis_figures/posthoc_update_v1/README.md) (`cancellation_controls`). All 19 cancellation review decisions have researcher approval; one UNCLEAR outcome remains explicit.

To archive the study:

```bash
tar -czf routegate_cancellation_controls_results.tar.gz -C "$ROUTEGATE_DATA_DIR" cancellation_controls_v1
```

## What is reported

- Harmful-assistance outcomes for each of the nine conditions on the full fixed 18-prompt frame.
- Paired outcomes for the original combination against attenuated repair and each of the three matched random controls.
- Fresh baseline and repair-only outcomes, including whether the previously selected regressions recur. No cases are dropped if they do not.
- Intended, dtype-cast and actually written edit norms and repair projections at every layer.
- Capped responses, uncertain judgements and technical failures separately.

Three random directions are comparison controls, not enough to estimate a broad random-direction distribution. Uncertainty intervals are descriptive under stated prompt-sampling assumptions; they do not remove the historical selection bias or establish a safety guarantee.

## Method

At each of layers 31–33, let r be the stored B_d1−E repair vector and s the stored H−E vector. The original vectors are retained without renormalisation. Define:

\[
q=\frac{s^\top r}{r^\top r},\qquad s_\perp=s-qr,\qquad a=0.15+0.10q.
\]

For a given prompt and layer, b is the norm of the freshly captured **unsteered layer input**. This same value is used for every condition. The edit is applied to the layer output at the final prompt token, only during the first forward pass used for generation. The internal package calls output of layer 31 'state 32'; the preceding 'state 31' supplies its input norm. No earlier steering changes b.

| Condition | Intended additive change |
|---|---|
| Baseline | 0 |
| Repair only | b(0.15r) |
| Original combination | b(0.15r+0.10s) |
| Attenuated repair | b(ar) |
| Orthogonal component with full repair | b(0.15r+0.10s_perp) |
| Orthogonal component only | b(0.10s_perp) |
| Random orthogonal 1–3 | b(ar+0.10\|s_perp\|u_j) |

Each u_j is a fixed unit random direction orthogonal to both r and s_perp. The three draws use seed 20260910 and are frozen before generation. These controls preserve the intended repair projection and total edit norm of the original combination, up to documented floating-point error, while changing its perpendicular direction.

The identity

\[
0.15r+0.10s=ar+0.10s_\perp
\]

separates cancellation along r from the remaining component. The previous direction bank has negative q at all three layers. Therefore adding H−E also reduces the effective repair coefficient. Keeping the nominal repair coefficient at 0.15 did not hold that projection constant.

### Primary and secondary comparisons

The four predeclared primary comparisons are original combination versus attenuated repair, and original combination versus each random control. Report the paired transitions and all uncertainty labels, rather than selecting only the most favourable comparison.

Orthogonal-full-repair versus repair-only and orthogonal-only versus baseline provide additional interpretation. They are secondary comparisons. The first is useful for asking whether the perpendicular component helps while restoring the full repair projection. The second checks whether that component alone changes harmful assistance.

Evidence consistent with a useful perpendicular component would be lower harmful assistance than attenuation and the matched random controls. Similar performance to attenuation would support a simpler cancellation explanation on these cases. Neither outcome is guaranteed. The full-repair orthogonal condition can still increase harmful assistance; its name does not imply a safety effect.

### Bounds on the claim

This is a prospective control comparison on a historically selected set. The 18 prompts were selected because they previously became harmful under repair. The coefficient 0.10 was chosen using previous development results. These are disclosed design choices, not an untouched test set.

Fresh baseline, repair-only and combination responses are generated alongside all new controls. Non-recurrence of a historical regression is reported rather than used to replace a case. General safety or over-refusal repair must subsequently be tested on broader E/B/H development material.

The theoretical vector identity does not guarantee identical realised changes after bfloat16 rounding, nor identical downstream states after previous layer edits. The runner records intended, cast and actually written changes. It uses the existing cast-delta-before-add arithmetic; no alternative arithmetic is introduced silently.

No policy probe is assumed to be a causal safety mechanism. No model weights are trained. No method is selected automatically from these results, and no existing thesis numbers or old assessments are changed.

### Model and measurement settings

- Qwen/Qwen3-30B-A3B at revision ad44e777bcd18fa416d9da3bd8f70d33ebb85d39; original tokeniser/template hashes checked.
- Eager attention, bfloat16, batch 1, no system prompt, thinking disabled, greedy decoding, maximum 768 new tokens.
- One new unsteered capture per prompt, saved for use across all nine conditions and resumes.
- The two original judges use the same frozen safety rubric and settings as the earlier development safety experiment. Exact duplicate material is assessed once per judge within this new study.
- Paired comparisons retain the full 18-prompt frame; uncertain/missing outcomes receive explicit bounds. Pointwise conservative intervals assume independent prompt sampling and do not compensate for the selected sample or annotation error.
