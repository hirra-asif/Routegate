# Parallel Opus judging (operational record)

Historical record of how the final-evaluation response assessment was scheduled.
It changed no model identity, sampling setting, rubric, prompt, generated answer,
blinded identifier or outcome definition.

Sonnet (requested as `anthropic.claude-sonnet-4-5-20250929-v1:0`, returned as
`eu.anthropic.claude-sonnet-4-5-20250929-v1:0`) ran in the main judging
workspace. Opus (`eu.anthropic.claude-opus-4-6-v1`) ran concurrently through the
same frozen runner in a separate workspace, `sealed_opus_parallel_v1/` under the
data root, using an exact copy of the prepared response packet and its fixed A/B
ordering. Each workspace kept its own single-writer lock.

`experiments/final_evaluation/run_opus_parallel.sh` performs three steps:

| Command | Effect |
| --- | --- |
| (no argument) | Validate and copy inputs, then run Opus in the separate workspace. Refuses to start if the main workspace already holds response-phase Opus evidence. |
| `progress` | Report Opus progress in the separate workspace. |
| `merge` | Acquire both runner locks, check that frozen inputs, packet, baseline adjudication, request hashes, verdicts and raw replies match, and import every attempted Opus case into the main workspace. No API calls. |

The merge preserves successful and failed attempts, refuses to overwrite
differing evidence and is idempotent. Its receipt is written to
`final_evaluation_v5_judging_v1/responses/parallel_opus_import_v1.json`. Failed
requests remain technical uncertainties and were not retried or relabelled by
this step.

The wrapper and `run_sealed_judging.sh` are hash-pinned by
`docs/parallel_opus_v1_validation.json` and
`manifests/routegate_sealed_judging_v1_release.json`, so they keep the absolute
Myriad paths with which they were run. Adapt a copy for another machine.

## Validation

Eight offline tests (`tests/test_parallel_opus_workspace.py`) cover packet
identity, concurrent separate workspaces, conflicting locks, original frozen
source hashes, resumption without repeating successful cases, preservation of
failed calls, refusal to import changed or incomplete evidence, and refusal to
overwrite conflicting evidence. A dry run on all 1,740 generation records and
the 187 approved baseline review decisions reproduced the 861-case Opus packet:
102 response-behaviour, 156 pairwise-quality and 603 safety assessments. No live
API calls were made during development or testing.

`docs/parallel_opus_v1_validation.json` pins the original text of this note,
which contained transfer instructions; that text is preserved in the privately
retained development history. The public repository begins with a clean initial
commit and does not include it.
