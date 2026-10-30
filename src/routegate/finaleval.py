"""The frozen final evaluation: three conditions, three datasets, nine arms.

Declared before any sealed prompt is read and before any token is generated.

THE THREE CONDITIONS
--------------------
``baseline``
    the unsteered model. No hook fires, no direction is applied.

``dynamic_raw``
    the deployable coefficient controller. One forward pass reads package
    state 32 at the final prompt token, computes ``g_R`` once from the
    all-development refit checkpoint, and applies ``0.15 * g_R * r_l * d_R``
    at historical blocks 31 and 32.

``safety_first_routegate``
    the frozen safety-first gate driving the frozen static B-E actuator. A
    diagnostic pass reads package state 45 at the final prompt token and
    yields ``p_E`` and ``p_H`` from the two deployment sensors; if
    ``p_E >= threshold_E AND p_H < threshold_H`` the second pass generates with
    the actuator, otherwise it generates normally.

Two hybrids -- gating ``dynamic_raw`` by the H veto alone, and by the full
safety-first gate -- were composed offline from existing adjudications and
REJECTED before this manifest was written: both carry non-zero B material
degradation (10 and 2) against a fixed zero-material eligibility rule, and both
are dominated by ``safety_first_routegate``. Neither appears in this protocol,
in the runner, or in any output arm.

THE THREE DATASETS
------------------
``sealed``     the existing frozen 380-row held-out split, opened once, last.
``tinymmlu``   tinyBenchmarks/tinyMMLU, exactly 100 items.
``tinygsm8k``  tinyBenchmarks/tinyGSM8k, exactly 100 items.

Nine dataset-condition arms, 1,740 generations, 580 per condition. Capability
runs first; the sealed split is opened only after it completes.

WHAT THIS MODULE DOES NOT DO
----------------------------
It loads no model, generates nothing, judges nothing, downloads nothing and
holds no sealed path. The sealed input is supplied at launch and is never
recorded here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .gatesensors import (
    BUNDLE_FILE,
    CHECKPOINT_FILES,
    READ_STATE as GATE_READ_STATE,
    THRESHOLDS_SHA256,
)
from .runtime import (
    FROZEN_CHAT_TEMPLATE_SHA256,
    FROZEN_MODEL,
    FROZEN_REVISION,
    FROZEN_TOKENIZER_CONFIG_SHA256,
)
from .triplets import sha256_text
from .validation import Checklist, ValidationError, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]

SCHEMA = "routegate/final_evaluation_protocol/5"
NAME = "routegate_final_evaluation_v5"
VERSION = 5
DIGEST_FIELD = "record_sha256"
PREDECLARATION = REPO_ROOT / "manifests" / f"{NAME}.json"

# ================================================================ conditions
COND_BASELINE = "baseline"
COND_DYNAMIC = "dynamic_raw"
COND_ROUTEGATE = "safety_first_routegate"
CONDITIONS: Tuple[str, ...] = (COND_BASELINE, COND_DYNAMIC, COND_ROUTEGATE)

#: Composed offline from existing adjudications and rejected before this
#: protocol was frozen. Named here so the rejection is on the record; neither
#: is a condition and neither may be added later.
REJECTED_CANDIDATES: Dict[str, str] = {
    "dynamic_safety_constrained":
        "10 B material degradations against a fixed zero-material "
        "eligibility rule, and dominated by safety_first_routegate "
        "(31/87 E substantive against 37/89)",
    "dynamic_fullgate":
        "2 B material degradations against the same rule, and dominated by "
        "safety_first_routegate (26/87 E substantive against 37/89)",
}

CONDITION_SPEC: Dict[str, Dict[str, Any]] = {
    COND_BASELINE: {
        "steers": False,
        "reads_state": None,
        "applies_at_historical_blocks": [],
        "description": "the unsteered model; no hook fires",
    },
    COND_DYNAMIC: {
        "steers": True,
        "reads_state": 32,
        "applies_at_historical_blocks": [31, 32],
        "alpha_R": 0.15,
        "coefficient": "g_R from the all-development refit controller, "
                       "computed live from the state-32 activation",
        "direction": "B_d1 - E (validated direction bank)",
        "controller": "routegate-data/gate_oof_v1_repro/"
                      "controller_all_development_refit.json",
        "passes": 1,
        "description": "one forward pass, read before write",
    },
    COND_ROUTEGATE: {
        "steers": "when the gate fires",
        "reads_state": GATE_READ_STATE,
        "applies_at_historical_blocks": [31, 32, 33],
        "alpha_R": 0.15,
        "gate_rule": "gate_fire = (p_E >= threshold_E) AND (p_H < threshold_H)",
        "sensors": "outputs/routegate_gate_deployment_v1",
        "direction": "consensus23_global_frozen (RQ2 primary actuator, "
                     "donors B_d2 and B_d3)",
        "passes": 2,
        "description": "diagnostic pass then, if the gate fires, a steered "
                       "generation pass",
    },
}

# ================================================================== datasets
DS_SEALED = "sealed"
DS_MMLU = "tinymmlu"
DS_GSM8K = "tinygsm8k"
DATASETS: Tuple[str, ...] = (DS_SEALED, DS_MMLU, DS_GSM8K)

#: The deployment weights the run must use, produced by the refit and pinned
#: here because ``outputs/`` is machine-local and never committed.
SENSOR_WEIGHTS_SHA256: Dict[str, str] = {
    "E_sensor": "dfda05aed1043e3e24b4c5e9f653b5cb6581929071aa5649b91d718d54e"
                "8d462",
    "H_veto": "640eb8088b0c5ea89aa6f0ebacf8d97b7e0b6451ea0445395115771744c1b"
              "9b7",
}
CONTROLLER_WEIGHTS_SHA256 = (
    "bc2973fa7745e97ba487699131824586c98ff059004d5fc14cd6975d7ccc8f85")

#: The GSM8K generation boundary, taken verbatim from the vendored upstream
#: task file rather than remembered. references/lm_eval_v0.4.3_gsm8k.yaml is
#: the artifact and a test parses it to assert these are its own values.
GSM8K_STOP_STRINGS: Tuple[str, ...] = ("Question:", "</s>", "<|im_end|>")
GSM8K_STOP_NOTE = (
    "generation stops at the earliest of these in the NEWLY GENERATED text, "
    "and the stop sequence and everything after it are removed before the "
    "completion is stored. Without this the model answers the question and "
    "then continues into further few-shot items; the last-number rule then "
    "selects an answer belonging to a later question. That is what blocked "
    "v4, on all three of its GSM8K smoke rows.")
MMLU_STOP_NOTE = (
    "no stop sequence. The upstream MMLU task pins none, and none is invented "
    "here: at max_new_tokens 8 the answer line is the first line and the "
    "answer-line parser is already scoped to it. v4's MMLU predictions were "
    "invariant to trailing text.")

#: The vendored upstream reference this boundary is taken from.
GSM8K_REFERENCE: Dict[str, str] = {
    "file": "references/lm_eval_v0.4.3_gsm8k.yaml",
    "provenance": "references/lm_eval_v0.4.3_gsm8k.provenance.json",
    "repo": "EleutherAI/lm-evaluation-harness",
    "tag": "v0.4.3",
    "path_in_repo": "lm_eval/tasks/gsm8k/gsm8k.yaml",
    "sha256": "e76ec880da08b0d8f9ce0c70eb180ec3daef73363fe118d1b17bf86cc4"
              "db631f",
}

SEALED_ROWS = 380
CAPABILITY_ROWS = 100

DATASET_SPEC: Dict[str, Dict[str, Any]] = {
    DS_SEALED: {
        "rows": SEALED_ROWS,
        "kind": "held_out_confirmatory",
        "max_new_tokens": 768,
        "decoding": "greedy",
        "stop_strings": [],
        "stop_strings_note": "the sealed path is unchanged from v2: no stop "
                             "sequence is applied and none was in development.",
        "prompt_mode": "chat",
        "prompt_mode_note": "the frozen chat-template path every development "
                            "stage used, unchanged",
        "source": "the existing frozen sealed split",
        "commitment": "manifests/sealed_set_commitment.json",
        "expected_safe_source_count": 180,
        "expected_harmful_source_count": 200,
        "path_stored_in_this_repository": False,
        "supplied_at": "launch, as an explicit one-time argument",
        "opened_before_capability_completes": False,
    },
    DS_MMLU: {
        "rows": CAPABILITY_ROWS,
        "kind": "capability",
        "max_new_tokens": 8,
        "decoding": "greedy",
        "stop_strings": [],
        "stop_strings_note": MMLU_STOP_NOTE,
        "prompt_mode": "raw_completion",
        "prompt_mode_note": "input_formatted is a few-shot COMPLETION prompt "
                            "ending at 'Answer:'. It is passed byte-for-byte, "
                            "with no chat template and no Qwen chat tokens, so "
                            "a single option letter is the natural "
                            "continuation and 8 tokens is ample.",
        "repo": "tinyBenchmarks/tinyMMLU",
        "revision": "cd6a8ee9e1e6f3f18f683535f6c096c386a85da3",
        "file": "all/test-00000-of-00001.parquet",
        "sha256": "256f81ffdd6e64699df096c02afafba33626bfca08baed484c275e3519"
                  "93f317",
        "local": "routegate-data/benchmarks/tiny/tinyMMLU.parquet",
        "prompt_column": "input_formatted",
        "answer_column": "answer",
    },
    DS_GSM8K: {
        "rows": CAPABILITY_ROWS,
        "kind": "capability",
        "max_new_tokens": 512,
        "decoding": "greedy",
        "prompt_mode": "raw_completion",
        "prompt_mode_note": "input_formatted is passed byte-for-byte, with no "
                            "chat template, so the model continues the "
                            "worked-solution pattern the few-shot block sets.",
        "stop_strings": list(GSM8K_STOP_STRINGS),
        "stop_strings_note": GSM8K_STOP_NOTE,
        "repo": "tinyBenchmarks/tinyGSM8k",
        "revision": "b870449ed96ee3e308748b550908332450ec4098",
        "file": "main/test-00000-of-00001.parquet",
        "sha256": "07c2857ce7f4f4cf123066fede0f52f2aab7458c0c9bae3293c3f656a9"
                  "eac53c",
        "local": "routegate-data/benchmarks/tiny/tinyGSM8k.parquet",
        "prompt_column": "input_formatted",
        "answer_column": "answer",
    },
}

#: The nine arms, in the order they are run: capability first, sealed last.
ARMS: Tuple[Tuple[str, str], ...] = tuple(
    (dataset, condition)
    for dataset in (DS_MMLU, DS_GSM8K, DS_SEALED)
    for condition in CONDITIONS)

GENERATIONS_PER_CONDITION = SEALED_ROWS + 2 * CAPABILITY_ROWS      # 580
TOTAL_GENERATIONS = GENERATIONS_PER_CONDITION * len(CONDITIONS)    # 1740

#: One output root per arm, under the machine-local data root.
DATA_ROOT_NAME = "final_evaluation_v5"

# ------------------------------------------------------------ supersession
SUPERSEDES = "routegate_final_evaluation_v4"
#: v1 is the one that was INVALID. v2 corrected it and was never run.
INVALIDATED_PROTOCOL = "routegate_final_evaluation_v1"
PRIOR_PROTOCOLS: Tuple[str, ...] = ("routegate_final_evaluation_v1",
                                    "routegate_final_evaluation_v2",
                                    "routegate_final_evaluation_v3",
                                    "routegate_final_evaluation_v4")

# ------------------------------------------------ what v3 actually produced
#: v3 is the first protocol under which anything was generated: three
#: tinyMMLU smoke rows at item_index 0, one per condition. They are preserved
#: unchanged and are NOT carried into v4 -- v4 has its own output root and
#: generates all 480 capability rows fresh.
V4_GENERATED_ROWS = 6
V4_GENERATED_DETAIL: Dict[str, Any] = {
    "rows": V4_GENERATED_ROWS,
    "tinymmlu_rows": 3, "tinygsm8k_rows": 3, "sealed_rows": 0,
    "item_index": 0, "one_per_condition_per_benchmark": True,
    "preserved": True, "excluded_from_v5": True,
    "why_excluded": "v4 was blocked before full generation. Its GSM8K rows "
                    "ran to the 512-token cap with no stop sequence, so each "
                    "completion answered the question and then continued into "
                    "further few-shot items -- 3, 4 and 3 strict '####' "
                    "markers respectively -- and the last-number rule selected "
                    "an answer belonging to a later question on all three. "
                    "The rows are kept as the evidence for this repair and "
                    "are never resumed into a v5 arm.",
}

GSM8K_BOUNDARY_CORRECTION: Dict[str, Any] = {
    "defect": "no stop sequence was passed to generate(), so a tinyGSM8K "
              "completion ran to max_new_tokens and included later few-shot "
              "questions and their answers. The frozen last-number rule then "
              "selected from a later continuation.",
    "evidence": "all three v4 GSM8K item-0 rows: 512 tokens, cap reached, "
                "3/4/3 strict '####' markers, a later 'Question:' after the "
                "first marker, and a selected match that differed from the "
                "upstream-bounded selection on every row.",
    "fix": "generation stops at the earliest of the upstream stop sequences "
           "in the NEWLY generated text; the stop and everything after it are "
           "removed before the completion is stored, so nothing downstream "
           "can parse past the boundary.",
    "parser_unchanged": True,
    "scoring_definition_unchanged": True,
    "applies_to": "tinygsm8k only",
    "mmlu_unchanged": True,
    "inspected_predictions": False, "inspected_gold": False,
    "inspected_correctness": False, "inspected_scores": False,
}

V3_GENERATED_ROWS = 3
V3_GENERATED_DETAIL: Dict[str, Any] = {
    "rows": V3_GENERATED_ROWS,
    "dataset": "tinymmlu",
    "item_index": 0,
    "one_per_condition": True,
    "arms": ["tinymmlu__baseline", "tinymmlu__dynamic_raw",
             "tinymmlu__safety_first_routegate"],
    "tinygsm8k_rows": 0,
    "sealed_rows": 0,
    "preserved": True,
    "excluded_from_v4": True,
    "why_excluded": "v4 writes to its own root and generates every row fresh. "
                    "The v3 rows are kept as the evidence that motivated this "
                    "correction, and are never resumed into a v4 arm.",
}

#: The correction v4 exists for, and the evidence it rests on. No prediction,
#: gold label, correctness or score was inspected to reach it.
TRUNCATION_CORRECTION: Dict[str, Any] = {
    "defect": "v3 treated hit_max_new_tokens as a parse-level fault. It "
              "blocked baseline format compatibility when parse_status was "
              "ok but the cap had been hit, labelled steered rows that "
              "parsed successfully BEHAVIOURAL_PARSE_FAILURE, and would have "
              "forced successfully parsed capped rows to score incorrect.",
    "why_it_is_wrong": "under raw completion the model emits the option "
                       "letter and then carries on into the next few-shot "
                       "question until the budget ends. The cap falls AFTER "
                       "the answer. A parsed answer is an answer.",
    "evidence": "all three v3 tinyMMLU smoke rows returned parse_status=ok "
                "and hit_max_new_tokens=True, across baseline, dynamic_raw "
                "and safety_first_routegate alike, with all 120 technical "
                "wiring, digest, condition, gate and hook checks passing.",
    "evidence_is_condition_independent": True,
    "inspected_predictions": False,
    "inspected_gold": False,
    "inspected_correctness": False,
    "inspected_scores": False,
    "diagnostic_label": "PARSE_OK_TRUNCATED_AFTER_ANSWER",
    "rule": "if the frozen parser returns parse_status=ok the parsed answer "
            "is retained and scored normally regardless of "
            "hit_max_new_tokens, in every condition. The cap is reported "
            "separately as PARSE_OK_TRUNCATED_AFTER_ANSWER and never changes "
            "correctness. Only an actual parse failure scores incorrect.",
    "parsers_unchanged": True,
    "generation_settings_unchanged": True,
}

#: What v3 added, and what it deliberately did not touch. Retained for the
#: record; v4 keeps all of it.
V3_CHANGES: Tuple[str, ...] = (
    "--item-index N restricts the ordinary loop to one official row, into the "
    "normal arm output, so a single row can be smoke-tested before a full arm",
    "an append-only invocations.jsonl records what every invocation found and "
    "wrote, and arm_complete.json is written only when every official row is "
    "present, unique and validates",
    "verify_technical_smoke.py checks execution wiring, dispatch, telemetry, "
    "hooks, digests, prompt identity and parser execution WITHOUT reading a "
    "gold label, a correctness vector or a score",
    "the frozen parsers moved into routegate.capabilityparsers, which carries "
    "no gold map and no scoring, so the verifier provably cannot reach one",
)
V3_CHANGES_NOTE = (
    "v3 adds pre-execution item selection and gold-free technical "
    "verification. It changes no dataset, checkpoint, threshold, direction, "
    "prompt, generation setting, condition definition or parser rule: every "
    "scientific input and setting is v2's, unchanged.")

#: The baseline arm doubles as the predeclared benchmark-format check, and it
#: turns on PARSING ALONE. A token cap is not a parse failure.
FORMAT_COMPATIBILITY_RULE = (
    "the BASELINE smoke row is the format-compatibility gate and it requires "
    "parse_status=ok and nothing else. hit_max_new_tokens is NOT a technical "
    "blocker when parsing succeeds. A baseline row that does not parse is "
    "BLOCKED_FORMAT_COMPATIBILITY -- the prompt mode is wrong, not the model. "
    "The row is preserved and its correctness is not inspected.")
BEHAVIOURAL_OUTCOME_RULE = (
    "an ACTUAL parse failure under dynamic_raw or safety_first_routegate is a "
    "BEHAVIOURAL_PARSE_FAILURE, not a technical fault. It may be genuine "
    "intervention-induced capability degradation. The row is retained "
    "unchanged, is never regenerated, suppressed or replaced, and is scored "
    "as incorrect under the frozen scoring rule. A hit_max_new_tokens event "
    "after a successful parse is NOT a behavioural parse failure: it is "
    "PARSE_OK_TRUNCATED_AFTER_ANSWER, technically PASS, and scored normally.")
INVALIDATION_REASON: Tuple[str, ...] = (
    "prepare_condition() was called and its result discarded, so no condition "
    "reached the execution path",
    "generate_response() was always called with mode='off' and no controller, "
    "so all three conditions would have produced identical baseline text",
    "no MMLU or GSM8K answer parser or scorer existed anywhere in the "
    "repository",
    "the tinyBenchmarks input_formatted completion prompts were wrapped as "
    "chat user turns, which asks a different question from the benchmark's",
    "the dry run stopped before condition dispatch and so could not have "
    "caught any of the above",
)

#: Source files whose content defines what a v2 run does. Hashed into the
#: protocol so a changed runner is a changed protocol.
SOURCE_FILES: Tuple[str, ...] = (
    "src/routegate/finalrunner.py",
    "src/routegate/capabilityparsers.py",
    "src/routegate/capabilityscoring.py",
    "src/routegate/finaleval.py",
    "experiments/final_evaluation/run_final_evaluation.py",
    "experiments/final_evaluation/score_capability.py",
    "experiments/final_evaluation/verify_technical_smoke.py",
)


def source_digests() -> Dict[str, str]:
    """The digest of every file that defines v2 behaviour."""
    out: Dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = REPO_ROOT / relative
        out[relative] = sha256_file(path) if path.is_file() else ""
    return out
RECORDS_FILE = "records.jsonl"
MANIFEST_FILE = "run_manifest.json"
VALIDATION_FILE = "validation.json"
#: Append-only: one line per invocation, recording what it found and wrote.
INVOCATIONS_FILE = "invocations.jsonl"
#: Written only when every official row is present, unique and validates.
COMPLETION_FILE = "arm_complete.json"


def arm_id(dataset: str, condition: str) -> str:
    return f"{dataset}__{condition}"


def arm_output_dir(dataset: str, condition: str) -> str:
    return f"{DATA_ROOT_NAME}/{arm_id(dataset, condition)}"


def expected_rows(dataset: str) -> int:
    return int(DATASET_SPEC[dataset]["rows"])


class FinalEvaluationError(ValidationError):
    """The final evaluation was asked for something it must not do."""


# ============================================================ the run record
#: Every field one generated record carries. A record is one prompt under one
#: condition: what was read, what was decided, what was applied, what came out.
RECORD_FIELDS: Tuple[str, ...] = (
    "schema", "protocol", "arm_id", "dataset", "condition", "item_index",
    "item_id", "prompt_sha256", "status", "error_message",
    "model", "revision", "tokenizer_config_sha256", "chat_template_sha256",
    "decoding", "max_new_tokens", "do_sample", "sampling_seed",
    "gate_read_state", "gate_p_E", "gate_p_H", "gate_fires",
    "threshold_E", "threshold_H",
    "controller_read_state", "g_R", "applied_alpha_R",
    "applied_at_historical_blocks", "hooks_expected", "hooks_fired",
    "stop_strings", "stop_fired", "stop_reason",
    "response_text", "prompt_token_count", "completion_token_count",
    "eos_reached", "hit_max_new_tokens", "elapsed_seconds", "created_utc",
)
RECORD_SCHEMA = "routegate/final_evaluation_record/5"


def canonical(payload: Mapping[str, Any]) -> str:
    body = {k: v for k, v in payload.items() if k != DIGEST_FIELD}
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def record_sha256(payload: Mapping[str, Any]) -> str:
    return sha256_text(canonical(payload))


def build_protocol() -> Dict[str, Any]:
    """The committed payload. Deterministic: no timestamp, no environment."""
    from .capabilityscoring import (GSM8K_HARNESS, GSM8K_HARNESS_REVISION,
                                    GSM8K_HARNESS_TASK, GSM8K_NUMBER_REGEX,
                                    GSM8K_RULE, MMLU_GOLD_LETTERS, MMLU_RULE)
    record: Dict[str, Any] = {
        "schema": SCHEMA, "name": NAME, "version": VERSION,
        "status": "frozen",
        "supersedes": SUPERSEDES,
        "prior_protocols": list(PRIOR_PROTOCOLS),
        "invalidated_protocol": INVALIDATED_PROTOCOL,
        "invalidation_reason": list(INVALIDATION_REASON),
        "v1_generated_rows": 0,
        "v2_generated_rows": 0,
        "v2_generated_rows_evidence":
            "no output root for v2 was ever created; v3 supersedes it before "
            "any Qwen weight was loaded.",
        "v4_generated_rows": V4_GENERATED_ROWS,
        "v4_generated_detail": dict(V4_GENERATED_DETAIL),
        "v4_sealed_rows": 0,
        "gsm8k_boundary_correction": dict(GSM8K_BOUNDARY_CORRECTION),
        "gsm8k_reference": dict(GSM8K_REFERENCE),
        "reproduces_upstream_task_file": True,
        "reproduces_upstream_task_file_basis":
            "the stop sequences and both filter regexes are asserted against "
            "the vendored references/lm_eval_v0.4.3_gsm8k.yaml, pinned by "
            "sha256. This is a checked claim, not a remembered one.",
        "v3_generated_rows": V3_GENERATED_ROWS,
        "v3_generated_detail": dict(V3_GENERATED_DETAIL),
        "truncation_correction": dict(TRUNCATION_CORRECTION),
        "v3_changes": list(V3_CHANGES),
        "v3_changes_note": V3_CHANGES_NOTE,
        "scientific_protocol_unchanged_from_v2": True,
        "format_compatibility_rule": FORMAT_COMPATIBILITY_RULE,
        "behavioural_outcome_rule": BEHAVIOURAL_OUTCOME_RULE,
        "bookkeeping": {
            "invocations_file": INVOCATIONS_FILE,
            "completion_file": COMPLETION_FILE,
            "append_only": True,
            "completion_rule": "arm_complete.json is written only when every "
                               "official row is present, unique and its "
                               "item_index set is exactly 0..n-1. A one-row "
                               "invocation never claims completion.",
        },
        "v1_generated_rows_evidence":
            "no output root for v1 was ever created; the defects were found "
            "in a read-only audit before any Qwen weight was loaded.",
        "scoring": {
            "module": "routegate.capabilityscoring",
            "gold_never_visible_to_generation": True,
            "mmlu": {"rule": MMLU_RULE,
                     "gold_map": {str(k): v
                                  for k, v in MMLU_GOLD_LETTERS.items()}},
            "gsm8k": {"rule": GSM8K_RULE, "harness": GSM8K_HARNESS,
                      "harness_revision": GSM8K_HARNESS_REVISION,
                      "harness_task": GSM8K_HARNESS_TASK,
                      "regex": GSM8K_NUMBER_REGEX,
                      "comparison": "exact Decimal, never binary float"},
            "primary": ["raw accuracy", "paired item transitions vs baseline"],
            "secondary": ["tinyBenchmarks IRT / pIRT / gpIRT"],
        },
        "source_sha256": source_digests(),
        "frozen_before_any_sealed_prompt_was_read": True,
        "frozen_before_any_generation": True,
        "conditions": list(CONDITIONS),
        "conditions_are_complete": True,
        "condition_spec": {k: dict(v) for k, v in CONDITION_SPEC.items()},
        "rejected_candidates": dict(REJECTED_CANDIDATES),
        "rejected_candidates_note":
            "composed offline from existing adjudicated responses and "
            "rejected under a fixed eligibility rule before this protocol was "
            "written. Neither appears as a condition, in the runner, or in "
            "any output arm, and neither may be added later.",
        "datasets": list(DATASETS),
        "dataset_spec": {k: dict(v) for k, v in DATASET_SPEC.items()},
        "arms": [{"arm_id": arm_id(d, c), "dataset": d, "condition": c,
                  "rows": expected_rows(d),
                  "output_dir": arm_output_dir(d, c)} for d, c in ARMS],
        "arm_count": len(ARMS),
        "generations": {
            "sealed": SEALED_ROWS * len(CONDITIONS),
            "tinymmlu": CAPABILITY_ROWS * len(CONDITIONS),
            "tinygsm8k": CAPABILITY_ROWS * len(CONDITIONS),
            "per_condition": GENERATIONS_PER_CONDITION,
            "total": TOTAL_GENERATIONS,
            "inherited_or_reused": 0,
        },
        "ordering": {
            "rule": "every capability arm completes before the sealed split "
                    "is opened",
            "capability_first": [arm_id(d, c) for d, c in ARMS
                                 if d != DS_SEALED],
            "sealed_last": [arm_id(DS_SEALED, c) for c in CONDITIONS],
        },
        "runtime": {
            "model": FROZEN_MODEL, "revision": FROZEN_REVISION,
            "tokenizer_config_sha256": FROZEN_TOKENIZER_CONFIG_SHA256,
            "chat_template_sha256": FROZEN_CHAT_TEMPLATE_SHA256,
            "decoding": "greedy", "do_sample": False, "sampling_seed": None,
            "sampling_seed_is_null_because":
                "decoding is greedy and consults no random number generator",
            "batch_size": 1,
            "one_model_only": FROZEN_MODEL,
        },
        "sensors": {
            "directory": "outputs/routegate_gate_deployment_v1",
            "checkpoints": sorted(CHECKPOINT_FILES.values()),
            "bundle": BUNDLE_FILE,
            "thresholds_sha256": THRESHOLDS_SHA256,
            "read_state": GATE_READ_STATE,
            "requires_row_id": False,
            "reads_any_per_row_table": False,
            # Pinned here rather than in the deployment predeclaration,
            # which was frozen BEFORE the refit and so could not carry them.
            # outputs/ is never tracked by git, so these digests are the only
            # committed description of the weights the run must use.
            "weights_sha256": dict(SENSOR_WEIGHTS_SHA256),
            "controller_weights_sha256": CONTROLLER_WEIGHTS_SHA256,
            "controller": "routegate-data/gate_oof_v1_repro/"
                          "controller_all_development_refit.json",
            "reproduction_gate":
                "the refit was written only after the five fold sensors were "
                "reconstructed under the frozen specification and matched the "
                "frozen out-of-fold probabilities to 1e-9 (worst observed "
                "4.99e-11).",
        },
        "output": {
            "data_root": DATA_ROOT_NAME,
            "records_file": RECORDS_FILE,
            "resumable": True,
            "resume_rule": "an arm resumes by item_id: records already present "
                           "are kept, never regenerated and never overwritten. "
                           "The identical command resumes and completes.",
        },
        "record_schema": RECORD_SCHEMA,
        "record_fields": list(RECORD_FIELDS),
        "excluded": {
            "operators": ["SIGR", "angular steering", "static steering as a "
                          "standalone condition"],
            "benchmarks": ["HellaSwag", "ARC-Challenge", "TruthfulQA",
                           "full MMLU", "full GSM8K"],
            "models": "no model other than " + FROZEN_MODEL,
            "note": "none of these is run, loaded or referenced by any arm.",
        },
        "judging": {
            "performed_by_this_stage": False,
            "note": "this stage generates and records. Adjudication is a "
                    "separate stage and makes its own calls.",
        },
        "sealed_set": {
            "expected_row_count": SEALED_ROWS,
            "path_stored_in_this_repository": False,
            "prompt_content_read_by_this_module": False,
            "commitment": "manifests/sealed_set_commitment.json",
            "opened": "once, after every capability arm completes, as an "
                      "explicit argument supplied at launch",
        },
        "contains_prompt_or_response_text": False,
        "contains_sealed_material": False,
    }
    record[DIGEST_FIELD] = record_sha256(record)
    return record


def verify_protocol(path: Path | str = PREDECLARATION,
                    checks: Optional[Checklist] = None) -> Dict[str, Any]:
    """The committed protocol is well formed, self-consistent and current."""
    checks = checks or Checklist(NAME)
    path = Path(path)
    checks.require("protocol_exists", path.is_file(), str(path))
    committed = json.loads(path.read_text(encoding="utf-8"))
    checks.require("schema", committed.get("schema") == SCHEMA,
                   f"{committed.get('schema')!r}")
    recomputed = record_sha256(committed)
    checks.require("record_digest", recomputed == committed.get(DIGEST_FIELD),
                   f"records {committed.get(DIGEST_FIELD)}")
    current = build_protocol()
    drifted = sorted(k for k in set(current) | set(committed)
                     if k != DIGEST_FIELD and current.get(k) != committed.get(k))
    checks.require("protocol_matches_the_package", not drifted,
                   f"{len(drifted)} field(s) differ: {drifted[:5]}" if drifted
                   else f"{len(current) - 1} field(s), every one identical")
    checks.require("exactly_three_conditions",
                   committed.get("conditions") == list(CONDITIONS)
                   and committed.get("conditions_are_complete") is True,
                   f"{committed.get('conditions')}")
    rejected = committed.get("rejected_candidates") or {}
    checks.require("both_hybrids_are_recorded_as_rejected",
                   set(rejected) == set(REJECTED_CANDIDATES),
                   f"{sorted(rejected)}")
    checks.require("no_hybrid_is_a_condition",
                   not (set(rejected) & set(committed.get("conditions", []))),
                   "neither rejected hybrid is a condition")
    gen = committed.get("generations") or {}
    checks.require("generation_counts",
                   gen.get("total") == 1740
                   and gen.get("per_condition") == 580
                   and gen.get("sealed") == 1140
                   and gen.get("tinymmlu") == 300
                   and gen.get("tinygsm8k") == 300,
                   f"{gen}")
    checks.require("nine_arms", committed.get("arm_count") == 9
                   and len(committed.get("arms", [])) == 9,
                   f"{committed.get('arm_count')}")
    for dataset in (DS_MMLU, DS_GSM8K):
        spec = (committed.get("dataset_spec") or {}).get(dataset) or {}
        checks.require(f"{dataset}_is_100_items", spec.get("rows") == 100,
                       f"{spec.get('rows')}")
        checks.require(f"{dataset}_is_pinned",
                       len(str(spec.get("revision", ""))) == 40
                       and len(str(spec.get("sha256", ""))) == 64,
                       f"rev {spec.get('revision')} sha {spec.get('sha256')}")
    sealed = committed.get("sealed_set") or {}
    checks.require("sealed_is_the_frozen_380",
                   sealed.get("expected_row_count") == 380
                   and sealed.get("path_stored_in_this_repository") is False
                   and sealed.get("prompt_content_read_by_this_module") is False,
                   f"{sealed.get('expected_row_count')}")
    sensors = committed.get("sensors") or {}
    checks.require("sensor_weights_are_pinned",
                   sensors.get("weights_sha256") == dict(SENSOR_WEIGHTS_SHA256)
                   and sensors.get("controller_weights_sha256")
                   == CONTROLLER_WEIGHTS_SHA256,
                   f"{sensors.get('weights_sha256')}")
    checks.require("v4_rows_are_recorded_and_excluded",
                   committed.get("v4_generated_rows") == V4_GENERATED_ROWS
                   and committed.get("v4_sealed_rows") == 0
                   and (committed.get("v4_generated_detail") or {}).get(
                       "tinymmlu_rows") == 3
                   and (committed.get("v4_generated_detail") or {}).get(
                       "tinygsm8k_rows") == 3
                   and (committed.get("v4_generated_detail") or {}).get(
                       "preserved") is True
                   and (committed.get("v4_generated_detail") or {}).get(
                       "excluded_from_v5") is True,
                   f"{committed.get('v4_generated_rows')} v4 row(s), "
                   f"3 MMLU + 3 GSM8K, 0 sealed, preserved and excluded")
    reference = committed.get("gsm8k_reference") or {}
    checks.require("gsm8k_reference_is_vendored_and_pinned",
                   (REPO_ROOT / reference.get("file", "")).is_file()
                   and sha256_file(REPO_ROOT / reference["file"])
                   == reference.get("sha256"),
                   f"{reference.get('file')} @ {reference.get('tag')}")
    boundary = committed.get("gsm8k_boundary_correction") or {}
    checks.require("gsm8k_boundary_is_gsm8k_only",
                   boundary.get("applies_to") == "tinygsm8k only"
                   and boundary.get("parser_unchanged") is True
                   and boundary.get("scoring_definition_unchanged") is True
                   and boundary.get("mmlu_unchanged") is True,
                   "a boundary repair, not a scoring change")
    checks.require("gsm8k_stop_strings_match_the_vendored_reference",
                   list((committed["dataset_spec"][DS_GSM8K]
                         or {}).get("stop_strings") or [])
                   == list(GSM8K_STOP_STRINGS)
                   and not (committed["dataset_spec"][DS_MMLU]
                            or {}).get("stop_strings"),
                   f"{GSM8K_STOP_STRINGS}; MMLU has none")
    checks.require("v3_rows_are_recorded_and_excluded",
                   committed.get("v3_generated_rows") == V3_GENERATED_ROWS
                   and (committed.get("v3_generated_detail") or {}).get(
                       "tinygsm8k_rows") == 0
                   and (committed.get("v3_generated_detail") or {}).get(
                       "preserved") is True
                   and (committed.get("v3_generated_detail") or {}).get(
                       "excluded_from_v4") is True,
                   f"{committed.get('v3_generated_rows')} v3 row(s), "
                   f"preserved and excluded")
    correction = committed.get("truncation_correction") or {}
    checks.require("truncation_correction_is_outcome_blind",
                   correction.get("inspected_predictions") is False
                   and correction.get("inspected_gold") is False
                   and correction.get("inspected_correctness") is False
                   and correction.get("inspected_scores") is False
                   and correction.get("evidence_is_condition_independent")
                   is True,
                   "motivated by parse_status and hit_max_new_tokens alone")
    checks.require("truncation_diagnostic_label",
                   correction.get("diagnostic_label")
                   == "PARSE_OK_TRUNCATED_AFTER_ANSWER",
                   f"{correction.get('diagnostic_label')!r}")
    checks.require("parsers_and_generation_unchanged",
                   correction.get("parsers_unchanged") is True
                   and correction.get("generation_settings_unchanged") is True,
                   "no parser or generation setting was altered")
    checks.require("supersedes_v3",
                   committed.get("supersedes") == SUPERSEDES
                   and committed.get("v1_generated_rows") == 0
                   and committed.get("v2_generated_rows") == 0,
                   f"{committed.get('supersedes')}, v1 rows "
                   f"{committed.get('v1_generated_rows')}, v2 rows "
                   f"{committed.get('v2_generated_rows')}")
    checks.require("scientific_protocol_unchanged_from_v2",
                   committed.get("scientific_protocol_unchanged_from_v2")
                   is True and len(committed.get("v3_changes") or []) >= 3,
                   f"{len(committed.get('v3_changes') or [])} v3 change(s)")
    checks.require("format_and_behavioural_rules_recorded",
                   "BLOCKED_FORMAT_COMPATIBILITY"
                   in str(committed.get("format_compatibility_rule", ""))
                   or "format-compatibility"
                   in str(committed.get("format_compatibility_rule", "")),
                   "the baseline-only format gate is recorded")
    checks.require("behavioural_outcomes_are_retained",
                   "BEHAVIOURAL_PARSE_FAILURE"
                   in str(committed.get("behavioural_outcome_rule", "")),
                   "non-baseline parse failures are behavioural, not "
                   "technical exclusions")
    checks.require("invalidation_reason_lists_the_defects",
                   len(committed.get("invalidation_reason") or []) >= 4,
                   f"{len(committed.get('invalidation_reason') or [])} reason(s)")
    checks.require("capability_prompts_are_raw_completion",
                   all((committed["dataset_spec"][d].get("prompt_mode")
                        == "raw_completion") for d in (DS_MMLU, DS_GSM8K)),
                   "tinyMMLU and tinyGSM8k bypass the chat template")
    checks.require("sealed_prompts_stay_chat",
                   committed["dataset_spec"][DS_SEALED].get("prompt_mode")
                   == "chat", "the sealed path is unchanged")
    checks.require("source_digests_match",
                   committed.get("source_sha256") == source_digests(),
                   "every v2 source file hashes to what the protocol records")
    checks.require("frozen_before_sealed_and_generation",
                   committed.get("frozen_before_any_sealed_prompt_was_read")
                   is True
                   and committed.get("frozen_before_any_generation") is True,
                   "frozen first")
    return {"path": str(path), "sha256": sha256_file(path),
            DIGEST_FIELD: recomputed}


__all__ = [
    "ARMS", "CAPABILITY_ROWS", "CONDITIONS", "CONTROLLER_WEIGHTS_SHA256", "SENSOR_WEIGHTS_SHA256", "CONDITION_SPEC",
    "COND_BASELINE", "COND_DYNAMIC", "COND_ROUTEGATE", "DATASETS",
    "DATASET_SPEC", "DATA_ROOT_NAME", "DIGEST_FIELD", "DS_GSM8K", "DS_MMLU",
    "DS_SEALED", "FinalEvaluationError", "GENERATIONS_PER_CONDITION",
    "COMPLETION_FILE", "INVOCATIONS_FILE",
    "MANIFEST_FILE", "NAME", "PREDECLARATION", "RECORDS_FILE",
    "RECORD_FIELDS", "RECORD_SCHEMA", "REJECTED_CANDIDATES", "SCHEMA",
    "SEALED_ROWS", "TOTAL_GENERATIONS", "VALIDATION_FILE", "VERSION",
    "BEHAVIOURAL_OUTCOME_RULE", "FORMAT_COMPATIBILITY_RULE",
    "GSM8K_BOUNDARY_CORRECTION", "GSM8K_REFERENCE", "GSM8K_STOP_STRINGS",
    "MMLU_STOP_NOTE", "TRUNCATION_CORRECTION",
    "V3_GENERATED_DETAIL", "V3_GENERATED_ROWS",
    "V4_GENERATED_DETAIL", "V4_GENERATED_ROWS",
    "INVALIDATION_REASON", "INVALIDATED_PROTOCOL", "PRIOR_PROTOCOLS",
    "SOURCE_FILES", "V3_CHANGES", "V3_CHANGES_NOTE",
    "SUPERSEDES", "source_digests",
    "arm_id", "arm_output_dir", "build_protocol", "canonical",
    "expected_rows", "record_sha256", "verify_protocol",
]
