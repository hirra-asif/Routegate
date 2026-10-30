"""A token cap after a parsed answer is diagnostic, not a failure.

v3 conflated the two. It blocked baseline format compatibility when the answer
had parsed but the cap had been hit, called successfully parsed steered rows
BEHAVIOURAL_PARSE_FAILURE, and would have forced parsed capped rows to score
incorrect. Under raw completion the model emits the option letter and then
carries on into the next few-shot question until the budget ends: the cap falls
AFTER the answer.

These tests hold v4 to the corrected rule -- parse status and cap status are
separate, and only an actual parse failure is a finding.
"""
from __future__ import annotations

import importlib.util
import json
from decimal import Decimal
from pathlib import Path

import pytest

from routegate.capabilityparsers import PARSE_OK, parse_gsm8k, parse_mmlu
from routegate.capabilityscoring import (
    TRUNCATED_AFTER_ANSWER, TRUNCATION_RULE, score_arm, score_item)
from routegate.finaleval import (
    ARMS, COND_BASELINE, COND_DYNAMIC, COND_ROUTEGATE, DATA_ROOT_NAME,
    DS_GSM8K, DS_MMLU, NAME, PRIOR_PROTOCOLS, RECORD_SCHEMA, SUPERSEDES,
    TRUNCATION_CORRECTION, V3_GENERATED_DETAIL, V3_GENERATED_ROWS,
    arm_output_dir, build_protocol)
from routegate.validation import Checklist, ValidationError
from conftest import needs_gate_thresholds

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "experiments/final_evaluation/run_final_evaluation.py"
VERIFIER = REPO / "experiments/final_evaluation/verify_technical_smoke.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load(RUNNER, "_runner_v4")
verifier = load(VERIFIER, "_verifier_v4")

# The exact shape the v3 smoke produced: a parsed answer, then the cap.
MMLU_CAPPED = "B\n\nQuestion: the next few-shot item begins and is cut off"
GSM8K_CAPPED = "We compute 6 * 7 = 42\n#### 42\n\nQuestion: the next item"


def _record(condition, dataset=DS_MMLU, text=MMLU_CAPPED, capped=True, **kw):
    from routegate.finalrunner import TELEMETRY_FIELDS
    from routegate.finaleval import DATASET_SPEC
    base = {
        "schema": RECORD_SCHEMA, "condition": condition, "item_index": 0,
        "item_id": f"{dataset}:0000", "prompt_sha256": "d" * 64,
        "prompt_mode": DATASET_SPEC[dataset]["prompt_mode"], "status": "ok",
        "response_text": text, "completion_token_count": 8,
        "eos_reached": not capped, "hit_max_new_tokens": capped,
        "mode": "off", "noop_reason": None,
        **{f: None for f in TELEMETRY_FIELDS},
    }
    base.update({"hooks_expected": 0, "hooks_fired": 0})
    base.update(kw)
    return base


# ---------------- 1, 2, 3. parse-ok + cap technically passes everywhere
def test_a_capped_but_parsed_completion_parses_ok():
    assert parse_mmlu(MMLU_CAPPED)[1] == PARSE_OK
    assert parse_gsm8k(GSM8K_CAPPED)[1] == PARSE_OK


def test_baseline_parse_ok_plus_cap_technically_passes():
    checks = Checklist("t")
    verifier.check_baseline(_record(COND_BASELINE), checks)
    assert checks.all_passed
    _p, status = parse_mmlu(MMLU_CAPPED)
    assert status == PARSE_OK
    # the corrected verdict: not a format block
    assert verifier.STATUS_FORMAT == "BLOCKED_FORMAT_COMPATIBILITY"
    assert verifier.TRUNCATED_AFTER_ANSWER == "PARSE_OK_TRUNCATED_AFTER_ANSWER"


def test_dynamic_parse_ok_plus_cap_technically_passes():
    record = _record(COND_DYNAMIC, mode="patch", controller_read_state=32,
                     g_R=0.5, applied_alpha_R=0.075,
                     applied_at_historical_blocks=[31, 32],
                     hooks_expected=2, hooks_fired=2)
    checks = Checklist("t")
    verifier.check_dynamic(record, checks)
    assert checks.all_passed
    assert parse_mmlu(record["response_text"])[1] == PARSE_OK


@needs_gate_thresholds
def test_safety_first_parse_ok_plus_cap_technically_passes():
    from routegate.gatesensors import load_thresholds
    t = load_thresholds()
    record = _record(COND_ROUTEGATE, mode="patch", gate_read_state=45,
                     gate_p_E=0.9, gate_p_H=0.01, gate_fires=True,
                     threshold_E=t.threshold_E, threshold_H=t.threshold_H,
                     applied_at_historical_blocks=[31, 32, 33],
                     applied_alpha_R=0.15, hooks_expected=3, hooks_fired=3)
    checks = Checklist("t")
    verifier.check_routegate(record, checks)
    assert checks.all_passed
    assert parse_mmlu(record["response_text"])[1] == PARSE_OK


def test_the_verifier_no_longer_blocks_on_a_cap_alone():
    source = VERIFIER.read_text(encoding="utf-8")
    # the v3 condition that coupled the cap to the baseline gate is gone
    assert "args.dataset == DS_MMLU and capped" not in source
    assert "if status == PARSE_OK:" in source
    assert "PARSE_OK_TRUNCATED_AFTER_ANSWER" in source


def test_parse_status_and_cap_status_are_reported_separately():
    source = VERIFIER.read_text(encoding="utf-8")
    assert '_parse_status", True, str(status))' in source
    assert '_hit_max_new_tokens", True, str(capped))' in source


# ------------------- 4, 5. a parsed capped answer is scored normally
def test_a_capped_mmlu_answer_is_scored_normally_not_forced_incorrect():
    capped = score_item(DS_MMLU, 0, "x", MMLU_CAPPED, 1, truncated=True)
    plain = score_item(DS_MMLU, 0, "x", "B", 1, truncated=False)
    assert capped.correct is plain.correct is True
    assert capped.parse_status == PARSE_OK
    assert capped.diagnostic == TRUNCATED_AFTER_ANSWER
    assert plain.diagnostic is None


def test_a_capped_gsm8k_answer_is_scored_normally():
    capped = score_item(DS_GSM8K, 0, "x", GSM8K_CAPPED, "work\n#### 42",
                        truncated=True)
    assert capped.correct is True
    assert capped.parse_status == PARSE_OK
    assert capped.diagnostic == TRUNCATED_AFTER_ANSWER


def test_truncation_never_changes_correctness_either_way():
    for gold, expected in ((1, True), (0, False)):
        with_cap = score_item(DS_MMLU, 0, "x", MMLU_CAPPED, gold,
                              truncated=True)
        without = score_item(DS_MMLU, 0, "x", MMLU_CAPPED, gold,
                             truncated=False)
        assert with_cap.correct is without.correct is expected


def test_score_arm_counts_truncations_without_penalising_them():
    golds = [1] * 100
    records = [{"item_index": i, "item_id": f"tinymmlu:{i:04d}",
                "response_text": MMLU_CAPPED, "hit_max_new_tokens": True}
               for i in range(100)]
    reading = score_arm(DS_MMLU, records, golds)
    assert reading["correct"] == 100
    assert reading["parsed_then_truncated"] == 100
    assert TRUNCATED_AFTER_ANSWER in reading["truncation_rule"]


# ------------------------------ 6, 7. real parse failures still matter
def test_a_baseline_parse_failure_still_blocks_format_compatibility():
    _p, status = parse_mmlu("Let me think about this carefully")
    assert status != PARSE_OK
    assert verifier.STATUS_FORMAT == "BLOCKED_FORMAT_COMPATIBILITY"
    source = VERIFIER.read_text(encoding="utf-8")
    assert "verdict = STATUS_FORMAT" in source


@pytest.mark.parametrize("condition", [COND_DYNAMIC, COND_ROUTEGATE])
def test_a_non_baseline_parse_failure_is_retained_and_scored_incorrect(
        condition):
    scored = score_item(DS_MMLU, 0, "x", "it refused to answer", 1,
                        truncated=False)
    assert scored.parse_status != PARSE_OK
    assert scored.correct is False
    assert scored.diagnostic is None
    assert verifier.BEHAVIOURAL == "BEHAVIOURAL_PARSE_FAILURE"


def test_a_gsm8k_parse_failure_is_scored_incorrect():
    scored = score_item(DS_GSM8K, 0, "x", "no numbers at all", "w\n#### 42")
    assert scored.parse_status != PARSE_OK and scored.correct is False


# ---------------------------------------- 8. outcome blindness preserved
def test_the_verifier_still_reads_no_gold_and_prints_no_prediction():
    source = VERIFIER.read_text(encoding="utf-8")
    imports = "\n".join(l for l in source.splitlines()
                        if l.strip().startswith(("import ", "from ")))
    assert "capabilityscoring" not in imports
    for banned in ("mmlu_gold_letter", "gsm8k_gold_value", "score_arm",
                   "score_item", "answer_column"):
        assert banned not in source
    for line in source.splitlines():
        if "print(" in line:
            assert "{_prediction" not in line and "{prediction" not in line


def test_the_correction_was_reached_without_inspecting_outcomes():
    assert TRUNCATION_CORRECTION["inspected_predictions"] is False
    assert TRUNCATION_CORRECTION["inspected_gold"] is False
    assert TRUNCATION_CORRECTION["inspected_correctness"] is False
    assert TRUNCATION_CORRECTION["inspected_scores"] is False
    assert TRUNCATION_CORRECTION["evidence_is_condition_independent"] is True


# ------------------------------- 9, 10. v4 isolation and resume arithmetic
def test_v3_rows_cannot_enter_or_be_resumed_into_v5():
    assert DATA_ROOT_NAME == "final_evaluation_v5"
    assert SUPERSEDES == "routegate_final_evaluation_v4"
    assert RECORD_SCHEMA.endswith("/5")
    for dataset, condition in ARMS:
        path = arm_output_dir(dataset, condition)
        assert path.startswith("final_evaluation_v5/")
        for older in ("final_evaluation_v1/", "final_evaluation_v2/",
                      "final_evaluation_v3/", "final_evaluation_v4/"):
            assert older not in path
    for older in ("routegate/final_evaluation_record/4",
                  "routegate/final_evaluation_record/3",
                  "routegate/final_evaluation_record/2"):
        assert older != RECORD_SCHEMA


def test_the_v3_rows_are_recorded_as_preserved_and_excluded():
    """v3's census is retained across successors; v4's sits beside it."""
    protocol = build_protocol()
    assert protocol["v3_generated_rows"] == V3_GENERATED_ROWS == 3
    assert protocol["v4_generated_rows"] == 6
    detail = protocol["v3_generated_detail"]
    assert detail["dataset"] == "tinymmlu" and detail["item_index"] == 0
    assert detail["one_per_condition"] is True
    assert detail["tinygsm8k_rows"] == 0 and detail["sealed_rows"] == 0
    assert detail["preserved"] is True and detail["excluded_from_v4"] is True
    assert len(detail["arms"]) == 3
    assert "routegate_final_evaluation_v3" in PRIOR_PROTOCOLS


def test_the_preserved_v3_rows_are_still_on_disk_untouched():
    root = Path.home() / "routegate-data" / "final_evaluation_v3"
    if not root.exists():
        pytest.skip("the v3 smoke rows are not present on this machine")
    total = 0
    for arm in V3_GENERATED_DETAIL["arms"]:
        path = root / arm / "records.jsonl"
        assert path.is_file(), arm
        rows = [json.loads(l) for l in
                path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(rows) == 1 and rows[0]["item_index"] == 0
        total += 1
    assert total == 3
    for arm in ("tinygsm8k__baseline", "tinygsm8k__dynamic_raw",
                "tinygsm8k__safety_first_routegate"):
        assert not (root / arm / "records.jsonl").exists()


def test_v4_smoke_then_resume_is_still_one_then_ninety_nine(tmp_path):
    items = [{"item_index": i, "item_id": f"tinymmlu:{i:04d}",
              "prompt": f"p{i}"} for i in range(100)]
    records = tmp_path / "records.jsonl"
    smoke = runner.select_items(items, 0, 100, Checklist("t"))
    assert len(smoke) == 1
    records.parent.mkdir(parents=True, exist_ok=True)
    records.write_text(json.dumps({"schema": RECORD_SCHEMA, "item_index": 0,
                                   "item_id": "tinymmlu:0000"}) + "\n")
    done = runner.completed_items(records)
    remaining = [i for i in runner.select_items(items, None, 100,
                                                Checklist("t"))
                 if i["item_id"] not in done]
    assert len(done) == 1 and len(remaining) == 99
    assert runner.finalise_if_complete(tmp_path, records, 100,
                                       Checklist("t")) is False


def test_the_frozen_settings_are_unchanged_from_v3():
    from routegate.finaleval import DATASET_SPEC, expected_rows
    assert DATASET_SPEC[DS_MMLU]["max_new_tokens"] == 8
    assert DATASET_SPEC[DS_GSM8K]["max_new_tokens"] == 512
    assert DATASET_SPEC[DS_MMLU]["prompt_mode"] == "raw_completion"
    assert DATASET_SPEC[DS_GSM8K]["prompt_mode"] == "raw_completion"
    assert DATASET_SPEC["sealed"]["prompt_mode"] == "chat"
    for spec in DATASET_SPEC.values():
        assert spec["decoding"] == "greedy"
    assert expected_rows(DS_MMLU) == expected_rows(DS_GSM8K) == 100
    assert build_protocol()["generations"]["total"] == 1740
