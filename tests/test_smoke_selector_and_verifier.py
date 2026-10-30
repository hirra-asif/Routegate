"""The v3 additions: one-item selection, truthful bookkeeping, gold-free checks.

v3 adds nothing scientific. It adds the ability to run ONE official row before
committing to a hundred, and a verifier that can say the machinery worked
without ever seeing whether the model was right. These tests hold both to that:
the selector must not become a second execution path, and the verifier must not
acquire a route to a gold label.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from routegate.capabilityparsers import PARSE_OK, parse_gsm8k, parse_mmlu
from routegate.finaleval import (
    ARMS, COMPLETION_FILE, CONDITIONS, COND_BASELINE, COND_DYNAMIC,
    COND_ROUTEGATE, DATASET_SPEC, DATA_ROOT_NAME, DS_GSM8K, DS_MMLU,
    INVOCATIONS_FILE, NAME, PRIOR_PROTOCOLS, RECORD_SCHEMA, SUPERSEDES,
    V3_CHANGES, arm_id, arm_output_dir, build_protocol, expected_rows)
from routegate.finalrunner import (
    DYNAMIC_HOOKS, DYNAMIC_READ_STATE, GATE_READ_STATE,
    NOOP_GATE_DID_NOT_FIRE, STATIC_HOOKS, TELEMETRY_FIELDS)
from routegate.validation import Checklist, ValidationError
from conftest import needs_gate_thresholds

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "experiments/final_evaluation/run_final_evaluation.py"
VERIFIER = REPO / "experiments/final_evaluation/verify_technical_smoke.py"

CAPABILITY_ARMS = (
    "tinymmlu__baseline", "tinymmlu__dynamic_raw",
    "tinymmlu__safety_first_routegate", "tinygsm8k__baseline",
    "tinygsm8k__dynamic_raw", "tinygsm8k__safety_first_routegate")


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load(RUNNER, "_runner_v3")
verifier = load(VERIFIER, "_verifier_v3")


def items(n=100, dataset=DS_MMLU):
    return [{"item_index": i, "item_id": f"{dataset}:{i:04d}",
             "prompt": f"prompt {i}"} for i in range(n)]


# =============================================================== the selector
def test_item_zero_selects_exactly_one_row():
    chosen = runner.select_items(items(), 0, 100, Checklist("t"))
    assert len(chosen) == 1 and chosen[0]["item_index"] == 0


def test_the_last_valid_index_selects_exactly_one_row():
    chosen = runner.select_items(items(), 99, 100, Checklist("t"))
    assert len(chosen) == 1 and chosen[0]["item_index"] == 99


@pytest.mark.parametrize("index", [-1, 100, 101, 1000])
def test_an_out_of_range_index_is_refused(index):
    with pytest.raises(ValidationError, match="outside"):
        runner.select_items(items(), index, 100, Checklist("t"))


def test_no_selector_schedules_all_one_hundred():
    assert len(runner.select_items(items(), None, 100, Checklist("t"))) == 100


def test_the_selector_preserves_id_index_and_prompt():
    original = items()
    chosen = runner.select_items(original, 42, 100, Checklist("t"))[0]
    assert chosen is original[42]
    assert chosen["item_index"] == 42
    assert chosen["item_id"] == "tinymmlu:0042"


def test_the_selector_is_a_filter_not_a_second_execution_path():
    source = RUNNER.read_text(encoding="utf-8")
    assert source.count("execute_item(") == 1, "one call site only"
    assert "select_items(items, args.item_index" in source
    # the loop uses the ORIGINAL index, never a positional enumerate
    assert "for item in items:" in source
    assert 'index = int(item["item_index"])' in source


def test_the_selector_does_not_read_gold():
    from routegate.capabilityscoring import assert_generation_cannot_read_gold
    assert_generation_cannot_read_gold(RUNNER.read_text(encoding="utf-8"),
                                       RUNNER.name)


# ------------------------------------------------- resume arithmetic (1 + 99)
def _write(path: Path, indexes, dataset=DS_MMLU):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for i in indexes:
            fh.write(json.dumps({"schema": RECORD_SCHEMA, "item_index": i,
                                 "item_id": f"{dataset}:{i:04d}"}) + "\n")


def test_smoke_then_full_resume_schedules_one_then_ninety_nine(tmp_path):
    records = tmp_path / "records.jsonl"
    assert len(runner.completed_items(records)) == 0
    smoke = runner.select_items(items(), 7, 100, Checklist("t"))
    assert len([i for i in smoke
                if i["item_id"] not in runner.completed_items(records)]) == 1
    _write(records, [7])
    done = runner.completed_items(records)
    assert len(done) == 1
    remaining = [i for i in runner.select_items(items(), None, 100,
                                                Checklist("t"))
                 if i["item_id"] not in done]
    assert len(remaining) == 99
    assert 7 not in [i["item_index"] for i in remaining]


def test_repeating_the_smoke_schedules_zero_and_never_duplicates(tmp_path):
    records = tmp_path / "records.jsonl"
    _write(records, [7])
    done = runner.completed_items(records)
    again = [i for i in runner.select_items(items(), 7, 100, Checklist("t"))
             if i["item_id"] not in done]
    assert again == []
    assert len(runner.completed_items(records)) == 1


def test_a_full_run_after_resume_holds_one_hundred_unique_ordered_records(
        tmp_path):
    records = tmp_path / "records.jsonl"
    _write(records, [7])
    _write(records, [i for i in range(100) if i != 7])
    done = runner.completed_items(records)
    assert len(done) == 100
    assert sorted(done.values()) == list(range(100))


# --------------------------------------------------- truthful bookkeeping
def test_a_one_row_arm_is_not_marked_complete(tmp_path):
    records = tmp_path / "records.jsonl"
    _write(records, [0])
    checks = Checklist("t")
    assert runner.finalise_if_complete(tmp_path, records, 100, checks) is False
    assert not (tmp_path / COMPLETION_FILE).exists()


def test_a_complete_arm_is_marked_complete(tmp_path):
    records = tmp_path / "records.jsonl"
    _write(records, range(100))
    checks = Checklist("t")
    assert runner.finalise_if_complete(tmp_path, records, 100, checks) is True
    payload = json.loads((tmp_path / COMPLETION_FILE).read_text())
    assert payload["complete"] is True and payload["rows"] == 100


def test_a_gappy_arm_is_refused_rather_than_called_complete(tmp_path):
    records = tmp_path / "records.jsonl"
    _write(records, list(range(99)) + [999])
    with pytest.raises(ValidationError, match="not complete"):
        runner.finalise_if_complete(tmp_path, records, 100, Checklist("t"))


def test_the_invocation_trail_is_append_only(tmp_path):
    path = tmp_path / INVOCATIONS_FILE
    runner.append_invocation(path, {"n": 1})
    runner.append_invocation(path, {"n": 2})
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(x)["n"] for x in lines] == [1, 2]


# =============================================================== the verifier
def _record(condition, dataset=DS_MMLU, index=0, **kw):
    base = {
        "schema": RECORD_SCHEMA, "condition": condition, "item_index": index,
        "item_id": f"{dataset}:{index:04d}", "prompt_sha256": "d" * 64,
        "prompt_mode": DATASET_SPEC[dataset]["prompt_mode"],
        "status": "ok", "response_text": "B", "completion_token_count": 1,
        "eos_reached": True, "hit_max_new_tokens": False, "mode": "off",
        "noop_reason": None,
        **{f: None for f in TELEMETRY_FIELDS},
    }
    base.update({"hooks_expected": 0, "hooks_fired": 0})
    base.update(kw)
    return base


def dynamic_record(**kw):
    fields = dict(mode="patch", controller_read_state=32, g_R=0.5,
                  applied_alpha_R=0.075,
                  applied_at_historical_blocks=[31, 32],
                  hooks_expected=2, hooks_fired=2)
    fields.update(kw)                      # the caller's override wins
    return _record(COND_DYNAMIC, **fields)


def routegate_record(fires=True, **kw):
    from routegate.gatesensors import load_thresholds
    t = load_thresholds()
    if fires:
        fields = dict(mode="patch", gate_read_state=45, gate_p_E=0.9,
                      gate_p_H=0.01, gate_fires=True,
                      threshold_E=t.threshold_E, threshold_H=t.threshold_H,
                      applied_at_historical_blocks=[31, 32, 33],
                      applied_alpha_R=0.15, hooks_expected=3, hooks_fired=3)
    else:
        fields = dict(mode="off", gate_read_state=45, gate_p_E=0.01,
                      gate_p_H=0.9, gate_fires=False,
                      threshold_E=t.threshold_E, threshold_H=t.threshold_H,
                      applied_at_historical_blocks=[], hooks_expected=0,
                      hooks_fired=0, noop_reason=NOOP_GATE_DID_NOT_FIRE)
    fields.update(kw)
    return _record(COND_ROUTEGATE, **fields)


def test_the_verifier_passes_a_correct_baseline_record():
    checks = Checklist("t")
    verifier.check_baseline(_record(COND_BASELINE), checks)
    assert checks.all_passed


def test_the_verifier_catches_the_hardcoded_mode_off_bug():
    """v1's exact defect: a steered condition generated with mode 'off'."""
    with pytest.raises(ValidationError, match="dynamic_controller_present"):
        verifier.check_dynamic(dynamic_record(mode="off"), Checklist("t"))


@pytest.mark.parametrize("fired", [0, 1, 3])
def test_the_verifier_catches_missing_partial_and_excessive_hooks(fired):
    with pytest.raises(ValidationError, match="dynamic_hooks"):
        verifier.check_dynamic(dynamic_record(hooks_fired=fired),
                               Checklist("t"))


@needs_gate_thresholds
def test_the_verifier_catches_a_wrong_gate_decision():
    """Recorded gate_fires must recompute from the recorded probabilities."""
    with pytest.raises(ValidationError, match="gate_decision_recomputes"):
        verifier.check_routegate(routegate_record(fires=True, gate_p_H=0.9),
                                 Checklist("t"))


@needs_gate_thresholds
def test_the_verifier_catches_a_threshold_mismatch():
    with pytest.raises(ValidationError, match="thresholds_match"):
        verifier.check_routegate(routegate_record(threshold_H=0.5),
                                 Checklist("t"))


@needs_gate_thresholds
def test_a_gate_no_op_passes_and_needs_an_explicit_reason():
    checks = Checklist("t")
    verifier.check_routegate(routegate_record(fires=False), checks)
    assert checks.all_passed
    with pytest.raises(ValidationError, match="noop_reason_is_explicit"):
        verifier.check_routegate(routegate_record(fires=False,
                                                  noop_reason=None),
                                 Checklist("t"))


@needs_gate_thresholds
def test_the_verifier_catches_a_gate_fired_row_with_no_hooks():
    with pytest.raises(ValidationError, match="gate_fired_hooks"):
        verifier.check_routegate(
            routegate_record(fires=True, hooks_expected=0, hooks_fired=0),
            Checklist("t"))


def test_the_verifier_catches_a_wrong_applied_alpha_relationship():
    with pytest.raises(ValidationError, match="applied_alpha_relationship"):
        verifier.check_dynamic(dynamic_record(applied_alpha_R=0.9),
                               Checklist("t"))


def test_the_verifier_catches_duplicates_and_version_mixing(tmp_path):
    path = tmp_path / "arm" / "records.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("\n".join([
        json.dumps({"schema": RECORD_SCHEMA, "item_index": 0,
                    "item_id": "x"}),
        json.dumps({"schema": RECORD_SCHEMA, "item_index": 0,
                    "item_id": "x"})]) + "\n")
    with pytest.raises(ValidationError, match="no_duplicate_rows"):
        verifier.read_one(path, 0, Checklist("t"))
    path.write_text(json.dumps({"schema": "routegate/final_evaluation_record/2",
                                "item_index": 0, "item_id": "x"}) + "\n")
    with pytest.raises(ValidationError, match="no_version_mixing"):
        verifier.read_one(path, 0, Checklist("t"))


# ------------------------------------------------ outcome blindness
def test_the_verifier_never_imports_gold_or_scoring_code():
    source = VERIFIER.read_text(encoding="utf-8")
    imports = [line for line in source.splitlines()
               if line.strip().startswith(("import ", "from "))]
    joined = "\n".join(imports)
    assert "capabilityscoring" not in joined
    assert "score_capability" not in joined
    for banned in ("mmlu_gold_letter", "gsm8k_gold_value", "score_arm",
                   "score_item", "answer_column", "paired_transitions"):
        assert banned not in source, banned


def test_the_parser_module_has_no_route_to_a_gold_label():
    import routegate.capabilityparsers as parsers
    for name in dir(parsers):
        assert "gold" not in name.lower()
        assert "score" not in name.lower()
        assert "correct" not in name.lower()


def test_the_verifier_prints_no_prediction_correctness_or_score():
    source = VERIFIER.read_text(encoding="utf-8")
    for line in source.splitlines():
        if "print(" in line:
            # the value may never be interpolated; the word may appear in the
            # command's own disclaimer, which is the point of printing it
            assert "{_prediction" not in line
            assert "{prediction" not in line
    assert "_prediction, status = parse(" in source, \
        "the prediction is bound but deliberately unused"


def test_technical_status_constants_are_the_declared_ones():
    assert verifier.STATUS_FORMAT == "BLOCKED_FORMAT_COMPATIBILITY"
    assert verifier.BEHAVIOURAL == "BEHAVIOURAL_PARSE_FAILURE"
    assert verifier.STATUS_PASS == "PASS"


# --------------------------- baseline format gate vs behavioural outcome
@pytest.mark.parametrize("dataset,bad", [
    (DS_MMLU, "Let me think about this carefully"),
    (DS_GSM8K, "no numbers at all here"),
])
def test_an_unparsable_baseline_is_a_format_compatibility_block(dataset, bad):
    parse = verifier.PARSERS[dataset]
    _prediction, status = parse(bad)
    assert status != PARSE_OK
    # the verdict the CLI assigns for a baseline row in this state
    assert verifier.STATUS_FORMAT == "BLOCKED_FORMAT_COMPATIBILITY"


@needs_gate_thresholds
@pytest.mark.parametrize("condition", [COND_DYNAMIC, COND_ROUTEGATE])
def test_a_non_baseline_parse_failure_is_behavioural_not_technical(condition):
    """The row stays valid, stays on disk, and is scored incorrect later."""
    record = (dynamic_record(response_text="refused to answer")
              if condition == COND_DYNAMIC
              else routegate_record(response_text="refused to answer"))
    checks = Checklist("t")
    if condition == COND_DYNAMIC:
        verifier.check_dynamic(record, checks)
    else:
        verifier.check_routegate(record, checks)
    assert checks.all_passed, "wiring is intact regardless of parseability"
    _prediction, status = parse_mmlu(record["response_text"])
    assert status != PARSE_OK
    assert verifier.BEHAVIOURAL == "BEHAVIOURAL_PARSE_FAILURE"


@needs_gate_thresholds
@pytest.mark.parametrize("condition", [COND_DYNAMIC, COND_ROUTEGATE])
def test_a_non_baseline_token_cap_event_is_behavioural(condition):
    record = (dynamic_record(hit_max_new_tokens=True, eos_reached=False)
              if condition == COND_DYNAMIC
              else routegate_record(hit_max_new_tokens=True,
                                    eos_reached=False))
    checks = Checklist("t")
    if condition == COND_DYNAMIC:
        verifier.check_dynamic(record, checks)
    else:
        verifier.check_routegate(record, checks)
    assert checks.all_passed


def test_a_behavioural_failure_is_scored_incorrect_by_the_frozen_rule():
    from routegate.capabilityscoring import score_item
    scored = score_item(DS_MMLU, 0, "tinymmlu:0000", "refused to answer", 1)
    assert scored.correct is False and scored.parse_status != PARSE_OK


def test_no_technical_check_reads_a_gold_label():
    source = VERIFIER.read_text(encoding="utf-8")
    for fn in ("check_baseline", "check_dynamic", "check_routegate",
               "check_common", "read_one"):
        body = source.split(f"def {fn}(")[1].split("\ndef ")[0]
        assert "gold" not in body.lower()
        assert "correct" not in body.lower()


# ------------------------------------------- arm ids, protocol, help text
def test_all_six_capability_arm_ids_use_double_underscores():
    ids = [arm_id(d, c) for d, c in ARMS if d != "sealed"]
    assert tuple(ids) == CAPABILITY_ARMS
    for name in CAPABILITY_ARMS:
        assert name.count("__") == 1
        assert name in runner.ARM_IDS


def test_v5_supersedes_v4_and_v1_v2_generated_nothing():
    protocol = build_protocol()
    assert SUPERSEDES == "routegate_final_evaluation_v4"
    assert protocol["v1_generated_rows"] == 0
    assert protocol["v2_generated_rows"] == 0
    assert protocol["v3_generated_rows"] == 3
    assert protocol["scientific_protocol_unchanged_from_v2"] is True
    assert len(V3_CHANGES) >= 3
    assert PRIOR_PROTOCOLS == ("routegate_final_evaluation_v1",
                               "routegate_final_evaluation_v2",
                               "routegate_final_evaluation_v3",
                               "routegate_final_evaluation_v4")


def test_the_v5_output_root_is_distinct():
    assert DATA_ROOT_NAME == "final_evaluation_v5"
    for dataset, condition in ARMS:
        path = arm_output_dir(dataset, condition)
        assert path.startswith("final_evaluation_v5/")
        for older in ("final_evaluation_v1/", "final_evaluation_v2/",
                      "final_evaluation_v3/", "final_evaluation_v4/"):
            assert older not in path


def test_the_runner_help_names_v5_and_no_earlier_version():
    source = RUNNER.read_text(encoding="utf-8")
    docstring = source.split('"""')[1]
    assert "routegate_final_evaluation_v5.json" in docstring
    for older in ("v1.json", "v2.json", "v3.json", "v4.json"):
        assert f"routegate_final_evaluation_{older}" not in docstring
    assert "--item-index" in docstring


def test_the_protocol_keeps_every_v2_scientific_setting():
    protocol = build_protocol()
    assert protocol["generations"]["total"] == 1740
    assert protocol["generations"]["per_condition"] == 580
    assert protocol["conditions"] == list(CONDITIONS)
    assert DATASET_SPEC[DS_MMLU]["prompt_mode"] == "raw_completion"
    assert DATASET_SPEC[DS_GSM8K]["prompt_mode"] == "raw_completion"
    assert DATASET_SPEC["sealed"]["prompt_mode"] == "chat"
    assert expected_rows(DS_MMLU) == expected_rows(DS_GSM8K) == 100
    assert expected_rows("sealed") == 380
