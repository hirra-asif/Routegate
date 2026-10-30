"""The GSM8K generation boundary, and the v4 failure it repairs.

v4 passed no stop sequence. Its three GSM8K completions therefore ran to the
512-token cap, answered the question, and carried on into further few-shot
items -- 3, 4 and 3 strict ``####`` markers respectively. The frozen
last-number rule then reached into a later continuation and selected an answer
belonging to a different question, on every row, silently, with
``parse_status=ok``.

v5 stops generation at the earliest upstream stop sequence in the NEWLY
generated text and removes it and everything after it before the completion is
stored. The parser is unchanged: with a bounded completion the last number IS
the current answer.

The upstream boundary is not remembered here. references/lm_eval_v0.4.3_gsm8k.yaml
is vendored byte-for-byte and pinned by sha256, and these tests read it.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

from routegate.capabilityparsers import PARSE_OK, parse_gsm8k, parse_mmlu
from routegate.finaleval import (
    ARMS, DATASET_SPEC, DATA_ROOT_NAME, DS_GSM8K, DS_MMLU, DS_SEALED,
    GSM8K_BOUNDARY_CORRECTION, GSM8K_REFERENCE, GSM8K_STOP_STRINGS,
    RECORD_SCHEMA, SUPERSEDES, V4_GENERATED_DETAIL, V4_GENERATED_ROWS,
    arm_id, arm_output_dir, build_protocol)
from routegate.runtime import (
    STOP_REASON_EOS, STOP_REASON_MAX_NEW_TOKENS, STOP_REASON_STOP_STRING,
    first_stop)
from routegate.validation import Checklist, ValidationError, sha256_file

REPO = Path(__file__).resolve().parents[1]
REFERENCE = REPO / "references/lm_eval_v0.4.3_gsm8k.yaml"
RUNNER = REPO / "experiments/final_evaluation/run_final_evaluation.py"
VERIFIER = REPO / "experiments/final_evaluation/verify_technical_smoke.py"
STRICT = re.compile(r"#### (\-?[0-9\.\,]+)")

CAPABILITY_ARMS = (
    "tinymmlu__baseline", "tinymmlu__dynamic_raw",
    "tinymmlu__safety_first_routegate", "tinygsm8k__baseline",
    "tinygsm8k__dynamic_raw", "tinygsm8k__safety_first_routegate")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verifier = load(VERIFIER, "_verifier_v5")
runner = load(RUNNER, "_runner_v5")

# The exact shape v4 produced: the current answer, then later few-shot items.
UNBOUNDED = ("We compute 6 * 7 = 42\n#### 42\n\n"
             "Question: a different question entirely\n"
             "Answer: we compute 3 + 4 = 7\n#### 7\n\n"
             "Question: a third question\nAnswer: 9 * 9 = 81\n#### 81")
BOUNDED = "We compute 6 * 7 = 42\n#### 42"


# ============================================ the vendored upstream reference
def test_the_upstream_task_file_is_vendored_and_pinned():
    assert REFERENCE.is_file()
    provenance = json.loads(
        (REPO / "references/lm_eval_v0.4.3_gsm8k.provenance.json").read_text())
    assert provenance["tag"] == "v0.4.3"
    assert provenance["repo"] == "EleutherAI/lm-evaluation-harness"
    assert provenance["path_in_repo"] == "lm_eval/tasks/gsm8k/gsm8k.yaml"
    assert sha256_file(REFERENCE) == provenance["sha256"]
    assert GSM8K_REFERENCE["sha256"] == provenance["sha256"]


def test_the_frozen_stop_strings_are_the_upstream_ones():
    import yaml
    spec = yaml.safe_load(REFERENCE.read_text(encoding="utf-8"))
    assert list(GSM8K_STOP_STRINGS) == spec["generation_kwargs"]["until"]
    assert DATASET_SPEC[DS_GSM8K]["stop_strings"] == list(GSM8K_STOP_STRINGS)


def test_the_frozen_parser_regex_is_the_upstream_flexible_extract():
    import yaml
    from routegate.capabilityparsers import GSM8K_NUMBER_REGEX
    spec = yaml.safe_load(REFERENCE.read_text(encoding="utf-8"))
    flexible = next(f for f in spec["filter_list"]
                    if f["name"] == "flexible-extract")
    assert flexible["filter"][0]["regex_pattern"] == GSM8K_NUMBER_REGEX
    assert flexible["filter"][0]["group_select"] == -1   # last match
    strict = next(f for f in spec["filter_list"]
                  if f["name"] == "strict-match")
    assert strict["filter"][0]["regex_pattern"] == STRICT.pattern


# ==================================================== the stopping mechanics
@pytest.mark.parametrize("stop", ["Question:", "</s>", "<|im_end|>"])
def test_each_stop_sequence_terminates_the_completion(stop):
    text = f"answer\n#### 42{stop} trailing material"
    kept, fired = first_stop(text, GSM8K_STOP_STRINGS)
    assert fired == stop
    assert stop not in kept and "trailing" not in kept


def test_the_earliest_stop_wins():
    text = "a <|im_end|> b </s> c Question: d"
    kept, fired = first_stop(text, GSM8K_STOP_STRINGS)
    assert fired == "<|im_end|>" and kept == "a "


def test_a_stop_present_only_in_the_prompt_does_not_terminate():
    """The criterion decodes ONLY the newly generated tokens."""
    source = (REPO / "src/routegate/runtime.py").read_text(encoding="utf-8")
    assert "input_ids[0, self.prompt_length:]" in source
    # and the completion-side cut only ever sees generated text
    kept, fired = first_stop("plain answer 42", GSM8K_STOP_STRINGS)
    assert fired is None and kept == "plain answer 42"


def test_a_multi_token_stop_sequence_is_matched_after_decoding():
    """Split across tokens is still one substring once decoded."""
    text = "answer 42<|im_end|>"
    kept, fired = first_stop(text, GSM8K_STOP_STRINGS)
    assert fired == "<|im_end|>" and kept == "answer 42"


def test_the_stop_and_everything_after_it_is_removed():
    kept, fired = first_stop(UNBOUNDED, GSM8K_STOP_STRINGS)
    assert fired == "Question:"
    assert "Question:" not in kept
    assert len(STRICT.findall(kept)) == 1


def test_no_stop_list_leaves_the_text_untouched():
    assert first_stop(UNBOUNDED, None) == (UNBOUNDED, None)
    assert first_stop(UNBOUNDED, []) == (UNBOUNDED, None)


# ============================== the v4 failure reproduced, and v5's repair
def test_the_v4_unbounded_failure_reproduces():
    """Last-match on an unbounded completion selects a later question's answer."""
    unbounded_value, status = parse_gsm8k(UNBOUNDED)
    assert status == PARSE_OK
    bounded_value, bounded_status = parse_gsm8k(BOUNDED)
    assert bounded_status == PARSE_OK
    # the defect: the two disagree
    assert unbounded_value != bounded_value
    assert len(STRICT.findall(UNBOUNDED)) == 3


def test_v5_bounding_restores_the_current_question_answer():
    kept, fired = first_stop(UNBOUNDED, GSM8K_STOP_STRINGS)
    assert fired == "Question:"
    v5_value, status = parse_gsm8k(kept)
    bounded_value, _ = parse_gsm8k(BOUNDED)
    assert status == PARSE_OK
    assert v5_value == bounded_value          # the current answer, not a later one
    assert len(STRICT.findall(kept)) == 1


def test_the_parser_itself_was_not_changed():
    assert GSM8K_BOUNDARY_CORRECTION["parser_unchanged"] is True
    assert GSM8K_BOUNDARY_CORRECTION["scoring_definition_unchanged"] is True
    assert GSM8K_BOUNDARY_CORRECTION["applies_to"] == "tinygsm8k only"


# ------------------------------------------------------- MMLU is untouched
def test_mmlu_has_no_stop_rule_and_keeps_its_answer_line_behaviour():
    assert DATASET_SPEC[DS_MMLU]["stop_strings"] == []
    assert DATASET_SPEC[DS_SEALED]["stop_strings"] == []
    assert GSM8K_BOUNDARY_CORRECTION["mmlu_unchanged"] is True
    trailing = "B\n\nQuestion: the next few-shot item\nAnswer: C"
    assert parse_mmlu(trailing) == parse_mmlu("B")


def test_max_new_tokens_and_decoding_are_unchanged():
    assert DATASET_SPEC[DS_MMLU]["max_new_tokens"] == 8
    assert DATASET_SPEC[DS_GSM8K]["max_new_tokens"] == 512
    for spec in DATASET_SPEC.values():
        assert spec["decoding"] == "greedy"


# ====================================================== the verifier checks
def _gsm_record(text=BOUNDED, **kw):
    base = {"response_text": text, "stop_strings": list(GSM8K_STOP_STRINGS),
            "stop_fired": True, "stop_reason": STOP_REASON_STOP_STRING,
            "hit_max_new_tokens": False}
    base.update(kw)
    return base


def test_a_bounded_gsm8k_row_passes_the_boundary_check():
    checks = Checklist("t")
    verifier.check_gsm8k_boundary(_gsm_record(), checks)
    assert checks.all_passed


def test_an_unbounded_gsm8k_row_is_technically_blocked():
    with pytest.raises(ValidationError, match="stop_sequence_survives|later_question"):
        verifier.check_gsm8k_boundary(
            _gsm_record(UNBOUNDED, stop_fired=False,
                        stop_reason=STOP_REASON_MAX_NEW_TOKENS,
                        hit_max_new_tokens=True), Checklist("t"))


def test_more_than_one_answer_marker_is_blocked():
    with pytest.raises(ValidationError, match="exactly_one_current_question"):
        verifier.check_gsm8k_boundary(
            _gsm_record("a\n#### 1\nb\n#### 2"), Checklist("t"))


def test_inconsistent_stop_telemetry_is_blocked():
    with pytest.raises(ValidationError, match="stop_telemetry_is_consistent"):
        verifier.check_gsm8k_boundary(
            _gsm_record(stop_fired=True, stop_reason=STOP_REASON_EOS),
            Checklist("t"))
    with pytest.raises(ValidationError, match="stop_telemetry_is_consistent"):
        verifier.check_gsm8k_boundary(
            _gsm_record(stop_fired=True, hit_max_new_tokens=True),
            Checklist("t"))


def test_missing_declared_stop_strings_are_blocked():
    with pytest.raises(ValidationError, match="stop_strings_declared"):
        verifier.check_gsm8k_boundary(_gsm_record(stop_strings=[]),
                                      Checklist("t"))


def test_the_verifier_still_prints_no_value_or_gold():
    source = VERIFIER.read_text(encoding="utf-8")
    imports = "\n".join(l for l in source.splitlines()
                        if l.strip().startswith(("import ", "from ")))
    assert "capabilityscoring" not in imports
    for banned in ("mmlu_gold_letter", "gsm8k_gold_value", "score_arm",
                   "answer_column"):
        assert banned not in source
    for line in source.splitlines():
        if "print(" in line:
            assert "{_prediction" not in line and "{prediction" not in line
    body = source.split("def check_gsm8k_boundary(")[1].split("\ndef ")[0]
    assert "print(" not in body
    assert "gold" not in body.lower()


# ============================== dispatch, resume and selector are unchanged
def test_the_three_conditions_still_use_their_exact_controller_paths():
    source = (REPO / "src/routegate/finalrunner.py").read_text(encoding="utf-8")
    body = source.split('"""', 2)[2]        # skip the module docstring
    assert 'controller=None, mode="off"' in body            # baseline
    assert body.count('mode="patch"') == 2                  # dynamic + gate
    assert body.count('mode="off"') == 2                    # baseline + no-op
    assert body.count("stop_strings=stop_strings") == 4     # every call site
    runner_src = RUNNER.read_text(encoding="utf-8")
    assert runner_src.count("execute_item(") == 1


def test_the_selector_and_resume_are_unchanged():
    items = [{"item_index": i, "item_id": f"tinygsm8k:{i:04d}",
              "prompt": f"p{i}"} for i in range(100)]
    assert len(runner.select_items(items, 0, 100, Checklist("t"))) == 1
    assert len(runner.select_items(items, None, 100, Checklist("t"))) == 100
    with pytest.raises(ValidationError):
        runner.select_items(items, 100, 100, Checklist("t"))


def test_a_v5_partial_smoke_cannot_claim_completion(tmp_path):
    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps({"schema": RECORD_SCHEMA, "item_index": 0,
                                   "item_id": "tinygsm8k:0000"}) + "\n")
    assert runner.finalise_if_complete(tmp_path, records, 100,
                                       Checklist("t")) is False
    assert not (tmp_path / "arm_complete.json").exists()


def test_all_six_double_underscore_arm_ids_remain_supported():
    ids = tuple(arm_id(d, c) for d, c in ARMS if d != DS_SEALED)
    assert ids == CAPABILITY_ARMS
    for name in CAPABILITY_ARMS:
        assert name in runner.ARM_IDS and name.count("__") == 1


# ================================================ v4 preserved and excluded
def test_v5_supersedes_v4_and_records_its_census():
    protocol = build_protocol()
    assert SUPERSEDES == "routegate_final_evaluation_v4"
    assert protocol["v4_generated_rows"] == V4_GENERATED_ROWS == 6
    assert protocol["v4_sealed_rows"] == 0
    detail = protocol["v4_generated_detail"]
    assert detail["tinymmlu_rows"] == 3 and detail["tinygsm8k_rows"] == 3
    assert detail["sealed_rows"] == 0 and detail["item_index"] == 0
    assert detail["preserved"] is True and detail["excluded_from_v5"] is True


def test_the_v5_root_cannot_collide_with_any_predecessor():
    assert DATA_ROOT_NAME == "final_evaluation_v5"
    assert RECORD_SCHEMA.endswith("/5")
    for dataset, condition in ARMS:
        path = arm_output_dir(dataset, condition)
        assert path.startswith("final_evaluation_v5/")
        for older in ("_v1/", "_v2/", "_v3/", "_v4/"):
            assert f"final_evaluation{older}" not in path


def test_the_v4_rows_are_still_on_disk_untouched():
    root = Path.home() / "routegate-data" / "final_evaluation_v4"
    if not root.exists():
        pytest.skip("the v4 smoke rows are not present on this machine")
    seen = 0
    for name in CAPABILITY_ARMS:
        path = root / name / "records.jsonl"
        assert path.is_file(), name
        rows = [l for l in path.read_text(encoding="utf-8").splitlines()
                if l.strip()]
        assert len(rows) == 1
        seen += 1
    assert seen == 6
    assert not list(root.glob("sealed__*/records.jsonl"))


def test_no_sealed_material_is_reachable():
    from routegate.validation import looks_sealed
    protocol = build_protocol()
    assert protocol["sealed_set"]["path_stored_in_this_repository"] is False
    assert not looks_sealed(str(GSM8K_REFERENCE["file"]))
    for spec in DATASET_SPEC.values():
        assert not looks_sealed(str(spec.get("local", "")))
