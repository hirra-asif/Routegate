"""The eighteen defects protocol v1 shipped with, each made impossible.

v1 prepared a condition and discarded it, always generated with mode="off" and
no controller, had no answer parser, wrapped completion prompts as chat, and
dry-ran without ever reaching dispatch. Nothing had been generated.

The tests below fail if any of that comes back. The last one is the one that
matters most: a deterministic instrumented fake model records exactly what the
runner asked for, so hardcoding mode="off" or dropping the controller shows up
as a wrong recorded call rather than as identical text nobody compares.
"""
from __future__ import annotations

import importlib.util
import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from routegate.capabilityscoring import (
    GSM8K_HARNESS_REVISION,
    MMLU_GOLD_LETTERS,
    PARSE_AMBIGUOUS,
    PARSE_NO_ANSWER,
    PARSE_OK,
    PARSE_PROSE_BEFORE_ANSWER,
    CapabilityScoringError,
    assert_generation_cannot_read_gold,
    gsm8k_gold_value,
    mmlu_gold_letter,
    paired_transitions,
    parse_gsm8k,
    parse_mmlu,
    score_arm,
)
from routegate.finaleval import (
    DATASET_SPEC, DS_GSM8K, DS_MMLU, DS_SEALED, NAME as PROTOCOL_NAME,
    RECORD_SCHEMA, SUPERSEDES)
from routegate.finalrunner import (
    COND_BASELINE,
    COND_DYNAMIC,
    COND_ROUTEGATE,
    DYNAMIC_BLOCKS,
    DYNAMIC_HOOKS,
    DYNAMIC_READ_STATE,
    DYNAMIC_STATES,
    GATE_READ_STATE,
    NOOP_GATE_DID_NOT_FIRE,
    STATIC_BLOCKS,
    STATIC_HOOKS,
    STATIC_STATES,
    TELEMETRY_FIELDS,
    ConditionRuntime,
    DispatchError,
    assert_record_is_complete,
    blank_telemetry,
    execute_item,
)
from routegate.runtime import (
    PROMPT_MODE_CHAT, PROMPT_MODE_RAW, GenerationConfigError, render_prompt)

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "experiments/final_evaluation/run_final_evaluation.py"


# ============================================================== fake harness
class FakeCall:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeController:
    """Records what it was asked to do and how many hooks 'fired'."""

    def __init__(self, capture_vector, hits_per_plan=1):
        self._capture = capture_vector
        self._hits_per_plan = hits_per_plan
        self.run = FakeCall(hits={}, applied={})
        self.calls = []
        self.closed = False

    def capture_for(self, name):
        return dict(self._capture)

    def reset_run(self, **kw):
        self.calls.append(("reset_run", kw))
        self.run = FakeCall(hits={}, applied={})

    def close(self):
        self.closed = True

    def fire(self, plans):
        self.run.hits = {s: self._hits_per_plan for s in plans}


class FakeModel:
    pass


def install_fake_runtime(monkeypatch, controller, record):
    """Patch the two runtime entry points execute_item calls."""
    import routegate.runtime as rt

    def fake_capture(model, tokenizer, prompt, ctrl, capture_name,
                     example_id, config=None, prompt_mode=PROMPT_MODE_CHAT,
                     stop_strings=None):
        record.append(("capture", prompt_mode, example_id))
        return 7

    def fake_generate(model, tokenizer, prompt, config=None, controller=None,
                      mode="off", plans=None, example_id=None,
                      prompt_mode=PROMPT_MODE_CHAT, stop_strings=None):
        record.append(("generate", mode, controller is not None,
                       tuple(sorted(plans or {})), prompt_mode))
        if controller is not None and mode == "patch":
            controller.fire(plans or {})
        fired = sum(int(v) for v in getattr(
            getattr(controller, "run", None), "hits", {}).values()) \
            if controller is not None else 0
        return {"response_text": "X", "prompt_token_count": 3,
                "completion_token_count": 1, "eos_reached": True,
                "hit_max_new_tokens": False, "elapsed_seconds": 0.0,
                "prompt_mode": prompt_mode, "hooks_fired": fired}

    monkeypatch.setattr(rt, "capture_prompt_states", fake_capture)
    monkeypatch.setattr(rt, "generate_response", fake_generate)


class FakeSensorBundle:
    def __init__(self, fires, p_e=0.9, p_h=0.01):
        self._fires, self._p_e, self._p_h = fires, p_e, p_h

    def decide(self, activation):
        return FakeCall(p_E=self._p_e, p_H=self._p_h, fires=self._fires)


class FakeThresholds:
    threshold_E = 0.29185573339357973
    threshold_H = 0.06975384354575279
    gate_rule = "gate_fire = (p_E >= threshold_E) AND (p_H < threshold_H)"


class FakeActuator:
    def direction_for_state(self, state):
        v = np.zeros(8, dtype=np.float32)
        v[0] = 1.0
        return v


def dynamic_runtime():
    return ConditionRuntime(
        condition=COND_DYNAMIC, steers=True, hook_states=DYNAMIC_STATES,
        capture_state=DYNAMIC_READ_STATE, apply_blocks=DYNAMIC_BLOCKS,
        hooks_expected=DYNAMIC_HOOKS, controller_checkpoint={"fake": True},
        direction_bank=object())


def routegate_runtime(fires=True):
    return ConditionRuntime(
        condition=COND_ROUTEGATE, steers=True,
        hook_states=tuple(sorted(set(STATIC_STATES) | {GATE_READ_STATE})),
        capture_state=GATE_READ_STATE, apply_blocks=STATIC_BLOCKS,
        hooks_expected=STATIC_HOOKS, bundle=FakeSensorBundle(fires),
        thresholds=FakeThresholds(), actuator=FakeActuator())


# ---------------------------- 1. the prepared condition reaches generation
def test_execute_item_requires_the_prepared_condition():
    with pytest.raises(DispatchError, match="requires the prepared"):
        execute_item(None, model=None, tokenizer=None, controller=None,
                     prompt="p", item_id="i", generation=None,
                     prompt_mode=PROMPT_MODE_RAW)


def test_the_runner_passes_prepared_into_execute_item():
    source = RUNNER.read_text(encoding="utf-8")
    assert "execute_item(\n                    prepared," in source \
        or "execute_item(prepared," in source
    assert "mode=\"off\"" not in source, "the runner must not hardcode off"


# ------------------------- 2. the three conditions take different paths
def test_baseline_installs_no_controller_and_runs_off(monkeypatch):
    seen = []
    install_fake_runtime(monkeypatch, None, seen)
    runtime = ConditionRuntime(condition=COND_BASELINE, steers=False)
    out = execute_item(runtime, model=FakeModel(), tokenizer=None,
                       controller=None, prompt="p", item_id="i",
                       generation=None, prompt_mode=PROMPT_MODE_RAW)
    assert seen == [("generate", "off", False, (), PROMPT_MODE_RAW)]
    assert out["hooks_expected"] == 0 and out["hooks_fired"] == 0
    assert out["mode"] == "off"


def test_baseline_refuses_a_controller(monkeypatch):
    install_fake_runtime(monkeypatch, None, [])
    runtime = ConditionRuntime(condition=COND_BASELINE, steers=False)
    with pytest.raises(DispatchError, match="installs no controller"):
        execute_item(runtime, model=FakeModel(), tokenizer=None,
                     controller=FakeController({}), prompt="p", item_id="i",
                     generation=None, prompt_mode=PROMPT_MODE_RAW)


def test_a_steered_condition_without_a_controller_is_refused(monkeypatch):
    install_fake_runtime(monkeypatch, None, [])
    with pytest.raises(DispatchError, match="unsteered generation is not"):
        execute_item(dynamic_runtime(), model=FakeModel(), tokenizer=None,
                     controller=None, prompt="p", item_id="i",
                     generation=None, prompt_mode=PROMPT_MODE_RAW)


# --------------- 3. a forced nonzero dynamic coefficient installs hooks
def test_dynamic_raw_installs_both_hooks(monkeypatch):
    seen = []
    ctrl = FakeController({DYNAMIC_READ_STATE: np.ones(8)})
    install_fake_runtime(monkeypatch, ctrl, seen)
    monkeypatch.setattr("routegate.livecontroller.gate_from_activation",
                        lambda a, c: np.array([0.5]))
    monkeypatch.setattr("routegate.twosite.build_two_site_plans",
                        lambda *a, **k: {32: object(), 33: object()})
    out = execute_item(dynamic_runtime(), model=FakeModel(), tokenizer=None,
                       controller=ctrl, prompt="p", item_id="i",
                       generation=None, prompt_mode=PROMPT_MODE_RAW)
    call = [c for c in seen if c[0] == "generate"][0]
    assert call[1] == "patch" and call[2] is True
    assert out["g_R"] == 0.5
    assert out["applied_alpha_R"] == pytest.approx(0.075)
    assert out["hooks_expected"] == DYNAMIC_HOOKS == 2
    assert out["hooks_fired"] == 2
    assert out["controller_read_state"] == DYNAMIC_READ_STATE


def test_dynamic_raw_with_a_zero_coefficient_fails_closed(monkeypatch):
    ctrl = FakeController({DYNAMIC_READ_STATE: np.ones(8)})
    install_fake_runtime(monkeypatch, ctrl, [])
    monkeypatch.setattr("routegate.livecontroller.gate_from_activation",
                        lambda a, c: np.array([0.0]))
    with pytest.raises(DispatchError, match="no coefficient"):
        execute_item(dynamic_runtime(), model=FakeModel(), tokenizer=None,
                     controller=ctrl, prompt="p", item_id="i",
                     generation=None, prompt_mode=PROMPT_MODE_RAW)


# ----------------- 4 & 5. gate fires -> actuator; gate no-op -> declared
def test_a_forced_fire_gate_installs_three_actuator_hooks(monkeypatch):
    seen = []
    ctrl = FakeController({GATE_READ_STATE: np.ones(8), 31: np.ones(8),
                           32: np.ones(8), 33: np.ones(8)})
    install_fake_runtime(monkeypatch, ctrl, seen)
    monkeypatch.setattr("routegate.steering.build_steering_plan",
                        lambda **k: object())
    out = execute_item(routegate_runtime(fires=True), model=FakeModel(),
                       tokenizer=None, controller=ctrl, prompt="p",
                       item_id="i", generation=None,
                       prompt_mode=PROMPT_MODE_RAW)
    call = [c for c in seen if c[0] == "generate"][0]
    assert call[1] == "patch"
    assert call[3] == tuple(b + 1 for b in STATIC_BLOCKS) == (32, 33, 34)
    assert out["gate_fires"] is True
    assert out["hooks_expected"] == STATIC_HOOKS == 3
    assert out["hooks_fired"] == 3
    assert out["noop_reason"] is None


def test_a_gate_no_op_is_declared_and_not_missing_wiring(monkeypatch):
    seen = []
    ctrl = FakeController({GATE_READ_STATE: np.ones(8)})
    install_fake_runtime(monkeypatch, ctrl, seen)
    out = execute_item(routegate_runtime(fires=False), model=FakeModel(),
                       tokenizer=None, controller=ctrl, prompt="p",
                       item_id="i", generation=None,
                       prompt_mode=PROMPT_MODE_RAW)
    assert out["gate_fires"] is False
    assert out["noop_reason"] == NOOP_GATE_DID_NOT_FIRE
    assert out["hooks_expected"] == 0 and out["hooks_fired"] == 0
    # the gate was still EVALUATED: its probabilities are on the record
    assert out["gate_p_E"] is not None and out["gate_p_H"] is not None
    assert out["threshold_E"] is not None
    assert_record_is_complete({**out, "condition": COND_ROUTEGATE})


# --------------------------------- 6. partial or missing hooks fail closed
def _record(condition, **kw):
    base = {**blank_telemetry(), "condition": condition, "mode": "patch",
            "noop_reason": None}
    base.update(kw)
    return base


def test_hook_disagreement_fails_closed():
    with pytest.raises(DispatchError, match="disagree"):
        assert_record_is_complete(_record(
            COND_DYNAMIC, hooks_expected=2, hooks_fired=1,
            controller_read_state=32, g_R=0.5, applied_alpha_R=0.075))


def test_a_gate_fired_row_with_no_hooks_fails_closed():
    with pytest.raises(DispatchError, match="disagree|missing wiring"):
        assert_record_is_complete(_record(
            COND_ROUTEGATE, hooks_expected=0, hooks_fired=0, gate_fires=True,
            gate_p_E=0.9, gate_p_H=0.01, gate_read_state=45,
            threshold_E=0.29, threshold_H=0.07))


def test_dynamic_raw_may_never_run_with_mode_off():
    with pytest.raises(DispatchError, match="never generate with mode 'off'"):
        assert_record_is_complete(_record(
            COND_DYNAMIC, mode="off", hooks_expected=0, hooks_fired=0,
            controller_read_state=32, g_R=0.5, applied_alpha_R=0.075))


def test_a_missing_telemetry_field_fails_closed():
    record = _record(COND_BASELINE, mode="off", hooks_expected=0,
                     hooks_fired=0)
    del record["g_R"]
    with pytest.raises(DispatchError, match="carries no g_R"):
        assert_record_is_complete(record)


# ------------------------------------------- 7. the twelve telemetry fields
def test_every_record_carries_all_twelve_telemetry_fields(monkeypatch):
    assert len(TELEMETRY_FIELDS) == 12
    seen = []
    install_fake_runtime(monkeypatch, None, seen)
    out = execute_item(ConditionRuntime(condition=COND_BASELINE, steers=False),
                       model=FakeModel(), tokenizer=None, controller=None,
                       prompt="p", item_id="i", generation=None,
                       prompt_mode=PROMPT_MODE_RAW)
    for field in TELEMETRY_FIELDS:
        assert field in out
    assert_record_is_complete({**out, "condition": COND_BASELINE})


# ------------------------------------------------- 8, 9, 10. prompt modes
def test_capability_prompts_bypass_the_chat_template():
    class Tok:
        def apply_chat_template(self, *a, **k):
            raise AssertionError("the chat template must not be applied")
    assert render_prompt(Tok(), "raw text", None, PROMPT_MODE_RAW) == "raw text"


def test_sealed_prompts_keep_the_chat_template():
    class Tok:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt,
                                enable_thinking):
            return "<|im_start|>user\n" + messages[0]["content"]
    out = render_prompt(Tok(), "hello", None, PROMPT_MODE_CHAT)
    assert out.startswith("<|im_start|>user")


def test_an_unknown_prompt_mode_is_refused():
    with pytest.raises(GenerationConfigError, match="unknown prompt mode"):
        render_prompt(None, "x", None, "something_else")


def test_the_protocol_pins_raw_for_capability_and_chat_for_sealed():
    assert DATASET_SPEC[DS_MMLU]["prompt_mode"] == PROMPT_MODE_RAW
    assert DATASET_SPEC[DS_GSM8K]["prompt_mode"] == PROMPT_MODE_RAW
    assert DATASET_SPEC[DS_SEALED]["prompt_mode"] == PROMPT_MODE_CHAT


def test_all_three_conditions_receive_identical_prompt_tokens(monkeypatch):
    """Prompt rendering depends on the dataset, never on the condition."""
    seen = []
    prompts = []

    import routegate.runtime as rt

    def fake_generate(model, tokenizer, prompt, config=None, controller=None,
                      mode="off", plans=None, example_id=None,
                      prompt_mode=PROMPT_MODE_CHAT, stop_strings=None):
        prompts.append((prompt, prompt_mode))
        if controller is not None and mode == "patch":
            controller.fire(plans or {})
        fired = sum(int(v) for v in getattr(
            getattr(controller, "run", None), "hits", {}).values()) \
            if controller is not None else 0
        return {"response_text": "X", "hooks_fired": fired}

    def fake_capture(model, tokenizer, prompt, ctrl, capture_name, example_id,
                     config=None, prompt_mode=PROMPT_MODE_CHAT,
                     stop_strings=None):
        prompts.append((prompt, prompt_mode))
        return 3

    monkeypatch.setattr(rt, "generate_response", fake_generate)
    monkeypatch.setattr(rt, "capture_prompt_states", fake_capture)
    monkeypatch.setattr("routegate.livecontroller.gate_from_activation",
                        lambda a, c: np.array([0.5]))
    monkeypatch.setattr("routegate.twosite.build_two_site_plans",
                        lambda *a, **k: {32: object(), 33: object()})
    monkeypatch.setattr("routegate.steering.build_steering_plan",
                        lambda **k: object())
    text = "Question ...\nAnswer:"
    cases = [
        (ConditionRuntime(condition=COND_BASELINE, steers=False), None),
        (dynamic_runtime(), FakeController({DYNAMIC_READ_STATE: np.ones(8)})),
        (routegate_runtime(True),
         FakeController({GATE_READ_STATE: np.ones(8), 31: np.ones(8),
                         32: np.ones(8), 33: np.ones(8)})),
    ]
    for runtime, ctrl in cases:
        execute_item(runtime, model=FakeModel(), tokenizer=None,
                     controller=ctrl, prompt=text, item_id="i",
                     generation=None, prompt_mode=PROMPT_MODE_RAW)
    assert {p for p, _ in prompts} == {text}
    assert {m for _, m in prompts} == {PROMPT_MODE_RAW}


# --------------------------------------- 11, 12. MMLU gold and parsing
def test_mmlu_gold_integers_map_to_letters():
    assert MMLU_GOLD_LETTERS == {0: "A", 1: "B", 2: "C", 3: "D"}
    assert [mmlu_gold_letter(i) for i in range(4)] == ["A", "B", "C", "D"]
    for bad in (-1, 4, "A", None):
        with pytest.raises(CapabilityScoringError):
            mmlu_gold_letter(bad)


@pytest.mark.parametrize("text,expected", [
    ("A", "A"), ("B.", "B"), ("  C  ", "C"), ("D)", "D"), ("b.", "B"),
    ("A\n\nQuestion: the next few-shot item", "A"),   # continuation is not the answer
])
def test_valid_mmlu_completions_parse(text, expected):
    letter, status = parse_mmlu(text)
    assert (letter, status) == (expected, PARSE_OK)


@pytest.mark.parametrize("text,status", [
    ("", PARSE_NO_ANSWER), ("   ", PARSE_NO_ANSWER),
    ("The answer is B", PARSE_PROSE_BEFORE_ANSWER),
    ("Let me think about this.", PARSE_NO_ANSWER),
    ("Bravo", PARSE_PROSE_BEFORE_ANSWER),
    ("B or C", PARSE_AMBIGUOUS), ("A, but possibly D", PARSE_AMBIGUOUS),
    ("C is correct because", PARSE_AMBIGUOUS),
])
def test_invalid_mmlu_completions_fail_closed(text, status):
    letter, got = parse_mmlu(text)
    assert letter is None and got == status


# ------------------------------------------------------ 13. GSM8K parsing
@pytest.mark.parametrize("text,expected", [
    ("42", Decimal("42")), ("The answer is 42.", Decimal("42")),
    ("#### 1,234", Decimal("1234")), ("-17", Decimal("-17")),
    ("3.5", Decimal("3.5")), ("$1,200.", Decimal("1200")),
    ("first 5 then 12", Decimal("12")), ("0.30", Decimal("0.30")),
])
def test_gsm8k_parses_commas_signs_integers_and_decimals(text, expected):
    value, status = parse_gsm8k(text)
    assert status == PARSE_OK and value == expected


@pytest.mark.parametrize("text", ["", "no numbers here", None])
def test_gsm8k_fails_closed_without_a_number(text):
    value, status = parse_gsm8k(text)
    assert value is None and status == PARSE_NO_ANSWER


def test_gsm8k_compares_by_exact_decimal_not_binary_float():
    assert parse_gsm8k("42")[0] == parse_gsm8k("42.0")[0]
    assert parse_gsm8k("42,000")[0] == parse_gsm8k("42000")[0]
    assert parse_gsm8k("0.1")[0] + parse_gsm8k("0.2")[0] == Decimal("0.3")
    assert 0.1 + 0.2 != 0.3          # the binary float this avoids


def test_gsm8k_gold_uses_the_final_number_after_the_separator():
    assert gsm8k_gold_value("work\n#### 1,234") == Decimal("1234")
    with pytest.raises(CapabilityScoringError):
        gsm8k_gold_value("no separator here")


def test_the_gsm8k_rule_pins_its_harness_revision():
    assert GSM8K_HARNESS_REVISION == "v0.4.3"
    assert DATASET_SPEC[DS_GSM8K]["max_new_tokens"] == 512
    assert DATASET_SPEC[DS_MMLU]["max_new_tokens"] == 8


# ------------------------------------- 14. generation cannot read the gold
def test_the_generating_runner_reads_no_gold_field():
    assert_generation_cannot_read_gold(RUNNER.read_text(encoding="utf-8"),
                                       RUNNER.name)


def test_the_hygiene_check_actually_catches_a_gold_read():
    with pytest.raises(CapabilityScoringError):
        assert_generation_cannot_read_gold('row["answer"]', "fake")


# ------------------------------ 15, 16. resume isolation and ordering
def test_v5_output_roots_cannot_collide_with_any_predecessor():
    from routegate.finaleval import DATA_ROOT_NAME, arm_output_dir
    assert DATA_ROOT_NAME == "final_evaluation_v5"
    assert SUPERSEDES == "routegate_final_evaluation_v4"
    for path in (arm_output_dir(d, c) for d, c in
                 [(DS_MMLU, COND_BASELINE), (DS_SEALED, COND_DYNAMIC)]):
        assert path.startswith("final_evaluation_v5/")
        for older in ("final_evaluation_v1/", "final_evaluation_v2/",
                      "final_evaluation_v3/", "final_evaluation_v4/"):
            assert older not in path


def test_an_earlier_record_cannot_be_resumed_into_a_v5_arm():
    assert RECORD_SCHEMA.endswith("/5")
    for older in ("routegate/final_evaluation_record/1",
                  "routegate/final_evaluation_record/2",
                  "routegate/final_evaluation_record/3",
                  "routegate/final_evaluation_record/4"):
        assert older != RECORD_SCHEMA


def test_scoring_preserves_dataset_order_and_the_100_vector():
    golds = [i % 4 for i in range(100)]
    records = [{"item_index": i, "item_id": f"tinymmlu:{i:04d}",
                "response_text": MMLU_GOLD_LETTERS[golds[i]]}
               for i in reversed(range(100))]          # deliberately shuffled
    reading = score_arm(DS_MMLU, records, golds)
    assert reading["denominator"] == 100
    assert len(reading["correctness_vector"]) == 100
    assert reading["correct"] == 100
    assert [it["item_index"] for it in reading["items"]] == list(range(100))


def test_an_incomplete_arm_is_not_scored():
    golds = [0] * 100
    with pytest.raises(CapabilityScoringError, match="absent"):
        score_arm(DS_MMLU, [{"item_index": 0, "item_id": "x",
                             "response_text": "A"}], golds)


def test_paired_transitions_are_item_level():
    base = [1, 1, 0, 0]
    cand = [1, 0, 1, 0]
    paired = paired_transitions(base, cand)
    assert paired["correct_to_correct"] == 1
    assert paired["correct_to_incorrect"] == 1
    assert paired["incorrect_to_correct"] == 1
    assert paired["incorrect_to_incorrect"] == 1
    assert paired["accuracy_delta"] == 0.0
    assert paired["delta_confidence_interval"]["discordant"] == 2


# --------------------------- 17, 18. dry run reaches dispatch; integration
def test_the_dry_run_validates_real_condition_dispatch():
    source = RUNNER.read_text(encoding="utf-8")
    assert "prepared_condition_is_used_not_discarded" in source
    assert "steered_conditions_declare_hooks" in source
    assert "capture_state_is_hooked" in source


def test_an_instrumented_fake_model_would_catch_the_v1_defect(monkeypatch):
    """The regression test for the exact bug: mode off / controller dropped."""
    seen = []
    ctrl = FakeController({DYNAMIC_READ_STATE: np.ones(8)})
    install_fake_runtime(monkeypatch, ctrl, seen)
    monkeypatch.setattr("routegate.livecontroller.gate_from_activation",
                        lambda a, c: np.array([0.5]))
    monkeypatch.setattr("routegate.twosite.build_two_site_plans",
                        lambda *a, **k: {32: object(), 33: object()})
    out = execute_item(dynamic_runtime(), model=FakeModel(), tokenizer=None,
                       controller=ctrl, prompt="p", item_id="i",
                       generation=None, prompt_mode=PROMPT_MODE_RAW)
    generate_calls = [c for c in seen if c[0] == "generate"]
    assert len(generate_calls) == 1
    _tag, mode, had_controller, plan_states, _pm = generate_calls[0]
    # v1 would have recorded ("generate", "off", False, ()) here.
    assert mode == "patch", "v1 hardcoded mode='off'"
    assert had_controller is True, "v1 discarded the controller"
    assert plan_states == (32, 33)
    assert out["hooks_fired"] == 2
    assert_record_is_complete({**out, "condition": COND_DYNAMIC})
