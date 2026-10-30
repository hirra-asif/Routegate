"""The one-direction controller's specification: one gate, one ceiling.

Nothing here trains, initialises or evaluates a controller -- there is none to
train. What is under test is the specification and the gates around it: how
many outputs it may have, how large the applied strength can get, where it
reads, what the safety head is allowed to do to it, and which configurations
the final path refuses because phase 2 ruled them out.

Organised by the failure each group prevents:

    1. outputs     a second coefficient returns under a new name
    2. ceiling     the applied strength climbs past the validated 0.15
    3. position    the read moves off state 32 or off the final prompt token
    4. brake       the safety reading starts steering instead of braking
    5. protocol    the out-of-fold or deployment order is quietly relaxed
    6. refusal     a ruled-out configuration reaches the final path
    7. committed   the file on disk and the package part company
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate import singledirection as S                          # noqa: E402
from routegate.phase2closure import PHASE2_OUTCOME                  # noqa: E402
from routegate.probeheads import (                                  # noqa: E402
    PRIMARY_READ_STATE,
    SENSITIVITY_STATES,
    layer_of_state,
    state_of_layer,
)
from routegate.validation import ValidationError                    # noqa: E402

DUAL_CONFIG = REPO_ROOT / "configs" / "dual_direction_controller.json"


def _minimal(**extra):
    """The smallest configuration the final path accepts, plus a mutation."""
    config = {"controller": {"outputs": ["g_R"], "max_applied_alpha_R": 0.15}}
    config.update(extra)
    return config


# =============================================================== 1. outputs
def test_the_controller_emits_exactly_one_output():
    assert S.CONTROLLER_OUTPUTS == ("g_R",)
    assert len(S.CONTROLLER_OUTPUTS) == 1
    specification = S.build_single_direction_specification()
    assert specification["controller"]["outputs"] == ["g_R"]
    assert specification["controller"]["output_count"] == 1
    assert specification["controller"]["steered_direction_count"] == 1


def test_no_safety_coefficient_is_emitted():
    specification = S.build_single_direction_specification()
    assert specification["controller"]["emits_a_safety_coefficient"] is False
    assert specification["retired_actuator"]["active"] is False
    assert specification["retired_actuator"]["retired_by"] == PHASE2_OUTCOME


@pytest.mark.parametrize("outputs", [["g_R", "g_S"], ["g_R", "a_S"], [],
                                     ["a_S"], ["g_R", "g_R"]])
def test_a_second_controller_output_is_refused(outputs):
    with pytest.raises(S.SingleDirectionError, match="controller outputs"):
        S.assert_final_controller_config(
            {"controller": {"outputs": outputs, "max_applied_alpha_R": 0.15}})


def test_the_retired_direction_is_not_an_active_actuator_anywhere():
    specification = S.build_single_direction_specification()
    assert specification["controller"]["steered_direction"] == "d_R = B_d1 - E"
    assert S.DIRECTION_SAFETY_RETIRED == "d_S = H - E"
    assert specification["retired_actuator"]["direction"] == \
        S.DIRECTION_SAFETY_RETIRED


# =============================================================== 2. ceiling
def test_the_applied_strength_is_the_gate_times_the_validated_strength():
    assert S.APPLIED_STRENGTH_FORMULA == "alpha_R = 0.15 * g_R"
    assert S.FROZEN_VALIDATED_REPAIR_STRENGTH == 0.15
    assert S.MAX_APPLIED_ALPHA_REPAIR == 0.15
    for gate, expected in ((0.0, 0.0), (0.5, 0.075), (1.0, 0.15)):
        assert S.applied_repair_strength(gate) == pytest.approx(expected)


@pytest.mark.parametrize("gate", [0.0, 0.01, 0.25, 0.5, 0.9999, 1.0])
def test_the_applied_strength_never_exceeds_the_validated_strength(gate):
    assert S.applied_repair_strength(gate) <= S.MAX_APPLIED_ALPHA_REPAIR


@pytest.mark.parametrize("gate", [1.0000001, 1.5, 2.0, -0.001, -1.0])
def test_a_gate_outside_the_sigmoid_range_is_refused(gate):
    """The ceiling is not a convention a caller can step around by handing the
    formula a number no sigmoid produces."""
    with pytest.raises(S.SingleDirectionError, match="outside"):
        S.applied_repair_strength(gate)


@pytest.mark.parametrize("gate", [float("nan"), float("inf"), "0.5", None,
                                  True])
def test_a_gate_that_is_not_a_number_in_range_is_refused(gate):
    with pytest.raises(S.SingleDirectionError):
        S.applied_repair_strength(gate)


def test_a_configuration_declaring_a_higher_ceiling_is_refused():
    with pytest.raises(S.SingleDirectionError, match="max_applied_alpha_R"):
        S.assert_final_controller_config(
            {"controller": {"outputs": ["g_R"], "max_applied_alpha_R": 0.2}})


# ============================================================== 3. position
def test_state_32_is_the_output_of_historical_block_31():
    assert S.READ_STATE == 32
    assert S.READ_LAYER == 31
    assert state_of_layer(31) == 32
    assert layer_of_state(32) == 31
    assert S.READ_STATE == PRIMARY_READ_STATE
    assert S.assert_read_state(32) == 32


@pytest.mark.parametrize("state", [33, 34])
def test_a_diagnostic_state_is_never_promoted_to_the_read_state(state):
    """States 33 and 34 stay sensitivity diagnostics. Selecting one after its
    metrics are known is the post-hoc choice this protocol refuses."""
    assert state in S.DIAGNOSTIC_STATES
    with pytest.raises(S.SingleDirectionError, match="DIAGNOSTIC"):
        S.assert_read_state(state)


@pytest.mark.parametrize("state", [0, 31, 45, 48])
def test_any_other_state_is_refused(state):
    with pytest.raises(S.SingleDirectionError):
        S.assert_read_state(state)


def test_the_diagnostic_states_are_the_probe_heads_sensitivity_states():
    assert S.DIAGNOSTIC_STATES == tuple(SENSITIVITY_STATES) == (33, 34)


def test_the_controller_reads_the_final_prompt_token_only():
    assert S.assert_token_position("final prompt token") == "final prompt token"
    for other in ("last generated token", "all prompt tokens", "mean pooled"):
        with pytest.raises(S.SingleDirectionError, match="final prompt token"):
            S.assert_token_position(other)


def test_the_specification_binds_the_read_position_and_refuses_decode():
    read = S.build_single_direction_specification()["read_position"]
    assert read["state"] == 32
    assert read["historical_layer"] == 31
    assert read["token_position"] == "final prompt token"
    assert read["prefill_only"] is True
    assert read["decode_positions_are_not_read"] is True
    assert read["diagnostic_states"] == [33, 34]


# ================================================================= 4. brake
def test_the_safety_reading_may_reduce_the_gate():
    assert S.apply_safety_brake(1.0, 0.5) == pytest.approx(0.5)
    assert S.apply_safety_brake(0.8, 0.25) == pytest.approx(0.2)


def test_the_safety_reading_may_veto_the_gate():
    assert S.apply_safety_brake(1.0, 0.0) == 0.0
    assert S.applied_repair_strength(S.apply_safety_brake(1.0, 0.0)) == 0.0


@pytest.mark.parametrize("gate,brake", [(0.4, 1.0), (0.4, 0.5), (1.0, 0.9),
                                        (0.0, 1.0)])
def test_a_braked_gate_is_never_larger_than_the_gate(gate, brake):
    assert S.apply_safety_brake(gate, brake) <= gate


@pytest.mark.parametrize("brake", [1.0001, 2.0, -0.1])
def test_a_brake_that_could_raise_the_gate_is_refused(brake):
    """A safety reading that can raise a repair gate is a steering input
    wearing a brake's name."""
    with pytest.raises(S.SingleDirectionError, match="brake"):
        S.apply_safety_brake(0.5, brake)


def test_the_safety_head_produces_no_direction_and_no_output():
    assert S.SAFETY_BRAKE["may_produce_a_steering_direction"] is False
    assert S.SAFETY_BRAKE["may_produce_a_controller_output"] is False
    assert S.SAFETY_BRAKE["may_increase_the_gate"] is False
    assert S.SAFETY_BRAKE["may_reduce_the_gate"] is True
    assert S.SAFETY_BRAKE["may_veto_the_gate"] is True


# ============================================================== 5. protocol
@pytest.mark.parametrize("fold", [0, 1, 2, 3, 4])
def test_every_component_of_a_held_out_fold_is_fitted_without_it(fold):
    plan = S.oof_protocol_for_fold(fold)
    for component in ("repair_probe", "safety_probe", "controller"):
        assert plan[component]["fitted_without_fold" if component != "controller"
                               else "trained_without_fold"] == fold
        assert fold not in plan[component]["training_folds"]
        assert len(plan[component]["training_folds"]) == 4
    assert plan["generated_folds"] == [fold]
    assert plan["evaluated_folds"] == [fold]
    assert S.assert_oof_protocol(plan, fold) == plan


@pytest.mark.parametrize("component", ["repair_probe", "safety_probe",
                                       "controller"])
def test_a_component_trained_on_the_held_out_fold_is_refused(component):
    plan = S.oof_protocol_for_fold(2)
    plan[component] = dict(plan[component])
    plan[component]["training_folds"] = [0, 1, 2, 3, 4]
    with pytest.raises(S.SingleDirectionError, match="include"):
        S.assert_oof_protocol(plan, 2)


@pytest.mark.parametrize("key", ["generated_folds", "evaluated_folds"])
def test_generating_or_evaluating_a_training_fold_is_refused(key):
    plan = S.oof_protocol_for_fold(2)
    plan[key] = [2, 3]
    with pytest.raises(S.SingleDirectionError, match=key):
        S.assert_oof_protocol(plan, 2)


def test_a_fold_outside_the_division_is_refused():
    with pytest.raises(S.SingleDirectionError, match="outside"):
        S.oof_protocol_for_fold(5)


def test_the_deployment_protocol_refits_then_trains_then_freezes():
    steps = S.build_single_direction_specification()["deployment_protocol"]
    assert steps["runs_only_after"] == "architecture freeze"
    assert len(steps["steps"]) == 3
    assert "refit the two probe heads on ALL development rows" in steps["steps"][0]
    assert "hyperparameter fixed" in steps["steps"][0]
    assert "final controller on ALL development rows" in steps["steps"][1]
    assert "BEFORE anything reads the sealed set" in steps["steps"][2]
    assert steps["sealed_set"]["accessed_before_the_freeze"] is False


def test_the_full_deployment_order_is_accepted():
    done = S.assert_deployment_preconditions(
        {step: True for step in S.DEPLOYMENT_STEPS})
    assert all(done.values())


def test_reading_the_sealed_set_before_the_freeze_is_refused():
    with pytest.raises(S.SingleDirectionError, match="sealed set"):
        S.assert_deployment_preconditions({
            "architecture_frozen": True,
            "probe_heads_refit_on_all_development_rows": True,
            "final_controller_trained_on_all_development_rows": True,
            "both_frozen_and_digested": False,
            "sealed_set_read": True})


def test_training_the_controller_before_the_heads_are_refit_is_refused():
    with pytest.raises(S.SingleDirectionError, match="preconditions"):
        S.assert_deployment_preconditions({
            "architecture_frozen": True,
            "probe_heads_refit_on_all_development_rows": False,
            "final_controller_trained_on_all_development_rows": True})


def test_refitting_before_the_architecture_is_frozen_is_refused():
    with pytest.raises(S.SingleDirectionError, match="preconditions"):
        S.assert_deployment_preconditions({
            "architecture_frozen": False,
            "probe_heads_refit_on_all_development_rows": True})


# =============================================================== 6. refusal
def test_the_historical_dual_configuration_is_refused_by_the_final_path():
    """It is preserved as phase-2 history and is not a final configuration."""
    config = json.loads(DUAL_CONFIG.read_text(encoding="utf-8"))
    with pytest.raises(S.SingleDirectionError):
        S.assert_final_controller_config(config, DUAL_CONFIG.name)
    with pytest.raises(S.SingleDirectionError):
        S.load_final_controller_config(DUAL_CONFIG)


@pytest.mark.parametrize("key", ["alpha_S", "safety_coefficient",
                                 "safety_steering_coefficient",
                                 "safety_strength", "a_S", "g_S"])
def test_a_safety_steering_coefficient_is_refused_whatever_it_is_set_to(key):
    with pytest.raises(S.SingleDirectionError,
                       match="safety steering coefficient"):
        S.assert_final_controller_config(_minimal(steering={key: 0.0}))


def test_alpha_s_above_zero_names_the_strength_it_refuses():
    with pytest.raises(S.SingleDirectionError, match="alpha_S = 0.2"):
        S.assert_final_controller_config(
            _minimal(arms=[{"arm_id": "combined", "alpha_S": 0.2}]))


@pytest.mark.parametrize("block", [
    {"actuator": {"direction": "H - E"}},
    {"actuators": {"directions": ["B_d1 - E", "H - E"]}},
    {"steering": {"safety_direction": "d_S = H - E"}},
    {"controller_extra": {"steered_directions": ["H − E"]}},
])
def test_an_active_h_minus_e_actuator_is_refused(block):
    with pytest.raises(S.SingleDirectionError, match="actuator slot"):
        S.assert_final_controller_config(_minimal(**block))


def test_the_retired_direction_may_still_be_recorded_as_retired():
    """A specification that cannot say what it retired is worse, not safer."""
    outcome = S.assert_final_controller_config(
        _minimal(retired_actuator={"direction": "d_S = H - E",
                                   "active": False}))
    assert outcome["recorded_retired_direction"]


def test_the_superseded_parameterisation_cannot_be_carried_forward():
    with pytest.raises(S.SingleDirectionError, match="superseded"):
        S.assert_final_controller_config(
            _minimal(note="two sigmoid values, each in [0, 1]"))


def test_a_configuration_with_no_controller_block_is_refused():
    with pytest.raises(S.SingleDirectionError, match="no 'controller' block"):
        S.assert_final_controller_config({"read_states": {"states": [32]}})


def test_the_minimal_accepted_configuration_is_accepted():
    outcome = S.assert_final_controller_config(_minimal())
    assert outcome["checks"].all_passed


# ============================================================= 7. committed
def test_the_committed_specification_passes_every_gate():
    outcome = S.verify_single_direction_specification()
    assert outcome["checks"].all_passed
    assert outcome["committed"]["name"] == S.CONTROLLER_NAME
    assert outcome["committed"]["schema"] == S.SCHEMA


def test_the_committed_specification_is_a_new_file_not_a_rewrite():
    committed = json.loads(S.SINGLE_DIRECTION_CONFIG.read_text("utf-8"))
    assert committed["supersedes"]["configuration"] == \
        "configs/dual_direction_controller.json"
    assert committed["supersedes"]["the_historical_file_is_not_rewritten"] is True
    assert DUAL_CONFIG.is_file()
    assert S.SINGLE_DIRECTION_CONFIG != DUAL_CONFIG


def test_the_committed_specification_trains_nothing_and_leaves_training_open():
    committed = json.loads(S.SINGLE_DIRECTION_CONFIG.read_text("utf-8"))
    assert committed["status"] == "SPECIFIED_NOT_IMPLEMENTED"
    for key in ("controller_input", "controller_hidden_size",
                "controller_depth", "controller_activation_function",
                "controller_loss", "controller_regularisation",
                "controller_optimiser", "controller_training_schedule",
                "successful_strength_targets_or_other_supervisory_signal",
                "controller_selection_criterion",
                "safety_brake_operating_point"):
        assert committed["unresolved"][key] == "UNRESOLVED", key
    assert "BEFORE the controller is trained" in committed["unresolved"]["note"]
    assert committed["resolved_by_phase_2"]["number_of_controller_outputs"] == 1


@pytest.mark.parametrize("field", ["controller", "read_position",
                                   "safety_brake", "oof_protocol",
                                   "deployment_protocol", "phase2"])
def test_an_edited_specification_field_is_refused(tmp_path, field):
    path = tmp_path / S.SINGLE_DIRECTION_CONFIG.name
    payload = json.loads(S.SINGLE_DIRECTION_CONFIG.read_text("utf-8"))
    if field == "controller":
        payload[field]["gate"] = {"moved": True}
    else:
        payload[field] = {"moved": True}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    with pytest.raises(ValidationError, match=f"specification_{field}"):
        S.verify_single_direction_specification(path)


def test_the_specification_holds_no_prompt_or_response_text():
    payload = json.loads(S.SINGLE_DIRECTION_CONFIG.read_text(encoding="utf-8"))
    assert payload["contains_prompt_or_response_text"] is False
    # the disclaimer's own key spells "prompt_or_response_text"; it is removed
    # before the scan so the check cannot fail on it.
    payload.pop("contains_prompt_or_response_text")
    text = json.dumps(payload)
    for banned in ("prompt_text", "response_text"):
        assert banned not in text, banned


def test_the_module_implements_no_trainer():
    """The architecture is fixed here. The controller is not."""
    import ast
    source = (REPO_ROOT / "src" / "routegate" / "singledirection.py").read_text(
        encoding="utf-8")
    tree = ast.parse(source)
    imported = {node.module for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module}
    imported |= {alias.name for node in ast.walk(tree)
                 if isinstance(node, ast.Import) for alias in node.names}
    for banned in ("torch", "sklearn", "transformers"):
        assert not any(name.startswith(banned) for name in imported), banned
    specification = S.build_single_direction_specification()
    assert "the controller itself" in specification["not_implemented_here"][0]
