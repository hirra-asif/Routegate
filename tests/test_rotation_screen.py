"""The rotational H - E screen: its sites, its arms, its records and its gates.

No model, no GPU and no corpus is needed for most of this: the site plans are
arithmetic over synthetic tensors and a synthetic bank, and the inventory,
resume, shard and consolidation logic are pure bookkeeping. The two tests that
need the real frozen inputs say so and skip without them.

The numbered comments correspond to the tests the screen's specification
requires of the experiment as opposed to of the operator.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import needs_real_inputs                              # noqa: E402

from routegate.directionbank import BANK_STATES, HISTORICAL_LAYERS
from routegate.dualsteer import (
    ARM_BY_ID,
    BUDGET_REFERENCE,
    ValidatedDirectionBank,
    build_plans,
)
from routegate.intervention import apply_delta_at_position
from routegate.phase2 import (
    PHASE2_DECISION_RULE,
    PHASE2_RECOVERY_THRESHOLD,
    PHASE2_SAFETY_ONLY_HARM_LIMIT,
    evaluate_phase2_decision,
)
from routegate.phase2closure import CLOSED_PROTOCOLS, assert_protocol_is_open
from routegate.rotation import ROTATION_TOLERANCES, rotate_toward
from routegate.rotationfreeze import (
    DECISION_RULE,
    PREDECLARATION,
    PRESERVED_ARTIFACTS,
    RECOVERY_THRESHOLD,
    ROTATION_ONLY_HARM_LIMIT,
    SAFETY_DIRECTION_SHA256,
    SUPERSEDED_PREDECLARATION,
    SUPERSEDED_PREDECLARATION_SHA256,
    RotationFreezeError,
    build_predeclaration,
    evaluate_rotation_decision,
    record_sha256,
    verify_predeclaration,
    verify_preserved_artifacts,
)
from routegate.rotationscreen import (
    ALPHA_REPAIR,
    ANALYSIS_TABLE_RECORDS,
    ARM_BY_ID_ROTATION,
    ARMS_BY_BETA,
    BETA_GRID,
    COMBINED_ARMS,
    DEFAULT_CONFIG,
    INTERVENTION_ORDER,
    INTERVENTION_TYPE,
    NEW_ARMS,
    NEW_GENERATIONS,
    N_ROWS,
    PROTOCOL,
    RECORD_SCHEMA,
    REFERENCE_ARM_IDS,
    REFERENCE_RECORDS,
    REPAIR_ARM_ID,
    ROTATION_ONLY_ARMS,
    RUN_LOCKED_FIELDS,
    SELECTED_ROWS,
    SELECTED_ROWS_SHA256,
    RotationScreenError,
    RotationSitePlan,
    append_record,
    assert_config_matches,
    assert_inventory,
    assert_safety_direction_is_h_minus_e,
    build_inventory,
    build_site_plans,
    claim_output_dir,
    declared_arms,
    implemented_arms,
    make_record,
    plan_work,
    read_records,
    record_key,
    rotation_controller,
    safety_direction_digests,
    safety_direction_for_state,
    selection_digest_holds,
    shard,
    verify_consistency,
    work_key,
)
from routegate.validation import SealedSetError, looks_sealed

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE = REPO_ROOT / "experiments" / "rotation_safety_direction"
BANK_DIR = Path.home() / "routegate-data" / "frozen" / "validated_direction_bank"
HIDDEN = 2048

needs_bank = pytest.mark.skipif(
    not (BANK_DIR / "validated_direction_bank.npz").is_file(),
    reason="the validated direction bank is not reachable on this machine")


# ------------------------------------------------------------------ fixtures
def unit_rows(rows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    array = rng.normal(size=(rows, HIDDEN)).astype(np.float32)
    return (array / np.linalg.norm(array, axis=1, keepdims=True)
            ).astype(np.float32)


@pytest.fixture
def bank() -> ValidatedDirectionBank:
    """A three-layer bank with three DISTINCT directions per array."""
    return ValidatedDirectionBank(
        repair=unit_rows(len(BANK_STATES), 11),
        safety=unit_rows(len(BANK_STATES), 22),
        artifact_sha256="a" * 64, manifest_sha256="b" * 64,
        manifest={
            "construction": {"subtraction_order": {"repair": "B_d1 - E",
                                                   "safety": "H - E"}},
            "tensors": {"read_from_capture": {
                "E": "safe_residual_stream",
                "B_d1": "donor_residual_stream",
                "H": "harmful_residual_stream"}},
            "orthogonalisation": {"applied": False},
        })


@pytest.fixture
def capture():
    """One captured activation per state a plan needs, including the budgets."""
    generator = torch.Generator().manual_seed(4242)
    return {state: torch.randn(HIDDEN, generator=generator,
                               dtype=torch.float32) * 6.0
            for state in range(min(BANK_STATES) - 1, max(BANK_STATES) + 1)}


@pytest.fixture
def rows():
    return [{"row_id": row_id, "row_class": "H",
             "prompt_sha256": f"{index:064d}"}
            for index, row_id in enumerate(SELECTED_ROWS)]


def activation(seed: int = 99, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    tensor = torch.randn(1, 5, HIDDEN, generator=generator,
                         dtype=torch.float32) * 4.0
    return tensor.to(dtype)


# ================================ 9. each layer uses its own H - E vector
def test_each_layer_rotates_toward_its_own_h_minus_e_vector(bank, capture):
    plans = build_site_plans(bank, ARM_BY_ID_ROTATION["rotation_only_015"],
                             capture)
    assert sorted(plans) == list(BANK_STATES)
    for index, state in enumerate(BANK_STATES):
        assert np.array_equal(plans[state].direction, bank.safety[index])
        assert plans[state].historical_layer == HISTORICAL_LAYERS[index]

    digests = {plans[state].direction_sha256 for state in BANK_STATES}
    assert len(digests) == len(BANK_STATES)


def test_no_layer_borrows_another_layers_direction(bank, capture):
    plans = build_site_plans(bank, ARM_BY_ID_ROTATION["rotation_only_015"],
                             capture)
    for index, state in enumerate(BANK_STATES):
        others = [bank.safety[other] for other in range(len(BANK_STATES))
                  if other != index]
        for other in others:
            assert not np.array_equal(plans[state].direction, other)


def test_the_safety_vector_for_a_state_is_the_banks_own_row(bank):
    for index, state in enumerate(BANK_STATES):
        assert np.array_equal(safety_direction_for_state(bank, state),
                              bank.safety[index])


def test_the_per_layer_digests_are_three_distinct_values(bank):
    digests = safety_direction_digests(bank)
    assert sorted(digests) == [str(state) for state in BANK_STATES]
    assert len(set(digests.values())) == len(BANK_STATES)


@needs_bank
def test_the_frozen_banks_per_layer_h_minus_e_digests_are_the_pinned_ones():
    from routegate.dualsteer import load_validated_bank
    from routegate.rotationfreeze import DIRECTION_BANK_SHA256

    real = load_validated_bank(BANK_DIR, DIRECTION_BANK_SHA256)
    assert safety_direction_digests(real) == SAFETY_DIRECTION_SHA256
    assert len(set(SAFETY_DIRECTION_SHA256.values())) == len(BANK_STATES)


def test_a_bank_whose_manifest_does_not_say_h_minus_e_is_refused(bank):
    manifest = json.loads(json.dumps(bank.manifest))
    manifest["construction"]["subtraction_order"]["safety"] = "E - H"
    flipped = ValidatedDirectionBank(
        repair=bank.repair, safety=bank.safety,
        artifact_sha256=bank.artifact_sha256,
        manifest_sha256=bank.manifest_sha256, manifest=manifest)
    with pytest.raises(RotationScreenError, match="H - E"):
        assert_safety_direction_is_h_minus_e(flipped)


def test_an_orthogonalised_bank_is_refused(bank):
    manifest = json.loads(json.dumps(bank.manifest))
    manifest["orthogonalisation"]["applied"] = True
    other = ValidatedDirectionBank(
        repair=bank.repair, safety=bank.safety,
        artifact_sha256=bank.artifact_sha256,
        manifest_sha256=bank.manifest_sha256, manifest=manifest)
    with pytest.raises(RotationScreenError, match="orthogonalis"):
        assert_safety_direction_is_h_minus_e(other)


# ================== 10. one beta is reused unchanged across layers 31-33
@pytest.mark.parametrize("arm_id", sorted(ARM_BY_ID_ROTATION))
def test_one_beta_is_shared_by_all_three_layers(bank, capture, arm_id):
    arm = ARM_BY_ID_ROTATION[arm_id]
    plans = build_site_plans(bank, arm, capture)
    betas = {plans[state].beta for state in BANK_STATES}
    assert betas == {arm.beta}
    assert [plans[state].historical_layer for state in BANK_STATES] == \
        list(HISTORICAL_LAYERS)


def test_the_realised_phi_differs_by_layer_because_theta_does(bank, capture):
    """One beta, three angles: beta is a fraction, not a displacement."""
    arm = ARM_BY_ID_ROTATION["rotation_only_020"]
    plans = build_site_plans(bank, arm, capture)
    tensor = activation()
    thetas, phis = [], []
    for state in BANK_STATES:
        _, diagnostics = plans[state].apply_at(tensor, 4)
        thetas.append(float(diagnostics["theta"]))
        phis.append(float(diagnostics["phi"]))
        assert diagnostics["phi"] == pytest.approx(
            arm.beta * diagnostics["theta"], rel=1e-12)
    assert len(set(round(value, 6) for value in thetas)) == len(BANK_STATES)
    assert len(set(round(value, 6) for value in phis)) == len(BANK_STATES)


# ==================== 11. the combined arms apply repair before rotation
@pytest.mark.parametrize("arm", COMBINED_ARMS, ids=lambda arm: arm.arm_id)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_a_combined_arm_repairs_first_and_rotates_second(bank, capture, arm,
                                                         dtype):
    plans = build_site_plans(bank, arm, capture)
    state = BANK_STATES[0]
    plan = plans[state]
    tensor = activation(dtype=dtype)
    position = 4

    changed, diagnostics = plan.apply_at(tensor, position)

    repaired = apply_delta_at_position(tensor, position, plan.repair_plan.delta)
    expected = rotate_toward(repaired[0, position], plan.direction, arm.beta,
                             tolerances=ROTATION_TOLERANCES).tensor
    assert bool(torch.equal(changed[0, position], expected.to(dtype)))
    assert diagnostics["steps_applied"] == ["repair", "rotation"]
    assert diagnostics["intervention_order"] == INTERVENTION_ORDER
    assert diagnostics["applies_repair"] is True


@pytest.mark.parametrize("arm", COMBINED_ARMS, ids=lambda arm: arm.arm_id)
def test_the_other_order_would_produce_a_different_vector(bank, capture, arm):
    """Rotation and addition do not commute, which is why the order is fixed."""
    plan = build_site_plans(bank, arm, capture)[BANK_STATES[0]]
    tensor = activation()
    position = 4

    repair_then_rotate, _ = plan.apply_at(tensor, position)
    rotated = rotate_toward(tensor[0, position], plan.direction,
                            arm.beta).tensor
    rotate_then_repair = rotated + plan.repair_plan.delta.to(rotated.dtype)

    difference = float(torch.linalg.vector_norm(
        repair_then_rotate[0, position].float() - rotate_then_repair.float()))
    scale = float(torch.linalg.vector_norm(tensor[0, position].float()))
    assert difference > 1e-4 * scale


def test_the_repair_half_is_the_existing_repair_015_plan(bank, capture):
    """Built by dualsteer, at the predeclared arm, with its budget unchanged."""
    arm = ARM_BY_ID_ROTATION["repair015_rotation010"]
    plans = build_site_plans(bank, arm, capture)
    inherited = build_plans(bank, ARM_BY_ID[REPAIR_ARM_ID], capture,
                            BUDGET_REFERENCE)
    for state in BANK_STATES:
        mine = plans[state].repair_plan
        theirs = inherited[state]
        assert bool(torch.equal(mine.delta, theirs.delta))
        assert mine.budget == theirs.budget
        assert mine.budget_reference_state == theirs.budget_reference_state
        assert mine.budget_reference_state == state - 1
        assert mine.strength == pytest.approx(ALPHA_REPAIR)


def test_the_combined_arms_pin_alpha_r_at_the_phase1_operating_point():
    assert {arm.alpha_repair for arm in COMBINED_ARMS} == {ALPHA_REPAIR}
    assert ALPHA_REPAIR == ARM_BY_ID[REPAIR_ARM_ID].alpha_repair
    assert ARM_BY_ID[REPAIR_ARM_ID].alpha_safety == 0.0


def test_a_combined_record_names_the_repair_direction_it_used(bank, capture):
    plans = build_site_plans(bank, ARM_BY_ID_ROTATION["repair015_rotation005"],
                             capture)
    for state in BANK_STATES:
        assert plans[state].repair_direction_sha256 == \
            bank.direction_sha256("repair_direction", state)


# ================== 12. the rotation-only arms never apply B - E
@pytest.mark.parametrize("arm", ROTATION_ONLY_ARMS, ids=lambda arm: arm.arm_id)
def test_a_rotation_only_arm_applies_no_repair(bank, capture, arm):
    plans = build_site_plans(bank, arm, capture)
    for state in BANK_STATES:
        plan = plans[state]
        assert plan.repair_plan is None
        assert plan.repair_direction_sha256 is None
        assert plan.applies_repair is False
        assert plan.order_applied == "rotation_only"


@pytest.mark.parametrize("arm", ROTATION_ONLY_ARMS, ids=lambda arm: arm.arm_id)
def test_a_rotation_only_edit_is_exactly_the_rotation(bank, capture, arm):
    plan = build_site_plans(bank, arm, capture)[BANK_STATES[1]]
    tensor = activation()
    changed, diagnostics = plan.apply_at(tensor, 4)
    expected = rotate_toward(tensor[0, 4], plan.direction, arm.beta).tensor
    assert bool(torch.equal(changed[0, 4], expected))
    assert diagnostics["steps_applied"] == ["rotation"]
    assert diagnostics["repair"] is None
    # Nothing outside the final prompt token moved.
    assert bool(torch.equal(changed[0, :4], tensor[0, :4]))


def test_the_rotation_only_arms_carry_alpha_r_zero():
    assert {arm.alpha_repair for arm in ROTATION_ONLY_ARMS} == {0.0}
    assert all(arm.applies_repair is False for arm in ROTATION_ONLY_ARMS)


def test_a_rotation_only_arm_leaves_the_norm_where_it_was(bank, capture):
    plan = build_site_plans(bank, ARM_BY_ID_ROTATION["rotation_only_020"],
                            capture)[BANK_STATES[0]]
    tensor = activation()
    changed, diagnostics = plan.apply_at(tensor, 4)
    before = float(torch.linalg.vector_norm(tensor[0, 4].float()))
    after = float(torch.linalg.vector_norm(changed[0, 4].float()))
    assert after == pytest.approx(
        before, rel=ROTATION_TOLERANCES.norm_preservation_rtol_float32)
    assert diagnostics["cosine_h_minus_e_realised"] > \
        diagnostics["cosine_h_minus_e_pre_rotation"]


def test_a_combined_arm_changes_the_norm_only_through_its_repair_half(bank,
                                                                      capture):
    """The rotation preserves whatever norm the repair produced, not the input's."""
    plan = build_site_plans(bank, ARM_BY_ID_ROTATION["repair015_rotation020"],
                            capture)[BANK_STATES[0]]
    tensor = activation()
    changed, diagnostics = plan.apply_at(tensor, 4)
    assert float(diagnostics["norm_realised_float32"]) == pytest.approx(
        float(diagnostics["norm_pre_rotation_float32"]),
        rel=ROTATION_TOLERANCES.norm_preservation_rtol_float32)
    assert diagnostics["norm_pre_repair_float32"] != pytest.approx(
        diagnostics["norm_pre_rotation_float32"], rel=1e-9)


# ================ 13. eight arms and exactly 18 frozen rows, covered once
def test_the_inventory_is_eighteen_rows_at_eight_arms(rows):
    inventory = build_inventory(rows)
    assert len(inventory) == NEW_GENERATIONS == 144
    assert len({(item["row_id"], item["arm_id"]) for item in inventory}) == 144
    assert sorted({item["row_id"] for item in inventory}) == list(SELECTED_ROWS)
    assert sorted({item["arm_id"] for item in inventory}) == \
        sorted(ARM_BY_ID_ROTATION)
    for arm_id in ARM_BY_ID_ROTATION:
        assert sum(1 for item in inventory
                   if item["arm_id"] == arm_id) == N_ROWS
    for row_id in SELECTED_ROWS:
        assert sum(1 for item in inventory
                   if item["row_id"] == row_id) == len(NEW_ARMS)


def test_the_arm_ids_are_the_eight_the_specification_names():
    assert [arm.arm_id for arm in NEW_ARMS] == [
        "rotation_only_005", "rotation_only_010", "rotation_only_015",
        "rotation_only_020", "repair015_rotation005", "repair015_rotation010",
        "repair015_rotation015", "repair015_rotation020"]
    assert list(BETA_GRID) == [0.05, 0.10, 0.15, 0.20]
    assert len(ARMS_BY_BETA) == len(BETA_GRID)
    for beta, pair in ARMS_BY_BETA.items():
        assert ARM_BY_ID_ROTATION[pair["rotation_only"]].beta == beta
        assert ARM_BY_ID_ROTATION[pair["combined"]].beta == beta


def test_the_counts_are_derived_and_agree(rows):
    counts = assert_inventory(build_inventory(rows))
    assert counts["total"] == NEW_GENERATIONS
    assert counts["reference_records"] == REFERENCE_RECORDS == 36
    assert counts["analysis_table_records"] == ANALYSIS_TABLE_RECORDS == 180
    assert NEW_GENERATIONS + REFERENCE_RECORDS == ANALYSIS_TABLE_RECORDS


def test_an_inventory_missing_a_row_is_refused(rows):
    with pytest.raises(RotationScreenError, match="frozen phase-2 rows"):
        assert_inventory(build_inventory(rows[:-1]))


def test_an_inventory_with_a_duplicate_is_refused(rows):
    inventory = build_inventory(rows)
    with pytest.raises(RotationScreenError, match="duplicated"):
        assert_inventory(inventory + [dict(inventory[0])])


def test_a_row_outside_the_selection_never_enters_the_inventory(rows):
    widened = rows + [{"row_id": "strongreject:not_selected", "row_class": "H",
                       "prompt_sha256": "0" * 64}]
    inventory = build_inventory(widened)
    assert len(inventory) == NEW_GENERATIONS
    assert "strongreject:not_selected" not in {item["row_id"]
                                               for item in inventory}


def test_a_row_of_another_class_is_refused():
    with pytest.raises(RotationScreenError, match="steers the"):
        build_inventory([{"row_id": SELECTED_ROWS[0], "row_class": "E",
                          "prompt_sha256": "0" * 64}])


def test_the_selection_digest_is_the_phase2_one():
    assert selection_digest_holds() == SELECTED_ROWS_SHA256
    assert len(SELECTED_ROWS) == N_ROWS == 18


def test_the_reference_arms_are_not_generated_arms():
    assert set(REFERENCE_ARM_IDS).isdisjoint(ARM_BY_ID_ROTATION)
    assert list(REFERENCE_ARM_IDS) == ["baseline", REPAIR_ARM_ID]


def test_a_record_carrying_a_reference_arm_is_refused():
    record = {"row_id": SELECTED_ROWS[0], "row_class": "H",
              "arm_id": "repair_015", "beta": 0.15, "alpha_R": 0.15,
              "direction_bank_sha256": "a", "generation_config_sha256": "b"}
    with pytest.raises(RotationScreenError, match="inherited reference arm"):
        verify_consistency([record], {})


# ==================================== 14. a dry run creates no output directory
def test_claiming_without_create_makes_no_directory(tmp_path):
    target = tmp_path / "never_created"
    assert claim_output_dir(target, create=False) == target
    assert not target.exists()


def test_claiming_with_create_makes_the_directory(tmp_path):
    target = tmp_path / "made"
    claim_output_dir(target, create=True)
    assert target.is_dir()


def test_a_non_empty_directory_is_refused_without_resume(tmp_path):
    target = tmp_path / "used"
    target.mkdir()
    (target / "rotation_records.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RotationScreenError, match="fresh record"):
        claim_output_dir(target, create=True, resume=False)


def test_resume_reaches_a_non_empty_directory(tmp_path):
    """The bug this repository already made once: --resume must be reachable."""
    target = tmp_path / "resumable"
    target.mkdir()
    (target / "rotation_records.jsonl").write_text("", encoding="utf-8")
    assert claim_output_dir(target, create=True, resume=True) == target


@needs_real_inputs
@needs_bank
def test_the_runner_dry_run_creates_no_output_directory(tmp_path):
    target = tmp_path / "dry_run_output"
    result = subprocess.run(
        [sys.executable,
         str(STAGE / "run_rotation_screen.py"),
         "--direction-bank", str(BANK_DIR),
         "--output-dir", str(target),
         "--dry-run"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stderr
    assert not target.exists()
    assert "no model loaded, nothing written" in result.stdout
    assert '"planned": 144' in result.stdout


# ========== 15. resume, sharding and consolidation are deterministic
@pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 6, 8, 9, 16, 144])
def test_the_shards_partition_the_inventory_exactly(rows, count):
    inventory = build_inventory(rows)
    pieces = [shard(inventory, index, count) for index in range(1, count + 1)]
    keys = [(item["row_id"], item["arm_id"])
            for piece in pieces for item in piece]
    expected = [(item["row_id"], item["arm_id"]) for item in inventory]
    assert keys == expected                      # exact, ordered cover
    assert len(set(keys)) == len(keys)           # no overlap
    assert sum(len(piece) for piece in pieces) == NEW_GENERATIONS
    assert max(len(piece) for piece in pieces) - \
        min(len(piece) for piece in pieces) <= 1


@pytest.mark.parametrize("count", [1, 4, 8])
def test_the_shard_split_is_deterministic(rows, count):
    first = build_inventory(rows)
    second = build_inventory(list(reversed(rows)))
    assert first == second                       # inventory order is sorted
    for index in range(1, count + 1):
        assert shard(first, index, count) == shard(second, index, count)


def test_an_out_of_range_shard_index_is_refused(rows):
    inventory = build_inventory(rows)
    with pytest.raises(RotationScreenError, match="outside"):
        shard(inventory, 0, 4)
    with pytest.raises(RotationScreenError, match="outside"):
        shard(inventory, 5, 4)
    with pytest.raises(RotationScreenError, match="at least 1"):
        shard(inventory, 1, 0)


def finished_record(item, **overrides):
    values = {
        "row_id": item["row_id"], "row_class": "H", "arm_id": item["arm_id"],
        "beta": item["beta"], "alpha_R": item["alpha_R"],
        "direction_bank_sha256": "bank", "generation_config_sha256": "gen",
        "model": "Qwen/Qwen3-30B-A3B", "revision": "rev",
        "tokenizer_config_sha256": "tok", "chat_template_sha256": "tpl",
        "intervention_type": INTERVENTION_TYPE,
        "intervention_order": INTERVENTION_ORDER,
        "budget_reference": BUDGET_REFERENCE, "states": list(BANK_STATES),
        "historical_layers": list(HISTORICAL_LAYERS),
        "dataset_sha256": "data", "smoke": False, "protocol": PROTOCOL,
        "tolerances_version": 1, "predeclaration_sha256": "pre",
        "selected_rows_sha256": SELECTED_ROWS_SHA256,
        "prompt_sha256": item["prompt_sha256"], "response_text": "no.",
        "response_sha256": "digest", "status": "ok",
    }
    values.update(overrides)
    return make_record(**values)


def run_locked_for(record):
    return {field: record[field] for field in RUN_LOCKED_FIELDS}


def test_resume_skips_exactly_what_is_already_recorded(rows, tmp_path):
    inventory = build_inventory(rows)
    path = tmp_path / "rotation_records.jsonl"
    done_items = inventory[:37]
    for item in done_items:
        append_record(path, finished_record(item))

    existing = read_records(path)
    assert len(existing) == 37
    verify_consistency(existing, run_locked_for(existing[0]))
    done = {record_key(record) for record in existing}
    todo = plan_work(inventory, done, "bank", "gen")
    assert len(todo) == NEW_GENERATIONS - 37
    assert {(item["row_id"], item["arm_id"]) for item in todo}.isdisjoint(
        {(item["row_id"], item["arm_id"]) for item in done_items})
    assert todo == [item for item in inventory if item not in done_items]


def test_resume_is_idempotent(rows, tmp_path):
    inventory = build_inventory(rows)
    path = tmp_path / "rotation_records.jsonl"
    for item in inventory:
        append_record(path, finished_record(item))
    done = {record_key(record) for record in read_records(path)}
    assert plan_work(inventory, done, "bank", "gen") == []


def test_a_resume_into_a_different_run_is_refused(rows, tmp_path):
    item = build_inventory(rows)[0]
    record = finished_record(item)
    with pytest.raises(RotationScreenError, match="different experiment"):
        verify_consistency([record],
                           {**run_locked_for(record),
                            "direction_bank_sha256": "another"})


def test_a_duplicate_record_is_refused_not_deduplicated(rows):
    item = build_inventory(rows)[0]
    record = finished_record(item)
    with pytest.raises(RotationScreenError, match="repeats key"):
        verify_consistency([record, dict(record)], run_locked_for(record))


def test_a_record_whose_coefficients_disagree_with_its_arm_is_refused(rows):
    item = build_inventory(rows)[0]
    record = finished_record(item, beta=0.99)
    with pytest.raises(RotationScreenError, match="predeclared arm"):
        verify_consistency([record], run_locked_for(record))


def test_a_record_for_an_unselected_row_is_refused(rows):
    item = build_inventory(rows)[0]
    record = finished_record(item, row_id="strongreject:not_selected")
    with pytest.raises(RotationScreenError, match="frozen phase-2 rows"):
        verify_consistency([record], run_locked_for(record))


def test_an_undeclared_record_field_is_refused(rows):
    item = build_inventory(rows)[0]
    with pytest.raises(RotationScreenError, match="unknown record field"):
        finished_record(item, invented_field=1)


def test_a_record_of_another_schema_is_refused(tmp_path):
    path = tmp_path / "rotation_records.jsonl"
    path.write_text(json.dumps({"schema": "something/else"}) + "\n",
                    encoding="utf-8")
    with pytest.raises(RotationScreenError, match="carries schema"):
        read_records(path)


def test_a_truncated_final_line_is_resumable_but_never_strict(tmp_path, rows):
    item = build_inventory(rows)[0]
    path = tmp_path / "rotation_records.jsonl"
    append_record(path, finished_record(item))
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"schema": "routegate/rotat')
    assert len(read_records(path, strict=False)) == 1
    with pytest.raises(RotationScreenError, match="not valid JSON"):
        read_records(path, strict=True)


def test_the_record_schema_is_stamped(rows):
    item = build_inventory(rows)[0]
    assert finished_record(item)["schema"] == RECORD_SCHEMA


def test_consolidation_order_is_deterministic_under_shuffling(rows):
    sys.path.insert(0, str(STAGE))
    try:
        import importlib
        module = importlib.import_module("consolidate_rotation_screen")
    finally:
        sys.path.remove(str(STAGE))

    inventory = build_inventory(rows)
    records = [finished_record(item) for item in inventory]
    forward = {record_key(record): record for record in records}
    backward = {record_key(record): record for record in reversed(records)}
    assert module.jsonl_text(module.order_records(forward)) == \
        module.jsonl_text(module.order_records(backward))
    ordered = module.order_records(forward)
    assert [(record["row_id"], record["arm_id"]) for record in ordered] == \
        [(item["row_id"], item["arm_id"]) for item in inventory]


def test_the_work_key_and_the_record_key_agree(rows):
    item = build_inventory(rows)[0]
    record = finished_record(item)
    assert work_key(item, "bank", "gen") == record_key(record)


# ================================== 16. no sealed-set path is reachable
def test_no_module_in_this_stage_carries_a_held_out_path_literal():
    permitted = {"manifests/sealed_set_commitment.json",
                 "/tmp/sealed_benchmark_out", "/tmp/final_benchmark.csv"}
    sources = sorted(STAGE.glob("*.py")) + [
        REPO_ROOT / "src" / "routegate" / "rotation.py",
        REPO_ROOT / "src" / "routegate" / "rotationscreen.py",
        REPO_ROOT / "src" / "routegate" / "rotationfreeze.py",
    ]
    offenders = []
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                      if isinstance(node, (ast.Module, ast.ClassDef,
                                           ast.FunctionDef,
                                           ast.AsyncFunctionDef))
                      and node.body and isinstance(node.body[0], ast.Expr)
                      and isinstance(node.body[0].value, ast.Constant)
                      and isinstance(node.body[0].value.value, str)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) \
                    and isinstance(node.value, str) \
                    and id(node) not in docstrings \
                    and ("/" in node.value or node.value.endswith(
                        (".csv", ".json", ".jsonl", ".npy", ".npz"))) \
                    and looks_sealed(node.value) \
                    and node.value not in permitted:
                offenders.append(f"{path.name}: {node.value!r}")
    assert offenders == []


def test_the_output_directory_gate_refuses_a_held_out_name():
    with pytest.raises(SealedSetError):
        claim_output_dir(Path("/tmp/sealed_benchmark_out"), create=False)


def test_the_configuration_names_no_held_out_material():
    from routegate.validation import assert_config_is_sealed_free
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert_config_is_sealed_free(config, "rotation screen configuration")
    assert config["evaluation_scope"]["held_out_material_accessible"] is False
    assert config["evaluation_scope"]["development_only"] is True


def test_the_predeclaration_states_the_held_out_split_is_unreachable():
    committed = json.loads(PREDECLARATION.read_text(encoding="utf-8"))
    assert committed["frozen_inputs"]["sealed_set"]["accessible"] is False


def test_the_runner_exposes_no_option_that_could_reach_held_out_material():
    sys.path.insert(0, str(STAGE))
    try:
        import importlib
        module = importlib.import_module("run_rotation_screen")
    finally:
        sys.path.remove(str(STAGE))
    parser = module.build_parser()
    for action in parser._actions:
        for option in action.option_strings:
            assert looks_sealed(option) is None
        assert looks_sealed(action.help or "") is None


# ================================================= the controller and its plans
class ToyBlock(torch.nn.Module):
    def __init__(self, index: int):
        super().__init__()
        self.register_buffer("bias", torch.full((HIDDEN,), float(index + 1)))

    def forward(self, hidden_states, **kwargs):
        return (hidden_states + self.bias,)


class ToyBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(8, HIDDEN)
        self.layers = torch.nn.ModuleList([ToyBlock(i) for i in range(3)])

    def forward(self, input_ids):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)[0]
        return hidden


class ToyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ToyBase()

    def forward(self, input_ids):
        return self.model(input_ids)


@pytest.fixture
def toy():
    torch.manual_seed(20260830)
    return ToyCausalLM()


def toy_plan(beta: float = 0.15, state: int = 2) -> RotationSitePlan:
    direction = unit_rows(1, 5)[0]
    return RotationSitePlan(
        state=state, historical_layer=state - 1, beta=beta,
        direction=direction, direction_sha256="c" * 64)


def test_the_controller_applies_the_rotation_at_the_final_prompt_token(toy):
    controller = rotation_controller(toy, states=[2], n_layers=3)
    try:
        plan = toy_plan()
        controller.reset_run(mode="patch", example_id="row", plans={2: plan})
        ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        with torch.no_grad():
            toy(ids)
        controller.verify_applied()
        assert set(controller.rotation_diagnostics) == {2}
        diagnostics = controller.rotation_diagnostics[2]
        assert diagnostics["state"] == 2
        assert diagnostics["intervention_order"] == "rotation_only"
        assert diagnostics["beta"] == 0.15
        assert controller.run.hits[2] == 1
    finally:
        controller.close()


def test_the_controller_touches_no_decode_forward(toy):
    controller = rotation_controller(toy, states=[2], n_layers=3)
    try:
        controller.reset_run(mode="patch", example_id="row",
                             plans={2: toy_plan()})
        ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        with torch.no_grad():
            toy(ids)
            toy(ids)
            toy(ids)
        assert controller.run.hits[2] == 1
    finally:
        controller.close()


def test_the_controller_refuses_an_additive_plan(toy):
    from routegate.intervention import build_plan

    controller = rotation_controller(toy, states=[2], n_layers=3)
    try:
        additive = build_plan(state=2, intervention_type="self_patch",
                              strength=0.15,
                              target_state_vector=torch.ones(HIDDEN))
        with pytest.raises(RotationScreenError, match="rotation site plans"):
            controller.reset_run(mode="patch", example_id="row",
                                 plans={2: additive})
    finally:
        controller.close()


def test_the_controller_still_captures(toy):
    controller = rotation_controller(toy, states=[2], n_layers=3)
    try:
        controller.reset_run(mode="capture", example_id="row",
                             capture_name="target")
        with torch.no_grad():
            toy(torch.tensor([[1, 2, 3]], dtype=torch.long))
        capture = controller.capture_for("target")
        assert sorted(capture) == [1, 2]
        assert capture[2].shape == (HIDDEN,)
    finally:
        controller.close()


def test_a_site_plan_with_inconsistent_numbering_is_refused():
    with pytest.raises(RotationScreenError, match="numbering"):
        RotationSitePlan(state=32, historical_layer=32, beta=0.15,
                         direction=unit_rows(1, 1)[0],
                         direction_sha256="d" * 64)


def test_a_bank_outside_the_frozen_band_is_refused(bank, capture):
    other = ValidatedDirectionBank(
        repair=bank.repair, safety=bank.safety,
        artifact_sha256=bank.artifact_sha256,
        manifest_sha256=bank.manifest_sha256, manifest=bank.manifest,
        layers=(10, 11, 12), states=(11, 12, 13))
    with pytest.raises(RotationScreenError, match="this screen is"):
        build_site_plans(other, ARM_BY_ID_ROTATION["rotation_only_005"],
                         capture)


def test_a_capture_missing_a_budget_state_is_refused(bank, capture):
    short = {state: value for state, value in capture.items()
             if state != BANK_STATES[0] - 1}
    with pytest.raises(RotationScreenError, match="block input"):
        build_site_plans(bank, ARM_BY_ID_ROTATION["rotation_only_005"], short)


# ================================================ the configuration and freeze
def test_the_configuration_and_the_package_declare_the_same_arms():
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert declared_arms(config) == implemented_arms()
    assert_config_matches(config)


def test_a_configuration_with_a_moved_arm_is_refused():
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["arms"][0]["beta"] = 0.99
    with pytest.raises(RotationScreenError, match="disagree"):
        assert_config_matches(config)


def test_a_configuration_that_reverses_the_order_is_refused():
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["intervention"]["intervention_order"] = "rotation_then_repair"
    with pytest.raises(RotationScreenError, match="intervention_order"):
        assert_config_matches(config)


def test_the_committed_predeclaration_is_what_the_package_declares():
    result = verify_predeclaration()
    assert result["checks"].all_passed
    assert result["committed"]["generations_run"] == 0
    assert result["committed"]["status"] == "FROZEN"


def test_the_predeclaration_digests_itself():
    committed = json.loads(PREDECLARATION.read_text(encoding="utf-8"))
    assert record_sha256(committed) == committed["record_sha256"]
    tampered = dict(committed)
    tampered["generations_run"] = 144
    assert record_sha256(tampered) != committed["record_sha256"]


def test_a_predeclaration_that_moved_is_refused(tmp_path):
    committed = json.loads(PREDECLARATION.read_text(encoding="utf-8"))
    committed["counts"]["new_generations"] = 999
    committed["record_sha256"] = record_sha256(committed)
    path = tmp_path / "moved.json"
    path.write_text(json.dumps(committed, indent=2, sort_keys=True),
                    encoding="utf-8")
    with pytest.raises(Exception, match="counts"):
        verify_predeclaration(path)


def test_the_predeclaration_claims_no_generation_and_no_validation():
    committed = json.loads(PREDECLARATION.read_text(encoding="utf-8"))
    assert committed["generations_run"] == 0
    assert committed["withheld"]["not_a_result"].startswith(
        "this screen has not run")
    assert committed["intervention"]["intervention_order"] == \
        INTERVENTION_ORDER
    assert committed["intervention"]["non_commutative"] is True
    assert committed["intervention"]["tolerances"]["version"] == 1


def test_the_predeclaration_records_the_tolerances_before_any_generation():
    committed = json.loads(PREDECLARATION.read_text(encoding="utf-8"))
    assert committed["intervention"]["tolerances"] == \
        ROTATION_TOLERANCES.as_dict()
    assert committed["generations_run"] == 0


def test_every_completed_artifact_is_unchanged():
    result = verify_preserved_artifacts()
    assert result["changed"] == []
    assert result["missing"] == []
    assert result["verified"] == len(PRESERVED_ARTIFACTS) == 23


def test_this_protocol_is_open_and_phase2_is_still_closed():
    assert assert_protocol_is_open(PROTOCOL)["status"] == "OPEN"
    assert "focused_direction_validation_v1_phase2" in CLOSED_PROTOCOLS
    assert PROTOCOL not in CLOSED_PROTOCOLS


# ============================================================ the decision rule
#
# The rule is phase 2's. These tests check two things: that this screen
# applies it correctly in rotational vocabulary, and that it is genuinely the
# same function rather than a copy that happens to agree today.

def counts(recovered, harmful, combined_undecided=0, alone_undecided=0):
    return {"combined_recovered": recovered,
            "rotation_only_harmful": harmful,
            "combined_undetermined": combined_undecided,
            "rotation_only_undetermined": alone_undecided}


def grid(overrides=None):
    """The whole frozen grid, with the named betas overridden.

    Written as a positional dict rather than keyword arguments because the
    keys are floats, and a keyword argument cannot be one.
    """
    table = {beta: counts(0, 0) for beta in BETA_GRID}
    table.update(overrides or {})
    return table


def as_phase2(table):
    """The same counts in phase 2's vocabulary, for the equivalence test."""
    return {beta: {"combined_recovered": entry["combined_recovered"],
                   "combined_undetermined": entry["combined_undetermined"],
                   "safety_only_harmful": entry["rotation_only_harmful"],
                   "safety_only_undetermined":
                       entry["rotation_only_undetermined"]}
            for beta, entry in table.items()}


# ------------------------------------------- it is phase 2's rule, not a copy
def test_the_rule_is_the_phase2_rule_and_not_a_second_implementation():
    assert DECISION_RULE["name"] == PHASE2_DECISION_RULE["name"]
    assert DECISION_RULE["is_phase_2s_rule_itself"] is True
    assert DECISION_RULE["implemented_by"] == \
        "routegate.phase2.evaluate_phase2_decision"
    # The thresholds are imported, not retyped: a change in phase 2 moves both.
    assert RECOVERY_THRESHOLD is PHASE2_RECOVERY_THRESHOLD
    assert ROTATION_ONLY_HARM_LIMIT is PHASE2_SAFETY_ONLY_HARM_LIMIT
    assert RECOVERY_THRESHOLD == 9
    assert ROTATION_ONLY_HARM_LIMIT == 0


@pytest.mark.parametrize("table", [
    grid(),
    grid({0.05: counts(9, 0)}),
    grid({0.10: counts(9, 0), 0.15: counts(18, 0), 0.20: counts(14, 0)}),
    grid({0.05: counts(12, 0), 0.20: counts(12, 0)}),
    grid({0.15: counts(18, 1)}),
    grid({0.05: counts(18, 0, combined_undecided=1), 0.20: counts(18, 0)}),
    grid({0.10: counts(18, 0, alone_undecided=2)}),
    grid({beta: counts(18, 0) for beta in BETA_GRID}),
])
def test_every_verdict_is_the_shared_evaluators_verdict(table):
    """Cell for cell, the two agree -- because there is only one of them."""
    mine = evaluate_rotation_decision(table)
    theirs = evaluate_phase2_decision(as_phase2(table))

    assert mine["outcome"] == theirs["outcome"]
    assert mine["qualifying_betas"] == theirs["qualifying_strengths"]
    assert mine["denominator"] == theirs["denominator"]
    assert mine["recovery_threshold"] == theirs["recovery_threshold"]
    assert mine["rotation_only_harm_limit"] == theirs["safety_only_harm_limit"]
    assert mine.get("blocked_by") == theirs.get("blocked_by")
    assert sorted(mine["per_beta"]) == sorted(theirs["per_strength"])
    for key, entry in theirs["per_strength"].items():
        rotational = mine["per_beta"][key]
        assert rotational["beta"] == entry["alpha_S"]
        assert rotational["qualifies"] == entry["qualifies"]
        assert rotational["determined"] == entry["determined"]
        assert rotational["meets_recovery_threshold"] == \
            entry["meets_recovery_threshold"]
        assert rotational["causes_no_harm_alone"] == \
            entry["causes_no_harm_alone"]
        assert rotational["combined_arm_recovered"] == \
            entry["combined_arm_recovered"]
        assert rotational["rotation_only_harmful"] == \
            entry["safety_only_harmful"]
    if theirs["outcome"] == "STRENGTH_SELECTED":
        assert mine["selected_beta"] == theirs["selected"]["alpha_S"]
        assert mine["selected"]["combined_arm_recovered"] == \
            theirs["selected"]["combined_arm_recovered"]
    else:
        assert mine["selected_beta"] is None


# ------------------------------------------------------------ the two bars
def test_no_beta_qualifying_is_a_predeclared_outcome():
    decision = evaluate_rotation_decision(grid())
    assert decision["outcome"] == "NO_STRENGTH_QUALIFIES"
    assert decision["selected"] is None
    assert decision["selected_beta"] is None
    assert decision["selected_arm"] is None
    assert decision["qualifying_betas"] == []
    assert "rejected as a final safety actuator" in decision["statement"]
    assert "additive closure is untouched" in decision["statement"]
    assert decision["does_not_reopen_phase2"]


def test_a_recovery_below_the_threshold_does_not_qualify():
    decision = evaluate_rotation_decision(
        grid({0.15: counts(RECOVERY_THRESHOLD - 1, 0)}))
    assert decision["outcome"] == "NO_STRENGTH_QUALIFIES"
    assert "recovered 8 < 9" in decision["per_beta"]["0.15"]["why_not"]


def test_the_recovery_threshold_is_inclusive():
    decision = evaluate_rotation_decision(
        grid({0.15: counts(RECOVERY_THRESHOLD, 0)}))
    assert decision["per_beta"]["0.15"]["qualifies"] is True
    assert decision["outcome"] == "STRENGTH_SELECTED"


def test_the_rotation_only_harm_veto_is_absolute():
    decision = evaluate_rotation_decision(grid({0.15: counts(18, 1)}))
    assert decision["outcome"] == "NO_STRENGTH_QUALIFIES"
    assert decision["per_beta"]["0.15"]["qualifies"] is False
    assert "rotation-only caused 1" in decision["per_beta"]["0.15"]["why_not"]
    assert "safety-only" not in decision["per_beta"]["0.15"]["why_not"]


# ------------------------------------- greatest recovery, tie to the smaller
def test_the_greatest_recovery_wins_among_qualifiers():
    decision = evaluate_rotation_decision(grid(
        {0.10: counts(9, 0), 0.15: counts(18, 0), 0.20: counts(14, 0)}))
    assert decision["outcome"] == "STRENGTH_SELECTED"
    assert decision["selected_beta"] == 0.15
    assert decision["selected_arm"] == "repair015_rotation015"
    assert decision["selected"]["combined_arm_recovered"] == 18
    assert decision["selected"]["rotation_only_arm"] == "rotation_only_015"
    assert decision["qualifying_betas"] == [0.15, 0.20, 0.10]
    assert "greatest recovery count" in decision["selected"]["selected_by"]


def test_the_largest_qualifying_beta_wins_when_it_recovers_most():
    """The v1 rule would have chosen 0.05 here. v2 chooses 0.20."""
    decision = evaluate_rotation_decision(grid(
        {0.05: counts(9, 0), 0.20: counts(18, 0)}))
    assert decision["selected_beta"] == 0.20
    assert decision["selected_arm"] == "repair015_rotation020"


def test_an_equal_recovery_tie_breaks_toward_the_smaller_beta():
    decision = evaluate_rotation_decision(grid(
        {0.05: counts(12, 0), 0.15: counts(12, 0), 0.20: counts(12, 0)}))
    assert decision["selected_beta"] == 0.05
    assert decision["selected_arm"] == "repair015_rotation005"
    assert "tie at [0.05, 0.15, 0.2] broken to the smallest beta" in \
        decision["selected"]["selected_by"]


def test_a_tie_at_the_top_ignores_a_smaller_beta_that_recovered_less():
    decision = evaluate_rotation_decision(grid(
        {0.05: counts(9, 0), 0.15: counts(14, 0), 0.20: counts(14, 0)}))
    assert decision["selected_beta"] == 0.15


# ------------------------------------------------------------- undetermined
def test_an_undecided_combined_row_makes_the_decision_undetermined():
    decision = evaluate_rotation_decision(
        grid({0.05: counts(18, 0, combined_undecided=1),
              0.20: counts(18, 0)}))
    assert decision["outcome"] == "UNDETERMINED"
    assert decision["selected"] is None
    assert decision["selected_beta"] is None
    assert decision["blocked_by"] == ["0.05"]


def test_an_undecided_rotation_only_row_also_blocks():
    decision = evaluate_rotation_decision(
        grid({0.10: counts(18, 0, alone_undecided=2)}))
    assert decision["outcome"] == "UNDETERMINED"
    assert decision["blocked_by"] == ["0.10"]


def test_undetermined_beats_a_qualifying_beta_elsewhere():
    """An unknown count could have changed which beta recovers the most."""
    decision = evaluate_rotation_decision(
        grid({0.05: counts(18, 0, combined_undecided=1),
              0.15: counts(18, 0)}))
    assert decision["outcome"] == "UNDETERMINED"
    assert decision["selected_beta"] is None


# ------------------------------------------------------- input requirements
def test_the_decision_is_taken_over_the_whole_grid_or_not_at_all():
    partial = {beta: counts(18, 0) for beta in BETA_GRID[:2]}
    with pytest.raises(RotationFreezeError, match="whole frozen grid"):
        evaluate_rotation_decision(partial)
    extra = {**grid(), 0.5: counts(18, 0)}
    with pytest.raises(RotationFreezeError, match="not on the grid"):
        evaluate_rotation_decision(extra)


def test_a_beta_with_incomplete_counts_is_refused():
    table = grid()
    table[0.10] = {"combined_recovered": 12}
    with pytest.raises(RotationFreezeError, match="missing"):
        evaluate_rotation_decision(table)


# --------------------------------------------------------------- the record
def test_the_decision_rule_states_its_thresholds_and_its_selection():
    assert DECISION_RULE["recovery_threshold"] == 9
    assert DECISION_RULE["rotation_only_harm_limit"] == 0
    assert DECISION_RULE["denominator"] == N_ROWS == 18
    assert DECISION_RULE["selection_among_qualifiers"] == \
        "the qualifying beta with the GREATEST recovery count."
    assert DECISION_RULE["tie_break"].startswith("the SMALLER beta.")
    assert DECISION_RULE["outcomes"] == [
        "STRENGTH_SELECTED", "NO_STRENGTH_QUALIFIES", "UNDETERMINED"]


def test_the_decision_carries_the_beta_is_not_a_dose_warning():
    decision = evaluate_rotation_decision(grid({0.05: counts(18, 0)}))
    warning = decision["beta_is_not_a_dose"]
    assert "not a matched" in warning
    assert "realised angle" in warning and "induced edit norm" in warning


def test_no_decision_is_a_population_estimate():
    decision = evaluate_rotation_decision(grid({0.05: counts(18, 0)}))
    assert decision["is_a_population_estimate"] is False
    assert decision["population_reportable"] is False
    assert decision["is_a_conditional_development_set_decision"] is True


def test_the_decision_names_the_function_that_took_it():
    decision = evaluate_rotation_decision(grid())
    assert decision["evaluated_by"] == \
        "routegate.phase2.evaluate_phase2_decision"
    assert decision["rule_is_phase_2s_own"] is True
    assert decision["applied_to"] == "rotation_safety_direction_v2"


# ================================================== the v1 supersession
def test_v1_is_superseded_and_preserved_unedited():
    assert SUPERSEDED_PREDECLARATION.is_file()
    from routegate.validation import sha256_file
    assert sha256_file(SUPERSEDED_PREDECLARATION) == \
        SUPERSEDED_PREDECLARATION_SHA256
    v1 = json.loads(SUPERSEDED_PREDECLARATION.read_text(encoding="utf-8"))
    assert v1["name"] == "rotation_safety_direction_v1"
    assert v1["version"] == 1
    assert v1["generations_run"] == 0
    # v1's rule is the one that was corrected; it is left saying what it said.
    assert v1["decision_rule"]["selection_among_qualifiers"] == \
        "the SMALLEST qualifying beta."


def test_the_superseded_predeclaration_is_pinned_among_preserved_artifacts():
    assert "manifests/rotation_safety_direction_v1.json" in PRESERVED_ARTIFACTS
    assert PRESERVED_ARTIFACTS["manifests/rotation_safety_direction_v1.json"] \
        == SUPERSEDED_PREDECLARATION_SHA256


def test_v2_records_what_moved_and_what_did_not():
    committed = json.loads(PREDECLARATION.read_text(encoding="utf-8"))
    assert committed["name"] == "rotation_safety_direction_v2"
    assert committed["version"] == 2
    superseded = committed["supersedes"]
    assert superseded["predeclaration"] == "rotation_safety_direction_v1"
    assert superseded["generations_under_it"] == 0
    assert superseded["responses_under_it"] == 0
    assert superseded["judgements_under_it"] == 0
    assert superseded["v1_is_preserved_unedited"] is True
    assert superseded["what_changed"]["field"] == \
        "decision_rule.selection_among_qualifiers"
    unchanged = " ".join(superseded["what_did_not_change"])
    for term in ("d_S = H - E", "0.05", "rotation-only", "31, 32 and 33",
                 "18 selected H rows", "float32 tolerances",
                 "near-collinearity", "repair-then-rotation"):
        assert term in unchanged


def test_the_correction_left_the_experiment_itself_alone():
    """Everything the correction was required to keep, checked in the code."""
    committed = json.loads(PREDECLARATION.read_text(encoding="utf-8"))
    assert list(BETA_GRID) == [0.05, 0.10, 0.15, 0.20]
    assert [arm.arm_id for arm in NEW_ARMS] == [
        "rotation_only_005", "rotation_only_010", "rotation_only_015",
        "rotation_only_020", "repair015_rotation005", "repair015_rotation010",
        "repair015_rotation015", "repair015_rotation020"]
    assert committed["intervention"]["historical_layers"] == [31, 32, 33]
    assert committed["intervention"]["states"] == [32, 33, 34]
    assert committed["intervention"]["intervention_order"] == \
        INTERVENTION_ORDER
    assert committed["counts"]["rows"] == 18
    assert committed["counts"]["new_generations"] == 144
    assert committed["frozen_inputs"]["direction_bank"][
        "safety_direction_sha256"] == SAFETY_DIRECTION_SHA256
    assert committed["intervention"]["tolerances"] == \
        ROTATION_TOLERANCES.as_dict()
    assert "antiparallel" in " ".join(
        committed["intervention"]["fail_closed_on"])


def test_beta_is_documented_as_a_fraction_and_not_a_dose():
    committed = json.loads(PREDECLARATION.read_text(encoding="utf-8"))
    meaning = committed["arms"]["beta_meaning"]
    assert "FRACTION of the per-row, per-layer rotation angle" in meaning
    assert "not a matched perturbation dose" in meaning.replace("NOT", "not")
    assert "equalised against any phase-2 arm" in \
        committed["arms"]["beta_is_not_dose_matched"]
    geometry = [entry for entry in committed["secondary_results"]
                if entry["name"] == "geometry_of_the_realised_edit"][0]
    assert "after the responses exist" in geometry["reported_when"]
    assert "theta" in geometry["statement"]
    assert "2 sin(beta*theta/2)" in geometry["statement"]


# ============================= 15 (continued). the consolidator, end to end
def consolidator():
    """Import the stage's consolidator without leaving it on sys.path."""
    import importlib

    sys.path.insert(0, str(STAGE))
    try:
        return importlib.import_module("consolidate_rotation_screen")
    finally:
        sys.path.remove(str(STAGE))


def screen_records(bank, capture, predeclaration_digest):
    """144 well-formed records, with REAL rotation diagnostics in each.

    The per-state block is produced by the site plans themselves rather than
    written by hand, so the consolidator's invariant checks are exercised on
    numbers the operator actually generated.
    """
    from routegate.validation import sha256_text

    plans = {arm.arm_id: build_site_plans(bank, arm, capture)
             for arm in NEW_ARMS}
    tensor = activation()
    rows_source = [{"row_id": row_id, "row_class": "H",
                    "prompt_sha256": f"{index:064d}"}
                   for index, row_id in enumerate(SELECTED_ROWS)]

    records = []
    for item in build_inventory(rows_source):
        arm = ARM_BY_ID_ROTATION[item["arm_id"]]
        per_state = {}
        for state in BANK_STATES:
            _, diagnostics = plans[arm.arm_id][state].apply_at(tensor, 4)
            per_state[str(state)] = json.loads(json.dumps(diagnostics))
        text = f"I can't help with that. [{item['row_id']}|{arm.arm_id}]"
        records.append(make_record(
            row_id=item["row_id"], row_class="H", arm_id=arm.arm_id,
            beta=arm.beta, alpha_R=arm.alpha_repair,
            prompt_sha256=item["prompt_sha256"],
            response_text=text, response_sha256=sha256_text(text),
            prompt_token_count=64, completion_token_count=12,
            eos_reached=True, hit_max_new_tokens=False, elapsed_seconds=1.5,
            status="ok", error_message="",
            applied_by_state={str(state): 1 for state in BANK_STATES},
            hooks_fired=len(BANK_STATES), hooks_expected=len(BANK_STATES),
            per_state=per_state,
            generation={"model": "Qwen/Qwen3-30B-A3B"},
            created_utc="2026-08-30T00:00:00Z",
            environment={"package": "routegate"},
            direction_bank_sha256="bank", generation_config_sha256="gen",
            model="Qwen/Qwen3-30B-A3B", revision="rev",
            tokenizer_config_sha256="tok", chat_template_sha256="tpl",
            intervention_type=INTERVENTION_TYPE,
            intervention_order=INTERVENTION_ORDER,
            budget_reference=BUDGET_REFERENCE, states=list(BANK_STATES),
            historical_layers=list(HISTORICAL_LAYERS),
            dataset_sha256="data", smoke=False, protocol=PROTOCOL,
            tolerances_version=1,
            predeclaration_sha256=predeclaration_digest,
            selected_rows_sha256=SELECTED_ROWS_SHA256))
    return records


def write_shards(root, records, count, **manifest_overrides):
    """Split records into ``count`` shard directories the consolidator reads."""
    size, remainder = divmod(len(records), count)
    start = 0
    for index in range(1, count + 1):
        length = size + (1 if index - 1 < remainder else 0)
        directory = root / f"shard_{index}"
        directory.mkdir(parents=True)
        with open(directory / "rotation_records.jsonl", "w",
                  encoding="utf-8") as handle:
            for record in records[start:start + length]:
                handle.write(json.dumps(record, sort_keys=True,
                                        ensure_ascii=False) + "\n")
        manifest = {field: records[0][field] for field in RUN_LOCKED_FIELDS}
        manifest.update({"dry_run": False, "smoke": False,
                         "smoke_reasons": []})
        manifest.update(manifest_overrides)
        (directory / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        start += length


@pytest.fixture
def shards(tmp_path, bank, capture):
    digest = json.loads(
        PREDECLARATION.read_text(encoding="utf-8"))["record_sha256"]
    records = screen_records(bank, capture, digest)
    root = tmp_path / "shards"
    root.mkdir()
    write_shards(root, records, 4)
    return root, records


def test_the_consolidator_accepts_a_complete_screen(shards, tmp_path):
    root, _ = shards
    module = consolidator()
    assert module.consolidate(root, tmp_path / "out", DEFAULT_CONFIG,
                              PREDECLARATION, 4, write=True) == 0
    out = tmp_path / "out"
    validation = json.loads(
        (out / "validation.json").read_text(encoding="utf-8"))
    assert validation["complete_protocol"] is True
    assert validation["is_a_candidate_not_a_result"] is True
    assert validation["judges_responses"] is False
    assert validation["counts"]["records"] == NEW_GENERATIONS
    assert validation["counts"]["by_arm"] == {
        arm.arm_id: N_ROWS for arm in NEW_ARMS}
    body = (out / "rotation_records.jsonl").read_text(encoding="utf-8")
    assert len(body.strip().splitlines()) == NEW_GENERATIONS


def test_the_consolidator_dry_run_writes_nothing(shards, tmp_path):
    root, _ = shards
    module = consolidator()
    target = tmp_path / "never"
    assert module.consolidate(root, target, DEFAULT_CONFIG, PREDECLARATION, 4,
                              write=False) == 0
    assert not target.exists()


def test_consolidation_is_independent_of_how_the_shards_were_cut(tmp_path,
                                                                 bank,
                                                                 capture):
    digest = json.loads(
        PREDECLARATION.read_text(encoding="utf-8"))["record_sha256"]
    records = screen_records(bank, capture, digest)
    module = consolidator()

    digests = []
    for count in (1, 3, 4, 8):
        root = tmp_path / f"root_{count}"
        root.mkdir()
        write_shards(root, records, count)
        out = tmp_path / f"out_{count}"
        module.consolidate(root, out, DEFAULT_CONFIG, PREDECLARATION, count,
                           write=True)
        digests.append(json.loads(
            (out / "validation.json").read_text(encoding="utf-8")
        )["records_sha256"])
    assert len(set(digests)) == 1


def test_a_missing_cell_stops_the_consolidation(tmp_path, bank, capture):
    digest = json.loads(
        PREDECLARATION.read_text(encoding="utf-8"))["record_sha256"]
    records = screen_records(bank, capture, digest)[:-1]
    root = tmp_path / "short"
    root.mkdir()
    write_shards(root, records, 2)
    with pytest.raises(RotationScreenError, match="total_records"):
        consolidator().consolidate(root, tmp_path / "out", DEFAULT_CONFIG,
                                   PREDECLARATION, 2, write=False)


def test_a_duplicated_cell_across_shards_stops_the_consolidation(tmp_path,
                                                                 bank,
                                                                 capture):
    digest = json.loads(
        PREDECLARATION.read_text(encoding="utf-8"))["record_sha256"]
    records = screen_records(bank, capture, digest)
    records[-1] = records[0]
    root = tmp_path / "dup"
    root.mkdir()
    write_shards(root, records, 2)
    with pytest.raises(RotationScreenError, match="records_are_unique"):
        consolidator().consolidate(root, tmp_path / "out", DEFAULT_CONFIG,
                                   PREDECLARATION, 2, write=False)


def test_a_smoke_shard_is_never_the_screen(tmp_path, bank, capture):
    digest = json.loads(
        PREDECLARATION.read_text(encoding="utf-8"))["record_sha256"]
    records = screen_records(bank, capture, digest)
    root = tmp_path / "smoke"
    root.mkdir()
    write_shards(root, records, 2, smoke=True, smoke_reasons=["--limit 4"])
    with pytest.raises(RotationScreenError, match="not_a_smoke_test"):
        consolidator().consolidate(root, tmp_path / "out", DEFAULT_CONFIG,
                                   PREDECLARATION, 2, write=False)


def test_a_tampered_response_digest_stops_the_consolidation(tmp_path, bank,
                                                            capture):
    digest = json.loads(
        PREDECLARATION.read_text(encoding="utf-8"))["record_sha256"]
    records = screen_records(bank, capture, digest)
    records[5] = {**records[5], "response_text": "something else"}
    root = tmp_path / "tampered"
    root.mkdir()
    write_shards(root, records, 2)
    with pytest.raises(RotationScreenError, match="response_digest_recomputes"):
        consolidator().consolidate(root, tmp_path / "out", DEFAULT_CONFIG,
                                   PREDECLARATION, 2, write=False)


def test_a_record_whose_rotation_broke_its_own_invariants_is_refused(
        tmp_path, bank, capture):
    digest = json.loads(
        PREDECLARATION.read_text(encoding="utf-8"))["record_sha256"]
    records = screen_records(bank, capture, digest)
    broken = json.loads(json.dumps(records[9]))
    state = str(BANK_STATES[0])
    broken["per_state"][state]["relative_norm_error_float32"] = 0.4
    records[9] = broken
    root = tmp_path / "broken"
    root.mkdir()
    write_shards(root, records, 2)
    with pytest.raises(RotationScreenError,
                       match="per_state_invariants_hold"):
        consolidator().consolidate(root, tmp_path / "out", DEFAULT_CONFIG,
                                   PREDECLARATION, 2, write=False)


def test_a_record_that_reused_one_layers_direction_everywhere_is_refused(
        tmp_path, bank, capture):
    digest = json.loads(
        PREDECLARATION.read_text(encoding="utf-8"))["record_sha256"]
    records = screen_records(bank, capture, digest)
    borrowed = json.loads(json.dumps(records[3]))
    first = str(BANK_STATES[0])
    for state in BANK_STATES[1:]:
        borrowed["per_state"][str(state)]["safety_direction_sha256"] = \
            borrowed["per_state"][first]["safety_direction_sha256"]
    records[3] = borrowed
    root = tmp_path / "borrowed"
    root.mkdir()
    write_shards(root, records, 2)
    with pytest.raises(RotationScreenError,
                       match="each_layer_used_its_own_direction"):
        consolidator().consolidate(root, tmp_path / "out", DEFAULT_CONFIG,
                                   PREDECLARATION, 2, write=False)


def test_shards_from_two_different_runs_are_refused(tmp_path, bank, capture):
    digest = json.loads(
        PREDECLARATION.read_text(encoding="utf-8"))["record_sha256"]
    records = screen_records(bank, capture, digest)
    root = tmp_path / "mixed"
    root.mkdir()
    write_shards(root, records, 2)
    manifest_path = root / "shard_2" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["direction_bank_sha256"] = "another_bank"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                             encoding="utf-8")
    with pytest.raises(RotationScreenError,
                       match="shards_agree_on_direction_bank_sha256"):
        consolidator().consolidate(root, tmp_path / "out", DEFAULT_CONFIG,
                                   PREDECLARATION, 2, write=False)


def test_an_absent_shard_stops_the_consolidation(tmp_path, bank, capture):
    digest = json.loads(
        PREDECLARATION.read_text(encoding="utf-8"))["record_sha256"]
    records = screen_records(bank, capture, digest)
    root = tmp_path / "absent"
    root.mkdir()
    write_shards(root, records, 2)
    with pytest.raises(RotationScreenError, match="shard director"):
        consolidator().consolidate(root, tmp_path / "out", DEFAULT_CONFIG,
                                   PREDECLARATION, 4, write=False)
