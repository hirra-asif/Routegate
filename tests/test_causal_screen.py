"""The dual-direction causal screen, on synthetic tensors and a stub model.

Nothing here loads a model, allocates a GPU, calls an API, judges a response or
reads the sealed set. The model is a stub whose only job is to let the hook
controller fire; every direction and activation is invented.

Organised by the failure each group is there to prevent:

    1. arithmetic   the combined edit is not the formula the screen declares
    2. arms         a coefficient, a layer or an arm silently differs
    3. inventory    the screen runs short and reports a clean smaller number
    4. bank         a direction bank nobody pinned is steered with
    5. records      a resume accepts a record made under other conditions
    6. abstinence   a model is loaded on a dry run, or the sealed set is read
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate import dualsteer as DS                                # noqa: E402
from routegate.directionbank import BANK_STATES, HISTORICAL_LAYERS   # noqa: E402
from routegate.dualsteer import DualSteerError                       # noqa: E402
from routegate.intervention import budget_reference_state            # noqa: E402
from routegate.steering import build_steering_plan                   # noqa: E402

HIDDEN = 2048
SEED = 20260822


def _load_script(name: str):
    path = REPO_ROOT / "experiments" / "dual_direction_controller" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_ddc_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _code_of(name: str) -> str:
    """A runner's executable source, with every docstring removed."""
    import ast
    path = REPO_ROOT / "experiments" / "dual_direction_controller" / f"{name}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)) and ast.get_docstring(node):
            node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(ast.fix_missing_locations(tree))


RUN = _load_script("run_causal_screen")
CONFIG = json.loads(
    (REPO_ROOT / "configs" / "dual_direction_controller.json").read_text("utf-8"))


# ---------------------------------------------------------------- fixtures
def _unit(rng, n=HIDDEN):
    vector = rng.standard_normal(n).astype(np.float32)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def _bank_arrays(seed=SEED):
    rng = np.random.default_rng(seed)
    repair = np.stack([_unit(rng) for _ in HISTORICAL_LAYERS])
    safety = np.stack([_unit(rng) for _ in HISTORICAL_LAYERS])
    return repair.astype(np.float32), safety.astype(np.float32)


def _reference(seed=SEED, scale=7.5):
    rng = np.random.default_rng(seed + 1)
    return torch.from_numpy(
        (rng.standard_normal(HIDDEN).astype(np.float32) * scale))


def _recover_coefficients(delta, repair, safety, reference_norm):
    """The (alpha_R, alpha_S) a delta was built from, by least squares.

    The two directions are not orthogonal, so ``dot(delta, d_R)`` alone is the
    repair coefficient plus a share of the safety one. Solving the 2x2 Gram
    system separates them.
    """
    d_r = torch.from_numpy(np.ascontiguousarray(repair)).float()
    d_s = torch.from_numpy(np.ascontiguousarray(safety)).float()
    gram = torch.tensor(
        [[float(torch.dot(d_r, d_r)), float(torch.dot(d_r, d_s))],
         [float(torch.dot(d_s, d_r)), float(torch.dot(d_s, d_s))]],
        dtype=torch.float64)
    rhs = torch.tensor([float(torch.dot(delta, d_r)) / reference_norm,
                        float(torch.dot(delta, d_s)) / reference_norm],
                       dtype=torch.float64)
    solved = torch.linalg.solve(gram, rhs)
    return float(solved[0]), float(solved[1])


def _rows(n_e=DS.N_E_ROWS, n_h=DS.N_H_ROWS):
    rows = []
    for index in range(n_e):
        rows.append({"row_id": f"e:{index:04d}", "row_class": DS.CLASS_E,
                     "prompt_sha256": f"{index:064x}"})
    for index in range(n_h):
        rows.append({"row_id": f"h:{index:04d}", "row_class": DS.CLASS_H,
                     "prompt_sha256": f"{index + 1000:064x}"})
    return rows


# ============================================================ 1. arithmetic
def test_the_combined_edit_is_exactly_the_declared_formula():
    repair, safety = _bank_arrays()
    reference = _reference()
    alpha_r, alpha_s = 0.15, 0.10

    plan = DS.build_dual_plan(state=BANK_STATES[0], repair=repair[0],
                              safety=safety[0], alpha_repair=alpha_r,
                              alpha_safety=alpha_s, reference_vector=reference)
    expected = torch.from_numpy(
        alpha_r * repair[0] + alpha_s * safety[0]).float()
    expected = expected * float(torch.linalg.vector_norm(reference.float()))
    assert torch.allclose(plan.delta, expected, rtol=0, atol=1e-6)
    assert plan.status == "ok"


def test_the_combined_delta_is_not_renormalised():
    """The two coefficients must not compete for one budget."""
    repair, safety = _bank_arrays()
    reference = _reference()
    norm = float(torch.linalg.vector_norm(reference.float()))

    weaker = DS.build_dual_plan(BANK_STATES[0], repair[0], safety[0],
                                0.15, 0.05, reference)
    stronger = DS.build_dual_plan(BANK_STATES[0], repair[0], safety[0],
                                  0.15, 0.20, reference)
    # raising alpha_S raises the length of the edit; it does not rebalance it
    assert stronger.applied_delta_norm > weaker.applied_delta_norm
    # and the repair coefficient is untouched by alpha_S. Recovered by solving
    # the two-direction system: d_R and d_S are not orthogonal, so projecting
    # on d_R alone would pick up part of the safety term.
    for plan in (weaker, stronger):
        recovered = _recover_coefficients(plan.delta, repair[0], safety[0],
                                          norm)
        assert abs(recovered[0] - 0.15) < 1e-4


def test_a_numpy_reference_is_accepted_as_well_as_a_tensor():
    """The pre-flight convention check hands it numpy; the run hands it a
    captured tensor. Both are the same edit."""
    repair, safety = _bank_arrays()
    reference = _reference()
    as_tensor = DS.build_dual_plan(BANK_STATES[0], repair[0], safety[0],
                                   0.15, 0.10, reference)
    as_numpy = DS.build_dual_plan(BANK_STATES[0], repair[0], safety[0],
                                  0.15, 0.10, reference.numpy())
    assert torch.equal(as_tensor.delta, as_numpy.delta)
    assert as_tensor.reference_norm == as_numpy.reference_norm


def test_zero_coefficients_apply_no_edit_at_all():
    repair, safety = _bank_arrays()
    plan = DS.build_dual_plan(BANK_STATES[0], repair[0], safety[0],
                              0.0, 0.0, _reference())
    assert plan.delta is None
    assert plan.applies_anything is False
    assert plan.status == "no_intervention"
    assert plan.applied_delta_norm == 0.0


@pytest.mark.parametrize("alpha_r,alpha_s", [(0.15, 0.0), (0.0, 0.15),
                                             (0.15, 0.10)])
def test_repair_only_safety_only_and_combined_each_move_where_they_should(
        alpha_r, alpha_s):
    repair, safety = _bank_arrays()
    reference = _reference()
    norm = float(torch.linalg.vector_norm(reference.float()))
    plan = DS.build_dual_plan(BANK_STATES[0], repair[0], safety[0],
                              alpha_r, alpha_s, reference)
    recovered = _recover_coefficients(plan.delta, repair[0], safety[0], norm)
    assert abs(recovered[0] - alpha_r) < 1e-4
    assert abs(recovered[1] - alpha_s) < 1e-4


def test_with_no_safety_the_edit_is_the_existing_actuator_edit():
    repair, _ = _bank_arrays()
    reference = _reference()
    mine = DS.build_dual_plan(BANK_STATES[0], repair[0],
                              np.zeros(HIDDEN, dtype=np.float32),
                              0.15, 0.0, reference)
    theirs = build_steering_plan(state=BANK_STATES[0], direction=repair[0],
                                 strength=0.15, reference_vector=reference,
                                 budget_reference=DS.BUDGET_REFERENCE)
    assert torch.allclose(mine.delta, theirs.delta, rtol=1e-6, atol=1e-8)
    assert mine.budget_reference_state == theirs.budget_reference_state
    measured = DS.assert_matches_actuator_convention(repair[0], reference)
    assert measured["max_abs_difference"] <= measured["tolerated"]


def test_a_convention_that_has_moved_stops_the_screen(monkeypatch):
    repair, _ = _bank_arrays()

    def halved(**kwargs):
        plan = build_steering_plan(**kwargs)
        plan.delta = plan.delta * 0.5
        return plan

    monkeypatch.setattr("routegate.steering.build_steering_plan", halved)
    with pytest.raises(DualSteerError, match="parted company"):
        DS.assert_matches_actuator_convention(repair[0], _reference())


def test_the_budget_reference_is_the_preceding_state_not_the_edited_one():
    for state in BANK_STATES:
        reference_state, _ = budget_reference_state(state,
                                                    DS.BUDGET_REFERENCE)
        assert reference_state == state - 1


@pytest.mark.parametrize("bad", [-0.05, -1.0])
def test_a_negative_coefficient_is_refused(bad):
    repair, safety = _bank_arrays()
    with pytest.raises(DualSteerError, match="non-negative"):
        DS.build_dual_plan(BANK_STATES[0], repair[0], safety[0], bad, 0.0,
                           _reference())


def test_a_non_finite_direction_is_refused():
    repair, safety = _bank_arrays()
    repair[0][3] = np.nan
    with pytest.raises(DualSteerError, match="non-finite"):
        DS.build_dual_plan(BANK_STATES[0], repair[0], safety[0], 0.15, 0.0,
                           _reference())


# ================================================================= 2. arms
def test_the_same_two_coefficients_are_used_at_all_three_layers():
    repair, safety = _bank_arrays()
    capture = {state - 1: _reference(seed=state) for state in BANK_STATES}
    bank = DS.ValidatedDirectionBank(repair=repair, safety=safety,
                                     artifact_sha256="a" * 64,
                                     manifest_sha256="b" * 64, manifest={})
    arm = DS.ARM_BY_ID["repair015_safety010"]
    plans = DS.build_plans(bank, arm, capture)

    assert sorted(plans) == list(BANK_STATES)
    for index, state in enumerate(BANK_STATES):
        norm = float(torch.linalg.vector_norm(capture[state - 1].float()))
        expected = torch.from_numpy(
            arm.alpha_repair * repair[index]
            + arm.alpha_safety * safety[index]).float() * norm
        assert torch.allclose(plans[state].delta, expected, rtol=0, atol=1e-5)
        # the directions differ by layer; the coefficients do not
        assert plans[state].strength == arm.alpha_repair + arm.alpha_safety


def test_every_layer_uses_its_own_direction():
    repair, safety = _bank_arrays()
    assert not np.array_equal(repair[0], repair[1])
    bank = DS.ValidatedDirectionBank(repair=repair, safety=safety,
                                     artifact_sha256="a" * 64,
                                     manifest_sha256="b" * 64, manifest={})
    for index, state in enumerate(BANK_STATES):
        got_r, got_s = bank.directions_for_state(state)
        assert np.array_equal(got_r, repair[index])
        assert np.array_equal(got_s, safety[index])
    with pytest.raises(DualSteerError, match="three-layer bank"):
        bank.directions_for_state(45)


def test_the_predeclared_arms_are_exactly_the_configured_ones():
    declared = CONFIG["causal_screen"]["arms"]
    for name in DS.SCREEN_CLASSES:
        configured = [(arm["arm_id"], arm["alpha_R"], arm["alpha_S"])
                      for arm in declared[name]]
        implemented = [(arm.arm_id, arm.alpha_repair, arm.alpha_safety)
                       for arm in DS.ARMS_BY_CLASS[name]]
        assert configured == implemented, name
    assert len(DS.ARMS_BY_CLASS[DS.CLASS_E]) == 5
    assert len(DS.ARMS_BY_CLASS[DS.CLASS_H]) == 13


def test_the_combined_arms_hold_the_repair_coefficient_fixed():
    """Question three depends on this and on nothing else."""
    combined = [arm for arm in DS.ARMS_BY_CLASS[DS.CLASS_H]
                if arm.alpha_repair > 0 and arm.alpha_safety > 0]
    assert len(combined) == 4
    assert {arm.alpha_repair for arm in combined} == {DS.COMBINED_ALPHA_REPAIR}
    assert sorted(arm.alpha_safety for arm in combined) == [0.05, 0.10, 0.15,
                                                            0.20]


def test_the_screen_covers_only_e_and_h():
    assert DS.SCREEN_CLASSES == ("E", "H")
    with pytest.raises(DualSteerError, match="B rows built"):
        DS.arms_for("B")


def test_the_layers_and_states_are_the_frozen_band():
    assert HISTORICAL_LAYERS == (31, 32, 33)
    assert BANK_STATES == (32, 33, 34)
    screen = CONFIG["causal_screen"]["intervention"]
    assert screen["historical_layers"] == list(HISTORICAL_LAYERS)
    assert screen["routegate_states"] == list(BANK_STATES)
    assert screen["combined_delta_renormalised"] is False
    assert screen["orthogonalisation_applied"] is False
    assert screen["prefill_only"] is True
    assert screen["generated_token_positions_are_never_edited"] is True


# ============================================================ 3. inventory
def test_the_inventory_is_exactly_three_thousand_two_hundred_and_fourteen():
    items = DS.build_inventory(_rows())
    counts = DS.assert_inventory(items)
    assert counts["total"] == DS.EXPECTED_GENERATIONS == 3214
    assert counts["E_generations"] == 89 * 5 == 445
    assert counts["H_generations"] == 213 * 13 == 2769
    assert counts["E_generations"] + counts["H_generations"] == 3214
    assert CONFIG["causal_screen"]["inventory"]["total"] == 3214


@pytest.mark.parametrize("n_e,n_h", [(88, 213), (89, 212), (90, 213)])
def test_a_corpus_of_the_wrong_size_is_refused(n_e, n_h):
    items = DS.build_inventory(_rows(n_e, n_h))
    with pytest.raises(DualSteerError, match="this screen is"):
        DS.assert_inventory(items)


def test_the_inventory_order_is_deterministic():
    rows = _rows()
    first = DS.build_inventory(rows)
    shuffled = list(reversed(rows))
    second = DS.build_inventory(shuffled)
    assert [(item["row_id"], item["arm_id"]) for item in first] == [
        (item["row_id"], item["arm_id"]) for item in second]
    # E before H, then row id, then the arm's declared position
    assert first[0]["row_class"] == DS.CLASS_E
    assert first[-1]["row_class"] == DS.CLASS_H
    assert [item["arm_id"] for item in first[:5]] == [
        arm.arm_id for arm in DS.ARMS_BY_CLASS[DS.CLASS_E]]


def test_a_duplicated_row_is_refused():
    rows = _rows()
    rows.append(dict(rows[0]))
    items = DS.build_inventory(rows)
    with pytest.raises(DualSteerError, match="duplicated"):
        DS.assert_inventory(items)


def test_the_shards_partition_the_inventory_exactly():
    items = DS.build_inventory(_rows())
    for count in (1, 4, 8, 7):
        pieces = [RUN.shard(items, index, count)
                  for index in range(1, count + 1)]
        assert sum(len(piece) for piece in pieces) == len(items)
        flat = [item for piece in pieces for item in piece]
        assert flat == items
    with pytest.raises(DualSteerError, match="outside 1"):
        RUN.shard(items, 0, 4)


# ================================================================= 4. bank
def _write_bank(tmp_path, repair=None, safety=None, orthogonalised=False):
    from routegate.directionbank import npz_bytes
    if repair is None or safety is None:
        repair, safety = _bank_arrays()
    directory = tmp_path / "bank"
    directory.mkdir(parents=True, exist_ok=True)
    payload = npz_bytes({"repair_direction": repair,
                         "safety_direction": safety})
    (directory / "validated_direction_bank.npz").write_bytes(payload)
    (directory / "direction_bank_manifest.json").write_text(json.dumps({
        "orthogonalisation": {"applied": orthogonalised}}), encoding="utf-8")
    import hashlib
    return directory, hashlib.sha256(payload).hexdigest()


def test_a_bank_whose_digest_is_not_the_pinned_one_is_refused(tmp_path):
    directory, digest = _write_bank(tmp_path)
    bank = DS.load_validated_bank(directory, digest)
    assert bank.artifact_sha256 == digest
    with pytest.raises(DualSteerError, match="is refused, not adopted"):
        DS.load_validated_bank(directory, "f" * 64)


def test_the_configured_pin_is_the_one_the_screen_uses():
    pinned = CONFIG["causal_screen"]["direction_bank"]["artifact_sha256"]
    assert pinned == (
        "d759574b5be630ea5170fb440f8663d8c4bfaa32b479ce5cf641b6dbd73516bb")
    assert len(pinned) == 64


def test_a_non_unit_direction_is_refused(tmp_path):
    repair, safety = _bank_arrays()
    repair[1] = repair[1] * 2.0
    directory, digest = _write_bank(tmp_path, repair, safety)
    with pytest.raises(DualSteerError, match="not unit norm"):
        DS.load_validated_bank(directory, digest)


def test_an_orthogonalised_bank_is_refused(tmp_path):
    directory, digest = _write_bank(tmp_path, orthogonalised=True)
    with pytest.raises(DualSteerError, match="no orthogonalisation"):
        DS.load_validated_bank(directory, digest)


def test_an_absent_bank_is_refused(tmp_path):
    with pytest.raises(DualSteerError, match="is not a file"):
        DS.load_validated_bank(tmp_path / "nowhere", "a" * 64)


# ============================================================== 5. records
def _record(**overrides):
    base = {
        "row_id": "e:0001", "row_class": DS.CLASS_E, "arm_id": "repair_015",
        "alpha_R": 0.15, "alpha_S": 0.0,
        "direction_bank_sha256": "a" * 64,
        "generation_config_sha256": "b" * 64,
        "model": "Qwen/Qwen3-30B-A3B", "revision": "c" * 40,
        "tokenizer_config_sha256": "d" * 64, "chat_template_sha256": "e" * 64,
        "intervention_type": DS.INTERVENTION_TYPE,
        "budget_reference": DS.BUDGET_REFERENCE,
        "states": list(BANK_STATES), "historical_layers": list(HISTORICAL_LAYERS),
        "dataset_sha256": "f" * 64, "smoke": False,
        "protocol": DS.DEFAULT_PROTOCOL,
    }
    base.update(overrides)
    return DS.make_record(**base)


def test_the_resume_key_binds_every_condition_that_could_differ():
    assert DS.KEY_FIELDS == (
        "row_id", "row_class", "arm_id", "alpha_R", "alpha_S",
        "direction_bank_sha256", "generation_config_sha256")
    assert list(CONFIG["causal_screen"]["resume_key"]) == list(DS.KEY_FIELDS)

    base = _record()
    for field, value in (("row_id", "e:0002"), ("row_class", DS.CLASS_H),
                         ("arm_id", "repair_010"), ("alpha_R", 0.10),
                         ("alpha_S", 0.05),
                         ("direction_bank_sha256", "9" * 64),
                         ("generation_config_sha256", "8" * 64)):
        assert DS.record_key(_record(**{field: value})) != DS.record_key(base)


def test_a_float_that_only_differs_by_rounding_is_the_same_arm():
    assert DS.record_key(_record(alpha_R=0.15)) == DS.record_key(
        _record(alpha_R=0.15000000000000002))


def test_a_completed_generation_is_not_run_again(tmp_path):
    inventory = DS.build_inventory(_rows(2, 2))
    done = {DS.work_key(inventory[0], "a" * 64, "b" * 64)}
    todo = DS.plan_work(inventory, done, "a" * 64, "b" * 64)
    assert len(todo) == len(inventory) - 1
    assert all(item != inventory[0] for item in todo)
    # a different bank means nothing is complete
    assert len(DS.plan_work(inventory, done, "9" * 64, "b" * 64)) == len(
        inventory)


def test_a_duplicate_record_is_refused(tmp_path):
    expected = {"direction_bank_sha256": "a" * 64}
    with pytest.raises(DualSteerError, match="repeats key"):
        DS.verify_consistency([_record(), _record()], expected)


@pytest.mark.parametrize("field,value", [
    ("direction_bank_sha256", "9" * 64),
    ("generation_config_sha256", "8" * 64),
    ("dataset_sha256", "7" * 64),
    ("chat_template_sha256", "6" * 64),
    ("states", [1, 2, 3]),
])
def test_a_record_from_another_run_is_refused(field, value):
    expected = {key: _record()[key] for key in DS.RUN_LOCKED_FIELDS}
    with pytest.raises(DualSteerError, match="different experiment"):
        DS.verify_consistency([_record(**{field: value})], expected)


def test_a_record_whose_coefficients_contradict_its_arm_is_refused():
    expected = {key: _record()[key] for key in DS.RUN_LOCKED_FIELDS}
    with pytest.raises(DualSteerError, match="the predeclared arm is"):
        DS.verify_consistency([_record(arm_id="repair_015", alpha_R=0.20)],
                              expected)


def test_an_unknown_arm_is_refused():
    expected = {key: _record()[key] for key in DS.RUN_LOCKED_FIELDS}
    with pytest.raises(DualSteerError, match="unknown arm"):
        DS.verify_consistency([_record(arm_id="repair_099")], expected)


def test_a_partially_written_final_line_is_resumable_but_a_torn_one_is_not(
        tmp_path):
    path = tmp_path / "screen_records.jsonl"
    DS.append_record(path, _record())
    DS.append_record(path, _record(row_id="e:0002"))
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"schema": "routegate/dual_direction_scre')

    assert len(DS.read_records(path, strict=False)) == 2
    with pytest.raises(DualSteerError, match="not valid JSON"):
        DS.read_records(path, strict=True)

    torn = tmp_path / "torn.jsonl"
    torn.write_text('{"broken\n' + json.dumps(_record()) + "\n",
                    encoding="utf-8")
    with pytest.raises(DualSteerError, match="two writers shared it"):
        DS.read_records(torn, strict=False)


def test_a_record_of_another_schema_is_refused(tmp_path):
    path = tmp_path / "other.jsonl"
    path.write_text(json.dumps({"schema": "routegate/steer_record/1"}) + "\n",
                    encoding="utf-8")
    with pytest.raises(DualSteerError, match="carries schema"):
        DS.read_records(path)


def test_an_unknown_record_field_is_refused():
    with pytest.raises(DualSteerError, match="unknown record field"):
        _record(judge_verdict="COMPLY")


# =========================================================== 6. abstinence
class _StubModel:
    """Just enough model for the hook controller to bind to."""

    def __init__(self, n_layers=48, hidden=HIDDEN):
        import types
        self.config = types.SimpleNamespace(num_hidden_layers=n_layers,
                                            hidden_size=hidden)


def test_a_dry_run_loads_no_model_and_writes_nothing(tmp_path, monkeypatch):
    """The gates that need no GPU must all run before one is asked for."""
    loaded = []
    monkeypatch.setattr(
        "routegate.runtime.load_model_and_tokenizer",
        lambda *a, **k: loaded.append(True) or (_StubModel(), None))

    source = (REPO_ROOT / "experiments" / "dual_direction_controller"
              / "run_causal_screen.py").read_text(encoding="utf-8")
    body = source.split("if args.dry_run:")[0]
    # every import of the model loader happens after the dry-run return
    assert "load_model_and_tokenizer" not in body
    assert "PatchController" not in body
    assert loaded == []


def test_the_runner_reaches_nothing_sealed():
    from routegate.validation import looks_sealed
    options = [option for action in RUN.build_parser()._actions
               for option in action.option_strings]
    assert options
    for option in options:
        assert looks_sealed(option) is None, option
    for required in ("--dry-run", "--output-dir", "--resume", "--limit",
                     "--device"):
        assert required in options, required
    source = (REPO_ROOT / "experiments" / "dual_direction_controller"
              / "run_causal_screen.py").read_text(encoding="utf-8")
    assert "assert_not_sealed" in source
    assert "/home/" not in source


def test_the_runner_judges_nothing():
    """Checked on the code, not the prose.

    The docstring deliberately *says* it judges nothing, so a substring scan
    over the raw text fires on the documentation and would pass if the
    documentation were deleted.
    """
    import ast
    path = (REPO_ROOT / "experiments" / "dual_direction_controller"
            / "run_causal_screen.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))

    imported = set()
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Call):
            func = node.func
            called.add(getattr(func, "id", None) or getattr(func, "attr", ""))
    # "label" is deliberately not banned here: POSITIVE_LABEL and
    # WITHHELD_LABEL are the corpus's own E/B/H class names, which the runner
    # must read to know which rows are E and which are H. A judgement is a
    # verdict about a response, and that is what is absent.
    for banned in ("judge", "score", "classif", "verdict", "refusal"):
        assert not [name for name in imported if banned in name.lower()], banned
        assert not [name for name in called if banned in name.lower()], banned

    # the only mention is the manifest key that states it does not judge
    code = _code_of("run_causal_screen").lower()
    assert code.count("judge") == 1 and "judges_responses" in code
    assert "verdict" not in code and "refusal" not in code
    assert CONFIG["causal_screen"]["judging"]["in_this_stage"] is False
    assert CONFIG["causal_screen"]["controller"][
        "trained_in_this_stage"] is False


def test_generated_token_positions_are_never_edited():
    """The controller applies at the prefill forward only, and refuses later."""
    from routegate.hooks import PREFILL_FORWARD_INDEX
    source = (REPO_ROOT / "src" / "routegate" / "hooks.py").read_text("utf-8")
    assert PREFILL_FORWARD_INDEX == 0
    assert "forward_index != PREFILL_FORWARD_INDEX" in source
    assert DS.INTERVENTION_TYPE == "validated_dual_direction"


def test_the_screen_reuses_the_frozen_generation_settings():
    from routegate.runtime import (FROZEN_MAX_NEW_TOKENS, FROZEN_MODEL,
                                   FROZEN_REVISION, GenerationConfig)
    config = GenerationConfig()
    config.validate()
    assert config.model == FROZEN_MODEL
    assert config.revision == FROZEN_REVISION == config.tokenizer_revision
    assert config.do_sample is False and config.temperature is None
    assert config.max_new_tokens == FROZEN_MAX_NEW_TOKENS
    assert config.enable_thinking is False

    # The runner constructs GenerationConfig exactly once, and passes it
    # exactly one keyword: the device. Checked on the call node rather than by
    # substring, because "max_new_tokens=" also matches the record field
    # "hit_max_new_tokens=".
    import ast
    tree = ast.parse((REPO_ROOT / "experiments" / "dual_direction_controller"
                      / "run_causal_screen.py").read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and getattr(node.func, "id", "") == "GenerationConfig"]
    assert len(calls) == 1
    assert [keyword.arg for keyword in calls[0].keywords] == ["device_map"]
    assert not calls[0].args


def test_the_configuration_and_the_implementation_agree_on_the_screen():
    screen = CONFIG["causal_screen"]
    assert screen["inventory"]["E_generations"] == 445
    assert screen["inventory"]["H_generations"] == 2769
    assert screen["corpus"]["E_rows"] == DS.N_E_ROWS
    assert screen["corpus"]["H_rows"] == DS.N_H_ROWS
    assert screen["corpus"]["B_rows_are_targets"] is False
    assert screen["intervention"]["budget_reference"] == DS.BUDGET_REFERENCE
    assert screen["intervention"]["formula"] == (
        "delta = ||block_input|| * (alpha_R * d_R + alpha_S * d_S)")
    assert screen["intervention"]["same_coefficients_at_every_layer"] is True
    for relative in screen["implemented_by"].values():
        assert (REPO_ROOT / relative).is_file(), relative


# ================================ 7. smoke selection and consolidation
#
# The screen runs as eight shards and is preceded by a hand-picked handful of
# real generations. Both need the same guarantee from opposite directions: a
# shard must be part of the screen, and a smoke test must never be mistaken for
# it.
CONSOLIDATE = _load_script("consolidate_causal_screen")


def _parser_options(module):
    return [option for action in module.build_parser()._actions
            for option in action.option_strings]


def test_the_runner_exposes_the_deterministic_filters():
    options = _parser_options(RUN)
    for required in ("--only-class", "--only-arm", "--only-row-id", "--limit",
                     "--output-dir", "--resume", "--dry-run", "--device"):
        assert required in options, required
    # --only-class is a second spelling of the existing --classes, not a
    # second filter: one dest, so they cannot disagree
    dests = {action.dest for action in RUN.build_parser()._actions
             if "--only-class" in action.option_strings}
    assert dests == {"classes"}
    assert "--classes" in options


def test_a_filter_value_with_a_stray_comma_is_refused():
    assert RUN.parse_selection("", "--only-arm") == set()
    assert RUN.parse_selection("a,b", "--only-arm") == {"a", "b"}
    with pytest.raises(DualSteerError, match="empty element"):
        RUN.parse_selection("a,,b", "--only-arm")
    with pytest.raises(DualSteerError, match="empty element"):
        RUN.parse_selection("a,", "--only-arm")


def _filtered(items, classes=None, only_arm=(), only_row=(), limit=0):
    """The runner's filter-then-limit order, applied to an inventory."""
    selected = list(items)
    if classes:
        selected = [i for i in selected if i["row_class"] in classes]
    if only_arm:
        selected = [i for i in selected if i["arm_id"] in set(only_arm)]
    if only_row:
        selected = [i for i in selected if i["row_id"] in set(only_row)]
    if limit:
        selected = selected[:limit]
    return selected


def test_the_filters_are_applied_before_the_limit():
    """A limit applied first would decide which rows the filter could see."""
    items = DS.build_inventory(_rows())
    wanted_row = "h:0100"

    correct = _filtered(items, only_row=[wanted_row],
                        only_arm=["repair015_safety015"], limit=1)
    assert len(correct) == 1
    assert correct[0]["row_id"] == wanted_row
    assert correct[0]["arm_id"] == "repair015_safety015"

    # limit first would truncate to E rows and select nothing at all
    truncated_first = [i for i in items[:1]
                       if i["row_id"] == wanted_row
                       and i["arm_id"] == "repair015_safety015"]
    assert truncated_first == []

    source = _code_of("run_causal_screen")
    assert source.index("only_row") < source.index("args.limit")


def test_each_filter_narrows_the_inventory_as_declared():
    items = DS.build_inventory(_rows())
    assert len(_filtered(items, classes=["E"])) == 89 * 5
    assert len(_filtered(items, classes=["H"])) == 213 * 13
    assert len(_filtered(items, only_arm=["baseline"])) == 89 + 213
    assert len(_filtered(items, only_arm=["repair015_safety015"])) == 213
    assert len(_filtered(items, only_row=["e:0000"])) == 5
    assert len(_filtered(items, only_row=["h:0000"])) == 13
    # the four-record smoke, as two selections
    assert len(_filtered(items, only_row=["e:0000"],
                         only_arm=["baseline", "repair_015"])) == 2
    assert len(_filtered(items, only_row=["h:0000"],
                         only_arm=["repair_015", "repair015_safety015"])) == 2


def test_a_smoke_run_is_marked_on_the_run_and_on_every_record():
    """Marked, not merely smaller: smoke is run-locked."""
    assert "smoke" in DS.RUN_LOCKED_FIELDS
    assert "smoke" in DS.RECORD_FIELDS
    source = _code_of("run_causal_screen")
    for marker in ("smoke_reasons", "complete_causal_screen", "frozen_protocol"):
        assert marker in source, marker
    # a smoke record and a full record cannot share an output directory
    expected = {key: _record(smoke=False)[key] for key in DS.RUN_LOCKED_FIELDS}
    with pytest.raises(DualSteerError, match="different experiment"):
        DS.verify_consistency([_record(smoke=True)], expected)


def test_an_unknown_arm_or_row_filter_is_refused():
    source = _code_of("run_causal_screen")
    assert "not predeclared" in source or "are not predeclared" in source
    assert "ARM_BY_ID" in source


# --- consolidation ---------------------------------------------------------
def _shard_record(row_id, row_class, arm_id, response="a reply", **overrides):
    arm = DS.ARM_BY_ID[arm_id]
    values = {
        "row_id": row_id, "row_class": row_class, "arm_id": arm_id,
        "alpha_R": arm.alpha_repair, "alpha_S": arm.alpha_safety,
        "direction_bank_sha256": CONFIG["causal_screen"]["direction_bank"][
            "artifact_sha256"],
        "generation_config_sha256": "b" * 64,
        "model": "Qwen/Qwen3-30B-A3B", "revision": "c" * 40,
        "tokenizer_config_sha256": "d" * 64, "chat_template_sha256": "e" * 64,
        "intervention_type": DS.INTERVENTION_TYPE,
        "budget_reference": DS.BUDGET_REFERENCE,
        "states": list(BANK_STATES),
        "historical_layers": list(HISTORICAL_LAYERS),
        "dataset_sha256": "f" * 64, "smoke": False,
        "protocol": DS.DEFAULT_PROTOCOL,
        "prompt_sha256": "1" * 64,
        "response_text": response,
        "response_sha256": __import__("hashlib").sha256(
            response.encode("utf-8")).hexdigest(),
        "status": "ok", "created_utc": "2026-08-22T00:00:00Z",
        "generation": {"model": "Qwen/Qwen3-30B-A3B"},
    }
    values.update(overrides)
    return DS.make_record(**values)


def _write_screen(tmp_path, shard_count=8, mutate=None, manifest_mutate=None):
    """Eight shards holding the complete 3,214-record screen."""
    rows = _rows()
    inventory = DS.build_inventory(rows)
    records = [_shard_record(item["row_id"], item["row_class"], item["arm_id"],
                             response=f"reply to {item['row_id']} "
                                      f"{item['arm_id']}")
               for item in inventory]
    if mutate is not None:
        records = mutate(records)

    root = tmp_path / "screen"
    size, remainder = divmod(len(records), shard_count)
    start = 0
    for index in range(1, shard_count + 1):
        length = size + (1 if index - 1 < remainder else 0)
        directory = root / f"shard_{index}"
        directory.mkdir(parents=True, exist_ok=True)
        for record in records[start:start + length]:
            DS.append_record(directory / "screen_records.jsonl", record)
        start += length
        manifest = {key: _shard_record("e:0000", "E", "baseline")[key]
                    for key in DS.RUN_LOCKED_FIELDS}
        manifest.update({"dry_run": False, "smoke": False,
                         "created_utc": "2026-08-22T00:00:00Z",
                         "generation": {"model": "Qwen/Qwen3-30B-A3B"}})
        if manifest_mutate is not None:
            manifest = manifest_mutate(index, manifest)
        (directory / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return root


def test_the_complete_screen_consolidates_to_three_thousand_two_hundred_and_fourteen(
        tmp_path):
    root = _write_screen(tmp_path)
    result = CONSOLIDATE.consolidate(root, tmp_path / "out",
                                     REPO_ROOT / "configs"
                                     / "dual_direction_controller.json",
                                     write=True)
    counts = result["validation"]["counts"]
    assert counts["total"] == 3214
    assert counts["by_class"]["E"] == 445
    assert counts["by_class"]["H"] == 2769
    assert counts["rows_by_class"] == {"E": 89, "H": 213}
    assert counts["by_arm"]["baseline"] == 302
    assert counts["by_arm"]["repair015_safety015"] == 213
    assert result["validation"]["complete_causal_screen"] is True
    assert result["validation"]["smoke"] is False
    assert result["validation"]["decides_no_behavioural_outcome"] is True

    out = tmp_path / "out"
    for name in ("screen_records.jsonl", "validation.json", "run_manifest.json"):
        assert (out / name).is_file(), name
    assert len(DS.read_records(out / "screen_records.jsonl")) == 3214


def test_the_consolidated_order_is_deterministic(tmp_path):
    first = CONSOLIDATE.consolidate(
        _write_screen(tmp_path / "a"), tmp_path / "out_a",
        REPO_ROOT / "configs" / "dual_direction_controller.json", write=True)
    second = CONSOLIDATE.consolidate(
        _write_screen(tmp_path / "b", shard_count=4), tmp_path / "out_b",
        REPO_ROOT / "configs" / "dual_direction_controller.json",
        shard_count=4, write=True)
    assert (first["validation"]["records_sha256"]
            == second["validation"]["records_sha256"])
    assert (tmp_path / "out_a" / "screen_records.jsonl").read_bytes() == (
        tmp_path / "out_b" / "screen_records.jsonl").read_bytes()

    keys = [(r["row_class"], r["row_id"]) for r in first["records"]]
    assert keys == sorted(keys, key=lambda k: (["E", "H"].index(k[0]), k[1]))


def test_a_duplicate_record_stops_the_consolidation(tmp_path):
    root = _write_screen(tmp_path, mutate=lambda rs: rs + [rs[0]])
    with pytest.raises(DualSteerError, match="no_duplicate_records"):
        CONSOLIDATE.consolidate(root, tmp_path / "out",
                                REPO_ROOT / "configs"
                                / "dual_direction_controller.json", write=False)


def test_a_missing_record_stops_the_consolidation(tmp_path):
    root = _write_screen(tmp_path, mutate=lambda rs: rs[:-1])
    with pytest.raises(DualSteerError, match="H_records: 2768"):
        CONSOLIDATE.consolidate(root, tmp_path / "out",
                                REPO_ROOT / "configs"
                                / "dual_direction_controller.json", write=False)


def test_a_class_that_is_short_is_named(tmp_path):
    root = _write_screen(
        tmp_path, mutate=lambda rs: [r for r in rs
                                     if not (r["row_class"] == "E"
                                             and r["row_id"] == "e:0000")])
    with pytest.raises(DualSteerError, match="E_records: 440"):
        CONSOLIDATE.consolidate(root, tmp_path / "out",
                                REPO_ROOT / "configs"
                                / "dual_direction_controller.json", write=False)


def test_an_absent_shard_stops_the_consolidation(tmp_path):
    root = _write_screen(tmp_path)
    import shutil
    shutil.rmtree(root / "shard_5")
    with pytest.raises(DualSteerError, match="shard directory"):
        CONSOLIDATE.consolidate(root, tmp_path / "out",
                                REPO_ROOT / "configs"
                                / "dual_direction_controller.json", write=False)


@pytest.mark.parametrize("field,value", [
    ("generation_config_sha256", "9" * 64),
    ("chat_template_sha256", "8" * 64),
    ("tokenizer_config_sha256", "7" * 64),
    ("model", "some/other-model"),
    ("dataset_sha256", "6" * 64),
])
def test_shards_from_different_runs_are_refused(tmp_path, field, value):
    def bend(index, manifest):
        if index == 3:
            manifest[field] = value
        return manifest

    root = _write_screen(tmp_path, manifest_mutate=bend)
    with pytest.raises(DualSteerError, match="not one experiment"):
        CONSOLIDATE.consolidate(root, tmp_path / "out",
                                REPO_ROOT / "configs"
                                / "dual_direction_controller.json", write=False)


def test_a_smoke_shard_is_never_the_screen(tmp_path):
    def bend(index, manifest):
        if index == 2:
            manifest["smoke"] = True
            manifest["smoke_reasons"] = ["--limit 4"]
        return manifest

    root = _write_screen(tmp_path, manifest_mutate=bend)
    with pytest.raises(DualSteerError, match="never accepted as"):
        CONSOLIDATE.consolidate(root, tmp_path / "out",
                                REPO_ROOT / "configs"
                                / "dual_direction_controller.json", write=False)


def test_a_dry_run_shard_cannot_contribute(tmp_path):
    def bend(index, manifest):
        if index == 1:
            manifest["dry_run"] = True
        return manifest

    root = _write_screen(tmp_path, manifest_mutate=bend)
    with pytest.raises(DualSteerError, match="written by a dry run"):
        CONSOLIDATE.consolidate(root, tmp_path / "out",
                                REPO_ROOT / "configs"
                                / "dual_direction_controller.json", write=False)


def test_a_response_digest_that_does_not_recompute_is_refused(tmp_path):
    def bend(records):
        records[7] = dict(records[7], response_sha256="0" * 64)
        return records

    root = _write_screen(tmp_path, mutate=bend)
    with pytest.raises(DualSteerError, match="response_digests_recompute"):
        CONSOLIDATE.consolidate(root, tmp_path / "out",
                                REPO_ROOT / "configs"
                                / "dual_direction_controller.json", write=False)


def test_an_incomplete_record_is_refused(tmp_path):
    def bend(records):
        records[11] = dict(records[11], status="")
        return records

    root = _write_screen(tmp_path, mutate=bend)
    with pytest.raises(DualSteerError, match="records_are_complete"):
        CONSOLIDATE.consolidate(root, tmp_path / "out",
                                REPO_ROOT / "configs"
                                / "dual_direction_controller.json", write=False)


def test_a_record_whose_coefficients_contradict_its_arm_is_refused(tmp_path):
    def bend(records):
        records[3] = dict(records[3], alpha_R=0.99)
        return records

    root = _write_screen(tmp_path, mutate=bend)
    with pytest.raises(DualSteerError, match="coefficients_match_their_arm"):
        CONSOLIDATE.consolidate(root, tmp_path / "out",
                                REPO_ROOT / "configs"
                                / "dual_direction_controller.json", write=False)


def test_a_consolidation_dry_run_writes_nothing(tmp_path):
    root = _write_screen(tmp_path)
    out = tmp_path / "out"
    result = CONSOLIDATE.consolidate(
        root, out, REPO_ROOT / "configs" / "dual_direction_controller.json",
        write=False)
    assert not out.exists()
    assert result["validation"]["counts"]["total"] == 3214
    # and the digest a real run would write is already known
    written = CONSOLIDATE.consolidate(
        root, out, REPO_ROOT / "configs" / "dual_direction_controller.json",
        write=True)
    assert (result["validation"]["records_sha256"]
            == written["validation"]["records_sha256"])


def test_a_non_empty_output_directory_is_refused(tmp_path):
    root = _write_screen(tmp_path)
    out = tmp_path / "out"
    config = REPO_ROOT / "configs" / "dual_direction_controller.json"
    CONSOLIDATE.consolidate(root, out, config, write=True)
    with pytest.raises(DualSteerError, match="written once"):
        CONSOLIDATE.consolidate(root, out, config, write=True)


def test_the_consolidator_judges_nothing():
    import ast
    path = (REPO_ROOT / "experiments" / "dual_direction_controller"
            / "consolidate_causal_screen.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    called = set()
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Call):
            func = node.func
            called.add(getattr(func, "id", None) or getattr(func, "attr", ""))
    for banned in ("judge", "score", "classif", "verdict", "refusal"):
        assert not [n for n in imported if banned in n.lower()], banned
        assert not [n for n in called if banned in n.lower()], banned
    options = _parser_options(CONSOLIDATE)
    assert "--dry-run" in options and "--shards-root" in options


def test_the_submission_script_pins_the_run():
    script = (REPO_ROOT / "experiments" / "dual_direction_controller"
              / "submit_causal_screen.sh").read_text(encoding="utf-8")
    assert "#$ -t 1-8" in script
    assert "EXPECT_HEAD" in script and "git rev-parse HEAD" in script
    assert "git status --porcelain" in script
    assert ("d759574b5be630ea5170fb440f8663d8c4bfaa32b479ce5cf641b6dbd73516bb"
            in script)
    assert "--nv" in script                       # the GPU container invocation
    assert 'OUTDIR="$SCREEN_ROOT/shard_${TASK}"' in script
    assert "--resume" in script
    # the array never runs a filtered subset
    for filter_flag in ("--only-class", "--only-arm", "--only-row-id",
                        "--limit"):
        assert f"{filter_flag} " not in script.split("apptainer exec")[1].split(
            '"\n')[0], filter_flag


# ============================= 8. what a run says it did
#
# The environment block is what a reader consults to know whether a number came
# from arithmetic over stored activations or from a model that was actually
# run. The first real H smoke run wrote "model_loaded: false" and "no language
# model, no generation" into a manifest for two real generations, which is the
# one thing that block must never get wrong.
def test_an_offline_tool_still_says_it_loaded_no_model():
    """The default is unchanged: almost every caller here is CPU-only."""
    from routegate.runrecord import OFFLINE_NOTE, environment_record
    record = environment_record("0.1.0")
    assert record["model_loaded"] is False
    assert record["generation_performed"] is False
    assert record["network_calls"] == 0
    assert record["note"] == OFFLINE_NOTE
    assert "no language model, no generation" in record["note"]


def test_a_generation_run_says_it_loaded_a_model_and_generated():
    from routegate.runrecord import GENERATION_NOTE, environment_record
    record = environment_record("0.1.0", model_loaded=True,
                                generation_performed=True)
    assert record["model_loaded"] is True
    assert record["generation_performed"] is True
    assert "no language model" not in record["note"]
    assert record["note"] == GENERATION_NOTE
    # a local snapshot is not a network call, and none is made either way
    assert record["network_calls"] == 0
    assert "no external service was called" in record["note"]


def test_the_stack_fields_are_the_same_either_way():
    """Only the three claims change; the recorded stack does not."""
    from routegate.runrecord import environment_record
    offline = environment_record("0.1.0")
    generating = environment_record("0.1.0", model_loaded=True,
                                    generation_performed=True)
    changed = {key for key in offline if offline[key] != generating[key]}
    assert changed == {"model_loaded", "generation_performed", "note"}


def test_the_runner_builds_one_environment_and_keys_it_to_the_dry_run():
    """The manifest and every record share it, so they cannot disagree."""
    source = _code_of("run_causal_screen")
    assert "model_loaded=not args.dry_run" in source
    assert "generation_performed=not args.dry_run" in source
    # one construction, reused
    assert source.count("environment_record(") == 1
    assert "'environment': environment" in source
    assert "environment=environment" in source


def test_a_dry_run_manifest_says_no_model_was_loaded(tmp_path, monkeypatch):
    """A dry run plans and loads nothing, and says so."""
    from routegate.runrecord import OFFLINE_NOTE, environment_record
    record = environment_record("0.1.0", model_loaded=False,
                               generation_performed=False)
    assert record["model_loaded"] is False and record["note"] == OFFLINE_NOTE

    source = _code_of("run_causal_screen")
    # the dry run returns before the loader is even imported
    before_return = source.split("if args.dry_run:")[0]
    assert "load_model_and_tokenizer" not in before_return


def test_the_manifest_is_written_only_after_the_model_is_loaded():
    """It claims a model was loaded; it must not be able to claim that of a
    run whose load then failed."""
    source = _code_of("run_causal_screen")
    load_at = source.index("load_model_and_tokenizer(generation)")
    write_at = source.index("write_json(out_dir / MANIFEST_FILENAME")
    assert load_at < write_at


def test_the_consolidator_is_still_an_offline_tool():
    """It reads JSONL and writes JSONL; nothing about it loads a model."""
    import ast
    path = (REPO_ROOT / "experiments" / "dual_direction_controller"
            / "consolidate_causal_screen.py")
    source = path.read_text(encoding="utf-8")
    assert "environment_record(PACKAGE_VERSION)" in source
    assert "model_loaded" not in source
    tree = ast.parse(source)
    called = {getattr(node.func, "id", None) or getattr(node.func, "attr", "")
              for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert "load_model_and_tokenizer" not in called
