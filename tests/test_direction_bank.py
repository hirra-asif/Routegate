"""The validated dual-direction bank, on synthetic activations.

Nothing here loads a model, allocates a GPU, generates, judges or reads the
sealed set. Every activation is invented in ``tmp_path`` and every digest is
recomputed for the synthetic inputs, so the real capture is never needed to
test the machinery that guards it.

Organised by the failure each group is there to prevent:

    1. signs        a reversed subtraction yields a well-formed wrong answer
    2. recipe       per-case normalisation, averaging, final renormalisation
    3. degenerate   zero and non-finite deltas are refused, never normalised
    4. eligibility  the authoritative record decides, and only it
    5. shape        wrong shape, dtype, layer or state
    6. hashes       an input whose digest moved is refused
    7. arms         a missing, substituted or extra arm or case
    8. artefact     unit norms, deterministic bytes, pickle-free loading
    9. parity       the rebuilt repair bank must reproduce the frozen d1
   10. abstinence   no strength and no orthogonalisation reach the artefact
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate import directionbank as DB                            # noqa: E402
from routegate.directionbank import DirectionBankError               # noqa: E402

CASES = [f"case_{i:02d}_alpha_vs_beta" for i in range(10)]
SEED = 20260822


def _load_script(name: str):
    path = REPO_ROOT / "experiments" / "dual_direction_controller" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_ddc_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BUILD = _load_script("build_validated_direction_bank")


# ---------------------------------------------------------------- fixtures
def _activations(seed=SEED, shared_scale=1.0):
    """Three arms whose two directions share a component.

    Random arms would make the repair and safety directions near-orthogonal in
    2048 dimensions by chance, and the check that nothing orthogonalised them
    would then pass whatever the code did. Giving both deltas a shared
    component -- with opposite sign, as the real captures have -- makes that
    check mean something.
    """
    rng = np.random.default_rng(seed)
    shape = DB.CAPTURE_SHAPE
    E = rng.standard_normal(shape).astype(np.float32)
    shared = (rng.standard_normal(shape[1:]).astype(np.float32) * shared_scale)
    B = E + shared + 0.3 * rng.standard_normal(shape).astype(np.float32)
    H = E - 0.8 * shared + 0.3 * rng.standard_normal(shape).astype(np.float32)
    return {"E": E.astype(np.float32), "B_d1": B.astype(np.float32),
            "H": H.astype(np.float32)}


def _inclusion(cases=CASES, validated=None, arms_prompt=None):
    """An eligibility record in the real manifest's shape."""
    validated = validated or {rule: list(cases) for rule in
                              ("comparator_actuator_source",
                               "h_e_direction_source",
                               "primary_actuator_source")}
    arms_prompt = arms_prompt or {}
    record = {"schema": "routegate/rq1_triplet_inclusion/1",
              "expected_case_ids": list(cases), "cases": {}}
    for case in cases:
        rules = {}
        for rule, members in validated.items():
            rules[rule] = {
                "inclusion_status": ("validated" if case in members
                                     else "invalid"),
                "exclusion_reason": None,
                "purpose": f"synthetic {rule}"}
        arms = {}
        for arm in ("E", "B_d1", "H", "B_d2", "B_d3"):
            digest = arms_prompt.get((case, arm)) or hashlib.sha256(
                f"{case}|{arm}".encode("utf-8")).hexdigest()
            arms[arm] = {"prompt_sha256": digest, "response_sha256": None,
                         "reached_max_tokens": None, "output_token_count": None,
                         "axes": {}}
        record["cases"][case] = {"arms": arms, "rules": rules,
                                 "cells_present": 11, "cells_expected": 11,
                                 "entirely_absent": False}
    return record


def _write_world(tmp_path, activations=None, inclusion=None, cases=CASES,
                 capture_overrides=None, meta_overrides=None,
                 source_overrides=None):
    """A complete synthetic input world: capture, records and the pin."""
    torch = pytest.importorskip("torch")

    activations = activations if activations is not None else _activations()
    inclusion = inclusion if inclusion is not None else _inclusion(cases)

    capture_dir = tmp_path / "rq2_geometry_capture"
    capture_dir.mkdir(parents=True, exist_ok=True)

    payload = {DB.ARM_TENSOR[arm]: torch.from_numpy(np.ascontiguousarray(block))
               for arm, block in activations.items()}
    payload["meta"] = {
        "case_ids": list(cases), "layers": [31, 32, 33],
        "sensor_layers": [20, 24, 25], "window": 1, "strength": 0.2,
        "slot_convention": ("window=1, slot 0 only: the final prefill row "
                            "(DecisionSlot(forward_index=0, row_index=-1, "
                            "slot=0)), i.e. the token position that decides "
                            "generated token 1.")}
    payload["meta"].update(meta_overrides or {})
    payload.update(capture_overrides or {})
    torch.save(payload, capture_dir / "capture.pt")

    capture_meta = {
        "model": DB.MODEL_NAME, "dtype": "bfloat16",
        "attention_implementation": "eager",
        "prompt_style": "chat_template_enable_thinking_false",
        "layers": [31, 32, 33], "window": 1,
        "case_ids": list(cases),
        "prompt_hashes": {
            case: {BUILD.CAPTURE_PROMPT_KEY[arm]:
                   inclusion["cases"][case]["arms"][arm]["prompt_sha256"]
                   for arm in DB.ARM_TENSOR}
            for case in cases}}
    for case in cases:
        capture_meta["prompt_hashes"][case]["donor_system"] = None
    (capture_dir / "manifest.json").write_text(
        json.dumps(capture_meta, indent=2, sort_keys=True), encoding="utf-8")

    inclusion_path = tmp_path / "rq1_triplet_inclusion.json"
    inclusion_path.write_text(json.dumps(inclusion, indent=2, sort_keys=True),
                              encoding="utf-8")

    source_path = tmp_path / "source_case_manifest.csv"
    columns = ["case_id"] + [BUILD.SOURCE_PROMPT_COLUMN[a]
                             for a in ("E", "B_d1", "H")]
    lines = [",".join(columns)]
    for case in cases:
        row = [case] + [
            (source_overrides or {}).get((case, arm))
            or inclusion["cases"][case]["arms"][arm]["prompt_sha256"]
            for arm in ("E", "B_d1", "H")]
        lines.append(",".join(row))
    source_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    pin = _pin(tmp_path, capture_dir, inclusion_path, source_path)
    return {"capture_dir": capture_dir, "inclusion": inclusion_path,
            "source": source_path, "pin": pin, "out": tmp_path / "bank",
            "root": tmp_path, "activations": activations}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pin(tmp_path, capture_dir, inclusion_path, source_path):
    """The committed record naming every input and its digest."""
    pin = {
        "schema": "routegate/rq2_geometry_capture/1",
        "files": {
            "capture.pt": {"sha256": _sha(capture_dir / "capture.pt"),
                           "bytes": (capture_dir / "capture.pt").stat().st_size},
            "manifest.json": {
                "sha256": _sha(capture_dir / "manifest.json"),
                "bytes": (capture_dir / "manifest.json").stat().st_size}},
        "eligibility_gate": {"sha256": _sha(inclusion_path),
                             "bytes": inclusion_path.stat().st_size},
        "case_digest_cross_check": {"sha256": _sha(source_path),
                                    "bytes": source_path.stat().st_size},
        "model": {"name": DB.MODEL_NAME, "revision": DB.MODEL_REVISION,
                  "dtype": "bfloat16",
                  "revision_provenance": "synthetic pin for tests"}}
    path = tmp_path / "rq2_geometry_capture.json"
    path.write_text(json.dumps(pin, indent=2, sort_keys=True), encoding="utf-8")
    return path


class _StubFrozen:
    """Stands in for the migrated RQ2 freeze in the parity check."""

    def __init__(self, directions):
        self.directions = directions


def _run(world, monkeypatch, parity=None, out=None, write=True):
    """Build, with the parity reference supplied rather than loaded."""
    expected = parity
    if expected is None:
        expected = DB.build_bank(
            world["activations"],
            {d: CASES for d in DB.DIRECTIONS}).repair
    monkeypatch.setattr(BUILD, "load_frozen_actuator",
                        lambda **kwargs: _StubFrozen(expected))
    return BUILD.build(world["capture_dir"], world["inclusion"],
                       world["source"], world["root"] / "actuator",
                       out or world["out"], write=write,
                       capture_manifest_path=world["pin"])


# ============================================================== 1. signs
def test_the_subtraction_is_arm_minus_baseline_not_the_reverse():
    acts = _activations()
    repair = DB.build_direction("repair", acts["B_d1"], acts["E"])
    reversed_ = DB.build_direction("repair", acts["E"], acts["B_d1"])
    per_layer = DB.per_layer_cosine(repair, reversed_)
    assert all(value < -0.99 for value in per_layer), per_layer


def test_each_direction_is_bound_to_its_own_arm():
    assert DB.DIRECTIONS["repair"]["arm"] == "B_d1"
    assert DB.DIRECTIONS["safety"]["arm"] == "H"
    assert DB.DIRECTIONS["repair"]["rule"] == "comparator_actuator_source"
    assert DB.DIRECTIONS["safety"]["rule"] == "h_e_direction_source"
    DB.assert_subtraction_order("repair", "B_d1")
    DB.assert_subtraction_order("safety", "H")
    with pytest.raises(DirectionBankError, match="repair is B_d1 - E"):
        DB.assert_subtraction_order("repair", "H")
    with pytest.raises(DirectionBankError, match="unknown direction"):
        DB.assert_subtraction_order("primary", "B_d2")


@pytest.mark.parametrize("arm", list(DB.BASELINE_ONLY_ARMS))
def test_the_baseline_only_actuator_arms_cannot_enter_the_bank(arm):
    assert arm not in {spec["arm"] for spec in DB.DIRECTIONS.values()}
    assert arm not in DB.ARM_TENSOR
    with pytest.raises(DirectionBankError, match="unknown direction"):
        DB.assert_subtraction_order(arm, arm)


def test_the_primary_actuator_rule_is_not_an_eligibility_source():
    with pytest.raises(DirectionBankError, match="comparison baseline"):
        DB.eligible_case_ids(_inclusion(), DB.BASELINE_ONLY_RULE)


# ============================================================= 2. recipe
def test_each_case_is_normalised_before_anything_is_averaged():
    acts = _activations()
    per_case = DB.case_directions(acts["B_d1"], acts["E"], "repair")
    norms = np.linalg.norm(per_case, axis=-1)
    assert per_case.shape == DB.CAPTURE_SHAPE
    assert np.allclose(norms, 1.0, atol=1e-5), norms.min()


def test_one_case_with_a_huge_delta_does_not_dominate_the_bank():
    """What per-case normalisation is for, stated as a test."""
    acts = _activations()
    inflated = {k: v.copy() for k, v in acts.items()}
    inflated["B_d1"][0] = acts["E"][0] + 1000.0 * (acts["B_d1"][0] - acts["E"][0])
    plain = DB.build_direction("repair", acts["B_d1"], acts["E"])
    scaled = DB.build_direction("repair", inflated["B_d1"], inflated["E"])
    # scaling one case's delta changes its length, not its direction, so the
    # bank is unmoved
    assert all(value > 0.999999 for value in DB.per_layer_cosine(plain, scaled))


def test_the_average_is_renormalised_to_unit_norm():
    acts = _activations()
    per_case = DB.case_directions(acts["H"], acts["E"], "safety")
    mean = per_case.mean(axis=0)
    assert not np.allclose(np.linalg.norm(mean, axis=-1), 1.0, atol=1e-3)
    direction = DB.aggregate(per_case, "safety")
    assert np.allclose(np.linalg.norm(direction, axis=-1), 1.0, atol=1e-6)
    assert all(value > 0.999999
               for value in DB.per_layer_cosine(direction, mean))


def test_every_layer_is_built_independently():
    """Layer 33's activations cannot move layer 31's direction."""
    acts = _activations()
    before = DB.build_direction("repair", acts["B_d1"], acts["E"])
    moved = {k: v.copy() for k, v in acts.items()}
    moved["B_d1"][:, 2, :] += 5.0
    after = DB.build_direction("repair", moved["B_d1"], moved["E"])
    cosines = DB.per_layer_cosine(before, after)
    assert cosines[0] > 0.999999 and cosines[1] > 0.999999
    assert cosines[2] < 0.99


def test_the_identical_recipe_produces_both_directions():
    acts = _activations()
    bank = DB.build_bank(acts, {d: CASES for d in DB.DIRECTIONS})
    manual_repair = DB.aggregate(
        DB.case_directions(acts["B_d1"], acts["E"], "r"), "r")
    manual_safety = DB.aggregate(
        DB.case_directions(acts["H"], acts["E"], "s"), "s")
    assert np.array_equal(bank.repair, manual_repair)
    assert np.array_equal(bank.safety, manual_safety)


# ========================================================= 3. degenerate
def test_a_zero_delta_is_refused_not_normalised():
    acts = _activations()
    acts["B_d1"][3, 1, :] = acts["E"][3, 1, :]
    with pytest.raises(DirectionBankError, match="carry no direction"):
        DB.build_direction("repair", acts["B_d1"], acts["E"])


def test_a_near_zero_delta_is_refused():
    acts = _activations()
    acts["H"][7, 0, :] = acts["E"][7, 0, :] + 1e-9
    with pytest.raises(DirectionBankError, match="shorter than"):
        DB.build_direction("safety", acts["H"], acts["E"])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_non_finite_activation_is_refused(bad):
    acts = _activations()
    acts["B_d1"][2, 0, 5] = bad
    with pytest.raises(DirectionBankError, match="non-finite"):
        DB.build_direction("repair", acts["B_d1"], acts["E"])


def test_ten_cases_that_cancel_leave_no_bank():
    """Not a degenerate delta -- a degenerate average."""
    shape = DB.CAPTURE_SHAPE
    rng = np.random.default_rng(7)
    delta = rng.standard_normal(shape).astype(np.float32)
    delta[5:] = -delta[:5]
    E = np.zeros(shape, dtype=np.float32)
    with pytest.raises(DirectionBankError, match="cancelled"):
        DB.build_direction("repair", delta, E)


# ======================================================== 4. eligibility
def test_only_cases_the_record_marks_validated_are_used():
    inclusion = _inclusion(validated={
        "comparator_actuator_source": CASES,
        "h_e_direction_source": CASES,
        "primary_actuator_source": CASES[:3]})
    assert DB.eligible_case_ids(
        inclusion, "comparator_actuator_source") == tuple(CASES)
    assert DB.eligible_case_ids(
        inclusion, "h_e_direction_source") == tuple(CASES)


def test_the_eligible_cases_come_back_in_the_records_own_order():
    shuffled = list(reversed(CASES))
    inclusion = _inclusion(cases=shuffled)
    assert DB.eligible_case_ids(
        inclusion, "comparator_actuator_source") == tuple(shuffled)


@pytest.mark.parametrize("n", [9, 11])
def test_a_rule_validating_other_than_ten_cases_is_refused(n):
    cases = [f"case_{i:02d}_alpha_vs_beta" for i in range(n)]
    inclusion = _inclusion(cases=cases)
    with pytest.raises(DirectionBankError, match="this bank is built from 10"):
        DB.eligible_case_ids(inclusion, "comparator_actuator_source")


def test_an_invalid_case_is_not_quietly_included():
    inclusion = _inclusion(validated={
        "comparator_actuator_source": CASES[:9],
        "h_e_direction_source": CASES,
        "primary_actuator_source": CASES})
    with pytest.raises(DirectionBankError, match="marks 9 case"):
        DB.eligible_case_ids(inclusion, "comparator_actuator_source")


def test_a_case_with_no_rule_record_is_refused_not_assumed():
    inclusion = _inclusion()
    del inclusion["cases"][CASES[4]]["rules"]["h_e_direction_source"]
    with pytest.raises(DirectionBankError, match="eligibility cannot be"):
        DB.eligible_case_ids(inclusion, "h_e_direction_source")


def test_the_eligibility_record_digest_pins_the_one_decision():
    inclusion = _inclusion()
    first = DB.eligibility_record_sha256(
        inclusion, CASES[0], "comparator_actuator_source")
    inclusion["cases"][CASES[0]]["rules"][
        "comparator_actuator_source"]["inclusion_status"] = "invalid"
    assert DB.eligibility_record_sha256(
        inclusion, CASES[0], "comparator_actuator_source") != first


def test_a_wrong_case_order_in_the_capture_is_refused(tmp_path, monkeypatch):
    reordered = list(CASES[5:]) + list(CASES[:5])
    world = _write_world(tmp_path, cases=CASES,
                         meta_overrides={"case_ids": reordered})
    with pytest.raises(DirectionBankError, match="capture_case_order"):
        _run(world, monkeypatch)


# ============================================================== 5. shape
@pytest.mark.parametrize("shape", [(9, 3, 2048), (10, 2, 2048), (10, 3, 1024)])
def test_a_wrong_activation_shape_is_refused(shape):
    with pytest.raises(DirectionBankError, match="shape is"):
        DB.assert_activation_block(np.zeros(shape, dtype=np.float32), "arm")


@pytest.mark.parametrize("dtype", ["float64", "float16", "int32"])
def test_a_wrong_dtype_is_refused_not_cast(dtype):
    block = np.zeros(DB.CAPTURE_SHAPE, dtype=dtype)
    with pytest.raises(DirectionBankError, match="dtype is"):
        DB.assert_activation_block(block, "arm")


def test_a_non_array_is_refused():
    with pytest.raises(DirectionBankError, match="expected a numpy array"):
        DB.assert_activation_block([[0.0]], "arm")


def test_the_layer_band_and_the_state_numbering_are_fixed():
    assert DB.HISTORICAL_LAYERS == (31, 32, 33)
    assert DB.BANK_STATES == (32, 33, 34)
    DB.assert_layers_and_states([31, 32, 33], [32, 33, 34])
    with pytest.raises(DirectionBankError, match="does not remap"):
        DB.assert_layers_and_states([30, 31, 32], [32, 33, 34])
    with pytest.raises(DirectionBankError, match="state = block \\+ 1"):
        DB.assert_layers_and_states([31, 32, 33], [31, 32, 33])


def test_the_state_mapping_comes_from_the_shared_convention():
    from routegate.intervention import historical_layer_to_state
    assert DB.BANK_STATES == tuple(
        historical_layer_to_state(layer) for layer in DB.HISTORICAL_LAYERS)


def test_a_capture_declaring_the_wrong_band_is_refused(tmp_path, monkeypatch):
    world = _write_world(tmp_path, meta_overrides={"layers": [30, 31, 32]})
    with pytest.raises(DirectionBankError, match="does not remap"):
        _run(world, monkeypatch)


def test_a_capture_declaring_the_wrong_token_position_is_refused(
        tmp_path, monkeypatch):
    world = _write_world(
        tmp_path, meta_overrides={"slot_convention": "all decode positions"})
    with pytest.raises(DirectionBankError, match="token_position"):
        _run(world, monkeypatch)


def test_a_capture_with_a_wider_window_is_refused(tmp_path, monkeypatch):
    world = _write_world(tmp_path, meta_overrides={"window": 4})
    with pytest.raises(DirectionBankError, match="window_is_one"):
        _run(world, monkeypatch)


# ============================================================= 6. hashes
@pytest.mark.parametrize("target", ["capture.pt", "manifest.json"])
def test_a_capture_file_whose_digest_moved_is_refused(tmp_path, monkeypatch,
                                                      target):
    world = _write_world(tmp_path)
    path = world["capture_dir"] / target
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(DirectionBankError, match="_sha256"):
        _run(world, monkeypatch)


def test_an_eligibility_record_whose_digest_moved_is_refused(tmp_path,
                                                             monkeypatch):
    world = _write_world(tmp_path)
    world["inclusion"].write_text(
        world["inclusion"].read_text(encoding="utf-8") + "\n",
        encoding="utf-8")
    with pytest.raises(DirectionBankError, match="inclusion_manifest_sha256"):
        _run(world, monkeypatch)


def test_a_source_case_manifest_whose_digest_moved_is_refused(tmp_path,
                                                              monkeypatch):
    world = _write_world(tmp_path)
    world["source"].write_text(
        world["source"].read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(DirectionBankError, match="source_case_manifest_sha256"):
        _run(world, monkeypatch)


def test_an_absent_input_is_refused_not_skipped(tmp_path, monkeypatch):
    world = _write_world(tmp_path)
    (world["capture_dir"] / "capture.pt").unlink()
    with pytest.raises(DirectionBankError, match="is not a file"):
        _run(world, monkeypatch)


def test_a_prompt_digest_that_disagrees_across_the_records_is_refused(
        tmp_path, monkeypatch):
    world = _write_world(
        tmp_path, source_overrides={(CASES[2], "H"): "f" * 64})
    with pytest.raises(DirectionBankError,
                       match="prompt_digests_agree_three_ways"):
        _run(world, monkeypatch)


# =============================================================== 7. arms
def test_a_missing_arm_is_refused_and_nothing_substitutes_for_it(tmp_path):
    torch = pytest.importorskip("torch")
    world = _write_world(tmp_path)
    payload = torch.load(world["capture_dir"] / "capture.pt",
                         map_location="cpu", weights_only=True)
    del payload[DB.ARM_TENSOR["H"]]
    torch.save(payload, world["capture_dir"] / "capture.pt")
    with pytest.raises(DirectionBankError, match="A missing arm is refused"):
        DB.capture_arrays(world["capture_dir"] / "capture.pt")


def test_build_bank_refuses_an_absent_baseline():
    acts = _activations()
    del acts["E"]
    with pytest.raises(DirectionBankError, match="baseline of both"):
        DB.build_bank(acts, {d: CASES for d in DB.DIRECTIONS})


def test_build_bank_refuses_an_absent_arm():
    acts = _activations()
    del acts["H"]
    with pytest.raises(DirectionBankError, match="safety needs the H"):
        DB.build_bank(acts, {d: CASES for d in DB.DIRECTIONS})


@pytest.mark.parametrize("ids", [CASES[:9], CASES + ["extra"],
                                 CASES[:9] + [CASES[0]]])
def test_build_bank_refuses_the_wrong_number_or_duplicate_cases(ids):
    acts = _activations()
    with pytest.raises(DirectionBankError, match="expected 10 distinct"):
        DB.build_bank(acts, {"repair": ids, "safety": CASES})


def test_an_unknown_arm_is_refused(tmp_path):
    world = _write_world(tmp_path)
    with pytest.raises(DirectionBankError, match="unknown arm"):
        DB.capture_arrays(world["capture_dir"] / "capture.pt", arms=("B_d2",))


# =========================================================== 8. artefact
def test_the_finished_directions_are_unit_norm(tmp_path, monkeypatch):
    manifest = _run(_write_world(tmp_path), monkeypatch)
    for name in ("repair", "safety"):
        for layer, value in manifest["norms"][name].items():
            assert abs(value - 1.0) < DB.UNIT_NORM_TOLERANCE, (name, layer)


def test_a_direction_that_is_not_unit_norm_is_refused():
    direction = np.ones((3, 2048), dtype=np.float32)
    with pytest.raises(DirectionBankError, match="not unit norm"):
        DB.assert_unit_norm(direction, "bad")


def test_the_artifact_bytes_are_deterministic(tmp_path, monkeypatch):
    world = _write_world(tmp_path)
    first = _run(world, monkeypatch, out=world["root"] / "run_a")
    second = _run(world, monkeypatch, out=world["root"] / "run_b")
    name = BUILD.ARTIFACT_OUT
    assert (world["root"] / "run_a" / name).read_bytes() == (
        world["root"] / "run_b" / name).read_bytes()
    assert (first["outputs"]["artifact"]["sha256"]
            == second["outputs"]["artifact"]["sha256"])
    assert first["outputs"]["arrays"] == second["outputs"]["arrays"]


def test_the_npz_helper_is_deterministic_on_its_own():
    arrays = {"repair_direction": np.eye(3, 2048, dtype=np.float32),
              "safety_direction": np.ones((3, 2048), dtype=np.float32)}
    assert DB.npz_bytes(arrays) == DB.npz_bytes(dict(reversed(list(
        arrays.items()))))


def test_the_artifact_is_read_back_without_pickle(tmp_path, monkeypatch):
    world = _write_world(tmp_path)
    manifest = _run(world, monkeypatch)
    path = world["out"] / BUILD.ARTIFACT_OUT
    arrays = DB.read_npz(path)
    assert sorted(arrays) == sorted(DB.ARTIFACT_ARRAYS)
    for name in DB.ARTIFACT_ARRAYS:
        assert arrays[name].shape == (3, 2048)
        assert arrays[name].dtype == np.float32
        assert DB.array_sha256(arrays[name]) == manifest[
            "outputs"]["arrays"][name]["sha256"]
    with zipfile.ZipFile(path) as archive:
        assert sorted(archive.namelist()) == sorted(
            f"{name}.npy" for name in DB.ARTIFACT_ARRAYS)
        for info in archive.infolist():
            assert info.date_time == DB.ZIP_TIMESTAMP
    assert b"pickle" not in path.read_bytes()


def test_the_artifact_holds_exactly_the_two_directions():
    with pytest.raises(DirectionBankError, match="unexpected"):
        DB.npz_bytes({"repair_direction": np.zeros((3, 2048), np.float32),
                      "safety_direction": np.zeros((3, 2048), np.float32),
                      "orthogonalised_repair": np.zeros((3, 2048), np.float32)})
    with pytest.raises(DirectionBankError, match="missing"):
        DB.npz_bytes({"repair_direction": np.zeros((3, 2048), np.float32)})


def test_the_capture_is_read_with_weights_only():
    source = (REPO_ROOT / "src" / "routegate" / "directionbank.py").read_text(
        encoding="utf-8")
    assert "weights_only=True" in source
    assert "weights_only=False" not in source
    assert "pickle.load" not in source
    assert "allow_pickle=False" in source
    assert "allow_pickle=True" not in source


def test_a_non_empty_output_directory_is_refused(tmp_path, monkeypatch):
    world = _write_world(tmp_path)
    _run(world, monkeypatch)
    with pytest.raises(DirectionBankError, match="written once"):
        _run(world, monkeypatch)


def test_a_dry_run_writes_nothing(tmp_path, monkeypatch):
    world = _write_world(tmp_path)
    manifest = _run(world, monkeypatch, write=False)
    assert not world["out"].exists()
    assert manifest["outputs"]["artifact"]["sha256"]


# ============================================================= 9. parity
def test_the_rebuilt_repair_bank_must_reproduce_the_frozen_comparator(
        tmp_path, monkeypatch):
    manifest = _run(_write_world(tmp_path), monkeypatch)
    parity = manifest["historical_d1_parity"]
    assert parity["agrees"] is True
    assert parity["variant"] == DB.PARITY_VARIANT
    assert set(parity["per_layer_cosine"]) == {"31", "32", "33"}
    for value in parity["per_layer_cosine"].values():
        assert value >= DB.PARITY_MIN_COSINE
    for value in parity["per_layer_max_abs_diff"].values():
        assert value <= DB.PARITY_MAX_ABS_DIFF


def test_a_repair_bank_that_does_not_match_the_frozen_comparator_is_refused(
        tmp_path, monkeypatch):
    world = _write_world(tmp_path)
    wrong = DB.build_bank(world["activations"],
                          {d: CASES for d in DB.DIRECTIONS}).safety
    with pytest.raises(DirectionBankError, match="different construction"):
        _run(world, monkeypatch, parity=wrong)


def test_parity_is_tight_enough_to_catch_a_sign_flip():
    acts = _activations()
    repair = DB.build_direction("repair", acts["B_d1"], acts["E"])
    with pytest.raises(DirectionBankError, match="float32 tolerance"):
        DB.parity_against_frozen(repair, -repair)


def test_parity_refuses_a_comparator_of_the_wrong_shape():
    acts = _activations()
    repair = DB.build_direction("repair", acts["B_d1"], acts["E"])
    with pytest.raises(DirectionBankError, match="expected \\(3, 2048\\)"):
        DB.parity_against_frozen(repair, np.zeros((2, 2048), np.float32))


# ========================================================= 10. abstinence
def _floats(node):
    if isinstance(node, dict):
        for value in node.values():
            yield from _floats(value)
    elif isinstance(node, list):
        for value in node:
            yield from _floats(value)
    elif isinstance(node, float):
        yield node


def test_no_application_strength_reaches_the_artifact(tmp_path, monkeypatch):
    world = _write_world(tmp_path)
    manifest = _run(world, monkeypatch)
    assert manifest["application_strength"]["applied"] is False
    assert manifest["application_strength"]["value"] is None
    banned = {0.15, 0.2}
    present = sorted(v for v in _floats(manifest) if v in banned)
    assert not present, present
    text = (world["out"] / BUILD.MANIFEST_OUT).read_text(encoding="utf-8")
    assert '"strength"' not in text


def test_no_orthogonalisation_is_applied(tmp_path, monkeypatch):
    world = _write_world(tmp_path)
    manifest = _run(world, monkeypatch)
    assert manifest["orthogonalisation"]["applied"] is False

    # the two directions are measurably non-orthogonal, so "we did not
    # orthogonalise" is a fact about the vectors and not only about the prose
    cosines = manifest["diagnostics"]["repair_safety_cosine_per_layer"]
    assert set(cosines) == {"31", "32", "33"}
    assert all(abs(value) > 0.2 for value in cosines.values()), cosines

    arrays = DB.read_npz(world["out"] / BUILD.ARTIFACT_OUT)
    measured = DB.per_layer_cosine(arrays["repair_direction"],
                                   arrays["safety_direction"])
    for layer, value in zip(("31", "32", "33"), measured):
        assert abs(value - cosines[layer]) < 1e-6


def test_the_manifest_records_the_recipe_and_the_exclusions(tmp_path,
                                                            monkeypatch):
    world = _write_world(tmp_path)
    manifest = _run(world, monkeypatch)

    construction = manifest["construction"]
    assert construction["subtraction_order"] == {"repair": "B_d1 - E",
                                                 "safety": "H - E"}
    assert construction["per_case_normalisation"] is True
    assert construction["identical_recipe_for_both_directions"] is True
    assert len(construction["recipe"]) == 4

    geometry = manifest["geometry"]
    assert geometry["historical_layers"] == [31, 32, 33]
    assert geometry["routegate_states"] == [32, 33, 34]
    assert geometry["layer_to_state"] == {"31": 32, "32": 33, "33": 34}
    assert geometry["token_position"] == DB.TOKEN_POSITION

    eligibility = manifest["eligibility"]
    assert eligibility["excluded"]["rule"] == DB.BASELINE_ONLY_RULE
    assert eligibility["excluded"]["arms"] == list(DB.BASELINE_ONLY_ARMS)
    for direction, spec in eligibility["directions"].items():
        assert spec["n_cases"] == 10
        assert spec["case_ids"] == CASES
        assert len(spec["eligibility_record_sha256"]) == 10
        assert spec["rule"] == DB.DIRECTIONS[direction]["rule"]

    assert manifest["model"]["revision"] == DB.MODEL_REVISION
    assert manifest["loads_a_model"] is False
    assert manifest["tensors"]["activation_shape"] == [10, 3, 2048]
    assert manifest["tensors"]["direction_shape"] == [3, 2048]
    assert manifest["checklist"]["all_passed"] is True


def test_the_builder_loads_no_model_and_names_no_machine_path():
    source = (REPO_ROOT / "experiments" / "dual_direction_controller"
              / "build_validated_direction_bank.py").read_text(encoding="utf-8")
    for banned in ("/home/", "from_pretrained", "AutoModel", "cuda",
                   "generate(", "requests", "urlopen"):
        assert banned not in source, banned
    for path in (REPO_ROOT / "src" / "routegate" / "directionbank.py",
                 REPO_ROOT / "manifests" / "rq2_geometry_capture.json"):
        assert "/home/" not in path.read_text(encoding="utf-8"), path.name


# ============================ 11. the committed configuration and the code
#
# `configs/dual_direction_controller.json` states the direction decision and
# `src/routegate/directionbank.py` performs it. Two records of one decision
# drift apart silently: the config is prose and JSON, nothing imports it, and a
# change to the module would leave it describing a construction that no longer
# happens. This binds them.
CONFIG_PATH = REPO_ROOT / "configs" / "dual_direction_controller.json"


def _named(module, fragment):
    """Public attributes of a module whose name contains a fragment."""
    return sorted(name for name in dir(module)
                  if not name.startswith("_") and fragment in name.lower())


def _by_the_configs_steps(arm_block, baseline_block):
    """The four steps the config declares, performed literally, in order.

    Written out here rather than called from the module, so that the comparison
    below is between the config's description and the module's behaviour and
    not between the module and itself.
    """
    delta = arm_block - baseline_block                                  # 1
    unit = delta / np.linalg.norm(delta, axis=-1, keepdims=True)        # 2
    mean = unit.mean(axis=0)                                            # 3
    return (mean / np.linalg.norm(mean, axis=-1, keepdims=True)         # 4
            ).astype(np.float32)


def test_the_committed_configuration_agrees_with_the_implementation():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    construction = config["direction_construction"]

    # --- the two directions: arm, baseline, sign, rule, population ---------
    for name, spec in construction["directions"].items():
        assert spec["arm"] == DB.DIRECTIONS[name]["arm"], name
        assert spec["baseline"] == DB.BASELINE_ARM, name
        assert spec["delta"] == f"{spec['arm']} - {DB.BASELINE_ARM}", name
        assert spec["inclusion_rule"] == DB.DIRECTIONS[name]["rule"], name
        assert spec["n_validated_cases"] == DB.N_CASES, name
    assert set(construction["directions"]) == set(DB.DIRECTIONS)
    assert construction["directions"]["repair"]["delta"] == "B_d1 - E"
    assert construction["directions"]["safety"]["delta"] == "H - E"

    # the arm binding is what keeps the 3-of-10 primary actuator out
    binding = construction["arm_binding_note"]
    assert DB.BASELINE_ONLY_RULE in binding
    for arm in DB.BASELINE_ONLY_ARMS:
        assert arm in binding, arm
        assert arm not in {s["arm"] for s in DB.DIRECTIONS.values()}

    # --- layers and states -------------------------------------------------
    recipe = construction["recipe"]
    assert recipe["layers"] == list(DB.HISTORICAL_LAYERS) == [31, 32, 33]
    read_states = config["read_states"]
    assert read_states["historical_layers"] == list(DB.HISTORICAL_LAYERS)
    assert read_states["states"] == list(DB.BANK_STATES) == [32, 33, 34]
    assert read_states["layer_to_state"] == {
        str(layer): state
        for layer, state in zip(DB.HISTORICAL_LAYERS, DB.BANK_STATES)}

    # --- the recipe: declared order, then declared order performed ---------
    steps = recipe["steps"]
    assert len(steps) == 4
    assert "delta =" in steps[0] and f"- {DB.BASELINE_ARM}_activation" in steps[0]
    assert "/ ||delta||_2" in steps[1]
    assert "average the ten" in steps[2]
    assert "L2-normalise" in steps[3] and "mean" in steps[3]
    assert recipe["per_case_normalisation_before_averaging"] is True
    assert recipe["identical_recipe_for_both_directions"] is True
    assert recipe["subtraction_order"].startswith("arm minus baseline")

    # One case is given a delta fifty times longer than the rest. On an even
    # fixture, normalising before or after averaging gives the same direction
    # to within rounding, and this comparison would then be discriminating on
    # float noise rather than on the order the config declares. Skewed, the
    # two orders genuinely diverge -- which is the property the config states
    # as "every case carries one vote".
    acts = _activations()
    skewed = {arm: block.copy() for arm, block in acts.items()}
    for arm in ("B_d1", "H"):
        skewed[arm][0] = acts[DB.BASELINE_ARM][0] + 50.0 * (
            acts[arm][0] - acts[DB.BASELINE_ARM][0])

    for name, spec in sorted(DB.DIRECTIONS.items()):
        arm_block = skewed[spec["arm"]]
        baseline_block = skewed[DB.BASELINE_ARM]
        declared = _by_the_configs_steps(arm_block, baseline_block)
        implemented = DB.build_direction(name, arm_block, baseline_block)
        assert np.array_equal(declared, implemented), name

        # averaging before normalising is a different direction, not a
        # different rounding: the order in the config is doing real work
        delta = arm_block - baseline_block
        mean_first = delta.mean(axis=0)
        mean_first = mean_first / np.linalg.norm(mean_first, axis=-1,
                                                 keepdims=True)
        assert all(value < 0.99 for value
                   in DB.per_layer_cosine(mean_first, implemented)), name

        # and dropping the final renormalisation leaves it off the unit sphere
        no_renorm = (delta / np.linalg.norm(delta, axis=-1, keepdims=True)
                     ).mean(axis=0)
        assert not np.allclose(np.linalg.norm(no_renorm, axis=-1), 1.0,
                               atol=DB.UNIT_NORM_TOLERANCE), name

    # --- no orthogonalisation in the primary method ------------------------
    assert construction["primary_method"]["orthogonalisation_applied"] is False
    assert _named(DB, "orthogon") == []
    assert _named(BUILD, "orthogon") == []
    assert DB.ARTIFACT_ARRAYS == ("repair_direction", "safety_direction")
    assert not any("orthogon" in name for name in DB.ARTIFACT_ARRAYS)

    # --- no steering strength embedded in the bank -------------------------
    strength = construction["steering_strength"]
    assert strength["embedded_in_the_direction_bank"] is False
    assert strength["value"] is None
    assert _named(DB, "strength") == []
    assert _named(BUILD, "strength") == []
    # and it is still an open question elsewhere, not silently answered here
    assert config["unresolved"][
        "sigmoid_to_maximum_strength_mapping"] == "UNRESOLVED"

    # --- orthogonalised safety: predeclared, unimplemented, unrun ----------
    ablation = config["predeclared_ablations"]["orthogonalised_safety"]
    assert ablation["status"] == "PREDECLARED_NOT_IMPLEMENTED"
    assert "not the primary method" in ablation["role"]
    assert "whether it is run at all" in ablation["open"]

    # --- the record points at the code that performs it --------------------
    implemented_by = construction["implemented_by"]
    assert implemented_by["arithmetic"] == "src/routegate/directionbank.py"
    for relative in implemented_by.values():
        assert (REPO_ROOT / relative).is_file(), relative
    assert construction["parity_gate"]["reference"] == DB.PARITY_VARIANT
    produces = config["phases"]["2_direction_pair"]["produces"]
    assert BUILD.ARTIFACT_OUT in produces and BUILD.MANIFEST_OUT in produces

    # --- the question is settled, and recorded as settled ------------------
    settled = "direction_normalisation_and_orthogonalisation"
    assert settled not in config["unresolved"]
    assert config["resolved"][settled]["recorded_in"] == "direction_construction"
