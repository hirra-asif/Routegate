"""The phase-2 closure: every way it could quietly stop being the closure.

Nothing here loads a model, generates a token, judges a cell, trains anything
or calls a service. The committed artifacts are read read-only; every mutation
is made on a copy in ``tmp_path``.

Organised by the failure each group prevents:

    1. outcome     the closure and the result it was taken from disagree
    2. immutable   the artifact is edited and nothing notices
    3. retirement  d_S or a_S survives the decision that removed them
    4. history     closing the phase deletes what it declared
    5. reruns      the completed protocol is run again by accident
    6. runner      the writer touches something it must not
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate import phase2 as P                                   # noqa: E402
from routegate import phase2closure as C                            # noqa: E402
from routegate.dualsteer import (                                   # noqa: E402
    EXPLORATORY_PROTOCOL,
    FOCUSED_PROTOCOL,
    PHASE2_PROTOCOL,
)
from routegate.singledirection import (                             # noqa: E402
    CONTROLLER_OUTPUTS,
    build_single_direction_specification,
)
from routegate.validation import ValidationError                    # noqa: E402

CLOSER = (REPO_ROOT / "experiments" / "dual_direction_controller"
          / "close_phase2.py")
RUNNER = (REPO_ROOT / "experiments" / "dual_direction_controller"
          / "run_causal_screen.py")
PATHS = REPO_ROOT / "configs" / "paths.local.json"


def _readings_path():
    """The completed readings on this machine, or None.

    They are a result under the private data root and are never committed, so
    their absence is a fact about the machine rather than a failure.
    """
    if not PATHS.is_file():
        return None
    try:
        root = json.loads(PATHS.read_text(encoding="utf-8")).get(
            "focused_direction_validation_dir")
    except Exception:
        return None
    if not root:
        return None
    path = (Path(root) / "phase2" / "judging" / "readings"
            / C.PHASE2_READINGS_FILENAME)
    return path if path.is_file() else None


needs_readings = pytest.mark.skipif(
    _readings_path() is None,
    reason="the completed phase-2 readings are not reachable on this machine")


@pytest.fixture
def closure(tmp_path):
    """A writable copy of the committed closure."""
    path = tmp_path / C.PHASE2_CLOSURE_MANIFEST.name
    path.write_text(C.PHASE2_CLOSURE_MANIFEST.read_text("utf-8"), "utf-8")
    return path


def _rewrite(path: Path, mutate) -> Path:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def _load(module_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# =============================================================== 1. outcome
def test_the_pinned_outcome_is_the_rules_own_failure_branch():
    """NO_STRENGTH_QUALIFIES is not a string somebody typed into the closure:
    it is what the frozen evaluator returns when nothing qualifies."""
    grid = {0.05: {"combined_recovered": 7, "safety_only_harmful": 1},
            0.10: {"combined_recovered": 10, "safety_only_harmful": 3},
            0.15: {"combined_recovered": 9, "safety_only_harmful": 3},
            0.20: {"combined_recovered": 10, "safety_only_harmful": 3}}
    out = P.evaluate_phase2_decision(grid)
    assert out["outcome"] == C.PHASE2_OUTCOME
    assert out["selected"] is None
    assert out["qualifying_strengths"] == []


def test_the_committed_closure_carries_the_outcome_and_the_readings_digest():
    payload = json.loads(C.PHASE2_CLOSURE_MANIFEST.read_text("utf-8"))
    assert payload["outcome"]["outcome"] == C.PHASE2_OUTCOME
    assert payload["outcome"]["selected"] is None
    assert payload["outcome"]["qualifying_strengths"] == []
    assert payload["readings"]["sha256"] == C.PHASE2_READINGS_SHA256
    assert len(C.PHASE2_READINGS_SHA256) == 64
    assert payload["outcome"]["population_reportable"] is False
    assert payload["outcome"]["is_a_population_estimate"] is False


def test_the_outcome_reaches_the_controller_specification():
    """Propagation, end to end: the readings' outcome is why the one-direction
    specification exists, and the specification says so by digest."""
    specification = build_single_direction_specification()
    assert specification["phase2"]["outcome"] == C.PHASE2_OUTCOME
    assert specification["phase2"]["readings_sha256"] == C.PHASE2_READINGS_SHA256
    assert specification["phase2"]["closure"] == C.CLOSURE_NAME
    committed = json.loads(C.PHASE2_CLOSURE_MANIFEST.read_text("utf-8"))
    assert committed["continuation"]["selected"] == specification["name"]


@needs_readings
def test_the_real_readings_hash_and_decide_exactly_what_the_closure_claims():
    path = _readings_path()
    summary = C.verify_phase2_readings(path)
    assert summary["sha256"] == C.PHASE2_READINGS_SHA256
    assert summary["outcome"] == C.PHASE2_OUTCOME
    assert summary["checks"].all_passed


@needs_readings
def test_re_evaluating_the_readings_own_counts_reproduces_the_outcome():
    """The closure's outcome is re-derived from the readings' per-strength
    counts through the frozen rule, not read off the readings' own label."""
    readings = json.loads(_readings_path().read_text(encoding="utf-8"))
    grid = {float(alpha): counts
            for alpha, counts in readings["c_counts_by_strength"].items()}
    out = P.evaluate_phase2_decision(grid)
    assert out["outcome"] == C.PHASE2_OUTCOME
    assert out["selected"] is None
    assert out["qualifying_strengths"] == []


def test_a_readings_file_with_another_outcome_is_refused(tmp_path):
    payload = {"schema": C.PHASE2_READINGS_SCHEMA, "phase": 2,
               "protocol": PHASE2_PROTOCOL,
               "d_decision": {"outcome": "STRENGTH_SELECTED"}}
    path = tmp_path / C.PHASE2_READINGS_FILENAME
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError, match="readings_sha256"):
        C.verify_phase2_readings(path)


def test_an_absent_readings_file_stops_the_closure(tmp_path):
    with pytest.raises(C.Phase2ClosureError, match="does not exist"):
        C.verify_phase2_readings(tmp_path / "nothing.json")


# ============================================================= 2. immutable
def test_the_committed_closure_passes_every_gate():
    outcome = C.verify_phase2_closure()
    assert outcome["checks"].all_passed
    assert outcome["checks"].as_dict()["checks_run"] > 10


def test_the_closure_digest_is_a_digest_of_the_closure():
    payload = json.loads(C.PHASE2_CLOSURE_MANIFEST.read_text("utf-8"))
    assert payload[C.CLOSURE_DIGEST_FIELD] == C.closure_sha256(payload)
    assert C.CLOSURE_DIGEST_FIELD not in C.closure_canonical_json(payload)


def test_an_edited_closure_fails_its_own_digest(closure):
    """Any edit, anywhere in the file, including one nothing else compares."""
    _rewrite(closure, lambda m: m.__setitem__("note_added_later", "hello"))
    with pytest.raises(ValidationError, match="closure_digest_recomputes"):
        C.verify_phase2_closure(closure)


@pytest.mark.parametrize("field", ["outcome", "readings", "actuators",
                                   "safety_probe", "controller_outputs",
                                   "protocol_closed", "probe_head_run"])
def test_an_edited_closure_field_is_refused(closure, field):
    def mutate(payload):
        payload[field] = {"moved": True}
        payload[C.CLOSURE_DIGEST_FIELD] = C.closure_sha256(payload)
    _rewrite(closure, mutate)
    with pytest.raises(ValidationError, match=f"closure_{field}"):
        C.verify_phase2_closure(closure)


def test_a_re_digested_edit_is_still_refused(closure):
    """Recomputing the digest after an edit does not launder it: the package
    is the second witness, and it did not move."""
    def mutate(payload):
        payload["readings"]["sha256"] = "0" * 64
        payload[C.CLOSURE_DIGEST_FIELD] = C.closure_sha256(payload)
    _rewrite(closure, mutate)
    with pytest.raises(ValidationError, match="closure_readings"):
        C.verify_phase2_closure(closure)


def test_an_absent_closure_stops_the_run(tmp_path):
    with pytest.raises(C.Phase2ClosureError, match="does not exist"):
        C.verify_phase2_closure(tmp_path / "nothing.json")


def test_the_closure_holds_no_prompt_or_response_text():
    payload = json.loads(C.PHASE2_CLOSURE_MANIFEST.read_text(encoding="utf-8"))
    assert payload["contains_prompt_or_response_text"] is False
    # scanned with the declaration itself removed: the field that says there is
    # no prompt text spells "prompt_or_response_text", and matching on it would
    # be the check failing on its own disclaimer.
    payload.pop("contains_prompt_or_response_text")
    text = json.dumps(payload)
    for banned in ("prompt_text", "response_text", "completion_text"):
        assert banned not in text, banned


def test_the_closure_carries_the_retained_probe_head_run_identity():
    payload = json.loads(C.PHASE2_CLOSURE_MANIFEST.read_text("utf-8"))
    run = payload["probe_head_run"]
    assert run["checks_run"] == 254 and run["all_passed"] is True
    assert run["refit_in_this_patch"] is False
    assert run["directory"] == C.PROBE_HEAD_RUN_DIRECTORY
    for name in ("run_manifest.json", "validation.json", "metrics.json",
                 "oof_predictions.csv", "fold_assignments.csv"):
        assert len(run["artifacts"][name]) == 64, name


def test_a_probe_head_run_that_has_moved_since_it_was_pinned_is_refused(
        tmp_path):
    run = tmp_path / "probe_heads_v1"
    run.mkdir()
    for name in C.PROBE_HEAD_RUN["artifacts"]:
        (run / name).write_text("not the pinned artifact", encoding="utf-8")
    with pytest.raises(ValidationError, match="probe_head_artifact_sha256"):
        C.verify_probe_head_run(run)


def test_an_unreachable_probe_head_run_is_a_fact_not_a_failure(tmp_path):
    """The run directory is an output and is not committed."""
    outcome = C.verify_probe_head_run(tmp_path / "absent")
    assert outcome["reachable"] is False


# ============================================================ 3. retirement
def test_the_closure_retires_h_minus_e_as_an_actuator():
    payload = json.loads(C.PHASE2_CLOSURE_MANIFEST.read_text("utf-8"))
    retired = payload["actuators"]["retired_actuator"]
    assert retired["direction"] == "d_S = H - E"
    assert retired["status"] == "RETIRED"
    assert retired["active"] is False
    assert retired["may_be_applied"] is False
    assert retired["may_carry_a_coefficient"] is False
    assert retired["retired_by"] == C.PHASE2_OUTCOME
    assert retired["the_bank_is_not_rewritten"] is True


def test_the_closure_emits_no_safety_coefficient():
    payload = json.loads(C.PHASE2_CLOSURE_MANIFEST.read_text("utf-8"))
    outputs = payload["controller_outputs"]
    assert outputs["outputs"] == list(CONTROLLER_OUTPUTS) == ["g_R"]
    assert outputs["output_count"] == 1
    assert outputs["a_S_is_emitted"] is False


def test_the_closure_retains_the_repair_direction_at_the_validated_strength():
    payload = json.loads(C.PHASE2_CLOSURE_MANIFEST.read_text("utf-8"))
    retained = payload["actuators"]["retained_actuator"]
    assert retained["direction"] == "d_R = B_d1 - E"
    assert retained["status"] == "RETAINED"
    assert retained["active"] is True
    assert retained["validated_strength"] == 0.15


def test_the_closure_keeps_the_safety_probe_as_a_sensor_and_not_a_steerer():
    """Dropping H - E as an ACTUATOR is not dropping the safety READING."""
    payload = json.loads(C.PHASE2_CLOSURE_MANIFEST.read_text("utf-8"))
    probe = payload["safety_probe"]
    assert probe["status"] == "RETAINED"
    assert probe["role"] == "sensor"
    assert probe["may_reduce_the_repair_gate"] is True
    assert probe["may_veto_the_repair_gate"] is True
    assert probe["may_produce_a_steering_direction"] is False
    assert probe["may_produce_a_controller_output"] is False


def test_the_retirement_is_the_rules_own_predeclared_branch():
    fallback = P.PHASE2_DECISION_RULE["if_no_strength_qualifies"]
    assert "REMOVED as a final steering actuator" in fallback["decision"]
    assert "one-direction controller" in fallback["what_continues"]
    assert "brake or veto" in fallback["the_probe_is_retained"]


# =============================================================== 4. history
def test_closing_the_phase_does_not_delete_what_it_declared():
    preserved = C.assert_historical_artifacts_are_preserved()
    assert set(preserved) == {"phase2_predeclaration", "phase2_selection",
                              "dual_direction_config"}
    for entry in preserved.values():
        assert len(entry["sha256"]) == 64


def test_the_predeclaration_still_passes_its_own_gates_unchanged():
    """The closure is a second file, not an edit to the first."""
    outcome = P.verify_phase2_predeclaration()
    assert outcome["checks"].all_passed
    assert outcome["committed"]["phase2_generations_run"] == 0
    assert outcome["committed"]["status"] == "FROZEN"


def test_the_predeclaration_carries_no_outcome():
    """A predeclaration edited to record its own result is not one."""
    text = P.PHASE2_PREDECLARATION.read_text(encoding="utf-8")
    assert C.PHASE2_OUTCOME not in text
    assert C.PHASE2_READINGS_SHA256 not in text
    payload = json.loads(text)
    assert "outcome" not in payload
    assert "closure" not in payload


def test_the_closure_says_it_is_not_the_predeclaration():
    payload = json.loads(C.PHASE2_CLOSURE_MANIFEST.read_text("utf-8"))
    block = payload["is_not_the_predeclaration"]
    assert block["predeclaration"] == P.PHASE2_PREDECLARATION.name
    assert block["both_preserved_unchanged"] is True
    assert payload["written_after_the_run"] is True
    assert payload["immutable"] is True


# ================================================================ 5. reruns
def test_the_completed_phase_2_protocol_is_closed():
    assert PHASE2_PROTOCOL in C.CLOSED_PROTOCOLS
    closed = C.CLOSED_PROTOCOLS[PHASE2_PROTOCOL]
    assert closed["status"] == "CLOSED"
    assert closed["outcome"] == C.PHASE2_OUTCOME
    assert closed["generations_run"] == 144
    assert closed["reproduction_is_preserved"] is True


def test_an_ordinary_rerun_of_the_closed_protocol_is_refused():
    with pytest.raises(C.Phase2ClosureError, match="is CLOSED"):
        C.assert_protocol_is_open(PHASE2_PROTOCOL)


def test_the_refusal_names_the_reproduction_path_rather_than_hiding_it():
    with pytest.raises(C.Phase2ClosureError) as caught:
        C.assert_protocol_is_open(PHASE2_PROTOCOL)
    assert C.REPRODUCTION_ONLY_FLAG in str(caught.value)


def test_an_explicit_reproduction_is_allowed_and_says_so():
    from routegate.validation import Checklist
    checks = Checklist("t")
    outcome = C.assert_protocol_is_open(PHASE2_PROTOCOL,
                                        reproduction_only=True, checks=checks)
    assert outcome["run_as"] == "reproduction_only"
    assert any(entry["check"] == "closed_protocol_reproduction"
               for entry in checks.entries)


@pytest.mark.parametrize("name", [EXPLORATORY_PROTOCOL, FOCUSED_PROTOCOL])
def test_the_open_protocols_are_untouched(name):
    assert C.assert_protocol_is_open(name)["status"] == "OPEN"


def test_the_runner_gates_the_closed_protocol_before_it_reads_anything():
    """Not "it is called somewhere" -- it is called before the configuration is
    opened, before the bank is hashed and before a corpus row is read."""
    source = RUNNER.read_text(encoding="utf-8")
    gate = source.index("assert_protocol_is_open(")
    for later in ('config = json.loads(Path(args.config)',
                  "bank = load_validated_bank(",
                  "rows = load_screen_rows(",
                  "if args.dry_run:",
                  "from routegate.runtime import load_model_and_tokenizer"):
        assert gate < source.index(later), later


def test_the_runner_exposes_the_reproduction_flag():
    source = RUNNER.read_text(encoding="utf-8")
    assert "REPRODUCTION_ONLY_FLAG" in source
    sys.path.insert(0, str(RUNNER.parent))
    try:
        module = _load(RUNNER, "_run_causal_screen")
    finally:
        sys.path.remove(str(RUNNER.parent))
    options = [option
               for action in module.build_parser()._actions
               for option in action.option_strings]
    assert C.REPRODUCTION_ONLY_FLAG in options


# ================================================================ 6. runner
def test_the_closure_entry_point_loads_no_model_and_trains_nothing():
    source = CLOSER.read_text(encoding="utf-8")
    for banned in ("torch", "load_model_and_tokenizer", "PatchController",
                   "cuda", "generate(", "sklearn", "fit("):
        assert banned not in source, banned


def test_the_closure_entry_point_writes_no_historical_artifact():
    """It reads the predeclaration, the selection and the dual config. It has
    no path that writes any of them."""
    source = CLOSER.read_text(encoding="utf-8")
    written = source[source.index("def write_one"):]
    for historical in ("PHASE2_PREDECLARATION", "PHASE2_SELECTION_MANIFEST",
                       "dual_direction_controller.json"):
        assert historical not in written, historical


def test_the_closure_writer_refuses_to_overwrite_a_different_artifact(tmp_path):
    module = _load(CLOSER, "_close_phase2")
    from routegate.validation import Checklist
    path = tmp_path / C.PHASE2_CLOSURE_MANIFEST.name
    path.write_text('{"schema": "something else"}\n', encoding="utf-8")
    with pytest.raises(C.Phase2ClosureError, match="immutable"):
        module.write_one(path, C.build_phase2_closure(), True,
                         Checklist("t"), "closure")


def test_the_closure_writer_is_idempotent_on_an_identical_artifact(tmp_path):
    module = _load(CLOSER, "_close_phase2")
    from routegate.validation import Checklist
    path = tmp_path / C.PHASE2_CLOSURE_MANIFEST.name
    payload = C.build_phase2_closure()
    module.write_one(path, payload, True, Checklist("t"), "closure")
    before = path.read_text(encoding="utf-8")
    module.write_one(path, payload, True, Checklist("t"), "closure")
    assert path.read_text(encoding="utf-8") == before
    assert before == C.PHASE2_CLOSURE_MANIFEST.read_text(encoding="utf-8")


def test_the_closure_is_derivable_and_carries_no_timestamp():
    """Re-derived on any machine at any time, it is the same bytes -- which is
    what makes "the committed file is what the package declares" a comparison
    rather than a judgement."""
    first = C.build_phase2_closure()
    second = C.build_phase2_closure()
    assert first == second
    assert "created_utc" not in first and "recorded_utc" not in first
