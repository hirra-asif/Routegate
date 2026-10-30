"""The named focused protocol: 871 generations, repair direction only.

Nothing here loads a model, allocates a GPU, calls an API, generates, judges or
reads sealed material. Every row is invented; the protocol arithmetic and the
completeness rules are what is under test.

Organised by the failure each group is there to prevent:

    1. inventory    the protocol is not exactly 871, split 445 / 426
    2. scope        the safety direction is tested, or alpha_S drifts above 0
    3. isolation    the two protocols contaminate one another
    4. reportable   a short, filtered or sharded run is treated as complete
    5. population   the 20 legacy regressions select or define the population
    6. declared     the config and the package disagree about the protocol
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate import dualsteer as DS                                # noqa: E402
from routegate.dualsteer import DualSteerError                       # noqa: E402
from routegate.validation import (                                   # noqa: E402
    Checklist,
    assert_config_is_sealed_free,
    looks_sealed,
)

FOCUSED = DS.FOCUSED_PROTOCOL
EXPLORATORY = DS.EXPLORATORY_PROTOCOL
CONFIG = json.loads(
    (REPO_ROOT / "configs" / "dual_direction_controller.json").read_text("utf-8"))


def _load_script(name: str):
    path = REPO_ROOT / "experiments" / "dual_direction_controller" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_ddc_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUN = _load_script("run_causal_screen")
CONSOLIDATE = _load_script("consolidate_causal_screen")


def _rows(n_e=DS.N_E_ROWS, n_h=DS.N_H_ROWS):
    rows = []
    for index in range(n_e):
        rows.append({"row_id": f"e:{index:04d}", "row_class": DS.CLASS_E,
                     "prompt_sha256": f"{index:064x}"})
    for index in range(n_h):
        rows.append({"row_id": f"h:{index:04d}", "row_class": DS.CLASS_H,
                     "prompt_sha256": f"{index + 1000:064x}"})
    return rows


# ============================================================ 1. inventory
def test_the_focused_protocol_is_exactly_eight_hundred_and_seventy_one():
    protocol = DS.protocol_for(FOCUSED)
    items = DS.build_inventory(_rows(), protocol)
    counts = DS.assert_inventory(items, protocol=protocol)
    assert counts["total"] == 871 == protocol.expected_generations
    assert counts["E_rows"] == 89 and counts["E_arms"] == 5
    assert counts["H_rows"] == 213 and counts["H_arms"] == 2
    assert counts["E_generations"] == 445
    assert counts["H_generations"] == 426
    assert counts["E_generations"] + counts["H_generations"] == 871
    assert counts["protocol"] == FOCUSED


def test_the_focused_arms_are_the_declared_five_and_two():
    protocol = DS.protocol_for(FOCUSED)
    assert [arm.arm_id for arm in protocol.arms_by_class[DS.CLASS_E]] == [
        "baseline", "repair_005", "repair_010", "repair_015", "repair_020"]
    assert [arm.arm_id for arm in protocol.arms_by_class[DS.CLASS_H]] == [
        "baseline", "repair_015"]


def test_every_focused_generation_appears_once():
    items = DS.build_inventory(_rows(), DS.protocol_for(FOCUSED))
    pairs = [(item["row_id"], item["arm_id"]) for item in items]
    assert len(pairs) == len(set(pairs)) == 871


def test_the_focused_inventory_order_is_deterministic():
    protocol = DS.protocol_for(FOCUSED)
    first = DS.build_inventory(_rows(), protocol)
    second = DS.build_inventory(list(reversed(_rows())), protocol)
    assert ([(i["row_id"], i["arm_id"]) for i in first]
            == [(i["row_id"], i["arm_id"]) for i in second])
    assert first[0]["row_class"] == DS.CLASS_E
    assert first[-1]["row_class"] == DS.CLASS_H


@pytest.mark.parametrize("n_e,n_h", [(88, 213), (89, 212), (90, 213)])
def test_a_corpus_of_the_wrong_size_is_refused(n_e, n_h):
    protocol = DS.protocol_for(FOCUSED)
    items = DS.build_inventory(_rows(n_e, n_h), protocol)
    with pytest.raises(DualSteerError, match=f"protocol {FOCUSED} is"):
        DS.assert_inventory(items, protocol=protocol)


def test_the_focused_shards_partition_the_inventory_exactly():
    items = DS.build_inventory(_rows(), DS.protocol_for(FOCUSED))
    for count in (1, 4, 8):
        pieces = [RUN.shard(items, index, count)
                  for index in range(1, count + 1)]
        assert sum(len(piece) for piece in pieces) == 871
        assert [item for piece in pieces for item in piece] == items


def test_four_shards_split_871_as_evenly_as_it_divides():
    items = DS.build_inventory(_rows(), DS.protocol_for(FOCUSED))
    sizes = [len(RUN.shard(items, index, 4)) for index in range(1, 5)]
    assert sum(sizes) == 871
    assert sizes == [218, 218, 218, 217]


# =============================================================== 2. scope
def test_the_focused_protocol_does_not_test_the_safety_direction():
    protocol = DS.protocol_for(FOCUSED)
    assert protocol.tests_safety_direction is False
    for arms in protocol.arms_by_class.values():
        for arm in arms:
            assert arm.alpha_safety == 0.0, arm.arm_id


def test_no_safety_or_combined_arm_is_in_the_focused_protocol():
    protocol = DS.protocol_for(FOCUSED)
    ids = {arm.arm_id for arms in protocol.arms_by_class.values()
           for arm in arms}
    assert not [a for a in ids if a.startswith("safety_")]
    assert not [a for a in ids if "safety" in a]
    assert ids == {"baseline", "repair_005", "repair_010", "repair_015",
                   "repair_020"}


def test_every_focused_arm_uses_only_the_repair_direction():
    items = DS.build_inventory(_rows(), DS.protocol_for(FOCUSED))
    assert {item["alpha_S"] for item in items} == {0.0}
    assert {item["alpha_R"] for item in items} == {0.0, 0.05, 0.10, 0.15, 0.20}


# ============================================================ 3. isolation
def test_the_exploratory_protocol_is_unchanged():
    protocol = DS.protocol_for(EXPLORATORY)
    assert protocol.expected_generations == 3214
    assert protocol.tests_safety_direction is True
    items = DS.build_inventory(_rows(), protocol)
    counts = DS.assert_inventory(items, protocol=protocol)
    assert counts["total"] == 3214
    assert counts["E_arms"] == 5 and counts["H_arms"] == 13
    # and the module default is still the exploratory screen
    assert DS.DEFAULT_PROTOCOL == EXPLORATORY
    assert DS.build_inventory(_rows()) == items


def test_one_output_directory_holds_one_protocol():
    assert "protocol" in DS.RUN_LOCKED_FIELDS
    base = {"row_id": "h:0001", "row_class": DS.CLASS_H, "arm_id": "baseline",
            "alpha_R": 0.0, "alpha_S": 0.0,
            "direction_bank_sha256": "a" * 64,
            "generation_config_sha256": "b" * 64, "protocol": FOCUSED}
    expected = dict(base)
    other = DS.make_record(**dict(base, protocol=EXPLORATORY))
    with pytest.raises(DualSteerError, match="different experiment"):
        DS.verify_consistency([other], expected)


def test_an_unknown_protocol_is_refused_never_defaulted():
    with pytest.raises(DualSteerError, match="unknown protocol"):
        DS.protocol_for("focused_direction_validation_v2")
    with pytest.raises(DualSteerError, match="unknown protocol"):
        DS.protocol_for("")


def test_both_runners_expose_the_protocol_by_name():
    for module in (RUN, CONSOLIDATE):
        options = [option for action in module.build_parser()._actions
                   for option in action.option_strings]
        assert "--protocol" in options
        choices = [action.choices for action in module.build_parser()._actions
                   if "--protocol" in action.option_strings][0]
        assert sorted(choices) == sorted(DS.PROTOCOLS)


def test_b_rows_are_not_steered_by_either_protocol():
    for name in DS.PROTOCOLS:
        protocol = DS.protocol_for(name)
        assert sorted(protocol.arms_by_class) == ["E", "H"]
        with pytest.raises(DualSteerError, match="B rows built"):
            protocol.arms_for("B")


# =========================================================== 4. reportable
def test_a_filtered_run_is_a_smoke_test_under_either_protocol():
    source = (REPO_ROOT / "experiments" / "dual_direction_controller"
              / "run_causal_screen.py").read_text(encoding="utf-8")
    assert "smoke_reasons" in source
    assert "--only-row-id" in source and "--only-arm" in source
    assert "--limit" in source
    # smoke is run-locked, so a smoke record cannot join a protocol's records
    assert "smoke" in DS.RUN_LOCKED_FIELDS


def test_an_arm_the_protocol_does_not_run_is_refused_by_name():
    source = (REPO_ROOT / "experiments" / "dual_direction_controller"
              / "run_causal_screen.py").read_text(encoding="utf-8")
    assert "protocol_arms" in source
    assert "does not run" in source
    # and an arm nobody declared is still its own, separate refusal
    assert "not predeclared" in source


def test_completeness_is_established_by_the_consolidator_not_a_shard():
    source = (REPO_ROOT / "experiments" / "dual_direction_controller"
              / "run_causal_screen.py").read_text(encoding="utf-8")
    assert "completeness is established by" in source
    consolidator = (REPO_ROOT / "experiments" / "dual_direction_controller"
                    / "consolidate_causal_screen.py").read_text("utf-8")
    assert "complete_protocol" in consolidator
    assert "protocol.expected_generations" in consolidator


def test_the_consolidator_counts_against_the_protocol_it_was_given():
    source = (REPO_ROOT / "experiments" / "dual_direction_controller"
              / "consolidate_causal_screen.py").read_text(encoding="utf-8")
    assert "protocol_matches_the_records" in source
    assert "expected_by_class()" in source
    # the default is still the exploratory split
    assert CONSOLIDATE.EXPECTED_BY_CLASS == {DS.CLASS_E: 445,
                                             DS.CLASS_H: 2769}


# =========================================================== 5. population
def test_the_focused_population_is_the_whole_development_corpus():
    spec = CONFIG["protocols"][FOCUSED]
    assert spec["inventory"]["E_rows"] == 89
    assert spec["inventory"]["H_rows"] == 213
    population = spec["population_definition"]
    assert "480-row development corpus" in population["source"]
    assert "targeted_h_legacy_regressions_v1" in population["not_defined_by"]


def test_the_legacy_stress_set_neither_selects_nor_defines_this_population():
    """The 20 legacy regressions came from a different actuator entirely."""
    legacy = json.loads(
        (REPO_ROOT / "manifests"
         / "targeted_h_legacy_regressions_v1.json").read_text("utf-8"))
    assert legacy["source_actuator"]["variant"] == "consensus23_global_frozen"
    assert legacy["source_actuator"]["donors"] == ["B_d2", "B_d3"]
    assert legacy["legacy_stress_set"] is True
    assert legacy["population_reportable"] is False

    # nothing in the focused protocol's code path reads that manifest
    for name in ("run_causal_screen", "consolidate_causal_screen"):
        source = (REPO_ROOT / "experiments" / "dual_direction_controller"
                  / f"{name}.py").read_text(encoding="utf-8")
        assert "targeted_h_legacy" not in source, name
        assert "consensus23" not in source, name
    assert "targeted_h_legacy" not in (
        REPO_ROOT / "src" / "routegate" / "dualsteer.py").read_text("utf-8")


# ============================================================= 6. declared
def test_the_config_and_the_package_agree_on_the_focused_protocol():
    spec = CONFIG["protocols"][FOCUSED]
    protocol = DS.protocol_for(FOCUSED)
    assert spec["expected_generations"] == protocol.expected_generations == 871
    assert spec["tests_safety_direction"] is False
    assert protocol.tests_safety_direction is False
    for row_class in DS.SCREEN_CLASSES:
        declared = [(arm["arm_id"], arm["alpha_R"], arm["alpha_S"])
                    for arm in spec["arms"][row_class]]
        implemented = [(arm.arm_id, arm.alpha_repair, arm.alpha_safety)
                       for arm in protocol.arms_by_class[row_class]]
        assert declared == implemented, row_class
    assert spec["alpha_safety_in_every_arm"] == 0.0
    assert spec["direction_bank"]["artifact_sha256"] == (
        "d759574b5be630ea5170fb440f8663d8c4bfaa32b479ce5cf641b6dbd73516bb")
    assert spec["direction_bank"]["repair_direction"] == "B_d1 - E"
    assert spec["direction_bank"]["orthogonalisation_applied"] is False


def test_the_config_records_the_layer_and_state_convention():
    spec = CONFIG["protocols"][FOCUSED]["intervention"]
    assert spec["historical_layers"] == list(DS.HISTORICAL_LAYERS) == [31, 32, 33]
    assert spec["routegate_states"] == list(DS.BANK_STATES) == [32, 33, 34]
    assert spec["token_position"] == "final prompt token"
    assert spec["prefill_only"] is True
    assert spec["generated_token_positions_are_never_edited"] is True
    assert spec["budget_reference"] == DS.BUDGET_REFERENCE


def test_the_config_keeps_phase_two_out_of_scope():
    spec = CONFIG["protocols"][FOCUSED]
    assert "H - E" in spec["does_not_test"]
    assert "phase 2" in spec["does_not_test"]
    assert "the safety direction d_S = H - E" in spec["out_of_scope"]
    assert "the coefficient controller" in spec["out_of_scope"]
    assert "any judging of the generated responses" in spec["out_of_scope"]


def test_the_exploratory_protocol_is_still_declared():
    spec = CONFIG["protocols"][EXPLORATORY]
    assert spec["expected_generations"] == 3214
    assert spec["tests_safety_direction"] is True
    assert CONFIG["causal_screen"]["inventory"]["total"] == 3214


# ============================================================ 7. invariants
# The three fields that restate the arm table -- expected_generations,
# tests_safety_direction, and the set of classes covered -- are checked
# against the arms at construction. These tests build deliberately
# inconsistent protocols and assert the construction is refused.
def _protocol(**overrides):
    spec = dict(
        name="probe_protocol",
        arms_by_class={
            DS.CLASS_E: (DS.ARM_BASELINE,),
            DS.CLASS_H: (DS.ARM_BASELINE,),
        },
        expected_generations=DS.N_E_ROWS + DS.N_H_ROWS,
        purpose="a protocol built only to be checked",
        tests_safety_direction=False)
    spec.update(overrides)
    return DS.Protocol(**spec)


def test_a_consistent_protocol_still_constructs():
    protocol = _protocol()
    assert protocol.expected_generations == 302 == DS.N_E_ROWS + DS.N_H_ROWS


def test_a_protocol_cannot_understate_its_own_size():
    with pytest.raises(DualSteerError, match="declares 871 generation"):
        _protocol(expected_generations=871)


def test_a_protocol_cannot_claim_a_scope_its_arms_do_not_keep():
    """alpha_S > 0 anywhere and tests_safety_direction=False is a false claim."""
    safety_arm = DS.ARM_BY_ID["safety_010"]
    with pytest.raises(DualSteerError, match="alpha_S > 0"):
        _protocol(arms_by_class={DS.CLASS_E: (DS.ARM_BASELINE,),
                                 DS.CLASS_H: (DS.ARM_BASELINE, safety_arm)},
                  expected_generations=DS.N_E_ROWS + 2 * DS.N_H_ROWS,
                  tests_safety_direction=False)


def test_a_protocol_cannot_claim_a_safety_scope_it_does_not_exercise():
    with pytest.raises(DualSteerError, match="tests_safety_direction=True"):
        _protocol(tests_safety_direction=True)


def test_a_protocol_must_cover_both_screened_classes():
    with pytest.raises(DualSteerError, match="declares no arms"):
        _protocol(arms_by_class={DS.CLASS_E: (DS.ARM_BASELINE,)},
                  expected_generations=DS.N_E_ROWS)


def test_adding_a_safety_arm_to_phase_one_is_refused_at_construction():
    """The concrete drift the invariant exists to stop.

    Phase 1 is defined as alpha_S = 0 throughout. If a later edit adds the
    safety ladder to its H arms, the protocol must fail to construct rather
    than run 871 generations that quietly test the safety direction while
    every manifest records tests_safety_direction=False.
    """
    focused = DS.protocol_for(FOCUSED)
    drifted = dict(focused.arms_by_class)
    drifted[DS.CLASS_H] = drifted[DS.CLASS_H] + (DS.ARM_BY_ID["safety_015"],)
    with pytest.raises(DualSteerError):
        DS.Protocol(name=focused.name, arms_by_class=drifted,
                    expected_generations=focused.expected_generations,
                    purpose=focused.purpose, tests_safety_direction=False)


def test_every_shipped_protocol_satisfies_the_invariant():
    """Not vacuous: PROTOCOLS is built at import, so this is already proven.

    Asserted anyway so that the guarantee is stated where it is read, and so
    that removing __post_init__ fails a test rather than silently widening
    what a protocol may claim.

    The row counts come from each protocol's own declared coverage, not from
    the corpus constants: a protocol bound to a selection covers what the
    selection returned. The corpus-wide protocols are checked against the
    corpus separately below, so the binding cannot quietly shrink one of them.
    """
    for name in DS.PROTOCOLS:
        protocol = DS.protocol_for(name)
        counted = sum(protocol.expected_rows(row_class)
                      * len(protocol.arms_by_class[row_class])
                      for row_class in DS.SCREEN_CLASSES)
        assert counted == protocol.expected_generations, name
        steers = any(arm.alpha_safety > 0.0
                     for arms in protocol.arms_by_class.values()
                     for arm in arms)
        assert steers == protocol.tests_safety_direction, name


def test_the_corpus_wide_protocols_still_cover_the_whole_corpus():
    """The row-binding machinery defaults to the corpus. Phase 1 and the
    exploratory screen must not have moved when phase 2 gained a subset."""
    for name in (FOCUSED, DS.EXPLORATORY_PROTOCOL):
        protocol = DS.protocol_for(name)
        assert protocol.expected_rows(DS.CLASS_E) == DS.N_E_ROWS, name
        assert protocol.expected_rows(DS.CLASS_H) == DS.N_H_ROWS, name
        assert protocol.selected_row_ids(DS.CLASS_H) is None, name
        assert protocol.reference_records() == 0, name
    assert DS.protocol_for(FOCUSED).expected_generations == 871
    assert DS.protocol_for(DS.EXPLORATORY_PROTOCOL).expected_generations == 3214


# =========================================================== 8. regressions
# One test per defect found when the protocol refactor was first run against
# the existing CPU suite. Each names the failure it prevents coming back.
def _full_record(arm_id, row_id="h:0001", row_class=DS.CLASS_H,
                 protocol=FOCUSED, **overrides):
    """A record complete enough to reach the consolidator's arm gates."""
    arm = DS.ARM_BY_ID[arm_id]
    text = f"reply {row_id} {arm_id}"
    values = {
        "row_id": row_id, "row_class": row_class, "arm_id": arm_id,
        "alpha_R": arm.alpha_repair, "alpha_S": arm.alpha_safety,
        "direction_bank_sha256": "a" * 64,
        "generation_config_sha256": "b" * 64,
        "model": "Qwen/Qwen3-30B-A3B", "revision": "c" * 40,
        "tokenizer_config_sha256": "d" * 64, "chat_template_sha256": "e" * 64,
        "intervention_type": DS.INTERVENTION_TYPE,
        "budget_reference": DS.BUDGET_REFERENCE,
        "states": list(DS.BANK_STATES),
        "historical_layers": list(DS.HISTORICAL_LAYERS),
        "dataset_sha256": "f" * 64, "smoke": False, "protocol": protocol,
        "prompt_sha256": "1" * 64,
        "response_text": text,
        "response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "status": "ok", "created_utc": "2026-08-23T00:00:00Z",
        "generation": {"model": "Qwen/Qwen3-30B-A3B"},
    }
    values.update(overrides)
    return DS.make_record(**values)


def test_declared_arms_follows_the_specification_indirection():
    """The KeyError this repair fixes.

    ``exploratory_dual_direction_v1`` carries no ``arms`` block of its own --
    it names the block that has held them since the screen was written. Code
    that reached straight for ``spec["arms"]`` raised KeyError on every
    consolidation, because the exploratory protocol is the default.
    """
    exploratory = DS.protocol_for(EXPLORATORY)
    spec = CONFIG["protocols"][EXPLORATORY]
    assert "arms" not in spec
    assert spec["specification"] == "causal_screen"
    assert (DS.declared_arms(CONFIG, exploratory)
            == DS.implemented_arms(exploratory))


def test_declared_arms_reads_a_protocol_that_carries_its_own_table():
    focused = DS.protocol_for(FOCUSED)
    assert "arms" in CONFIG["protocols"][FOCUSED]
    assert DS.declared_arms(CONFIG, focused) == DS.implemented_arms(focused)


def test_a_configuration_written_before_protocols_still_reads():
    """causal_screen.arms alone is the exploratory table by definition."""
    historical = {key: value for key, value in CONFIG.items()
                  if key != "protocols"}
    exploratory = DS.protocol_for(EXPLORATORY)
    assert (DS.declared_arms(historical, exploratory)
            == DS.implemented_arms(exploratory))
    # the fallback belongs to the default protocol and to nothing else
    with pytest.raises(DualSteerError, match="declares no protocol"):
        DS.declared_arms(historical, DS.protocol_for(FOCUSED))


def test_an_undeclared_protocol_is_not_inferred():
    config = dict(CONFIG, protocols={
        EXPLORATORY: CONFIG["protocols"][EXPLORATORY]})
    with pytest.raises(DualSteerError, match="declares no protocol"):
        DS.declared_arms(config, DS.protocol_for(FOCUSED))


def test_contradictory_arm_tables_are_refused_not_reconciled():
    """Own table and a reused one, disagreeing: neither silently wins."""
    spec = dict(CONFIG["protocols"][FOCUSED], specification="causal_screen")
    config = dict(CONFIG,
                  protocols=dict(CONFIG["protocols"], **{FOCUSED: spec}))
    with pytest.raises(DualSteerError, match="disagree"):
        DS.declared_arms(config, DS.protocol_for(FOCUSED))


def test_a_protocol_with_no_arm_table_at_all_is_refused():
    spec = {"expected_generations": 871}
    config = dict(CONFIG,
                  protocols=dict(CONFIG["protocols"], **{FOCUSED: spec}))
    with pytest.raises(DualSteerError, match="declares neither"):
        DS.declared_arms(config, DS.protocol_for(FOCUSED))


def test_the_refusal_names_both_the_screen_and_the_protocol():
    """The established wording is preserved; the protocol's count is added.

    The exploratory suite matches on "this screen is" and has since the
    screen was written. Rewording it to "protocol X is" broke three cases
    that were testing fail-closed behaviour, not prose.
    """
    for name in (EXPLORATORY, FOCUSED):
        protocol = DS.protocol_for(name)
        items = DS.build_inventory(_rows(88, DS.N_H_ROWS), protocol)
        with pytest.raises(DualSteerError) as caught:
            DS.assert_inventory(items, protocol=protocol)
        message = str(caught.value)
        assert "this screen is" in message, name
        assert f"protocol {name} is {protocol.expected_generations}" in message
        assert "is refused rather than run short" in message, name


def test_consolidation_checks_coefficients_against_the_selected_arm_table():
    """A safety arm is a record under one protocol and not under the other."""
    accepted = _full_record("safety_010", protocol=EXPLORATORY)
    locked = {field: accepted[field] for field in DS.RUN_LOCKED_FIELDS}
    result = CONSOLIDATE.verify_records(
        [accepted], locked, Checklist("regression"),
        DS.protocol_for(EXPLORATORY))
    assert result["unique"] == 1

    refused = _full_record("safety_010", protocol=FOCUSED)
    locked = {field: refused[field] for field in DS.RUN_LOCKED_FIELDS}
    with pytest.raises(DualSteerError, match="no_unexpected_arms"):
        CONSOLIDATE.verify_records([refused], locked, Checklist("regression"),
                                   DS.protocol_for(FOCUSED))


def test_a_focused_arm_still_passes_its_own_coefficient_gate():
    """The counterpart: repair_015 is phase 1's H arm and is accepted."""
    record = _full_record("repair_015", protocol=FOCUSED)
    locked = {field: record[field] for field in DS.RUN_LOCKED_FIELDS}
    result = CONSOLIDATE.verify_records(
        [record], locked, Checklist("regression"), DS.protocol_for(FOCUSED))
    assert result["unique"] == 1


def test_the_out_of_scope_wording_carries_no_forbidden_token():
    """It said "sealed", which the configuration walker refuses outright."""
    spec = CONFIG["protocols"][FOCUSED]
    assert ("any evaluation material outside the development corpus"
            in spec["out_of_scope"])
    for line in spec["out_of_scope"]:
        assert looks_sealed(line) is None, line
    assert_config_is_sealed_free(CONFIG, "dual direction controller config")


def test_the_protocol_arm_index_is_the_protocol_s_own():
    focused = DS.protocol_for(FOCUSED)
    exploratory = DS.protocol_for(EXPLORATORY)
    assert set(focused.arm_by_id()) == {
        "baseline", "repair_005", "repair_010", "repair_015", "repair_020"}
    assert set(exploratory.arm_by_id()) == set(DS.ARM_BY_ID)
    assert "safety_010" in exploratory.arm_by_id()
    assert "safety_010" not in focused.arm_by_id()
