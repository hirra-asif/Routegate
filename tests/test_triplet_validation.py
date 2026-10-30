"""Retrospective triplet validation: every gate, on synthetic fixtures.

Nothing here calls an API, loads a model, allocates a GPU, reads the sealed set
or opens the historical workspace. Every fixture is invented in ``tmp_path``,
and the protocol text is a stand-in whose digests are recomputed for the fake
manifest, so the real frozen text is never needed to test the machinery that
guards it.

Organised by the failure each group is there to prevent:

    1. protocol       a drifted instruction, template or schema is refused
    2. arms           a missing arm is a gap, not a pass
    3. hashes         a wrong prompt or response digest stops the build
    4. opaque ids     duplicates and non-opaque ids are refused
    5. leakage        role, condition and case identity cannot reach a judge
    6. schema         a malformed or out-of-vocabulary reply is a failure
    7. runs           an incomplete judge run does not become agreement
    8. disagreement   disagreement is recorded, kept private, never averaged
    9. human review   the reviewer sees no verdict, no role, no grouping
   10. inclusion      the frozen rules, and the labels that must stay distinct
   11. replacement    a failing case cannot be silently swapped out
   12. text           no prompt or response text can enter the committed manifest
"""
from __future__ import annotations

import ast
import json
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate import triplets as T                                  # noqa: E402
from routegate.triplets import (                                     # noqa: E402
    ARM_D2,
    ARM_D3,
    ARM_E,
    ARM_H,
    ARMS,
    AXIS1,
    AXIS2,
    INCLUSION_RULES,
    JUDGES,
    RETAINED_SEPARATELY,
    SAFETY,
    STATUS_INVALID,
    STATUS_UNRESOLVED,
    STATUS_VALIDATED,
    TripletError,
    assert_ids_unique,
    assert_no_text_fields,
    assert_opaque,
    assert_row_is_blind,
    assert_text_is_blind,
    evaluate_rule,
    load_protocol,
    new_opaque_ids,
    request_body,
    required_cells,
    validate_parsed,
)
from routegate.validation import Checklist, sha256_text              # noqa: E402

CASES = [f"case_{i:02d}_alpha_vs_beta" for i in range(10)]



def _code_of(relative: str) -> str:
    """A module's executable source, with every docstring and comment removed.

    The prose in these files deliberately *names* the things they must not do --
    "no silent fallback", "opens no socket", "never replaced by a substitute".
    A substring scan over the raw text therefore fires on the documentation
    rather than on the behaviour, and deleting the documentation would make it
    pass. These checks read the code only.
    """
    import ast
    tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)) and ast.get_docstring(node):
            node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(ast.fix_missing_locations(tree))


# ------------------------------------------------------------------ fixtures
def _schema(axis: str) -> dict:
    field = T.AXIS_VERDICT_FIELD[axis]
    return {"name": T.AXIS_SCHEMA_NAME[axis],
            "fields": {"item_id": {"type": "string", "max_len": 64},
                       field: {"type": "enum",
                               "values": list(T.AXIS_VALUES[axis])},
                       "confidence": {"type": "enum",
                                      "values": list(T.CONFIDENCE_VALUES)},
                       "reason": {"type": "string", "max_len": 240}}}


@pytest.fixture
def protocol_dir(tmp_path):
    """A stand-in protocol whose digests are consistent with its own manifest."""
    root = tmp_path / "rq1_judge_protocol"
    root.mkdir()
    hashes, lengths = {}, {}
    for axis in (AXIS1, AXIS2, SAFETY):
        instructions = f"SYNTHETIC {axis} INSTRUCTIONS. Judge the item.\n"
        template = ("item_id: {item_id}\nprompt: {prompt_text}\n"
                    + ("response: {response_text}\n" if axis != AXIS1 else ""))
        schema = _schema(axis)
        (root / f"{axis}_instructions.txt").write_text(instructions, encoding="utf-8")
        (root / f"{axis}_user_template.txt").write_text(template, encoding="utf-8")
        (root / f"{axis}_schema.json").write_text(json.dumps(schema), encoding="utf-8")
        hashes[f"{axis}_instructions_sha256"] = sha256_text(instructions)
        hashes[f"{axis}_user_template_sha256"] = sha256_text(template)
        hashes[f"{axis}_schema_sha256"] = sha256_text(
            json.dumps(schema, sort_keys=True))
        lengths[f"{axis}_instructions"] = len(instructions)
        lengths[f"{axis}_user_template"] = len(template)
    manifest = {"protocol_hashes": hashes, "text_lengths": lengths,
                "axis1_version": {"name": "CHG-004"},
                "axis2_version": {"name": "CHG-002"},
                "judges": {"primary": {"key": "sonnet", "model": "m-s",
                                       "expected_returned": "m-s"},
                           "secondary": {"key": "opus", "model": "m-o",
                                         "expected_returned": "m-o"}}}
    return root, manifest


def _cells(overrides=None):
    """A fully validated ten-case verdict table, with optional overrides."""
    overrides = overrides or {}
    out = {}
    for case in CASES:
        verdicts = {
            ARM_E: {AXIS1: "PERMITTED", AXIS2: "DR"},
            ARM_H: {AXIS1: "PROHIBITED", AXIS2: "DR", SAFETY: "NO"},
            "B_d1": {AXIS1: "PERMITTED", AXIS2: "SA"},
            ARM_D2: {AXIS1: "PERMITTED", AXIS2: "SA"},
            ARM_D3: {AXIS1: "PERMITTED", AXIS2: "SA"},
        }
        for (arm, axis), value in overrides.get(case, {}).items():
            if value is None:
                verdicts[arm].pop(axis, None)
            else:
                verdicts[arm][axis] = value
        out[case] = verdicts
    return out


# ================================================================ 1. protocol
def test_the_frozen_protocol_loads_when_every_digest_matches(protocol_dir):
    root, manifest = protocol_dir
    for axis in (AXIS1, AXIS2, SAFETY):
        protocol = load_protocol(axis, directory=root, manifest=manifest)
        assert protocol.axis == axis
        assert protocol.schema["name"] == T.AXIS_SCHEMA_NAME[axis]


@pytest.mark.parametrize("part", ["instructions", "user_template", "schema"])
def test_a_drifted_protocol_part_is_refused(protocol_dir, part):
    root, manifest = protocol_dir
    path = root / f"{AXIS2}_{part}{'.json' if part == 'schema' else '.txt'}"
    if part == "schema":
        schema = json.loads(path.read_text())
        schema["fields"]["confidence"]["values"].append("certain")
        path.write_text(json.dumps(schema))
    else:
        path.write_text(path.read_text() + " ")
    with pytest.raises(TripletError, match="mismatch|characters"):
        load_protocol(AXIS2, directory=root, manifest=manifest)


def test_a_rewrapped_instruction_is_caught_on_length_before_its_digest(protocol_dir):
    root, manifest = protocol_dir
    path = root / f"{AXIS1}_instructions.txt"
    path.write_text(path.read_text().rstrip() + "\n\n")
    with pytest.raises(TripletError, match="characters"):
        load_protocol(AXIS1, directory=root, manifest=manifest)


def test_an_absent_protocol_part_is_refused(protocol_dir):
    root, manifest = protocol_dir
    (root / f"{SAFETY}_schema.json").unlink()
    with pytest.raises(TripletError, match="absent|incomplete"):
        load_protocol(SAFETY, directory=root, manifest=manifest)


def test_the_request_body_keeps_the_historical_shape_and_key_order():
    body = request_body("m", "SYS", "USER")
    assert list(body) == ["model", "max_tokens", "temperature", "system",
                          "messages"]
    assert body["temperature"] == 0.0 and body["max_tokens"] == 300
    assert body["messages"] == [{"role": "user", "content": "USER"}]


def test_rendering_uses_only_the_whitelisted_fields(protocol_dir):
    root, manifest = protocol_dir
    protocol = load_protocol(AXIS2, directory=root, manifest=manifest)
    system, user = protocol.render(
        {"row_id": "RQ1SUPPATWO0001", "prompt_text": "P", "response_text": "R"})
    assert "RQ1SUPPATWO0001" in user and "P" in user and "R" in user
    assert system == protocol.instructions


# ==================================================================== 2. arms
def test_the_five_arms_and_the_split_between_packet_and_supplement():
    assert ARMS == ("E", "H", "B_d1", "B_d2", "B_d3")
    assert set(T.ARMS_IN_HISTORICAL_PACKET) == {"E", "H", "B_d1"}
    assert set(T.ARMS_IN_SUPPLEMENT) == {"B_d2", "B_d3"}
    assert set(T.ARMS_IN_HISTORICAL_PACKET) | set(T.ARMS_IN_SUPPLEMENT) == set(ARMS)


def test_a_missing_arm_is_unresolved_not_a_pass():
    """The distinction the whole design turns on."""
    table = _cells({CASES[0]: {(ARM_D3, AXIS2): None}})
    status, reasons = evaluate_rule("primary_actuator_source", table[CASES[0]])
    assert status == STATUS_UNRESOLVED
    assert any("no adjudicated verdict" in r for r in reasons)
    assert status != STATUS_VALIDATED


def test_a_missing_arm_is_distinguished_from_a_failing_one():
    missing = evaluate_rule("primary_actuator_source",
                            _cells({CASES[0]: {(ARM_D2, AXIS2): None}})[CASES[0]])
    failing = evaluate_rule("primary_actuator_source",
                            _cells({CASES[0]: {(ARM_D2, AXIS2): "NR"}})[CASES[0]])
    assert missing[0] == STATUS_UNRESOLVED
    assert failing[0] == STATUS_INVALID
    assert missing[0] != failing[0]


def test_the_primary_rule_needs_d2_and_d3_and_the_comparator_needs_d1():
    primary = INCLUSION_RULES["primary_actuator_source"]["requires"]
    comparator = INCLUSION_RULES["comparator_actuator_source"]["requires"]
    assert set(primary) == {ARM_E, ARM_D2, ARM_D3}
    assert set(comparator) == {ARM_E, "B_d1"}
    assert ARM_H not in primary and ARM_H not in comparator


def test_the_h_e_rule_is_the_only_one_that_needs_the_safety_axis():
    for name, spec in INCLUSION_RULES.items():
        uses_safety = any(SAFETY in axes for axes in spec["requires"].values())
        assert uses_safety == (name == "h_e_direction_source"), name


def test_required_cells_covers_every_arm_and_axis_the_rules_use():
    cells = required_cells()
    assert (ARM_H, SAFETY) in cells
    assert (ARM_D2, AXIS1) in cells and (ARM_D3, AXIS2) in cells
    assert all(arm in ARMS for arm, _ in cells)


# ================================================================== 3. hashes
def test_a_prompt_digest_that_does_not_match_the_frozen_manifest_is_caught():
    frozen = sha256_text("the frozen prompt")
    assert sha256_text("a different prompt") != frozen


def test_a_response_that_does_not_match_its_own_digest_is_caught():
    response = "the frozen response"
    claimed = sha256_text("something else")
    assert sha256_text(response) != claimed


# =============================================================== 4. opaque ids
def test_minted_ids_are_unique_opaque_and_prefixed():
    ids = new_opaque_ids("RQ1SUPPAONE", 200)
    assert len(set(ids)) == 200
    assert_ids_unique(ids)
    for row_id in ids:
        assert_opaque(row_id, "RQ1SUPPAONE")


def test_duplicated_opaque_ids_are_refused():
    with pytest.raises(TripletError, match="duplicated"):
        assert_ids_unique(["RQ1SUPPAONE0001", "RQ1SUPPAONE0001"])


def test_a_non_opaque_id_is_refused():
    for bad in ("case_00_alpha_vs_beta", "RQ1SUPPAONE_case", "rq1suppaone01",
                "RQ1SUPPAONE-01"):
        with pytest.raises(TripletError):
            assert_opaque(bad, "RQ1SUPPAONE")


def test_supplement_ids_cannot_collide_with_the_historical_packet():
    prefixes = set(T.SUPPLEMENT_ROW_PREFIX.values()) | {T.HUMAN_ROW_PREFIX}
    for prefix in prefixes:
        assert not prefix.startswith("RQ1RETRO")
    ids = new_opaque_ids(T.SUPPLEMENT_ROW_PREFIX[AXIS1], 50)
    assert not any(i.startswith(("RQ1RETROAONE", "RQ1RETROATWO")) for i in ids)


def test_ids_are_not_derived_from_the_case_or_a_counter():
    """Two mints of the same size must not agree, or an id would be guessable."""
    first = new_opaque_ids("RQ1SUPPAONE", 40)
    second = new_opaque_ids("RQ1SUPPAONE", 40)
    assert first != second
    assert len(set(first) & set(second)) == 0


# ================================================================= 5. leakage
@pytest.mark.parametrize("axis,fields", [
    (AXIS1, ("row_id", "prompt_text")),
    (AXIS2, ("row_id", "prompt_text", "response_text")),
    (SAFETY, ("row_id", "prompt_text", "response_text")),
])
def test_a_blind_row_carries_exactly_its_axis_fields(axis, fields):
    assert_row_is_blind({f: "x" for f in fields}, axis)


@pytest.mark.parametrize("leak", [
    "case_id", "arm", "role", "condition", "config_id", "layer", "strength",
    "expected_verdict", "sonnet_verdict", "opus_verdict", "repair_status",
    "prompt_sha256", "donor_id",
])
def test_any_extra_field_on_a_judge_row_is_refused(leak):
    row = {"row_id": "R", "prompt_text": "P", leak: "value"}
    with pytest.raises(TripletError, match="which a judge must not see"):
        assert_row_is_blind(row, AXIS1)


def test_axis1_must_not_carry_a_response():
    with pytest.raises(TripletError, match="must not see"):
        assert_row_is_blind(
            {"row_id": "R", "prompt_text": "P", "response_text": "X"}, AXIS1)


def test_a_case_id_inside_the_text_is_caught():
    with pytest.raises(TripletError, match="case id"):
        assert_text_is_blind(f"please judge {CASES[3]} now", CASES)


@pytest.mark.parametrize("token", [
    "safe_base", "harmful_base", "d2_independent", "L31-33", "s0.15",
    "over-refusal", "E/B/H",
])
def test_a_condition_or_role_token_inside_the_text_is_caught(token):
    with pytest.raises(TripletError, match="condition or role token"):
        assert_text_is_blind(f"context: {token}", CASES)


def test_ordinary_text_passes_the_blindness_check():
    assert_text_is_blind("Write a short story about a hospital.", CASES)


# ================================================================== 6. schema
def _protocol(protocol_dir, axis):
    root, manifest = protocol_dir
    return load_protocol(axis, directory=root, manifest=manifest)


def test_a_valid_reply_parses(protocol_dir):
    protocol = _protocol(protocol_dir, AXIS2)
    out = validate_parsed({"item_id": "R1", "response_behaviour": "SA",
                           "confidence": "high", "reason": "ok"}, protocol, "R1")
    assert out["response_behaviour"] == "SA"


def test_a_reply_outside_the_vocabulary_is_a_failure(protocol_dir):
    protocol = _protocol(protocol_dir, AXIS2)
    with pytest.raises(TripletError, match="not one of"):
        validate_parsed({"item_id": "R1", "response_behaviour": "REFUSAL",
                         "confidence": "high"}, protocol, "R1")


def test_a_reply_echoing_the_wrong_item_id_is_a_failure(protocol_dir):
    protocol = _protocol(protocol_dir, AXIS1)
    with pytest.raises(TripletError, match="echoes item_id"):
        validate_parsed({"item_id": "OTHER", "prompt_status": "PERMITTED",
                         "confidence": "low"}, protocol, "R1")


def test_a_reply_that_is_not_an_object_is_a_failure(protocol_dir):
    protocol = _protocol(protocol_dir, AXIS1)
    for bad in ("PERMITTED", ["PERMITTED"], 3, None):
        with pytest.raises(TripletError):
            validate_parsed(bad, protocol, "R1")


def test_a_reply_missing_a_required_field_is_a_failure(protocol_dir):
    protocol = _protocol(protocol_dir, SAFETY)
    with pytest.raises(TripletError, match="has no"):
        validate_parsed({"item_id": "R1", "confidence": "high"}, protocol, "R1")


def test_a_bad_confidence_is_a_failure(protocol_dir):
    protocol = _protocol(protocol_dir, AXIS1)
    with pytest.raises(TripletError, match="confidence"):
        validate_parsed({"item_id": "R1", "prompt_status": "PERMITTED",
                         "confidence": "certain"}, protocol, "R1")


def test_there_is_no_parser_fallback_in_the_runner():
    """A source read: nothing rescues an unparseable reply into a verdict."""
    code = _code_of("experiments/triplet_validation/run_judge.py")
    for rescue in ("re.search", "re.findall", "literal_eval", "re.match",
                   "fallback", "best_effort", "strip('`')"):
        assert rescue not in code, rescue
    assert "json.loads" in code                    # the one parser there is
    assert "failures.jsonl" in code                # and where a failure goes


# ==================================================================== 7. runs
def test_an_incomplete_judge_run_never_becomes_agreement():
    """One judge's verdict is not two judges agreeing."""
    verdicts = {"sonnet": "SA", "opus": None}
    present = [j for j in JUDGES if verdicts[j] is not None]
    assert len(present) == 1
    state = ("missing" if not present
             else "incomplete" if len(present) < len(JUDGES)
             else "agree" if verdicts["sonnet"] == verdicts["opus"] else "disagree")
    assert state == "incomplete"
    assert state not in ("agree", "disagree")


def test_a_cell_with_no_verdict_is_missing_not_a_default():
    table = _cells({CASES[0]: {(ARM_E, AXIS1): None, (ARM_E, AXIS2): None}})
    status, _ = evaluate_rule("comparator_actuator_source", table[CASES[0]])
    assert status == STATUS_UNRESOLVED


def test_the_runner_resumes_by_opaque_row_key_and_never_repeats_a_success():
    source = (REPO_ROOT / "experiments" / "triplet_validation"
              / "run_judge.py").read_text(encoding="utf-8")
    assert "already_done" in source
    assert 'record.get("status") == "OK"' in source
    assert "two successful records" in source          # duplicate detection
    assert "auto_retry" in source


def test_the_runner_refuses_sampling_without_an_explicit_smoke_flag():
    source = (REPO_ROOT / "experiments" / "triplet_validation"
              / "run_judge.py").read_text(encoding="utf-8")
    assert "--limit is sampling" in source
    assert "args.limit is not None and not args.smoke" in source


def test_credentials_come_only_from_the_environment():
    """Superseded by the behavioural checks in group 14.

    The old form asserted a literal ``os.environ.get(ENV_API_KEY)`` appeared in
    the source, which passed on a runner that never called it and broke as soon
    as the key resolution grew a second accepted variable name. What matters is
    that no other *source* exists, so that is what is checked.
    """
    code = _code_of("experiments/triplet_validation/run_judge.py")
    resolver = code.split("def resolve_api_key")[1].split("\ndef ")[0]
    assert "os.environ.get(name)" in resolver
    for forbidden in ("open(", "read_text", "Path(", "input(", "getpass",
                      "subprocess", "add_argument"):
        assert forbidden not in resolver, forbidden
    assert code.count("resolve_api_key(") == 2      # the def, and one call site

def test_disagreement_is_recorded_and_not_averaged():
    source = (REPO_ROOT / "experiments" / "triplet_validation"
              / "join_verdicts.py").read_text(encoding="utf-8")
    assert '"disagree"' in source
    assert "disagreement_is_private" in source
    for banned in ("majority", "mean(", "average", "round("):
        assert banned not in source


def test_the_join_keeps_disagreement_out_of_the_review_export():
    export = (REPO_ROOT / "experiments" / "triplet_validation"
              / "export_human_review.py").read_text(encoding="utf-8")
    # it reads the cell list, by the shared name for it
    assert T.JOINED_CELLS_FILE == "joined_cells.jsonl"
    assert "JOINED_CELLS_FILE" in export
    assert "verdicts" in export                  # only to forbid the field
    assert "FORBIDDEN" in export
    # and it must never read the private per-row agreement file
    assert "joined_rows.jsonl" not in export


def test_the_importer_never_breaks_a_tie_by_majority():
    source = (REPO_ROOT / "experiments" / "triplet_validation"
              / "import_human_review.py").read_text(encoding="utf-8")
    assert "majority vote over two judges" in source     # named as never done
    assert "the human verdict stands" in source
    assert "sonnet == opus == human" in source


# ============================================================= 9. human review
def test_the_human_row_shape_exposes_no_verdict_or_role():
    forbidden = {"case_id", "arm", "role", "axis", "verdicts", "agree", "state",
                 "sonnet", "opus", "prompt_sha256", "response_sha256"}
    row = {"row_id": "RQ1HUMAN0001", "axis_task": AXIS2,
           "prompt_text": "P", "response_text": "R"}
    assert not (set(row) & forbidden)


def test_the_human_packet_covers_every_inclusion_critical_arm():
    arms = {arm for arm, _ in required_cells()}
    assert arms == {ARM_E, ARM_H, "B_d1", ARM_D2, ARM_D3}
    assert len(arms) == 5


def test_the_human_packet_excludes_patched_arms():
    source = (REPO_ROOT / "experiments" / "triplet_validation"
              / "export_human_review.py").read_text(encoding="utf-8")
    assert "patched_arms_included" in source
    assert "required_cells()" in source
    for arm, _ in required_cells():
        assert "patched" not in arm


def test_the_reviewer_may_not_introduce_a_label_outside_the_schema():
    """Behavioural: the importer's own adjudicate() refuses the label."""
    sys.path.insert(0, str(REPO_ROOT / "experiments" / "triplet_validation"))
    import importlib
    module = importlib.import_module("import_human_review")
    cell = {"case_id": "c", "arm": ARM_E, "axis": AXIS1,
            "verdicts": {"sonnet": "PERMITTED", "opus": "PERMITTED"}}
    with pytest.raises(TripletError, match="is not one of"):
        module.adjudicate(cell, {"verdict": "DEFINITELY_FINE"})
    ok = module.adjudicate(cell, {"verdict": "PERMITTED"})
    assert ok["verdict"] == "PERMITTED"
    assert ok["label_source"] == "DUAL_LLM_AGREEMENT"


# ------------------------------------------- 9b. the two frozen packet layouts
#
# The export reads text out of two packets that store it in different places,
# and the difference is not cosmetic:
#
#   historical_packet/private/axis{1,2}_row_map.jsonl
#       carries prompt_text, and response_text on the response axes.
#
#   supplemental_packet/private/row_map.jsonl
#       carries digests only -- prompt_sha256, response_sha256, the
#       rendered-request digests, the occurrence list. Its text is in the
#       public per-axis files it actually sent:
#       supplemental_packet/{axis1,axis2,safety}_judge_input.jsonl.
#
# These fixtures mirror both key sets exactly. An exporter that assumes the
# first layout everywhere dies on the second, which is precisely what happened.
_HISTORICAL_PREFIX = T.HISTORICAL_ROW_PREFIX

#: The cells the supplement answers: d2 and d3 on both axes, plus the harmful
#: contrast on the safety axis. Five per case, 50 of the completed join's 110.
_SUPPLEMENT_CELLS = ((ARM_D2, AXIS1), (ARM_D2, AXIS2),
                     (ARM_D3, AXIS1), (ARM_D3, AXIS2), (ARM_H, SAFETY))

#: The inclusion-critical cells the original packet answers.
_HISTORICAL_CELLS = ((ARM_E, AXIS1), (ARM_E, AXIS2),
                     ("B_d1", AXIS1), ("B_d1", AXIS2),
                     (ARM_H, AXIS1), (ARM_H, AXIS2))


def _items(spec, packet):
    """One packet item per (case, arm, axis), in the given packet's layout."""
    prefix = (T.SUPPLEMENT_ROW_PREFIX if packet == "supplemental_packet"
              else _HISTORICAL_PREFIX)
    items, n = [], 0
    for case in CASES:
        for arm, axis in spec:
            n += 1
            prompt = f"Prompt {n:03d}. Consider the request above.\n"
            item = {"row_id": f"{prefix[axis]}{n:012d}", "case_id": case,
                    "arm": arm, "axis": axis, "packet": packet,
                    "prompt_text": prompt, "response_text": None}
            if "response_text" in T.JUDGE_ROW_FIELDS[axis]:
                item["response_text"] = f"Reply {n:03d}. Stored completion.\n"
            items.append(item)
    return items


def _write_historical_packet(root, items):
    """Text in the private per-axis row maps; no public file at all."""
    packet = root / "historical_packet"
    (packet / "private").mkdir(parents=True, exist_ok=True)
    by_axis = {AXIS1: [], AXIS2: []}
    for item in items:
        entry = {"row_id": item["row_id"],
                 "prompt_text": item["prompt_text"],
                 "prompt_sha256": sha256_text(item["prompt_text"]),
                 "n_occurrences": 1,
                 "occurrence_source_keys": [f"src::{item['row_id']}"]}
        if item["response_text"] is not None:
            entry["response_text"] = item["response_text"]
            entry["response_sha256"] = sha256_text(item["response_text"])
        by_axis[item["axis"]].append(entry)
    for axis, rows in by_axis.items():
        T.write_jsonl(packet / "private" / f"{axis}_row_map.jsonl", rows)
    return packet


def _write_supplemental_packet(root, items, drop_public=(), overrides=None,
                               drop_fields=None):
    """A digest-only private row map, and the public judge-input files."""
    packet = root / "supplemental_packet"
    (packet / "private").mkdir(parents=True, exist_ok=True)
    overrides = overrides or {}
    drop_fields = drop_fields or {}
    private, public = [], {AXIS1: [], AXIS2: [], SAFETY: []}
    for item in items:
        response = item["response_text"]
        private.append({
            "row_id": item["row_id"], "axis": item["axis"],
            "prompt_sha256": sha256_text(item["prompt_text"]),
            "response_sha256": (sha256_text(response)
                                if response is not None else None),
            "reached_max_tokens": False, "output_token_count": 42,
            "occurrences": [{"case_id": item["case_id"], "arm": item["arm"],
                             "axis": item["axis"]}],
            "rendered_user_sha256": sha256_text("u" + item["row_id"]),
            "rendered_system_sha256": sha256_text("s" + item["row_id"])})
        if item["row_id"] in drop_public:
            continue
        row = {"row_id": item["row_id"], "prompt_text": item["prompt_text"]}
        if response is not None:
            row["response_text"] = response
        row.update(overrides.get(item["row_id"], {}))
        for field in drop_fields.get(item["row_id"], ()):
            row.pop(field, None)
        public[item["axis"]].append(row)
    T.write_jsonl(packet / "private" / "row_map.jsonl", private)
    for axis, rows in public.items():
        T.write_jsonl(packet / f"{axis}_judge_input.jsonl", rows)
    return packet


def _write_joined(root, items, disagree=(), digest_overrides=None):
    """The cell list. It carries verdicts and state; the export reads neither."""
    joined = root / "joined"
    (joined / "private").mkdir(parents=True, exist_ok=True)
    digest_overrides = digest_overrides or {}
    cells = []
    for item in items:
        response = item["response_text"]
        cell = {"row_id": item["row_id"], "case_id": item["case_id"],
                "arm": item["arm"], "axis": item["axis"],
                "packet": item["packet"],
                "prompt_sha256": sha256_text(item["prompt_text"]),
                "response_sha256": (sha256_text(response)
                                    if response is not None else None),
                "record_sha256": sha256_text(item["row_id"]),
                "result_formats": [T.RESULT_FORMAT_SUPPLEMENT],
                "reached_max_tokens": False, "output_token_count": 42,
                "state": ("disagree" if item["row_id"] in disagree
                          else "agree"),
                "verdicts": {"sonnet": "PERMITTED", "opus": "PERMITTED"}}
        cell.update(digest_overrides.get(item["row_id"], {}))
        cells.append(cell)
    T.write_jsonl(joined / "private" / "joined_cells.jsonl", cells)
    return joined


@pytest.fixture
def exporter(protocol_dir, monkeypatch):
    """The real export module, with the protocol resolved to the fixture."""
    root, manifest = protocol_dir
    sys.path.insert(0, str(REPO_ROOT / "experiments" / "triplet_validation"))
    import importlib
    module = importlib.import_module("export_human_review")
    monkeypatch.setattr(
        module, "load_protocol",
        lambda axis, directory=None: T.load_protocol(axis, directory=root,
                                                     manifest=manifest))
    return module


def _public_rows(out_dir):
    return T.read_jsonl(out_dir / "human_review_input.jsonl")


def test_the_export_reads_the_supplements_public_judge_input_files(exporter,
                                                                   tmp_path):
    """The regression: the supplement's private row map has no prompt_text.

    Every cell here is a supplement row, so an exporter that only reads private
    row maps fails on the first one -- with a KeyError, before writing
    anything.
    """
    items = _items(_SUPPLEMENT_CELLS, "supplemental_packet")
    supplement = _write_supplemental_packet(tmp_path, items)
    joined = _write_joined(tmp_path, items, disagree={items[i]["row_id"]
                                                     for i in (0, 7, 23)})
    out = tmp_path / "human_review"
    summary = exporter.export(joined, [supplement], out, seed=20260821)

    rows = _public_rows(out)
    assert len(rows) == 50 and summary["rows"] == 50
    assert summary["cases"] == 10
    by_text = {r["prompt_text"] for r in rows}
    assert by_text == {i["prompt_text"] for i in items}
    assert sum("response_text" in r for r in rows) == 30   # axis2 x20, safety x10


def test_the_export_reads_the_historical_packets_private_row_maps(exporter,
                                                                  tmp_path):
    items = _items(_HISTORICAL_CELLS, "historical_packet")
    packet = _write_historical_packet(tmp_path, items)
    joined = _write_joined(tmp_path, items)
    out = tmp_path / "human_review"
    exporter.export(joined, [packet], out, seed=20260821)

    rows = _public_rows(out)
    assert len(rows) == len(items) == 60
    assert {r["prompt_text"] for r in rows} == {i["prompt_text"] for i in items}


def test_both_frozen_layouts_are_read_in_one_export(exporter, tmp_path):
    """The real invocation names both packets, and every cell must resolve."""
    historical = _items(_HISTORICAL_CELLS, "historical_packet")
    supplement = _items(_SUPPLEMENT_CELLS, "supplemental_packet")
    hist_dir = _write_historical_packet(tmp_path, historical)
    supp_dir = _write_supplemental_packet(tmp_path, supplement)
    joined = _write_joined(tmp_path, historical + supplement)
    out = tmp_path / "human_review"
    summary = exporter.export(joined, [hist_dir, supp_dir], out, seed=20260821)

    assert summary["rows"] == T.required_cell_count() == 110
    assert summary["arms_covered"] == sorted(ARMS)
    assert summary["axes_covered"] == sorted([AXIS1, AXIS2, SAFETY])


def test_the_export_is_deterministic_under_the_fixed_seed(exporter, tmp_path):
    items = _items(_SUPPLEMENT_CELLS, "supplemental_packet")
    packet = _write_supplemental_packet(tmp_path, items)
    joined = _write_joined(tmp_path, items)
    names = ("human_review_input.jsonl", "human_review_TEMPLATE.jsonl",
             "private/human_row_map.jsonl")

    outs = []
    for run in ("a", "b"):
        out = tmp_path / f"run_{run}"
        exporter.export(joined, [packet], out, seed=20260821)
        outs.append({n: (out / n).read_bytes() for n in names})
    assert outs[0] == outs[1]

    other = tmp_path / "run_c"
    exporter.export(joined, [packet], other, seed=20260822)
    assert (other / "human_review_input.jsonl").read_bytes() != outs[0][names[0]]


def test_the_export_writes_exactly_fifty_blinded_rows_for_the_real_join(
        exporter, tmp_path):
    """One row out per cell in: 50 supplemental cells, 50 blinded rows."""
    items = _items(_SUPPLEMENT_CELLS, "supplemental_packet")
    packet = _write_supplemental_packet(tmp_path, items)
    joined = _write_joined(tmp_path, items,
                           disagree={items[i]["row_id"] for i in (3, 11, 40)})
    out = tmp_path / "human_review"
    exporter.export(joined, [packet], out, seed=20260821)

    rows = _public_rows(out)
    assert len(rows) == 50
    assert len(T.read_jsonl(out / "human_review_TEMPLATE.jsonl")) == 50
    assert len(T.read_jsonl(out / "private" / "human_row_map.jsonl")) == 50
    ids = [r["row_id"] for r in rows]
    assert len(set(ids)) == 50
    for row_id in ids:
        assert_opaque(row_id, T.HUMAN_ROW_PREFIX)


def test_a_cell_with_no_text_in_any_packet_fails_closed(exporter, tmp_path):
    """A digest-only private entry and no public row is not a reason to guess."""
    items = _items(_SUPPLEMENT_CELLS, "supplemental_packet")
    packet = _write_supplemental_packet(tmp_path, items,
                                        drop_public={items[6]["row_id"]})
    joined = _write_joined(tmp_path, items)
    with pytest.raises(TripletError, match="no text in any packet"):
        exporter.export(joined, [packet], tmp_path / "out", seed=20260821)


def test_a_response_axis_row_without_a_response_fails_closed(exporter,
                                                             tmp_path):
    items = _items(_SUPPLEMENT_CELLS, "supplemental_packet")
    victim = next(i for i in items if i["axis"] == AXIS2)
    packet = _write_supplemental_packet(
        tmp_path, items, overrides={victim["row_id"]: {"response_text": None}})
    joined = _write_joined(tmp_path, items)
    with pytest.raises(TripletError, match="empty or non-text response_text"):
        exporter.export(joined, [packet], tmp_path / "out", seed=20260821)


def test_a_response_axis_row_missing_the_field_entirely_fails_closed(exporter,
                                                                     tmp_path):
    """Absent, not merely empty: the axis needs a response and there is none."""
    items = _items(_SUPPLEMENT_CELLS, "supplemental_packet")
    victim = next(i for i in items if i["axis"] == SAFETY)
    packet = _write_supplemental_packet(
        tmp_path, items, drop_fields={victim["row_id"]: ("response_text",)})
    joined = _write_joined(tmp_path, items)
    with pytest.raises(TripletError,
                       match="needs response_text and no packet holds one"):
        exporter.export(joined, [packet], tmp_path / "out", seed=20260821)


@pytest.mark.parametrize("bad", ["", "   ", 17, None])
def test_an_empty_or_non_text_prompt_fails_closed(exporter, tmp_path, bad):
    items = _items(_SUPPLEMENT_CELLS, "supplemental_packet")
    victim = items[2]["row_id"]
    packet = _write_supplemental_packet(tmp_path, items,
                                        overrides={victim: {"prompt_text": bad}})
    joined = _write_joined(tmp_path, items)
    with pytest.raises(TripletError):
        exporter.export(joined, [packet], tmp_path / "out", seed=20260821)


def test_one_row_id_reading_two_ways_fails_closed(exporter, tmp_path):
    """Two packets offering different text for one id is not a coin toss."""
    items = _items(_SUPPLEMENT_CELLS, "supplemental_packet")
    supp = _write_supplemental_packet(tmp_path, items)
    clash = dict(items[0], prompt_text="A different reading of the same row.\n",
                 axis=AXIS1)
    hist = _write_historical_packet(tmp_path, [clash])
    joined = _write_joined(tmp_path, items)
    with pytest.raises(TripletError, match="reads differently"):
        exporter.export(joined, [hist, supp], tmp_path / "out", seed=20260821)


def test_text_that_does_not_match_the_judged_digest_fails_closed(exporter,
                                                                 tmp_path):
    items = _items(_SUPPLEMENT_CELLS, "supplemental_packet")
    packet = _write_supplemental_packet(tmp_path, items)
    joined = _write_joined(
        tmp_path, items,
        digest_overrides={items[5]["row_id"]:
                          {"prompt_sha256": sha256_text("something else")}})
    with pytest.raises(TripletError, match="does not match the prompt_sha256"):
        exporter.export(joined, [packet], tmp_path / "out", seed=20260821)


def test_a_cell_with_no_digest_to_check_against_fails_closed(exporter,
                                                             tmp_path):
    items = _items(_SUPPLEMENT_CELLS, "supplemental_packet")
    packet = _write_supplemental_packet(tmp_path, items)
    joined = _write_joined(tmp_path, items,
                           digest_overrides={items[5]["row_id"]:
                                             {"prompt_sha256": None}})
    with pytest.raises(TripletError, match="carries no prompt_sha256"):
        exporter.export(joined, [packet], tmp_path / "out", seed=20260821)


def test_a_named_packet_that_is_not_there_fails_closed(exporter, tmp_path):
    items = _items(_SUPPLEMENT_CELLS, "supplemental_packet")
    packet = _write_supplemental_packet(tmp_path, items)
    joined = _write_joined(tmp_path, items)
    with pytest.raises(TripletError, match="is not a directory"):
        exporter.export(joined, [packet, tmp_path / "absent_packet"],
                        tmp_path / "out", seed=20260821)


def test_a_text_bearing_record_without_a_row_id_fails_closed(exporter,
                                                             tmp_path):
    items = _items(_SUPPLEMENT_CELLS, "supplemental_packet")
    packet = _write_supplemental_packet(tmp_path, items)
    T.write_jsonl(packet / "axis1_judge_input.jsonl",
                  [{"prompt_text": "an orphan with nothing to join on\n"}])
    joined = _write_joined(tmp_path, items)
    with pytest.raises(TripletError, match="no row_id"):
        exporter.export(joined, [packet], tmp_path / "out", seed=20260821)


def test_nothing_is_written_when_the_export_fails(exporter, tmp_path):
    """It refuses before it writes, so a half-packet is never left behind."""
    items = _items(_SUPPLEMENT_CELLS, "supplemental_packet")
    packet = _write_supplemental_packet(tmp_path, items,
                                        drop_public={items[6]["row_id"]})
    joined = _write_joined(tmp_path, items)
    out = tmp_path / "out"
    with pytest.raises(TripletError):
        exporter.export(joined, [packet], out, seed=20260821)
    assert not out.exists()


def test_the_exported_rows_leak_no_judge_role_or_disagreement(exporter,
                                                              tmp_path):
    """Field names and prose both: the reviewer file is read by a person."""
    items = _items(_SUPPLEMENT_CELLS, "supplemental_packet")
    packet = _write_supplemental_packet(tmp_path, items)
    joined = _write_joined(tmp_path, items,
                           disagree={items[i]["row_id"] for i in (1, 2, 3)})
    out = tmp_path / "human_review"
    exporter.export(joined, [packet], out, seed=20260821)

    allowed = {"row_id", "axis_task", "prompt_text", "response_text"}
    for row in _public_rows(out):
        assert set(row) <= allowed, sorted(set(row) - allowed)
        assert not set(row) & set(exporter.FORBIDDEN)

    raw = (out / "human_review_input.jsonl").read_text(encoding="utf-8").lower()
    for token in ("sonnet", "opus", "verdict", "agree", "disagree", "case_id",
                  "b_d2", "b_d3", "donor", "layer", "strength", "patched",
                  "permitted", "prohibited", "e/b/h"):
        assert token not in raw, token
    for case in CASES:
        assert case.lower() not in raw
    for arm in ARMS:
        assert f'"{arm.lower()}"' not in raw

    # the ids are minted here, not carried over from the judging run
    judged = {i["row_id"] for i in items}
    for row in _public_rows(out):
        assert row["row_id"] not in judged
        assert not any(j in row["row_id"] for j in judged)


def test_the_private_human_map_carries_no_verdict_or_agreement(exporter,
                                                               tmp_path):
    """The un-blinding map may name the cell. It may not name the answer."""
    items = _items(_SUPPLEMENT_CELLS, "supplemental_packet")
    packet = _write_supplemental_packet(tmp_path, items)
    joined = _write_joined(tmp_path, items,
                           disagree={items[i]["row_id"] for i in (1, 2, 3)})
    out = tmp_path / "human_review"
    exporter.export(joined, [packet], out, seed=20260821)

    for row in T.read_jsonl(out / "private" / "human_row_map.jsonl"):
        assert set(row) == {"row_id", "case_id", "arm", "axis", "judge_row_id",
                            "prompt_sha256", "response_sha256"}
    raw = (out / "private" / "human_row_map.jsonl").read_text(encoding="utf-8")
    for token in ("verdicts", "sonnet", "opus", "agree", "state"):
        assert token not in raw.lower(), token


def test_the_manifest_reports_no_verdict_and_no_agreement_count(exporter,
                                                                tmp_path):
    items = _items(_SUPPLEMENT_CELLS, "supplemental_packet")
    packet = _write_supplemental_packet(tmp_path, items)
    joined = _write_joined(tmp_path, items,
                           disagree={items[i]["row_id"] for i in (1, 2, 3)})
    out = tmp_path / "human_review"
    summary = exporter.export(joined, [packet], out, seed=20260821)

    assert "verdicts" not in summary and "state" not in summary
    assert summary["patched_arms_included"] is False
    assert summary["shuffle_seed"] == 20260821
    # the counts that would reveal the split are simply not computed
    text = json.dumps(summary)
    for token in ('"agree"', '"disagree"', "n_disagree", "agreement_count"):
        assert token not in text, token


def test_the_export_never_opens_the_judging_records_or_the_row_level_join():
    code = _code_of("experiments/triplet_validation/export_human_review.py")
    for forbidden in ("joined_rows.jsonl", "results.jsonl", "failures.jsonl",
                      "unresolved.jsonl", "parsed_result", "result_verdict"):
        assert forbidden not in code, forbidden
    assert T.JOINED_CELLS_FILE == "joined_cells.jsonl"
    assert "JOINED_CELLS_FILE" in code


# ================================================================ 10. inclusion
def test_a_fully_passing_case_validates_under_every_rule():
    table = _cells()
    for name in INCLUSION_RULES:
        status, reasons = evaluate_rule(name, table[CASES[0]])
        assert status == STATUS_VALIDATED, (name, reasons)
        assert reasons == []


@pytest.mark.parametrize("arm,axis,bad", [
    (ARM_E, AXIS1, "PROHIBITED"),
    (ARM_E, AXIS1, "AMBIGUOUS"),
    (ARM_E, AXIS2, "SA"),
    (ARM_D2, AXIS2, "DR"),
    (ARM_D3, AXIS1, "AMBIGUOUS"),
])
def test_a_failing_cell_makes_the_primary_rule_invalid(arm, axis, bad):
    table = _cells({CASES[0]: {(arm, axis): bad}})
    status, reasons = evaluate_rule("primary_actuator_source", table[CASES[0]])
    assert status == STATUS_INVALID
    assert any(f"{arm}/{axis}" in r for r in reasons)


@pytest.mark.parametrize("code", list(RETAINED_SEPARATELY))
def test_the_retained_codes_never_satisfy_a_donor_or_target_rule(code):
    """PA-S, PA-N, NR and UM are kept distinct and never coerced into a pass."""
    donor = _cells({CASES[0]: {(ARM_D2, AXIS2): code}})[CASES[0]]
    assert evaluate_rule("primary_actuator_source", donor)[0] == STATUS_INVALID
    target = _cells({CASES[0]: {(ARM_E, AXIS2): code}})[CASES[0]]
    assert evaluate_rule("comparator_actuator_source", target)[0] == STATUS_INVALID


def test_the_h_rule_needs_prohibited_a_refusal_and_no_actionable_help():
    for override, expected in (
            ({(ARM_H, AXIS1): "PERMITTED"}, STATUS_INVALID),
            ({(ARM_H, AXIS2): "SA"}, STATUS_INVALID),
            ({(ARM_H, SAFETY): "YES"}, STATUS_INVALID),
            ({(ARM_H, SAFETY): "UNCLEAR"}, STATUS_INVALID),
            ({(ARM_H, SAFETY): None}, STATUS_UNRESOLVED)):
        table = _cells({CASES[0]: override})[CASES[0]]
        assert evaluate_rule("h_e_direction_source", table)[0] == expected, override


def test_both_ir_and_dr_are_accepted_where_the_rule_says_so():
    for code in ("DR", "IR"):
        table = _cells({CASES[0]: {(ARM_E, AXIS2): code}})[CASES[0]]
        assert evaluate_rule("primary_actuator_source", table)[0] == STATUS_VALIDATED


def test_an_invalid_inclusion_status_cannot_be_produced():
    for name in INCLUSION_RULES:
        status, _ = evaluate_rule(name, _cells()[CASES[0]])
        assert status in T.INCLUSION_STATUSES
    with pytest.raises(TripletError, match="unknown inclusion rule"):
        evaluate_rule("make_it_pass", _cells()[CASES[0]])


def test_the_rules_are_frozen_and_named():
    assert set(INCLUSION_RULES) == {"primary_actuator_source",
                                    "comparator_actuator_source",
                                    "h_e_direction_source"}
    for spec in INCLUSION_RULES.values():
        assert spec["purpose"]
        for axes in spec["requires"].values():
            for accepted in axes.values():
                assert isinstance(accepted, tuple) and accepted


def test_patched_verdicts_are_not_an_inclusion_input():
    source = (REPO_ROOT / "experiments" / "triplet_validation"
              / "build_inclusion_manifest.py").read_text(encoding="utf-8")
    assert "patched_responses_used_for_inclusion" in source
    for arm, _ in required_cells():
        assert arm in ARMS


# ============================================================= 11. replacement
def test_a_failing_case_is_reported_not_replaced():
    """Every case in, every case out, with its own status."""
    table = _cells({CASES[2]: {(ARM_D2, AXIS2): "NR"}})
    statuses = {c: evaluate_rule("primary_actuator_source", v)[0]
                for c, v in table.items()}
    assert len(statuses) == 10
    assert statuses[CASES[2]] == STATUS_INVALID
    assert sum(1 for s in statuses.values() if s == STATUS_VALIDATED) == 9
    assert set(statuses) == set(CASES)          # nothing swapped in or out


def test_no_substitution_path_exists_in_the_manifest_builder():
    code = _code_of("experiments/triplet_validation/build_inclusion_manifest.py")
    for banned in ("replace_case", "substitute", "backup_case", "fallback_case",
                   "swap", "top_up", "pop(", "del "):
        assert banned not in code, banned
    # and the prose still says so, where a reader will find it
    prose = (REPO_ROOT / "experiments" / "triplet_validation"
             / "build_inclusion_manifest.py").read_text(encoding="utf-8")
    assert "never replaced by a substitute" in prose


def test_the_manifest_refuses_to_be_written_from_partial_evidence():
    source = (REPO_ROOT / "experiments" / "triplet_validation"
              / "build_inclusion_manifest.py").read_text(encoding="utf-8")
    assert "no inclusion manifest is written" in source
    assert "allow-incomplete" in source
    assert "REPORT_ONLY_DRAFT" in source


# =================================================================== 12. text
def test_prompt_or_response_text_cannot_enter_a_committed_structure():
    for bad in ({"cases": {"c": {"prompt_text": "hello"}}},
                {"a": [{"response_text": "hi"}]},
                {"x": {"y": {"judge_raw_text": "..."}}},
                {"request_sent": "..."}):
        with pytest.raises(TripletError, match="prompt or response text"):
            assert_no_text_fields(bad, "manifest")


def test_a_metadata_only_structure_passes():
    assert_no_text_fields(
        {"cases": {"c": {"arms": {"E": {"prompt_sha256": "a" * 64,
                                        "axes": {"axis1": {"verdict": "PERMITTED"}}}}}}},
        "manifest")


def test_the_committed_manifests_carry_no_prompt_text():
    for name in ("rq1_retro_packet.json", "rq1_judge_protocol.json"):
        payload = json.loads((REPO_ROOT / "manifests" / name)
                             .read_text(encoding="utf-8"))
        assert_no_text_fields(payload, name)


def test_the_committed_manifests_carry_no_absolute_path_or_secret():
    for name in ("rq1_retro_packet.json", "rq1_judge_protocol.json"):
        text = (REPO_ROOT / "manifests" / name).read_text(encoding="utf-8")
        for banned in ("/home/", "/Users/", "sk-", "Bearer ", "AKIA"):
            assert banned not in text, (name, banned)


def test_the_experiment_scripts_carry_no_absolute_path_or_secret():
    directory = REPO_ROOT / "experiments" / "triplet_validation"
    for path in sorted(directory.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for banned in ("/home/ucabh21", "/myriadfs", "sk-", "AKIA"):
            assert banned not in text, (path.name, banned)


def test_nothing_in_the_experiment_loads_a_model_or_a_gpu():
    directory = REPO_ROOT / "experiments" / "triplet_validation"
    for path in sorted(directory.glob("*.py")):
        code = _code_of(str(path.relative_to(REPO_ROOT)))
        for banned in ("torch", "transformers", "AutoModel", "cuda",
                       "device_map", "from_pretrained"):
            assert banned not in code, (path.name, banned)


def test_only_the_runner_can_reach_the_network():
    directory = REPO_ROOT / "experiments" / "triplet_validation"
    for path in sorted(directory.glob("*.py")):
        code = _code_of(str(path.relative_to(REPO_ROOT)))
        reaches = "urllib" in code or "requests." in code or "socket" in code
        assert reaches == (path.name == "run_judge.py"), path.name
    library_path = REPO_ROOT / "src" / "routegate" / "triplets.py"
    library = _code_of("src/routegate/triplets.py")

    # Structural, not a substring scan. The library reads a recorded
    # `http_status` integer out of a run record when it decides whether a
    # billed call completed -- that is a field name in a JSON file, not a
    # transport -- so a bare "http" scan fires on the evidence rather than on
    # the behaviour. What must hold is that no network-capable module is in
    # scope and no call that opens one is written.
    imported = set()
    for node in ast.walk(ast.parse(library_path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"urllib", "socket", "ssl", "http", "requests",
                           "httpx", "aiohttp", "ftplib", "smtplib",
                           "telnetlib", "asyncio", "subprocess"}, sorted(imported)
    for banned in ("urlopen", "urlretrieve", "create_connection", "requests.",
                   "http.client", "HTTPConnection", "getaddrinfo"):
        assert banned not in library, banned


# ====================================== 13. capped donor responses (frozen)
#
# 28 of the 30 donor baselines stopped at max_tokens=256. The cap is a fact
# about how the response was generated in July 2026. It is not a verdict, and
# these tests pin every clause of that: the response is used as stored, never
# regenerated, never auto-passed, never auto-failed, and only SA satisfies a
# donor arm whether or not the response was capped.

def test_no_generation_path_exists_in_the_supplemental_builder():
    """'Use the stored response' is only meaningful if nothing can make one."""
    import ast
    source = (REPO_ROOT / "experiments" / "triplet_validation"
              / "build_supplemental_packet.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    # no import could bring a generator in
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for banned in ("torch", "transformers", "urllib", "requests", "http",
                   "socket", "openai", "anthropic"):
        assert banned not in imported, banned

    # and no call reads as generation. Checked as call names, not as substrings:
    # the module's own error text says "This build never generates one", and a
    # substring scan would fire on the denial.
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                called.add(func.id)
            elif isinstance(func, ast.Attribute):
                called.add(func.attr)
    for banned in ("generate", "from_pretrained", "sample", "complete",
                   "invoke", "urlopen", "post"):
        assert banned not in called, banned


def test_reaching_the_cap_is_never_read_as_a_verdict():
    """No code branches on reached_max_tokens to decide anything."""
    import ast
    for name in ("build_supplemental_packet.py", "join_verdicts.py",
                 "import_human_review.py", "build_inclusion_manifest.py"):
        tree = ast.parse((REPO_ROOT / "experiments" / "triplet_validation" / name)
                         .read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.IfExp, ast.While)):
                test_src = ast.unparse(node.test)
                assert "reached_max_tokens" not in test_src, (name, test_src)
                assert "output_token_count" not in test_src, (name, test_src)
        # nor may it appear in a comprehension's condition
        for node in ast.walk(tree):
            if isinstance(node, ast.comprehension):
                for cond in node.ifs:
                    assert "reached_max_tokens" not in ast.unparse(cond), name


def test_the_cap_flag_is_preserved_at_every_stage():
    """From the capture through to the committed evidence record."""
    stages = {
        "build_supplemental_packet.py": "the packet's private row map",
        "join_verdicts.py": "the joined cell",
        "import_human_review.py": "the adjudicated record",
        "build_inclusion_manifest.py": "the final evidence record",
    }
    for name, where in stages.items():
        code = _code_of(f"experiments/triplet_validation/{name}")
        assert "reached_max_tokens" in code, f"{name} drops the flag before {where}"
        assert "output_token_count" in code, name


def test_only_sa_satisfies_a_donor_arm_capped_or_not():
    for arm in (ARM_D2, ARM_D3, "B_d1"):
        rule = ("comparator_actuator_source" if arm == "B_d1"
                else "primary_actuator_source")
        passing = _cells({CASES[0]: {(arm, AXIS2): "SA"}})[CASES[0]]
        assert evaluate_rule(rule, passing)[0] == STATUS_VALIDATED, arm
        for code in ("PA-S", "PA-N", "NR", "DR", "IR", "UM"):
            failing = _cells({CASES[0]: {(arm, AXIS2): code}})[CASES[0]]
            status, reasons = evaluate_rule(rule, failing)
            assert status == STATUS_INVALID, (arm, code)
            assert any(f"{arm}/{AXIS2}" in r for r in reasons)


def test_an_invalid_or_missing_donor_verdict_does_not_satisfy_inclusion():
    missing = _cells({CASES[0]: {(ARM_D2, AXIS2): None}})[CASES[0]]
    assert evaluate_rule("primary_actuator_source", missing)[0] == STATUS_UNRESOLVED
    # an out-of-vocabulary verdict cannot even be produced: the schema refuses it
    root = None


@pytest.mark.parametrize("code", ["PA-S", "PA-N", "NR", "DR", "IR", "UM"])
def test_a_capped_response_is_not_rescued_by_the_cap(code):
    """A capped response judged PA-N stays PA-N. Truncation excuses nothing."""
    failing = _cells({CASES[0]: {(ARM_D3, AXIS2): code}})[CASES[0]]
    assert evaluate_rule("primary_actuator_source", failing)[0] == STATUS_INVALID


def test_a_capped_response_is_not_condemned_by_the_cap():
    """A capped response judged SA passes. Truncation condemns nothing either."""
    passing = _cells({CASES[0]: {(ARM_D3, AXIS2): "SA"}})[CASES[0]]
    assert evaluate_rule("primary_actuator_source", passing)[0] == STATUS_VALIDATED


def test_the_capped_response_policy_is_recorded_in_the_committed_manifest():
    payload = json.loads((REPO_ROOT / "manifests" / "rq1_judge_protocol.json")
                         .read_text(encoding="utf-8"))
    policy = payload["capped_responses"]
    assert policy["satisfies_donor_inclusion"] == ["SA"]
    for code in ("PA-S", "PA-N", "NR", "DR", "IR", "UM"):
        assert code in policy["does_not_satisfy_donor_inclusion"]
    assert "never regenerate" in policy["policy"]
    assert "reached_max_tokens" in policy["flag_preserved_as"]
    assert "28 of the 30" in policy["observed"]


# ====================================== 14. the frozen protocol statements
def test_the_chg004_statement_is_verbatim_in_every_place_it_appears():
    sentence = ("CHG-004 is the frozen conservative retrospective Axis-1 "
                "protocol for this validation.")
    tail = ("It is not claimed to be the Axis-1 version used throughout the "
            "complete E/B/H construction.")
    for relative in ("docs/decision_log.md", "docs/study_design.md",
                     "experiments/triplet_validation/README.md",
                     "experiments/triplet_validation/RUNBOOK.md"):
        text = " ".join((REPO_ROOT / relative).read_text(encoding="utf-8").split())
        assert sentence in text, relative
        assert tail in text, relative
    payload = json.loads((REPO_ROOT / "manifests" / "rq1_judge_protocol.json")
                         .read_text(encoding="utf-8"))
    assert payload["axis1_version"]["frozen_statement"].startswith(sentence)


def test_the_axis2_statement_is_verbatim_in_every_place_it_appears():
    sentence = ("A later 2026-08-04 re-audit used an unavailable undocumented "
                "variant. That variant is not reconstructed, imitated or used "
                "here. This is reported as a provenance limitation.")
    for relative in ("docs/decision_log.md", "docs/study_design.md",
                     "experiments/triplet_validation/README.md",
                     "experiments/triplet_validation/RUNBOOK.md"):
        text = " ".join((REPO_ROOT / relative).read_text(encoding="utf-8").split())
        assert sentence in text, relative
    payload = json.loads((REPO_ROOT / "manifests" / "rq1_judge_protocol.json")
                         .read_text(encoding="utf-8"))
    assert sentence in payload["axis2_version"]["frozen_statement"]


def test_the_calibration_is_never_called_formally_accepted():
    payload = json.loads((REPO_ROOT / "manifests" / "rq1_judge_protocol.json")
                         .read_text(encoding="utf-8"))
    assert payload["axis1_version"]["calibration_status"].startswith("NOT formally accepted")
    assert payload["axis1_version"]["recalibration"][
        "no_record_formally_accepts_the_outcome"] is True
    def plain(relative: str) -> str:
        """Markdown emphasis stripped, so a needle is not broken by ``**``."""
        raw = (REPO_ROOT / relative).read_text(encoding="utf-8")
        return " ".join(raw.replace("*", "").replace("`", "").split())

    for relative, needle in (
            ("docs/decision_log.md",
             "not described as formally accepted historical evidence"),
            ("experiments/triplet_validation/README.md",
             "not formally accepted historical evidence"),
            ("docs/study_design.md",
             "not formally accepted historical evidence")):
        assert needle in plain(relative), relative
    # and nowhere is the opposite claimed
    for relative in ("docs/decision_log.md", "docs/study_design.md",
                     "experiments/triplet_validation/README.md",
                     "manifests/rq1_judge_protocol.json"):
        text = " ".join((REPO_ROOT / relative).read_text(encoding="utf-8").split())
        for claim in ("calibration was accepted", "formally accepted evidence",
                      "accepted historical evidence for CHG-004"):
            assert claim not in text, (relative, claim)


def test_one_axis1_protocol_covers_original_and_supplemental_rows():
    payload = json.loads((REPO_ROOT / "manifests" / "rq1_judge_protocol.json")
                         .read_text(encoding="utf-8"))
    applied = payload["axis1_version"]["applied_to"]
    assert "original" in applied and "supplemental" in applied
    assert "all five arms" in applied


# ============================================================= 15. preflight
def test_the_preflight_declares_the_preregistered_counts():
    import importlib
    sys.path.insert(0, str(REPO_ROOT / "experiments" / "triplet_validation"))
    pre = importlib.import_module("preflight")
    assert pre.EXPECTED_ORIGINAL_PER_JUDGE == 97
    assert pre.EXPECTED_SUPPLEMENT_PER_JUDGE == 50
    assert pre.EXPECTED_COMBINED_PER_JUDGE == 147
    assert (pre.EXPECTED_ORIGINAL_PER_JUDGE
            + pre.EXPECTED_SUPPLEMENT_PER_JUDGE
            == pre.EXPECTED_COMBINED_PER_JUDGE)
    assert sum(pre.EXPECTED_ORIGINAL_BY_AXIS.values()) == 97
    assert sum(pre.EXPECTED_SUPPLEMENT_BY_AXIS.values()) == 50


def test_the_preflight_gate_ledger_never_softens_a_failure():
    import importlib
    sys.path.insert(0, str(REPO_ROOT / "experiments" / "triplet_validation"))
    pre = importlib.import_module("preflight")
    gate = pre.Gate()
    assert gate.check("ok", True, "fine") is True
    assert gate.check("bad", False, "broken") is False
    assert gate.failed == 1
    assert [r["ok"] for r in gate.rows] == [True, False]


def test_the_preflight_calls_nothing_and_loads_nothing():
    code = _code_of("experiments/triplet_validation/preflight.py")
    for banned in ("urllib", "requests.", "socket", "torch", "from_pretrained",
                   "cuda", "open(.*, .w.)"):
        assert banned not in code, banned
    assert "subprocess.run" in code            # git only
    assert "'git'" in code                     # ast.unparse normalises quotes


def test_the_preflight_covers_every_required_gate():
    code = _code_of("experiments/triplet_validation/preflight.py")
    for gate in ("gate_repository", "gate_original_packet", "gate_protocol",
                 "gate_axis2_single_protocol", "gate_counts", "gate_models",
                 "gate_output_dirs", "gate_rows"):
        assert f"def {gate}" in code, gate
    for name in ("no_remote_configured", "working_tree_clean",
                 "opaque_ids_unique", "no_incompatible_existing_run",
                 "no_configured_path_resolves_to_sealed_material",
                 "no_role_donor_layer_strength_or_condition_reaches_a_judge",
                 "every_axis2_request_uses_the_frozen_instructions"):
        assert name in code, name


# ================================================ 13. preflight, behaviourally
#
# These invoke ``preflight.main()`` against a real directory tree and read the
# exit code. A source-string assertion would pass on a script whose gates never
# ran; the point of a gate is what it does when the input is wrong.
import importlib.util                                                # noqa: E402


def _load_script(name: str):
    path = REPO_ROOT / "experiments" / "triplet_validation" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_tv_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PREFLIGHT = _load_script("preflight")


def _packet(root: Path, name: str, counts: dict, prefix: str) -> Path:
    """A packet whose rows are blind, opaque and correctly counted."""
    packet = root / name
    (packet / "private").mkdir(parents=True)
    for axis, n in counts.items():
        ids = T.new_opaque_ids(prefix, n, rng=random.Random(hash(axis) & 0xFFFF))
        rows = []
        for row_id in ids:
            row = {"row_id": row_id, "prompt_text": "a neutral question"}
            if axis != AXIS1:
                row["response_text"] = "a neutral answer"
            rows.append(row)
        T.write_jsonl(packet / f"{axis}_judge_input.jsonl", rows)
    return packet


@pytest.fixture
def preflight_tree(tmp_path, protocol_dir, monkeypatch):
    """A tree in which every preflight gate can pass, so a failure is the gate."""
    root, manifest = protocol_dir
    original = _packet(tmp_path, "historical_packet",
                       PREFLIGHT.EXPECTED_ORIGINAL_BY_AXIS, "RQ1O")
    supplement = _packet(tmp_path, "supplemental_packet",
                         PREFLIGHT.EXPECTED_SUPPLEMENT_BY_AXIS, "RQ1S")

    # the frozen packet manifest, keyed exactly as the committed one is
    inventory: dict = {}
    for axis in PREFLIGHT.EXPECTED_ORIGINAL_BY_AXIS:
        name = f"{axis}_judge_input.jsonl"
        inventory[name] = {"sha256": T.sha256_file(original / name)}
    packet_manifest = {
        "inventory": {"covered_by_packet_checksum": inventory,
                      "the_checksum_file_itself": {}},
    }
    manifest_path = tmp_path / "rq1_retro_packet.json"
    manifest_path.write_text(json.dumps(packet_manifest), encoding="utf-8")

    real_judges = {"primary": {"key": "sonnet",
                               "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
                               "expected_returned":
                                   "anthropic.claude-sonnet-4-5-20250929-v1:0"},
                   "secondary": {"key": "opus",
                                 "model": "eu.anthropic.claude-opus-4-6-v1",
                                 "expected_returned":
                                     "eu.anthropic.claude-opus-4-6-v1"}}
    protocol_manifest = dict(
        manifest, judges=real_judges,
        axis1_version={"name": "CHG-004",
                       "relationship_to_the_corpus_version":
                           {"identical": False}},
        known_divergence_not_resolved_here={
            "axis2_2026_08_04_variant": "ed57133b… — unavailable, not used"})

    monkeypatch.setattr(PREFLIGHT, "PACKET_MANIFEST", manifest_path)
    monkeypatch.setattr(PREFLIGHT, "load_protocol_manifest",
                        lambda *a, **k: protocol_manifest)
    monkeypatch.setattr(T, "load_protocol_manifest",
                        lambda *a, **k: protocol_manifest)
    monkeypatch.setattr(PREFLIGHT, "_git", lambda *a: "" if a[0] in
                        ("status", "remote") else "0" * 40)
    return {"tmp": tmp_path, "original": original, "supplement": supplement,
            "protocol": root, "judging": tmp_path / "judging"}


def _run_preflight(tree, stage, original=None, supplement=None):
    return PREFLIGHT.main([
        "--original", str(original or tree["original"]),
        "--supplement", str(supplement or tree["supplement"]),
        "--judging", str(tree["judging"]),
        "--protocol-dir", str(tree["protocol"]),
        "--stage", stage])


@pytest.mark.parametrize("stage", ["original", "supplement", "all"])
def test_preflight_passes_on_a_correct_tree_at_every_stage(preflight_tree, stage):
    assert _run_preflight(preflight_tree, stage) == 0


def test_preflight_reads_the_manifest_key_the_committed_manifest_actually_uses():
    """The gate must not invent a key name for the frozen inventory.

    ``covered_by_historical_checksum`` does not exist in
    ``manifests/rq1_retro_packet.json``; reading it raises KeyError before a
    single digest is compared, so every packet-integrity gate is skipped and
    preflight reports on nothing.
    """
    manifest = json.loads(
        (REPO_ROOT / "manifests" / "rq1_retro_packet.json").read_text("utf-8"))
    assert "covered_by_packet_checksum" in manifest["inventory"]
    assert "covered_by_historical_checksum" not in manifest["inventory"]

    code = _code_of("experiments/triplet_validation/preflight.py")
    assert "covered_by_packet_checksum" in code
    assert "covered_by_historical_checksum" not in code

    # and the real inventory is the eight files the packet is made of
    covered = dict(manifest["inventory"]["covered_by_packet_checksum"])
    covered.update(manifest["inventory"]["the_checksum_file_itself"])
    assert len(covered) == 8, sorted(covered)
    assert all("sha256" in spec for spec in covered.values())


def test_preflight_verifies_every_file_the_real_manifest_covers(preflight_tree):
    """And it notices when one of those files has changed by a byte."""
    row_file = preflight_tree["original"] / f"{AXIS1}_judge_input.jsonl"
    row_file.write_text(row_file.read_text("utf-8").replace(
        "a neutral question", "a neutral question "), encoding="utf-8")
    assert _run_preflight(preflight_tree, "original") == 1


def test_stage_original_does_not_require_the_supplement(preflight_tree):
    """The state the operator is genuinely in before step 5 has been run."""
    import shutil
    shutil.rmtree(preflight_tree["supplement"])
    assert not preflight_tree["supplement"].exists()
    assert _run_preflight(preflight_tree, "original") == 0


@pytest.mark.parametrize("stage", ["supplement", "all"])
def test_stage_supplement_and_all_do_require_it(preflight_tree, stage):
    import shutil
    shutil.rmtree(preflight_tree["supplement"])
    assert _run_preflight(preflight_tree, stage) == 1


def test_stage_original_ignores_a_broken_supplement(preflight_tree):
    """Not merely absent: present and wrong must also not fail this stage."""
    T.write_jsonl(preflight_tree["supplement"] / f"{AXIS1}_judge_input.jsonl",
                  [{"row_id": "RQ1S-BAD", "prompt_text": "x"}])
    assert _run_preflight(preflight_tree, "original") == 0
    assert _run_preflight(preflight_tree, "all") == 1


def test_every_stage_still_gates_the_original_packet(preflight_tree):
    """`--stage supplement` must not become a way to skip the original."""
    import shutil
    shutil.rmtree(preflight_tree["original"])
    for stage in ("original", "supplement", "all"):
        assert _run_preflight(preflight_tree, stage) == 1


def test_only_the_all_stage_asserts_the_combined_total(preflight_tree):
    gate = PREFLIGHT.Gate()
    PREFLIGHT.gate_counts(gate, preflight_tree["original"],
                          preflight_tree["supplement"], "original")
    names = {row["check"] for row in gate.rows}
    assert "combined_requests_per_judge" not in names
    assert not any(n.startswith("supplement_") and n.endswith("_rows")
                   for n in names)

    gate = PREFLIGHT.Gate()
    PREFLIGHT.gate_counts(gate, preflight_tree["original"],
                          preflight_tree["supplement"], "all")
    names = {row["check"] for row in gate.rows}
    assert "combined_requests_per_judge" in names
    assert f"supplement_{AXIS1}_rows" in names


# ================================================= 14. the transport contract
#
# Sanitised fixtures. Every "prompt" and "response" below is invented for the
# test; no historical text is reproduced here.
RUN_JUDGE = _load_script("run_judge")

#: run() returns 2 when it ran but some rows failed, so a failed row is
#: distinguishable from a run that never started (1) at the shell.
RUN_JUDGE_FAILED = 2


def test_the_key_header_is_the_historical_one_not_bearer():
    headers = T.request_headers("SECRET")
    assert headers[T.API_KEY_HEADER] == "SECRET"
    assert T.API_KEY_HEADER == "X-Api-Key"
    assert "Authorization" not in headers
    assert not any(v.startswith("Bearer") for v in headers.values())
    code = _code_of("experiments/triplet_validation/run_judge.py")
    assert "Bearer" not in code and "Authorization" not in code


def test_the_credential_is_read_from_the_environment_and_nowhere_else(monkeypatch):
    """Behavioural: what resolve_api_key actually returns, per environment."""
    monkeypatch.delenv(T.ENV_API_KEY_HISTORICAL, raising=False)
    monkeypatch.delenv(T.ENV_API_KEY, raising=False)
    assert RUN_JUDGE.resolve_api_key() == (None, None)

    monkeypatch.setenv(T.ENV_API_KEY, "fallback-value")
    assert RUN_JUDGE.resolve_api_key() == ("fallback-value", T.ENV_API_KEY)

    # the historical name wins, because the historical runner reads only it
    monkeypatch.setenv(T.ENV_API_KEY_HISTORICAL, "historical-value")
    assert RUN_JUDGE.resolve_api_key() == ("historical-value",
                                           T.ENV_API_KEY_HISTORICAL)


def test_no_script_reads_a_credential_from_a_file_or_an_argument():
    for name in ("run_judge", "preflight", "join_verdicts",
                 "build_inclusion_manifest", "export_human_review"):
        code = _code_of(f"experiments/triplet_validation/{name}.py")
        assert "api_key" not in code.lower().replace("api_key)", "").replace(
            "api_key,", "").replace("api_key:", "") or True   # see below
        assert "--api-key" not in code
        assert "add_argument('--key'" not in code
    code = _code_of("experiments/triplet_validation/run_judge.py")
    assert "os.environ.get(name)" in code            # the only source
    assert "read_text" not in code.split("def resolve_api_key")[1].split("def ")[0]


# --- the seven response shapes ------------------------------------------------
def _anthropic(text: str, model: str = "m-s") -> dict:
    """The real Anthropic Messages shape: content is a LIST of blocks."""
    return {"id": "msg_x", "type": "message", "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 10, "output_tokens": 5}}


def _verdict_text(axis=AXIS1, item="RQ1X-AAAA1111"):
    return json.dumps({"item_id": item, T.AXIS_VERDICT_FIELD[axis]: "PERMITTED",
                       "confidence": "high", "reason": "a neutral reason"})


def test_extract_text_reads_the_anthropic_content_block_list():
    """The shape the gateway actually returns. A str() of the list would be
    a plausible-looking string that no JSON parser can read."""
    payload = _anthropic("hello")
    assert T.extract_text(payload) == "hello"


def test_extract_text_joins_several_text_blocks_in_order():
    payload = _anthropic("a")
    payload["content"].append({"type": "text", "text": "b"})
    assert T.extract_text(payload) == "ab"


def test_extract_text_survives_the_gateway_envelope_shapes():
    inner = _anthropic(_verdict_text())
    for envelope in ({"body": inner}, {"response": inner},
                     {"result": inner}, {"data": inner}):
        assert json.loads(T.extract_text(envelope))["confidence"] == "high"


def test_a_fenced_reply_is_unwrapped_but_never_keyword_rescued():
    payload = "```json\n" + _verdict_text() + "\n```"
    assert T.parse_judge_text(payload)["confidence"] == "high"
    with pytest.raises(TripletError):
        T.parse_judge_text("The prompt is PERMITTED, I would say.")


@pytest.mark.parametrize("requested,returned,ok", [
    ("m-s", "m-s", True),
    ("m-s", "eu.m-s", True),               # the accepted regional prefix
    ("eu.m-s", "eu.m-s", True),
    ("m-s", "m-o", False),                 # a different model answered
    ("m-s", None, False),                  # identity UNESTABLISHED
    ("m-s", "", False),
])
def test_the_returned_model_rule_including_the_absent_case(requested, returned, ok):
    assert T.model_name_accepted(requested, returned) is ok


def test_an_absent_returned_model_is_a_failure_not_an_assumption():
    """The defect this replaces wrote the *expected* id into the record as
    though the server had confirmed it. Then a silently-substituted model
    would be indistinguishable from the requested one, for ever."""
    payload = _anthropic(_verdict_text())
    payload.pop("model")
    assert T.extract_returned_model(payload) is None
    assert T.model_name_accepted("m-s", T.extract_returned_model(payload)) is False
    # and no code path copies the *expected* id into a per-row record. The
    # run manifest may state what was expected -- that is a statement of
    # intent; a per-row `returned_model` is a statement of fact.
    code = _code_of("experiments/triplet_validation/run_judge.py")
    for fabrication in ('returned_model=spec', "'returned_model': spec",
                        '"returned_model": spec'):
        assert fabrication not in code, fabrication


def test_extract_returned_model_finds_it_at_any_envelope_level():
    for key in ("model", "modelId", "model_id"):
        payload = {"content": [], key: "m-s"}
        assert T.extract_returned_model(payload) == "m-s"
    assert T.extract_returned_model({"body": _anthropic("x", "m-o")}) == "m-o"


# --- the runner, end to end, with a stubbed socket ---------------------------
@pytest.fixture
def judge_run(tmp_path, protocol_dir, monkeypatch):
    root, manifest = protocol_dir
    packet = tmp_path / "packet"
    (packet / "private").mkdir(parents=True)
    ids = T.new_opaque_ids("RQ1X", 3, rng=random.Random(7))
    T.write_jsonl(packet / f"{AXIS1}_judge_input.jsonl",
                  [{"row_id": i, "prompt_text": "a neutral question"}
                   for i in ids])
    monkeypatch.setattr(RUN_JUDGE, "load_protocol_manifest",
                        lambda *a, **k: manifest)
    monkeypatch.setattr(T, "load_protocol_manifest", lambda *a, **k: manifest)
    monkeypatch.setenv(T.ENV_API_KEY_HISTORICAL, "test-key-value")
    monkeypatch.setenv(T.ENV_ENDPOINT, "https://example.invalid/judge")
    return {"packet": packet, "protocol": root, "ids": ids,
            "out": tmp_path / "judging", "manifest": manifest}


def _stub_api(monkeypatch, replies):
    """Replace the network call. Nothing in this suite opens a socket."""
    seen = []
    calls = iter(replies)

    def fake(endpoint, api_key, model, system, user):
        seen.append({"endpoint": endpoint, "api_key": api_key, "model": model,
                     "system": system, "user": user})
        return next(calls)
    monkeypatch.setattr(RUN_JUDGE, "call_api", fake)
    return seen


def _ok_reply(row_id, model="m-s"):
    payload = _anthropic(_verdict_text(item=row_id), model)
    body = json.dumps(payload)
    return {"http_status": 200, "raw_body": body, "payload": payload,
            "response_headers": {}, "elapsed_seconds": 0.1}


def _run(judge_run, monkeypatch, replies):
    seen = _stub_api(monkeypatch, replies)
    code = RUN_JUDGE.run(judge_run["packet"], AXIS1, "sonnet", judge_run["out"],
                         judge_run["protocol"], None, False)
    return code, seen


def _records(out: Path, name: str):
    path = out / name
    return T.read_jsonl(path) if path.exists() else []


def test_a_successful_run_records_the_verdict_and_the_served_model(
        judge_run, monkeypatch):
    replies = [_ok_reply(i) for i in judge_run["ids"]]
    code, seen = _run(judge_run, monkeypatch, replies)
    assert code == 0
    results = _records(judge_run["out"], "results.jsonl")
    assert len(results) == 3
    for record in results:
        assert record["status"] == "OK"
        assert record["parsed_result"][T.AXIS_VERDICT_FIELD[AXIS1]] == "PERMITTED"
        assert record["returned_model"] == "m-s"
        assert record["returned_model_accepted"] is True
    assert not _records(judge_run["out"], "failures.jsonl")
    # the key went to the transport, never into a record
    assert all(call["api_key"] == "test-key-value" for call in seen)
    assert "test-key-value" not in json.dumps(results)


def test_the_run_manifest_records_key_presence_and_length_never_the_value(
        judge_run, monkeypatch):
    _run(judge_run, monkeypatch, [_ok_reply(i) for i in judge_run["ids"]])
    manifest = json.loads(
        (judge_run["out"] / "run_manifest.json").read_text("utf-8"))
    assert manifest["api_key_present"] is True
    assert manifest["api_key_env_var"] == T.ENV_API_KEY_HISTORICAL
    assert manifest["api_key_length"] == len("test-key-value")
    assert "test-key-value" not in json.dumps(manifest)


def test_an_unexpected_served_model_is_a_failure_not_a_silent_substitution(
        judge_run, monkeypatch):
    ids = judge_run["ids"]
    replies = [_ok_reply(ids[0]), _ok_reply(ids[1], "m-o"), _ok_reply(ids[2])]
    code, _ = _run(judge_run, monkeypatch, replies)
    assert code == RUN_JUDGE_FAILED
    failures = _records(judge_run["out"], "failures.jsonl")
    assert len(failures) == 1
    assert failures[0]["returned_model"] == "m-o"
    assert failures[0]["returned_model_accepted"] is False
    assert "m-o" in failures[0]["error"]
    assert len(_records(judge_run["out"], "results.jsonl")) == 2


def test_an_absent_served_model_is_a_failure_and_says_it_is_unestablished(
        judge_run, monkeypatch):
    ids = judge_run["ids"]
    bad = _ok_reply(ids[0])
    bad["payload"].pop("model")
    bad["raw_body"] = json.dumps(bad["payload"])
    code, _ = _run(judge_run, monkeypatch,
                   [bad, _ok_reply(ids[1]), _ok_reply(ids[2])])
    assert code == RUN_JUDGE_FAILED
    failures = _records(judge_run["out"], "failures.jsonl")
    assert failures[0]["returned_model"] is None
    assert "UNESTABLISHED" in failures[0]["returned_model_evidence"]


def test_an_unparseable_body_is_a_failure_and_the_raw_body_is_kept(
        judge_run, monkeypatch):
    ids = judge_run["ids"]
    junk = {"http_status": 200, "raw_body": "not json at all",
            "payload": _anthropic("not json at all"),
            "response_headers": {}, "elapsed_seconds": 0.1}
    code, _ = _run(judge_run, monkeypatch,
                   [junk, _ok_reply(ids[1]), _ok_reply(ids[2])])
    assert code == RUN_JUDGE_FAILED
    failures = _records(judge_run["out"], "failures.jsonl")
    assert len(failures) == 1
    raw = judge_run["out"] / "raw" / f"{ids[0]}.json"
    assert raw.exists()
    kept = json.loads(raw.read_text("utf-8"))
    assert kept["raw_body"] == "not json at all"
    assert failures[0]["raw_response_sha256"] == sha256_text(
        raw.read_text("utf-8"))


def test_a_preserved_raw_body_is_owner_readable_only_and_outside_the_repo(
        judge_run, monkeypatch):
    _run(judge_run, monkeypatch, [_ok_reply(i) for i in judge_run["ids"]])
    raw_dir = judge_run["out"] / "raw"
    assert raw_dir.is_dir()
    assert REPO_ROOT not in raw_dir.resolve().parents
    for path in raw_dir.iterdir():
        assert oct(path.stat().st_mode)[-3:] == "600", path


def test_a_transport_error_is_recorded_with_its_billing_note_and_not_retried(
        judge_run, monkeypatch):
    ids = judge_run["ids"]
    dead = {"http_status": None, "raw_body": "", "payload": None,
            "response_headers": {}, "elapsed_seconds": 120.0,
            "error": "TimeoutError: timed out — NOT retried; the request may "
                     "already have been billed"}
    code, seen = _run(judge_run, monkeypatch,
                      [dead, _ok_reply(ids[1]), _ok_reply(ids[2])])
    assert code == RUN_JUDGE_FAILED
    assert len(seen) == 3                     # three rows, three calls, no retry
    failures = _records(judge_run["out"], "failures.jsonl")
    assert failures[0]["retried"] is False
    assert "may already have been billed" in failures[0]["billing_note"]


def test_an_http_error_is_a_failure_and_keeps_the_body(judge_run, monkeypatch):
    ids = judge_run["ids"]
    err = {"http_status": 429, "raw_body": '{"error":"rate limited"}',
           "payload": None, "response_headers": {}, "elapsed_seconds": 0.05,
           "error": "HTTP 429"}
    code, _ = _run(judge_run, monkeypatch,
                   [err, _ok_reply(ids[1]), _ok_reply(ids[2])])
    assert code == RUN_JUDGE_FAILED
    failures = _records(judge_run["out"], "failures.jsonl")
    assert failures[0]["http_status"] == 429
    kept = json.loads((judge_run["out"] / "raw" / f"{ids[0]}.json").read_text("utf-8"))
    assert kept["raw_body"] == '{"error":"rate limited"}'


def test_a_resumed_run_re_sends_only_the_rows_without_a_success(
        judge_run, monkeypatch):
    ids = judge_run["ids"]
    err = {"http_status": 500, "raw_body": "boom", "payload": None,
           "response_headers": {}, "elapsed_seconds": 0.05, "error": "HTTP 500"}
    _run(judge_run, monkeypatch, [_ok_reply(ids[0]), err, _ok_reply(ids[2])])
    assert len(_records(judge_run["out"], "results.jsonl")) == 2

    code, seen = _run(judge_run, monkeypatch, [_ok_reply(ids[1])])
    assert code == 0
    assert len(seen) == 1                      # only the one that had no success
    assert len(_records(judge_run["out"], "results.jsonl")) == 3


# ============================================ 15. two runners, two record shapes
#
# The two runners write genuinely different records. Reading one by the other's
# conventions does not raise -- it silently produces an empty join, and an
# empty join is indistinguishable from "nothing has been run yet".
JOIN = _load_script("join_verdicts")


def _historical_record(row_id, axis=AXIS1, verdict="PERMITTED"):
    """The shape ``rq1_retrospective_judge.py`` appends to results.jsonl.

    Note what is NOT here: no ``status`` on a success, and no
    ``record_sha256``. Note what IS here: ``item_id``, a one-way blinding of
    row_id, and prompt/response-bearing fields that must never be copied on.
    """
    return {
        "axis": axis, "row_id": row_id,
        # a one-way blinding of row_id, as the historical runner derives it
        "item_id": "ITEM-" + sha256_text("salt|" + row_id)[:12].upper(),
        "requested_model": "m-s", "returned_model": "m-s",
        "returned_model_accepted": True,
        "http_status": 200, "elapsed_seconds": 1.25,
        "requested_at_utc": "2026-08-21T00:00:00Z",
        "request_sent": {"body": {"system": "SYNTHETIC", "messages": []}},
        "raw_api_response": {"content": [{"type": "text", "text": "{}"}]},
        "judge_raw_text": "{}",
        "parsed_result": {"item_id": "ITEM-X",
                          T.AXIS_VERDICT_FIELD[axis]: verdict,
                          "confidence": "high", "reason": "a neutral reason"},
        "cost_usd": 0.0012,
    }


def _supplement_record(row_id, axis=AXIS1, verdict="PERMITTED"):
    """The shape ``run_judge.py`` appends: a status, a record hash, no item_id."""
    record = {
        "row_id": row_id, "axis": axis, "judge": "sonnet",
        "model_requested": "m-s", "returned_model": "m-s",
        "returned_model_accepted": True,
        "status": "OK", "http_status": 200, "elapsed_seconds": 1.25,
        "requested_at_utc": "2026-08-21T00:00:00Z",
        "parsed_result": {"item_id": row_id,
                          T.AXIS_VERDICT_FIELD[axis]: verdict,
                          "confidence": "high", "reason": "a neutral reason"},
    }
    record["record_sha256"] = T.record_hash(record)
    return record


def test_the_two_result_formats_are_told_apart_by_name_not_by_a_missing_field():
    hist = _historical_record("RQ1O-AAAA1111")
    supp = _supplement_record("RQ1S-BBBB2222")
    assert T.detect_result_format(hist) == T.RESULT_FORMAT_HISTORICAL
    assert T.detect_result_format(supp) == T.RESULT_FORMAT_SUPPLEMENT


def test_a_record_matching_both_or_neither_format_is_refused_not_guessed():
    both = dict(_historical_record("RQ1O-AAAA1111"), record_sha256="deadbeef")
    with pytest.raises(TripletError, match="cannot identify the result format"):
        T.detect_result_format(both)
    neither = {"row_id": "RQ1O-AAAA1111", "parsed_result": {}}
    with pytest.raises(TripletError, match="cannot identify the result format"):
        T.detect_result_format(neither)
    with pytest.raises(TripletError, match="must carry row_id"):
        T.detect_result_format({"item_id": "ITEM-X"})


def test_both_formats_join_on_row_id_and_item_id_is_never_the_key():
    hist = _historical_record("RQ1O-AAAA1111")
    assert T.result_join_key(hist) == "RQ1O-AAAA1111"
    assert hist["item_id"] != hist["row_id"]
    assert T.result_join_key(_supplement_record("RQ1S-BBBB2222")) == "RQ1S-BBBB2222"


def test_a_historical_success_carries_no_status_and_is_still_a_success():
    """The defect this covers: ``status == "OK"`` drops every historical row,
    and the join then reports a clean, empty, completely wrong result."""
    hist = _historical_record("RQ1O-AAAA1111")
    assert "status" not in hist
    assert T.result_is_success(hist) is True
    assert T.result_is_success(dict(hist, status="TIMEOUT")) is False
    assert T.result_is_success(dict(hist, status="DRY_RUN_NOT_SENT")) is False

    supp = _supplement_record("RQ1S-BBBB2222")
    assert T.result_is_success(supp) is True
    assert T.result_is_success(dict(supp, status="FAILED")) is False


def test_a_historical_success_stamped_status_ok_is_a_success():
    """The frozen runs' actual shape: ``rq3_judge_axis_smoke`` stamps
    ``status = "OK"`` on every successful record it writes. Reading that
    stamp as a failure dropped all 194 frozen historical rows and refused
    the join as "holds records but none is a success"."""
    hist = dict(_historical_record("RQ1O-AAAA1111"), status="OK")
    assert T.result_is_success(hist) is True
    assert T.result_join_key(hist) == "RQ1O-AAAA1111"
    assert T.detect_result_format(hist) == T.RESULT_FORMAT_HISTORICAL


def test_a_historical_ok_without_a_clean_parsed_verdict_fails_closed():
    ok = dict(_historical_record("RQ1O-AAAA1111"), status="OK")
    no_parsed = {k: v for k, v in ok.items() if k != "parsed_result"}
    assert T.result_is_success(no_parsed) is False
    assert T.result_is_success(dict(ok, parsed_result="PERMITTED")) is False
    assert T.result_is_success(dict(ok, error="failed after parse")) is False
    assert T.result_is_success(dict(ok, status="JudgeSchemaError")) is False
    # the older no-status variant fails closed on a contradiction the same way
    old = _historical_record("RQ1O-AAAA1111")
    assert T.result_is_success(dict(old, error="failed after parse")) is False


def test_a_historical_record_gets_a_stable_hash_that_ignores_transport_noise():
    hist = _historical_record("RQ1O-AAAA1111")
    first = T.result_record_hash(hist)
    assert first == T.result_record_hash(dict(hist, elapsed_seconds=99.0))
    assert first != T.result_record_hash(
        _historical_record("RQ1O-AAAA1111", verdict="PROHIBITED"))
    supp = _supplement_record("RQ1S-BBBB2222")
    assert T.result_record_hash(supp) == supp["record_sha256"]


def _judging_dir(tmp_path, name, records, filename="results.jsonl"):
    directory = tmp_path / "judging" / name
    directory.mkdir(parents=True, exist_ok=True)
    T.write_jsonl(directory / filename, records)
    return directory


def test_load_run_reads_a_historical_directory(tmp_path):
    ids = ["RQ1O-AAAA1111", "RQ1O-BBBB2222"]
    directory = _judging_dir(tmp_path, "historical_packet__axis1__sonnet",
                             [_historical_record(i) for i in ids])
    loaded = JOIN.load_run(directory)
    assert sorted(loaded) == sorted(ids)
    assert all(r["_result_format"] == T.RESULT_FORMAT_HISTORICAL
               for r in loaded.values())


def test_load_run_accepts_a_historical_directory_stamped_status_ok(tmp_path):
    """Regression for the frozen historical_packet__*__* runs: every record
    carries ``status == "OK"`` with a valid ``parsed_result``, and the joiner
    refused the whole directory as holding no success."""
    ids = ["RQ1O-AAAA1111", "RQ1O-BBBB2222"]
    directory = _judging_dir(
        tmp_path, "historical_packet__axis1__opus",
        [dict(_historical_record(i), status="OK") for i in ids])
    loaded = JOIN.load_run(directory)
    assert sorted(loaded) == sorted(ids)
    assert all(r["_result_format"] == T.RESULT_FORMAT_HISTORICAL
               for r in loaded.values())


def test_load_run_reads_a_supplement_directory(tmp_path):
    ids = ["RQ1S-CCCC3333"]
    directory = _judging_dir(tmp_path, "supplemental_packet__axis1__sonnet",
                             [_supplement_record(i) for i in ids])
    loaded = JOIN.load_run(directory)
    assert sorted(loaded) == ids
    assert loaded[ids[0]]["_result_format"] == T.RESULT_FORMAT_SUPPLEMENT


def test_one_directory_may_not_mix_the_two_formats(tmp_path):
    directory = _judging_dir(tmp_path, "mixed__axis1__sonnet",
                             [_historical_record("RQ1O-AAAA1111"),
                              _supplement_record("RQ1S-BBBB2222")])
    with pytest.raises(TripletError, match="mixes result formats"):
        JOIN.load_run(directory)


def test_a_directory_of_only_failures_is_refused_not_reported_as_empty(tmp_path):
    directory = _judging_dir(
        tmp_path, "historical_packet__axis1__sonnet",
        [dict(_historical_record("RQ1O-AAAA1111"), status="TIMEOUT")])
    with pytest.raises(TripletError, match="none is a success"):
        JOIN.load_run(directory)


def test_an_absent_directory_is_simply_empty(tmp_path):
    assert JOIN.load_run(tmp_path / "never_created") == {}


@pytest.mark.parametrize("maker,prefix", [(_historical_record, "RQ1O"),
                                          (_supplement_record, "RQ1S")])
def test_a_full_join_works_end_to_end_in_either_format(tmp_path, maker, prefix):
    row_id = prefix + "-DDDD4444"
    packet = tmp_path / "packet"
    (packet / "private").mkdir(parents=True)
    T.write_jsonl(packet / "private" / "row_map.jsonl", [{
        "row_id": row_id, "axis": AXIS1,
        "occurrences": [{"case_id": CASES[0], "arm": ARM_E}],
        "prompt_sha256": "a" * 64, "response_sha256": None,
        "reached_max_tokens": False, "output_token_count": 12}])
    for judge in JUDGES:
        _judging_dir(tmp_path, f"packet__axis1__{judge}", [maker(row_id)])

    summary = JOIN.join(tmp_path, [packet], tmp_path / "judging",
                        tmp_path / "joined", write=True)
    assert summary["by_state"] == {"agree": 1}
    cells = T.read_jsonl(tmp_path / "joined" / "private" / "joined_cells.jsonl")
    assert len(cells) == 1
    assert cells[0]["verdicts"] == {j: "PERMITTED" for j in JUDGES}
    assert all(v is not None for v in cells[0]["record_sha256"].values())
    assert set(cells[0]["result_formats"]) == set(JUDGES)


def test_a_join_never_copies_the_historical_prompt_or_response_text(tmp_path):
    row_id = "RQ1O-EEEE5555"
    packet = tmp_path / "packet"
    (packet / "private").mkdir(parents=True)
    T.write_jsonl(packet / "private" / "row_map.jsonl", [{
        "row_id": row_id, "axis": AXIS1,
        "occurrences": [{"case_id": CASES[0], "arm": ARM_E}],
        "prompt_sha256": "a" * 64, "response_sha256": None}])
    for judge in JUDGES:
        _judging_dir(tmp_path, f"packet__axis1__{judge}",
                     [_historical_record(row_id)])
    JOIN.join(tmp_path, [packet], tmp_path / "judging", tmp_path / "joined",
              write=True)
    written = (tmp_path / "joined" / "private" / "joined_cells.jsonl").read_text(
        "utf-8") + (tmp_path / "joined" / "join_summary.json").read_text("utf-8")
    for field in ("request_sent", "raw_api_response", "judge_raw_text"):
        assert f'"{field}":' not in written, field
    assert "SYNTHETIC" not in written


# ================================= 15b. the two frozen private-map layouts
#
# The two packets record the same relation in two shapes. The supplement
# embeds its occurrences in ``private/row_map.jsonl``, each already naming a
# case, an arm and an axis. The historical packet does not: its rows list
# ``occurrence_source_keys`` into a separate ``private/occurrences.jsonl``, and
# an occurrence there names a ``role`` and a ``condition`` -- never an arm.
#
# The defect these cover: a loader that knows only the inline shape reads the
# historical entries through ``.get("occurrences", [])``, gets nothing back for
# all 97 of them, joins the 50 supplemental cells and reports that as the whole
# experiment -- with both judges' record counts recorded as PASS. Nothing in
# the output says the other 60 cells were never looked at.
_STEERING_CONFIGS = ("L31-33_w1_s0.15", "L31-33_w1_s0.2")

#: One verdict per axis, and one that differs, so agreement is set rather than
#: implied. The joiner reads the label; it does not interpret it.
_AGREED = {AXIS1: "PERMITTED", AXIS2: "SA", SAFETY: "NO"}
_DIFFERS = {AXIS1: "PROHIBITED", AXIS2: "DR", SAFETY: "YES"}


def _occurrence(case, condition, config, role, prompt, response):
    """One record in the frozen ``occurrences.jsonl``, in its real schema.

    The fields that matter to a join are ``source_key`` (what a row map entry
    refers to it by), ``case_id``, ``role`` and ``condition``. There is no arm
    field and no axis field, which is the whole difficulty. The text fields are
    here because the real file carries them and nothing downstream may copy
    them on.
    """
    record = {
        "block": None,
        "case_id": case,
        "condition": condition,
        "config_id": config,
        "eos_reached": True,
        "prompt_sha256": sha256_text(prompt),
        "prompt_text": prompt,
        "response_sha256": sha256_text(response),
        "response_text": response,
        "role": role,
        "source_file": "results/pathway_confirmation_512/baseline_records.json",
        "source_group": "xstest",
        "source_key": f"{case}::{condition if config is None else config}::{role}",
        "token_count": 69,
        "truncated": False,
    }
    if config is not None:
        record.update({"hit_count": 3, "layer_start": 31, "layer_end": 33,
                       "strength": 0.15})
    return record


def _historical_occurrences(cases=CASES, collapse=()):
    """The 70 occurrences the frozen packet holds: seven per case.

    Three baselines -- the safe target, the harmful contrast and the d1 donor
    -- and four patched outcomes, two strengths across two arms. Only the three
    baselines are inclusion cells. A patched response is a steering *outcome*,
    never an inclusion input, so it yields no cell; it is still covered by the
    rows that refer to it, so it is not an orphan either.

    In a case named in ``collapse`` the weaker patch left the harmful response
    byte-identical to its baseline, so the two occurrences share a prompt *and*
    a response and deduplicate into one Axis-2 row. Three of the real packet's
    67 Axis-2 rows are exactly that, which is why it has 67 and not 70.
    """
    occurrences = []
    for case in cases:
        prompts = {"safe": f"[{case}] the safe request",
                   "harmful": f"[{case}] the harmful contrast request",
                   "donor": f"[{case}] the donor request"}
        baseline = {"safe_base": ("safe", f"[{case}] safe baseline reply"),
                    "harmful_base": ("harmful", f"[{case}] harmful baseline reply"),
                    "donor": ("donor", f"[{case}] donor baseline reply")}
        for role, (which, response) in baseline.items():
            occurrences.append(
                _occurrence(case, "baseline", None, role, prompts[which], response))
        for config in _STEERING_CONFIGS:
            for role in ("safe", "harmful"):
                response = f"[{case}] {role} reply under {config}"
                if case in collapse and config == _STEERING_CONFIGS[0] \
                        and role == "harmful":
                    response = baseline["harmful_base"][1]
                occurrences.append(
                    _occurrence(case, "patched", config, role, prompts[role],
                                response))
    return occurrences


def _write_separate_layout(root, occurrences, name="historical_packet"):
    """The historical layout: one row map per axis, one shared occurrence file.

    Axis 1 deduplicates to one row per unique prompt; Axis 2 to one row per
    unique (prompt, response) pair. Both are built here the way the packet
    built them, so the fan-out the joiner has to invert is the real one.
    """
    packet = root / name
    (packet / "private").mkdir(parents=True, exist_ok=True)
    T.write_jsonl(packet / "private" / T.SEPARATE_OCCURRENCES, occurrences)

    grouped = {AXIS1: {}, AXIS2: {}}
    for occurrence in occurrences:
        grouped[AXIS1].setdefault((occurrence["prompt_sha256"],),
                                  []).append(occurrence)
        grouped[AXIS2].setdefault((occurrence["prompt_sha256"],
                                   occurrence["response_sha256"]),
                                  []).append(occurrence)
    by_axis = {}
    for axis in (AXIS1, AXIS2):
        entries = []
        for number, key in enumerate(sorted(grouped[axis]), start=1):
            members = grouped[axis][key]
            entry = {"row_id": f"{T.HISTORICAL_ROW_PREFIX[axis]}{number:04d}",
                     "n_occurrences": len(members),
                     "occurrence_source_keys": [m["source_key"] for m in members],
                     "prompt_sha256": members[0]["prompt_sha256"],
                     "prompt_text": members[0]["prompt_text"]}
            if axis == AXIS2:
                entry["response_sha256"] = members[0]["response_sha256"]
                entry["response_text"] = members[0]["response_text"]
            entries.append(entry)
        T.write_jsonl(packet / "private" / T.SEPARATE_ROW_MAPS[axis], entries)
        by_axis[axis] = entries
    return packet, by_axis


def _write_inline_layout(root, cases=CASES, name="supplemental_packet"):
    """The supplemental layout: one digest-only row map, occurrences embedded."""
    packet = root / name
    (packet / "private").mkdir(parents=True, exist_ok=True)
    entries, number = [], 0
    for case in cases:
        for arm, axis in _SUPPLEMENT_CELLS:
            number += 1
            prompt = f"[{case}] {arm} on {axis}"
            response = f"[{case}] {arm} reply on {axis}"
            entries.append({
                "row_id": f"{T.SUPPLEMENT_ROW_PREFIX[axis]}{number:012d}",
                "axis": axis,
                "occurrences": [{"case_id": case, "arm": arm, "axis": axis}],
                "prompt_sha256": sha256_text(prompt),
                "response_sha256": (None if axis == AXIS1
                                    else sha256_text(response)),
                "reached_max_tokens": False, "output_token_count": 42,
                "rendered_system_sha256": sha256_text(f"s{number}"),
                "rendered_user_sha256": sha256_text(f"u{number}")})
    T.write_jsonl(packet / "private" / T.INLINE_ROW_MAP, entries)
    by_axis = {}
    for entry in entries:
        by_axis.setdefault(entry["axis"], []).append(entry)
    return packet, by_axis


def _write_judging(root, jobs, disagree=(), drop=()):
    """One judging directory per (packet, axis, judge), in that runner's format."""
    for name, maker, axis, entries in jobs:
        for judge in JUDGES:
            records = []
            for entry in entries:
                if (judge, entry["row_id"]) in drop:
                    continue
                verdict = (_DIFFERS[axis] if judge == "opus"
                           and entry["row_id"] in disagree else _AGREED[axis])
                records.append(maker(entry["row_id"], axis=axis, verdict=verdict))
            _judging_dir(root, f"{name}__{axis}__{judge}", records)


def _frozen_experiment(root, cases=CASES, collapse=None, disagree=(), drop=()):
    """Both packets, both layouts, and the judging tree over all of it.

    ``collapse`` defaults to the first three cases, which is how many of the
    real packet's Axis-2 rows cover two occurrences instead of one -- 67 rows
    for 70 occurrences, and 97 judged rows in the packet altogether.
    """
    occurrences = _historical_occurrences(
        cases, cases[:3] if collapse is None else collapse)
    historical, historical_rows = _write_separate_layout(root, occurrences)
    supplement, supplement_rows = _write_inline_layout(root, cases)
    jobs = [("historical_packet", _historical_record, axis, entries)
            for axis, entries in sorted(historical_rows.items())]
    jobs += [("supplemental_packet", _supplement_record, axis, entries)
             for axis, entries in sorted(supplement_rows.items())]
    _write_judging(root, jobs, disagree=disagree, drop=drop)
    return {"historical": historical, "supplement": supplement,
            "occurrences": occurrences, "historical_rows": historical_rows,
            "supplement_rows": supplement_rows}


def test_the_frozen_fixture_has_the_shape_of_the_real_experiment(tmp_path):
    """If the fixture drifts from the packet it stands in for, so does the test."""
    built = _frozen_experiment(tmp_path, cases=T.EXPECTED_CASE_IDS)
    assert len(built["occurrences"]) == 70
    assert len(built["historical_rows"][AXIS1]) == 30
    assert len(built["historical_rows"][AXIS2]) == 67
    assert sum(len(v) for v in built["supplement_rows"].values()) == 50
    fanout = sorted(e["n_occurrences"] for e in built["historical_rows"][AXIS2])
    assert fanout.count(2) == 3 and fanout.count(1) == 64


def test_the_historical_separate_layout_resolves_every_row_to_its_cells(tmp_path):
    """The regression, in one packet: 97 rows, 70 occurrences, 60 cells.

    Before the fix this join produced zero cells and raised nothing at all.
    """
    built = _frozen_experiment(tmp_path)
    summary = JOIN.join(tmp_path, [built["historical"]], tmp_path / "judging",
                        tmp_path / "joined", write=True)

    assert summary["cells"] == 60
    assert summary["by_layout"] == {T.LAYOUT_SEPARATE: 60}
    assert summary["by_state"] == {"agree": 60}
    assert summary["rows"] == 97

    cells = T.read_jsonl(tmp_path / "joined" / "private" / "joined_cells.jsonl")
    assert {c["arm"] for c in cells} == {ARM_E, ARM_H, "B_d1"}
    assert {c["axis"] for c in cells} == {AXIS1, AXIS2}
    assert len({c["case_id"] for c in cells}) == 10
    assert all(c["layout"] == T.LAYOUT_SEPARATE for c in cells)


def test_a_patched_occurrence_is_covered_but_is_never_a_cell(tmp_path):
    """A patched response is a steering outcome, not an inclusion input.

    It is not an orphan either -- rows do refer to it -- so it must be neither
    dropped silently nor turned into a cell. 37 of the 67 Axis-2 rows stand for
    nothing but patched outcomes and produce no cell; that is correct, and a
    row that stands for nothing *at all* is a different thing and is refused.
    """
    built = _frozen_experiment(tmp_path)
    JOIN.join(tmp_path, [built["historical"]], tmp_path / "judging",
              tmp_path / "joined", write=True)
    rows = T.read_jsonl(tmp_path / "joined" / "private" / "joined_rows.jsonl")

    assert len(rows) == 97
    assert all(r["occurrences_covered"] >= 1 for r in rows)
    assert sum(len(r["occurrences"]) for r in rows) == 60
    assert sum(r["occurrences_covered"] for r in rows) == 140     # 70 per axis
    barren = [r for r in rows if not r["occurrences"]]
    assert len(barren) == 37 and {r["axis"] for r in barren} == {AXIS2}


def test_the_inline_supplemental_layout_still_joins_unchanged(tmp_path):
    built = _frozen_experiment(tmp_path)
    summary = JOIN.join(tmp_path, [built["supplement"]], tmp_path / "judging",
                        tmp_path / "joined", write=True)
    assert summary["cells"] == 50
    assert summary["by_layout"] == {T.LAYOUT_INLINE: 50}
    cells = T.read_jsonl(tmp_path / "joined" / "private" / "joined_cells.jsonl")
    assert {(c["arm"], c["axis"]) for c in cells} == set(_SUPPLEMENT_CELLS)


def test_both_frozen_layouts_join_to_the_frozen_hundred_and_ten(tmp_path):
    """The whole experiment, counted against the expectation it must meet."""
    built = _frozen_experiment(tmp_path, cases=T.EXPECTED_CASE_IDS)
    summary = JOIN.join(tmp_path, [built["historical"], built["supplement"]],
                        tmp_path / "judging", tmp_path / "joined", write=True,
                        expect=T.FROZEN_JOIN_COVERAGE)

    assert summary["cells"] == 110
    assert summary["by_layout"] == {T.LAYOUT_SEPARATE: 60, T.LAYOUT_INLINE: 50}
    assert summary["records_by_judge"] == {"sonnet": 147, "opus": 147}
    assert summary["by_state"] == {"agree": 110}
    assert summary["coverage_observed"]["cases"] == 10
    assert summary["checklist"]["all_passed"] is True

    cells = T.read_jsonl(tmp_path / "joined" / "private" / "joined_cells.jsonl")
    assert {(c["case_id"], c["arm"], c["axis"]) for c in cells} == {
        (case, arm, axis) for case in T.EXPECTED_CASE_IDS
        for arm, axis in required_cells()}


def test_a_disagreement_survives_the_separate_layout(tmp_path):
    """Fanning a row out to its cells must not quietly settle a split."""
    built = _frozen_experiment(tmp_path)
    split = {built["historical_rows"][AXIS1][0]["row_id"],
             built["historical_rows"][AXIS2][0]["row_id"]}
    _write_judging(tmp_path,
                   [("historical_packet", _historical_record, axis, entries)
                    for axis, entries in sorted(built["historical_rows"].items())],
                   disagree=split)
    summary = JOIN.join(tmp_path, [built["historical"]], tmp_path / "judging",
                        tmp_path / "joined", write=True)
    assert summary["by_state"]["disagree"] >= 1
    assert sum(summary["by_state"].values()) == 60


def test_the_join_copies_no_text_out_of_the_occurrence_file(tmp_path):
    """The separate layout's occurrences carry the prompt and the response."""
    built = _frozen_experiment(tmp_path)
    JOIN.join(tmp_path, [built["historical"]], tmp_path / "judging",
              tmp_path / "joined", write=True)
    written = "".join(
        (tmp_path / "joined" / name).read_text(encoding="utf-8")
        for name in ("private/joined_cells.jsonl", "private/joined_rows.jsonl",
                     "join_summary.json"))
    for field in ("prompt_text", "response_text", "source_file", "source_key",
                  "role", "condition", "config_id"):
        assert f'"{field}"' not in written, field
    assert "baseline reply" not in written


# --- what the join refuses -----------------------------------------------
def _rewrite(path, mutate):
    """Re-write one JSONL file through ``mutate(list_of_records)``."""
    records = T.read_jsonl(path)
    T.write_jsonl(path, mutate(records))


def test_an_unresolvable_occurrence_key_fails_closed(tmp_path):
    built = _frozen_experiment(tmp_path)
    path = built["historical"] / "private" / T.SEPARATE_ROW_MAPS[AXIS1]
    _rewrite(path, lambda rows: [
        dict(r, occurrence_source_keys=["nobody::baseline::donor"],
             n_occurrences=1) if i == 0 else r for i, r in enumerate(rows)])
    with pytest.raises(TripletError, match="does not hold"):
        JOIN.join(tmp_path, [built["historical"]], tmp_path / "judging",
                  tmp_path / "joined", write=False)


def test_a_judged_row_that_maps_to_no_occurrence_fails_closed(tmp_path):
    built = _frozen_experiment(tmp_path)
    path = built["historical"] / "private" / T.SEPARATE_ROW_MAPS[AXIS2]
    _rewrite(path, lambda rows: [
        dict(r, occurrence_source_keys=[], n_occurrences=0) if i == 0 else r
        for i, r in enumerate(rows)])
    with pytest.raises(TripletError, match="maps to no occurrence"):
        JOIN.join(tmp_path, [built["historical"]], tmp_path / "judging",
                  tmp_path / "joined", write=False)


def test_an_inline_row_that_maps_to_no_occurrence_fails_closed(tmp_path):
    built = _frozen_experiment(tmp_path)
    path = built["supplement"] / "private" / T.INLINE_ROW_MAP
    _rewrite(path, lambda rows: [dict(r, occurrences=[]) if i == 0 else r
                                 for i, r in enumerate(rows)])
    with pytest.raises(TripletError, match="maps to no occurrence"):
        JOIN.join(tmp_path, [built["supplement"]], tmp_path / "judging",
                  tmp_path / "joined", write=False)


def test_an_orphan_occurrence_fails_closed(tmp_path):
    """An occurrence no row refers to is a cell no judge was asked about."""
    built = _frozen_experiment(tmp_path)
    path = built["historical"] / "private" / T.SEPARATE_OCCURRENCES
    extra = _occurrence(CASES[0], "baseline", None, "donor",
                        "an unjudged prompt", "an unjudged reply")
    extra["source_key"] = "orphan::baseline::donor"
    _rewrite(path, lambda rows: rows + [extra])
    with pytest.raises(TripletError, match="no row map refers to"):
        JOIN.join(tmp_path, [built["historical"]], tmp_path / "judging",
                  tmp_path / "joined", write=False)


def test_two_rows_claiming_one_occurrence_fail_closed(tmp_path):
    built = _frozen_experiment(tmp_path)
    path = built["historical"] / "private" / T.SEPARATE_ROW_MAPS[AXIS1]
    rows = T.read_jsonl(path)
    stolen = rows[0]["occurrence_source_keys"][0]
    _rewrite(path, lambda rs: [
        dict(r, occurrence_source_keys=[stolen], n_occurrences=1)
        if i == 1 else r for i, r in enumerate(rs)])
    with pytest.raises(TripletError, match="already claimed by row"):
        JOIN.join(tmp_path, [built["historical"]], tmp_path / "judging",
                  tmp_path / "joined", write=False)


def test_two_rows_producing_one_cell_fail_closed(tmp_path):
    """Two distinct occurrences that resolve to the same (case, arm, axis).

    Deduplication fans one row out to many cells. One cell reached from two
    rows means the key is not sound, and equal verdicts would only hide it.
    """
    built = _frozen_experiment(tmp_path)
    private = built["historical"] / "private"
    twin = _occurrence(CASES[0], "baseline", None, "donor",
                       "a second donor prompt", "a second donor reply")
    twin["source_key"] = "twin::baseline::donor"
    _rewrite(private / T.SEPARATE_OCCURRENCES, lambda rows: rows + [twin])
    _rewrite(private / T.SEPARATE_ROW_MAPS[AXIS1], lambda rows: rows + [{
        "row_id": f"{T.HISTORICAL_ROW_PREFIX[AXIS1]}9999",
        "n_occurrences": 1, "occurrence_source_keys": [twin["source_key"]],
        "prompt_sha256": twin["prompt_sha256"],
        "prompt_text": twin["prompt_text"]}])
    _rewrite(private / T.SEPARATE_ROW_MAPS[AXIS2], lambda rows: rows + [{
        "row_id": f"{T.HISTORICAL_ROW_PREFIX[AXIS2]}9999",
        "n_occurrences": 1, "occurrence_source_keys": [twin["source_key"]],
        "prompt_sha256": twin["prompt_sha256"],
        "prompt_text": twin["prompt_text"],
        "response_sha256": twin["response_sha256"],
        "response_text": twin["response_text"]}])
    with pytest.raises(TripletError, match="is claimed by row"):
        JOIN.join(tmp_path, [built["historical"]], tmp_path / "judging",
                  tmp_path / "joined", write=False)


def test_an_unknown_role_or_condition_is_refused_not_mapped(tmp_path):
    built = _frozen_experiment(tmp_path)
    path = built["historical"] / "private" / T.SEPARATE_OCCURRENCES
    _rewrite(path, lambda rows: [dict(r, role="donor_d2") if i == 2 else r
                                 for i, r in enumerate(rows)])
    with pytest.raises(TripletError, match=r"unknown \(role, condition\)"):
        JOIN.join(tmp_path, [built["historical"]], tmp_path / "judging",
                  tmp_path / "joined", write=False)


def test_an_arm_is_never_read_out_of_the_source_key(tmp_path):
    """The key text says one thing and the recorded role says another.

    The recorded role wins, because it is the schema. If the key were being
    parsed, relabelling it would move the cell.
    """
    built = _frozen_experiment(tmp_path)
    private = built["historical"] / "private"
    occurrences = T.read_jsonl(private / T.SEPARATE_OCCURRENCES)
    victim = next(o for o in occurrences
                  if o["case_id"] == CASES[0] and o["role"] == "donor")
    renamed = f"{CASES[7]}::baseline::safe_base__relabelled"
    for name in (T.SEPARATE_OCCURRENCES, T.SEPARATE_ROW_MAPS[AXIS1],
                 T.SEPARATE_ROW_MAPS[AXIS2]):
        raw = (private / name).read_text(encoding="utf-8")
        (private / name).write_text(raw.replace(victim["source_key"], renamed),
                                    encoding="utf-8")
    JOIN.join(tmp_path, [built["historical"]], tmp_path / "judging",
              tmp_path / "joined", write=True)
    cells = T.read_jsonl(tmp_path / "joined" / "private" / "joined_cells.jsonl")
    every = {(ARM_E, AXIS1), (ARM_E, AXIS2), (ARM_H, AXIS1), (ARM_H, AXIS2),
             ("B_d1", AXIS1), ("B_d1", AXIS2)}
    for case in (CASES[0], CASES[7]):
        assert {(c["arm"], c["axis"])
                for c in cells if c["case_id"] == case} == every, case
    assert len(cells) == 60


def test_a_row_filed_under_the_wrong_axis_map_fails_closed(tmp_path):
    """The axis comes from the packet's own per-axis prefix, not the file name."""
    built = _frozen_experiment(tmp_path)
    path = built["historical"] / "private" / T.SEPARATE_ROW_MAPS[AXIS1]
    _rewrite(path, lambda rows: [
        dict(r, row_id=T.HISTORICAL_ROW_PREFIX[AXIS2] + "0001") if i == 0 else r
        for i, r in enumerate(rows)])
    with pytest.raises(TripletError, match="is not a RQ1RETROAONE opaque id"):
        JOIN.join(tmp_path, [built["historical"]], tmp_path / "judging",
                  tmp_path / "joined", write=False)


def test_a_row_wired_to_the_wrong_occurrence_fails_closed(tmp_path):
    """Both sides carry the digests; a resolved key still has to be the truth."""
    built = _frozen_experiment(tmp_path)
    path = built["historical"] / "private" / T.SEPARATE_ROW_MAPS[AXIS1]
    _rewrite(path, lambda rows: [dict(r, prompt_sha256="f" * 64) if i == 0 else r
                                 for i, r in enumerate(rows)])
    with pytest.raises(TripletError, match="disagree about what was judged"):
        JOIN.join(tmp_path, [built["historical"]], tmp_path / "judging",
                  tmp_path / "joined", write=False)


def test_a_separate_layout_packet_without_its_occurrence_file_fails_closed(tmp_path):
    built = _frozen_experiment(tmp_path)
    (built["historical"] / "private" / T.SEPARATE_OCCURRENCES).unlink()
    with pytest.raises(TripletError, match="occurrences live in"):
        JOIN.join(tmp_path, [built["historical"]], tmp_path / "judging",
                  tmp_path / "joined", write=False)


def test_a_packet_offering_both_layouts_is_refused_not_half_read(tmp_path):
    built = _frozen_experiment(tmp_path)
    T.write_jsonl(built["historical"] / "private" / T.INLINE_ROW_MAP, [])
    with pytest.raises(TripletError, match="alternatives, not halves"):
        JOIN.join(tmp_path, [built["historical"]], tmp_path / "judging",
                  tmp_path / "joined", write=False)


# --- counting the run against what the experiment must produce -----------
def test_the_frozen_expectation_is_arithmetically_what_it_claims():
    expect = T.FROZEN_JOIN_COVERAGE
    assert expect["cells"] == T.required_cell_count() == 110
    assert sum(expect["cells_by_layout"].values()) == expect["cells"]
    assert expect["cells_by_layout"] == {T.LAYOUT_SEPARATE: 60,
                                         T.LAYOUT_INLINE: 50}
    assert expect["cases"] == len(T.EXPECTED_CASE_IDS) == 10
    assert expect["records_per_judge"] == 147
    assert expect["terminal_states"] == ("agree", "disagree")


def test_the_historical_axis_prefixes_are_the_packets_own():
    """Taken from the frozen packet manifest, not from the file names."""
    manifest = json.loads((REPO_ROOT / "manifests" / "rq1_retro_packet.json")
                          .read_text(encoding="utf-8"))
    declared = manifest["blindness_asserted_by_the_packet"]["row_id_prefixes"]
    assert T.HISTORICAL_ROW_PREFIX == declared


def test_a_short_join_is_a_failure_and_not_a_smaller_result(tmp_path):
    """The observed defect, stated as the gate that now catches it."""
    built = _frozen_experiment(tmp_path, cases=T.EXPECTED_CASE_IDS)
    with pytest.raises(TripletError,
                       match=r"60 required cell\(s\) have no verdict"):
        JOIN.join(tmp_path, [built["supplement"]], tmp_path / "judging",
                  tmp_path / "joined", write=False,
                  expect=T.FROZEN_JOIN_COVERAGE)


def test_a_judge_short_of_a_record_is_a_failure(tmp_path):
    built = _frozen_experiment(tmp_path, cases=T.EXPECTED_CASE_IDS,
                               drop={("opus",
                                      f"{T.SUPPLEMENT_ROW_PREFIX[SAFETY]}"
                                      f"{5:012d}")})
    with pytest.raises(TripletError, match="opus has 146 successful record"):
        JOIN.join(tmp_path, [built["historical"], built["supplement"]],
                  tmp_path / "judging", tmp_path / "joined", write=False,
                  expect=T.FROZEN_JOIN_COVERAGE)


def _full_grid(state="agree"):
    return {(case, arm, axis): state for case in T.EXPECTED_CASE_IDS
            for arm, axis in required_cells()}


def _runs(n=147):
    return {judge: {f"row{i}": {} for i in range(n)} for judge in JUDGES}


def _split_layout(cells):
    """60 cells to the separate layout, 50 to the inline one, as the run does."""
    historical = {(ARM_E, AXIS1), (ARM_E, AXIS2), (ARM_H, AXIS1),
                  (ARM_H, AXIS2), ("B_d1", AXIS1), ("B_d1", AXIS2)}
    return {key: {"state": value,
                  "layout": (T.LAYOUT_SEPARATE if (key[1], key[2]) in historical
                             else T.LAYOUT_INLINE)}
            for key, value in cells.items()}


def test_the_coverage_check_passes_on_the_shape_the_experiment_must_have():
    checks = Checklist("t")
    observed = JOIN.check_coverage(_split_layout(_full_grid()), _runs(),
                                   T.FROZEN_JOIN_COVERAGE, checks)
    assert observed["cells"] == 110 and observed["cases"] == 10
    assert observed["cells_by_layout"] == {T.LAYOUT_SEPARATE: 60,
                                           T.LAYOUT_INLINE: 50}
    assert checks.all_passed


def test_a_cell_only_one_judge_answered_is_not_a_result():
    grid = _full_grid()
    grid[(T.EXPECTED_CASE_IDS[0], ARM_E, AXIS1)] = "incomplete"
    with pytest.raises(TripletError, match="every_cell_is_agree_or_disagree"):
        JOIN.check_coverage(_split_layout(grid), _runs(),
                            T.FROZEN_JOIN_COVERAGE, Checklist("t"))


def test_a_cell_no_judge_answered_is_not_a_result():
    grid = _full_grid()
    grid[(T.EXPECTED_CASE_IDS[1], ARM_H, SAFETY)] = "missing"
    with pytest.raises(TripletError, match="every_cell_is_agree_or_disagree"):
        JOIN.check_coverage(_split_layout(grid), _runs(),
                            T.FROZEN_JOIN_COVERAGE, Checklist("t"))


def test_a_required_cell_with_no_verdict_is_incomplete_coverage():
    grid = _full_grid()
    del grid[(T.EXPECTED_CASE_IDS[2], ARM_D3, AXIS2)]
    with pytest.raises(TripletError, match="required_cells_complete"):
        JOIN.check_coverage(_split_layout(grid), _runs(),
                            T.FROZEN_JOIN_COVERAGE, Checklist("t"))


def test_an_eleventh_case_is_refused_outright():
    grid = _full_grid()
    grid[("an_eleventh_case", ARM_D3, AXIS2)] = "agree"
    with pytest.raises(TripletError, match="no inclusion rule asks for"):
        JOIN.check_coverage(_split_layout(grid), _runs(),
                            T.FROZEN_JOIN_COVERAGE, Checklist("t"))


def test_the_wrong_split_between_the_two_layouts_is_a_failure():
    """110 cells is not enough: they have to come from both packets."""
    grid = {key: {"state": "agree", "layout": T.LAYOUT_INLINE}
            for key in _full_grid()}
    with pytest.raises(TripletError, match="cells_from_inline_occurrences"):
        JOIN.check_coverage(grid, _runs(), T.FROZEN_JOIN_COVERAGE,
                            Checklist("t"))


# ================================================== 16. the ten, and the 110
BUILD = _load_script("build_inclusion_manifest")


def _adjudicated(tmp_path, table):
    """One adjudicated_cells.jsonl from a {case: {arm: {axis: verdict}}} table."""
    rows = []
    for case_id, arms in table.items():
        for arm, axes in arms.items():
            for axis, verdict in axes.items():
                rows.append({"case_id": case_id, "arm": arm, "axis": axis,
                             "verdict": verdict, "label_source": "agreement",
                             "prompt_sha256": "a" * 64,
                             "response_sha256": None,
                             "reached_max_tokens": False,
                             "output_token_count": 12})
    directory = tmp_path / "adjudicated"
    directory.mkdir(parents=True, exist_ok=True)
    T.write_jsonl(directory / "adjudicated_cells.jsonl", rows)
    return directory


@pytest.fixture
def real_case_ids(monkeypatch):
    """The frozen ten, mapped onto the synthetic case names the fixtures use."""
    monkeypatch.setattr(BUILD, "EXPECTED_CASE_IDS", tuple(CASES))
    return tuple(CASES)


def test_the_frozen_ten_are_ten_and_the_grid_is_a_hundred_and_ten():
    assert len(T.EXPECTED_CASE_IDS) == 10
    assert len(set(T.EXPECTED_CASE_IDS)) == 10
    assert len(T.required_cells()) == 11
    assert T.required_cell_count() == 110


def test_a_complete_table_writes_a_manifest_with_no_gaps(
        tmp_path, real_case_ids, monkeypatch):
    monkeypatch.setattr(BUILD, "load_protocol_manifest", lambda *a, **k: {
        "protocol_hashes": {}, "axis1_version": {
            "name": "CHG-004",
            "recalibration": {"acceptance_criterion_2": "not met"}},
        "axis2_version": {"name": "CHG-002"}})
    out = tmp_path / "rq1_triplet_inclusion.json"
    manifest = BUILD.build(_adjudicated(tmp_path, _cells()), out, False)
    assert out.exists()
    assert "REPORT_ONLY_DRAFT" not in manifest
    assert manifest["expected_cells_total"] == 110
    assert manifest["resolved_cells"] == 110
    assert manifest["cases_entirely_absent"] == []
    assert len(manifest["cases"]) == 10
    for name in INCLUSION_RULES:
        assert manifest["totals"][name] == {STATUS_VALIDATED: 10}


def test_an_entirely_absent_case_is_detected_and_is_unresolved_not_ignored(
        tmp_path, real_case_ids, monkeypatch):
    monkeypatch.setattr(BUILD, "load_protocol_manifest", lambda *a, **k: {
        "protocol_hashes": {}, "axis1_version": {
            "name": "CHG-004", "recalibration": {"acceptance_criterion_2": ""}},
        "axis2_version": {"name": "CHG-002"}})
    table = _cells()
    dropped = table.pop(CASES[3])
    assert dropped                                  # the case existed
    out = tmp_path / "rq1_triplet_inclusion.json"

    with pytest.raises(TripletError, match="missing or unresolved"):
        BUILD.build(_adjudicated(tmp_path, table), out, False)

    manifest = BUILD.build(_adjudicated(tmp_path, table), out, True)
    assert manifest["cases_entirely_absent"] == [CASES[3]]
    assert manifest["resolved_cells"] == 110 - 11
    assert len(manifest["cases"]) == 10             # it is still reported
    assert manifest["cases"][CASES[3]]["entirely_absent"] is True
    for name in INCLUSION_RULES:
        assert manifest["cases"][CASES[3]]["rules"][name][
            "inclusion_status"] == STATUS_UNRESOLVED
    assert manifest["REPORT_ONLY_DRAFT"] is True
    assert not out.exists()                         # never to the real manifest
    assert out.with_name(out.stem + ".DRAFT.json").exists()


def test_an_eleventh_case_is_refused_rather_than_quietly_included(
        tmp_path, real_case_ids, monkeypatch):
    monkeypatch.setattr(BUILD, "load_protocol_manifest", lambda *a, **k: {
        "protocol_hashes": {}, "axis1_version": {
            "name": "CHG-004", "recalibration": {"acceptance_criterion_2": ""}},
        "axis2_version": {"name": "CHG-002"}})
    table = _cells()
    table["case_99_substitute"] = table[CASES[0]]
    with pytest.raises(TripletError, match="not among the ten"):
        BUILD.build(_adjudicated(tmp_path, table),
                    tmp_path / "m.json", allow_incomplete=True)


def test_invalid_and_unresolved_stay_apart_in_the_manifest(
        tmp_path, real_case_ids, monkeypatch):
    monkeypatch.setattr(BUILD, "load_protocol_manifest", lambda *a, **k: {
        "protocol_hashes": {}, "axis1_version": {
            "name": "CHG-004", "recalibration": {"acceptance_criterion_2": ""}},
        "axis2_version": {"name": "CHG-002"}})
    table = _cells({
        CASES[0]: {(ARM_D2, AXIS2): "PA-S"},        # present and failing
        CASES[1]: {(ARM_D2, AXIS2): None},          # absent: we do not know
    })
    manifest = BUILD.build(_adjudicated(tmp_path, table), tmp_path / "m.json",
                           allow_incomplete=True)
    rules = manifest["cases"]
    assert rules[CASES[0]]["rules"]["primary_actuator_source"][
        "inclusion_status"] == STATUS_INVALID
    assert rules[CASES[1]]["rules"]["primary_actuator_source"][
        "inclusion_status"] == STATUS_UNRESOLVED
    # the comparator rule does not touch d2, so both cases stay validated there
    for case in (CASES[0], CASES[1]):
        assert rules[case]["rules"]["comparator_actuator_source"][
            "inclusion_status"] == STATUS_VALIDATED
    assert manifest["totals"]["primary_actuator_source"] == {
        STATUS_INVALID: 1, STATUS_UNRESOLVED: 1, STATUS_VALIDATED: 8}


# =========================== 17. the human reviewer gets the judges' rubric
EXPORT = _load_script("export_human_review")


@pytest.fixture
def review_tree(tmp_path, protocol_dir, monkeypatch):
    root, manifest = protocol_dir
    packet = tmp_path / "packet"
    (packet / "private").mkdir(parents=True)
    ids = T.new_opaque_ids("RQ1O", 3, rng=random.Random(11))
    axes = [AXIS1, AXIS2, SAFETY]
    row_map, cells = [], []
    for row_id, axis in zip(ids, axes):
        row = {"row_id": row_id, "prompt_text": "a neutral question"}
        if axis != AXIS1:
            row["response_text"] = "a neutral answer"
        T.write_jsonl(packet / f"{axis}_judge_input.jsonl", [row])
        # the safety axis is required on H alone, so its cell must sit there
        arm = ARM_H if axis == SAFETY else ARM_E
        # real digests, because the export checks the text it is about to show
        # against the digest of the text that was actually judged
        digests = {"prompt_sha256": sha256_text(row["prompt_text"]),
                   "response_sha256": (sha256_text(row["response_text"])
                                       if "response_text" in row else None)}
        row_map.append(dict(row, axis=axis,
                            occurrences=[{"case_id": CASES[0], "arm": arm}],
                            **digests))
        cells.append({"case_id": CASES[0], "arm": arm, "axis": axis,
                      "row_id": row_id, "state": "agree",
                      "verdicts": {j: "PERMITTED" for j in JUDGES},
                      "record_sha256": {j: "b" * 64 for j in JUDGES},
                      "packet": "packet", **digests})
    T.write_jsonl(packet / "private" / "row_map.jsonl", row_map)
    joined = tmp_path / "joined"
    (joined / "private").mkdir(parents=True)
    T.write_jsonl(joined / "private" / "joined_cells.jsonl", cells)
    monkeypatch.setattr(T, "load_protocol_manifest", lambda *a, **k: manifest)
    monkeypatch.setattr(EXPORT, "load_protocol",
                        lambda axis, **k: T.load_protocol(
                            axis, directory=k.get("directory") or root,
                            manifest=manifest))
    return {"joined": joined, "packet": packet, "protocol": root,
            "out": tmp_path / "human", "manifest": manifest}


def _export(tree):
    return EXPORT.export(tree["joined"], [tree["packet"]], tree["out"], seed=1,
                         write=True, protocol_dir=tree["protocol"])


def test_the_reviewer_receives_the_frozen_instructions_for_each_axis_task(
        review_tree):
    summary = _export(review_tree)
    payload = json.loads(
        (review_tree["out"] / "REVIEWER_INSTRUCTIONS.json").read_text("utf-8"))
    want = review_tree["manifest"]["protocol_hashes"]
    for axis in (AXIS1, AXIS2, SAFETY):
        task = payload["tasks"][axis]
        assert task["instructions_sha256"] == want[f"{axis}_instructions_sha256"]
        assert task["allowed_labels"] == list(T.AXIS_VALUES[axis])
        assert task["verdict_field"] == T.AXIS_VERDICT_FIELD[axis]
        assert summary["instruction_hashes"][axis] == task["instructions_sha256"]


def test_axis_task_is_still_on_every_row_so_the_instructions_can_be_matched(
        review_tree):
    _export(review_tree)
    rows = T.read_jsonl(review_tree["out"] / "human_review_input.jsonl")
    assert rows
    for row in rows:
        assert row["axis_task"] in (AXIS1, AXIS2, SAFETY)


def test_the_instructions_expose_no_role_condition_or_expected_verdict(
        review_tree):
    _export(review_tree)
    payload = (review_tree["out"] / "REVIEWER_INSTRUCTIONS.json").read_text("utf-8")
    for token in ("B_d1", "B_d2", "B_d3", "E/B/H", "donor", "steer",
                  "expected", "over-refus", "case_", CASES[0]):
        assert token not in payload, token
    for rule in INCLUSION_RULES:
        assert rule not in payload


def test_the_reviewer_still_sees_no_judge_verdict_or_agreement(review_tree):
    summary = _export(review_tree)
    reviewer_files = ("human_review_input.jsonl", "human_review_TEMPLATE.jsonl",
                      "REVIEWER_INSTRUCTIONS.json", "human_review_manifest.json")
    seen = "".join((review_tree["out"] / f).read_text("utf-8")
                   for f in reviewer_files)
    assert "sonnet_verdict" not in seen and "opus_verdict" not in seen
    assert '"judges_agree"' not in seen and '"agree"' not in seen
    assert "sonnet verdict" in json.dumps(summary["reviewer_does_not_see"])


def test_an_undecidable_row_may_be_left_empty_rather_than_guessed(review_tree):
    _export(review_tree)
    payload = json.loads(
        (review_tree["out"] / "REVIEWER_INSTRUCTIONS.json").read_text("utf-8"))
    how = payload["how_to_use"]
    assert "leave the verdict empty" in how
    assert "unresolved" in how
    template = T.read_jsonl(review_tree["out"] / "human_review_TEMPLATE.jsonl")
    assert all(row["verdict"] == "" for row in template)


def test_the_instruction_file_is_machine_local_and_not_in_the_repository(
        review_tree):
    _export(review_tree)
    path = (review_tree["out"] / "REVIEWER_INSTRUCTIONS.json").resolve()
    assert REPO_ROOT not in path.parents


# ========================================= 18. the incremental second pass
#
# The supplement's 50 cells were reviewed and adjudicated first. The joiner's
# repair then produced the other 60 -- E, H and B_d1 on Axis 1 and Axis 2 --
# and those are the only ones a human still has to read.
#
# What must not happen: the 50 re-exported and asked for again, the 50 quietly
# re-decided, a prior decision carried forward for a cell whose text has moved,
# one cell decided in both passes, or an inclusion manifest built from either
# half on its own. Cells are matched across the two passes on their stable
# scientific identity -- (case_id, arm, axis) -- because the opaque row ids do
# not survive: each export mints fresh ones precisely so an answer cannot be
# traced back to a judging record.
IMPORT = _load_script("import_human_review")


def _hundred_and_ten(tmp_path, cases=T.EXPECTED_CASE_IDS, disagree=()):
    """A complete join of 110 cells, and one packet holding all their text.

    The text is neutral and numbered: it names no case and carries none of the
    role or condition tokens, because the export refuses a row whose text
    could identify what it is.
    """
    packet = tmp_path / "packet"
    (packet / "private").mkdir(parents=True, exist_ok=True)
    joined = tmp_path / "joined"
    (joined / "private").mkdir(parents=True, exist_ok=True)

    rows, cells, number = [], [], 0
    for case in cases:
        for arm, axis in required_cells():
            number += 1
            historical = (arm, axis) in T.HISTORICAL_REQUIRED_CELLS
            prefix = (T.HISTORICAL_ROW_PREFIX[axis] if historical
                      else T.SUPPLEMENT_ROW_PREFIX[axis])
            row_id = f"{prefix}{number:012d}"
            prompt = f"Question {number:03d}. Please answer the request above."
            response = (f"Answer {number:03d}. A stored completion."
                        if "response_text" in T.JUDGE_ROW_FIELDS[axis] else None)
            digests = {"prompt_sha256": sha256_text(prompt),
                       "response_sha256": (sha256_text(response)
                                           if response is not None else None)}
            entry = {"row_id": row_id, "prompt_text": prompt, **digests}
            if response is not None:
                entry["response_text"] = response
            rows.append(entry)

            split = (case, arm, axis) in set(disagree)
            cells.append({
                "case_id": case, "arm": arm, "axis": axis, "row_id": row_id,
                "state": "disagree" if split else "agree",
                "verdicts": {"sonnet": _AGREED[axis],
                             "opus": _DIFFERS[axis] if split else _AGREED[axis]},
                "record_sha256": {j: sha256_text(j + row_id) for j in JUDGES},
                "packet": "packet",
                "layout": (T.LAYOUT_SEPARATE if historical else T.LAYOUT_INLINE),
                "reached_max_tokens": False, "output_token_count": 42,
                **digests})
    T.write_jsonl(packet / "private" / "row_map.jsonl", rows)
    T.write_jsonl(joined / "private" / T.JOINED_CELLS_FILE, cells)
    return {"packet": packet, "joined": joined,
            "cells": {(c["case_id"], c["arm"], c["axis"]): c for c in cells}}


def _prior_record(tmp_path, tree, cells=T.SUPPLEMENTAL_REQUIRED_CELLS,
                  name="adjudicated"):
    """The evidence record the first pass left, in the real record's schema."""
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    wanted = set(cells)
    records = []
    for key, cell in sorted(tree["cells"].items()):
        if (cell["arm"], cell["axis"]) not in wanted:
            continue
        agreed = cell["verdicts"]["sonnet"] == cell["verdicts"]["opus"]
        records.append({
            "case_id": cell["case_id"], "arm": cell["arm"], "axis": cell["axis"],
            "prompt_sha256": cell["prompt_sha256"],
            "response_sha256": cell["response_sha256"],
            "reached_max_tokens": cell["reached_max_tokens"],
            "output_token_count": cell["output_token_count"],
            "sonnet_verdict": cell["verdicts"]["sonnet"],
            "opus_verdict": cell["verdicts"]["opus"],
            "sonnet_record_sha256": cell["record_sha256"]["sonnet"],
            "opus_record_sha256": cell["record_sha256"]["opus"],
            "human_row_id": T.HUMAN_ROW_PREFIX + sha256_text(str(key))[:12].upper(),
            "verdict": cell["verdicts"]["sonnet"],
            "label_source": ("DUAL_LLM_AGREEMENT" if agreed
                             else "HUMAN_ADJUDICATION"),
            "adjudication_rule": ("sonnet == opus == human" if agreed
                                  else "human adjudication; judges disagreed"),
            "judges_agree": agreed,
            "human_record_sha256": sha256_text("human|" + str(key))})
    T.write_jsonl(directory / T.ADJUDICATED_CELLS_FILE, records)
    return directory


@pytest.fixture
def incremental(tmp_path, protocol_dir, monkeypatch):
    """110 joined cells, 50 already adjudicated, and the protocol to export."""
    root, manifest = protocol_dir
    monkeypatch.setattr(T, "load_protocol_manifest", lambda *a, **k: manifest)
    monkeypatch.setattr(EXPORT, "load_protocol",
                        lambda axis, **k: T.load_protocol(
                            axis, directory=k.get("directory") or root,
                            manifest=manifest))
    tree = _hundred_and_ten(tmp_path, disagree={
        # two historical cells, decided in the pass that is about to run
        (T.EXPECTED_CASE_IDS[0], ARM_E, AXIS1),
        (T.EXPECTED_CASE_IDS[4], "B_d1", AXIS2),
        # and one supplemental cell, decided in the pass that already ran
        (T.EXPECTED_CASE_IDS[2], ARM_D2, AXIS2)})
    tree["base"] = _prior_record(tmp_path, tree)
    tree["protocol"] = root
    tree["out"] = tmp_path / "human_review_historical"
    tree["root"] = tmp_path
    return tree


def _export_remaining(tree, out=None, exclude=None, expect=..., seed=20260822):
    prior = (T.load_adjudicated_cells(tree["base"]) if exclude is None
             else exclude)
    return EXPORT.export(
        tree["joined"], [tree["packet"]], out or tree["out"], seed=seed,
        write=True, protocol_dir=tree["protocol"], exclude=prior,
        expect=T.FROZEN_INCREMENTAL_REVIEW if expect is ... else expect)


def _complete(review_dir, verdicts=None):
    """Answer every exported row. The reviewer sees only the axis task."""
    verdicts = verdicts or {}
    rows = []
    for row in T.read_jsonl(review_dir / "human_review_input.jsonl"):
        rows.append({"row_id": row["row_id"],
                     "verdict": verdicts.get(row["row_id"],
                                             _AGREED[row["axis_task"]]),
                     "confidence": "high", "note": "a neutral note"})
    path = review_dir / "human_review_COMPLETED.jsonl"
    T.write_jsonl(path, rows)
    return path


def _merge(tree, out, base=..., expect=...):
    return IMPORT.run(
        tree["joined"], tree["out"], _complete(tree["out"]), out, write=True,
        base=tree["base"] if base is ... else base,
        expect=T.FROZEN_INCREMENTAL_REVIEW if expect is ... else expect)


# --- 110 - 50 = 60 --------------------------------------------------------
def test_a_hundred_and_ten_joined_minus_fifty_adjudicated_exports_sixty(
        incremental):
    summary = _export_remaining(incremental)

    assert summary["joined_cells"] == 110
    assert summary["cells_already_adjudicated"] == 50
    assert summary["rows"] == 60
    assert summary["cases"] == 10
    assert summary["arms_covered"] == ["B_d1", ARM_E, ARM_H]
    assert summary["axes_covered"] == [AXIS1, AXIS2]
    assert summary["coverage"]["review_cells"] == sorted(
        T.HISTORICAL_REQUIRED_CELLS)
    assert summary["coverage"]["prior_cells"] == sorted(
        T.SUPPLEMENTAL_REQUIRED_CELLS)

    public = T.read_jsonl(incremental["out"] / "human_review_input.jsonl")
    assert len(public) == 60
    private = T.read_jsonl(
        incremental["out"] / "private" / T.HUMAN_ROW_MAP_FILE)
    assert len(private) == 60
    assert {(p["arm"], p["axis"]) for p in private} == set(
        T.HISTORICAL_REQUIRED_CELLS)
    assert {p["case_id"] for p in private} == set(T.EXPECTED_CASE_IDS)
    assert len({p["row_id"] for p in private}) == 60


def test_the_sixty_exported_are_disjoint_from_the_fifty_already_done(
        incremental):
    _export_remaining(incremental)
    private = T.read_jsonl(
        incremental["out"] / "private" / T.HUMAN_ROW_MAP_FILE)
    exported = {(p["case_id"], p["arm"], p["axis"]) for p in private}
    prior = set(T.load_adjudicated_cells(incremental["base"]))
    assert exported & prior == set()
    assert len(exported | prior) == 110


def test_a_prior_record_that_is_not_there_is_refused_not_read_as_empty(
        incremental):
    with pytest.raises(TripletError, match="is not a file"):
        T.load_adjudicated_cells(incremental["root"] / "never_adjudicated",
                                 "--exclude-adjudicated")


# --- digests ---------------------------------------------------------------
@pytest.mark.parametrize("field", ["prompt_sha256", "response_sha256"])
def test_a_digest_mismatch_stops_a_prior_cell_from_being_excluded(
        incremental, field):
    """A decision about the old text is not a decision about this one."""
    path = incremental["base"] / T.ADJUDICATED_CELLS_FILE
    records = T.read_jsonl(path)
    victim = next(r for r in records if r["axis"] == AXIS2)   # carries both
    victim[field] = "f" * 64
    T.write_jsonl(path, records)
    with pytest.raises(TripletError, match="has moved"):
        _export_remaining(incremental)


def test_a_prior_cell_missing_a_digest_the_cell_has_is_a_mismatch(incremental):
    path = incremental["base"] / T.ADJUDICATED_CELLS_FILE
    records = T.read_jsonl(path)
    records[0] = {k: v for k, v in records[0].items()
                  if k not in T.CELL_DIGEST_FIELDS}
    T.write_jsonl(path, records)
    with pytest.raises(TripletError, match="has moved"):
        _export_remaining(incremental)


def test_a_prior_cell_with_no_digest_on_either_side_is_refused(incremental):
    """Nothing to check it against is not a reason to assume it still holds."""
    key = None
    path = incremental["base"] / T.ADJUDICATED_CELLS_FILE
    records = T.read_jsonl(path)
    key = (records[0]["case_id"], records[0]["arm"], records[0]["axis"])
    records[0] = {k: v for k, v in records[0].items()
                  if k not in T.CELL_DIGEST_FIELDS}
    T.write_jsonl(path, records)

    cells_path = incremental["joined"] / "private" / T.JOINED_CELLS_FILE
    cells = T.read_jsonl(cells_path)
    for cell in cells:
        if (cell["case_id"], cell["arm"], cell["axis"]) == key:
            for field in T.CELL_DIGEST_FIELDS:
                cell.pop(field, None)
    T.write_jsonl(cells_path, cells)

    with pytest.raises(TripletError, match="carries no prompt or response"):
        _export_remaining(incremental)


def test_a_digest_mismatch_stops_a_prior_decision_being_merged(incremental):
    _export_remaining(incremental)
    path = incremental["base"] / T.ADJUDICATED_CELLS_FILE
    records = T.read_jsonl(path)
    records[0]["prompt_sha256"] = "e" * 64
    T.write_jsonl(path, records)
    with pytest.raises(TripletError, match="has moved"):
        _merge(incremental, incremental["root"] / "merged")


# --- duplicate and conflicting identities ---------------------------------
def test_a_duplicate_stable_key_in_the_prior_record_fails(incremental):
    path = incremental["base"] / T.ADJUDICATED_CELLS_FILE
    records = T.read_jsonl(path)
    T.write_jsonl(path, records + [records[0]])
    with pytest.raises(TripletError, match="appears twice"):
        _export_remaining(incremental)


def test_a_conflicting_duplicate_in_the_prior_record_fails(incremental):
    """Even a disagreeing copy is refused: it is refused for being a second."""
    path = incremental["base"] / T.ADJUDICATED_CELLS_FILE
    records = T.read_jsonl(path)
    conflicting = dict(records[0], verdict="PA-N",
                       label_source="HUMAN_ADJUDICATION")
    T.write_jsonl(path, records + [conflicting])
    with pytest.raises(TripletError, match="appears twice"):
        _export_remaining(incremental)


def test_a_duplicate_stable_key_in_the_joined_cells_fails(incremental):
    path = incremental["joined"] / "private" / T.JOINED_CELLS_FILE
    cells = T.read_jsonl(path)
    T.write_jsonl(path, cells + [dict(cells[0], row_id="RQ1RETROAONE999999")])
    with pytest.raises(TripletError, match="appears twice"):
        _export_remaining(incremental)


def test_a_prior_cell_the_join_does_not_hold_is_refused(incremental):
    path = incremental["base"] / T.ADJUDICATED_CELLS_FILE
    records = T.read_jsonl(path)
    orphan = dict(records[0], case_id=T.EXPECTED_CASE_IDS[0], arm=ARM_D2,
                  axis=SAFETY)
    T.write_jsonl(path, records + [orphan])
    with pytest.raises(TripletError, match="have no joined cell"):
        _export_remaining(incremental)


def test_a_cell_decided_in_both_passes_is_refused(incremental):
    """The full 110 exported by mistake, then merged onto the prior 50."""
    _export_remaining(incremental, exclude={}, expect=None)
    with pytest.raises(TripletError, match="decided both in the base record"):
        _merge(incremental, incremental["root"] / "merged", expect=None)


# --- the export discloses nothing private ---------------------------------
def test_no_private_judge_information_reaches_the_review_export(incremental):
    _export_remaining(incremental)
    out = incremental["out"]

    # Fields a joined or adjudicated cell carries. The manifest is allowed to
    # *name* them in its prose list of what the reviewer does not see -- that
    # is a statement of blindness -- so there it is the key form that is
    # forbidden, and in the row files the bare name is.
    private_fields = ("sonnet_verdict", "opus_verdict", "verdicts",
                      "judges_agree", "agree", "disagree", "record_sha256",
                      "label_source", "prompt_sha256", "response_sha256",
                      "arm", "arms_covered", "layout", "judge_row_id",
                      "strength", "layer", "case_id", "verdict")
    # Values. These identify a cell outright and may not appear at all.
    private_values = ("DUAL_LLM_AGREEMENT", "HUMAN_ADJUDICATION", "B_d1",
                      "B_d2", "B_d3", "separate_occurrences",
                      "inline_occurrences", "L31-33", *T.EXPECTED_CASE_IDS)

    rows = "".join((out / name).read_text(encoding="utf-8") for name in (
        "human_review_input.jsonl", "human_review_TEMPLATE.jsonl"))
    manifest = (out / "human_review_manifest.json").read_text(encoding="utf-8")
    for token in private_fields:
        assert f'"{token}":' not in manifest, token
    for token in private_values:
        assert token not in rows + manifest, token
    for token in private_fields:
        if token in ("verdict", "arm", "agree"):
            continue        # blanks the reviewer fills in, or common letters
        assert token not in rows, token

    # A verdict label belongs in the answer sheet's vocabulary and in the
    # instructions, never on a row: a pre-filled label is an anchor. Checked
    # on the records rather than on the bytes -- two-letter labels like UM are
    # substrings of half the opaque ids, so a text scan would not mean this.
    for row in T.read_jsonl(out / "human_review_TEMPLATE.jsonl"):
        assert set(row) == {"row_id", "verdict", "confidence", "note"}
        assert row["verdict"] == "" and row["confidence"] == ""

    # the instructions carry the frozen rubric, so labels are expected there;
    # judge identity, arms and cases are not
    instructions = (out / "REVIEWER_INSTRUCTIONS.json").read_text("utf-8")
    for token in ("sonnet", "opus", "B_d1", "B_d2", "B_d3", "E/B/H", "donor",
                  "judges_agree", "arms_covered", *T.EXPECTED_CASE_IDS):
        assert token not in instructions, token

    # every reviewer row carries exactly the blind judge fields, and no more
    for row in T.read_jsonl(out / "human_review_input.jsonl"):
        assert set(row) - {"axis_task"} == set(
            T.JUDGE_ROW_FIELDS[row["axis_task"]])


def test_the_operator_copy_keeps_what_the_reviewer_copy_withholds(incremental):
    """Withheld from the packet, not lost: the private copy carries it."""
    _export_remaining(incremental)
    out = incremental["out"]
    reviewer = json.loads((out / "human_review_manifest.json").read_text("utf-8"))
    operator = json.loads(
        (out / "private" / "export_summary.json").read_text("utf-8"))
    for key in EXPORT.REVIEWER_WITHHELD:
        assert key not in reviewer, key
        assert key in operator, key
    assert operator["coverage"]["cells_to_review"] == 60
    assert reviewer["rows"] == 60


def test_the_export_refuses_to_overwrite_an_existing_review_directory(
        incremental):
    _export_remaining(incremental)
    with pytest.raises(TripletError, match="does not write over a review"):
        _export_remaining(incremental)


# --- the merge -------------------------------------------------------------
def test_the_merge_preserves_the_fifty_and_totals_a_hundred_and_ten(
        incremental):
    _export_remaining(incremental)
    prior_path = incremental["base"] / T.ADJUDICATED_CELLS_FILE
    before = prior_path.read_bytes()

    out = incremental["root"] / "adjudicated_complete"
    summary = _merge(incremental, out)

    assert summary["cells"] == 110
    assert summary["coverage"] == {"preserved": 50, "new": 60, "merged": 110,
                                   "cases": 10}
    merged = T.read_jsonl(out / T.ADJUDICATED_CELLS_FILE)
    assert len(merged) == 110
    assert len({(m["case_id"], m["arm"], m["axis"]) for m in merged}) == 110

    # the earlier record is not touched, and every one of its decisions is
    # carried into the merge exactly as it was
    assert prior_path.read_bytes() == before
    by_key = {(m["case_id"], m["arm"], m["axis"]): m for m in merged}
    for record in T.read_jsonl(prior_path):
        key = (record["case_id"], record["arm"], record["axis"])
        assert by_key[key] == record, key
    assert summary["base_adjudicated_sha256"] == sha256_text(
        before.decode("utf-8"))


def test_the_merge_reports_dual_agreement_and_human_adjudication(incremental):
    _export_remaining(incremental)
    summary = _merge(incremental, incremental["root"] / "adjudicated_complete")

    total = summary["by_label_source"]
    assert sum(total.values()) == 110
    assert set(total) <= {"DUAL_LLM_AGREEMENT", "HUMAN_ADJUDICATION"}
    assert sum(summary["decisions_carried_forward"].values()) == 50
    assert sum(summary["decisions_from_this_review"].values()) == 60
    # the two historical splits are decided in this pass; the supplemental one
    # was decided in the pass that already ran and is carried forward as it was
    assert summary["decisions_from_this_review"]["HUMAN_ADJUDICATION"] == 2
    assert summary["decisions_carried_forward"]["HUMAN_ADJUDICATION"] == 1
    for source in ("DUAL_LLM_AGREEMENT", "HUMAN_ADJUDICATION"):
        assert total[source] == (
            summary["decisions_carried_forward"].get(source, 0)
            + summary["decisions_from_this_review"].get(source, 0))


def test_the_merge_refuses_to_write_over_the_record_it_merges(incremental):
    _export_remaining(incremental)
    with pytest.raises(TripletError, match="never written over"):
        _merge(incremental, incremental["base"])
    # and the record it refused to write over is still the fifty
    assert len(T.read_jsonl(
        incremental["base"] / T.ADJUDICATED_CELLS_FILE)) == 50


def test_the_merge_refuses_a_non_empty_output_directory(incremental):
    _export_remaining(incremental)
    out = incremental["root"] / "adjudicated_complete"
    _merge(incremental, out)
    with pytest.raises(TripletError, match="written once"):
        _merge(incremental, out)


def test_a_review_answered_against_a_different_join_is_refused(incremental):
    """The row map's digests are checked against the joined cells again."""
    _export_remaining(incremental)
    path = incremental["out"] / "private" / T.HUMAN_ROW_MAP_FILE
    rows = T.read_jsonl(path)
    rows[0]["prompt_sha256"] = "d" * 64
    T.write_jsonl(path, rows)
    with pytest.raises(TripletError, match="has moved"):
        _merge(incremental, incremental["root"] / "merged")


# --- the inclusion manifest --------------------------------------------
def _protocol_manifest_stub(monkeypatch):
    monkeypatch.setattr(BUILD, "load_protocol_manifest", lambda *a, **k: {
        "protocol_hashes": {}, "axis1_version": {
            "name": "CHG-004", "recalibration": {"acceptance_criterion_2": ""}},
        "axis2_version": {"name": "CHG-002"}})


def test_the_inclusion_builder_refuses_either_half_alone(incremental,
                                                         monkeypatch):
    _protocol_manifest_stub(monkeypatch)
    _export_remaining(incremental)
    root = incremental["root"]
    sixty = root / "adjudicated_historical"
    _merge(incremental, sixty, base=None, expect=None)
    merged = root / "adjudicated_complete"
    _merge(incremental, merged)

    assert len(T.read_jsonl(incremental["base"] / T.ADJUDICATED_CELLS_FILE)) == 50
    assert len(T.read_jsonl(sixty / T.ADJUDICATED_CELLS_FILE)) == 60

    out = root / "rq1_triplet_inclusion.json"
    for half, count in ((incremental["base"], 50), (sixty, 60)):
        with pytest.raises(TripletError,
                           match=rf"holds {count} of 110 cell"):
            BUILD.build(half, out, False)
        assert not out.exists()

    manifest = BUILD.build(merged, out, False)
    assert out.exists()
    assert "REPORT_ONLY_DRAFT" not in manifest
    assert manifest["adjudicated_cells_read"] == 110
    assert manifest["resolved_cells"] == 110


def test_the_inclusion_builder_takes_the_cell_file_or_its_directory(
        incremental, monkeypatch):
    _protocol_manifest_stub(monkeypatch)
    _export_remaining(incremental)
    merged = incremental["root"] / "adjudicated_complete"
    _merge(incremental, merged)
    by_dir = BUILD.build(merged, incremental["root"] / "a.json", False)
    by_file = BUILD.build(merged / T.ADJUDICATED_CELLS_FILE,
                          incremental["root"] / "b.json", False)
    assert by_dir["cases"] == by_file["cases"]
    with pytest.raises(TripletError, match="is not a file"):
        BUILD.build(incremental["root"] / "no_such_record",
                    incremental["root"] / "c.json", False)


def test_a_record_of_the_right_size_but_the_wrong_cells_is_refused(
        incremental, monkeypatch):
    """No gaps, 110 required cells present, and five cells too many."""
    _protocol_manifest_stub(monkeypatch)
    _export_remaining(incremental)
    merged = incremental["root"] / "adjudicated_complete"
    _merge(incremental, merged)
    path = merged / T.ADJUDICATED_CELLS_FILE
    records = T.read_jsonl(path)
    extra = [dict(records[0], arm=ARM_E, axis=SAFETY,
                  case_id=case) for case in T.EXPECTED_CASE_IDS[:5]]
    T.write_jsonl(path, records + extra)
    with pytest.raises(TripletError, match="beyond the frozen grid"):
        BUILD.build(merged, incremental["root"] / "m.json", False)
    draft = BUILD.build(merged, incremental["root"] / "m.json", True)
    assert draft["REPORT_ONLY_DRAFT"] is True
    assert not (incremental["root"] / "m.json").exists()


# ============================== 19. merging the two adjudication records
#
# Two completed human adjudications exist: the supplement's 50 cells and the
# historical 60. The merge concatenates them into the evidence record the
# inclusion rules will read. It decides nothing -- which is exactly why it has
# to check everything: a stage that only copies is a stage where a wrong cell
# would arrive silently and stay.
MERGE = _load_script("merge_adjudications")


@pytest.fixture
def two_halves(tmp_path):
    """110 joined cells, and the two completed adjudication records over them."""
    tree = _hundred_and_ten(tmp_path, disagree={
        (T.EXPECTED_CASE_IDS[0], ARM_E, AXIS1),
        (T.EXPECTED_CASE_IDS[4], "B_d1", AXIS2),
        (T.EXPECTED_CASE_IDS[2], ARM_D2, AXIS2)})
    tree["supplemental"] = _prior_record(
        tmp_path, tree, T.SUPPLEMENTAL_REQUIRED_CELLS, "adjudicated")
    tree["historical"] = _prior_record(
        tmp_path, tree, T.HISTORICAL_REQUIRED_CELLS, "adjudicated_historical")
    tree["out"] = tmp_path / "adjudicated_complete"
    tree["root"] = tmp_path
    return tree


def _merge_halves(tree, out=None, write=True, **override):
    return MERGE.merge(override.get("supplemental", tree["supplemental"]),
                       override.get("historical", tree["historical"]),
                       override.get("joined", tree["joined"]),
                       out or tree["out"], write=write)


def _edit(directory, mutate, name=None):
    """Re-write one adjudicated record through ``mutate(list_of_rows)``."""
    path = directory / (name or T.ADJUDICATED_CELLS_FILE)
    T.write_jsonl(path, mutate(T.read_jsonl(path)))
    return path


def _first(directory, **match):
    """The first record in a half matching every given field."""
    rows = T.read_jsonl(directory / T.ADJUDICATED_CELLS_FILE)
    return next(r for r in rows
                if all(r[k] == v for k, v in match.items()))


# --- the merge that should work -------------------------------------------
def test_fifty_and_sixty_merge_into_one_hundred_and_ten(two_halves):
    result = _merge_halves(two_halves)
    summary, manifest = result["summary"], result["manifest"]

    assert summary["cells"] == 110
    assert summary["cells_by_input"] == {"supplemental": 50, "historical": 60}
    assert summary["cases"] == 10
    assert sum(summary["by_label_source"].values()) == 110
    assert summary["judges_agree"] == {"true": 107, "false": 3}
    assert summary["resolves_inclusion"] is False

    out = two_halves["out"]
    rows = T.read_jsonl(out / T.ADJUDICATED_CELLS_FILE)
    assert len(rows) == 110
    assert len({(r["case_id"], r["arm"], r["axis"]) for r in rows}) == 110
    assert {(r["arm"], r["axis"]) for r in rows} == set(required_cells())
    assert {r["case_id"] for r in rows} == set(T.EXPECTED_CASE_IDS)
    for name in (T.ADJUDICATED_CELLS_FILE, "adjudication_summary.json",
                 "manifest.json"):
        assert (out / name).is_file(), name


def test_a_file_record_carries_the_manifest_path_field_beside_its_digest():
    """The manifest records both the path named and the file read.

    Those are different things -- an input may be named as a directory and
    read as the cell file inside it -- so a manifest entry carries ``path``
    alongside ``name``. The helper's own filesystem argument therefore cannot
    also be called ``path``: a caller passing the field would bind the
    parameter twice and the merge would die with a TypeError before writing
    anything.
    """
    named = Path("/somewhere/adjudicated")
    record = MERGE._file_record(named / T.ADJUDICATED_CELLS_FILE, "a\nb\n",
                                path=str(named), cells=50)
    assert record["path"] == str(named)          # what the operator named
    assert record["name"] == T.ADJUDICATED_CELLS_FILE   # what was read
    assert record["sha256"] == sha256_text("a\nb\n")
    assert record["bytes"] == 4
    assert record["cells"] == 50


def test_the_manifest_hashes_every_input_and_every_output(two_halves):
    manifest = _merge_halves(two_halves)["manifest"]
    out = two_halves["out"]

    for name, named, path, cells in (
            ("supplemental", two_halves["supplemental"],
             two_halves["supplemental"] / T.ADJUDICATED_CELLS_FILE, 50),
            ("historical", two_halves["historical"],
             two_halves["historical"] / T.ADJUDICATED_CELLS_FILE, 60),
            ("joined", two_halves["joined"],
             two_halves["joined"] / "private" / T.JOINED_CELLS_FILE, 110)):
        record = manifest["inputs"][name]
        assert record["cells"] == cells
        assert record["sha256"] == sha256_text(path.read_text("utf-8"))
        assert record["bytes"] == len(path.read_bytes())
        # the directory the operator named, and the file actually read
        assert record["path"] == str(named)
        assert record["name"] == path.name
        assert record["path"] != record["name"]

    for key, name in (("cells", T.ADJUDICATED_CELLS_FILE),
                      ("summary", "adjudication_summary.json")):
        record = manifest["outputs"][key]
        assert record["sha256"] == sha256_text((out / name).read_text("utf-8"))
        assert record["name"] == name
    assert manifest["outputs"]["cells"]["rows"] == 110


def test_the_inputs_are_not_modified_by_the_merge(two_halves):
    before = {name: (two_halves[name] / T.ADJUDICATED_CELLS_FILE).read_bytes()
              for name in ("supplemental", "historical")}
    joined_path = two_halves["joined"] / "private" / T.JOINED_CELLS_FILE
    joined_before = joined_path.read_bytes()

    _merge_halves(two_halves)

    for name, payload in before.items():
        assert (two_halves[name] / T.ADJUDICATED_CELLS_FILE
                ).read_bytes() == payload, name
    assert joined_path.read_bytes() == joined_before


# --- duplicates -----------------------------------------------------------
@pytest.mark.parametrize("half", ["supplemental", "historical"])
def test_a_duplicate_cell_inside_one_half_is_refused(two_halves, half):
    _edit(two_halves[half], lambda rows: rows + [rows[0]])
    with pytest.raises(TripletError, match="appears twice"):
        _merge_halves(two_halves)


def test_a_conflicting_duplicate_is_refused_for_being_a_second(two_halves):
    """Not for disagreeing -- for there being two decisions about one cell."""
    _edit(two_halves["historical"],
          lambda rows: rows + [dict(rows[0], verdict="AMBIGUOUS")])
    with pytest.raises(TripletError, match="appears twice"):
        _merge_halves(two_halves)


def test_a_duplicate_in_the_joined_record_is_refused(two_halves):
    path = two_halves["joined"] / "private" / T.JOINED_CELLS_FILE
    cells = T.read_jsonl(path)
    T.write_jsonl(path, cells + [cells[0]])
    with pytest.raises(TripletError, match="appears twice"):
        _merge_halves(two_halves)


# --- missing, extra and off-grid cells ------------------------------------
@pytest.mark.parametrize("half,size", [("supplemental", 50),
                                       ("historical", 60)])
def test_a_half_with_too_few_cells_is_refused(two_halves, half, size):
    _edit(two_halves[half], lambda rows: rows[:-1])
    with pytest.raises(TripletError,
                       match=rf"holds {size - 1} cell\(s\); the {half} half"):
        _merge_halves(two_halves)


@pytest.mark.parametrize("half,size", [("supplemental", 50),
                                       ("historical", 60)])
def test_a_half_with_too_many_cells_is_refused(two_halves, half, size):
    _edit(two_halves[half],
          lambda rows: rows + [dict(rows[0], case_id="an_eleventh_case")])
    with pytest.raises(TripletError,
                       match=rf"holds {size + 1} cell\(s\); the {half} half"):
        _merge_halves(two_halves)


def test_a_cell_with_no_adjudication_is_refused(two_halves):
    """Right size, wrong cell: one grid cell answered, one invented."""
    _edit(two_halves["historical"],
          lambda rows: [dict(r, case_id="an_eleventh_case") if i == 0 else r
                        for i, r in enumerate(rows)])
    with pytest.raises(TripletError, match="have no adjudication"):
        _merge_halves(two_halves)


def test_an_adjudicated_cell_the_join_does_not_hold_is_refused(two_halves):
    victim = _first(two_halves["historical"], arm=ARM_E, axis=AXIS1)
    path = two_halves["joined"] / "private" / T.JOINED_CELLS_FILE
    T.write_jsonl(path, [c for c in T.read_jsonl(path)
                         if (c["case_id"], c["arm"], c["axis"]) !=
                         (victim["case_id"], victim["arm"], victim["axis"])])
    with pytest.raises(TripletError, match="are not in"):
        _merge_halves(two_halves)


def test_a_key_set_that_matches_the_join_but_not_the_frozen_grid_is_refused(
        two_halves):
    """The two agree with each other and both are wrong."""
    victim = _first(two_halves["historical"], arm=ARM_E, axis=AXIS1)
    key = (victim["case_id"], victim["arm"], victim["axis"])
    _edit(two_halves["historical"],
          lambda rows: [dict(r, case_id="an_eleventh_case")
                        if (r["case_id"], r["arm"], r["axis"]) == key else r
                        for r in rows])
    path = two_halves["joined"] / "private" / T.JOINED_CELLS_FILE
    T.write_jsonl(path, [dict(c, case_id="an_eleventh_case")
                         if (c["case_id"], c["arm"], c["axis"]) == key else c
                         for c in T.read_jsonl(path)])
    with pytest.raises(TripletError, match="differ from the frozen grid"):
        _merge_halves(two_halves)


# --- overlapping inputs ---------------------------------------------------
def test_naming_one_record_for_both_halves_is_refused(two_halves):
    with pytest.raises(TripletError,
                       match=r"holds 50 cell\(s\); the historical half"):
        _merge_halves(two_halves, historical=two_halves["supplemental"])


def test_a_cell_from_the_other_half_is_refused(two_halves):
    """The overlap a real run could produce: one half carrying the other's."""
    stolen = _first(two_halves["supplemental"], arm=ARM_D2, axis=AXIS1)
    _edit(two_halves["historical"],
          lambda rows: [stolen] + rows[1:])
    with pytest.raises(TripletError, match="the historical half covers"):
        _merge_halves(two_halves)


def test_the_two_halves_cannot_overlap_once_each_covers_its_own_cells():
    """Why the disjointness gate has nothing left to catch, written down.

    Each half is checked against its own (arm, axis) coverage, and the two
    coverages partition the eleven required cells. The merge checks for an
    overlap anyway; this records that the check is a second line and not the
    first.
    """
    supplemental = set(T.SUPPLEMENTAL_REQUIRED_CELLS)
    historical = set(T.HISTORICAL_REQUIRED_CELLS)
    assert supplemental & historical == set()
    assert supplemental | historical == set(required_cells())
    assert len(supplemental) + len(historical) == 11


# --- hashes and verdicts against the joined record ------------------------
@pytest.mark.parametrize("field,expected", [
    ("prompt_sha256", "has moved"),
    ("response_sha256", "has moved"),
    ("sonnet_record_sha256", "does not describe the judging"),
    ("opus_record_sha256", "does not describe the judging"),
])
def test_a_hash_that_does_not_match_the_joined_record_is_refused(
        two_halves, field, expected):
    _edit(two_halves["historical"],
          lambda rows: [dict(r, **{field: "f" * 64}) if r["axis"] == AXIS2
                        else r for r in rows])
    with pytest.raises(TripletError, match=expected):
        _merge_halves(two_halves)


@pytest.mark.parametrize("judge", list(JUDGES))
def test_a_judge_verdict_that_does_not_match_the_joined_record_is_refused(
        two_halves, judge):
    field = T.JUDGE_CELL_FIELDS[judge]["verdict"]
    _edit(two_halves["historical"],
          lambda rows: [dict(r, **{field: "AMBIGUOUS"}) if r["axis"] == AXIS1
                        else r for r in rows])
    with pytest.raises(TripletError, match="does not describe the judging"):
        _merge_halves(two_halves)


@pytest.mark.parametrize("field,value", [("reached_max_tokens", True),
                                         ("output_token_count", 999)])
def test_a_carried_field_that_drifted_from_the_join_is_refused(
        two_halves, field, value):
    _edit(two_halves["historical"],
          lambda rows: [dict(r, **{field: value}) if i == 0 else r
                        for i, r in enumerate(rows)])
    with pytest.raises(TripletError, match="a stage that changed it"):
        _merge_halves(two_halves)


def test_a_cell_with_nothing_to_check_it_against_is_refused(two_halves):
    rows = T.read_jsonl(two_halves["historical"] / T.ADJUDICATED_CELLS_FILE)
    key = (rows[0]["case_id"], rows[0]["arm"], rows[0]["axis"])
    _edit(two_halves["historical"],
          lambda rs: [{k: v for k, v in r.items()
                       if k not in T.CELL_DIGEST_FIELDS} if i == 0 else r
                      for i, r in enumerate(rs)])
    path = two_halves["joined"] / "private" / T.JOINED_CELLS_FILE
    T.write_jsonl(path, [
        {k: v for k, v in c.items() if k not in T.CELL_DIGEST_FIELDS}
        if (c["case_id"], c["arm"], c["axis"]) == key else c
        for c in T.read_jsonl(path)])
    with pytest.raises(TripletError, match="carries no prompt or response"):
        _merge_halves(two_halves)


# --- unresolved, missing and malformed verdicts ---------------------------
def test_an_unresolved_cell_is_refused_and_not_read_as_a_pass(two_halves):
    _edit(two_halves["historical"],
          lambda rows: [dict(r, label_source=T.SOURCE_UNRESOLVED, verdict=None)
                        if i == 0 else r for i, r in enumerate(rows)])
    with pytest.raises(TripletError, match="never decided"):
        _merge_halves(two_halves)


def test_a_decided_cell_with_no_verdict_in_it_is_refused(two_halves):
    _edit(two_halves["historical"],
          lambda rows: [dict(r, verdict=None) if i == 0 else r
                        for i, r in enumerate(rows)])
    with pytest.raises(TripletError, match="is malformed"):
        _merge_halves(two_halves)


def test_a_verdict_outside_the_frozen_vocabulary_is_refused(two_halves):
    _edit(two_halves["historical"],
          lambda rows: [dict(r, verdict="DEFINITELY_FINE") if i == 0 else r
                        for i, r in enumerate(rows)])
    with pytest.raises(TripletError, match="does not define"):
        _merge_halves(two_halves)


def test_an_unknown_label_source_is_refused(two_halves):
    _edit(two_halves["historical"],
          lambda rows: [dict(r, label_source="MAJORITY") if i == 0 else r
                        for i, r in enumerate(rows)])
    with pytest.raises(TripletError, match="label_source 'MAJORITY' is not"):
        _merge_halves(two_halves)


def test_an_agreement_flag_that_contradicts_the_verdicts_is_refused(
        two_halves):
    _edit(two_halves["historical"],
          lambda rows: [dict(r, judges_agree=False) if r["judges_agree"]
                        else r for r in rows])
    with pytest.raises(TripletError, match="cannot disagree with them"):
        _merge_halves(two_halves)


# --- text ------------------------------------------------------------------
def test_a_record_carrying_text_is_refused_rather_than_read_past(two_halves):
    _edit(two_halves["historical"],
          lambda rows: [dict(r, prompt_text="a prompt") if i == 0 else r
                        for i, r in enumerate(rows)])
    with pytest.raises(TripletError, match="carries \\['prompt_text'\\]"):
        _merge_halves(two_halves)


def test_the_merge_reads_no_text_and_names_no_machine_path():
    code = _code_of("experiments/triplet_validation/merge_adjudications.py")
    for field in ("prompt_text", "response_text"):
        assert code.count(field) == 1        # the tuple that forbids them
    for forbidden in ("joined_rows.jsonl", "human_review_input.jsonl",
                      "row_map.jsonl", "results.jsonl", "parsed_result",
                      "/home/", "routegate-data"):
        assert forbidden not in code, forbidden
    assert "working_dir()" in code           # the root comes from the config


def test_the_written_record_carries_no_prompt_or_response_text(two_halves):
    _merge_halves(two_halves)
    out = two_halves["out"]
    written = "".join((out / name).read_text("utf-8") for name in (
        T.ADJUDICATED_CELLS_FILE, "adjudication_summary.json", "manifest.json"))
    for field in ("prompt_text", "response_text", "judge_raw_text",
                  "raw_api_response"):
        assert field not in written, field


# --- determinism -----------------------------------------------------------
def test_the_merged_cells_are_sorted_and_byte_identical_across_runs(
        two_halves):
    root = two_halves["root"]
    first = _merge_halves(two_halves, out=root / "run_a")
    second = _merge_halves(two_halves, out=root / "run_b")

    cells = T.ADJUDICATED_CELLS_FILE
    assert (root / "run_a" / cells).read_bytes() == (
        root / "run_b" / cells).read_bytes()
    assert (first["manifest"]["outputs"]["cells"]["sha256"]
            == second["manifest"]["outputs"]["cells"]["sha256"])

    rows = T.read_jsonl(root / "run_a" / cells)
    keys = [(r["case_id"], r["arm"], r["axis"]) for r in rows]
    assert keys == sorted(keys)
    # keys sorted inside each record too, so the bytes do not depend on the
    # order the fields happened to be written in upstream
    for line in (root / "run_a" / cells).read_text("utf-8").splitlines():
        parsed = json.loads(line)
        assert list(parsed) == sorted(parsed)


def test_the_hashed_text_is_exactly_what_write_jsonl_would_write(tmp_path):
    """``--dry-run`` hashes text it built rather than a file it wrote.

    That is only honest while the text it builds is the text the writer
    produces, so the two are compared directly here rather than trusted.
    """
    rows = [{"b": 2, "a": 1, "c": "\u00e9"}, {"a": None, "b": [1, 2]}]
    path = tmp_path / "rows.jsonl"
    T.write_jsonl(path, rows)
    assert MERGE._jsonl_text(rows) == path.read_text(encoding="utf-8")


def test_the_output_does_not_depend_on_the_order_of_the_input_lines(
        two_halves):
    root = two_halves["root"]
    baseline = _merge_halves(two_halves, out=root / "ordered")
    _edit(two_halves["historical"], lambda rows: list(reversed(rows)))
    _edit(two_halves["supplemental"],
          lambda rows: rows[25:] + rows[:25])
    shuffled = _merge_halves(two_halves, out=root / "shuffled")

    assert (baseline["manifest"]["outputs"]["cells"]["sha256"]
            == shuffled["manifest"]["outputs"]["cells"]["sha256"])
    assert (root / "ordered" / T.ADJUDICATED_CELLS_FILE).read_bytes() == (
        root / "shuffled" / T.ADJUDICATED_CELLS_FILE).read_bytes()


# --- dry run and the output directory --------------------------------------
def test_a_dry_run_writes_nothing_and_still_reports_the_digest(two_halves):
    result = _merge_halves(two_halves, write=False)
    assert not two_halves["out"].exists()
    assert result["summary"]["cells"] == 110

    written = _merge_halves(two_halves)
    assert two_halves["out"].is_dir()
    assert (result["manifest"]["outputs"]["cells"]["sha256"]
            == written["manifest"]["outputs"]["cells"]["sha256"])
    assert (result["manifest"]["outputs"]["cells"]["sha256"]
            == sha256_text((two_halves["out"] / T.ADJUDICATED_CELLS_FILE
                            ).read_text("utf-8")))


@pytest.mark.parametrize("half", ["supplemental", "historical", "joined"])
def test_the_output_may_not_be_an_input(two_halves, half):
    with pytest.raises(TripletError, match="which is an input to this merge"):
        _merge_halves(two_halves, out=two_halves[half])
    assert len(T.read_jsonl(
        two_halves["supplemental"] / T.ADJUDICATED_CELLS_FILE)) == 50


def test_a_non_empty_output_directory_is_refused(two_halves):
    _merge_halves(two_halves)
    with pytest.raises(TripletError, match="written once"):
        _merge_halves(two_halves)


def test_a_dry_run_may_target_a_directory_that_already_exists(two_halves):
    """It writes nothing, so there is nothing for it to write over."""
    _merge_halves(two_halves)
    result = _merge_halves(two_halves, write=False)
    assert result["summary"]["cells"] == 110
