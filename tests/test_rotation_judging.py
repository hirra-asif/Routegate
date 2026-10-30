"""Registering the completed rotation screen with the common judging engine.

The screen is finished and pinned; what is under test here is the small amount
of code that lets the COMMON engine judge it, and the reading that comes back
out. Six things, which are the six the task names:

* the **counts** -- 144 judged cells, 36 inherited, 180 in the analysis table,
  288 billed calls and not 360;
* the **hashes** -- the consolidated screen, the frozen rubric text (under both
  digest conventions in this repository), the phase-1 adjudication and the
  adapter's own output;
* **blindness** -- a judge row carries the whitelist and nothing that names an
  arm, a beta, a class, an expected outcome or a corpus row;
* the **inherited references** -- read from phase 1, never re-judged, and
  refused if anything about them has moved;
* **technical failures** -- UNRESOLVED_TECHNICAL is terminal, is never a
  verdict, and takes its beta to UNDETERMINED with it;
* **exact reuse of the authoritative decision function** -- the report calls
  ``rotationfreeze.evaluate_rotation_decision`` and holds no threshold, veto,
  tie-break or undetermined branch of its own.

Everything that can run on synthetic material does. The tests that need the
private data root are marked and skipped where it is absent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate import judgingstages as S
from routegate import phase2judging as P2
from routegate import rotationjudging as RJ
from routegate import rotationfreeze as RF
from routegate import triplets as T
from routegate.judgingprotocol import (
    SINGLE_ROW_FIELDS,
    JudgingProtocolError,
    load_manifest,
)
from routegate.rotationscreen import (
    ARMS_BY_BETA,
    BETA_GRID,
    NEW_GENERATIONS,
    N_ROWS,
    REFERENCE_ARM_IDS,
    REFERENCE_RECORDS,
    SELECTED_ROWS,
)
from routegate.validation import Checklist, ValidationError

PATHS_PATH = REPO_ROOT / "configs" / "paths.local.json"
MANIFEST_PATH = RJ.JUDGING_MANIFEST


def _local_paths():
    if not PATHS_PATH.exists():
        return {}
    try:
        return json.loads(PATHS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _consolidated():
    root = _local_paths().get(RJ.PATHS_KEY_CONSOLIDATED)
    if not root:
        return None
    directory = Path(root)
    return (directory if (directory / RJ.CONSOLIDATED_RECORDS).is_file()
            else None)


def _rubric_dir():
    root = _local_paths().get("triplet_validation_dir")
    if not root:
        return None
    directory = Path(root) / "frozen" / "rq1_judge_protocol"
    return directory if directory.is_dir() else None


def _adapted():
    root = _local_paths().get(RJ.PATHS_KEY_JUDGING)
    if not root:
        return None
    path = Path(root) / RJ.INPUTS_SUBDIR / RJ.CANDIDATE_RECORDS
    return path if path.is_file() else None


needs_consolidated = pytest.mark.skipif(
    _consolidated() is None,
    reason="the consolidated rotation screen is not reachable")
needs_rubric = pytest.mark.skipif(
    _rubric_dir() is None,
    reason="the frozen safety rubric text is not staged on this machine")
needs_adapted = pytest.mark.skipif(
    _adapted() is None,
    reason="the adapted candidate records have not been written here")


# ================================================================= fixtures
def prompt_for(row_id):
    """Synthetic prompt text that does NOT contain the corpus row id.

    The engine refuses a judge row whose text names a corpus row, so a fixture
    that spelled the row id into its prompt would fail that gate for the wrong
    reason. Indexing by position keeps the fixture blind the way a real prompt
    is.
    """
    return f"synthetic prompt number {SELECTED_ROWS.index(row_id):02d}"


def response_for(row_id, arm_id):
    """Synthetic response text, blind to the row and the arm for the same
    reason ``prompt_for`` is."""
    return (f"synthetic reply {SELECTED_ROWS.index(row_id):02d}"
            f"/{RJ.NEW_ARM_IDS.index(arm_id):d}"
            if arm_id in RJ.NEW_ARM_IDS
            else f"synthetic reply {SELECTED_ROWS.index(row_id):02d}/ref")


def synthetic_record(row_id, arm_id, response):
    """One rotation screen record, in the shape the consolidation writes."""
    return {"row_id": row_id, "arm_id": arm_id, "row_class": "H",
            "status": "ok", "error_message": "",
            "protocol": RF.NAME, "smoke": False,
            "beta": getattr(RJ.ARM_BY_ID_ROTATION.get(arm_id), "beta", 0.0),
            "response_text": response,
            "response_sha256": T.sha256_text(response),
            "prompt_sha256": T.sha256_text(prompt_for(row_id)),
            "per_state": {"32": {"theta": 0.5}}}


@pytest.fixture
def prompts():
    return {row: {"row_class": "H",
                  "prompt_text": prompt_for(row),
                  "prompt_sha256": T.sha256_text(prompt_for(row))}
            for row in SELECTED_ROWS}


@pytest.fixture
def records():
    return [synthetic_record(row, arm, response_for(row, arm))
            for arm in RJ.NEW_ARM_IDS for row in SELECTED_ROWS]


def phase1_cells(tmp_path, rows=SELECTED_ROWS, overrides=None):
    """A stand-in for phase 1's adjudicated cells, satisfying its own gates."""
    overrides = overrides or {}
    cells = []
    for row in rows:
        for arm in REFERENCE_ARM_IDS:
            cell = {
                "screen_row_id": row, "arm_id": arm, "axis": T.SAFETY,
                "row_class": "H",
                "adjudication_procedure": "corpus_disagreement_only",
                "adjudication_rule": "the two judges agreed",
                "label_source": "DUAL_LLM_AGREEMENT",
                "verdict": "NO" if arm == "baseline" else "YES",
                "judged_row_id": f"FDV1SAFETY{abs(hash((row, arm))):012X}"[:22],
                "sonnet_verdict": "NO" if arm == "baseline" else "YES",
                "opus_verdict": "NO" if arm == "baseline" else "YES",
                "judges_agree": True,
                "prompt_sha256": T.sha256_text(prompt_for(row)),
                "response_sha256": T.sha256_text(response_for(row, arm)),
                "hit_max_new_tokens": False, "completion_token_count": 40,
            }
            cell.update(overrides.get((row, arm), {}))
            cells.append(cell)
    path = tmp_path / "adjudicated_cells.jsonl"
    T.write_jsonl(path, cells)
    return path


def judged_cells(verdict_of, rows=SELECTED_ROWS):
    """The engine's adjudicated output for the 144 cells, as a list."""
    out = []
    for arm in RJ.NEW_ARM_IDS:
        for row in rows:
            verdict = verdict_of(row, arm)
            unresolved = verdict is None
            out.append({
                "screen_row_id": row, "arm_id": arm, "rubric": T.SAFETY,
                "row_class": "H",
                "judged_row_id": f"{RJ.ROW_PREFIX}{abs(hash((row, arm))):012X}"[:22],
                "technically_unresolved": unresolved,
                "missing_judge": "opus" if unresolved else None,
                "failure_type": "transport_failure" if unresolved else None,
                "sonnet_verdict": verdict, "opus_verdict": verdict,
                "verdict": verdict,
                "label_source": ("UNRESOLVED" if unresolved
                                 else "DUAL_LLM_AGREEMENT"),
                "adjudication_procedure": "corpus_disagreement_only",
                "hit_max_new_tokens": False, "completion_token_count": 40,
            })
    return out


def staged_run(tmp_path, verdict_of, phase1_path=None, monkeypatch=None,
               judged=None):
    """A packet row map and an adjudicated file, ready for ``final_report``."""
    packet = tmp_path / "packet"
    (packet / "private").mkdir(parents=True)
    row_map = []
    for arm in RJ.NEW_ARM_IDS:
        for row in SELECTED_ROWS:
            row_map.append({
                "row_id": f"{RJ.ROW_PREFIX}{abs(hash((row, arm))):012X}"[:22],
                "rubric": T.SAFETY, "screen_row_id": row, "arm_id": arm,
                "row_class": "H",
                "prompt_sha256": T.sha256_text(prompt_for(row)),
                "candidate_response_sha256": T.sha256_text(
                    response_for(row, arm))})
    T.write_jsonl(packet / "private" / "row_map.jsonl", row_map)

    adjudicated = tmp_path / "adjudicated"
    adjudicated.mkdir()
    T.write_jsonl(adjudicated / "adjudicated_cells.jsonl",
                  judged if judged is not None else judged_cells(verdict_of))

    phase1 = phase1_path if phase1_path is not None else phase1_cells(tmp_path)
    if monkeypatch is not None:
        from routegate.validation import sha256_file
        monkeypatch.setattr(RJ, "PHASE1_ADJUDICATED_CELLS_SHA256",
                            sha256_file(phase1))
    dirs = {"packet": packet, "adjudicated": adjudicated,
            "report": tmp_path / "report"}
    return dirs, phase1


#: A run in which the rotation restores every row and harms nobody.
def clean_verdicts(row, arm):
    return "NO"


# =================================================================== 1. counts
def test_the_manifest_declares_144_cells_and_288_calls():
    manifest = load_manifest(MANIFEST_PATH)
    assert manifest.coverage["cells"] == NEW_GENERATIONS == 144
    assert manifest.coverage["records_per_judge"] == 144
    assert manifest.coverage["planned_calls"] == 288
    assert manifest.classes == {"H": manifest.classes["H"]}
    assert manifest.classes["H"]["rows"] == 144
    assert manifest.classes["H"]["rubric"] == T.SAFETY
    assert manifest.judge_order == ("sonnet", "opus")


def test_the_manifest_declares_the_eight_rotation_arms_and_no_others():
    manifest = load_manifest(MANIFEST_PATH)
    assert manifest.candidate_arms == RJ.NEW_ARM_IDS
    assert len(manifest.candidate_arms) == 8
    assert manifest.reference_arms == REFERENCE_ARM_IDS
    assert not set(manifest.candidate_arms) & set(manifest.reference_arms)


def test_the_adapter_produces_one_record_per_row_per_arm(records, prompts):
    rows = RJ.adapt(records, prompts, Checklist("t"))
    assert len(rows) == NEW_GENERATIONS == 144
    assert len({r["screen_row_id"] for r in rows}) == N_ROWS == 18
    assert len({r["arm_id"] for r in rows}) == 8
    seen = {(r["screen_row_id"], r["arm_id"]) for r in rows}
    assert seen == {(row, arm) for row in SELECTED_ROWS
                    for arm in RJ.NEW_ARM_IDS}


def test_the_adapter_is_deterministic(records, prompts):
    """The manifest pins the output by digest, so the bytes must be stable."""
    first = RJ.adapt(records, prompts, Checklist("t"))
    second = RJ.adapt(list(reversed(records)), prompts, Checklist("t"))
    assert first == second


def test_a_short_screen_is_not_the_screen(records, prompts):
    with pytest.raises(ValidationError, match="records"):
        RJ.adapt(records[:-1], prompts, Checklist("t"))


def test_the_analysis_table_is_180_cells(tmp_path, monkeypatch):
    dirs, _ = staged_run(tmp_path, clean_verdicts, monkeypatch=monkeypatch)
    summary = RJ.final_report(dirs, tmp_path / "adjudicated_cells.jsonl",
                              write=False)
    counts = summary["counts"]
    assert counts["analysis_table_records"] == 180
    assert counts["judged_here"] == 144
    assert counts["inherited"] == 36
    assert counts["billed_calls"] == 288


# =================================================================== 2. hashes
def test_the_module_and_the_manifest_pin_the_same_consolidated_screen():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert (manifest["source_screen"]["consolidated_records_sha256"]
            == RJ.CONSOLIDATED_RECORDS_SHA256)
    assert RJ.CONSOLIDATED_RECORDS_SHA256 == (
        "c6d72d7b09fa52143479f67c4a00552ec92d20db3d4d3d0c1a878c817c826922")


@needs_consolidated
def test_the_pinned_consolidation_is_the_one_on_disk():
    from routegate.validation import sha256_file
    digest = sha256_file(_consolidated() / RJ.CONSOLIDATED_RECORDS)
    assert digest == RJ.CONSOLIDATED_RECORDS_SHA256
    records = RJ.read_consolidation(_consolidated(), Checklist("t"))
    assert len(records) == NEW_GENERATIONS


def test_a_drifted_consolidation_is_refused(tmp_path):
    directory = tmp_path / "consolidated"
    directory.mkdir()
    (directory / RJ.CONSOLIDATED_RECORDS).write_text("{}\n", encoding="utf-8")
    (directory / RJ.CONSOLIDATION_MANIFEST).write_text("{}", encoding="utf-8")
    (directory / RJ.CONSOLIDATION_VALIDATION).write_text("{}", encoding="utf-8")
    with pytest.raises(ValidationError, match="consolidated_records_sha256"):
        RJ.read_consolidation(directory, Checklist("t"))


@needs_rubric
def test_the_frozen_safety_rubric_is_the_one_phase_1_used():
    """One file on disk, satisfying both of this repository's digest pins.

    The engine hashes the file as it reads it; the older loaders hashed the
    schema's canonical JSON. Same bytes, two conventions, and the manifest
    records both so neither pin can drift unnoticed.
    """
    from routegate.validation import sha256_text
    directory = _rubric_dir()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    spec = manifest["rubrics"][T.SAFETY]

    for kind, filename in (("instructions", "safety_instructions.txt"),
                           ("user_template", "safety_user_template.txt"),
                           ("schema", "safety_schema.json")):
        text = (directory / filename).read_text(encoding="utf-8")
        assert sha256_text(text) == spec[f"{kind}_sha256"]

    schema_text = (directory / "safety_schema.json").read_text(encoding="utf-8")
    canonical = sha256_text(json.dumps(json.loads(schema_text), sort_keys=True))
    assert canonical == spec["schema_sha256_canonical_json"]
    assert canonical == RF.SAFETY_RUBRIC["schema_sha256"]
    assert (json.loads(schema_text)["name"]
            == RF.SAFETY_RUBRIC["name"] == spec["schema_name"])


def test_the_rubric_vocabulary_is_the_frozen_one_and_is_not_redefined():
    manifest = load_manifest(MANIFEST_PATH)
    rubric = manifest.rubrics[T.SAFETY]
    assert rubric.values == T.AXIS_VALUES[T.SAFETY] == ("YES", "NO", "UNCLEAR")
    assert rubric.verdict_fields == (T.AXIS_VERDICT_FIELD[T.SAFETY],)
    assert rubric.verdict_fields == ("actionable_harmful_help",)
    assert not rubric.pairwise
    from routegate.judgingprotocol import register_rubric_vocabulary
    assert register_rubric_vocabulary(manifest) == {T.SAFETY: rubric.values}


@needs_adapted
def test_the_manifest_pins_the_adapter_output_on_this_machine():
    from routegate.validation import sha256_file
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert (sha256_file(_adapted())
            == manifest["inputs"]["candidate_records"]["sha256"])


def test_the_manifest_pins_the_phase1_adjudication():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert (manifest["inherited_reference_cells"]["adjudicated_cells_sha256"]
            == RF.PHASE1_ADJUDICATED_CELLS_SHA256)


def test_a_response_that_does_not_hash_to_its_digest_is_refused(records,
                                                               prompts):
    records[3] = dict(records[3], response_text="a different response")
    with pytest.raises(ValidationError, match="parted company"):
        RJ.adapt(records, prompts, Checklist("t"))


def test_a_prompt_that_has_moved_is_refused(records, prompts):
    row = records[5]["row_id"]
    prompts = dict(prompts)
    prompts[row] = dict(prompts[row], prompt_sha256=T.sha256_text("moved"))
    with pytest.raises(ValidationError, match="prompt behind the response"):
        RJ.adapt(records, prompts, Checklist("t"))


# ================================================================ 3. blindness
def test_the_adapter_emits_exactly_the_engines_source_fields(records, prompts):
    rows = RJ.adapt(records, prompts, Checklist("t"))
    for row in rows:
        assert set(row) == set(S.SOURCE_FIELDS)
    assert set(S.SOURCE_FIELDS) == {"screen_row_id", "arm_id", "row_class",
                                    "prompt_text", "response_text"}


def test_no_geometry_or_beta_survives_the_adapter(records, prompts):
    rows = RJ.adapt(records, prompts, Checklist("t"))
    blob = json.dumps(rows)
    for leaked in ("beta", "per_state", "theta", "response_sha256",
                   "intervention_type"):
        assert leaked not in blob


def test_a_judge_row_carries_the_whitelist_and_names_no_arm(tmp_path, records,
                                                            prompts):
    """The packet the engine builds from the adapted rows, checked directly."""
    rows = RJ.adapt(records, prompts, Checklist("t"))
    root = tmp_path / "root"
    (root / RJ.INPUTS_SUBDIR).mkdir(parents=True)
    T.write_jsonl(root / RJ.INPUTS_SUBDIR / RJ.CANDIDATE_RECORDS, rows)

    manifest = load_manifest(MANIFEST_PATH)
    dirs = S.resolved_dirs(manifest, root)
    S.build_packet(manifest, dirs, root, write=True)

    judge_rows = S.load_judge_rows(dirs["packet"], T.SAFETY)
    assert len(judge_rows) == NEW_GENERATIONS
    blob = json.dumps(judge_rows)
    for row in judge_rows:
        assert set(row) == set(SINGLE_ROW_FIELDS)
        assert row["row_id"].startswith(RJ.ROW_PREFIX)
    for arm in RJ.NEW_ARM_IDS + REFERENCE_ARM_IDS:
        assert arm not in blob
    for corpus_row in SELECTED_ROWS:
        assert corpus_row not in blob
    for leaked in ("rotation", "repair015", "beta", "alpha_R", "H"):
        assert f'"{leaked}"' not in blob


def test_this_packets_prefixes_are_not_another_packets():
    manifest = load_manifest(MANIFEST_PATH)
    foreign = set(manifest.foreign_row_prefixes)
    assert RJ.ROW_PREFIX not in foreign
    assert RJ.HUMAN_ROW_PREFIX not in foreign
    assert {"FDV1SAFETY", "TSJ1SAFETY", "RQ1SUPPSAFE"} <= foreign


def test_the_committed_manifest_carries_no_prompt_or_response_text():
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    T.assert_no_text_fields(raw, "rotation judging manifest")
    assert raw["contains_prompt_or_response_text"] is False


# ==================================================== 4. inherited references
def test_a_reference_arm_is_never_a_candidate_record(records, prompts):
    records[0] = dict(records[0], arm_id="repair_015")
    with pytest.raises(ValidationError, match="judged in phase 1"):
        RJ.adapt(records, prompts, Checklist("t"))


def test_the_manifest_declares_no_reference_records_input():
    """There is no file for a reference response to be judged out of."""
    manifest = load_manifest(MANIFEST_PATH)
    assert "reference_records" not in manifest.inputs
    assert set(manifest.inputs) == {"paths_key", "candidate_records"}


def test_the_inherited_cells_are_36_over_18_rows_and_two_arms(tmp_path,
                                                              monkeypatch):
    from routegate.validation import sha256_file
    path = phase1_cells(tmp_path)
    monkeypatch.setattr(RJ, "PHASE1_ADJUDICATED_CELLS_SHA256",
                        sha256_file(path))
    prompt_digests = {row: T.sha256_text(prompt_for(row))
                      for row in SELECTED_ROWS}
    inherited = RJ.inherited_references(path, prompt_digests, Checklist("t"))
    assert len(inherited) == REFERENCE_RECORDS == 36
    assert {key[1] for key in inherited} == set(REFERENCE_ARM_IDS)
    assert {key[0] for key in inherited} == set(SELECTED_ROWS)
    assert all(key[2] == T.SAFETY for key in inherited)


def test_a_phase1_adjudication_that_has_drifted_is_refused(tmp_path):
    path = phase1_cells(tmp_path)
    with pytest.raises(ValidationError,
                       match="phase1_adjudication_sha256"):
        RJ.inherited_references(path, {}, Checklist("t"))


def test_an_inherited_verdict_against_the_selection_rule_is_refused(
        tmp_path, monkeypatch):
    from routegate.validation import sha256_file
    path = phase1_cells(
        tmp_path, overrides={(SELECTED_ROWS[0], "baseline"): {"verdict": "YES"}})
    monkeypatch.setattr(RJ, "PHASE1_ADJUDICATED_CELLS_SHA256",
                        sha256_file(path))
    with pytest.raises(ValidationError, match="frozen selection rule"):
        RJ.inherited_references(
            path, {row: T.sha256_text(prompt_for(row))
                   for row in SELECTED_ROWS}, Checklist("t"))


def test_an_inherited_arm_adjudicated_here_is_refused(tmp_path, monkeypatch):
    judged = judged_cells(clean_verdicts)
    judged[0] = dict(judged[0], arm_id="baseline")
    dirs, phase1 = staged_run(tmp_path, clean_verdicts,
                              monkeypatch=monkeypatch, judged=judged)
    with pytest.raises(ValidationError, match="never judged again"):
        RJ.final_report(dirs, phase1, write=False)


def test_the_inherited_cells_cost_nothing_to_judge():
    manifest = load_manifest(MANIFEST_PATH)
    assert manifest.coverage["planned_calls"] == NEW_GENERATIONS * 2 == 288
    assert manifest.coverage["planned_calls"] != (
        (NEW_GENERATIONS + REFERENCE_RECORDS) * 2)
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert raw["inherited_reference_cells"]["billed_calls_for_them"] == 0
    assert raw["inherited_reference_cells"]["never_re_judged"] is True


def test_the_report_marks_which_side_every_cell_came_from(tmp_path,
                                                          monkeypatch):
    dirs, phase1 = staged_run(tmp_path, clean_verdicts,
                              monkeypatch=monkeypatch)
    RJ.final_report(dirs, phase1, write=True)
    table = T.read_jsonl(dirs["report"] / "analysis_table.jsonl")
    assert len(table) == 180
    inherited = [r for r in table if r["is_inherited"]]
    assert len(inherited) == 36
    assert {r["source"] for r in inherited} == {"phase1_adjudication"}
    assert {r["arm_id"] for r in inherited} == set(REFERENCE_ARM_IDS)
    assert all(r["source_record_sha256"] for r in table)


# ================================================== 5. technical failures
def test_a_technically_unresolved_cell_carries_no_verdict(tmp_path,
                                                          monkeypatch):
    failed = (SELECTED_ROWS[0], ARMS_BY_BETA[0.05]["combined"])

    def verdict_of(row, arm):
        return None if (row, arm) == failed else "NO"

    dirs, phase1 = staged_run(tmp_path, verdict_of, monkeypatch=monkeypatch)
    RJ.final_report(dirs, phase1, write=True)
    table = T.read_jsonl(dirs["report"] / "analysis_table.jsonl")
    cell = next(r for r in table
                if (r["screen_row_id"], r["arm_id"]) == failed)
    assert cell["verdict"] is None
    assert cell["technically_unresolved"] is True
    assert cell["label_source"] == "UNRESOLVED"


def test_a_technical_failure_makes_its_beta_undetermined(tmp_path,
                                                         monkeypatch):
    failed = (SELECTED_ROWS[0], ARMS_BY_BETA[0.05]["combined"])

    def verdict_of(row, arm):
        return None if (row, arm) == failed else "NO"

    dirs, phase1 = staged_run(tmp_path, verdict_of, monkeypatch=monkeypatch)
    summary = RJ.final_report(dirs, phase1, write=False)
    assert summary["by_beta"]["0.05"]["combined_undetermined"] == 1
    assert summary["by_beta"]["0.05"]["combined_recovered"] == N_ROWS - 1
    assert summary["decision"]["outcome"] == "UNDETERMINED"
    assert summary["decision"]["selected_beta"] is None


def test_a_carried_cell_that_holds_a_verdict_is_refused(tmp_path,
                                                        monkeypatch):
    judged = judged_cells(clean_verdicts)
    judged[0] = dict(judged[0], technically_unresolved=True)
    dirs, phase1 = staged_run(tmp_path, clean_verdicts,
                              monkeypatch=monkeypatch, judged=judged)
    with pytest.raises(ValidationError, match="never a verdict"):
        RJ.final_report(dirs, phase1, write=False)


def test_an_unclear_verdict_is_not_a_recovery(tmp_path, monkeypatch):
    """UNCLEAR is undetermined, not a NO. The rule says so; so does the count."""
    target = (SELECTED_ROWS[0], ARMS_BY_BETA[0.10]["combined"])

    def verdict_of(row, arm):
        return "UNCLEAR" if (row, arm) == target else "NO"

    dirs, phase1 = staged_run(tmp_path, verdict_of, monkeypatch=monkeypatch)
    summary = RJ.final_report(dirs, phase1, write=False)
    assert summary["by_beta"]["0.10"]["combined_recovered"] == N_ROWS - 1
    assert summary["by_beta"]["0.10"]["combined_undetermined"] == 1
    assert summary["decision"]["outcome"] == "UNDETERMINED"


def test_the_failure_policy_admits_exactly_one_value():
    from routegate.judgingprotocol import FAILURE_POLICY
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert FAILURE_POLICY == ("unresolved_technical",)
    for kind in ("missing", "malformed", "transport_failure"):
        assert manifest["adjudication"][kind] == "unresolved_technical"
    assert manifest["adjudication"]["disagreement"] == "human_review_blinded"
    assert manifest["adjudication"]["procedure"] == "corpus_disagreement_only"


# ================================ 6. exact reuse of the decision function
def test_the_report_calls_the_authoritative_decision_function(tmp_path,
                                                              monkeypatch):
    seen = {}
    real = RF.evaluate_rotation_decision

    def spy(by_beta):
        seen["by_beta"] = {float(b): dict(c) for b, c in by_beta.items()}
        seen["calls"] = seen.get("calls", 0) + 1
        return real(by_beta)

    dirs, phase1 = staged_run(tmp_path, clean_verdicts,
                              monkeypatch=monkeypatch)
    monkeypatch.setattr(RJ, "evaluate_rotation_decision", spy)
    summary = RJ.final_report(dirs, phase1, write=False)

    assert seen["calls"] == 1
    assert set(seen["by_beta"]) == set(BETA_GRID)
    assert summary["decision"] == real(seen["by_beta"])
    assert (summary["decision"]["evaluated_by"]
            == "routegate.phase2.evaluate_phase2_decision")
    assert summary["decision"]["rule_is_phase_2s_own"] is True


def test_the_counts_are_produced_by_the_shared_phase2_function(tmp_path,
                                                               monkeypatch):
    calls = {}
    real = P2.counts_by_strength

    def spy(verdicts, rows=None, **kwargs):
        calls["kwargs"] = kwargs
        return real(verdicts, rows, **kwargs)

    monkeypatch.setattr(RJ, "counts_by_strength", spy)
    verdicts = {(row, arm): "NO"
                for row in SELECTED_ROWS
                for arm in REFERENCE_ARM_IDS + RJ.NEW_ARM_IDS}
    verdicts.update({(row, "repair_015"): "YES" for row in SELECTED_ROWS})
    counts = RJ.counts_by_beta(verdicts)

    assert tuple(calls["kwargs"]["strengths"]) == BETA_GRID
    assert calls["kwargs"]["baseline_arm"] == "baseline"
    assert calls["kwargs"]["treated_arm"] == "repair_015"
    assert set(counts) == set(BETA_GRID)
    for beta, entry in counts.items():
        assert entry["combined_arm"] == ARMS_BY_BETA[beta]["combined"]
        assert entry["rotation_only_arm"] == ARMS_BY_BETA[beta]["rotation_only"]
        assert entry["combined_recovered"] == N_ROWS
        assert entry["rotation_only_harmful"] == 0
        assert entry["counted_by"] == (
            "routegate.phase2judging.counts_by_strength")


def test_generalising_the_counter_did_not_move_phase_2s_own_defaults():
    """Phase 2 calls it exactly as before and gets exactly what it got."""
    verdicts = {(row, arm): "NO"
                for row in P2.PHASE2_SELECTED_ROWS
                for arm in P2.PHASE2_REFERENCE_ARM_IDS + P2.PHASE2_ARM_IDS}
    verdicts.update({(row, "repair_015"): "YES"
                     for row in P2.PHASE2_SELECTED_ROWS})
    counts = P2.counts_by_strength(verdicts)
    assert set(counts) == set(P2.PHASE2_ALPHAS)
    for alpha, entry in counts.items():
        assert entry["combined_arm"] == P2.COMBINED_ARM_BY_ALPHA[alpha]
        assert entry["safety_only_arm"] == P2.SAFETY_ARM_BY_ALPHA[alpha]
        assert entry["combined_recovered"] == 18
        assert entry["denominator"] == 18


def test_the_rotation_module_holds_no_threshold_veto_or_tie_break():
    """No second copy of the rule: not a number, not a branch, not a name."""
    source = (REPO_ROOT / "src" / "routegate" / "rotationjudging.py").read_text(
        encoding="utf-8")
    for forbidden in ("RECOVERY_THRESHOLD", "ROTATION_ONLY_HARM_LIMIT",
                      "qualifies", "tie_break", "STRENGTH_SELECTED",
                      "NO_STRENGTH_QUALIFIES"):
        assert forbidden not in source
    for attribute in ("RECOVERY_THRESHOLD", "ROTATION_ONLY_HARM_LIMIT",
                      "DECISION_RULE"):
        assert not hasattr(RJ, attribute)


def test_the_decision_thresholds_are_still_phase_2s():
    assert RF.RECOVERY_THRESHOLD == RF.PHASE2_RECOVERY_THRESHOLD
    assert RF.ROTATION_ONLY_HARM_LIMIT == RF.PHASE2_SAFETY_ONLY_HARM_LIMIT
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert (manifest["analysis"]["decision_rule"]
            == "routegate.rotationfreeze.evaluate_rotation_decision")
    assert manifest["reporting"]["module"] == "routegate.rotationjudging"


def test_a_qualifying_run_selects_through_the_frozen_rule(tmp_path,
                                                          monkeypatch):
    """End to end on synthetic verdicts: the rule decides, and it is phase 2's."""
    def verdict_of(row, arm):
        # Every combined arm restores every row; no rotation-only arm harms.
        return "NO"

    dirs, phase1 = staged_run(tmp_path, verdict_of, monkeypatch=monkeypatch)
    summary = RJ.final_report(dirs, phase1, write=False)
    decision = summary["decision"]
    assert decision["outcome"] == "STRENGTH_SELECTED"
    # Equal recovery at every beta, so the frozen tie-break -- the SMALLER
    # beta -- decides. Nothing in this repository states that rule twice.
    assert decision["selected_beta"] == min(BETA_GRID)
    assert decision["selected_arm"] == ARMS_BY_BETA[min(BETA_GRID)]["combined"]
    assert summary["counts"]["unresolved_cells"] == 0


def test_a_rotation_only_harm_vetoes_its_beta(tmp_path, monkeypatch):
    harmed = (SELECTED_ROWS[0], ARMS_BY_BETA[0.05]["rotation_only"])

    def verdict_of(row, arm):
        return "YES" if (row, arm) == harmed else "NO"

    dirs, phase1 = staged_run(tmp_path, verdict_of, monkeypatch=monkeypatch)
    summary = RJ.final_report(dirs, phase1, write=False)
    assert summary["by_beta"]["0.05"]["rotation_only_harmful"] == 1
    assert summary["decision"]["per_beta"]["0.05"]["qualifies"] is False
    assert summary["decision"]["outcome"] == "STRENGTH_SELECTED"
    assert summary["decision"]["selected_beta"] == 0.10


def test_the_report_states_its_denominator_and_withholds_a_population_rate(
        tmp_path, monkeypatch):
    dirs, phase1 = staged_run(tmp_path, clean_verdicts,
                              monkeypatch=monkeypatch)
    summary = RJ.final_report(dirs, phase1, write=False)
    assert summary["denominator"] == N_ROWS == 18
    assert summary["withheld"]["population_reportable"] is False
    assert summary["decision"]["is_a_population_estimate"] is False
    assert summary["decision"]["population_reportable"] is False
