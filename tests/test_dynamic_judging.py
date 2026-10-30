"""Registering the completed dynamic controller arm with the common engine.

The arm is finished and pinned; what is under test is the small amount of code
that lets the COMMON engine judge it, the audit that had to pass before it
could be registered, and the reading that comes back out.

* the **audit**, and above all the 267/213 split. The run manifest's
  ``already_present`` is a RESUME count and the corpus happens to hold 267 E/B
  rows, so the arithmetic invites exactly one wrong reading -- that the arm
  refused to steer its H rows. These tests hold the split to being a sorted
  row-id prefix, hold every record to being this arm's own with both hooks
  fired and a non-zero edit, and fail if either side of the split is one class.
* the **counts** -- 480 cells, 178 references, 960 billed calls, 0 reused.
* the **hashes** -- the completed run's four files, the frozen rubric text
  under both digest conventions, the two completed adjudications the report
  reads and the adapter's own output.
* **blindness** -- a judge row carries the whitelist and nothing that names an
  arm, a coefficient, a fold, a site or a corpus row.
* the **reuse rule** -- item 12 as a gate: everything must match, the
  scientific cell names the arm, and a byte-identical response on another arm
  is a diagnostic rather than a licence.
* **technical failures** -- UNRESOLVED_TECHNICAL is terminal, never a verdict,
  and never something a human can fill in.
* **exact reuse of the frozen readings** -- the report calls
  ``twositejudging.e_reading`` / ``b_reading`` / ``h_reading`` and holds no
  threshold or classification rule of its own.

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

from routegate import dynamicjudging as DJ
from routegate import judgingstages as S
from routegate import triplets as T
from routegate import twositejudging as TS
from routegate.gatecontroller import ALPHA_R
from routegate.judgingprotocol import (
    CELL_FIELDS,
    FAILURE_POLICY,
    PAIRWISE_ROW_FIELDS,
    SINGLE_ROW_FIELDS,
    JudgingProtocolError,
    decode_pairwise,
    load_manifest,
    register_rubric_vocabulary,
)
from routegate.validation import Checklist, ValidationError

PATHS_PATH = REPO_ROOT / "configs" / "paths.local.json"
MANIFEST_PATH = DJ.JUDGING_MANIFEST


def _local_paths():
    if not PATHS_PATH.exists():
        return {}
    try:
        return json.loads(PATHS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _dir_for(key, marker):
    root = _local_paths().get(key)
    if not root:
        return None
    directory = Path(root)
    return directory if (directory / marker).is_file() else None


def _run_dir():
    return _dir_for(DJ.PATHS_KEY_RUN, DJ.RUN_RECORDS)


def _rubric_source_dirs():
    paths = _local_paths()
    out = {}
    for key in (DJ.PATHS_KEY_TRIPLETS, DJ.PATHS_KEY_STATIC):
        if paths.get(key):
            out[key] = Path(paths[key])
    return out


needs_run = pytest.mark.skipif(
    _run_dir() is None,
    reason="the completed dynamic_raw run is not reachable on this machine")

MANIFEST = load_manifest(MANIFEST_PATH)


# ================================================================ fixtures
def _frame():
    """480 synthetic rows in the real class proportions, sorted by row id."""
    rows = []
    for row_class, n in sorted(TS.ROWS_BY_CLASS.items()):
        rows.extend((f"{row_class.lower()}row:{i:05d}", row_class)
                    for i in range(n))
    return sorted(rows)


#: Judge-facing text that does NOT contain the row id, because the engine's
#: blindness gate refuses a judge row that names a corpus row -- and a fixture
#: that could not pass that gate would test nothing about the real packet.
_INDEX = {row_id: i for i, (row_id, _c) in enumerate(_frame())}


def _prompt(row_id):
    return f"question {_INDEX[row_id]:04d}"


def _response(row_id):
    return f"answer {_INDEX[row_id]:04d}"


def _baseline_text(row_id):
    return f"prior answer {_INDEX[row_id]:04d}"


def _record(row_id, row_class, response=None, *, g_r=0.4, fold=0, arm=None,
            hooks=2, delta=1.0, alpha=None):
    """One dynamic screen record, in the shape the run wrote."""
    gate = float(g_r)
    applied = float(ALPHA_R * gate) if alpha is None else float(alpha)
    controller = {"fold": fold, "g_R": gate, "applied_alpha_R": applied,
                  "is_out_of_fold": True, "read_state": 32,
                  "checkpoint": f"controller_fold_{fold}.json"}
    return {
        "row_id": row_id, "row_class": row_class,
        "arm_id": arm or DJ.CANDIDATE_ARM, "status": "ok",
        "alpha_R": applied, "alpha_S": 0.0,
        "prompt_sha256": T.sha256_text(_prompt(row_id)),
        "response_text": _response(row_id) if response is None else response,
        "hooks_expected": 2, "hooks_fired": hooks,
        "per_state": {
            "32": {"applied_delta_norm": delta, "controller": dict(controller)},
            "33": {"applied_delta_norm": delta, "controller": dict(controller)},
        },
    }


def _corpus(rows):
    return {row_id: {"row_class": row_class,
                     "prompt_text": _prompt(row_id),
                     "prompt_sha256": T.sha256_text(_prompt(row_id))}
            for row_id, row_class in rows}


def _records():
    return [_record(row_id, row_class) for row_id, row_class in _frame()]


# ================================================================= the audit
def test_the_audit_says_nothing_was_inherited_or_reused():
    assert DJ.AUDIT["verdict"] == "PASS"
    assert DJ.AUDIT["generated_by_this_arm"] == 480
    assert DJ.AUDIT["inherited_or_reused"] == 0
    assert DJ.AUDIT["rows_with_an_exactly_zero_coefficient"] == 0
    assert DJ.AUDIT["rows_with_an_exactly_zero_applied_delta"] == 0


def test_the_267_and_the_213_are_not_classes():
    """The whole point: neither side of the resume split is one class."""
    present = DJ.AUDIT["already_present_by_class"]
    written = DJ.AUDIT["generated_here_by_class"]
    assert sum(present.values()) == DJ.AUDIT["already_present_on_resume"] == 213
    assert sum(written.values()) == DJ.AUDIT["generated_here_last_invocation"]
    assert len(present) == 3 and len(written) == 3
    assert present["H"] == 32 and written["H"] == 181
    assert present["H"] + written["H"] == TS.ROWS_BY_CLASS["H"]
    for row_class, total in TS.ROWS_BY_CLASS.items():
        assert present[row_class] + written[row_class] == total


def test_the_three_invocations_add_up_to_the_split():
    invocations = DJ.AUDIT["invocations"]
    assert [i["records"] for i in invocations] == [22, 191, 267]
    assert (invocations[0]["records"] + invocations[1]["records"]
            == DJ.AUDIT["already_present_on_resume"])
    assert invocations[2]["records"] == DJ.AUDIT["generated_here_last_invocation"]


def test_a_resume_prefix_that_is_one_class_is_refused():
    """A split made by class is what item 7 of the audit stops on."""
    records = _records()
    by_class = sorted(records, key=lambda r: (r["row_class"] != "H",
                                              r["row_id"]))
    with pytest.raises(ValidationError, match="sorted row-id order"):
        DJ.assert_the_split_is_a_resume(by_class, 213, Checklist("t"))


def test_a_reused_row_with_a_non_zero_edit_is_refused():
    """An exactly zero edit is the only thing a baseline reuse could be."""
    rows = _frame()
    records = _records()
    records[7]["per_state"]["32"]["applied_delta_norm"] = 0.0
    with pytest.raises(ValidationError, match="exactly zero edit"):
        DJ.assert_the_split_is_a_resume(records, 213, Checklist("t"))


def test_a_record_that_did_not_fire_both_hooks_is_refused():
    rows = _frame()
    records = _records()
    records[3]["hooks_fired"] = 1
    with pytest.raises(ValidationError, match="fired"):
        DJ.assert_the_split_is_a_resume(records, 213, Checklist("t"))


def test_two_coefficients_at_the_two_sites_are_refused():
    rows = _frame()
    records = _records()
    records[5]["per_state"]["33"]["controller"]["g_R"] = 0.9
    with pytest.raises(ValidationError, match="one_coefficient|carry two"):
        DJ.assert_the_split_is_a_resume(records, 213, Checklist("t"))


def test_an_applied_strength_that_is_not_alpha_times_g_is_refused():
    rows = _frame()
    records = _records()
    records[9]["alpha_R"] = 0.15
    with pytest.raises(ValidationError, match="disagree"):
        DJ.assert_the_split_is_a_resume(records, 213, Checklist("t"))


def test_an_in_fold_coefficient_is_refused():
    rows = _frame()
    records = _records()
    records[11]["per_state"]["32"]["controller"]["is_out_of_fold"] = False
    with pytest.raises(ValidationError, match="out_of_fold|were not"):
        DJ.assert_the_split_is_a_resume(records, 213, Checklist("t"))


def test_a_clean_resume_prefix_passes_every_general_gate():
    """The gates that hold for any run of this arm, on synthetic records."""
    checks = Checklist("t")
    out = DJ.assert_the_split_is_a_resume(_records(), 213, checks)
    assert checks.all_passed
    assert sum(out["already_present_by_class"].values()) == 213
    assert sum(out["generated_here_by_class"].values()) == 480 - 213
    assert set(out["invariants"]) == set(DJ.AUDIT_INVARIANTS)
    for name in DJ.AUDIT_INVARIANTS:
        assert any(e["check"] == name and e["status"] == "PASS"
                   for e in checks.entries)


def test_the_published_audit_numbers_are_asserted_separately():
    """A run whose split is clean but different is still held to the audit."""
    checks = Checklist("t")
    out = DJ.assert_the_split_is_a_resume(_records(), 213, checks)
    with pytest.raises(ValidationError,
                       match="the_split_matches_the_published_audit"):
        DJ.assert_matches_the_audit(out, checks)
    DJ.assert_matches_the_audit(
        {"already_present_by_class":
             dict(sorted(DJ.AUDIT["already_present_by_class"].items())),
         "generated_here_by_class":
             dict(sorted(DJ.AUDIT["generated_here_by_class"].items()))},
        Checklist("t"))


# ================================================================ the adapter
def test_adapt_carries_exactly_the_source_fields():
    rows = _frame()
    records = _records()
    out = DJ.adapt(records, _corpus(rows), Checklist("t"))
    assert len(out) == 480
    assert all(sorted(r) == sorted(S.SOURCE_FIELDS) for r in out)
    assert {r["arm_id"] for r in out} == {DJ.CANDIDATE_ARM}


def test_adapt_is_deterministic():
    rows = _frame()
    records = _records()
    corpus = _corpus(rows)
    first = DJ.adapt(records, corpus, Checklist("t"))
    second = DJ.adapt(list(reversed(records)), corpus, Checklist("t"))
    assert first == second


def test_adapt_refuses_a_record_on_another_arm():
    rows = _frame()
    records = _records()
    records[0]["arm_id"] = DJ.BASELINE_ARM
    with pytest.raises(DJ.DynamicJudgingError, match="already carry"):
        DJ.adapt(records, _corpus(rows), Checklist("t"))


def test_adapt_refuses_a_prompt_that_has_moved():
    rows = _frame()
    records = _records()
    records[0]["prompt_sha256"] = T.sha256_text("something else")
    with pytest.raises(DJ.DynamicJudgingError, match="has moved"):
        DJ.adapt(records, _corpus(rows), Checklist("t"))


def test_adapt_refuses_a_class_the_corpus_disagrees_with():
    rows = _frame()
    records = _records()
    records[0]["row_class"] = "H" if records[0]["row_class"] != "H" else "E"
    with pytest.raises(DJ.DynamicJudgingError, match="the corpus calls it"):
        DJ.adapt(records, _corpus(rows), Checklist("t"))


def test_only_the_B_baselines_become_references():
    rows = _frame()
    baselines = [{"row_id": row_id, "arm_id": DJ.BASELINE_ARM,
                  "row_class": row_class,
                  "prompt_sha256": T.sha256_text(_prompt(row_id)),
                  "response_text": _baseline_text(row_id)}
                 for row_id, row_class in rows]
    out = DJ.adapt_baselines(baselines, _corpus(rows), Checklist("t"))
    assert len(out) == TS.ROWS_BY_CLASS["B"]
    assert {r["row_class"] for r in out} == {"B"}


def test_a_reference_on_the_wrong_arm_is_refused():
    rows = _frame()
    baselines = [{"row_id": rows[0][0], "arm_id": DJ.CANDIDATE_ARM,
                  "row_class": rows[0][1],
                  "prompt_sha256": T.sha256_text(_prompt(rows[0][0])),
                  "response_text": "x"}]
    with pytest.raises(DJ.DynamicJudgingError, match="expected 'baseline'"):
        DJ.adapt_baselines(baselines, _corpus(rows), Checklist("t"))


# ============================================================== the manifest
def test_the_manifest_loads_and_its_counts_add_up():
    assert MANIFEST.experiment_id == DJ.EXPERIMENT_ID
    assert MANIFEST.packet_id == DJ.PACKET_ID
    assert MANIFEST.candidate_arms == (DJ.CANDIDATE_ARM,)
    assert MANIFEST.reference_arms == (DJ.BASELINE_ARM,)
    assert int(MANIFEST.coverage["cells"]) == 480
    assert int(MANIFEST.coverage["records_per_judge"]) == 480
    assert int(MANIFEST.coverage["planned_calls"]) == 960


def test_the_manifest_uses_the_frozen_sampling_and_procedure():
    assert MANIFEST.temperature == T.TEMPERATURE == 0.0
    assert MANIFEST.max_tokens == T.MAX_TOKENS == 300
    assert MANIFEST.procedure == "corpus_disagreement_only"
    assert MANIFEST.judge_order == T.JUDGES


def test_every_failure_is_unresolved_technical():
    adjudication = MANIFEST.raw["adjudication"]
    for kind in ("missing", "malformed", "transport_failure"):
        assert adjudication[kind] == FAILURE_POLICY[0] == "unresolved_technical"
    assert adjudication["disagreement"] == "human_review_blinded"


def test_a_manifest_that_turns_a_failure_into_a_verdict_is_refused(tmp_path):
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    raw["adjudication"]["transport_failure"] = "NO"
    path = tmp_path / "loosened.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(JudgingProtocolError, match="no manifest value"):
        load_manifest(path)


def test_the_class_to_rubric_map_is_the_one_the_task_names():
    assert MANIFEST.rubric_for_class("E").name == T.AXIS2
    assert MANIFEST.rubric_for_class("B").name == DJ.PAIRWISE_QUALITY
    assert MANIFEST.rubric_for_class("H").name == T.SAFETY
    assert MANIFEST.rubrics["B" and DJ.PAIRWISE_QUALITY].pairwise is True
    assert MANIFEST.rubrics[T.AXIS2].pairwise is False
    assert MANIFEST.rubrics[T.SAFETY].pairwise is False


def test_the_rubric_vocabularies_are_the_frozen_ones():
    assert MANIFEST.rubrics[T.AXIS2].values == T.AXIS2_VALUES
    assert MANIFEST.rubrics[T.SAFETY].values == T.SAFETY_VALUES
    quality = MANIFEST.rubrics[DJ.PAIRWISE_QUALITY]
    assert set(quality.judge_values) == set(DJ.PAIRWISE_JUDGE_VALUES)
    assert set(quality.values) == set(DJ.PAIRWISE_VALUES)
    assert len(quality.values) == len(T.QUALITY_VALUES) == 9


def test_registering_this_manifest_does_not_redefine_an_existing_vocabulary():
    before = dict(T.AXIS_VALUES)
    register_rubric_vocabulary(MANIFEST)
    assert T.AXIS_VALUES[T.AXIS2] == before[T.AXIS2]
    assert T.AXIS_VALUES[T.SAFETY] == before[T.SAFETY]
    assert T.AXIS_VALUES[T.QUALITY] == before[T.QUALITY]
    assert T.AXIS_VALUES[DJ.PAIRWISE_QUALITY] == DJ.PAIRWISE_VALUES


def test_the_pairwise_rubric_could_not_have_reused_the_quality_name():
    """Why the new name exists at all, asserted rather than asserted in prose."""
    assert T.AXIS_VALUES[T.QUALITY] != DJ.PAIRWISE_VALUES
    assert DJ.PAIRWISE_QUALITY != T.QUALITY
    assert (DJ.RUBRIC_SOURCE[DJ.PAIRWISE_QUALITY]["schema_name"]
            == T.AXIS_SCHEMA_NAME[T.QUALITY])


def test_the_row_prefixes_cannot_join_another_packet():
    mine = set(MANIFEST.rubrics[name].row_prefix for name in MANIFEST.rubrics)
    assert mine == set(DJ.ROW_PREFIX.values())
    assert not mine & set(MANIFEST.foreign_row_prefixes)
    assert MANIFEST.human_row_prefix == DJ.HUMAN_ROW_PREFIX
    assert MANIFEST.human_row_prefix not in mine


def test_the_manifest_carries_no_prompt_or_response_text():
    raw = MANIFEST_PATH.read_text(encoding="utf-8")
    assert MANIFEST.raw["contains_prompt_or_response_text"] is False
    T.assert_no_text_fields(json.loads(raw), "manifest")


def test_the_manifest_pins_the_completed_run_by_digest():
    source = MANIFEST.raw["source_run"]
    assert source["records_sha256"] == DJ.RUN_RECORDS_SHA256
    assert source["arm"] == DJ.CANDIDATE_ARM
    assert source["alpha_R"] == 0.15 and source["alpha_S"] == 0.0
    assert source["historical_blocks"] == [31, 32]
    assert source["read_state"] == 32


def test_the_manifest_records_the_audit_it_was_registered_under():
    audit = MANIFEST.raw["audit"]
    assert audit["verdict"] == "PASS"
    assert audit["inherited_or_reused"] == 0
    assert audit["already_present_by_class"] == DJ.AUDIT["already_present_by_class"]
    assert audit["generated_here_by_class"] == DJ.AUDIT["generated_here_by_class"]


# ================================================================= the rubrics
def test_the_schema_digest_is_pinned_under_both_conventions():
    """The same bytes; the older loaders hashed the canonical JSON."""
    for name, spec in DJ.RUBRIC_SOURCE.items():
        declared = MANIFEST.raw["rubrics"][name]
        assert declared["schema_sha256"] == spec["schema_sha256"]
        assert (declared["schema_sha256_canonical_json"]
                == spec["schema_sha256_canonical_json"])


@pytest.mark.skipif(not _rubric_source_dirs(),
                    reason="the frozen rubric text is not reachable")
def test_the_frozen_rubric_files_still_hash_to_what_is_pinned():
    from routegate.validation import sha256_file
    sources = _rubric_source_dirs()
    checked = 0
    for name, spec in DJ.RUBRIC_SOURCE.items():
        root = sources.get(str(spec["paths_key"]))
        if root is None:
            continue
        directory = root / str(spec["subdir"])
        for part, (source_name, _staged) in DJ.rubric_filenames(name).items():
            path = directory / source_name
            if not path.is_file():
                continue
            assert sha256_file(path) == spec[f"{part}_sha256"]
            checked += 1
    assert checked


@pytest.mark.skipif(not _rubric_source_dirs(),
                    reason="the frozen rubric text is not reachable")
def test_the_one_schema_file_satisfies_both_digest_conventions():
    from routegate.validation import sha256_file
    sources = _rubric_source_dirs()
    for name in (T.AXIS2, T.SAFETY):
        spec = DJ.RUBRIC_SOURCE[name]
        root = sources.get(str(spec["paths_key"]))
        if root is None:
            continue
        path = root / str(spec["subdir"]) / f"{spec['source_stem']}_schema.json"
        if not path.is_file():
            continue
        assert sha256_file(path) == spec["schema_sha256"]
        canonical = T.sha256_text(json.dumps(
            json.loads(path.read_text(encoding="utf-8")), sort_keys=True))
        assert canonical == spec["schema_sha256_canonical_json"]


@pytest.mark.skipif(not _rubric_source_dirs(),
                    reason="the frozen rubric text is not reachable")
def test_staging_copies_the_rubrics_byte_for_byte(tmp_path):
    from routegate.validation import sha256_file
    sources = _rubric_source_dirs()
    if len(sources) < 2:
        pytest.skip("both frozen rubric roots are needed")
    checks = Checklist("t")
    out = DJ.stage_rubrics(sources, tmp_path / "judge_rubrics", checks)
    assert out["files"] == 9
    for name, spec in DJ.RUBRIC_SOURCE.items():
        for part, (source_name, staged_name) in DJ.rubric_filenames(name).items():
            source = sources[str(spec["paths_key"])] / str(spec["subdir"]) / source_name
            staged = tmp_path / "judge_rubrics" / staged_name
            assert staged.read_bytes() == source.read_bytes()
            assert sha256_file(staged) == spec[f"{part}_sha256"]
    assert checks.all_passed


@pytest.mark.skipif(not _rubric_source_dirs(),
                    reason="the frozen rubric text is not reachable")
def test_staging_refuses_a_rubric_whose_bytes_moved(tmp_path, monkeypatch):
    sources = _rubric_source_dirs()
    if len(sources) < 2:
        pytest.skip("both frozen rubric roots are needed")
    moved = dict(DJ.RUBRIC_SOURCE[T.SAFETY])
    moved["instructions_sha256"] = "0" * 64
    monkeypatch.setitem(DJ.RUBRIC_SOURCE, T.SAFETY, moved)
    with pytest.raises(ValidationError, match="frozen_safety_instructions"):
        DJ.stage_rubrics(sources, tmp_path / "out", Checklist("t"))


# ================================================================= blindness
def _packet(tmp_path, records, references):
    root = tmp_path / "root"
    dirs = S.resolved_dirs(MANIFEST, root)
    inputs = root / DJ.INPUTS_SUBDIR
    inputs.mkdir(parents=True)
    T.write_jsonl(inputs / DJ.CANDIDATE_RECORDS, records)
    T.write_jsonl(inputs / DJ.REFERENCE_RECORDS, references)
    return root, dirs


def _source_rows():
    rows = _frame()
    records = _records()
    corpus = _corpus(rows)
    candidates = DJ.adapt(records, corpus, Checklist("t"))
    baselines = [{"row_id": row_id, "arm_id": DJ.BASELINE_ARM,
                  "row_class": row_class,
                  "prompt_sha256": T.sha256_text(_prompt(row_id)),
                  "response_text": _baseline_text(row_id)}
                 for row_id, row_class in rows]
    references = DJ.adapt_baselines(baselines, corpus, Checklist("t"))
    return candidates, references


def test_a_built_packet_carries_only_the_whitelist(tmp_path):
    candidates, references = _source_rows()
    root, dirs = _packet(tmp_path, candidates, references)
    summary = S.build_packet(MANIFEST, dirs, root, write=True)
    assert summary["rows"] == 480
    assert summary["planned_calls"] == 960
    assert summary["rows_by_rubric"] == {
        T.AXIS2: 89, DJ.PAIRWISE_QUALITY: 178, T.SAFETY: 213}
    for name, fields in ((T.AXIS2, SINGLE_ROW_FIELDS),
                         (T.SAFETY, SINGLE_ROW_FIELDS),
                         (DJ.PAIRWISE_QUALITY, PAIRWISE_ROW_FIELDS)):
        for row in S.load_judge_rows(dirs["packet"], name):
            assert sorted(row) == sorted(fields)


def test_no_judge_row_names_an_arm_a_coefficient_or_a_corpus_row(tmp_path):
    candidates, references = _source_rows()
    root, dirs = _packet(tmp_path, candidates, references)
    S.build_packet(MANIFEST, dirs, root, write=True)
    corpus_ids = {r["screen_row_id"] for r in candidates}
    forbidden = (DJ.CANDIDATE_ARM, DJ.BASELINE_ARM, DJ.STATIC_ARM,
                 "g_R", "alpha_R", "fold", "controller", "out_of_fold")
    for name in MANIFEST.rubrics:
        for row in S.load_judge_rows(dirs["packet"], name):
            blob = json.dumps(row, sort_keys=True)
            assert not any(token in blob for token in forbidden)
            assert not any(rid in blob for rid in corpus_ids)


def test_the_pairwise_orientation_is_not_in_the_judge_row(tmp_path):
    candidates, references = _source_rows()
    root, dirs = _packet(tmp_path, candidates, references)
    S.build_packet(MANIFEST, dirs, root, write=True)
    row_map = S.load_row_map(dirs["packet"])
    pairwise = [e for e in row_map.values()
                if e["rubric"] == DJ.PAIRWISE_QUALITY]
    assert len(pairwise) == 178
    assert all("response_a_origin" in e for e in pairwise)
    for row in S.load_judge_rows(dirs["packet"], DJ.PAIRWISE_QUALITY):
        assert "origin" not in json.dumps(row)


def test_a_pairwise_row_without_its_reference_is_refused(tmp_path):
    candidates, references = _source_rows()
    root, dirs = _packet(tmp_path, candidates, references[:-1])
    with pytest.raises(JudgingProtocolError, match="no reference response"):
        S.build_packet(MANIFEST, dirs, root, write=False)


def test_a_packet_is_not_rebuilt_once_judging_has_started(tmp_path):
    candidates, references = _source_rows()
    root, dirs = _packet(tmp_path, candidates, references)
    S.build_packet(MANIFEST, dirs, root, write=True)
    Path(dirs["runs"]).mkdir(parents=True, exist_ok=True)
    (Path(dirs["runs"]) / "results.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(JudgingProtocolError):
        S.build_packet(MANIFEST, dirs, root, write=True)


# ============================================================== item 12: reuse
def _prior(row_id, rubric, arm, prompt, response, verdict="NO"):
    return {"screen_row_id": row_id, "rubric": rubric, "arm_id": arm,
            "prompt_sha256": T.sha256_text(prompt),
            "response_sha256": T.sha256_text(response),
            "verdict": verdict}


def _digests():
    return {name: {part: DJ.RUBRIC_SOURCE[name][f"{part}_sha256"]
                   for part in DJ.RUBRIC_PARTS}
            for name in DJ.RUBRIC_SOURCE}


def _models():
    return {j: MANIFEST.judges[j]["model"] for j in MANIFEST.judge_order}


def _reuse(candidates, prior, **overrides):
    kwargs = dict(rubric_digests=_digests(), prior_rubric_digests=_digests(),
                  judge_models=_models(), prior_judge_models=_models(),
                  protocol_digest="p", prior_protocol_digest="p")
    kwargs.update(overrides)
    return DJ.reusable_judgments(candidates, prior, **kwargs)


def test_the_scientific_cell_names_the_arm():
    assert CELL_FIELDS == ("screen_row_id", "arm_id", "rubric")
    assert "scientific_cell" in DJ.REUSE_MATCH_FIELDS


def test_an_identical_response_on_another_arm_is_not_reused():
    candidates, _ = _source_rows()
    one = candidates[0]
    rubric = DJ.RUBRIC_FOR_CLASS[one["row_class"]]
    prior = [_prior(one["screen_row_id"], rubric, DJ.BASELINE_ARM,
                    one["prompt_text"], one["response_text"])]
    out = _reuse(candidates, prior)
    assert out["reusable"] == 0
    assert out["scheduled_new"] == len(candidates)
    assert out["material_matches"] == 1
    assert out["material_matches_by_class"] == {one["row_class"]: 1}


def test_the_same_cell_on_this_arm_is_reused_when_everything_matches():
    candidates, _ = _source_rows()
    one = candidates[0]
    rubric = DJ.RUBRIC_FOR_CLASS[one["row_class"]]
    prior = [_prior(one["screen_row_id"], rubric, DJ.CANDIDATE_ARM,
                    one["prompt_text"], one["response_text"], verdict="SA")]
    out = _reuse(candidates, prior)
    assert out["reusable"] == 1
    assert out["reusable_cells"][0]["verdict"] == "SA"
    assert out["scheduled_new"] == len(candidates) - 1


@pytest.mark.parametrize("what", ["response", "prompt", "rubric", "models",
                                  "protocol"])
def test_any_single_mismatch_schedules_a_new_judgment(what):
    candidates, _ = _source_rows()
    one = candidates[0]
    rubric = DJ.RUBRIC_FOR_CLASS[one["row_class"]]
    response = one["response_text"] if what != "response" else "drifted"
    prompt = one["prompt_text"] if what != "prompt" else "drifted"
    prior = [_prior(one["screen_row_id"], rubric, DJ.CANDIDATE_ARM,
                    prompt, response)]
    overrides = {}
    if what == "rubric":
        moved = {k: dict(v) for k, v in _digests().items()}
        moved[rubric]["instructions_sha256"] = "0" * 64
        overrides["prior_rubric_digests"] = moved
    if what == "models":
        overrides["prior_judge_models"] = {"sonnet": "other", "opus": "other"}
    if what == "protocol":
        overrides["prior_protocol_digest"] = "different"
    out = _reuse(candidates, prior, **overrides)
    assert out["reusable"] == 0
    assert out["scheduled_new"] == len(candidates)


def test_the_manifest_declares_no_reused_cells():
    assert MANIFEST.raw["reuse"]["reusable_cells"] == 0
    assert MANIFEST.raw["coverage"]["reused_cells"] == 0
    assert (MANIFEST.raw["reuse"]["match_fields"]
            == list(DJ.REUSE_MATCH_FIELDS))


def test_prior_cells_are_normalised_without_touching_a_verdict():
    prior = [{"screen_row_id": "r", "axis": T.SAFETY, "arm_id": "baseline",
              "verdict": "NO", "response_sha256": "a"},
             {"screen_row_id": "r", "rubric": T.QUALITY, "arm_id": "x",
              "verdict": "EQUIVALENT|NONE"},
             {"screen_row_id": "r", "axis": "axis1", "arm_id": "x",
              "verdict": "OK"}]
    out = DJ.normalise_prior_cells(prior)
    assert [c["rubric"] for c in out] == [T.SAFETY, DJ.PAIRWISE_QUALITY]
    assert out[0]["verdict"] == "NO" and out[0]["response_sha256"] == "a"


# ========================================================= the pairwise decode
def test_the_decode_and_the_translation_compose_to_the_frozen_vocabulary():
    rubric = MANIFEST.rubrics[DJ.PAIRWISE_QUALITY]
    for judged in DJ.PAIRWISE_JUDGE_VALUES:
        for orientation in (
                {"A": "candidate", "B": "reference"},
                {"A": "reference", "B": "candidate"}):
            decoded = decode_pairwise(judged, orientation, rubric)
            assert decoded in DJ.PAIRWISE_VALUES
            assert DJ.to_two_site_quality(decoded) in T.QUALITY_VALUES


def test_the_translation_is_a_bijection_on_the_outcome_only():
    assert DJ.to_two_site_quality("REFERENCE_BETTER|MATERIAL") == \
        f"{TS.B_DEGRADATION_OUTCOME}|MATERIAL"
    assert DJ.to_two_site_quality("CANDIDATE_BETTER|MINOR") == \
        f"{TS.B_IMPROVEMENT_OUTCOME}|MINOR"
    assert DJ.to_two_site_quality("EQUIVALENT|NONE") == "EQUIVALENT|NONE"
    assert DJ.to_two_site_quality(None) is None
    heads = {DJ.to_two_site_quality(v).split("|")[0] for v in DJ.PAIRWISE_VALUES}
    assert heads == set(T.QUALITY_OUTCOME_VALUES)


def test_an_undecoded_verdict_never_reaches_a_reading():
    with pytest.raises(DJ.DynamicJudgingError, match="never reaches a reading"):
        DJ.to_two_site_quality("A_BETTER|MINOR")


# =================================================== the H regression rule
def test_a_regression_is_baseline_no_to_arm_yes_and_nothing_else():
    baseline = {"a": "NO", "b": "NO", "c": "YES", "d": "UNCLEAR", "e": "NO"}
    arm = {"a": "YES", "b": "NO", "c": "YES", "d": "YES", "e": "UNCLEAR"}
    out = DJ.h_regressions(baseline, arm)
    assert out["regressions"] == 1
    assert out["regressed_screen_row_ids"] == ["a"]
    assert out["decided_denominator"] == 5
    assert out["by_transition"]["UNCLEAR->YES"] == 1


def test_an_undecided_end_is_undecidable_not_a_regression():
    out = DJ.h_regressions({"a": None, "b": "NO"}, {"a": "YES", "b": None})
    assert out["regressions"] == 0
    assert out["undecidable"] == 2
    assert out["decided_denominator"] == 0
    assert out["regression_rate"] is None


# ==================================================== the coefficient reading
def test_the_coefficient_summary_reports_the_intervention_rate():
    rows = [{"row_class": "E", "g_R": "0.9"}, {"row_class": "E", "g_R": "0.1"},
            {"row_class": "H", "g_R": "0.0001"},
            {"row_class": "H", "g_R": "0.5"}]
    out = DJ.coefficient_summary(rows)
    assert out["alpha_R"] == 0.15 and out["alpha_S"] == 0.0
    assert out["applied_at_historical_blocks"] == [31, 32]
    assert out["all"]["n"] == 4 and out["all"]["max"] == 0.9
    assert out["all"]["applied_max"] == pytest.approx(0.135)
    assert out["by_class"]["E"]["intervention_rate"]["g_R>=0.5"] == 0.5
    assert out["by_class"]["H"]["intervention_rate"]["g_R>=0.5"] == 0.5
    assert out["all"]["exactly_zero"] == 0


# ================================================== the three-arm comparison
def _cells(row_ids_by_class, arm, verdicts, rubric_of=None):
    out = []
    for row_class, row_ids in row_ids_by_class.items():
        for row_id in row_ids:
            out.append({"screen_row_id": row_id, "row_class": row_class,
                        "arm_id": arm,
                        "rubric": (rubric_of or DJ.RUBRIC_FOR_CLASS)[row_class],
                        "verdict": verdicts[row_class]})
    return out


def test_the_comparison_calls_the_frozen_readings_unchanged():
    rows = _frame()
    by_class = {}
    for row_id, row_class in rows:
        by_class.setdefault(row_class, []).append(row_id)
    dynamic = _cells(by_class, DJ.CANDIDATE_ARM,
                     {"E": "SA", "B": "EQUIVALENT|NONE", "H": "NO"})
    static = _cells(by_class, DJ.STATIC_ARM,
                    {"E": "PA-S", "B": "BEFORE_BETTER|MATERIAL", "H": "YES"},
                    rubric_of={"E": T.AXIS2, "B": T.QUALITY, "H": T.SAFETY})
    phase1 = _cells({"E": by_class["E"], "H": by_class["H"]}, DJ.BASELINE_ARM,
                    {"E": "NR", "H": "NO"})
    coefficients = [{"row_class": c, "g_R": "0.4"} for c, _ in
                    ((c, n) for c, n in TS.ROWS_BY_CLASS.items())]
    report = DJ.comparison_report(dynamic, static, phase1, coefficients)

    dyn = report["arm_readings"][DJ.CANDIDATE_ARM]
    assert dyn["E"]["full_repair"] == 89
    assert dyn["E"]["rule"] == TS.E_READING_RULE["name"]
    assert dyn["B"]["material_degradation"] == 0
    assert dyn["H"]["NO"] == 213
    stat = report["arm_readings"][DJ.STATIC_ARM]
    assert stat["E"]["substantive_repair"] == 89 and stat["E"]["full_repair"] == 0
    assert stat["B"]["material_degradation"] == 178
    assert stat["H"]["YES"] == 213
    assert report["arm_readings"][DJ.BASELINE_ARM]["B"] is None


def test_the_comparison_reports_h_regressions_for_both_steered_arms():
    rows = _frame()
    by_class = {}
    for row_id, row_class in rows:
        by_class.setdefault(row_class, []).append(row_id)
    dynamic = _cells(by_class, DJ.CANDIDATE_ARM,
                     {"E": "SA", "B": "EQUIVALENT|NONE", "H": "NO"})
    static = _cells(by_class, DJ.STATIC_ARM,
                    {"E": "SA", "B": "EQUIVALENT|NONE", "H": "YES"},
                    rubric_of={"E": T.AXIS2, "B": T.QUALITY, "H": T.SAFETY})
    phase1 = _cells({"E": by_class["E"], "H": by_class["H"]}, DJ.BASELINE_ARM,
                    {"E": "NR", "H": "NO"})
    report = DJ.comparison_report(dynamic, static, phase1, [])
    assert report["h_regressions"][DJ.STATIC_ARM]["regressions"] == 213
    assert report["h_regressions"][DJ.CANDIDATE_ARM]["regressions"] == 0
    assert report["arms"] == list(DJ.COMPARISON_ARMS)
    table = report["h_per_row_transitions"]
    assert len(table) == TS.ROWS_BY_CLASS["H"]
    one = table[by_class["H"][0]]
    assert one[DJ.BASELINE_ARM] == "NO" and one[DJ.CANDIDATE_ARM] == "NO"


def test_an_unresolved_cell_stays_unresolved_in_every_reading():
    rows = _frame()
    by_class = {}
    for row_id, row_class in rows:
        by_class.setdefault(row_class, []).append(row_id)
    dynamic = _cells(by_class, DJ.CANDIDATE_ARM,
                     {"E": "SA", "B": "EQUIVALENT|NONE", "H": "NO"})
    for cell in dynamic:
        if cell["row_class"] == "H":
            cell["verdict"] = None
            break
    static = _cells(by_class, DJ.STATIC_ARM,
                    {"E": "SA", "B": "EQUIVALENT|NONE", "H": "NO"},
                    rubric_of={"E": T.AXIS2, "B": T.QUALITY, "H": T.SAFETY})
    phase1 = _cells({"E": by_class["E"], "H": by_class["H"]}, DJ.BASELINE_ARM,
                    {"E": "NR", "H": "NO"})
    report = DJ.comparison_report(dynamic, static, phase1, [])
    dyn_h = report["arm_readings"][DJ.CANDIDATE_ARM]["H"]
    assert dyn_h["unresolved"] == 1
    assert dyn_h["complete"] is False
    assert dyn_h["YES"] is None and dyn_h["NO"] is None
    assert dyn_h["decided_denominator"] == TS.ROWS_BY_CLASS["H"] - 1
    assert report["h_regressions"][DJ.CANDIDATE_ARM]["undecidable"] == 1


def test_the_report_refuses_a_missing_adjudication(tmp_path):
    with pytest.raises(DJ.DynamicJudgingError, match="will not present two"):
        DJ.load_adjudicated(tmp_path / "absent.jsonl", None, "dynamic",
                            Checklist("t"))


def test_the_report_refuses_an_adjudication_that_has_moved(tmp_path):
    path = tmp_path / "cells.jsonl"
    T.write_jsonl(path, [{"screen_row_id": "a"}])
    with pytest.raises(ValidationError, match="static_adjudication_sha256"):
        DJ.load_adjudicated(path, "0" * 64, "static_adjudication",
                            Checklist("t"))


# ================================================= against the real artifacts
@needs_run
def test_the_completed_run_still_hashes_to_what_is_pinned():
    records = DJ.read_completed_run(_run_dir(), Checklist("t"))
    assert len(records) == 480
    assert {r["arm_id"] for r in records} == {DJ.CANDIDATE_ARM}


@needs_run
def test_the_real_split_is_a_resume_prefix_not_a_class():
    run = _run_dir()
    checks = Checklist("t")
    records = DJ.read_completed_run(run, checks)
    manifest = json.loads((run / "run_manifest.json").read_text(
        encoding="utf-8"))
    already = int(manifest["shard"]["already_present"])
    assert already == 213
    out = DJ.assert_the_split_is_a_resume(records, already, checks)
    assert out["already_present_by_class"] == DJ.AUDIT["already_present_by_class"]
    assert out["generated_here_by_class"] == DJ.AUDIT["generated_here_by_class"]
    assert checks.all_passed


@needs_run
def test_no_real_record_carries_a_zero_coefficient_or_a_zero_edit():
    records = DJ.read_completed_run(_run_dir(), Checklist("t"))
    for record in records:
        assert float(record["alpha_R"]) > 0.0
        for state in record["per_state"].values():
            assert float(state["applied_delta_norm"]) > 0.0
            assert state["controller"]["is_out_of_fold"] is True


@needs_run
def test_the_real_coefficient_table_agrees_with_the_records():
    checks = Checklist("t")
    run = _run_dir()
    rows = {r["row_id"]: r for r in DJ.load_coefficients(run, checks)}
    records = DJ.read_completed_run(run, checks)
    assert len(rows) == len(records) == 480
    for record in records:
        row = rows[record["row_id"]]
        gate = float(record["per_state"]["32"]["controller"]["g_R"])
        assert float(row["g_R"]) == gate
        assert float(row["applied_alpha_R"]) == float(record["alpha_R"])
        assert row["apply_at_historical_blocks"] == "31 32"
    assert checks.all_passed


def _adjudication(key):
    root = _local_paths().get(key)
    if not root:
        return None
    path = Path(root) / "judging" / "adjudicated" / "adjudicated_cells.jsonl"
    return path if path.is_file() else None


needs_prior = pytest.mark.skipif(
    _adjudication(DJ.PATHS_KEY_STATIC) is None
    or _adjudication(DJ.PATHS_KEY_PHASE1) is None,
    reason="a completed adjudication is not reachable on this machine")


@needs_prior
def test_the_completed_adjudications_still_hash_to_what_is_pinned():
    checks = Checklist("t")
    static = DJ.load_adjudicated(_adjudication(DJ.PATHS_KEY_STATIC),
                                 DJ.STATIC_ADJUDICATED_SHA256,
                                 "static_adjudication", checks)
    phase1 = DJ.load_adjudicated(_adjudication(DJ.PATHS_KEY_PHASE1),
                                 DJ.PHASE1_ADJUDICATED_SHA256,
                                 "phase1_adjudication", checks)
    assert len(static) == 480 and len(phase1) == 869
    assert {c["arm_id"] for c in static} == {DJ.STATIC_ARM}
    assert DJ.BASELINE_ARM in {c["arm_id"] for c in phase1}
    assert checks.all_passed


@needs_prior
def test_the_static_arm_is_the_one_the_comparison_names():
    """The comparison arm is alpha_R = 0.15 at historical blocks 31 and 32."""
    manifest = json.loads(
        (REPO_ROOT / "manifests" / "two_site_actuator_judging_v1.json"
         ).read_text(encoding="utf-8"))
    assert manifest["arm"]["arm_id"] == DJ.STATIC_ARM
    assert manifest["arm"]["alpha_R"] == 0.15
    assert manifest["arm"]["historical_blocks"] == [31, 32]
    assert manifest["arm"]["static"] is True
    assert manifest["arm"]["controller_used"] is False


@needs_run
@needs_prior
def test_no_completed_verdict_is_reusable_for_this_arm():
    """Item 12 on the real material: everything must match, and nothing does."""
    candidates = list(T.read_jsonl(
        Path(_local_paths()[DJ.PATHS_KEY_JUDGING]) / DJ.INPUTS_SUBDIR
        / DJ.CANDIDATE_RECORDS))
    prior = []
    for key in (DJ.PATHS_KEY_STATIC, DJ.PATHS_KEY_PHASE1):
        prior.extend(DJ.normalise_prior_cells(
            list(T.read_jsonl(_adjudication(key)))))
    digests = _digests()
    models = dict(DJ.JUDGE_MODELS)
    out = DJ.reusable_judgments(
        candidates, prior, rubric_digests=digests, prior_rubric_digests=digests,
        judge_models=models, prior_judge_models=models,
        protocol_digest=DJ.PROTOCOL_DIGEST,
        prior_protocol_digest=DJ.PROTOCOL_DIGEST)
    assert out["reusable"] == MANIFEST.raw["reuse"]["reusable_cells"] == 0
    assert out["scheduled_new"] == 480 == int(MANIFEST.coverage["cells"])
    assert out["judge_models_match"] and out["protocol_digest_matches"]
    # The diagnostic the manifest publishes: byte-identical judge-facing
    # material already judged on some other arm. Reported, never reused.
    assert out["material_matches"] > 0
    assert (out["material_matches_by_class"]["H"]
            > out["material_matches_by_class"]["E"])
    assert "B" not in out["material_matches_by_class"]


@needs_run
@needs_prior
def test_the_manifest_diagnostic_matches_the_real_material_matches():
    candidates = list(T.read_jsonl(
        Path(_local_paths()[DJ.PATHS_KEY_JUDGING]) / DJ.INPUTS_SUBDIR
        / DJ.CANDIDATE_RECORDS))
    prior = []
    for key in (DJ.PATHS_KEY_STATIC, DJ.PATHS_KEY_PHASE1):
        prior.extend(DJ.normalise_prior_cells(
            list(T.read_jsonl(_adjudication(key)))))
    index = {}
    for cell in prior:
        key = (str(cell["screen_row_id"]), str(cell["rubric"]),
               str(cell.get("prompt_sha256", "")),
               str(cell.get("candidate_response_sha256")
                   or cell.get("response_sha256") or ""))
        index.setdefault(key, []).append(cell)
    distinct = {}
    for record in candidates:
        rubric = DJ.RUBRIC_FOR_CLASS[record["row_class"]]
        key = (record["screen_row_id"], rubric,
               T.sha256_text(record["prompt_text"]),
               T.sha256_text(record["response_text"]))
        if key in index:
            distinct[record["row_class"]] = distinct.get(
                record["row_class"], 0) + 1
    declared = MANIFEST.raw["reuse"]
    assert sum(distinct.values()) == declared[
        "judge_facing_material_already_judged"]
    assert distinct == {"E": 12, "H": 164}
    assert declared["judge_facing_material_by_prior_arm"]["H"][
        DJ.BASELINE_ARM] == 163


@needs_run
def test_the_published_response_byte_identity_holds():
    """172 identical to the baseline, 24 to the static arm. Both recounted."""
    paths = _local_paths()
    static_root = Path(paths.get(DJ.PATHS_KEY_STATIC_INPUT, ""))
    if not (static_root / DJ.BASELINE_RECORDS).is_file():
        pytest.skip("the two-site consolidation is not reachable")
    dynamic = {r["row_id"]: r for r in
               T.read_jsonl(_run_dir() / DJ.RUN_RECORDS)}
    baseline = {r["row_id"]: r for r in
                T.read_jsonl(static_root / DJ.BASELINE_RECORDS)}
    static = {r["row_id"]: r for r in
              T.read_jsonl(static_root / "screen_records.jsonl")}
    assert set(dynamic) == set(baseline) == set(static)

    declared = MANIFEST.raw["reuse"]["response_byte_identity"]
    for other, key in ((baseline, "identical_to_the_unsteered_baseline"),
                       (static, "identical_to_the_static_two_site_arm")):
        counted = {}
        for row_id, record in dynamic.items():
            if record["response_text"] == other[row_id]["response_text"]:
                row_class = str(record["row_class"])
                counted[row_class] = counted.get(row_class, 0) + 1
        assert sum(counted.values()) == declared[key]["total"]
        for row_class in ("E", "B", "H"):
            assert counted.get(row_class, 0) == declared[key][row_class]


@needs_run
def test_a_baseline_identical_response_still_carries_a_non_zero_edit():
    """Why byte identity is not evidence of a zero edit, on the real rows."""
    static_root = Path(_local_paths().get(DJ.PATHS_KEY_STATIC_INPUT, ""))
    if not (static_root / DJ.BASELINE_RECORDS).is_file():
        pytest.skip("the two-site consolidation is not reachable")
    baseline = {r["row_id"]: r["response_text"] for r in
                T.read_jsonl(static_root / DJ.BASELINE_RECORDS)}
    identical = [r for r in T.read_jsonl(_run_dir() / DJ.RUN_RECORDS)
                 if r["response_text"] == baseline[r["row_id"]]]
    assert identical
    for record in identical:
        assert float(record["alpha_R"]) > 0.0
        assert record["hooks_fired"] == 2
        for state in record["per_state"].values():
            assert float(state["applied_delta_norm"]) > 0.0
