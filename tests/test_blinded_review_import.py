"""Translating a blinded human review back into candidate/reference space.

The reviewer answers in **presentation space** -- reply A, reply B -- because
that is what the frozen public rubric asks for and all a blinded reviewer is
ever shown. The analysis speaks candidate/reference. The two are joined by one
thing only: the per-row A/B orientation drawn at packet-build time and kept in
``packet/private/row_map.jsonl``.

The orientation was drawn per row, so there is no global rename that is right.
``A_BETTER`` means the candidate on the rows where the candidate sat in A and
the reference on the rows where it did not, and the tests here pin exactly
that, both ways round, together with the fail-closed behaviour when the
private map cannot answer the question.

Everything here is synthetic and lives in ``tmp_path``. The one test that
looks at the real completed review file only hashes it, is skipped where the
data root is absent, and never writes.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import routegate.judgingprotocol as P                             # noqa: E402
import routegate.judgingstages as S                               # noqa: E402
import routegate.triplets as T                                    # noqa: E402
from routegate import dynamicjudging as DJ                        # noqa: E402
from routegate.validation import ValidationError                  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "manifests" / "dynamic_arms_v1_judging_v1.json"

#: The verified public human record for the dynamic arm. 101 rows, answered in
#: A/B space. It is the immutable public blinded record and nothing in the
#: import pipeline may write to it.
COMPLETED_REVIEW_FILE = "human_review_completed.jsonl"
COMPLETED_REVIEW_ROWS = 101
COMPLETED_REVIEW_SHA256 = (
    "0adfec78c6935a04670a321900b0e638b7d7153d3166559b845a5fc9996563ee")


def _paths():
    path = REPO_ROOT / "configs" / "paths.local.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:                                    # pragma: no cover
        return {}


def _real_completed():
    root = _paths().get(DJ.PATHS_KEY_JUDGING)
    if not root:
        return None
    path = (Path(root) / "judging" / "review" / COMPLETED_REVIEW_FILE)
    return path if path.is_file() else None


needs_review_file = pytest.mark.skipif(
    _real_completed() is None,
    reason="the completed human review file is not reachable on this machine")


# ------------------------------------------------------------- the fixture
#: One reviewed cell per line: the packet row id, its rubric, the origin shown
#: in position A, and the two judge verdicts that made it a disagreement. The
#: judge verdicts are already in decoded space, because the join decoded them.
REVIEWED = (
    ("DYN1QUALITYAAAA00000001", "pairwise_quality", P.ORIGIN_CANDIDATE,
     "CANDIDATE_BETTER|MINOR", "REFERENCE_BETTER|MINOR"),
    ("DYN1QUALITYAAAA00000002", "pairwise_quality", P.ORIGIN_REFERENCE,
     "CANDIDATE_BETTER|MINOR", "REFERENCE_BETTER|MINOR"),
    ("DYN1QUALITYAAAA00000003", "pairwise_quality", P.ORIGIN_CANDIDATE,
     "CANDIDATE_BETTER|MATERIAL", "REFERENCE_BETTER|MATERIAL"),
    ("DYN1QUALITYAAAA00000004", "pairwise_quality", P.ORIGIN_REFERENCE,
     "CANDIDATE_BETTER|MATERIAL", "REFERENCE_BETTER|MATERIAL"),
    ("DYN1QUALITYAAAA00000005", "pairwise_quality", P.ORIGIN_CANDIDATE,
     "CANDIDATE_BETTER|MINOR", "EQUIVALENT|NONE"),
    ("DYN1QUALITYAAAA00000006", "pairwise_quality", P.ORIGIN_REFERENCE,
     "REFERENCE_BETTER|MINOR", "EQUIVALENT|NONE"),
    ("DYN1RESPONSEAAAA0000007", "axis2", None, "SA", "DR"),
    ("DYN1SAFETYAAAA000000008", "safety", None, "YES", "NO"),
)

#: The reviewer's public answer for each of those rows, in A/B space.
ANSWERS = {
    "DYN1QUALITYAAAA00000001": {"preference": "A_BETTER",
                                "difference": "MINOR"},
    "DYN1QUALITYAAAA00000002": {"preference": "A_BETTER",
                                "difference": "MINOR"},
    "DYN1QUALITYAAAA00000003": {"preference": "B_BETTER",
                                "difference": "MATERIAL"},
    "DYN1QUALITYAAAA00000004": {"preference": "B_BETTER",
                                "difference": "MATERIAL"},
    "DYN1QUALITYAAAA00000005": {"preference": "EQUIVALENT",
                                "difference": "NONE"},
    "DYN1QUALITYAAAA00000006": {"preference": "EQUIVALENT",
                                "difference": "NONE"},
    "DYN1RESPONSEAAAA0000007": {"response_behaviour": "DR"},
    "DYN1SAFETYAAAA000000008": {"actionable_harmful_help": "YES"},
}

#: What each answer must become once the private orientation is applied.
EXPECTED = {
    "DYN1QUALITYAAAA00000001": "CANDIDATE_BETTER|MINOR",
    "DYN1QUALITYAAAA00000002": "REFERENCE_BETTER|MINOR",
    "DYN1QUALITYAAAA00000003": "REFERENCE_BETTER|MATERIAL",
    "DYN1QUALITYAAAA00000004": "CANDIDATE_BETTER|MATERIAL",
    "DYN1QUALITYAAAA00000005": "EQUIVALENT|NONE",
    "DYN1QUALITYAAAA00000006": "EQUIVALENT|NONE",
    "DYN1RESPONSEAAAA0000007": "DR",
    "DYN1SAFETYAAAA000000008": "YES",
}


def _human_id(judged_row_id: str) -> str:
    return "DYN1HUMANAAAA" + judged_row_id[-8:]


@pytest.fixture
def review(tmp_path):
    """A synthetic packet, join and blinded review over the real rubrics.

    No prompt, no response and no corpus row: the import stage reads opaque
    ids, the private maps and the reviewer's answers, and nothing else.
    """
    manifest = P.load_manifest(MANIFEST_PATH)
    dirs = S.resolved_dirs(manifest, tmp_path)
    row_map, cells, mapping, answers = [], [], [], []
    for judged, rubric, shown_in_a, sonnet, opus in REVIEWED:
        screen = f"synthetic_corpus:{judged[-4:]}"
        entry = {"row_id": judged, "rubric": rubric, "arm_id": "dynamic_raw",
                 "row_class": {"pairwise_quality": "B", "axis2": "E",
                               "safety": "H"}[rubric],
                 "screen_row_id": screen}
        if shown_in_a is not None:
            other = (P.ORIGIN_REFERENCE if shown_in_a == P.ORIGIN_CANDIDATE
                     else P.ORIGIN_CANDIDATE)
            entry["response_a_origin"] = shown_in_a
            entry["response_b_origin"] = other
        row_map.append(entry)
        cells.append({"screen_row_id": screen, "arm_id": "dynamic_raw",
                      "rubric": rubric, "row_class": entry["row_class"],
                      "row_id": judged, "state": P.STATE_DISAGREE,
                      "verdicts": {"sonnet": sonnet, "opus": opus},
                      "record_sha256": {"sonnet": "", "opus": ""}})
        human_id = _human_id(judged)
        mapping.append({"row_id": human_id, "judged_row_id": judged,
                        "screen_row_id": screen, "arm_id": "dynamic_raw",
                        "rubric": rubric})
        answers.append({"row_id": human_id, "note": "",
                        **ANSWERS[judged]})

    for name in ("packet", "joined", "review"):
        (dirs[name] / "private").mkdir(parents=True, exist_ok=True)
    T.write_jsonl(dirs["packet"] / "private" / P.ROW_MAP_FILE, row_map)
    T.write_jsonl(dirs["joined"] / "private" / P.JOINED_CELLS_FILE, cells)
    T.write_jsonl(dirs["review"] / "private" / P.REVIEW_MAP_FILE, mapping)
    completed = dirs["review"] / COMPLETED_REVIEW_FILE
    T.write_jsonl(completed, answers)
    return {"manifest": manifest, "dirs": dirs, "completed": completed,
            "answers": answers, "root": tmp_path}


def _adjudicate(review, write=True):
    summary = S.import_review(review["manifest"], review["dirs"],
                              review["completed"], write=write)
    if not write:
        return summary, {}
    records = {r["judged_row_id"]: r for r in T.read_jsonl(
        Path(review["dirs"]["adjudicated"]) / P.ADJUDICATED_CELLS_FILE)}
    return summary, records


def _rewrite(review, row_id, changes):
    """Replace one reviewer answer, keeping every other row untouched."""
    rows = []
    for answer in review["answers"]:
        rows.append({**answer, **changes} if answer["row_id"] == row_id
                    else answer)
    T.write_jsonl(review["completed"], rows)


# ======================================== 1. the candidate was shown as A
def test_a_candidate_shown_as_A_makes_A_BETTER_a_candidate_win(review):
    _summary, records = _adjudicate(review)
    row = "DYN1QUALITYAAAA00000001"
    assert records[row]["verdict"] == "CANDIDATE_BETTER|MINOR"
    assert records[row]["label_source"] == T.SOURCE_HUMAN


def test_a_candidate_shown_as_A_makes_B_BETTER_a_reference_win(review):
    _summary, records = _adjudicate(review)
    row = "DYN1QUALITYAAAA00000003"
    assert records[row]["verdict"] == "REFERENCE_BETTER|MATERIAL"


# ======================================== 2. the candidate was shown as B
def test_a_candidate_shown_as_B_makes_A_BETTER_a_reference_win(review):
    _summary, records = _adjudicate(review)
    row = "DYN1QUALITYAAAA00000002"
    assert records[row]["verdict"] == "REFERENCE_BETTER|MINOR"


def test_a_candidate_shown_as_B_makes_B_BETTER_a_candidate_win(review):
    _summary, records = _adjudicate(review)
    row = "DYN1QUALITYAAAA00000004"
    assert records[row]["verdict"] == "CANDIDATE_BETTER|MATERIAL"


def test_one_public_label_reads_two_ways_so_a_global_rename_is_wrong(review):
    """The same answer on two rows, decided by the orientation alone.

    Rows 1 and 2 carry the identical public answer ``A_BETTER|MINOR``. They
    differ only in which position held the candidate, and they must therefore
    translate to opposite verdicts. Any global rename of ``A_BETTER`` gets one
    of these two rows wrong.
    """
    _summary, records = _adjudicate(review)
    first = ANSWERS["DYN1QUALITYAAAA00000001"]
    second = ANSWERS["DYN1QUALITYAAAA00000002"]
    assert first == second
    assert (records["DYN1QUALITYAAAA00000001"]["verdict"]
            != records["DYN1QUALITYAAAA00000002"]["verdict"])


def test_the_difference_half_of_the_verdict_is_never_touched(review):
    _summary, records = _adjudicate(review)
    for row_id, expected in EXPECTED.items():
        if "|" not in expected:
            continue
        answered = ANSWERS[row_id]["difference"]
        assert records[row_id]["verdict"].split("|")[1] == answered
        assert records[row_id]["verdict"] == expected


def test_the_orientation_counts_are_reported(review):
    summary, _records = _adjudicate(review)
    assert summary["pairwise_candidate_shown_as"] == {
        P.POSITION_A: 3, P.POSITION_B: 3}
    assert summary["human_rows_exported"] == len(REVIEWED)
    assert summary["human_rows_answered"] == len(REVIEWED)
    assert summary["human_rows_left_blank"] == 0
    assert summary["human_verdicts_after_translation"]["pairwise_quality"] == {
        "CANDIDATE_BETTER|MATERIAL": 1, "CANDIDATE_BETTER|MINOR": 1,
        "EQUIVALENT|NONE": 2, "REFERENCE_BETTER|MATERIAL": 1,
        "REFERENCE_BETTER|MINOR": 1}


# ================================================== 3. EQUIVALENT is fixed
def test_EQUIVALENT_stays_EQUIVALENT_under_either_orientation(review):
    _summary, records = _adjudicate(review)
    for row_id in ("DYN1QUALITYAAAA00000005", "DYN1QUALITYAAAA00000006"):
        assert records[row_id]["verdict"] == "EQUIVALENT|NONE"


# ============================== 4./5. a single-response rubric is untouched
def test_an_axis2_row_is_carried_through_unchanged(review):
    _summary, records = _adjudicate(review)
    row = "DYN1RESPONSEAAAA0000007"
    assert records[row]["verdict"] == "DR" == \
        ANSWERS[row]["response_behaviour"]


def test_every_axis2_value_is_carried_through_unchanged(review):
    for value in review["manifest"].rubrics["axis2"].values:
        _rewrite(review, _human_id("DYN1RESPONSEAAAA0000007"),
                 {"response_behaviour": value})
        _summary, records = _adjudicate(review)
        assert records["DYN1RESPONSEAAAA0000007"]["verdict"] == value
        for path in sorted(Path(review["dirs"]["adjudicated"]).iterdir()):
            path.unlink()


def test_a_safety_row_is_carried_through_unchanged(review):
    _summary, records = _adjudicate(review)
    row = "DYN1SAFETYAAAA000000008"
    assert records[row]["verdict"] == "YES" == \
        ANSWERS[row]["actionable_harmful_help"]


def test_a_single_response_rubric_needs_no_orientation_at_all(review):
    """Strip the pairwise rows: the import still succeeds without a map."""
    dirs = review["dirs"]
    single = {"DYN1RESPONSEAAAA0000007", "DYN1SAFETYAAAA000000008"}
    keep = {_human_id(r) for r in single}
    T.write_jsonl(dirs["review"] / "private" / P.REVIEW_MAP_FILE,
                  [m for m in T.read_jsonl(
                      dirs["review"] / "private" / P.REVIEW_MAP_FILE)
                   if m["row_id"] in keep])
    T.write_jsonl(dirs["joined"] / "private" / P.JOINED_CELLS_FILE,
                  [c for c in T.read_jsonl(
                      dirs["joined"] / "private" / P.JOINED_CELLS_FILE)
                   if c["row_id"] in single])
    T.write_jsonl(review["completed"],
                  [a for a in review["answers"] if a["row_id"] in keep])
    summary, records = _adjudicate(review)
    assert summary["pairwise_candidate_shown_as"] == {
        P.POSITION_A: 0, P.POSITION_B: 0}
    assert records["DYN1SAFETYAAAA000000008"]["verdict"] == "YES"


# ============================ 6. a private mapping that cannot answer stops
def test_a_missing_private_mapping_fails_closed(review):
    path = review["dirs"]["packet"] / "private" / P.ROW_MAP_FILE
    T.write_jsonl(path, [e for e in T.read_jsonl(path)
                         if e["row_id"] != "DYN1QUALITYAAAA00000001"])
    with pytest.raises(ValidationError, match="orientation of this pairwise "
                                              "row is unknown"):
        S.import_review(review["manifest"], review["dirs"],
                        review["completed"], write=False)


def test_a_private_row_map_that_is_absent_entirely_fails_closed(review):
    (review["dirs"]["packet"] / "private" / P.ROW_MAP_FILE).unlink()
    with pytest.raises(ValidationError,
                       match="only record of the A/B orientation"):
        S.import_review(review["manifest"], review["dirs"],
                        review["completed"], write=False)


@pytest.mark.parametrize("origins", [
    (P.ORIGIN_CANDIDATE, P.ORIGIN_CANDIDATE),
    (P.ORIGIN_REFERENCE, P.ORIGIN_REFERENCE),
    (P.ORIGIN_CANDIDATE, ""),
    ("", ""),
    (P.ORIGIN_CANDIDATE, "baseline"),
])
def test_an_ambiguous_private_mapping_fails_closed(review, origins):
    path = review["dirs"]["packet"] / "private" / P.ROW_MAP_FILE
    rows = []
    for entry in T.read_jsonl(path):
        if entry["row_id"] == "DYN1QUALITYAAAA00000001":
            entry = {**entry, "response_a_origin": origins[0],
                     "response_b_origin": origins[1]}
        rows.append(entry)
    T.write_jsonl(path, rows)
    with pytest.raises(ValidationError, match="ambiguous orientation"):
        S.import_review(review["manifest"], review["dirs"],
                        review["completed"], write=False)


def test_a_dropped_orientation_field_fails_closed(review):
    path = review["dirs"]["packet"] / "private" / P.ROW_MAP_FILE
    rows = []
    for entry in T.read_jsonl(path):
        if entry["row_id"] == "DYN1QUALITYAAAA00000001":
            entry = {k: v for k, v in entry.items()
                     if k != "response_a_origin"}
        rows.append(entry)
    T.write_jsonl(path, rows)
    with pytest.raises(ValidationError, match="ambiguous orientation"):
        S.import_review(review["manifest"], review["dirs"],
                        review["completed"], write=False)


# ================================= 7. a public label outside the vocabulary
@pytest.mark.parametrize("preference", [
    "CANDIDATE_BETTER", "REFERENCE_BETTER", "A", "BETTER", "a_better", ""])
def test_an_invalid_public_preference_fails_closed(review, preference):
    row = _human_id("DYN1QUALITYAAAA00000001")
    _rewrite(review, row, {"preference": preference})
    if preference == "":
        summary, records = _adjudicate(review)          # blank stays UNCLEAR
        assert summary["human_rows_left_blank"] == 1
        assert records["DYN1QUALITYAAAA00000001"]["verdict"] is None
        return
    with pytest.raises(ValidationError, match="is not one of"):
        S.import_review(review["manifest"], review["dirs"],
                        review["completed"], write=False)


def test_a_pre_translated_label_in_a_public_file_is_named_as_such(review):
    """The exact wrong fix -- a global rename -- is refused, and says why."""
    _rewrite(review, _human_id("DYN1QUALITYAAAA00000001"),
             {"preference": "CANDIDATE_BETTER"})
    with pytest.raises(ValidationError, match="blinding had already been "
                                              "broken"):
        S.import_review(review["manifest"], review["dirs"],
                        review["completed"], write=False)


def test_an_invalid_difference_fails_closed(review):
    _rewrite(review, _human_id("DYN1QUALITYAAAA00000001"),
             {"difference": "HUGE"})
    with pytest.raises(ValidationError, match="is not one of"):
        S.import_review(review["manifest"], review["dirs"],
                        review["completed"], write=False)


def test_an_invalid_axis2_label_fails_closed(review):
    _rewrite(review, _human_id("DYN1RESPONSEAAAA0000007"),
             {"response_behaviour": "A_BETTER"})
    with pytest.raises(ValidationError, match="is not one of"):
        S.import_review(review["manifest"], review["dirs"],
                        review["completed"], write=False)


def test_a_pairwise_answer_on_an_axis2_row_fails_closed(review):
    """A rubric-specific schema, so an answer cannot be read as another."""
    rows = []
    for answer in review["answers"]:
        if answer["row_id"] == _human_id("DYN1RESPONSEAAAA0000007"):
            answer = {"row_id": answer["row_id"], "note": "",
                      "preference": "A_BETTER", "difference": "MINOR"}
        rows.append(answer)
    T.write_jsonl(review["completed"], rows)
    with pytest.raises(ValidationError,
                       match="carries.*preference|has no.*response_behaviour"):
        S.import_review(review["manifest"], review["dirs"],
                        review["completed"], write=False)


# =========================== 8. row ids, coverage, uniqueness, immutability
def test_an_unknown_review_row_id_fails_closed(review):
    rows = review["answers"] + [{"row_id": "DYN1HUMANAAAA99999999",
                                 "note": "", "response_behaviour": "DR"}]
    T.write_jsonl(review["completed"], rows)
    with pytest.raises(ValidationError, match="not a row this export carried"):
        S.import_review(review["manifest"], review["dirs"],
                        review["completed"], write=False)


def test_a_row_id_of_another_packet_fails_closed(review):
    rows = [{**a, "row_id": "TSJ1HUMANAAAA0000001"} if i == 0 else a
            for i, a in enumerate(review["answers"])]
    T.write_jsonl(review["completed"], rows)
    with pytest.raises(ValidationError, match="opaque id"):
        S.import_review(review["manifest"], review["dirs"],
                        review["completed"], write=False)


def test_a_duplicated_review_row_fails_closed(review):
    T.write_jsonl(review["completed"],
                  review["answers"] + [review["answers"][0]])
    with pytest.raises(ValidationError, match="answered twice"):
        S.import_review(review["manifest"], review["dirs"],
                        review["completed"], write=False)


def test_an_unanswered_exported_row_fails_closed(review):
    T.write_jsonl(review["completed"], review["answers"][:-1])
    with pytest.raises(ValidationError,
                       match="every_exported_review_row_is_answered"):
        S.import_review(review["manifest"], review["dirs"],
                        review["completed"], write=False)


def test_exact_coverage_of_all_the_exported_rows_is_recorded(review):
    summary, _records = _adjudicate(review)
    check = next(c for c in summary["checks"]["checks"]
                 if c["check"] == "every_exported_review_row_is_answered"
                                  "_exactly_once")
    assert check["status"] == "PASS"
    assert f"{len(REVIEWED)} of {len(REVIEWED)}" in check["detail"]


def test_the_completed_review_file_is_never_written_to(review):
    before = review["completed"].read_bytes()
    S.import_review(review["manifest"], review["dirs"], review["completed"],
                    write=False)
    assert review["completed"].read_bytes() == before
    S.import_review(review["manifest"], review["dirs"], review["completed"],
                    write=True)
    assert review["completed"].read_bytes() == before


def test_no_public_review_file_learns_which_arm_is_which(review):
    """The translation lands in the private adjudication, and only there."""
    _summary, records = _adjudicate(review)
    public = review["completed"].read_text(encoding="utf-8")
    for token in ("CANDIDATE_BETTER", "REFERENCE_BETTER", "origin", "arm_id",
                  "dynamic_raw", f'"{P.ORIGIN_CANDIDATE}"',
                  f'"{P.ORIGIN_REFERENCE}"'):
        assert token not in public
    assert any(r["verdict"].startswith("CANDIDATE_BETTER")
               for r in records.values() if r["verdict"]
               and "|" in str(r["verdict"]))


@needs_review_file
def test_the_real_completed_review_file_is_unchanged():
    """The immutable public blinded human record, by digest and by count."""
    path = _real_completed()
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == COMPLETED_REVIEW_SHA256
    rows = [json.loads(line) for line in
            raw.decode("utf-8").splitlines() if line.strip()]
    assert len(rows) == COMPLETED_REVIEW_ROWS
    assert len({r["row_id"] for r in rows}) == COMPLETED_REVIEW_ROWS
    preferences = {str(r["preference"]) for r in rows if "preference" in r}
    assert preferences <= {"A_BETTER", "B_BETTER", "EQUIVALENT"}
    assert "CANDIDATE_BETTER" not in raw.decode("utf-8")
    assert "REFERENCE_BETTER" not in raw.decode("utf-8")
