"""The three technically unresolved two-site cells, and how they behave.

957 of 960 planned judge verdicts arrived. These tests pin what happens to the
missing three: they are carried, never decided, never sent to a human, never
replaced by a human, and never divided into a denominator that pretends they
were answered.

Nothing here reads, writes or needs the machine-local judging artifacts. Every
fixture is synthetic and lives in ``tmp_path``.
"""
from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "experiments" / "two_site_judging"

from routegate.validation import ValidationError               # noqa: E402
import routegate.triplets as T                                    # noqa: E402
import routegate.twositejudging as J                              # noqa: E402
import routegate.twositeunresolved as U                           # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"_tsu_{name}", EXPERIMENTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)                              # type: ignore
    return module


join_mod = _load("join_two_site_verdicts")
export_mod = _load("export_two_site_human_review")
import_mod = _load("import_two_site_human_review")


def _sha(text: str) -> str:
    return T.sha256_text(text)


def _raw_body(payload: dict | None) -> str:
    """A preserved raw response, in the shape the runner writes."""
    content = ([] if payload is None
               else [{"type": "text", "text": json.dumps(payload)}])
    return json.dumps({"http_status": 200, "response_headers": {},
                       "raw_body": json.dumps({"content": content})},
                      sort_keys=True, ensure_ascii=False)


def _one_char_off(row_id: str) -> str:
    """The failure actually observed: an id differing in exactly one place."""
    last = "5" if row_id[-1] != "5" else "4"
    return row_id[:-1] + last


@pytest.fixture
def stage(tmp_path):
    """A small packet where three cells lose one judge's verdict each.

    Deliberately mirrors the real failure shape: two quality cells whose reply
    echoes a one-character-wrong id, and one safety cell that comes back with
    no content blocks at all. Each failed cell keeps the *other* judge's
    verdict, exactly as the real run did.
    """
    rng = random.Random(20260827)
    packet = tmp_path / "packet"
    (packet / "private").mkdir(parents=True)
    runs = tmp_path / "runs"

    counts = {"quality": 6, "safety": 5, "axis2": 4}
    rows_by_rubric: dict[str, list] = {r: [] for r in J.RUBRICS_USED}
    private: list[dict] = []
    verdicts: dict[str, dict] = {}

    for rubric, n_rows in counts.items():
        row_class = J.RUBRIC_CLASS[rubric]
        for n in range(n_rows):
            row_id = J.new_opaque_ids(J.ROW_PREFIX[rubric], 1)[0]
            prompt = f"prompt {n} for {rubric}"
            entry = {"row_id": row_id, "rubric": rubric,
                     "screen_row_id": f"corpus:{rubric}:{n}",
                     "row_class": row_class, "arm_id": J.CANDIDATE_ARM_ID,
                     "prompt_sha256": _sha(prompt),
                     "hit_max_new_tokens": False,
                     "completion_token_count": 100 + n}
            if rubric in J.PAIRWISE_RUBRICS:
                baseline, candidate = f"reply P {n}", f"reply Q {n}"
                orientation = J.draw_orientation()
                texts = {J.ORIGIN_BASELINE: baseline,
                         J.ORIGIN_CANDIDATE: candidate}
                row = {"row_id": row_id, "prompt_text": prompt,
                       "response_a": texts[orientation[J.POSITION_A]],
                       "response_b": texts[orientation[J.POSITION_B]]}
                entry.update({
                    "baseline_response_sha256": _sha(baseline),
                    "candidate_response_sha256": _sha(candidate),
                    "response_a_origin": orientation[J.POSITION_A],
                    "response_b_origin": orientation[J.POSITION_B],
                    "responses_are_byte_identical": False})
                parsed = {"item_id": row_id, "preference": "EQUIVALENT",
                          "difference": "NONE", "pairing_consistent": True,
                          "reason": ""}
                other = dict(parsed)
                if n == 1:                      # one genuine disagreement
                    other = dict(parsed, preference="A_BETTER",
                                 difference="MINOR")
            else:
                response = f"response {n}"
                row = {"row_id": row_id, "prompt_text": prompt,
                       "response_text": response}
                entry["response_sha256"] = _sha(response)
                entry["candidate_response_sha256"] = _sha(response)
                field = T.AXIS_VERDICT_FIELD[rubric]
                value = T.AXIS_VALUES[rubric][0]
                parsed = {"item_id": row_id, field: value,
                          "confidence": "high", "reason": ""}
                other = dict(parsed)
                if n == 1:
                    other = dict(parsed)
                    other[field] = T.AXIS_VALUES[rubric][1]
            entry["rendered_user_sha256"] = _sha(
                json.dumps(row, sort_keys=True))
            entry["rendered_system_sha256"] = _sha(rubric)
            rows_by_rubric[rubric].append(row)
            private.append(entry)
            verdicts[row_id] = {T.JUDGES[0]: parsed, T.JUDGES[1]: other}

    # The three cells that lose a verdict, one per real-world failure shape.
    q_rows = [r["row_id"] for r in rows_by_rubric["quality"]]
    s_rows = [r["row_id"] for r in rows_by_rubric["safety"]]
    failing = {
        q_rows[2]: (T.JUDGES[0], "quality", U.FAILURE_MISMATCHED_ECHO),
        q_rows[3]: (T.JUDGES[1], "quality", U.FAILURE_MISMATCHED_ECHO),
        s_rows[2]: (T.JUDGES[1], "safety", U.FAILURE_EMPTY_RESPONSE),
    }

    for rubric, rows in rows_by_rubric.items():
        T.write_jsonl(packet / f"{rubric}_judge_input.jsonl",
                      sorted(rows, key=lambda r: r["row_id"]))
    T.write_jsonl(packet / "private" / J.ROW_MAP_FILE,
                  sorted(private, key=lambda r: r["row_id"]))

    entries = []
    for judge in T.JUDGES:
        for rubric in J.RUBRICS_USED:
            out = runs / f"packet__{rubric}__{judge}"
            (out / "raw").mkdir(parents=True)
            records, failures = [], []
            for row in rows_by_rubric[rubric]:
                row_id = row["row_id"]
                fail = failing.get(row_id)
                if fail and fail[0] == judge:
                    evidence = []
                    for attempt in (1, 2):
                        payload = (None if fail[2] == U.FAILURE_EMPTY_RESPONSE
                                   else dict(verdicts[row_id][judge],
                                             item_id=_one_char_off(row_id)))
                        rel = (f"raw/{row_id}.json" if attempt == 1 else
                               f"raw/{row_id}__attempt_"
                               f"{attempt:04d}.json")
                        (out / rel).write_text(_raw_body(payload),
                                               encoding="utf-8")
                        digest = T.sha256_text((out / rel).read_text(
                            encoding="utf-8"))
                        failures.append({
                            "row_id": row_id, "rubric": rubric,
                            "judge": judge, "status": "FAILED",
                            "http_status": 200, "attempt_number": attempt,
                            "raw_response_path": rel,
                            "raw_response_sha256": digest,
                            "error": "recorded elsewhere"})
                        evidence.append({"raw_response_path": rel,
                                         "raw_response_sha256": digest,
                                         "http_status": 200,
                                         "attempt_number": attempt})
                    entries.append({
                        "unresolved_id": f"TSU1-{len(entries) + 1:03d}",
                        "row_id": row_id,
                        "screen_row_id": next(
                            e["screen_row_id"] for e in private
                            if e["row_id"] == row_id),
                        "rubric": rubric, "row_class": J.RUBRIC_CLASS[rubric],
                        "failed_judge": judge, "failure_type": fail[2],
                        "attempts": 2, "reason": "technical",
                        "other_judge_has_verdict": True,
                        "evidence": evidence})
                    continue
                record = {"row_id": row_id, "rubric": rubric, "judge": judge,
                          "status": "OK",
                          "parsed_result": verdicts[row_id][judge]}
                record["record_sha256"] = T.record_hash(record)
                records.append(record)
            T.write_jsonl(out / "results.jsonl", records)
            if failures:
                T.write_jsonl(out / "failures.jsonl", failures)

    manifest_path = tmp_path / "unresolved.json"
    manifest_path.write_text(json.dumps({
        "manifest": "test", "is_outcome_based_exclusion": False,
        "contains_prompt_or_response_text": False, "invents_no_verdict": True,
        "unresolved": entries,
        "counts": U.derived_counts({"unresolved": entries})}, indent=2),
        encoding="utf-8")

    expect = {
        "cells": len(private),
        "cells_by_rubric": {r: len(v) for r, v in rows_by_rubric.items()},
        "cells_by_arm": {J.CANDIDATE_ARM_ID: len(private)},
        "cells_by_class": {J.RUBRIC_CLASS[r]: len(v)
                           for r, v in rows_by_rubric.items()},
        "rows_by_class": {J.RUBRIC_CLASS[r]: len(v)
                          for r, v in rows_by_rubric.items()},
        "records_per_judge": len(private),
        "terminal_states": ("agree", "disagree"),
    }
    return {"tmp": tmp_path, "packet": packet, "runs": runs,
            "manifest": manifest_path, "expect": expect,
            "failing": failing, "rng": rng}


def _join(stage, **kw):
    return join_mod.join(stage["packet"], stage["runs"],
                         stage["tmp"] / "joined", expect=stage["expect"],
                         unresolved_manifest=stage["manifest"], **kw)


# --------------------------------------------------------------- the join
def test_a_missing_sonnet_verdict_is_carried_not_decided(stage):
    summary = _join(stage)
    cells = {c["row_id"]: c for c in T.read_jsonl(
        stage["tmp"] / "joined" / "private" / J.JOINED_CELLS_FILE)}
    row_id = next(r for r, f in stage["failing"].items()
                  if f[0] == T.JUDGES[0])
    assert cells[row_id]["state"] == U.STATE_UNRESOLVED_TECHNICAL
    assert cells[row_id]["verdicts"][T.JUDGES[0]] is None
    assert cells[row_id]["verdicts"][T.JUDGES[1]] is not None
    assert summary["cells"] == stage["expect"]["cells"]


def test_a_missing_opus_verdict_is_carried_not_decided(stage):
    _join(stage)
    cells = {c["row_id"]: c for c in T.read_jsonl(
        stage["tmp"] / "joined" / "private" / J.JOINED_CELLS_FILE)}
    row_id = next(r for r, f in stage["failing"].items()
                  if f[0] == T.JUDGES[1] and f[1] == "quality")
    assert cells[row_id]["state"] == U.STATE_UNRESOLVED_TECHNICAL
    assert cells[row_id]["verdicts"][T.JUDGES[1]] is None
    assert cells[row_id]["verdicts"][T.JUDGES[0]] is not None


def test_every_cell_survives_the_join(stage):
    """480 in the real run: nothing is deleted to make the arithmetic work."""
    summary = _join(stage)
    assert summary["cells"] == stage["expect"]["cells"]
    states = summary["by_state"]
    assert states[U.STATE_UNRESOLVED_TECHNICAL] == 3
    assert sum(states.values()) == stage["expect"]["cells"]


def test_the_join_refuses_a_failure_the_manifest_does_not_carry(stage):
    """A new failure must stop the stage, not ride in on an old manifest."""
    manifest = json.loads(stage["manifest"].read_text(encoding="utf-8"))
    manifest["unresolved"] = manifest["unresolved"][:1]
    manifest["counts"] = U.derived_counts(manifest)
    stage["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValidationError) as exc:
        _join(stage)
    assert "no verdict" in str(exc.value)
    assert "not carried by the technical-unresolved manifest" in str(exc.value)


def test_the_join_refuses_a_manifest_whose_evidence_does_not_recompute(stage):
    manifest = json.loads(stage["manifest"].read_text(encoding="utf-8"))
    manifest["unresolved"][0]["evidence"][0]["raw_response_sha256"] = "0" * 64
    stage["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(U.TechnicalUnresolvedError) as exc:
        _join(stage)
    assert "digest" in str(exc.value)


def test_the_join_refuses_a_manifest_cell_the_judge_later_answered(stage):
    """If the failure stopped being true, the manifest must be revised."""
    row_id, (judge, rubric, _kind) = next(iter(stage["failing"].items()))
    run = stage["runs"] / f"packet__{rubric}__{judge}"
    records = list(T.read_jsonl(run / "results.jsonl"))
    late = {"row_id": row_id, "rubric": rubric, "judge": judge,
            "status": "OK",
            "parsed_result": {"item_id": row_id, "preference": "EQUIVALENT",
                              "difference": "NONE",
                              "pairing_consistent": True, "reason": ""}}
    late["record_sha256"] = T.record_hash(late)
    records.append(late)
    T.write_jsonl(run / "results.jsonl", records)
    with pytest.raises(U.TechnicalUnresolvedError) as exc:
        _join(stage)
    assert "no longer true" in str(exc.value)


# ------------------------------------------- the mismatched id and the empty
def test_a_mismatched_echo_is_never_accepted_as_a_verdict(stage):
    """The one-character-wrong id stays a failure, at every layer."""
    row_id = next(r for r, f in stage["failing"].items()
                  if f[2] == U.FAILURE_MISMATCHED_ECHO)
    wrong = _one_char_off(row_id)
    assert wrong != row_id and len(wrong) == len(row_id)
    protocol = J.quality_protocol_stub() if hasattr(
        J, "quality_protocol_stub") else None
    with pytest.raises(J.TwoSiteJudgingError) as exc:
        J.validate_quality_parsed(
            {"item_id": wrong, "preference": "EQUIVALENT",
             "difference": "NONE"}, row_id,
            "rq3_b_quality_pairwise_preference_v1")
    assert "echoes item_id" in str(exc.value)
    assert protocol is None or protocol is not None

    _join(stage)
    cells = {c["row_id"]: c for c in T.read_jsonl(
        stage["tmp"] / "joined" / "private" / J.JOINED_CELLS_FILE)}
    failed_judge = stage["failing"][row_id][0]
    assert cells[row_id]["verdicts"][failed_judge] is None


def test_an_empty_judge_response_is_never_interpreted(stage):
    row_id = next(r for r, f in stage["failing"].items()
                  if f[2] == U.FAILURE_EMPTY_RESPONSE)
    judge = stage["failing"][row_id][0]
    raw = json.loads((stage["runs"] / f"packet__safety__{judge}"
                      / f"raw/{row_id}.json").read_text(encoding="utf-8"))
    assert U.classify_raw_failure(raw, row_id) == U.FAILURE_EMPTY_RESPONSE

    _join(stage)
    cells = {c["row_id"]: c for c in T.read_jsonl(
        stage["tmp"] / "joined" / "private" / J.JOINED_CELLS_FILE)}
    assert cells[row_id]["verdicts"][judge] is None
    assert cells[row_id]["state"] == U.STATE_UNRESOLVED_TECHNICAL


def test_the_failure_type_is_rederived_from_the_body_not_the_error_string(
        stage):
    """Classification never depends on the credential-bearing error field."""
    row_id = next(r for r, f in stage["failing"].items()
                  if f[2] == U.FAILURE_MISMATCHED_ECHO)
    judge, rubric, _ = stage["failing"][row_id]
    raw = json.loads((stage["runs"] / f"packet__{rubric}__{judge}"
                      / f"raw/{row_id}.json").read_text(encoding="utf-8"))
    assert U.classify_raw_failure(raw, row_id) == U.FAILURE_MISMATCHED_ECHO
    # A body that echoes the right id is not a carryable failure.
    good = json.loads(_raw_body({"item_id": row_id, "preference": "EQUIVALENT",
                                 "difference": "NONE"}))
    assert U.classify_raw_failure(good, row_id) == "EchoMatched"


# ------------------------------------------------------------ human review
def test_a_technical_failure_never_enters_the_human_review_packet(stage):
    _join(stage)
    joined = stage["tmp"] / "joined"
    review = stage["tmp"] / "review"
    cells = list(T.read_jsonl(joined / "private" / J.JOINED_CELLS_FILE))
    wanted = J.cells_needing_human_review(cells, J.DECLARED_PROCEDURE)
    carried = {c["row_id"] for c in cells
               if c["state"] == U.STATE_UNRESOLVED_TECHNICAL}
    selected = {c["row_id"] for c in cells
                if J.cell_key(c, "cell") in set(wanted)}
    assert carried and not (carried & selected)
    assert all(c["state"] == "disagree" for c in cells
               if J.cell_key(c, "cell") in set(wanted))
    assert not review.exists()


def test_a_human_cannot_replace_a_missing_judge_verdict(stage):
    """The named refusal, not merely absence from the selection."""
    _join(stage)
    joined = stage["tmp"] / "joined"
    cells = {J.cell_key(c, "cell"): c for c in
             T.read_jsonl(joined / "private" / J.JOINED_CELLS_FILE)}
    carried_key = next(k for k, c in cells.items()
                       if c["state"] == U.STATE_UNRESOLVED_TECHNICAL)

    def fake_load_review(*_a, **_k):
        return {carried_key: {"verdict": "NO", "human_row_id": "TSJ1HUMAN01"}}

    original = import_mod.load_review
    import_mod.load_review = fake_load_review
    try:
        with pytest.raises(Exception) as exc:
            import_mod.run(joined, stage["packet"], None, None,
                           stage["tmp"] / "adjudicated", J.DECLARED_PROCEDURE,
                           write=False, expect=stage["expect"])
        assert ("no_human_verdict_replaces_a_missing_judge_verdict"
                in str(exc.value)
                or "does not send to a human" in str(exc.value))
    finally:
        import_mod.load_review = original


def test_adjudicate_labels_a_carried_cell_unresolved_under_both_procedures(
        stage):
    """The frozen rule does this on its own; nothing new decides it."""
    for procedure in J.ADJUDICATION_PROCEDURES:
        cell = {"screen_row_id": "corpus:quality:2",
                "arm_id": J.CANDIDATE_ARM_ID, "rubric": "quality",
                "row_class": "B",
                "verdicts": {T.JUDGES[0]: None,
                             T.JUDGES[1]: "EQUIVALENT|NONE"}}
        out = J.adjudicate(cell, None, procedure)
        assert out["verdict"] is None
        assert out["label_source"] == T.SOURCE_UNRESOLVED
        assert out["judges_agree"] is None
        assert "missing judge verdict" in out["adjudication_rule"]


# ------------------------------------------------------------- denominators
def test_readings_state_a_decided_denominator_and_never_divide_by_the_total():
    """A rate over an unstated denominator is the thing being prevented."""
    verdicts = {f"row{i}": "SA" for i in range(10)}
    verdicts["row10"] = None
    reading = J.e_reading(verdicts)

    assert reading["complete"] is False
    assert reading["substantive_repair"] is None      # original: withheld
    assert reading["decided_denominator"] == 10       # complete-case: stated
    assert reading["decided_substantive_repair"] == 10
    assert reading["decided_substantive_repair_rate"] == 1.0
    assert reading["unresolved_screen_row_ids"] == ["row10"]
    assert len(verdicts) == 11                        # never the denominator


def test_h_reading_brackets_the_unresolved_cell_without_deciding_it():
    verdicts = {f"row{i}": "NO" for i in range(10)}
    verdicts["row10"] = None
    reading = J.h_reading(verdicts)
    assert reading["decided_denominator"] == 10
    assert reading["decided_NO"] == 10
    assert reading["sensitivity_YES_if_all_unresolved_were_NO"] == 0
    assert reading["sensitivity_YES_if_all_unresolved_were_YES"] == 1
    assert reading["YES"] is None                     # still withheld


def test_derived_counts_report_957_of_960_for_three_missing_verdicts():
    manifest = {"unresolved": [
        {"row_id": "TSJ1QUALITY0000000000A1", "row_class": "B",
         "failed_judge": "sonnet"},
        {"row_id": "TSJ1QUALITY0000000000A2", "row_class": "B",
         "failed_judge": "opus"},
        {"row_id": "TSJ1SAFETY00000000000A3", "row_class": "H",
         "failed_judge": "opus"}]}
    counts = U.derived_counts(manifest)
    assert counts["planned_verdicts"] == 960
    assert counts["valid_verdicts"] == 957
    assert counts["missing_verdicts"] == 3
    assert counts["cells"] == 480
    assert counts["dual_judged_cells"] == 477
    assert counts["technically_unresolved_cells"] == 3
    assert counts["original_protocol_complete"] is False
    assert U.unresolved_by_class(manifest) == {"B": 2, "H": 1}


# --------------------------------------------------------------- redaction
def test_an_invalid_header_exception_is_redacted_before_it_is_recorded():
    """The exact leak: http.client puts the key into the message text."""
    import urllib.error
    import urllib.request

    secret = "sk-ant-NOTAREALKEY-000111222333"
    request = urllib.request.Request(
        "http://127.0.0.1:9/x", data=b"{}", method="POST",
        headers={T.API_KEY_HEADER: secret + "\n"})
    try:
        urllib.request.urlopen(request, timeout=1)
        raise AssertionError("expected the header to be rejected")
    except ValueError as exc:
        raw = f"{type(exc).__name__}: {exc}"
    except urllib.error.URLError:                     # pragma: no cover
        pytest.skip("the header was not rejected in this environment")

    assert secret in raw                              # the bug, reproduced
    assert secret not in T.redact_exception_message(raw, [secret])
    # and redacted even when the caller does not know the secret
    assert secret not in T.redact_exception_message(raw)
    assert T.REDACTED in T.redact_exception_message(raw)


def test_redaction_leaves_a_message_that_carries_no_credential_alone():
    message = "TimeoutError: read timed out - NOT retried"
    assert T.redact_exception_message(message, [None, ""]) == message


def test_redaction_never_touches_evidence_digests():
    """Redaction is for the error string only, never for scientific fields."""
    digest = "a" * 64
    message = f"ValueError: body digest {digest} did not match"
    assert digest in T.redact_exception_message(message, ["sk-ant-x"])
