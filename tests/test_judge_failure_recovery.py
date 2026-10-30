"""Append-only raw evidence, and outcome-blind recovery of a slipped item_id echo.

Nothing here calls an API, opens a socket, loads a model, allocates a GPU or
reads the real packet. Every fixture is invented in ``tmp_path``, the protocol
text is a stand-in whose digests are recomputed for a fake manifest, and the
one test that exercises ``run_judge.run`` replaces its transport with a stub.

Organised by the failure each group is there to prevent:

    1. paths      a retry overwrites the body of the attempt before it
    2. runner     a record points at a file that is not the one its call wrote
    3. echoes     a malformation is named, or refused, on the string alone
    4. gates      a row is recovered on anything weaker than its provenance
    5. blindness  the decision, or the report, depends on the verdict
    6. idempotent a second recovery run duplicates a successful record
    7. hygiene    the recovery path can reach a network or edit history
"""
from __future__ import annotations

import ast
import importlib
import json
import socket
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate import triplets as T                                  # noqa: E402
from routegate.triplets import (                                     # noqa: E402
    AXIS2,
    CORRECTION_DUPLICATED,
    CORRECTION_EXACT,
    CORRECTION_SUBSTITUTED,
    CORRECTION_TRUNCATED,
    RECOVERY_GATES,
    SAFETY,
    TripletError,
    assess_failed_row,
    build_request_index,
    classify_item_id_echo,
    load_protocol,
    next_raw_response_path,
    raw_attempt_number,
    raw_path_belongs_to,
    raw_response_path,
    read_jsonl,
    record_hash,
    request_hash,
)
from routegate.validation import sha256_text                         # noqa: E402

STAGE_DIR = REPO_ROOT / "experiments" / "triplet_validation"
RECOVERY_SCRIPT = STAGE_DIR / "recover_failed_rows.py"
RUNNER_SCRIPT = STAGE_DIR / "run_judge.py"
STAMP = "2026-08-24T21:00:00Z"


def _module(name: str):
    if str(STAGE_DIR) not in sys.path:
        sys.path.insert(0, str(STAGE_DIR))
    return importlib.import_module(name)


# ------------------------------------------------------------------ fixtures
def _schema(axis: str) -> dict:
    field = T.AXIS_VERDICT_FIELD[axis]
    return {"name": T.AXIS_SCHEMA_NAME[axis],
            "fields": {"item_id": {"type": "string", "max_len": 64},
                       field: {"type": "enum", "values": list(T.AXIS_VALUES[axis])},
                       "confidence": {"type": "enum",
                                      "values": list(T.CONFIDENCE_VALUES)},
                       "reason": {"type": "string", "max_len": 240}}}


@pytest.fixture
def protocol_dir(tmp_path):
    """A stand-in protocol whose digests are consistent with its own manifest."""
    root = tmp_path / "rq1_judge_protocol"
    root.mkdir()
    hashes, lengths = {}, {}
    for axis in (T.AXIS1, AXIS2, SAFETY):
        instructions = f"SYNTHETIC {axis} INSTRUCTIONS. Judge the item.\n"
        template = ("item_id: {item_id}\nprompt: {prompt_text}\n"
                    + ("response: {response_text}\n" if axis != T.AXIS1 else ""))
        schema = _schema(axis)
        (root / f"{axis}_instructions.txt").write_text(instructions, encoding="utf-8")
        (root / f"{axis}_user_template.txt").write_text(template, encoding="utf-8")
        (root / f"{axis}_schema.json").write_text(json.dumps(schema), encoding="utf-8")
        hashes[f"{axis}_instructions_sha256"] = sha256_text(instructions)
        hashes[f"{axis}_user_template_sha256"] = sha256_text(template)
        hashes[f"{axis}_schema_sha256"] = sha256_text(json.dumps(schema, sort_keys=True))
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


#: Four opaque ids. Uppercase alphanumeric, as an opaque id must be, and long
#: enough that a substitution or a duplication inside one is unambiguous.
ROWS = ["TESTROWAAAA0001", "TESTROWBBBB0002", "TESTROWCCCC0003", "TESTROWDDDD0004"]


@pytest.fixture
def packet(tmp_path):
    """A four-row axis2 packet. Each row renders to a distinct request."""
    directory = tmp_path / "packet"
    directory.mkdir()
    rows = [{"row_id": row_id, "prompt_text": f"prompt for {i}",
             "response_text": f"response for {i}"}
            for i, row_id in enumerate(ROWS)]
    (directory / f"{AXIS2}_judge_input.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    return directory, rows


def _envelope(item_id, verdict="SA", confidence="high", axis=AXIS2, model="m-s",
              extra=None):
    inner = {"item_id": item_id, T.AXIS_VERDICT_FIELD[axis]: verdict,
             "confidence": confidence, "reason": "because"}
    if extra is not None:
        inner.update(extra)
    return json.dumps({"model": model,
                       "content": [{"type": "text", "text": json.dumps(inner)}]})


def _dup(row_id: str) -> str:
    """A duplicated-substring echo of ``row_id``: its own tail, repeated.

    Tests that mean to reach a *later* gate need an echo the classifier can
    name, or they stop at the classifier instead and pass for the wrong
    reason. ``row_id + "AA"`` is not such an echo -- the appended block
    repeats nothing next to it -- and that is deliberate: see
    ``test_an_unexplained_echo_with_perfect_provenance_is_still_refused``.
    """
    return row_id + row_id[-6:]


def _empty_envelope(model="m-s"):
    return json.dumps({"model": model, "content": []})


def _write_raw(run_dir: Path, rel: str, body: str, http_status=200) -> str:
    """Write a preserved body exactly as ``run_judge.preserve_raw`` writes it."""
    payload = json.dumps({"http_status": http_status, "response_headers": {},
                          "raw_body": body}, sort_keys=True, ensure_ascii=False)
    path = run_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return sha256_text(payload)


def _failure(run_dir, protocol, row, body, *, rel=None, attempt=1,
             http_status=200, request_row=None, error="echo", judge="sonnet",
             model="m-s", raw_sha=None, write_raw=True):
    """One failure record, wired to a raw file the way the runner wires it."""
    rel = rel or raw_response_path(row["row_id"], attempt)
    written = _write_raw(run_dir, rel, body, http_status) if write_raw else None
    system, user = protocol.render(request_row or row)
    record = {
        "row_id": row["row_id"], "axis": protocol.axis, "judge": judge,
        "model_requested": model,
        "request_sha256": request_hash(system, user, model),
        "response_sha256": sha256_text(body),
        "raw_response_sha256": raw_sha if raw_sha is not None else written,
        "raw_response_path": rel, "attempt_number": attempt,
        "http_status": http_status, "elapsed_seconds": 1.0,
        "requested_at_utc": "2026-08-24T18:00:00Z",
        "temperature": T.TEMPERATURE, "max_tokens": T.MAX_TOKENS,
        "protocol_hashes": protocol.hashes(),
        "returned_model": model, "returned_model_accepted": True,
        "returned_model_evidence": "read from the response payload",
        "status": "FAILED", "error": error, "error_type": "TripletError",
        "retried": False, "billing_note": "may already have been billed",
    }
    record["record_sha256"] = record_hash(record)
    return record


def _run_dir(tmp_path, protocol, failures, *, results=(), axis=AXIS2,
             judge="sonnet", name="run"):
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run_manifest.json").write_text(
        json.dumps({"axis": axis, "judge": judge, "model_requested": "m-s",
                    "protocol_hashes": protocol.hashes()}), encoding="utf-8")
    (directory / "failures.jsonl").write_text(
        "".join(json.dumps(f, sort_keys=True) + "\n" for f in failures),
        encoding="utf-8")
    (directory / "results.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in results),
        encoding="utf-8")
    return directory


def _assess(row, run_dir, protocol, rows, attempts, **kw):
    index = build_request_index(rows, protocol, "m-s")
    claims_p, claims_r = {}, {}
    for record in attempts:
        claims_p.setdefault(record["raw_response_path"], []).append(record["row_id"])
        claims_r.setdefault(record["request_sha256"], []).append(record["row_id"])
    return assess_failed_row(
        row["row_id"], [a for a in attempts if a["row_id"] == row["row_id"]],
        run_dir, protocol, "m-s", index, [r["row_id"] for r in rows], STAMP,
        claimed_paths=kw.pop("claimed_paths", claims_p),
        claimed_requests=kw.pop("claimed_requests", claims_r), **kw)


# ======================================================== 1. attempt paths
def test_the_first_attempt_keeps_the_original_bare_path():
    assert raw_response_path("ABC123", 1) == "raw/ABC123.json"


def test_a_retry_gets_its_own_numbered_path():
    assert raw_response_path("ABC123", 2) == "raw/ABC123__attempt_0002.json"
    assert raw_response_path("ABC123", 11) == "raw/ABC123__attempt_0011.json"


def test_attempt_zero_is_refused():
    with pytest.raises(TripletError, match=">= 1"):
        raw_response_path("ABC123", 0)


def test_the_attempt_number_is_derived_from_the_history():
    history = [{"row_id": "A"}, {"row_id": "B"}, {"row_id": "A"}]
    assert raw_attempt_number("A", history) == 3
    assert raw_attempt_number("B", history) == 2
    assert raw_attempt_number("C", history) == 1


def test_the_next_path_walks_past_anything_already_on_disk(tmp_path):
    """A file on disk outranks a history that disagrees with it."""
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "A.json").write_text("first", encoding="utf-8")
    rel, attempt = next_raw_response_path(tmp_path, "A", [])
    assert (rel, attempt) == ("raw/A__attempt_0002.json", 2)
    (tmp_path / rel).write_text("second", encoding="utf-8")
    assert next_raw_response_path(tmp_path, "A", []) == (
        "raw/A__attempt_0003.json", 3)


def test_a_path_is_matched_to_its_own_row_and_not_to_a_prefix_of_it():
    assert raw_path_belongs_to("raw/ABC123.json", "ABC123")
    assert raw_path_belongs_to("raw/ABC123__attempt_0002.json", "ABC123")
    assert not raw_path_belongs_to("raw/ABC123456.json", "ABC123")
    assert not raw_path_belongs_to("raw/ABC123__attempt_00X2.json", "ABC123")
    assert not raw_path_belongs_to("raw/OTHER.json", "ABC123")


# ============================================================ 2. the runner
@pytest.fixture
def stub_run(monkeypatch, protocol_dir, packet, tmp_path):
    """``run_judge.run`` with its transport replaced. No socket is opened."""
    runner = _module("run_judge")
    root, manifest = protocol_dir
    packet_dir, rows = packet
    monkeypatch.setattr(runner, "load_protocol_manifest", lambda *a, **k: manifest)
    monkeypatch.setenv(T.ENV_API_KEY_HISTORICAL, "not-a-real-key")
    monkeypatch.setenv(T.ENV_ENDPOINT, "https://example.invalid/never-called")
    out_dir = tmp_path / "runout"

    def go(replies):
        """``replies`` maps row_id -> raw body string, consumed in order."""
        queue = {k: list(v) for k, v in replies.items()}

        def fake_call_api(endpoint, api_key, model, system, user):
            row_id = user.split("item_id: ", 1)[1].split("\n", 1)[0]
            body = queue[row_id].pop(0)
            return {"http_status": 200, "raw_body": body,
                    "payload": json.loads(body), "response_headers": {},
                    "elapsed_seconds": 0.1}

        monkeypatch.setattr(runner, "call_api", fake_call_api)
        return runner.run(packet_dir, AXIS2, "sonnet", out_dir, root, None, False)

    return go, out_dir, rows


def test_a_retry_does_not_alter_the_first_raw_response(stub_run):
    """1. The body of the attempt before it is untouched, byte for byte."""
    go, out_dir, rows = stub_run
    bad = {r["row_id"]: [_envelope("WRONGECHO" + r["row_id"])] for r in rows}
    assert go(bad) == 2                       # every row failed on its echo
    first = {p.name: (p.read_text(encoding="utf-8"), p.stat().st_mtime_ns)
             for p in sorted((out_dir / "raw").iterdir())}
    assert set(first) == {f"{r['row_id']}.json" for r in rows}

    good = {r["row_id"]: [_envelope(r["row_id"])] for r in rows}
    assert go(good) == 0
    for name, (text, mtime) in first.items():
        path = out_dir / "raw" / name
        assert path.read_text(encoding="utf-8") == text
        assert path.stat().st_mtime_ns == mtime


def test_both_attempt_paths_and_hashes_remain_verifiable(stub_run):
    """2. Every record's digest recomputes from the exact file it named."""
    go, out_dir, rows = stub_run
    go({r["row_id"]: [_envelope("WRONGECHO" + r["row_id"])] for r in rows})
    go({r["row_id"]: [_envelope(r["row_id"])] for r in rows})

    records = (read_jsonl(out_dir / "failures.jsonl")
               + read_jsonl(out_dir / "results.jsonl"))
    assert len(records) == 2 * len(rows)
    paths = set()
    for record in records:
        path = out_dir / record["raw_response_path"]
        assert path.is_file(), record["raw_response_path"]
        assert sha256_text(path.read_text(encoding="utf-8")) == \
            record["raw_response_sha256"]
        assert raw_path_belongs_to(record["raw_response_path"], record["row_id"])
        paths.add(record["raw_response_path"])
    assert len(paths) == 2 * len(rows)        # no two records share a file


def test_a_successful_retry_points_to_its_own_raw_response(stub_run):
    """3. The success names attempt 2; the failure still names attempt 1."""
    go, out_dir, rows = stub_run
    row_id = rows[0]["row_id"]
    go({r["row_id"]: [_envelope("WRONGECHO" + r["row_id"])] for r in rows})
    go({r["row_id"]: [_envelope(r["row_id"])] for r in rows})

    failure = [r for r in read_jsonl(out_dir / "failures.jsonl")
               if r["row_id"] == row_id][0]
    success = [r for r in read_jsonl(out_dir / "results.jsonl")
               if r["row_id"] == row_id][0]
    assert failure["raw_response_path"] == f"raw/{row_id}.json"
    assert failure["attempt_number"] == 1
    assert success["raw_response_path"] == f"raw/{row_id}__attempt_0002.json"
    assert success["attempt_number"] == 2
    assert failure["raw_response_sha256"] != success["raw_response_sha256"]
    body = json.loads((out_dir / success["raw_response_path"]).read_text(
        encoding="utf-8"))["raw_body"]
    assert json.loads(body)["content"][0]["text"].count(row_id) == 1


def test_resume_still_skips_a_successful_row(stub_run):
    """A third run calls nothing: the success is not re-billed, or re-pathed."""
    go, out_dir, rows = stub_run
    go({r["row_id"]: [_envelope(r["row_id"])] for r in rows})
    before = sorted(p.name for p in (out_dir / "raw").iterdir())
    assert go({}) == 0                        # empty queue: any call would KeyError
    assert sorted(p.name for p in (out_dir / "raw").iterdir()) == before
    assert len(read_jsonl(out_dir / "results.jsonl")) == len(rows)


def test_preserve_raw_refuses_to_stand_on_an_existing_file(tmp_path):
    runner = _module("run_judge")
    reply = {"http_status": 200, "response_headers": {}, "raw_body": "x"}
    runner.preserve_raw(tmp_path, "raw/A.json", reply)
    with pytest.raises(TripletError, match="never overwritten"):
        runner.preserve_raw(tmp_path, "raw/A.json", reply)


# ============================================================== 3. echoes
@pytest.mark.parametrize("expected,echoed,kind", [
    ("FDV1RESPONSE148E0F1ADD28", "FDV1RESPONSE148E0F1ADD28", CORRECTION_EXACT),
    ("FDV1RESPONSE148E0F1ADD28", "FDV1RESPONSE148E0F1ADD28E0F1ADD28",
     CORRECTION_DUPLICATED),
    ("FDV1RESPONSEB4B9B02A6536", "FDV1RESPONSEB4B9B02A6B9B02A6536",
     CORRECTION_DUPLICATED),
    ("FDV1RESPONSEE97E56B40F47", "FDV1RESPONSEE97E56B40F", CORRECTION_TRUNCATED),
    ("FDV1SAFETY1E570D056658", "FDV1SAFETY1E570D056058", CORRECTION_SUBSTITUTED),
])
def test_the_observed_malformations_are_named(expected, echoed, kind):
    assert classify_item_id_echo(expected, echoed) == kind


@pytest.mark.parametrize("expected,echoed", [
    ("ABCDEF", "ABCXYZ"),                     # two substitutions
    ("ABCDEF", "ABCDEFXY"),                   # an insertion that repeats nothing
    ("ABCDEF", "BCDEF"),                      # a suffix, not a prefix
    ("ABCDEF", "FEDCBA"),                     # an anagram
    ("ABCDEF", ""),                           # nothing at all
])
def test_an_unexplained_echo_is_not_named_a_transcription_error(expected, echoed):
    assert classify_item_id_echo(expected, echoed) is None


# =============================================================== 4. gates
def test_a_duplicated_substring_echo_is_recovered(tmp_path, protocol_dir, packet):
    root, manifest = protocol_dir
    protocol = load_protocol(AXIS2, directory=root, manifest=manifest)
    _, rows = packet
    row = rows[0]
    echo = row["row_id"] + row["row_id"][-6:]
    run = tmp_path / "r"
    run.mkdir()
    failures = [_failure(run, protocol, row, _envelope(echo))]
    decision = _assess(row, run, protocol, rows, failures)
    assert decision.recoverable
    assert decision.correction_kind == CORRECTION_DUPLICATED
    assert all(decision.gates[name] for name in RECOVERY_GATES)
    correction = decision.record["item_id_correction"]
    assert correction["applied"] is True
    assert correction["correction_mapping"] == {"from": echo, "to": row["row_id"]}
    assert decision.record["parsed_result"]["item_id"] == row["row_id"]
    assert decision.record["status"] == "OK"
    assert T.detect_result_format(decision.record) == T.RESULT_FORMAT_SUPPLEMENT


@pytest.mark.parametrize("make_echo,kind", [
    (lambda r: r, CORRECTION_EXACT),
    (lambda r: r[:-2], CORRECTION_TRUNCATED),
    (lambda r: r[:-1] + ("9" if r[-1] != "9" else "8"), CORRECTION_SUBSTITUTED),
])
def test_each_recognised_malformation_recovers(tmp_path, protocol_dir, packet,
                                               make_echo, kind):
    root, manifest = protocol_dir
    protocol = load_protocol(AXIS2, directory=root, manifest=manifest)
    _, rows = packet
    row = rows[1]
    run = tmp_path / "r"
    run.mkdir()
    failures = [_failure(run, protocol, row, _envelope(make_echo(row["row_id"])))]
    decision = _assess(row, run, protocol, rows, failures)
    assert decision.recoverable and decision.correction_kind == kind
    assert decision.record["item_id_correction"]["applied"] is (kind != CORRECTION_EXACT)


def test_an_echo_equal_to_another_real_packet_id_is_refused(tmp_path, protocol_dir,
                                                            packet):
    """The reply may genuinely belong to that other row. Similarity is not evidence."""
    root, manifest = protocol_dir
    protocol = load_protocol(AXIS2, directory=root, manifest=manifest)
    _, rows = packet
    row, other = rows[0], rows[1]
    run = tmp_path / "r"
    run.mkdir()
    failures = [_failure(run, protocol, row, _envelope(other["row_id"]))]
    decision = _assess(row, row and run, protocol, rows, failures)
    assert not decision.recoverable
    assert "echo_is_not_another_packet_id" in decision.blocked_by
    assert decision.record is None


def test_an_ambiguous_packet_binding_is_refused(tmp_path, protocol_dir):
    """Two packet rows that render identically bind the same request digest."""
    root, manifest = protocol_dir
    protocol = load_protocol(AXIS2, directory=root, manifest=manifest)
    rows = [{"row_id": ROWS[0], "prompt_text": "same", "response_text": "same"},
            {"row_id": ROWS[1], "prompt_text": "same", "response_text": "same"}]
    run = tmp_path / "r"
    run.mkdir()
    # Both rows render the same user text apart from the item_id line, so make
    # the second row's request digest collide by rendering it from the first.
    index = build_request_index(rows, protocol, "m-s")
    row = rows[0]
    failures = [_failure(run, protocol, row, _envelope(row["row_id"] + "AA"))]
    index[failures[0]["request_sha256"]] = [ROWS[0], ROWS[1]]
    decision = assess_failed_row(
        row["row_id"], failures, run, protocol, "m-s", index,
        [r["row_id"] for r in rows], STAMP)
    assert not decision.recoverable
    assert "request_digest_recomputes_from_one_packet_row" in decision.blocked_by


def test_a_request_digest_that_does_not_recompute_is_refused(tmp_path, protocol_dir,
                                                             packet):
    root, manifest = protocol_dir
    protocol = load_protocol(AXIS2, directory=root, manifest=manifest)
    _, rows = packet
    row = rows[0]
    run = tmp_path / "r"
    run.mkdir()
    failure = _failure(run, protocol, row, _envelope(row["row_id"] + "AA"))
    failure["request_sha256"] = "0" * 64      # a request no packet row renders
    decision = _assess(row, run, protocol, rows, [failure])
    assert not decision.recoverable
    assert "request_digest_recomputes_from_one_packet_row" in decision.blocked_by


def test_a_raw_response_digest_that_does_not_recompute_is_refused(
        tmp_path, protocol_dir, packet):
    root, manifest = protocol_dir
    protocol = load_protocol(AXIS2, directory=root, manifest=manifest)
    _, rows = packet
    row = rows[0]
    run = tmp_path / "r"
    run.mkdir()
    failure = _failure(run, protocol, row, _envelope(row["row_id"] + "AA"),
                       raw_sha="f" * 64)
    decision = _assess(row, run, protocol, rows, [failure])
    assert not decision.recoverable
    assert "raw_response_digest_recomputes" in decision.blocked_by
    assert decision.gates["raw_response_file_exists"] is True


def test_a_missing_raw_response_is_refused(tmp_path, protocol_dir, packet):
    root, manifest = protocol_dir
    protocol = load_protocol(AXIS2, directory=root, manifest=manifest)
    _, rows = packet
    row = rows[0]
    run = tmp_path / "r"
    run.mkdir()
    failure = _failure(run, protocol, row, _envelope(row["row_id"] + "AA"),
                       write_raw=False, raw_sha="a" * 64)
    decision = _assess(row, run, protocol, rows, [failure])
    assert not decision.recoverable
    assert "raw_response_file_exists" in decision.blocked_by


def test_a_request_that_never_completed_is_refused(tmp_path, protocol_dir, packet):
    root, manifest = protocol_dir
    protocol = load_protocol(AXIS2, directory=root, manifest=manifest)
    _, rows = packet
    row = rows[0]
    run = tmp_path / "r"
    run.mkdir()
    failure = _failure(run, protocol, row, "", http_status=503)
    decision = _assess(row, run, protocol, rows, [failure])
    assert not decision.recoverable
    assert "http_request_completed" in decision.blocked_by


def test_a_malformed_verdict_schema_is_refused(tmp_path, protocol_dir, packet):
    """Out of vocabulary, once the item_id is set aside. Never a verdict."""
    root, manifest = protocol_dir
    protocol = load_protocol(AXIS2, directory=root, manifest=manifest)
    _, rows = packet
    row = rows[0]
    run = tmp_path / "r"
    run.mkdir()
    body = _envelope(_dup(row["row_id"]), verdict="NOT_IN_THE_VOCABULARY")
    decision = _assess(row, run, protocol, rows, [_failure(run, protocol, row, body)])
    assert not decision.recoverable
    assert "parsed_result_conforms_to_frozen_schema" in decision.blocked_by
    assert decision.record is None


def test_a_reply_with_no_assistant_text_is_refused(tmp_path, protocol_dir, packet):
    """The two safety/opus rows: HTTP 200, zero content blocks, nothing to recover."""
    root, manifest = protocol_dir
    protocol = load_protocol(AXIS2, directory=root, manifest=manifest)
    _, rows = packet
    row = rows[0]
    run = tmp_path / "r"
    run.mkdir()
    failure = _failure(run, protocol, row, _empty_envelope(),
                       error="EmptyJudgeResponse")
    decision = _assess(row, run, protocol, rows, [failure])
    assert not decision.recoverable
    assert decision.gates["binding_is_unique"] is True      # provenance is fine
    assert "EmptyJudgeResponse" in decision.reason          # the reply is not


def test_a_second_row_claiming_the_response_is_refused(tmp_path, protocol_dir,
                                                       packet):
    root, manifest = protocol_dir
    protocol = load_protocol(AXIS2, directory=root, manifest=manifest)
    _, rows = packet
    row = rows[0]
    run = tmp_path / "r"
    run.mkdir()
    failure = _failure(run, protocol, row, _envelope(_dup(row["row_id"])))
    decision = _assess(row, run, protocol, rows, [failure],
                       claimed_paths={failure["raw_response_path"]:
                                      [row["row_id"], rows[2]["row_id"]]})
    assert not decision.recoverable
    assert "no_second_row_could_receive_it" in decision.blocked_by


def test_duplicate_failure_attempts_for_one_row_are_history_not_rows(
        tmp_path, protocol_dir, packet):
    """Two attempts, one row. The later supersedes; both are preserved."""
    root, manifest = protocol_dir
    protocol = load_protocol(AXIS2, directory=root, manifest=manifest)
    _, rows = packet
    row = rows[0]
    run = tmp_path / "r"
    run.mkdir()
    first = _failure(run, protocol, row, _envelope(_dup(row["row_id"])), attempt=1)
    second = _failure(run, protocol, row, _envelope(row["row_id"][:-3]), attempt=2)
    decision = _assess(row, run, protocol, rows, [first, second])
    assert decision.recoverable
    assert decision.attempts == 2
    assert decision.correction_kind == CORRECTION_TRUNCATED   # the later attempt
    history = decision.record["recovered_from_failure"]
    assert history["attempt_count"] == 2
    assert history["recovered_attempt_number"] == 2
    assert history["recovered_attempt_record_sha256"] == second["record_sha256"]
    assert [a["record_sha256"] for a in history["attempts"]] == \
        [first["record_sha256"], second["record_sha256"]]


def test_an_overwritten_first_attempt_selects_the_one_that_recomputes(
        tmp_path, protocol_dir, packet):
    """The legacy single-path runs: only the surviving body can be bound."""
    root, manifest = protocol_dir
    protocol = load_protocol(AXIS2, directory=root, manifest=manifest)
    _, rows = packet
    row = rows[0]
    run = tmp_path / "r"
    run.mkdir()
    rel = raw_response_path(row["row_id"], 1)
    first = _failure(run, protocol, row, _envelope(_dup(row["row_id"])), rel=rel)
    second = _failure(run, protocol, row, _envelope(row["row_id"][:-3]), rel=rel)
    assert first["raw_response_sha256"] != second["raw_response_sha256"]
    decision = _assess(row, run, protocol, rows, [first, second])
    assert decision.recoverable
    assert decision.raw_response_sha256 == second["raw_response_sha256"]
    assert decision.correction_kind == CORRECTION_TRUNCATED


def test_an_already_recovered_row_is_reported_and_not_reappended(
        tmp_path, protocol_dir, packet):
    root, manifest = protocol_dir
    protocol = load_protocol(AXIS2, directory=root, manifest=manifest)
    _, rows = packet
    row = rows[0]
    run = tmp_path / "r"
    run.mkdir()
    failure = _failure(run, protocol, row, _envelope(_dup(row["row_id"])))
    decision = _assess(row, run, protocol, rows, [failure],
                       recorded_ok=[row["row_id"]])
    assert decision.already_recovered is True
    assert decision.recoverable is False
    assert decision.record is None
    assert decision.blocked_by == ()



def test_an_unexplained_echo_with_perfect_provenance_is_still_refused(
        tmp_path, protocol_dir, packet):
    """Every digest binds, and the echo is still not corrected.

    ``row_id + "AA"`` appends a block that repeats nothing beside it. The
    provenance chain is intact -- this is exactly the row a "closest match"
    rescue would take -- and it stays failed, because an unexplained echo is
    not a transcription error.
    """
    root, manifest = protocol_dir
    protocol = load_protocol(AXIS2, directory=root, manifest=manifest)
    _, rows = packet
    row = rows[0]
    run = tmp_path / "r"
    run.mkdir()
    failure = _failure(run, protocol, row, _envelope(row["row_id"] + "AA"))
    decision = _assess(row, run, protocol, rows, [failure])
    assert not decision.recoverable
    assert decision.record is None
    assert decision.gates["echo_is_not_another_packet_id"] is True
    assert "not a recognised transcription malformation" in decision.reason


# =========================================================== 5. blindness
def test_the_decision_does_not_change_with_the_verdict(tmp_path, protocol_dir,
                                                       packet):
    """Every allowed value, same gates, same correction. Nothing else changes."""
    root, manifest = protocol_dir
    protocol = load_protocol(AXIS2, directory=root, manifest=manifest)
    _, rows = packet
    row = rows[0]
    seen = set()
    for value in T.AXIS_VALUES[AXIS2]:
        run = tmp_path / f"r_{value}"
        run.mkdir()
        failure = _failure(run, protocol, row,
                           _envelope(_dup(row["row_id"]), verdict=value))
        decision = _assess(row, run, protocol, rows, [failure])
        assert decision.recoverable
        assert decision.gates["correction_is_verdict_blind"] is True
        # Everything about the correction except the digests, which differ
        # because the *bytes* differ -- a different verdict is a different
        # body. What must not move is the decision and the mapping.
        correction = dict(decision.record["item_id_correction"])
        evidence = correction.pop("evidence")
        seen.add((decision.correction_kind, tuple(sorted(decision.gates.items())),
                  json.dumps(correction, sort_keys=True),
                  json.dumps(evidence["gates"], sort_keys=True),
                  evidence["packet_rows_matching_request_sha256"],
                  evidence["raw_response_path"]))
    assert len(seen) == 1, "the recovery decision moved with the verdict"


def test_the_printable_summary_carries_no_verdict(tmp_path, protocol_dir, packet):
    root, manifest = protocol_dir
    protocol = load_protocol(AXIS2, directory=root, manifest=manifest)
    _, rows = packet
    row = rows[0]
    run = tmp_path / "r"
    run.mkdir()
    failure = _failure(run, protocol, row,
                       _envelope(_dup(row["row_id"]), verdict="DR"))
    decision = _assess(row, run, protocol, rows, [failure])
    summary = decision.summary()
    assert "parsed_result" not in summary and "record" not in summary
    blob = json.dumps(summary)
    for value in T.AXIS_VALUES[AXIS2]:
        assert f'"{value}"' not in blob
    assert "because" not in blob              # the judge's free-text reason
    assert row["row_id"] in blob
    # and the verdict really is in the record the summary omits
    assert decision.record["parsed_result"][T.AXIS_VERDICT_FIELD[AXIS2]] == "DR"


# ========================================================== 6. idempotence
def _cli(packet_dir, protocol_root, run_dir, *extra):
    module = _module("recover_failed_rows")
    return module, module.main(["--packet", str(packet_dir),
                                "--protocol-dir", str(protocol_root),
                                "--run-dir", str(run_dir), *extra])


@pytest.fixture
def live_run(tmp_path, protocol_dir, packet, monkeypatch):
    """A run directory with one recoverable row and one that must stay failed."""
    root, manifest = protocol_dir
    protocol = load_protocol(AXIS2, directory=root, manifest=manifest)
    packet_dir, rows = packet
    run = tmp_path / "packet__axis2__sonnet"
    run.mkdir()
    good = _failure(run, protocol, rows[0],
                    _envelope(rows[0]["row_id"] + rows[0]["row_id"][-6:]))
    bad = _failure(run, protocol, rows[1], _empty_envelope(),
                   error="EmptyJudgeResponse")
    _run_dir(tmp_path, protocol, [good, bad], name="packet__axis2__sonnet")
    module = _module("recover_failed_rows")
    monkeypatch.setattr(module, "load_protocol_manifest", lambda *a, **k: manifest)
    return run, packet_dir, root, rows


def test_a_dry_run_writes_nothing_and_reports_both_outcomes(live_run, capsys):
    run, packet_dir, root, rows = live_run
    before = {p: p.read_bytes() for p in sorted(run.rglob("*")) if p.is_file()}
    module, code = _cli(packet_dir, root, run, "--dry-run")
    assert code == 2                          # one row stays unresolved
    assert {p: p.read_bytes() for p in sorted(run.rglob("*")) if p.is_file()} == before
    assert not (run / "recovery_manifest.json").exists()
    out = capsys.readouterr().out
    assert "RECOVERABLE" in out and "STILL-UNRESOLVED" in out
    assert rows[0]["row_id"] in out and rows[1]["row_id"] in out
    for value in T.AXIS_VALUES[AXIS2]:
        assert f" {value} " not in out


def test_recovery_appends_once_and_a_second_run_is_a_no_op(live_run):
    run, packet_dir, root, rows = live_run
    failures_before = (run / "failures.jsonl").read_bytes()
    raw_before = {p.name: p.read_bytes() for p in sorted((run / "raw").iterdir())}

    module, code = _cli(packet_dir, root, run)
    assert code == 2
    results = read_jsonl(run / "results.jsonl")
    assert [r["row_id"] for r in results] == [rows[0]["row_id"]]
    assert results[0]["status"] == "OK"

    module, code = _cli(packet_dir, root, run)
    assert code == 2
    assert [r["row_id"] for r in read_jsonl(run / "results.jsonl")] == \
        [rows[0]["row_id"]]                   # not appended twice

    # history is untouched, and the resume key stays unique
    assert (run / "failures.jsonl").read_bytes() == failures_before
    assert {p.name: p.read_bytes() for p in sorted((run / "raw").iterdir())} == \
        raw_before
    runner = _module("run_judge")
    assert set(runner.already_done(run / "results.jsonl")) == {rows[0]["row_id"]}
    manifest = json.loads((run / "recovery_manifest.json").read_text(encoding="utf-8"))
    assert manifest["recovered"] == []        # the second run recovered nothing
    assert manifest["already_recovered"] == [rows[0]["row_id"]]
    assert manifest["still_unresolved"] == [rows[1]["row_id"]]
    assert manifest["network_calls"] == 0


def test_a_directory_whose_manifest_and_records_disagree_is_refused(live_run):
    run, packet_dir, root, rows = live_run
    path = run / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["judge"] = "opus"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    module, code = _cli(packet_dir, root, run)
    assert code == 1


# ============================================================== 7. hygiene
def test_no_api_is_called_by_the_offline_recovery_command(live_run, monkeypatch):
    """5. Every socket the process could open is armed to fail. It still succeeds."""
    def refuse(*args, **kwargs):
        raise AssertionError("the offline recovery opened a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    run, packet_dir, root, rows = live_run
    module, code = _cli(packet_dir, root, run)
    assert code == 2
    assert [r["row_id"] for r in read_jsonl(run / "results.jsonl")] == \
        [rows[0]["row_id"]]


def test_the_recovery_module_imports_nothing_that_can_reach_a_network():
    """The static half of the same claim: no transport is even in scope."""
    tree = ast.parse(RECOVERY_SCRIPT.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"urllib", "requests", "http", "socket", "ssl",
                           "httpx", "aiohttp", "ftplib", "telnetlib"}


def test_the_recovery_module_never_opens_history_for_writing():
    """``failures.jsonl`` and ``raw/`` are read-only to this script."""
    source = RECOVERY_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    written = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "open"):
            continue
        mode = next((a.value for a in node.keywords if a.arg == "mode"), None)
        if mode is None and len(node.args) > 1:
            mode = node.args[1]
        text = ast.get_source_segment(source, node) or ""
        if isinstance(mode, ast.Constant) and set(str(mode.value)) & set("wax+"):
            written.append(text)
    assert len(written) == 1, written
    assert "results.jsonl" in written[0]
    assert not any(name in source for name in
                   ('"failures.jsonl", "a"', "unlink", "rename", "shutil"))


def test_the_runner_never_writes_a_raw_path_it_did_not_choose():
    """``preserve_raw`` takes the path; it does not compose one of its own."""
    source = RUNNER_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "preserve_raw")
    assert [a.arg for a in function.args.args] == ["out_dir", "rel_path", "reply"]
    body = ast.get_source_segment(source, function) or ""
    assert "f\"{row_id}.json\"" not in body
    assert "path.exists()" in body            # the refusal to overwrite
