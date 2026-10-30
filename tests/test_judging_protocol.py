"""The common judging engine: what a manifest may say, and what it may not.

Every fixture here is synthetic and lives in ``tmp_path``. No test reads the
data root, calls an API, loads a model or touches a completed run's artifacts.
The judge stage is driven through an injected transport, so the billed path is
exercised end to end without a network or a credential.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "experiments" / "judging" / "make_demo_fixture.py"

import routegate.judgingprotocol as P                            # noqa: E402
import routegate.judgingstages as S                              # noqa: E402
import routegate.triplets as T                                   # noqa: E402
from routegate.validation import SealedSetError                  # noqa: E402


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)                              # type: ignore
    return module


demo = _load(FIXTURE, "_judging_demo_fixture")


@pytest.fixture(autouse=True)
def _fake_credentials(monkeypatch):
    """A placeholder key and endpoint, so the billed stage can be driven.

    Nothing is ever sent: every test injects its own transport. The values are
    obviously not credentials, and the redaction test supplies its own.
    """
    monkeypatch.setenv(T.ENV_API_KEY_HISTORICAL, "fixture-not-a-real-key")
    monkeypatch.setenv(T.ENV_ENDPOINT, "http://fixture.invalid")


@pytest.fixture
def fixture(tmp_path):
    """A synthetic experiment: 6 pairwise cells, 5 single-response cells."""
    manifest_path = demo.build(tmp_path)
    manifest = P.load_manifest(manifest_path)
    dirs = S.resolved_dirs(manifest, tmp_path)
    return {"root": tmp_path, "manifest": manifest,
            "manifest_path": manifest_path, "dirs": dirs}


def _edit(fixture, mutate):
    """Rewrite the manifest with one change, and reload it."""
    raw = json.loads(fixture["manifest_path"].read_text(encoding="utf-8"))
    mutate(raw)
    fixture["manifest_path"].write_text(json.dumps(raw), encoding="utf-8")
    return P.load_manifest(fixture["manifest_path"])


def _reply(row_id, model, values, blocks=1):
    """A judge reply in the gateway's envelope shape.

    ``model`` is echoed back, because the runner refuses a reply from a model
    it did not ask for -- a gate worth not defeating in the fixture.
    """
    payload = {"item_id": row_id, **values}
    content = ([{"type": "text", "text": json.dumps(payload)}]
               if blocks else [])
    body = {"model": model, "content": content}
    return {"http_status": 200, "response_headers": {},
            "raw_body": json.dumps(body), "payload": body,
            "elapsed_seconds": 0.01}


def _transport(answers, empty=(), wrong_echo=()):
    """A fake gateway. ``answers`` maps row_id -> verdict field values."""
    def call(endpoint, api_key, model, system, user, manifest):
        row_id = json.loads(user.splitlines()[0])["item_id"]
        if row_id in empty:
            return _reply(row_id, model, {}, blocks=0)
        values = answers[row_id]
        if row_id in wrong_echo:
            bad = row_id[:-1] + ("5" if row_id[-1] != "5" else "4")
            payload = {"item_id": bad, **values}
            body = {"model": model,
                    "content": [{"type": "text",
                                 "text": json.dumps(payload)}]}
            return {"http_status": 200, "response_headers": {},
                    "raw_body": json.dumps(body), "payload": body,
                    "elapsed_seconds": 0.01}
        return _reply(row_id, model, values)
    return call


def _answers(fixture, split=(), judge_shift=False):
    """Verdicts per row; ``split`` names rows the judges differ on."""
    manifest, dirs = fixture["manifest"], fixture["dirs"]
    out = {}
    for name, rubric in manifest.rubrics.items():
        for row in S.load_judge_rows(dirs["packet"], name):
            row_id = row["row_id"]
            if rubric.pairwise:
                base = {"preference": "EQUIVALENT", "difference": "NONE"}
                other = {"preference": "A_BETTER", "difference": "MINOR"}
            else:
                base = {"harmful": "NO"}
                other = {"harmful": "YES"}
            use = other if (judge_shift and row_id in split) else base
            out[row_id] = use
    return out


def _run_both(fixture, answers, empty=(), wrong=()):
    manifest, dirs = fixture["manifest"], fixture["dirs"]
    for judge in manifest.judge_order:
        S.run_judge(manifest, dirs, fixture["root"], judge, write=True,
                    transport=_transport(answers[judge], empty.get(judge, ()),
                                         wrong.get(judge, ())))


# ------------------------------------------------------------ manifest gates
def test_a_manifest_of_another_dialect_is_refused(fixture):
    with pytest.raises(P.JudgingProtocolError) as exc:
        _edit(fixture, lambda m: m.__setitem__("schema", "something_else"))
    assert "does not guess" in str(exc.value)


def test_a_manifest_carrying_a_credential_is_refused(fixture):
    with pytest.raises(P.JudgingProtocolError) as exc:
        _edit(fixture, lambda m: m["judges"].__setitem__(
            "api_key", "sk-ant-NOTREAL-000"))
    assert "credential" in str(exc.value)


def test_a_credential_shaped_value_anywhere_is_refused(fixture):
    with pytest.raises(P.JudgingProtocolError) as exc:
        _edit(fixture, lambda m: m.__setitem__("description",
                                               "sk-ant-NOTREAL-0001"))
    assert "credential-shaped" in str(exc.value)


def test_a_manifest_naming_the_sealed_set_is_refused(fixture):
    with pytest.raises(SealedSetError):
        _edit(fixture, lambda m: m["inputs"]["candidate_records"].__setitem__(
            "path", "sealed_eval/holdout.jsonl"))


def test_a_manifest_cannot_make_a_failure_into_a_verdict(fixture):
    for kind in ("missing", "malformed", "transport_failure"):
        with pytest.raises(P.JudgingProtocolError) as exc:
            _edit(fixture,
                  lambda m, k=kind: m["adjudication"].__setitem__(k, "accept"))
        assert "no manifest value that turns one into a verdict" in str(
            exc.value)
        _edit(fixture, lambda m, k=kind: m["adjudication"].__setitem__(
            k, "unresolved_technical"))


def test_a_manifest_cannot_route_a_disagreement_away_from_a_human(fixture):
    with pytest.raises(P.JudgingProtocolError):
        _edit(fixture, lambda m: m["adjudication"].__setitem__(
            "disagreement", "take_the_first_judge"))


def test_a_manifest_cannot_invent_an_adjudication_procedure(fixture):
    with pytest.raises(P.JudgingProtocolError) as exc:
        _edit(fixture, lambda m: m["adjudication"].__setitem__(
            "procedure", "whatever_we_like"))
    assert "not one of" in str(exc.value)


def test_counts_that_do_not_add_up_are_refused(fixture):
    with pytest.raises(P.JudgingProtocolError) as exc:
        _edit(fixture,
              lambda m: m["coverage"].__setitem__("planned_calls", 99))
    assert "planned_calls" in str(exc.value)


def test_a_row_prefix_shared_with_another_packet_is_refused(fixture):
    with pytest.raises(P.JudgingProtocolError) as exc:
        _edit(fixture, lambda m: m["identity"]["row_prefixes"].__setitem__(
            "demosafety", "TSJ1SAFETY"))
    assert "another packet" in str(exc.value)


def test_sampling_must_match_the_frozen_transport(fixture):
    with pytest.raises(P.JudgingProtocolError) as exc:
        _edit(fixture, lambda m: m["sampling"].__setitem__("temperature", 0.7))
    assert "frozen transport" in str(exc.value)


def test_a_pairwise_rubric_needs_a_reference_arm(fixture):
    with pytest.raises(P.JudgingProtocolError) as exc:
        _edit(fixture, lambda m: m["arms"].__setitem__("reference", []))
    assert "reference arm" in str(exc.value)


def test_a_drifted_input_digest_is_refused(fixture):
    path = fixture["root"] / "candidate_records.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(P.JudgingProtocolError) as exc:
        S.preflight(fixture["manifest"], fixture["dirs"], fixture["root"],
                    fixture["root"])
    assert "drifted input" in str(exc.value)


def test_a_drifted_rubric_digest_is_refused(fixture):
    path = fixture["root"] / "frozen" / "demosafety_instructions.txt"
    path.write_text("edited\n", encoding="utf-8")
    with pytest.raises(P.JudgingProtocolError) as exc:
        S.preflight(fixture["manifest"], fixture["dirs"], fixture["root"],
                    fixture["root"])
    assert "never edited in place" in str(exc.value)


# ------------------------------------------------------------------ blindness
def test_every_judge_row_carries_only_its_whitelist(fixture):
    manifest, dirs = fixture["manifest"], fixture["dirs"]
    S.build_packet(manifest, dirs, fixture["root"])
    corpus = {str(r["screen_row_id"]) for r in
              T.read_jsonl(fixture["root"] / "candidate_records.jsonl")}
    for name, rubric in manifest.rubrics.items():
        rows = S.load_judge_rows(dirs["packet"], name)
        assert rows
        for row in rows:
            assert set(row) == set(rubric.row_fields)
            S.assert_row_is_blind(row, rubric, corpus)
        blob = json.dumps(rows)
        for banned in ("arm_id", "row_class", "screen_row_id", "origin",
                       "demo_static_repair", "demo_no_steering"):
            assert banned not in blob


def test_the_orientation_lives_only_in_the_private_map(fixture):
    manifest, dirs = fixture["manifest"], fixture["dirs"]
    S.build_packet(manifest, dirs, fixture["root"])
    rows = json.dumps(S.load_judge_rows(dirs["packet"], "demoquality"))
    assert "origin" not in rows
    private = S.load_row_map(dirs["packet"])
    pairwise = [e for e in private.values() if "response_a_origin" in e]
    assert len(pairwise) == 6


def test_two_builds_mint_different_ids_and_orientations(fixture, tmp_path):
    manifest, dirs = fixture["manifest"], fixture["dirs"]
    first = S.build_packet(manifest, dirs, fixture["root"], write=False)
    second = S.build_packet(manifest, dirs, fixture["root"], write=False)
    assert first["rows"] == second["rows"]
    # ids are drawn from `secrets`, so two builds cannot coincide
    assert first["created_utc"] is not None


# ------------------------------------------------------------ packet identity
def test_a_packet_is_not_rebuilt_once_judging_has_started(fixture):
    manifest, dirs = fixture["manifest"], fixture["dirs"]
    S.build_packet(manifest, dirs, fixture["root"])
    run = Path(dirs["runs"]) / "packet__demosafety__sonnet"
    run.mkdir(parents=True)
    (run / "results.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(P.JudgingProtocolError) as exc:
        S.build_packet(manifest, dirs, fixture["root"])
    assert "orphaning every verdict already billed" in str(exc.value)


# ------------------------------------------------------------ echo validation
def test_an_exact_echo_is_required(fixture):
    rubric = fixture["manifest"].rubrics["demosafety"]
    good = P.validate_reply({"item_id": "DEMOSAFETY01", "harmful": "NO"},
                            rubric, "DEMOSAFETY01")
    assert good["harmful"] == "NO"
    with pytest.raises(P.JudgingProtocolError) as exc:
        P.validate_reply({"item_id": "DEMOSAFETY02", "harmful": "NO"},
                         rubric, "DEMOSAFETY01")
    assert "never repaired and never accepted" in str(exc.value)


def test_a_verdict_outside_the_vocabulary_is_refused(fixture):
    rubric = fixture["manifest"].rubrics["demosafety"]
    with pytest.raises(P.JudgingProtocolError):
        P.validate_reply({"item_id": "X", "harmful": "PROBABLY"}, rubric, "X")


def test_a_non_object_reply_is_refused(fixture):
    rubric = fixture["manifest"].rubrics["demosafety"]
    with pytest.raises(P.JudgingProtocolError) as exc:
        P.validate_reply("YES", rubric, "X")
    assert "not converted into a verdict" in str(exc.value)


# ------------------------------------------------------- judging, resumability
def test_a_successful_row_is_never_called_twice(fixture):
    manifest, dirs = fixture["manifest"], fixture["dirs"]
    S.build_packet(manifest, dirs, fixture["root"])
    answers = _answers(fixture)
    calls = []

    def counting(endpoint, api_key, model, system, user, mf):
        row_id = json.loads(user.splitlines()[0])["item_id"]
        calls.append(row_id)
        return _reply(row_id, model, answers[row_id])

    first = S.run_judge(manifest, dirs, fixture["root"], "sonnet",
                        write=True, transport=counting)
    assert first["succeeded"] == 11 and len(calls) == 11
    second = S.run_judge(manifest, dirs, fixture["root"], "sonnet",
                         write=True, transport=counting)
    assert second["attempted"] == 0 and second["skipped"] == 11
    assert len(calls) == 11          # not one extra billed call


def test_raw_evidence_is_immutable(fixture):
    manifest, dirs = fixture["manifest"], fixture["dirs"]
    S.build_packet(manifest, dirs, fixture["root"])
    out = Path(dirs["runs"]) / "packet__demosafety__sonnet"
    out.mkdir(parents=True)
    reply = _reply("DEMOSAFETY01", "m", {"harmful": "NO"})
    P.preserve_raw(out, "raw/x.json", reply)
    with pytest.raises(P.JudgingProtocolError) as exc:
        P.preserve_raw(out, "raw/x.json", reply)
    assert "immutable" in str(exc.value)


def test_a_dry_run_writes_nothing_and_calls_nothing(fixture):
    manifest, dirs = fixture["manifest"], fixture["dirs"]
    S.build_packet(manifest, dirs, fixture["root"])

    def explode(*a, **k):                       # pragma: no cover
        raise AssertionError("a dry run must not call the gateway")

    out = S.run_judge(manifest, dirs, fixture["root"], "sonnet",
                      write=False, transport=explode)
    assert out["attempted"] == 0
    assert not Path(dirs["runs"]).exists() or not any(
        Path(dirs["runs"]).iterdir())
    rendered = S.dry_run(manifest, dirs, fixture["root"])
    assert rendered["calls"] == 22 and rendered["called_nothing"]


def test_the_judge_stage_refuses_to_run_without_a_credential(fixture,
                                                             monkeypatch):
    manifest, dirs = fixture["manifest"], fixture["dirs"]
    S.build_packet(manifest, dirs, fixture["root"])
    monkeypatch.delenv(T.ENV_API_KEY_HISTORICAL, raising=False)
    monkeypatch.delenv(T.ENV_ENDPOINT, raising=False)
    with pytest.raises(P.JudgingProtocolError) as exc:
        S.run_judge(manifest, dirs, fixture["root"], "sonnet", write=True)
    assert "is ever read from a manifest or a file" in str(exc.value)


# ----------------------------------------------------- failures stay technical
def _judged(fixture, monkeypatch, empty=None, wrong=None, split=()):
    manifest, dirs = fixture["manifest"], fixture["dirs"]
    monkeypatch.setenv(T.ENV_API_KEY_HISTORICAL, "not-a-real-key")
    monkeypatch.setenv(T.ENV_ENDPOINT, "http://fixture.invalid")
    S.build_packet(manifest, dirs, fixture["root"])
    base = _answers(fixture)
    per_judge = {"sonnet": base,
                 "opus": _answers(fixture, split=split, judge_shift=True)}
    _run_both(fixture, per_judge, empty or {}, wrong or {})
    return per_judge


def _first_row(fixture, rubric):
    return S.load_judge_rows(fixture["dirs"]["packet"], rubric)[0]["row_id"]


def test_an_empty_reply_never_becomes_a_verdict(fixture, monkeypatch):
    row = None
    manifest, dirs = fixture["manifest"], fixture["dirs"]
    monkeypatch.setenv(T.ENV_API_KEY_HISTORICAL, "k")
    monkeypatch.setenv(T.ENV_ENDPOINT, "http://fixture.invalid")
    S.build_packet(manifest, dirs, fixture["root"])
    row = _first_row(fixture, "demosafety")
    answers = _answers(fixture)
    S.run_judge(manifest, dirs, fixture["root"], "sonnet", write=True,
                transport=_transport(answers, empty=(row,)))
    results = Path(dirs["runs"]) / "packet__demosafety__sonnet"
    ok = {r["row_id"] for r in T.read_jsonl(results / "results.jsonl")}
    failed = list(T.read_jsonl(results / "failures.jsonl"))
    assert row not in ok
    assert len(failed) == 1 and failed[0]["row_id"] == row
    assert failed[0]["status"] == "FAILED"


def test_a_mismatched_echo_never_becomes_a_verdict(fixture, monkeypatch):
    manifest, dirs = fixture["manifest"], fixture["dirs"]
    monkeypatch.setenv(T.ENV_API_KEY_HISTORICAL, "k")
    monkeypatch.setenv(T.ENV_ENDPOINT, "http://fixture.invalid")
    S.build_packet(manifest, dirs, fixture["root"])
    row = _first_row(fixture, "demosafety")
    answers = _answers(fixture)
    S.run_judge(manifest, dirs, fixture["root"], "sonnet", write=True,
                transport=_transport(answers, wrong_echo=(row,)))
    results = Path(dirs["runs"]) / "packet__demosafety__sonnet"
    ok = {r["row_id"] for r in T.read_jsonl(results / "results.jsonl")}
    failed = list(T.read_jsonl(results / "failures.jsonl"))
    assert row not in ok
    assert "echoes item_id" in failed[0]["error"]


def test_an_uncarried_failure_stops_the_join(fixture, monkeypatch):
    manifest, dirs = fixture["manifest"], fixture["dirs"]
    row = None
    monkeypatch.setenv(T.ENV_API_KEY_HISTORICAL, "k")
    monkeypatch.setenv(T.ENV_ENDPOINT, "http://fixture.invalid")
    S.build_packet(manifest, dirs, fixture["root"])
    row = _first_row(fixture, "demosafety")
    answers = _answers(fixture)
    for judge in manifest.judge_order:
        S.run_judge(manifest, dirs, fixture["root"], judge, write=True,
                    transport=_transport(
                        answers, empty=(row,) if judge == "opus" else ()))
    with pytest.raises(Exception) as exc:
        S.join_evidence(manifest, dirs, write=False)
    assert "not carried as technically unresolved" in str(exc.value)


def test_a_carried_failure_becomes_unresolved_technical(fixture, monkeypatch):
    manifest, dirs = fixture["manifest"], fixture["dirs"]
    monkeypatch.setenv(T.ENV_API_KEY_HISTORICAL, "k")
    monkeypatch.setenv(T.ENV_ENDPOINT, "http://fixture.invalid")
    S.build_packet(manifest, dirs, fixture["root"])
    row = _first_row(fixture, "demosafety")
    answers = _answers(fixture)
    for judge in manifest.judge_order:
        S.run_judge(manifest, dirs, fixture["root"], judge, write=True,
                    transport=_transport(
                        answers, empty=(row,) if judge == "opus" else ()))
    carried = [{"row_id": row, "failed_judge": "opus",
                "failure_type": "EmptyJudgeResponse"}]
    summary = S.join_evidence(manifest, dirs, write=True, carried=carried)
    assert summary["cells"] == 11
    assert summary["by_state"][P.STATE_UNRESOLVED_TECHNICAL] == 1
    assert summary["valid_verdicts"] == 21
    assert summary["planned_verdicts"] == 22
    cells = list(T.read_jsonl(
        Path(dirs["joined"]) / "private" / P.JOINED_CELLS_FILE))
    carried_cells = [c for c in cells
                     if c["state"] == P.STATE_UNRESOLVED_TECHNICAL]
    assert len(carried_cells) == 1
    assert carried_cells[0]["verdicts"]["opus"] is None
    assert carried_cells[0]["verdicts"]["sonnet"] is not None


# ------------------------------------------------------------- review, denoms
def _pipeline(fixture, monkeypatch, carried_row=None, split=None):
    manifest, dirs = fixture["manifest"], fixture["dirs"]
    monkeypatch.setenv(T.ENV_API_KEY_HISTORICAL, "k")
    monkeypatch.setenv(T.ENV_ENDPOINT, "http://fixture.invalid")
    S.build_packet(manifest, dirs, fixture["root"])
    split = split or {_first_row(fixture, "demoquality")}
    sonnet = _answers(fixture)
    opus = _answers(fixture, split=split, judge_shift=True)
    for judge, table in (("sonnet", sonnet), ("opus", opus)):
        S.run_judge(manifest, dirs, fixture["root"], judge, write=True,
                    transport=_transport(
                        table,
                        empty=((carried_row,)
                               if (carried_row and judge == "opus") else ())))
    carried = ([{"row_id": carried_row, "failed_judge": "opus",
                 "failure_type": "EmptyJudgeResponse"}] if carried_row else [])
    S.join_evidence(manifest, dirs, write=True, carried=carried)
    return split


def test_only_genuine_disagreements_reach_the_reviewer(fixture, monkeypatch):
    manifest, dirs = fixture["manifest"], fixture["dirs"]
    carried = _first_row(fixture, "demosafety") if False else None
    S.build_packet(manifest, dirs, fixture["root"])
    row = S.load_judge_rows(dirs["packet"], "demosafety")[0]["row_id"]
    monkeypatch.setenv(T.ENV_API_KEY_HISTORICAL, "k")
    monkeypatch.setenv(T.ENV_ENDPOINT, "http://fixture.invalid")
    split = {S.load_judge_rows(dirs["packet"], "demoquality")[0]["row_id"]}
    sonnet = _answers(fixture)
    opus = _answers(fixture, split=split, judge_shift=True)
    for judge, table in (("sonnet", sonnet), ("opus", opus)):
        S.run_judge(manifest, dirs, fixture["root"], judge, write=True,
                    transport=_transport(
                        table, empty=(row,) if judge == "opus" else ()))
    S.join_evidence(manifest, dirs, write=True,
                    carried=[{"row_id": row, "failed_judge": "opus",
                              "failure_type": "EmptyJudgeResponse"}])
    summary = S.export_review(manifest, dirs, write=True)

    assert summary["rows"] == 1                      # the one disagreement
    assert summary["technically_unresolved_excluded"] == 1
    exported = json.dumps(list(T.read_jsonl(
        Path(dirs["review"]) / P.REVIEW_INPUT_FILE)))
    assert row not in exported                       # the technical cell
    assert "demo_corpus" not in exported             # no corpus id
    assert "arm_id" not in exported and "origin" not in exported


def test_a_human_cannot_replace_a_missing_judge_verdict(fixture, monkeypatch):
    manifest, dirs = fixture["manifest"], fixture["dirs"]
    S.build_packet(manifest, dirs, fixture["root"])
    row = S.load_judge_rows(dirs["packet"], "demosafety")[0]["row_id"]
    monkeypatch.setenv(T.ENV_API_KEY_HISTORICAL, "k")
    monkeypatch.setenv(T.ENV_ENDPOINT, "http://fixture.invalid")
    split = {S.load_judge_rows(dirs["packet"], "demoquality")[0]["row_id"]}
    for judge, table in (("sonnet", _answers(fixture)),
                         ("opus", _answers(fixture, split=split,
                                           judge_shift=True))):
        S.run_judge(manifest, dirs, fixture["root"], judge, write=True,
                    transport=_transport(
                        table, empty=(row,) if judge == "opus" else ()))
    S.join_evidence(manifest, dirs, write=True,
                    carried=[{"row_id": row, "failed_judge": "opus",
                              "failure_type": "EmptyJudgeResponse"}])
    S.export_review(manifest, dirs, write=True)

    # Forge a reviewer answer for the technical cell by pointing the review
    # map at it. The named gate must refuse it.
    map_path = Path(dirs["review"]) / "private" / P.REVIEW_MAP_FILE
    mapping = list(T.read_jsonl(map_path))
    cells = {c["row_id"]: c for c in T.read_jsonl(
        Path(dirs["joined"]) / "private" / P.JOINED_CELLS_FILE)}
    technical = cells[row]
    mapping.append({"row_id": "DEMOHUMANFORGED", "judged_row_id": row,
                    "screen_row_id": technical["screen_row_id"],
                    "arm_id": technical["arm_id"], "rubric": "demosafety"})
    T.write_jsonl(map_path, mapping)
    completed = Path(dirs["review"]) / P.REVIEW_COMPLETED_FILE
    T.write_jsonl(completed, [{"row_id": "DEMOHUMANFORGED", "harmful": "NO"}])

    with pytest.raises(Exception) as exc:
        S.import_review(manifest, dirs, completed, write=False)
    message = str(exc.value)
    # Two independent gates cover this. Whichever fires first, a human answer
    # for a cell no judge answered never reaches adjudication.
    assert ("no_human_verdict_replaces_a_missing_judge_verdict" in message
            or "no_review_of_a_cell_this_procedure_does_not_send_to_a_human"
            in message)


def test_the_report_states_explicit_denominators(fixture, monkeypatch):
    manifest, dirs = fixture["manifest"], fixture["dirs"]
    S.build_packet(manifest, dirs, fixture["root"])
    row = S.load_judge_rows(dirs["packet"], "demosafety")[0]["row_id"]
    monkeypatch.setenv(T.ENV_API_KEY_HISTORICAL, "k")
    monkeypatch.setenv(T.ENV_ENDPOINT, "http://fixture.invalid")
    for judge in manifest.judge_order:
        S.run_judge(manifest, dirs, fixture["root"], judge, write=True,
                    transport=_transport(
                        _answers(fixture),
                        empty=(row,) if judge == "opus" else ()))
    S.join_evidence(manifest, dirs, write=True,
                    carried=[{"row_id": row, "failed_judge": "opus",
                              "failure_type": "EmptyJudgeResponse"}])
    S.export_review(manifest, dirs, write=True)
    S.import_review(manifest, dirs, None, write=True)
    out = S.report(manifest, dirs, write=True)

    assert out["cells"] == 11 and out["planned_cells"] == 11
    assert out["planned_verdicts"] == 22
    assert out["unresolved_cells"] == 1
    assert out["unresolved_cell_ids"] == [row]
    assert out["original_protocol_complete"] is False

    t_reading = out["readings"]["T"]
    assert t_reading["planned_rows"] == 5           # every planned cell kept
    assert t_reading["rows"] == 5
    assert t_reading["decided_denominator"] == 4    # complete-case
    assert sum(t_reading["by_verdict"].values()) == 4
    for value, rate in t_reading["rates"].items():
        if rate is not None:
            assert abs(rate - t_reading["by_verdict"][value] / 4) < 1e-12
    # the unresolved cell is named by opaque id and rubric, and by nothing else
    listed = t_reading["unresolved_cells"]
    assert listed == [{"row_id": row, "rubric": "demosafety", "row_class": "T",
                       "missing_judge": "opus",
                       "failure_type": "EmptyJudgeResponse"}]
    blob = json.dumps(out)
    assert "prompt" not in blob and "response" not in blob
    assert "demo_corpus" not in blob


def test_a_technical_cell_never_carries_a_verdict(fixture, monkeypatch):
    manifest, dirs = fixture["manifest"], fixture["dirs"]
    S.build_packet(manifest, dirs, fixture["root"])
    row = S.load_judge_rows(dirs["packet"], "demosafety")[0]["row_id"]
    monkeypatch.setenv(T.ENV_API_KEY_HISTORICAL, "k")
    monkeypatch.setenv(T.ENV_ENDPOINT, "http://fixture.invalid")
    for judge in manifest.judge_order:
        S.run_judge(manifest, dirs, fixture["root"], judge, write=True,
                    transport=_transport(
                        _answers(fixture),
                        empty=(row,) if judge == "opus" else ()))
    S.join_evidence(manifest, dirs, write=True,
                    carried=[{"row_id": row, "failed_judge": "opus",
                              "failure_type": "EmptyJudgeResponse"}])
    S.import_review(manifest, dirs, None, write=True)
    records = {r["judged_row_id"]: r for r in T.read_jsonl(
        Path(dirs["adjudicated"]) / P.ADJUDICATED_CELLS_FILE)}
    assert records[row]["verdict"] is None
    assert records[row]["label_source"] == T.SOURCE_UNRESOLVED
    assert records[row]["judges_agree"] is None


# -------------------------------------------------------------- multi-arm use
def test_the_manifest_supports_several_steering_arms(fixture):
    manifest = _edit(fixture, lambda m: m["arms"].__setitem__(
        "candidate", ["demo_static_repair", "demo_cast_gate",
                      "demo_dynamic_controller", "demo_safety_projected",
                      "demo_layer_ablation_31_32"]))
    assert len(manifest.candidate_arms) == 5
    assert manifest.reference_arms == ("demo_no_steering",)


def test_registering_a_vocabulary_twice_is_fine_but_redefining_is_not(fixture):
    manifest = fixture["manifest"]
    P.register_rubric_vocabulary(manifest)
    P.register_rubric_vocabulary(manifest)          # idempotent
    changed = _edit(fixture, lambda m: m["rubrics"]["demosafety"].__setitem__(
        "values", ["YES", "NO"]))
    with pytest.raises(P.JudgingProtocolError) as exc:
        P.register_rubric_vocabulary(changed)
    assert "never redefined" in str(exc.value)


# ---------------------------------------------------------------- redaction
def test_a_transport_exception_is_redacted_before_it_is_recorded(fixture,
                                                                 monkeypatch):
    manifest, dirs = fixture["manifest"], fixture["dirs"]
    S.build_packet(manifest, dirs, fixture["root"])
    secret = "sk-ant-NOTAREALKEY-1234567890"
    monkeypatch.setenv(T.ENV_API_KEY_HISTORICAL, secret)
    monkeypatch.setenv(T.ENV_ENDPOINT, "http://fixture.invalid")

    def leaky(endpoint, api_key, model, system, user, mf):
        return {"http_status": None, "response_headers": {}, "raw_body": "",
                "payload": None, "elapsed_seconds": 0.0,
                "error": T.redact_exception_message(
                    f"ValueError: Invalid header value b'{api_key}'",
                    [api_key, endpoint])}

    S.run_judge(manifest, dirs, fixture["root"], "sonnet", write=True,
                transport=leaky)
    for path in Path(dirs["runs"]).rglob("failures.jsonl"):
        blob = path.read_text(encoding="utf-8")
        assert secret not in blob
        assert T.REDACTED in blob


def test_a_record_carrying_a_secret_is_refused_before_it_is_written():
    with pytest.raises(P.JudgingProtocolError):
        P.assert_no_secrets({"api_key": "sk-ant-x"}, "record")
    with pytest.raises(P.JudgingProtocolError):
        P.assert_no_secrets({"note": "Bearer abc"}, "record")
    P.assert_no_secrets({"paths_key": "two_site_judging_dir"}, "record")
    P.assert_no_secrets({"root_key": "x", "sha256": "a" * 64}, "record")


def test_a_technical_cell_is_excluded_even_when_every_cell_is_reviewed():
    """Under ``triplet_independent_review`` a human sees every cell -- except.

    The exclusion is not a property of the corpus procedure; it is a property
    of what a human can do. No procedure sends a cell with missing judge
    evidence to a person: there is nothing there for them to adjudicate.
    """
    cells = [
        {"screen_row_id": "r1", "arm_id": "a", "rubric": "demosafety",
         "state": P.STATE_AGREE},
        {"screen_row_id": "r2", "arm_id": "a", "rubric": "demosafety",
         "state": P.STATE_DISAGREE},
        {"screen_row_id": "r3", "arm_id": "a", "rubric": "demosafety",
         "state": P.STATE_UNRESOLVED_TECHNICAL},
    ]
    for procedure in P.PROCEDURES:
        selected = S.cells_needing_review(cells, procedure)
        assert ("r3", "a", "demosafety") not in selected
    assert S.cells_needing_review(cells, "corpus_disagreement_only") == [
        ("r2", "a", "demosafety")]
    everything = S.cells_needing_review(cells, "triplet_independent_review")
    assert len(everything) == 2
