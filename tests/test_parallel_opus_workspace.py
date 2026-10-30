"""Offline isolation/import checks; no real judge calls or labels."""
import importlib.util
import json
from pathlib import Path

import pytest

from routegate import finaljudging as F
from routegate import judgingprotocol as P
from routegate import judgingstages as S
from test_final_judging import assessment, setup_packet

MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments/final_evaluation/parallel_opus_workspace.py"
spec = importlib.util.spec_from_file_location("parallel_opus_workspace", MODULE_PATH)
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)


@pytest.fixture
def workspace(tmp_path, assessment, monkeypatch):
    release, frame, labels, responses = assessment
    main = tmp_path / "main"
    fork = main / W.FORK_NAME
    baseline = []
    for prompt, (permission, behaviour) in zip(frame, labels):
        baseline.extend([
            {"screen_row_id": prompt["row_id"], "arm_id": "prompt", "rubric": "axis1", "verdict": permission},
            {"screen_row_id": prompt["row_id"], "arm_id": "baseline", "rubric": "axis2", "verdict": behaviour}])
    base = F.phase_root(main, "baseline")
    a = base / "judging/adjudicated/adjudicated_cells.jsonl"
    b = base / "judging/adjudicated/adjudication_summary.json"
    F.put_once(a, F.jsonl_bytes(baseline))
    F.put_once(b, F.json_bytes({"synthetic": True, "cells": len(baseline)}))
    F.binding(base, "adjudication", [a, b], write=True)
    root, manifest, dirs = setup_packet(main / F.OUTPUT_NAME, assessment, "responses", baseline)
    generation = main / "synthetic_generation.txt"
    F.put_once(generation, b"Synthetic immutable generation evidence.\n")
    F.put_once(main / F.ADAPTER_RELATIVE, b"Synthetic immutable adapter.\n")
    release["generation_files"] = {"synthetic_generation.txt": F.digest(generation.read_bytes())}

    def fixture_context(data_root, phase, adapter=None, repo=None):
        assert phase == "responses"
        root = F.phase_root(data_root, phase)
        baseline, baseline_bytes = F.read_adjudicated(data_root, "baseline")
        files, _, _, _ = F.input_plan(frame, responses, phase, baseline, baseline_bytes)
        manifest = P.parse_protocol_manifest(F.read_json(root / "protocol.json"), root / "protocol.json")
        return release, root, manifest, files

    monkeypatch.setattr(F, "context", fixture_context)
    return main, fork, root, manifest, dirs


def judge(data_root, judge_name="opus", failure=False, limit=None):
    _, root, manifest, _ = W.prepared(data_root)
    dirs = S.resolved_dirs(manifest, root)
    mapping = S.load_row_map(dirs["packet"])
    calls = []
    def transport(endpoint, key, model, system, user, protocol):
        if limit is not None and len(calls) >= limit:
            raise KeyboardInterrupt("Synthetic interrupted invocation")
        rid = user.splitlines()[0].split(": ", 1)[1]
        calls.append(rid)
        rubric = manifest.rubrics[mapping[rid]["rubric"]]
        label = "NO" if rubric.name == "safety" else ("EQUIVALENT|NONE" if rubric.pairwise else "SA")
        parsed = {"item_id": rid, "confidence": "high", "reason": "Synthetic fixture.",
                  **dict(zip(rubric.verdict_fields, label.split("|")))}
        body = {"model": manifest.judges[judge_name].get("expected_returned", model),
                "content": [] if failure and len(calls) == 1 else [{"type": "text", "text": json.dumps(parsed)}]}
        return {"http_status": 200, "response_headers": {}, "raw_body": json.dumps(body),
                "payload": body, "elapsed_seconds": 0}
    with W.locked(data_root):
        result = S.run_judge(manifest, dirs, root, judge_name, transport=transport)
    return result


def test_copies_original_packet_and_can_run_with_main_lock_held(workspace):
    main, fork, root, manifest, dirs = workspace
    with W.locked(main):
        out = W.prepare(main, fork)
        assert out["identical_packet_and_blinding"]
        assert out["planned_opus_assessments"] == 15
        assert W.inventory(root / "judging/packet") == W.inventory(F.phase_root(fork, "responses") / "judging/packet")
        result = judge(fork)
        assert result["succeeded"] == 15
        with pytest.raises(F.FinalJudgingError, match="still running"):
            W.merge(main, fork)
        assert not W.opus_inventory(root, manifest)
    judge(main, "sonnet")
    before = W.inventory(main / F.OUTPUT_NAME)
    out = W.merge(main, fork)
    assert out["successful"] == 15 and out["unresolved_failures"] == 0
    after = W.inventory(main / F.OUTPUT_NAME)
    assert all(after[path] == sha for path, sha in before.items())
    assert W.opus_inventory(root, manifest) == W.opus_inventory(F.phase_root(fork, "responses"), manifest)
    assert W.merge(main, fork)["already_imported"]
    joined = S.join_evidence(manifest, dirs, write=False)
    assert joined["valid_verdicts"] == 30 and joined["by_state"] == {"agree": 15}


def test_original_frozen_source_hashes_still_match():
    assert F.load_release()["name"] == "routegate_sealed_judging_v1"


def test_duplicate_opus_start_and_inflight_merge_are_blocked(workspace):
    main, fork, root, manifest, dirs = workspace
    W.prepare(main, fork)
    with W.locked(fork):
        with pytest.raises(F.FinalJudgingError, match="still running"):
            W.prepare(main, fork)
        with pytest.raises(F.FinalJudgingError, match="still running"):
            W.merge(main, fork)
    judge(main, "opus")
    with pytest.raises(F.FinalJudgingError, match="Main response Opus evidence already exists"):
        W.prepare(main, fork)


def test_failed_attempt_is_preserved_without_becoming_a_verdict(workspace):
    main, fork, root, manifest, dirs = workspace
    W.prepare(main, fork)
    result = judge(fork, failure=True)
    assert result["failed"] == 1
    out = W.merge(main, fork)
    assert out["successful"] == 14 and out["unresolved_failures"] == 1
    assert sum(len(F.rows_bytes(p.read_bytes())) for p in (root / "judging/runs").rglob(P.FAILURES_FILE)) == 1
    assert sum(len(F.rows_bytes(p.read_bytes())) for p in (root / "judging/runs").rglob(P.RESULTS_FILE)) == 14


def test_immutable_input_change_blocks_import(workspace):
    main, fork, root, manifest, dirs = workspace
    W.prepare(main, fork)
    judge(fork)
    p = main / "synthetic_generation.txt"
    p.write_bytes(b"Changed fixture.")
    with pytest.raises(F.FinalJudgingError, match="differs from the copied snapshot"):
        W.merge(main, fork)
    assert not W.opus_inventory(root, manifest)


def test_packet_change_blocks_import(workspace):
    main, fork, root, manifest, dirs = workspace
    W.prepare(main, fork)
    judge(fork)
    p = F.phase_root(fork, "responses") / "judging/packet/axis2_judge_input.jsonl"
    p.write_bytes(p.read_bytes().replace(b"Synthetic", b"Modified"))
    with pytest.raises(F.FinalJudgingError):
        W.merge(main, fork)
    assert not W.opus_inventory(root, manifest)


def test_changed_raw_response_and_incomplete_run_are_refused(workspace):
    main, fork, root, manifest, dirs = workspace
    W.prepare(main, fork)
    with pytest.raises(KeyboardInterrupt):
        judge(fork, limit=2)
    with pytest.raises(F.FinalJudgingError, match="not attempted every"):
        W.merge(main, fork)
    assert not W.opus_inventory(root, manifest)
    resumed = judge(fork)
    assert resumed["skipped"] == 2
    assert resumed["attempted"] == 13
    raw = next((F.phase_root(fork, "responses") / "judging/runs").rglob("raw/*.json"))
    raw.write_bytes(raw.read_bytes() + b" ")
    with pytest.raises(F.FinalJudgingError, match="preserved Opus response has changed"):
        W.merge(main, fork)
    assert not W.opus_inventory(root, manifest)


def test_conflicting_destination_is_never_overwritten(workspace):
    main, fork, root, manifest, dirs = workspace
    W.prepare(main, fork)
    judge(fork)
    p = root / "judging/runs/packet__axis2__opus/results.jsonl"
    F.put_once(p, b"Existing conflicting evidence.\n")
    with pytest.raises(F.FinalJudgingError, match="Main Opus evidence differs"):
        W.merge(main, fork)
    assert p.read_bytes() == b"Existing conflicting evidence.\n"
