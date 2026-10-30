"""Synthetic final-judging checks. No API calls or real evaluation data."""
import copy
import csv
import io
import json
from pathlib import Path

import pytest

from routegate import finaljudging as F
from routegate import judgingprotocol as P
from routegate import judgingstages as S
from routegate.finalrunner import blank_telemetry
from routegate.validation import SealedSetError


@pytest.fixture
def assessment(tmp_path, monkeypatch):
    release = json.loads((Path(__file__).resolve().parents[1] / F.RELEASE_FILE).read_text())
    frame = [{"row_id": "CORPUS{:02d}".format(i), "prompt_text": "Invented request number {}".format(i),
              "source_side": "harmful_source" if i < 3 else "safe_source"} for i in range(8)]
    labels = [("PERMITTED", "DR"), ("PERMITTED", "SA"), ("PERMITTED", "PA-S"),
              ("PROHIBITED", "SA"), ("AMBIGUOUS", "IR"), ("PERMITTED", "NR"),
              ("PERMITTED", "IR"), ("PERMITTED", "UM")]
    responses = {c: {p["row_id"]: {"response_text": "Synthetic {} reply {}".format(c, i)}
                     for i, p in enumerate(frame)} for c in F.CONDITIONS}
    monkeypatch.setenv("MODEL_API_KEY", "x" * 51)
    monkeypatch.setenv("ROUTEGATE_JUDGE_ENDPOINT", "https://fixture.invalid")
    return release, frame, labels, responses


def setup_packet(tmp_path, assessment, phase, baseline=None):
    release, frame, labels, responses = assessment
    root = tmp_path / phase
    files, classes, arms, refs = F.input_plan(frame, responses, phase, baseline,
                                           F.jsonl_bytes(baseline) if baseline else None)
    for name in {s["rubric"] for s in classes.values()}:
        spec = release["rubrics"][name]
        fields = "\n{response_a}\n{response_b}" if spec["pairwise"] else ("" if spec["prompt_only"] else "\n{response_text}")
        parts = {"instructions": ("Synthetic " + name + " instructions").encode(),
                 "user_template": ("ITEM_ID: {item_id}\n{prompt_text}" + fields).encode(),
                 "schema": b'{"fixture": true}'}
        for part, data in parts.items():
            spec[part + "_sha256"] = F.digest(data)
            ext = ".json" if part == "schema" else ".txt"
            F.put_once(root / "frozen/judge_rubrics" / (name + "_" + part + ext), data)
    document = F.protocol_document(release, phase, files, classes, arms, refs)
    manifest = P.parse_protocol_manifest(document, root / "protocol.json")
    F.put_once(root / "protocol.json", F.json_bytes(document))
    for name, data in files.items():
        F.put_once(root / name, data)
    dirs = S.resolved_dirs(manifest, root)
    S.build_packet(manifest, dirs, root)
    F.verify_packet(root, manifest, create_binding=True)
    return root, manifest, dirs


def fake_judges(root, manifest, dirs, assessments):
    private = S.load_row_map(dirs["packet"])
    by_id = {r["row_id"]: r for r in private.values()}
    def transport(endpoint, key, model, system, user, protocol):
        row_id = user.splitlines()[0].split(": ", 1)[1]
        entry = by_id[row_id]
        label = assessments[(entry["screen_row_id"], entry["arm_id"], entry["rubric"])]
        rubric = manifest.rubrics[entry["rubric"]]
        values = dict(zip(rubric.verdict_fields, label.split("|")))
        parsed = {"item_id": row_id, "confidence": "high", "reason": "Synthetic fixture.", **values}
        expected = next(j.get("expected_returned", model) for j in manifest.judges.values() if j["model"] == model)
        body = {"model": expected, "content": [{"type": "text", "text": json.dumps(parsed)}]}
        return {"http_status": 200, "response_headers": {}, "raw_body": json.dumps(body),
                "payload": body, "elapsed_seconds": 0}
    for judge in manifest.judge_order:
        result = S.run_judge(manifest, dirs, root, judge, transport=transport)
        assert result["failed"] == 0
    S.join_evidence(manifest, dirs)
    export = S.export_review(manifest, dirs)
    assert export["rows"] == 0
    S.import_review(manifest, dirs)
    return list(F.read_jsonl(dirs["adjudicated"] / P.ADJUDICATED_CELLS_FILE))


def test_permission_is_prompt_only_and_both_phases_use_existing_stages(tmp_path, assessment):
    release, frame, labels, responses = assessment
    root, manifest, dirs = setup_packet(tmp_path, assessment, "baseline")
    assert manifest.coverage["cells"] == 16
    for row in S.load_judge_rows(dirs["packet"], "axis1"):
        assert set(row) == {"row_id", "prompt_text"}
    assert S.dry_run(manifest, dirs, root)["calls"] == 32
    answers = {}
    for prompt, (permission, behaviour) in zip(frame, labels):
        answers[(prompt["row_id"], "prompt", "axis1")] = permission
        answers[(prompt["row_id"], "baseline", "axis2")] = behaviour
    baseline = fake_judges(root, manifest, dirs, answers)
    groups = F.derive_groups(frame, baseline)
    assert [r["group"] for r in groups] == ["E", "B", "permitted_other", "H", "permission_ambiguous", "permitted_other", "E", "permitted_other"]
    root2, mf2, dirs2 = setup_packet(tmp_path, assessment, "responses", baseline)
    assert mf2.coverage["cells"] == 15  # 2E*2 + 1B*2 + 3other*2 + 1H*3
    answers2 = {}
    for entry in S.load_row_map(dirs2["packet"]).values():
        rubric = entry["rubric"]
        label = "NO" if rubric == "safety" else ("EQUIVALENT|NONE" if rubric == "pairwise_quality" else "SA")
        answers2[(entry["screen_row_id"], entry["arm_id"], rubric)] = label
    adjudicated2 = fake_judges(root2, mf2, dirs2, answers2)
    assert len(adjudicated2) == 15
    assert S.report(mf2, dirs2)["unresolved_cells"] == 0


def test_unknown_permission_and_baseline_are_retained_separately(assessment):
    release, frame, labels, responses = assessment
    baseline = []
    for prompt, (permission, behaviour) in zip(frame, labels):
        baseline.extend([{"screen_row_id": prompt["row_id"], "arm_id": "prompt", "rubric": "axis1", "verdict": permission},
                         {"screen_row_id": prompt["row_id"], "arm_id": "baseline", "rubric": "axis2", "verdict": behaviour}])
    baseline[0]["verdict"] = None
    baseline[3]["verdict"] = None
    groups = F.derive_groups(frame, baseline)
    assert groups[0]["group"] == "permission_unresolved"
    assert groups[1]["group"] == "permitted_baseline_unresolved"
    assert len(groups) == len(frame)


def test_default_development_loader_still_refuses_final_manifest(tmp_path, assessment):
    root, manifest, dirs = setup_packet(tmp_path, assessment, "baseline")
    with pytest.raises(SealedSetError):
        P.load_manifest(root / "protocol.json")


def test_packet_text_change_is_refused(tmp_path, assessment):
    root, manifest, dirs = setup_packet(tmp_path, assessment, "baseline")
    path = dirs["packet"] / "axis2_judge_input.jsonl"
    rows = F.rows_bytes(path.read_bytes())
    rows[0]["response_text"] += " changed"
    path.write_bytes(F.jsonl_bytes(rows))
    with pytest.raises(F.FinalJudgingError, match="differs from the generated"):
        F.verify_packet(root, manifest)


def test_prompt_only_manifest_cannot_be_pairwise(tmp_path, assessment):
    root, manifest, dirs = setup_packet(tmp_path, assessment, "baseline")
    raw = copy.deepcopy(manifest.raw)
    raw["rubrics"]["axis1"]["pairwise"] = True
    with pytest.raises(P.JudgingProtocolError, match="prompt-only"):
        P.parse_protocol_manifest(raw, root / "bad.json")


def test_authentication_failure_is_recorded_before_calls_stop(tmp_path, assessment, monkeypatch):
    root, manifest, dirs = setup_packet(tmp_path, assessment, "baseline")
    called = []
    def failed(*args, **kwargs):
        called.append(1)
        return {"http_status": 403, "raw_body": "Forbidden", "payload": None,
                "response_headers": {}, "elapsed_seconds": 0, "error": "HTTP 403"}
    monkeypatch.setattr(S, "call_judge", failed)
    with pytest.raises(F.FinalJudgingError, match="Authentication failed"):
        S.run_judge(manifest, dirs, root, "sonnet", transport=F.SafeProgressTransport("sonnet"))
    assert len(called) == 1
    assert sum(len(F.rows_bytes(p.read_bytes())) for p in dirs["runs"].rglob("failures.jsonl")) == 1


@pytest.fixture
def generation_fixture(tmp_path):
    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir(); data.mkdir(); (repo / "manifests").mkdir()
    runtime = {"model": "synthetic", "revision": "synthetic", "chat_template_sha256": "fixture",
               "tokenizer_config_sha256": "fixture", "do_sample": False, "sampling_seed": None}
    F.put_once(repo / "synthetic_source.py", b"# Synthetic generation source.\n")
    gen = {"arms": [], "runtime": runtime, "record_schema": "synthetic",
           "source_sha256": {"synthetic_source.py": F.digest((repo / "synthetic_source.py").read_bytes())}}
    frame = [{"row_id": "FIXTURE0", "prompt_text": "A synthetic permitted request.", "source_side": "safe_source", "canonical_prompt_sha256": "historical-convention"},
             {"row_id": "FIXTURE1", "prompt_text": "A second synthetic request.", "source_side": "harmful_source", "canonical_prompt_sha256": "historical-convention"}]
    release = {"generation_files": {}, "expected_prompts": 2,
               "source_composition": {"safe_source": 1, "harmful_source": 1}}
    for dataset in ("tinymmlu", "tinygsm8k", "sealed"):
        for condition in F.CONDITIONS:
            name = dataset + "__" + condition; relative = "final_evaluation_v5/" + name
            arm = {"arm_id": name, "rows": 2, "output_dir": relative, "dataset": dataset, "condition": condition}
            gen["arms"].append(arm)
            rows = []
            for i, prompt in enumerate(frame):
                row = {**runtime, **blank_telemetry(), "item_id": prompt["row_id"], "item_index": i,
                       "arm_id": name, "dataset": dataset, "condition": condition, "status": "ok",
                       "protocol": "routegate_final_evaluation_v5", "schema": "synthetic",
                       "prompt_sha256": F.sha256_text(prompt["prompt_text"]), "response_text": "Synthetic answer.",
                       "mode": "off", "hooks_expected": 0, "hooks_fired": 0}
                if condition == "dynamic_raw":
                    row.update(mode="patch", hooks_expected=2, hooks_fired=2, controller_read_state=32,
                               g_R=0.5, applied_alpha_R=0.075)
                elif condition == "safety_first_routegate":
                    row.update(mode="off", gate_read_state=45, gate_p_E=0.1, gate_p_H=0.1,
                               gate_fires=False, threshold_E=0.5, threshold_H=0.1, noop_reason="gate_did_not_fire")
                rows.append(row)
            raw = F.jsonl_bytes(rows)
            complete = {"complete": True, "rows": 2, "records_sha256": F.digest(raw)}
            for filename, content in (("records.jsonl", raw), ("arm_complete.json", F.json_bytes(complete))):
                rel = relative + "/" + filename
                F.put_once(data / rel, content); release["generation_files"][rel] = F.digest(content)
    F.put_once(repo / F.GENERATION_FILE, F.json_bytes(gen))
    release["generation_manifest_sha256"] = F.digest((repo / F.GENERATION_FILE).read_bytes())
    csv_text = io.StringIO(newline=""); writer = csv.DictWriter(csv_text, fieldnames=list(frame[0])); writer.writeheader(); writer.writerows(frame)
    adapter = data / F.ADAPTER_RELATIVE; F.put_once(adapter, csv_text.getvalue().encode())
    release["adapter_sha256"] = F.digest(adapter.read_bytes())
    return repo, data, release


def test_generation_scope_checks_all_nine_completions_and_raw_prompt_joins(generation_fixture):
    repo, data, release = generation_fixture
    frame, responses = F.audit_generations(release, data, repo=repo)
    assert len(frame) == 2 and set(responses) == set(F.CONDITIONS)


def test_a_changed_completed_generation_is_refused(generation_fixture):
    repo, data, release = generation_fixture
    file = data / "final_evaluation_v5/sealed__baseline/records.jsonl"
    file.write_bytes(file.read_bytes() + b"\n")
    with pytest.raises(F.FinalJudgingError, match="digest changed"):
        F.audit_generations(release, data, repo=repo)


def test_missing_capability_completion_prevents_final_judging(generation_fixture):
    repo, data, release = generation_fixture
    (data / "final_evaluation_v5/tinymmlu__baseline/arm_complete.json").unlink()
    with pytest.raises(F.FinalJudgingError, match="artifact absent"):
        F.audit_generations(release, data, repo=repo)


def test_frozen_generation_source_change_is_refused(generation_fixture):
    repo, data, release = generation_fixture
    (repo / "synthetic_source.py").write_bytes(b"# Changed generation source.\n")
    with pytest.raises(F.FinalJudgingError, match="generation source changed"):
        F.audit_generations(release, data, repo=repo)
