"""Final-evaluation input adapter for the existing common judging stages.

The development entry point keeps its sealed-data exclusion. This entry point
accepts only the released v5 files, whose digests are pinned after generation
and before judging. No direction, checkpoint, generation or rubric is changed.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path

from . import judgingprotocol as P
from . import judgingstages as S
from .finalrunner import assert_record_is_complete
from .triplets import assert_no_text_fields, read_jsonl, sha256_text
from .validation import ValidationError


REPO = Path(__file__).resolve().parents[2]
RELEASE_FILE = "manifests/routegate_sealed_judging_v1_release.json"
GENERATION_FILE = "manifests/routegate_final_evaluation_v5.json"
OUTPUT_NAME = "final_evaluation_v5_judging_v1"
ADAPTER_RELATIVE = "final_evaluation_v5/sealed_input_release/rq3_sealed_final_manifest_prompt_text.csv"
PHASES = ("baseline", "responses")
CONDITIONS = ("baseline", "dynamic_raw", "safety_first_routegate")
PREFIXES = {"baseline": {"axis1": "FJ1A", "axis2": "FJ1B"},
            "responses": {"axis2": "FJ2A", "pairwise_quality": "FJ2B", "safety": "FJ2C"}}


class FinalJudgingError(ValidationError):
    pass


def require(condition, message):
    if not condition:
        raise FinalJudgingError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def jsonl_bytes(rows):
    return "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")


def rows_bytes(data):
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def read_json(path):
    require(Path(path).is_file(), "Required file is absent: " + str(path))
    return json.loads(Path(path).read_text(encoding="utf-8"))


def put_once(path, data):
    """Preserve an existing identical file; never replace different bytes."""
    path = Path(path)
    if path.exists():
        require(path.read_bytes() == data, "Existing file differs; refusing to overwrite: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    require(not temporary.exists(), "Incomplete earlier write: " + str(temporary))
    with temporary.open("xb") as handle:
        handle.write(data)
    temporary.replace(path)


def load_release(repo=REPO):
    release = read_json(repo / RELEASE_FILE)
    require(release.get("schema") == "routegate/final_judging_release/1"
            and release.get("data_scope") == "released_final_evaluation_v5", "Wrong final judging release.")
    expected = digest(json.dumps({k: v for k, v in release.items() if k != "record_sha256"}, sort_keys=True).encode())
    require(release.get("record_sha256") == expected, "Judging release digest differs.")
    P.assert_no_secrets(release, "final judging release")
    assert_no_text_fields(release, "final judging release")
    for name, expected in release["integration_sources"].items():
        require((repo / name).is_file() and digest((repo / name).read_bytes()) == expected,
                "Installed judging source differs from this release: " + name)
    return release


def audit_generations(release, data_root, adapter=None, repo=REPO):
    """Read-only checks; do not print or interpret any generated answer."""
    gen_path = repo / GENERATION_FILE
    require(digest(gen_path.read_bytes()) == release["generation_manifest_sha256"], "Generation manifest changed.")
    gen = read_json(gen_path)
    for relative, expected in gen["source_sha256"].items():
        path = repo / relative
        require(path.is_file() and digest(path.read_bytes()) == expected,
                "Frozen generation source changed: " + relative)
    records = {}
    for relative, expected in release["generation_files"].items():
        path = data_root / relative
        require(path.is_file(), "Required generation artifact absent: " + str(path))
        require(digest(path.read_bytes()) == expected, "Generation artifact digest changed: " + relative)
    require(len(gen["arms"]) == 9, "Expected all nine completed arms.")
    for arm in gen["arms"]:
        name, count = arm["arm_id"], arm["rows"]
        directory = data_root / arm["output_dir"]
        raw = (directory / "records.jsonl").read_bytes()
        rows = rows_bytes(raw)
        complete = read_json(directory / "arm_complete.json")
        require(len(rows) == count == len({r["item_id"] for r in rows}), "Missing or duplicated records in " + name)
        require(sorted(r["item_index"] for r in rows) == list(range(count)), "Wrong item-index set in " + name)
        require(complete.get("complete") is True and complete.get("rows") == count
                and complete.get("records_sha256") == digest(raw), "Completion artifact mismatch in " + name)
        for row in rows:
            require(row["protocol"] == "routegate_final_evaluation_v5" and row["schema"] == gen["record_schema"]
                    and row["arm_id"] == name and row["dataset"] == arm["dataset"]
                    and row["condition"] == arm["condition"] and row["status"] == "ok"
                    and isinstance(row["response_text"], str), "Record identity/schema/status mismatch in " + name)
            for key in ("model", "revision", "chat_template_sha256", "tokenizer_config_sha256", "do_sample", "sampling_seed"):
                require(row[key] == gen["runtime"][key], "Frozen runtime mismatch in " + name)
            assert_record_is_complete(row)
        records[name] = {row["item_id"]: row for row in rows}
    adapter = adapter or data_root / ADAPTER_RELATIVE
    raw = Path(adapter).read_bytes()
    require(digest(raw) == release["adapter_sha256"], "The prompt CSV is not the released adapter.")
    frame = list(csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")))
    require(len(frame) == release["expected_prompts"] == len({r["row_id"] for r in frame}), "Prompt count or uniqueness mismatch.")
    require(dict(Counter(r["source_side"] for r in frame)) == release["source_composition"], "Source composition changed.")
    for index, row in enumerate(frame):
        prompt_hash = sha256_text(row["prompt_text"])
        # The archived canonical hash uses a different historical convention.
        # The whole CSV is pinned. Join its exact text to the raw-text digest
        # actually written by the current generation runner.
        for condition in CONDITIONS:
            record = records["sealed__" + condition][row["row_id"]]
            require(record["item_index"] == index and record["prompt_sha256"] == prompt_hash,
                    "A sealed record does not match its released prompt.")
    return frame, {c: records["sealed__" + c] for c in CONDITIONS}


def rubric_files(release, repo=REPO, override=None):
    paths = read_json(repo / "configs/paths.local.json") if override is None else {}
    files = {}
    for name, spec in release["rubrics"].items():
        if override is not None:
            directory = Path(override) / spec["source_key"]
        else:
            require(spec["source_key"] in paths, "Missing rubric path key: " + spec["source_key"])
            directory = Path(paths[spec["source_key"]]).expanduser() / spec["source_subdir"]
        for part, extension in (("instructions", ".txt"), ("user_template", ".txt"), ("schema", ".json")):
            filename = name + "_" + part + extension
            path = directory / filename
            require(path.is_file(), "Frozen rubric file absent: " + str(path))
            data = path.read_bytes()
            require(digest(data) == spec[part + "_sha256"], "Frozen rubric changed: " + filename)
            files["frozen/judge_rubrics/" + filename] = data
    return files


def phase_root(data_root, phase):
    require(phase in PHASES, "Unknown judging phase.")
    return Path(data_root) / OUTPUT_NAME / phase


def binding(root, kind, paths, write=False):
    actual = {str(p.relative_to(root)): digest(p.read_bytes()) for p in paths}
    name = root / (kind + "_binding.json")
    if write:
        put_once(name, json_bytes(actual))
    else:
        require(read_json(name) == actual, kind + " files changed after being bound.")
    return actual


def read_adjudicated(data_root, phase):
    root = phase_root(data_root, phase)
    file = root / "judging/adjudicated/adjudicated_cells.jsonl"
    summary = root / "judging/adjudicated/adjudication_summary.json"
    binding(root, "adjudication", [file, summary])
    rows = rows_bytes(file.read_bytes())
    P.index_cells(rows, "adjudicated cells")
    return rows, file.read_bytes()


def derive_groups(frame, baseline):
    """Use adjudicated permission and actual baseline behaviour, never source."""
    index = P.index_cells(baseline, "baseline adjudication")
    require(len(index) == 2 * len(frame), "Baseline adjudication must retain both cells for every prompt.")
    groups = []
    for prompt in frame:
        rid = prompt["row_id"]
        require((rid, "prompt", "axis1") in index and (rid, "baseline", "axis2") in index,
                "A baseline assessment cell is missing.")
        permission = index[(rid, "prompt", "axis1")].get("verdict")
        behaviour = index[(rid, "baseline", "axis2")].get("verdict")
        require(permission in (None, "PERMITTED", "PROHIBITED", "AMBIGUOUS"), "Invalid permission label.")
        require(behaviour in (None, "SA", "PA-S", "PA-N", "DR", "IR", "NR", "UM"), "Invalid baseline behaviour label.")
        if permission is None:
            group = "permission_unresolved"
        elif permission == "AMBIGUOUS":
            group = "permission_ambiguous"
        elif permission == "PROHIBITED":
            group = "H"
        elif behaviour in ("DR", "IR"):
            group = "E"
        elif behaviour == "SA":
            group = "B"
        elif behaviour is None:
            group = "permitted_baseline_unresolved"
        else:
            group = "permitted_other"
        groups.append({"row_id": rid, "group": group, "permission": permission,
                       "baseline_behaviour": behaviour, "source_side": prompt["source_side"]})
    return groups


def input_plan(frame, responses, phase, baseline=None, baseline_bytes=None):
    candidates, references = [], []
    def source(prompt, arm, group, response=True):
        row = {"screen_row_id": prompt["row_id"], "arm_id": arm,
               "row_class": group, "prompt_text": prompt["prompt_text"]}
        if response:
            row["response_text"] = responses[arm][prompt["row_id"]]["response_text"]
        return row
    if phase == "baseline":
        for prompt in frame:
            candidates.append(source(prompt, "prompt", "permission", response=False))
            candidates.append(source(prompt, "baseline", "baseline_behaviour"))
        groups = None
        group_rubrics = {"permission": "axis1", "baseline_behaviour": "axis2"}
    else:
        require(baseline is not None, "Finish baseline adjudication before preparing response assessment.")
        groups = derive_groups(frame, baseline)
        by_id = {p["row_id"]: p for p in frame}
        group_rubrics = {"E": "axis2", "B": "pairwise_quality", "H": "safety",
                         "permitted_other": "axis2", "permitted_baseline_unresolved": "axis2"}
        for item in groups:
            group = item["group"]
            if group not in group_rubrics:
                continue
            prompt = by_id[item["row_id"]]
            if group == "B":
                references.append(source(prompt, "baseline_reference", group, response=False))
                references[-1]["response_text"] = responses["baseline"][item["row_id"]]["response_text"]
            arms = CONDITIONS if group == "H" else CONDITIONS[1:]
            for arm in arms:
                candidates.append(source(prompt, arm, group))
    files = {"inputs/candidate_records.jsonl": jsonl_bytes(candidates)}
    if references:
        files["inputs/reference_records.jsonl"] = jsonl_bytes(references)
    if groups is not None:
        files["inputs/eligibility.json"] = json_bytes(groups)
        files["inputs/baseline_adjudicated.jsonl"] = baseline_bytes
    counts = Counter(r["row_class"] for r in candidates)
    classes = {g: {"rows": n, "rubric": group_rubrics[g]} for g, n in sorted(counts.items())}
    return files, classes, sorted({r["arm_id"] for r in candidates}), bool(references)


def protocol_document(release, phase, files, classes, arms, has_references):
    used = {s["rubric"] for s in classes.values()}
    rubrics = {}
    for name in sorted(used):
        spec = copy.deepcopy(release["rubrics"][name])
        spec.pop("source_key"); spec.pop("source_subdir")
        spec.update(protocol_dir_key="final_evaluation_judging", subdir="frozen/judge_rubrics")
        rubrics[name] = spec
    total = sum(s["rows"] for s in classes.values())
    require(total > 0, "No eligible response cells; retain the eligibility report and do not make judge calls.")
    inputs = {"paths_key": "final_evaluation_judging"}
    for filename, data in files.items():
        key = Path(filename).stem
        inputs[key] = {"path": filename, "sha256": digest(data)}
    return {"schema": P.SCHEMA, "experiment_id": "sealed_final_v1_" + phase,
            "packet_id": "sealed_final_v1_" + phase, "data_scope": "released_final_evaluation_v5",
            "release_sha256": release["record_sha256"],
            "identity": {"row_prefixes": {n: PREFIXES[phase][n] for n in used},
                         "human_row_prefix": "FJ1H" if phase == "baseline" else "FJ2H",
                         "foreign_row_prefixes": []},
            "arms": {"candidate": arms, "reference": ["baseline_reference"] if has_references else []},
            "classes": classes, "rubrics": rubrics, "inputs": inputs,
            "judges": release["judges"], "sampling": release["sampling"],
            "adjudication": release["adjudication"],
            "coverage": {"cells": total, "records_per_judge": total, "planned_calls": total * 2},
            "outputs": {"root_key": "final_evaluation_judging",
                        **{name: "judging/" + name for name in ("packet", "runs", "joined", "review", "adjudicated", "report")}}}


def context(data_root, phase, adapter=None, repo=REPO):
    release = load_release(repo)
    frame, responses = audit_generations(release, data_root, adapter, repo)
    baseline, raw = (read_adjudicated(data_root, "baseline") if phase == "responses" else (None, None))
    files, classes, arms, refs = input_plan(frame, responses, phase, baseline, raw)
    document = protocol_document(release, phase, files, classes, arms, refs)
    root = phase_root(data_root, phase)
    manifest = P.parse_protocol_manifest(document, root / "protocol.json")
    return release, root, manifest, files


def verify_prepared(root, manifest, files):
    require(read_json(root / "protocol.json") == manifest.raw, "Prepared protocol differs from the frozen input plan.")
    for name, data in files.items():
        require((root / name).is_file() and (root / name).read_bytes() == data,
                "Prepared input differs from the frozen source: " + name)


def prepare(data_root, phase, adapter=None, rubrics_root=None, write=True, repo=REPO):
    release, root, manifest, files = context(data_root, phase, adapter, repo)
    staged = rubric_files(release, repo, rubrics_root)
    payloads = {**files, **staged, "protocol.json": json_bytes(manifest.raw)}
    for name, data in payloads.items():
        if (root / name).exists():
            require((root / name).read_bytes() == data, "Preparation would change an existing file: " + name)
    if write:
        for name, data in payloads.items():
            put_once(root / name, data)
    return {"stage": "prepare", "phase": phase, **P.expected_coverage(manifest),
            "generation_records_verified": 1740, "wrote_files": write, "api_calls": 0}


def verify_packet(root, manifest, create_binding=False):
    """Check every displayed string against its source and bind opaque IDs."""
    directory = root / "judging/packet"
    row_map = S.load_row_map(directory)
    candidates = rows_bytes((root / "inputs/candidate_records.jsonl").read_bytes())
    source_index = {(r["screen_row_id"], r["arm_id"], r["row_class"]): r for r in candidates}
    require(len(source_index) == len(candidates), "Duplicate source cells.")
    references_path = root / "inputs/reference_records.jsonl"
    references = {(r["screen_row_id"], r["row_class"]): r for r in rows_bytes(references_path.read_bytes())} if references_path.exists() else {}
    public = {}
    packet_files = [directory / P.MANIFEST_FILENAME, directory / "private" / P.ROW_MAP_FILE]
    for name, rubric in manifest.rubrics.items():
        packet_files.append(directory / (name + "_judge_input.jsonl"))
        for row in S.load_judge_rows(directory, name):
            P.assert_row_is_blind(row, rubric)
            require(row["row_id"] not in public, "Duplicate judge row ID.")
            public[row["row_id"]] = row
    require(set(public) == set(row_map) and len(public) == len(candidates), "Packet coverage or private map mismatch.")
    seen = set()
    for row_id, entry in row_map.items():
        key = (entry["screen_row_id"], entry["arm_id"], entry["row_class"])
        require(key in source_index and key not in seen, "Packet cell does not map uniquely to a source.")
        seen.add(key)
        source, row = source_index[key], public[row_id]
        rubric = manifest.rubric_for_class(entry["row_class"])
        require(entry["rubric"] == rubric.name and row["prompt_text"] == source["prompt_text"], "Prompt or rubric mapping mismatch.")
        if rubric.prompt_only:
            require(set(row) == {"row_id", "prompt_text"}, "Permission judgement contains response information.")
        elif rubric.pairwise:
            ref = references[(source["screen_row_id"], source["row_class"])]
            origins = {"candidate": source["response_text"], "reference": ref["response_text"]}
            require({entry["response_a_origin"], entry["response_b_origin"]} == {"candidate", "reference"}, "Invalid pairwise orientation.")
            require(row["response_a"] == origins[entry["response_a_origin"]]
                    and row["response_b"] == origins[entry["response_b_origin"]], "Pairwise texts differ from their recorded origins.")
        else:
            require(row["response_text"] == source["response_text"], "Judge response differs from the generated text.")
    require(read_json(directory / P.MANIFEST_FILENAME)["manifest_sha256"] == manifest.digest(), "Packet belongs to another protocol.")
    binding(root, "packet", [root / "protocol.json"] + packet_files, write=create_binding)


def progress(root, manifest):
    dirs = S.resolved_dirs(manifest, root)
    result = {"phase": root.name, "planned_per_judge": manifest.coverage["records_per_judge"], "judges": {}}
    for judge in manifest.judge_order:
        done, open_failures = 0, 0
        for name in manifest.rubrics:
            directory = dirs["runs"] / ("packet__" + name + "__" + judge)
            successes = P.already_done(directory / P.RESULTS_FILE)
            failures = list(read_jsonl(directory / P.FAILURES_FILE)) if (directory / P.FAILURES_FILE).exists() else []
            done += len(successes)
            open_failures += len({r["row_id"] for r in failures if r["row_id"] not in successes})
        result["judges"][judge] = {"successful": done, "remaining": result["planned_per_judge"] - done,
                                    "failed_without_success": open_failures}
    return result


class SafeProgressTransport:
    """Use the existing transport; stop after recording an authentication error."""
    def __init__(self, judge):
        self.judge, self.calls, self.auth_failed = judge, 0, False

    def __call__(self, *args, **kwargs):
        require(not self.auth_failed, "Authentication failed. The failed attempt was recorded; fix the key before an explicit retry.")
        reply = S.call_judge(*args, **kwargs)
        self.calls += 1
        if self.calls == 1 or self.calls % 25 == 0:
            print("[judge] {}: {} calls attempted in this invocation".format(self.judge, self.calls), flush=True)
        self.auth_failed = reply.get("http_status") in (401, 403)
        return reply


def run_stage(data_root, phase, stage, judge=None, completed=None, carried=None,
              dry_run=False, retry_failures=False, adapter=None, rubrics_root=None, repo=REPO):
    if stage == "prepare":
        return prepare(data_root, phase, adapter, rubrics_root, not dry_run, repo)
    release, root, manifest, files = context(data_root, phase, adapter, repo)
    verify_prepared(root, manifest, files)
    dirs = S.resolved_dirs(manifest, root)
    S.preflight(manifest, dirs, root, root)
    if stage == "build":
        if (root / "packet_binding.json").exists():
            verify_packet(root, manifest)
            return {"stage": "build", "existing_packet_preserved": True, "cells": manifest.coverage["cells"]}
        result = S.build_packet(manifest, dirs, root, write=not dry_run)
        if not dry_run:
            verify_packet(root, manifest, create_binding=True)
        return result
    if stage in ("preflight", "progress"):
        if (root / "packet_binding.json").exists():
            verify_packet(root, manifest)
        return progress(root, manifest) if stage == "progress" else {"stage": stage, "phase": phase, "coverage": P.expected_coverage(manifest)}
    verify_packet(root, manifest)
    if stage == "dryrun":
        return S.dry_run(manifest, dirs, root)
    if stage == "judge":
        require(judge in manifest.judge_order, "Select --judge sonnet or --judge opus.")
        state = progress(root, manifest)["judges"][judge]
        require(not state["failed_without_success"] or retry_failures or dry_run,
                "Recorded failed rows remain. Inspect the failures, then use --retry-failures for a deliberate retry.")
        if not dry_run:
            import os
            key = os.environ.get("MODEL_API_KEY", "")
            require(len(key) == 51 and key.isascii() and not any(c.isspace() for c in key),
                    "The container needs the 51-character MODEL_API_KEY entered in the current terminal.")
            require(bool(os.environ.get("ROUTEGATE_JUDGE_ENDPOINT")), "Judge endpoint is missing from the container.")
        S.dry_run(manifest, dirs, root, judge)
        return S.run_judge(manifest, dirs, root, judge, write=not dry_run, transport=SafeProgressTransport(judge))
    if stage == "join":
        technical = read_json(carried).get("unresolved", []) if carried else []
        return S.join_evidence(manifest, dirs, write=not dry_run, carried=technical)
    if stage == "export":
        return S.export_review(manifest, dirs, write=not dry_run)
    if stage == "import":
        review_map = dirs["review"] / "private" / P.REVIEW_MAP_FILE
        require(review_map.is_file(), "Export the blinded human-review packet before importing it.")
        review_file = Path(completed) if completed else dirs["review"] / P.REVIEW_COMPLETED_FILE
        require(not rows_bytes(review_map.read_bytes()) or review_file.is_file(),
                "Human-review rows exist, but the completed review file is absent. Supply --completed with the reviewed file.")
        result = S.import_review(manifest, dirs, completed, write=not dry_run)
        if not dry_run:
            binding(root, "adjudication", [dirs["adjudicated"] / P.ADJUDICATED_CELLS_FILE,
                    dirs["adjudicated"] / "adjudication_summary.json"], write=True)
        return result
    if stage == "report":
        read_adjudicated(data_root, phase)
        return S.report(manifest, dirs, write=not dry_run)
    raise FinalJudgingError("Unknown stage: " + stage)
