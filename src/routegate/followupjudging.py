"""Prospective E/B/H assessment through the project's existing judging engine.

The inventory is development data. This module neither chooses an intervention
nor generates responses. It binds an explicit future plan, validates fresh
material, and keeps judgement evidence separate from scientific interpretation.
"""
from __future__ import annotations

import copy
import json
import re
from collections import Counter
from pathlib import Path

from . import cancellationjudging as C
from . import judgingprotocol as P
from . import judgingstages as S
from .validation import assert_not_sealed
from .judgingprotocol import assert_no_secrets

REPO = Path(__file__).resolve().parents[2]
OUTPUT_NAME = "assessment_v1"
PLAN_FILE = "assessment_plan.json"
STAGES = C.STAGES
require, digest, _json, _jsonl, _read, _rows, _put_once = C.require, C.digest, C._json, C._jsonl, C._read, C._rows, C._put_once
RUBRICS = ("axis2", "pairwise_quality", "safety")


def _inventory(root, repo_root=None):
    from . import followupinputs
    return followupinputs.verify(root)


def frozen_settings(repo_root=None):
    path = Path(repo_root or REPO) / "manifests/dynamic_arms_v1_judging_v1.json"
    raw = _read(path)
    P.load_manifest(path)
    return {"rubrics": {r: copy.deepcopy(raw["rubrics"][r]) for r in RUBRICS},
            "judges": copy.deepcopy(raw["judges"]), "sampling": copy.deepcopy(raw["sampling"]),
            "adjudication": copy.deepcopy(raw["adjudication"]),
            "source_manifest_sha256": digest(path.read_bytes())}


def _rubric_files(data_root, settings):
    roots = {"axis2": ("triplet_validation/frozen/rq1_judge_protocol", "axis2"),
             "safety": ("triplet_validation/frozen/rq1_judge_protocol", "safety"),
             "pairwise_quality": ("two_site_judging_v1/frozen/b_quality_rubric", "quality")}
    result = {}
    for rubric, (folder, stem) in roots.items():
        for part, ext in (("instructions", "txt"), ("user_template", "txt"), ("schema", "json")):
            data = (Path(data_root) / folder / (stem + "_" + part + "." + ext)).read_bytes()
            require(digest(data) == settings["rubrics"][rubric][part + "_sha256"], "Frozen rubric differs: " + rubric + " " + part)
            result[rubric + "_" + part + "." + ext] = data
    return result


def _safe_identifier(value):
    return isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value) is not None


def _source_hashes(repo):
    names = ("src/routegate/followupinputs.py", "src/routegate/followupjudging.py", "src/routegate/followupscoring.py",
             "src/routegate/judgingprotocol.py", "src/routegate/judgingstages.py", "src/routegate/triplets.py",
             "src/routegate/focusedjudging.py", "src/routegate/cancellationjudging.py")
    return {name: digest((Path(repo) / name).read_bytes()) for name in names}


def bind_plan(study_root, plan_file, *, repo_root=None, data_root=None, dry_run=False):
    """Freeze the user's specified comparison after the diagnostic is assessed.

    No observed score selects, removes or changes an arm. A complete diagnostic
    can justify proceeding or a different question; a favourable result is not
    a condition imposed by this function.
    """
    root, repo = Path(study_root).resolve(), Path(repo_root or REPO).resolve()
    inventory, rows = _inventory(root, repo)
    request = _read(plan_file)
    assert_no_secrets(request, "follow-up comparison plan")
    require(not (root / "generations/records.jsonl").exists() and not (root / OUTPUT_NAME).exists(), "Bind the plan before any generation or assessment exists.")
    candidates = request.get("candidate_arms")
    require(isinstance(candidates, list) and bool(candidates), "Choose at least one candidate explicitly after reviewing the H diagnostic.")
    arms = ["baseline"]
    for arm in candidates:
        require(isinstance(arm, dict) and _safe_identifier(arm.get("arm_id")) and arm["arm_id"] not in arms, "Candidate arm IDs must be unique lower-case identifiers, excluding baseline.")
        require(isinstance(arm.get("description"), str) and bool(arm["description"].strip()) and isinstance(arm.get("settings"), dict) and bool(arm["settings"]), "Each candidate needs its description and complete intervention settings.")
        arms.append(arm["arm_id"])
    comparisons = request.get("primary_comparisons")
    require(isinstance(comparisons, list) and bool(comparisons), "Predeclare at least one comparison.")
    names = set()
    for item in comparisons:
        require(_safe_identifier(item.get("name")) and item["name"] not in names and item.get("reference_arm") in arms and item.get("candidate_arm") in arms and item["reference_arm"] != item["candidate_arm"], "Invalid or repeated primary comparison.")
        names.add(item["name"])
    generation = request.get("generation_settings")
    require(isinstance(generation, dict) and all(k in generation for k in ("model", "revision", "dtype", "max_new_tokens", "do_sample", "enable_thinking")), "Generation settings need model, revision, dtype, max_new_tokens, do_sample and enable_thinking, plus the future runner's other settings.")
    require(type(generation["max_new_tokens"]) is int and generation["max_new_tokens"] > 0 and type(generation["do_sample"]) is bool and type(generation["enable_thinking"]) is bool, "Invalid generation settings types.")
    code = request.get("generation_code_hashes")
    require(isinstance(code, dict) and bool(code), "Pin the actual future generation runner and intervention implementation in generation_code_hashes before binding.")
    for name, sha in code.items():
        path = (repo / name).resolve()
        require(path.is_relative_to(repo) and path.is_file() and digest(path.read_bytes()) == sha, "Generation code hash differs or path is outside repo: " + name)
    report_path = Path(request.get("diagnostic_report", "")).expanduser().resolve()
    require(report_path.name == "scientific_report.json" and report_path.parent.name == "assessment_v1", "Supply the completed cancellation diagnostic scientific_report.json.")
    diagnostic = C.run_stage(report_path.parent.parent, "report", dry_run=True, repo_root=repo)
    require(_read(report_path) == diagnostic, "Diagnostic report differs from verified evidence.")
    require(all(v["unresolved_review"] == 0 for v in diagnostic["by_arm"].values()), "Finish exported disagreements before binding; explicitly carried technical missingness and UNCLEAR remain recorded.")
    diagnostic_missing = {a: v["technical_missing"] for a, v in diagnostic["by_arm"].items()}
    if any(diagnostic_missing.values()):
        require(isinstance(request.get("diagnostic_interpretation"), str) and bool(request["diagnostic_interpretation"].strip()), "Explain the diagnostic uncertainty in diagnostic_interpretation before binding; no extra retry is required.")
    settings = frozen_settings(repo)
    data = Path(data_root or inventory.get("data_root_at_init", "/home/ucabh21/routegate-data"))
    files = _rubric_files(data, settings)
    plan = {"schema": "routegate/development_followup_plan/1", "inventory_sha256": inventory["inventory_sha256"],
            "study_sha256": inventory["inventory_sha256"], "baseline_arm": "baseline", "arm_ids": arms,
            "candidate_arms": arms[1:], "candidate_definitions": candidates,
            "arm_settings_sha256": {a["arm_id"]: digest(_json(a["settings"])) for a in candidates},
            "baseline_settings": {"intervention": "none"}, "generation_settings": generation,
            "generation_settings_sha256": digest(_json(generation)), "generation_code_hashes": code,
            "primary_comparisons": comparisons, "judging": settings, "source_hashes": _source_hashes(repo),
            "diagnostic_report": {"path_at_binding": str(report_path), "sha256": digest(report_path.read_bytes()), "study_sha256": diagnostic["study_sha256"], "technical_missing_by_arm": diagnostic_missing, "unclear_by_arm": {a: v["UNCLEAR"] for a, v in diagnostic["by_arm"].items()}, "interpretation": request.get("diagnostic_interpretation", "No unresolved technical assessments.")},
            "selection_policy": "explicit researcher choice using development diagnostic; no automatic selection; all 480 original E/B/H rows retained",
            "deduplication": "exact prompt, response and rubric within this new assessment; pairwise reference and candidate roles retained",
            "historical_verdicts_imported": False, "expected_generations": 480 * len(arms), "external_380_used": False,
            "rubric_files": {name: digest(content) for name, content in files.items()}}
    plan["arm_settings_sha256"]["baseline"] = digest(_json(plan["baseline_settings"]))
    plan["plan_sha256"] = digest(_json(plan))
    for name, content in files.items():
        if (root / "judge_rubrics" / name).exists():
            require((root / "judge_rubrics" / name).read_bytes() == content, "Existing frozen rubric differs.")
    if not dry_run:
        for name, content in files.items():
            _put_once(root / "judge_rubrics" / name, content)
        _put_once(root / PLAN_FILE, _json(plan))
    return plan


def verify_plan(study_root, repo_root=None):
    root, repo = Path(study_root), Path(repo_root or REPO)
    inventory, rows = _inventory(root, repo)
    plan = _read(root / PLAN_FILE)
    require(plan.get("plan_sha256") == digest(_json({k: v for k, v in plan.items() if k != "plan_sha256"})), "Comparison plan checksum differs.")
    require(plan["inventory_sha256"] == inventory["inventory_sha256"], "Comparison plan belongs to another development inventory.")
    require(plan["source_hashes"] == _source_hashes(repo), "Follow-up or judging code changed after the comparison was bound.")
    for name, sha in plan["generation_code_hashes"].items():
        path = (repo / name).resolve()
        require(path.is_relative_to(repo.resolve()) and digest(path.read_bytes()) == sha, "Bound generation code changed: " + name)
    require(plan["judging"] == frozen_settings(repo), "Frozen judging settings changed.")
    for name, sha in plan["rubric_files"].items():
        require(digest((root / "judge_rubrics" / name).read_bytes()) == sha, "Frozen rubric changed: " + name)
    return plan, rows


def generation_record(plan, row, arm_id, response_text, completion_token_count,
                      eos_reached, hit_max_new_tokens, extra_metadata=None):
    """Record actual future-run outputs; this function generates no text.

    Counts and stop flags must come from the model runner. The record binds the
    fixed comparison and arm settings, and preserves optional run diagnostics.
    """
    require(arm_id in plan["arm_ids"], "Record arm is outside the bound plan.")
    require(isinstance(response_text, str), "The actual decoded response must be text.")
    require(type(completion_token_count) is int and 0 <= completion_token_count <= plan["generation_settings"]["max_new_tokens"], "Invalid observed completion-token count.")
    require(type(eos_reached) is bool and type(hit_max_new_tokens) is bool and hit_max_new_tokens == (completion_token_count == plan["generation_settings"]["max_new_tokens"]), "Observed cap/EOS flags differ from the frozen runtime convention.")
    require(row.get("row_class") in ("E", "B", "H") and digest(row["prompt_text"].encode()) == row["prompt_sha256"], "Invalid fixed development row.")
    record = {k: row[k] for k in ("row_id", "row_class", "prompt_text", "prompt_sha256")}
    record.update(arm_id=arm_id, status="ok", study_sha256=plan["study_sha256"], plan_sha256=plan["plan_sha256"],
                  generation_settings_sha256=plan["generation_settings_sha256"], arm_settings_sha256=plan["arm_settings_sha256"][arm_id],
                  response_text=response_text, response_sha256=digest(response_text.encode()),
                  completion_token_count=completion_token_count, eos_reached=eos_reached, hit_max_new_tokens=hit_max_new_tokens)
    if extra_metadata is not None:
        require(isinstance(extra_metadata, dict), "Extra generation metadata must be an object.")
        assert_no_secrets(extra_metadata, "generation diagnostics")
        json.dumps(extra_metadata, allow_nan=False)
        record["metadata"] = copy.deepcopy(extra_metadata)
    record["record_sha256"] = digest(_json(record))
    return record


def load_generation_records(study_root, plan, rows):
    path = Path(study_root) / "generations/records.jsonl"
    require(path.is_file(), "No fresh generation records yet. This package prepares evaluation; the future runner follows the H diagnostic.")
    records = _rows(path)
    by_id = {r["row_id"]: r for r in rows}
    require(len(by_id) == len(rows) == 480 and Counter(r["row_class"] for r in rows) == {"E": 89, "B": 178, "H": 213}, "Development frame differs from E89/B178/H213.")
    expected = {(r, a) for r in by_id for a in plan["arm_ids"]}
    require(len(records) == len(expected), "Prepare requires complete fresh coverage: {} generations; found {}.".format(len(expected), len(records)))
    seen = {}
    for record in records:
        key = (record.get("row_id"), record.get("arm_id"))
        require(key in expected and key not in seen, "Duplicate or foreign generation cell.")
        row = by_id[key[0]]
        require(record.get("status") == "ok" and record.get("study_sha256") == plan["study_sha256"] and record.get("plan_sha256") == plan["plan_sha256"], "Generation status or prospective plan binding differs.")
        require(record.get("generation_settings_sha256") == plan["generation_settings_sha256"] and record.get("arm_settings_sha256") == plan["arm_settings_sha256"][key[1]], "Generation settings binding differs.")
        require(record.get("row_class") == row["row_class"] and record.get("prompt_text") == row["prompt_text"] and record.get("prompt_sha256") == row["prompt_sha256"], "Generation prompt differs from the fixed inventory.")
        require(isinstance(record.get("response_text"), str) and digest(record["response_text"].encode()) == record.get("response_sha256"), "Generation response checksum differs.")
        require(type(record.get("hit_max_new_tokens")) is bool and type(record.get("eos_reached")) is bool and type(record.get("completion_token_count")) is int and 0 <= record["completion_token_count"] <= plan["generation_settings"]["max_new_tokens"], "Completion length and cap/EOS fields are required.")
        require(record["hit_max_new_tokens"] == (record["completion_token_count"] == plan["generation_settings"]["max_new_tokens"]), "Cap flag conflicts with token count/EOS under the frozen runtime convention.")
        payload = {k: v for k, v in record.items() if k != "record_sha256"}
        require(record.get("record_sha256") == digest(_json(payload)), "Generation record checksum differs.")
        seen[key] = record
    require(set(seen) == expected, "Generation coverage differs.")
    return seen


def context(study_root, repo_root=None):
    root = Path(study_root).resolve()
    assert_not_sealed(root)
    plan, rows = verify_plan(root, repo_root)
    records = load_generation_records(root, plan, rows)
    settings = plan["judging"]
    unique, references, mapping = {}, {}, []
    for row in rows:
        reference = records[(row["row_id"], "baseline")]
        for arm in plan["arm_ids"]:
            record = records[(row["row_id"], arm)]
            rubric = "safety" if row["row_class"] == "H" else ("pairwise_quality" if row["row_class"] == "B" and arm != "baseline" else "axis2")
            fingerprint = {p: settings["rubrics"][rubric][p + "_sha256"] for p in ("instructions", "user_template", "schema")}
            material = {"prompt_text": record["prompt_text"], "response_text": record["response_text"], "rubric": rubric, "rubric_fingerprint": fingerprint}
            if rubric == "pairwise_quality":
                material["reference_response_text"] = reference["response_text"]
            content_id = digest(_json(material))
            candidate = {"screen_row_id": content_id, "arm_id": "unique_material", "row_class": rubric, "prompt_text": record["prompt_text"], "response_text": record["response_text"]}
            require(content_id not in unique or unique[content_id] == candidate, "Material digest collision.")
            unique[content_id] = candidate
            cell = {"row_id": row["row_id"], "row_class": row["row_class"], "arm_id": arm, "rubric": rubric, "content_id": content_id,
                    "prompt_sha256": row["prompt_sha256"], "response_sha256": record["response_sha256"],
                    "hit_max_new_tokens": record["hit_max_new_tokens"], "eos_reached": record["eos_reached"], "completion_token_count": record["completion_token_count"]}
            if rubric == "pairwise_quality":
                references[content_id] = {**candidate, "arm_id": "reference_material", "response_text": reference["response_text"]}
                cell.update(reference_response_sha256=reference["response_sha256"], reference_hit_max_new_tokens=reference["hit_max_new_tokens"])
            mapping.append(cell)
    mapping.sort(key=lambda r: (r["row_id"], r["arm_id"]))
    counts = Counter(r["row_class"] for r in unique.values())
    files = {"inputs/candidate_records.jsonl": _jsonl([unique[k] for k in sorted(unique)]),
             "inputs/reference_records.jsonl": _jsonl([references[k] for k in sorted(references)]),
             "inputs/scientific_map.jsonl": _jsonl(mapping)}
    specs = copy.deepcopy(settings["rubrics"])
    for spec in specs.values():
        spec.update(protocol_dir_key="development_followup", subdir="frozen/judge_rubrics")
    protocol = {"schema": P.SCHEMA, "experiment_id": "development_followup_v1", "packet_id": "development_followup_content_v1", "data_scope": "fixed_development_480",
                "study_sha256": plan["study_sha256"], "plan_sha256": plan["plan_sha256"],
                "identity": {"row_prefixes": {r: "DFV1" + r.replace("_", "").upper() for r in RUBRICS}, "human_row_prefix": "DFV1REVIEW", "foreign_row_prefixes": []},
                "arms": {"candidate": ["unique_material"], "reference": ["reference_material"]},
                "classes": {r: {"rows": counts[r], "rubric": r} for r in RUBRICS}, "rubrics": specs,
                "inputs": {"paths_key": "development_followup", **{Path(k).stem: {"path": k, "sha256": digest(v)} for k, v in files.items()}},
                "judges": settings["judges"], "sampling": settings["sampling"], "adjudication": settings["adjudication"],
                "coverage": {"cells": len(unique), "records_per_judge": len(unique), "planned_calls": 2 * len(unique)},
                "outputs": {"root_key": "development_followup", **{k: "judging/" + k for k in ("packet", "runs", "joined", "review", "adjudicated", "report")}}}
    assessment = root / OUTPUT_NAME
    for name in plan["rubric_files"]:
        files["frozen/judge_rubrics/" + name] = (root / "judge_rubrics" / name).read_bytes()
    files["protocol.json"] = _json(protocol)
    metadata = {"schema": "routegate/development_followup_assessment/1", "inventory_root": str(root), "inventory_sha256": plan["inventory_sha256"], "study_sha256": plan["study_sha256"],
                "plan_sha256": plan["plan_sha256"], "evaluation_plan_sha256": plan["plan_sha256"], "baseline_arm": "baseline", "candidate_arms": plan["candidate_arms"], "arm_ids": plan["arm_ids"], "primary_comparisons": plan["primary_comparisons"],
                "generation_records_sha256": digest((root / "generations/records.jsonl").read_bytes()), "scientific_cells": len(mapping), "unique_materials": len(unique),
                "inputs_sha256": {k: digest(v) for k, v in files.items()}}
    files["assessment_manifest.json"] = _json(metadata)
    return plan, assessment, P.parse_protocol_manifest(protocol, assessment / "protocol.json"), files, mapping


def _verify_packet(root, manifest, create=False):
    packet = root / "judging/packet"
    private = S.load_row_map(packet)
    sources = {r["screen_row_id"]: r for r in _rows(root / "inputs/candidate_records.jsonl")}
    references = {r["screen_row_id"]: r for r in _rows(root / "inputs/reference_records.jsonl")}
    public = {r["row_id"]: r for rubric in RUBRICS for r in S.load_judge_rows(packet, rubric)}
    require(len(private) == len(public) == len(sources) and set(private) == set(public), "Packet coverage differs.")
    seen = set()
    for opaque, entry in private.items():
        content_id, rubric = entry["screen_row_id"], entry["rubric"]
        require(content_id in sources and content_id not in seen and rubric in RUBRICS, "Packet content identity differs.")
        seen.add(content_id)
        row, source = public[opaque], sources[content_id]
        P.assert_row_is_blind(row, manifest.rubrics[rubric])
        require(entry["arm_id"] == "unique_material" and entry["row_class"] == rubric == source["row_class"] and row["prompt_text"] == source["prompt_text"], "Packet metadata or prompt differs.")
        if rubric == "pairwise_quality":
            texts = {P.ORIGIN_CANDIDATE: source["response_text"], P.ORIGIN_REFERENCE: references[content_id]["response_text"]}
            require(set((entry["response_a_origin"], entry["response_b_origin"])) == set(texts) and row["response_a"] == texts[entry["response_a_origin"]] and row["response_b"] == texts[entry["response_b_origin"]], "Pairwise orientation or material differs.")
        else:
            require(row["response_text"] == source["response_text"], "Packet response differs.")
    require(_read(packet / P.MANIFEST_FILENAME)["manifest_sha256"] == manifest.digest(), "Packet manifest differs.")
    C._binding(root, "packet", [root / "protocol.json", packet / P.MANIFEST_FILENAME, packet / "private" / P.ROW_MAP_FILE] + [packet / (r + "_judge_input.jsonl") for r in RUBRICS], create)


def progress(root, manifest):
    summary = {"planned_per_judge": manifest.coverage["cells"], "judges": {}}
    for judge in manifest.judge_order:
        done, failed = 0, 0
        for rubric in RUBRICS:
            directory = root / "judging/runs" / ("packet__" + rubric + "__" + judge)
            success = P.already_done(directory / P.RESULTS_FILE)
            failures = _rows(directory / P.FAILURES_FILE) if (directory / P.FAILURES_FILE).exists() else []
            done += len(success)
            failed += len({r["row_id"] for r in failures} - set(success))
        summary["judges"][judge] = {"successful": done, "remaining": manifest.coverage["cells"] - done, "failed_without_success": failed}
    return summary


def _expanded(root, manifest, mapping):
    C._binding(root, "adjudication", C._adjudication_paths(root))
    raw = _rows(root / "judging/adjudicated" / P.ADJUDICATED_CELLS_FILE)
    unique = {r["screen_row_id"]: r for r in raw}
    require(len(unique) == len(raw) == manifest.coverage["cells"] and set(unique) == {r["content_id"] for r in mapping}, "Adjudicated material coverage differs.")
    expanded = []
    for row in mapping:
        cell = unique[row["content_id"]]
        require(cell["rubric"] == row["rubric"] and cell.get("verdict") in (*manifest.rubrics[row["rubric"]].values, None), "Foreign rubric or verdict.")
        require(not cell.get("technically_unresolved") or cell.get("verdict") is None, "Technical missingness cannot carry a verdict.")
        expanded.append({**row, **{k: cell.get(k) for k in ("verdict", "label_source", "judged_row_id", "sonnet_verdict", "opus_verdict")}, "technically_unresolved": bool(cell.get("technically_unresolved"))})
    return expanded


def verify_scoring_inputs(assessment_root):
    root = Path(assessment_root).resolve()
    metadata = _read(root / "assessment_manifest.json")
    _, expected_root, manifest, files, mapping = context(metadata["inventory_root"])
    require(root == expected_root, "Assessment location differs from its declared inventory.")
    C._verify_prepared(root, files)
    _verify_packet(root, manifest)
    expanded = _expanded(root, manifest, mapping)
    require((root / "scientific_adjudicated_cells.jsonl").read_bytes() == _jsonl(expanded), "Expanded verdict evidence differs.")
    return metadata, expanded


def run_stage(study_root, stage, *, judge=None, completed=None, reviewer_provenance=None, carried=None, dry_run=False, retry_failures=False, repo_root=None, transport=None):
    require(stage in STAGES, "Unknown assessment stage.")
    plan, root, manifest, files, mapping = context(study_root, repo_root)
    if stage == "prepare":
        for name, data in files.items():
            if (root / name).exists():
                require((root / name).read_bytes() == data, "Preparation would alter existing evidence.")
        if not dry_run:
            for name, data in files.items():
                _put_once(root / name, data)
        return {"stage": stage, "scientific_cells": len(mapping), "unique_materials": manifest.coverage["cells"], "planned_calls": manifest.coverage["planned_calls"], "api_calls": 0, "wrote_files": not dry_run}
    C._verify_prepared(root, files)
    dirs = S.resolved_dirs(manifest, root)
    check = S.preflight(manifest, dirs, root, root)
    if stage == "build":
        if (root / "packet_binding.json").exists():
            _verify_packet(root, manifest)
            return {"stage": stage, "existing_packet_preserved": True}
        result = S.build_packet(manifest, dirs, root, write=not dry_run)
        if not dry_run:
            _verify_packet(root, manifest, create=True)
        return result
    if stage == "preflight":
        if (root / "packet_binding.json").exists():
            _verify_packet(root, manifest)
        return check
    _verify_packet(root, manifest)
    if stage == "progress":
        return progress(root, manifest)
    if stage == "dryrun":
        return S.dry_run(manifest, dirs, root, judge)
    if stage == "judge":
        require(judge in manifest.judge_order, "Select sonnet or opus.")
        state = progress(root, manifest)["judges"][judge]
        require(not state["failed_without_success"] or retry_failures or dry_run, "Inspect saved failures before --retry-failures.")
        S.dry_run(manifest, dirs, root, judge)
        return S.run_judge(manifest, dirs, root, judge, write=not dry_run, transport=transport or C.ProgressTransport(judge))
    if stage == "join":
        technical = _read(carried).get("unresolved", []) if carried else []
        known = set(S.load_row_map(dirs["packet"]))
        require(all(c.get("row_id") in known and c.get("failed_judge") in manifest.judge_order for c in technical), "Foreign technical failure.")
        return S.join_evidence(manifest, dirs, write=not dry_run, carried=technical)
    if stage == "export":
        return S.export_review(manifest, dirs, write=not dry_run)
    if stage == "adjudicate":
        review_map = dirs["review"] / "private" / P.REVIEW_MAP_FILE
        require(review_map.is_file(), "Export the review packet first.")
        if _rows(review_map):
            require(completed is not None and reviewer_provenance is not None, "Supply completed disagreements and truthful reviewer provenance.")
            provenance = _read(reviewer_provenance)
            require(provenance.get("schema") == "routegate/development_followup_review_provenance/1" and type(provenance.get("ai_assisted")) is bool and type(provenance.get("blinded")) is bool and isinstance(provenance.get("reviewer"), str) and bool(provenance["reviewer"].strip()), "Provenance needs schema, reviewer, ai_assisted and blinded.")
            completed_bytes = Path(completed).read_bytes()
            require(provenance.get("completed_sha256") == digest(completed_bytes), "Review provenance does not bind completed labels.")
        else:
            require(completed is None, "No disagreements exist; do not import unrelated labels.")
            provenance = {"procedure": "dual_judge_agreement_or_technical_unresolved", "manual_adjudication_performed": False}
        # Existing adjudication is immutable. Repeating this stage with the
        # same review verifies and returns it; changed review material never
        # overwrites a previously accepted decision.
        if (root / "adjudication_binding.json").exists():
            require((root / "review_provenance.json").read_bytes() == _json(provenance), "Existing reviewer provenance differs; use a separately versioned assessment.")
            if completed is not None:
                require((dirs["review"] / P.REVIEW_COMPLETED_FILE).read_bytes() == completed_bytes, "Existing completed review differs; original decisions are preserved.")
            C._binding(root, "adjudication", C._adjudication_paths(root))
            return {**_read(dirs["adjudicated"] / "adjudication_summary.json"), "existing_adjudication_preserved": True}
        result = S.import_review(manifest, dirs, Path(completed) if completed else None, write=not dry_run)
        if not dry_run:
            if completed is not None:
                require(Path(completed).read_bytes() == completed_bytes, "Review changed during import.")
                _put_once(dirs["review"] / P.REVIEW_COMPLETED_FILE, completed_bytes)
            _put_once(root / "review_provenance.json", _json(provenance))
            C._binding(root, "adjudication", C._adjudication_paths(root), create=True)
        return result
    if stage == "report":
        expanded = _expanded(root, manifest, mapping)
        result = {"stage": "report", "scientific_cells": len(expanded), "unique_materials": manifest.coverage["cells"], "unresolved_cells": sum(r["verdict"] is None for r in expanded), "technically_unresolved_cells": sum(r["technically_unresolved"] for r in expanded), "scientific_scoring": "routegate.followupscoring"}
        S.report(manifest, dirs, write=not dry_run)
        if not dry_run:
            _put_once(root / "scientific_adjudicated_cells.jsonl", _jsonl(expanded))
        return result
    raise ValueError("Unreachable assessment stage")
