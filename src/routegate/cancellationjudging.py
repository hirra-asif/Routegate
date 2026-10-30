"""New-study safety judging through the existing frozen judging engine.

Only the new 9 x 18 generations enter this adapter. Exact material is deduplicated
before opaque packet IDs are minted, then adjudications are expanded back to all
scientific cells. Historical judgements are never loaded or imported.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from . import judgingprotocol as P
from . import judgingstages as S
from .validation import ValidationError, assert_not_sealed

REPO = Path(__file__).resolve().parents[2]
STUDY_ID = "cancellation_controls_v1"
OUTPUT_NAME = "assessment_v1"
FROZEN_JUDGING_MANIFEST = "manifests/rotation_safety_direction_v2_judging_v1.json"
ARM_IDS = ("baseline", "repair_only", "original_combination", "attenuated_repair",
           "orthogonal_full_repair", "orthogonal_only", "random_orthogonal_1",
           "random_orthogonal_2", "random_orthogonal_3")
STAGES = ("prepare", "preflight", "build", "dryrun", "judge", "progress", "join",
          "export", "adjudicate", "report")


class CancellationJudgingError(ValidationError):
    pass


def require(condition, message):
    if not condition:
        raise CancellationJudgingError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _json(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()


def _jsonl(rows):
    return b"".join((json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n").encode() for r in rows)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def _put_once(path, data):
    path = Path(path)
    if path.exists():
        require(path.read_bytes() == data, "Existing immutable file differs: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def frozen_settings(repo_root=None):
    """Settings to embed in the study manifest before any new generations."""
    path = Path(repo_root or REPO) / FROZEN_JUDGING_MANIFEST
    raw = _read(path)
    P.load_manifest(path)  # Preserve the common development-scope guard.
    return {"rubrics": {"safety": copy.deepcopy(raw["rubrics"]["safety"])},
            "judges": copy.deepcopy(raw["judges"]),
            "sampling": copy.deepcopy(raw["sampling"]),
            "adjudication": copy.deepcopy(raw["adjudication"]),
            "source_manifest_sha256": digest(path.read_bytes())}


def _verify_study(study_root, repo_root):
    # Late import: this adapter does not load models, tensors or controls.
    from .cancellationexperiment import verify_study
    return verify_study(study_root, repo_root=repo_root)


def _load_records(study_root, study):
    # Use the generation runner's provenance gate, including capture/environment
    # links and record digests; the judging adapter cannot waive those checks.
    from .cancellationexperiment import load_records
    return list(load_records(study_root, study).values())


def context(study_root, repo_root=None, rubrics_root=None):
    """Validate complete fresh material and deterministically plan deduplication."""
    study_root = Path(study_root).expanduser().resolve()
    assert_not_sealed(study_root)
    study = _verify_study(study_root, repo_root)
    require(study.get("study_id") == STUDY_ID and bool(study.get("study_sha256")), "Wrong study identity.")
    arms = [a["arm_id"] for a in study["arms"]]
    require(len(arms) == 9 and set(arms) == set(ARM_IDS), "The nine predeclared arms are required.")
    selected = _rows(study_root / "selected_rows.jsonl")
    selected_by_id = {r["row_id"]: r for r in selected}
    require(len(selected) == len(selected_by_id) == 18, "Exactly 18 unique selected rows are required.")
    for row in selected:
        require(row.get("row_class") == "H", "Only the selected H frame is supported.")
        require(digest(row["prompt_text"].encode()) == row["prompt_sha256"], "Selected prompt checksum differs.")
    source = study_root / "generations/records.jsonl"
    records = _load_records(study_root, study)
    expected = {(r, a) for r in selected_by_id for a in arms}
    require(len(records) == 162, "Prepare requires all 162 successful fresh generations.")
    seen = set()
    settings = frozen_settings(repo_root)
    require(study.get("judging") == settings, "Study judging settings must equal the frozen settings before preparation.")
    rubric = settings["rubrics"]["safety"]
    fingerprint = digest(_json({k: rubric[k] for k in ("instructions_sha256", "user_template_sha256", "schema_sha256")}))
    unique, mapping = {}, []
    for record in records:
        key = (record["row_id"], record["arm_id"])
        require(key in expected and key not in seen, "Unexpected or duplicate scientific generation cell.")
        seen.add(key)
        require(record.get("status") == "ok", "A failed or incomplete generation is not judgeable material.")
        require(record.get("study_sha256") == study["study_sha256"], "Generation belongs to another study snapshot.")
        selected_row = selected_by_id[key[0]]
        require(record["prompt_text"] == selected_row["prompt_text"] and record["prompt_sha256"] == selected_row["prompt_sha256"], "Generation prompt differs from selected material.")
        require(isinstance(record["response_text"], str) and digest(record["response_text"].encode()) == record["response_sha256"], "Generation response checksum differs.")
        material = {"prompt_text": record["prompt_text"], "response_text": record["response_text"], "rubric_fingerprint": fingerprint}
        content_id = digest(_json(material))
        candidate = {"screen_row_id": content_id, "arm_id": "unique_material", "row_class": "H",
                     "prompt_text": record["prompt_text"], "response_text": record["response_text"]}
        require(content_id not in unique or unique[content_id] == candidate, "Material hash collision.")
        unique[content_id] = candidate
        mapping.append({"row_id": key[0], "arm_id": key[1], "content_id": content_id,
                        "prompt_sha256": record["prompt_sha256"], "response_sha256": record["response_sha256"]})
    require(seen == expected, "Generation coverage differs from 9 x 18.")
    for contrast in study["primary_comparisons"]:
        require(contrast["reference_arm"] in arms and contrast["candidate_arm"] in arms,
                "Comparison names an undeclared arm.")
    mapping.sort(key=lambda r: (r["row_id"], r["arm_id"]))
    candidates = [unique[k] for k in sorted(unique)]
    files = {"inputs/candidate_records.jsonl": _jsonl(candidates),
             "inputs/scientific_map.jsonl": _jsonl(mapping)}
    n = len(unique)
    spec = copy.deepcopy(rubric)
    spec.update(protocol_dir_key="cancellation_assessment", subdir="frozen/judge_rubrics")
    protocol = {"schema": P.SCHEMA, "experiment_id": STUDY_ID + "_judging_v1",
                "packet_id": STUDY_ID + "_content_packet_v1", "data_scope": "selected_development_H",
                "study_sha256": study["study_sha256"], "generation_file_sha256": digest(source.read_bytes()),
                "scientific_cells": 162, "deduplication": "exact prompt, response and frozen rubric content; new study only",
                "identity": {"row_prefixes": {"safety": "CCV1SAFETY"}, "human_row_prefix": "CCV1REVIEW", "foreign_row_prefixes": []},
                "arms": {"candidate": ["unique_material"], "reference": []},
                "classes": {"H": {"rows": n, "rubric": "safety"}}, "rubrics": {"safety": spec},
                "inputs": {"paths_key": "cancellation_assessment", **{Path(k).stem: {"path": k, "sha256": digest(v)} for k, v in files.items()}},
                "judges": settings["judges"], "sampling": settings["sampling"], "adjudication": settings["adjudication"],
                "coverage": {"cells": n, "records_per_judge": n, "planned_calls": 2 * n},
                "outputs": {"root_key": "cancellation_assessment", **{k: "judging/" + k for k in ("packet", "runs", "joined", "review", "adjudicated", "report")}}}
    root = study_root / OUTPUT_NAME
    manifest = P.parse_protocol_manifest(protocol, root / "protocol.json")
    rubric_dir = Path(rubrics_root) if rubrics_root else study_root / "judge_rubrics"
    for kind, ext in (("instructions", "txt"), ("user_template", "txt"), ("schema", "json")):
        name = "safety_" + kind + "." + ext
        data = (rubric_dir / name).read_bytes()
        require(digest(data) == rubric[kind + "_sha256"], "Frozen rubric differs: " + name)
        files["frozen/judge_rubrics/" + name] = data
    files["protocol.json"] = _json(protocol)
    return study, root, manifest, files, mapping


def _verify_prepared(root, files):
    for name, data in files.items():
        require((root / name).is_file() and (root / name).read_bytes() == data, "Prepared snapshot differs: " + name)


def _binding(root, name, paths, create=False):
    values = {str(p.relative_to(root)): digest(p.read_bytes()) for p in paths}
    path = root / (name + "_binding.json")
    if create:
        _put_once(path, _json(values))
    else:
        require(_read(path) == values, "Changed " + name + " evidence; start a separately versioned assessment.")


def _adjudication_paths(root):
    paths = [root / "judging/adjudicated" / P.ADJUDICATED_CELLS_FILE,
             root / "judging/adjudicated/adjudication_summary.json", root / "review_provenance.json"]
    completed = root / "judging/review" / P.REVIEW_COMPLETED_FILE
    if completed.exists():
        paths.append(completed)
    return paths


def _verify_packet(root, manifest, create=False):
    packet = root / "judging/packet"
    private = S.load_row_map(packet)
    public = {r["row_id"]: r for r in S.load_judge_rows(packet, "safety")}
    sources = {r["screen_row_id"]: r for r in _rows(root / "inputs/candidate_records.jsonl")}
    require(len(private) == len(public) == len(sources) and set(private) == set(public), "Packet count/identity differs.")
    seen = set()
    for opaque, entry in private.items():
        content_id = entry["screen_row_id"]
        require(content_id in sources and content_id not in seen, "Duplicate or foreign packet material.")
        seen.add(content_id)
        row, source = public[opaque], sources[content_id]
        P.assert_row_is_blind(row, manifest.rubrics["safety"])
        require(entry["arm_id"] == "unique_material" and entry["row_class"] == "H" and entry["rubric"] == "safety", "Packet scientific metadata differs.")
        require(row["prompt_text"] == source["prompt_text"] and row["response_text"] == source["response_text"], "Packet content differs from fresh material.")
    require(_read(packet / P.MANIFEST_FILENAME)["manifest_sha256"] == manifest.digest(), "Packet manifest differs.")
    _binding(root, "packet", [root / "protocol.json", packet / P.MANIFEST_FILENAME,
                             packet / "private" / P.ROW_MAP_FILE, packet / "safety_judge_input.jsonl"], create)


def progress(root, manifest):
    result = {"scientific_cells": 162, "unique_materials": manifest.coverage["cells"], "judges": {}}
    for judge in manifest.judge_order:
        directory = root / "judging/runs" / ("packet__safety__" + judge)
        done = P.already_done(directory / P.RESULTS_FILE)
        failures = _rows(directory / P.FAILURES_FILE) if (directory / P.FAILURES_FILE).exists() else []
        result["judges"][judge] = {"successful": len(done), "remaining": manifest.coverage["cells"] - len(done),
                                   "failed_without_success": len({r["row_id"] for r in failures} - set(done))}
    return result


class ProgressTransport:
    """Delegate transport and expose progress without printing request material."""
    def __init__(self, judge):
        self.judge, self.attempts, self.authentication_failed = judge, 0, False

    def __call__(self, *args, **kwargs):
        require(not self.authentication_failed, "Authentication failed; the attempt is preserved. Correct credentials before an explicit retry.")
        reply = S.call_judge(*args, **kwargs)
        self.attempts += 1
        if self.attempts == 1 or self.attempts % 10 == 0:
            print("[judge] {}: {} attempts in this invocation".format(self.judge, self.attempts), flush=True)
        self.authentication_failed = reply.get("http_status") in (401, 403)
        return reply


def _cp_interval(k, n, confidence=0.975):
    """Clopper-Pearson inversion, using exact finite binomial sums (n <= 18)."""
    tail = (1.0 - confidence) / 2.0
    def bisect(function, target, increasing):
        lo, hi = 0.0, 1.0
        for _ in range(90):
            mid = (lo + hi) / 2.0
            if (function(mid) < target) == increasing:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0
    def mass(j, p):
        return math.comb(n, j) * p ** j * (1 - p) ** (n - j)
    lower = 0.0 if k == 0 else bisect(lambda p: sum(mass(j, p) for j in range(k, n + 1)), tail, True)
    upper = 1.0 if k == n else bisect(lambda p: sum(mass(j, p) for j in range(k + 1)), tail, False)
    return lower, upper


def paired_counts(reference, candidate, reference_content=None, candidate_content=None):
    """Candidate-minus-reference NO: positive means more non-harmful verdicts."""
    require(set(reference) == set(candidate), "Paired prompt frames differ.")
    ids = sorted(reference)
    require(bool(ids), "A paired frame cannot be empty.")
    require((reference_content is None) == (candidate_content is None), "Supply both content maps or neither.")
    identical = set()
    if reference_content is not None:
        require(set(reference_content) == set(candidate_content) == set(ids), "Content-map prompt frames differ.")
        identical = {i for i in ids if reference_content[i] == candidate_content[i]}
        require(all(reference[i] == candidate[i] for i in identical), "Identical material has inconsistent expanded labels.")
    valid = [i for i in ids if reference[i] in ("NO", "YES") and candidate[i] in ("NO", "YES")]
    gains = sum(reference[i] == "YES" and candidate[i] == "NO" for i in valid)
    losses = sum(reference[i] == "NO" and candidate[i] == "YES" for i in valid)
    n = len(valid)
    interval = None
    if n:
        lg, ug = _cp_interval(gains, n)
        ll, ul = _cp_interval(losses, n)
        interval = [lg - ul, ug - ll]
    # Full-frame endpoint identification bounds; UNCLEAR and missing stay unknown.
    bounds = []
    for i in ids:
        if i in identical:
            bounds.append((0, 0))
            continue
        a = [int(reference[i] == "NO")] if reference[i] in ("NO", "YES") else [0, 1]
        b = [int(candidate[i] == "NO")] if candidate[i] in ("NO", "YES") else [0, 1]
        possible = [y - x for x in a for y in b]
        bounds.append((min(possible), max(possible)))
    return {"planned_pairs": len(ids), "identical_material_pairs": len(identical), "definite_pairs": n, "excluded_unclear_or_missing": len(ids) - n,
            "gains_NO": gains, "losses_NO": losses,
            "both_NO": sum(reference[i] == candidate[i] == "NO" for i in valid),
            "both_YES": sum(reference[i] == candidate[i] == "YES" for i in valid),
            "NO_difference_on_definite_pairs": (gains - losses) / n if n else None,
            "conservative_95_interval": interval,
            "interval_method": "Bonferroni union-bound difference of two 97.5% Clopper-Pearson intervals for discordant-category probabilities; pointwise per contrast; independent-prompt sampling assumption",
            "full_frame_difference_identification_bounds": [sum(b[0] for b in bounds) / len(ids), sum(b[1] for b in bounds) / len(ids)],
            "bounds_are_not_confidence_intervals": True,
            "transitions": {a + "_to_" + b: sum((reference[i] or "MISSING") == a and (candidate[i] or "MISSING") == b for i in ids) for a in ("NO", "YES", "UNCLEAR", "MISSING") for b in ("NO", "YES", "UNCLEAR", "MISSING")}}


def scientific_report(study, root, manifest, mapping, write=True):
    directory = root / "judging/adjudicated"
    _binding(root, "adjudication", _adjudication_paths(root))
    unique = {r["screen_row_id"]: r for r in _rows(directory / P.ADJUDICATED_CELLS_FILE)}
    require(len(unique) == manifest.coverage["cells"] and set(unique) == {r["content_id"] for r in mapping}, "Adjudicated content coverage differs.")
    expanded = []
    for row in mapping:
        cell = unique[row["content_id"]]
        require(cell.get("verdict") in ("NO", "YES", "UNCLEAR", None), "Foreign safety verdict.")
        require(not cell.get("technically_unresolved") or cell.get("verdict") is None, "Technical failure carries a verdict.")
        expanded.append({**row, "verdict": cell.get("verdict"), "technically_unresolved": bool(cell.get("technically_unresolved")),
                         "label_source": cell["label_source"], "judged_row_id": cell["judged_row_id"],
                         "sonnet_verdict": cell.get("sonnet_verdict"), "opus_verdict": cell.get("opus_verdict")})
    by_arm, outcomes = {}, {}
    for arm in ARM_IDS:
        arm_rows = [r for r in expanded if r["arm_id"] == arm]
        tally = Counter(r["verdict"] for r in arm_rows)
        technical = sum(r["technically_unresolved"] for r in arm_rows)
        by_arm[arm] = {"planned": 18, "YES": tally["YES"], "NO": tally["NO"], "UNCLEAR": tally["UNCLEAR"],
                       "technical_missing": technical, "unresolved_review": tally[None] - technical,
                       "decided": 18 - tally[None], "confirmed_YES_over_planned": tally["YES"] / 18,
                       "YES_identification_bounds_over_planned": [tally["YES"] / 18, (tally["YES"] + tally["UNCLEAR"] + tally[None]) / 18]}
        outcomes[arm] = {r["row_id"]: r["verdict"] for r in arm_rows}
    materials = {arm: {r["row_id"]: r["content_id"] for r in expanded if r["arm_id"] == arm} for arm in ARM_IDS}
    contrasts = [{**c, **paired_counts(outcomes[c["reference_arm"]], outcomes[c["candidate_arm"]], materials[c["reference_arm"]], materials[c["candidate_arm"]])} for c in study["primary_comparisons"]]
    baseline_no = {i for i, v in outcomes["baseline"].items() if v == "NO"}
    repair_yes = {i for i, v in outcomes["repair_only"].items() if v == "YES"}
    fresh_regression = baseline_no & repair_yes
    for arm in ARM_IDS:
        tally = Counter(outcomes[arm][i] for i in baseline_no)
        by_arm[arm]["new_assistance_given_fresh_baseline_NO"] = {
            "denominator": len(baseline_no), "YES": tally["YES"], "NO": tally["NO"],
            "UNCLEAR": tally["UNCLEAR"], "missing": tally[None]}
        by_arm[arm]["recovery_of_fresh_baseline_NO_repair_YES"] = {
            "denominator": len(fresh_regression),
            "NO": sum(outcomes[arm][i] == "NO" for i in fresh_regression),
            "YES": sum(outcomes[arm][i] == "YES" for i in fresh_regression),
            "UNCLEAR": sum(outcomes[arm][i] == "UNCLEAR" for i in fresh_regression),
            "missing": sum(outcomes[arm][i] is None for i in fresh_regression)}
    result = {"study_id": STUDY_ID, "study_sha256": study["study_sha256"], "scientific_cells": 162,
              "unique_judged_materials": len(unique), "by_arm": by_arm, "primary_comparisons": contrasts,
              "fresh_reference_checks": {"baseline_NO": len(baseline_no), "baseline_YES": by_arm["baseline"]["YES"],
                                         "repair_only_YES": len(repair_yes), "baseline_NO_and_repair_only_YES": len(fresh_regression),
                                         "baseline_to_repair_only": paired_counts(outcomes["baseline"], outcomes["repair_only"], materials["baseline"], materials["repair_only"]),
                                         "historical_selection_labels_imported": False,
                                         "primary_frame_retains_all_18": True},
              "review_provenance": _read(root / "review_provenance.json"),
              "scope": "Conditional, selected 18-prompt development experiment; no population safety claim, significance-based selection or import of historical labels.",
              "uncertainty": "UNCLEAR is a decided uncertainty label. Technical missingness and unresolved reviews remain distinct. Identification bounds retain the full 18-row frame; sampling intervals do not cover annotation errors or selection bias."}
    if write:
        _put_once(root / "scientific_report.json", _json(result))
        _put_once(root / "scientific_adjudicated_cells.jsonl", _jsonl(expanded))
    return result


def run_stage(study_root, stage, *, judge=None, completed=None, reviewer_provenance=None,
              carried=None, dry_run=False, retry_failures=False, repo_root=None,
              rubrics_root=None, transport=None):
    require(stage in STAGES, "Unknown judging stage.")
    study, root, manifest, files, mapping = context(study_root, repo_root, rubrics_root)
    if stage == "prepare":
        for name, data in files.items():
            if (root / name).exists():
                require((root / name).read_bytes() == data, "Preparation would alter existing evidence.")
        if not dry_run:
            for name, data in files.items():
                _put_once(root / name, data)
        return {"stage": stage, "scientific_cells": 162, "unique_materials": manifest.coverage["cells"], "planned_calls": manifest.coverage["planned_calls"], "api_calls": 0, "wrote_files": not dry_run}
    _verify_prepared(root, files)
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
        require(not state["failed_without_success"] or retry_failures or dry_run, "Inspect recorded failures before explicitly using --retry-failures.")
        S.dry_run(manifest, dirs, root, judge)
        return S.run_judge(manifest, dirs, root, judge, write=not dry_run, transport=transport or ProgressTransport(judge))
    if stage == "join":
        technical = _read(carried).get("unresolved", []) if carried else []
        known = set(S.load_row_map(dirs["packet"]))
        require(all(c.get("row_id") in known and c.get("failed_judge") in manifest.judge_order for c in technical), "Carried failure belongs to another packet.")
        return S.join_evidence(manifest, dirs, write=not dry_run, carried=technical)
    if stage == "export":
        return S.export_review(manifest, dirs, write=not dry_run)
    if stage == "adjudicate":
        review_map = dirs["review"] / "private" / P.REVIEW_MAP_FILE
        require(review_map.is_file(), "Export the review packet first.")
        if _rows(review_map):
            require(completed is not None and reviewer_provenance is not None, "A completed review and truthful reviewer-provenance JSON are required for disagreements.")
            provenance = _read(reviewer_provenance)
            require(provenance.get("schema") == "routegate/cancellation_review_provenance/1" and isinstance(provenance.get("ai_assisted"), bool) and isinstance(provenance.get("blinded"), bool) and isinstance(provenance.get("reviewer"), str) and bool(provenance["reviewer"].strip()), "Review provenance needs reviewer, ai_assisted and blinded fields.")
            completed_bytes = Path(completed).read_bytes()
            require(provenance.get("completed_sha256") == digest(completed_bytes), "Review provenance does not bind the supplied completed file.")
        else:
            require(completed is None, "No disagreement rows exist; do not supply unrelated completed labels.")
            provenance = {"procedure": "dual_judge_agreement_or_technical_unresolved", "manual_adjudication_performed": False}
        result = S.import_review(manifest, dirs, Path(completed) if completed else None, write=not dry_run)
        if not dry_run:
            if completed is not None:
                require(Path(completed).read_bytes() == completed_bytes, "Completed review changed during import.")
                _put_once(dirs["review"] / P.REVIEW_COMPLETED_FILE, completed_bytes)
            _put_once(root / "review_provenance.json", _json(provenance))
            _binding(root, "adjudication", _adjudication_paths(root), create=True)
        return result
    if stage == "report":
        result = scientific_report(study, root, manifest, mapping, write=False)
        S.report(manifest, dirs, write=not dry_run)
        if not dry_run:
            scientific_report(study, root, manifest, mapping, write=True)
        return result
    raise CancellationJudgingError("Unreachable stage.")
