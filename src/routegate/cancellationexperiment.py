"""Isolated generation runner for the fixed cancellation-control diagnostic.

Existing data and evaluation outputs are read only. Study inputs, new prompt
captures and successful generations are committed atomically in a new folder.
No judge calls, training, or automatic changes to the design occur here.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import tempfile
from datetime import datetime, timezone

import numpy as np

from . import cancellationcontrols as core
from .runtime import GenerationConfig


class StudyError(ValueError):
    """A study input, committed record or execution setting did not verify."""


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def text_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, sort_keys=True, indent=2,
                                  ensure_ascii=False, allow_nan=False) + "\n").encode())


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (ValueError, UnicodeError) as exc:
        raise StudyError(f"Invalid JSONL in {path}; preserve the file and inspect it") from exc


def write_jsonl(path, rows):
    atomic_bytes(path, "".join(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                        allow_nan=False) + "\n" for row in rows).encode())


@contextlib.contextmanager
def study_lock(study_root):
    import fcntl
    root = Path(study_root)
    if not (root / "study_manifest.json").is_file():
        raise StudyError("Initialise this study before claiming its generation lock")
    with (root / "generation.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StudyError("Another generation process owns this study; do not run two writers") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _repo_root():
    return Path(__file__).resolve().parents[2]


def source_hashes(repo_root):
    """Pin the entire small package, including indirect generation/judge imports."""
    root = Path(repo_root)
    paths = sorted((root / "src/routegate").glob("*.py"))
    paths += sorted((root / "experiments/cancellation_controls_v1").glob("*.py"))
    paths += sorted((root / "tests").glob("test_cancellation*.py"))
    if not paths:
        raise StudyError("No RouteGate source found")
    return {str(path.relative_to(root)): core.sha256_file(path) for path in paths}


def init_study(repo_root, data_root, study_root):
    from .cancellationjudging import frozen_settings
    repo, data, root = (Path(value).resolve() for value in (repo_root, data_root, study_root))
    # Avoid accidentally placing writes into the repository or frozen evidence.
    if root == data or root == repo or repo in root.parents or data / "frozen" in root.parents:
        raise StudyError("Use a separate new study folder outside the repository and frozen inputs")
    if root.exists():
        if (root / "study_manifest.json").is_file():
            return verify_study(root, repo)
        raise StudyError(f"Output folder already exists without this study's manifest: {root}")
    rows = core.load_selected_rows(repo, data)
    arrays, geometry = core.build_control_bank(data / core.BANK_RELATIVE_PATH)
    judging = frozen_settings(repo)
    rubric_source = data / "triplet_validation/frozen/rq1_judge_protocol"
    names = ("safety_instructions.txt", "safety_user_template.txt", "safety_schema.json")
    rubric_bytes = {name: (rubric_source / name).read_bytes() for name in names}
    config = GenerationConfig()
    config.validate()
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".cancellation-init-", dir=root.parent))
    try:
        write_jsonl(temporary / "selected_rows.jsonl", rows)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        atomic_bytes(temporary / "controls.npz", buffer.getvalue())
        write_json(temporary / "control_metadata.json", geometry)
        for name, content in rubric_bytes.items():
            atomic_bytes(temporary / "judge_rubrics" / name, content)
        artifacts = {str(p.relative_to(temporary)): core.sha256_file(p)
                     for p in sorted(temporary.rglob("*")) if p.is_file()}
        comparisons = [{"name": "combination_vs_attenuation", "reference_arm": "attenuated_repair",
                        "candidate_arm": "original_combination"}]
        comparisons += [{"name": f"combination_vs_random_{i}", "reference_arm": f"random_orthogonal_{i}",
                         "candidate_arm": "original_combination"} for i in range(1, 4)]
        manifest = {
            "study_id": core.PROTOCOL, "created_utc": now(), "schema_version": 1,
            "study_root": str(root), "repo_root_at_init": str(repo), "data_root_at_init": str(data),
            "arm_ids": list(core.ARM_IDS), "arms": [{"arm_id": arm} for arm in core.ARM_IDS],
            "row_ids": [row["row_id"] for row in rows], "expected_rows": 18,
            "expected_generations": 162, "row_order": "frozen selected ID order",
            "arm_order": "fixed listed order within each prompt",
            "generation": config.as_dict(), "generation_fingerprint": config.fingerprint(),
            "source_hashes": source_hashes(repo), "artifacts": artifacts,
            "judging": judging, "primary_comparisons": comparisons,
            "design": {
                "scope": "mechanism diagnostic on 18 previously selected development H regressions",
                "selection_is_historical": True, "alpha_S_selected_using_previous_development_results": True,
                "held_out_generalisation_test": False, "external_380_used": False,
                "training_or_model_weight_changes": False,
                "budget": "same freshly captured unsteered layer-input norm for all arms of each prompt",
                "capture_persistence": "one atomic capture per prompt, reused unchanged on resume",
                "write": "float32 weighted sum, scale by unsteered input norm, cast delta to live dtype before addition",
                "normalise_combined_edit": False, "intervention": "final prompt token, prefill only",
                "numerical_scope": "intended local matching; realised bfloat16 edits measured at every site",
                "random_control_count": 3, "seed": core.RANDOM_SEED,
                "historical_responses_or_verdicts_imported": False,
                "material_deduplication": "within this new study, exact prompt/response/rubric shared by arms is judged once per judge",
                "unclear_and_technical_failures": "reported separately, never counted as no harmful assistance",
                "fresh_reference_changes": "reported on fixed 18; no dropping or replacement of selected cases",
                "decision": "compare paired outcomes with attenuation and all three random controls; no automated method selection",
            },
        }
        manifest["study_sha256"] = digest(manifest)
        write_json(temporary / "study_manifest.json", manifest)
        # Validate pinned rubric bytes and geometry before publishing the folder.
        verify_study(temporary, repo)
        if root.exists():
            raise StudyError("Output appeared during initialisation; refusing to replace it")
        os.rename(temporary, root)
    finally:
        if temporary.exists():
            import shutil
            shutil.rmtree(temporary)
    return manifest


def verify_study(study_root, repo_root=None):
    root = Path(study_root)
    manifest = json.loads((root / "study_manifest.json").read_text())
    content = {k: v for k, v in manifest.items() if k != "study_sha256"}
    if manifest.get("study_sha256") != digest(content):
        raise StudyError("Study manifest hash mismatch")
    if manifest.get("study_id") != core.PROTOCOL or manifest.get("arm_ids") != list(core.ARM_IDS):
        raise StudyError("Wrong study or changed arm inventory")
    if manifest.get("generation") != GenerationConfig().as_dict():
        raise StudyError("Generation settings differ from the implemented fixed protocol")
    if manifest.get("generation_fingerprint") != GenerationConfig().fingerprint():
        raise StudyError("Generation fingerprint mismatch")
    for relative, expected in manifest["artifacts"].items():
        path = root / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts or not path.is_file():
            raise StudyError(f"Missing or invalid study artifact: {relative}")
        if core.sha256_file(path) != expected:
            raise StudyError(f"Study artifact hash mismatch: {relative}")
    repo = Path(repo_root) if repo_root is not None else _repo_root()
    for relative, expected in manifest["source_hashes"].items():
        if core.sha256_file(repo / relative) != expected:
            raise StudyError(f"Source changed since initialisation: {relative}")
    rows = read_jsonl(root / "selected_rows.jsonl")
    if [r["row_id"] for r in rows] != manifest["row_ids"] or len(rows) != 18:
        raise StudyError("Selected row inventory mismatch")
    if core.selected_rows_digest(manifest["row_ids"]) != core.SELECTED_ROWS_SHA256:
        raise StudyError("Rows differ from the historical 18-case selection")
    for row in rows:
        if row.get("row_class") != "H" or text_hash(row["prompt_text"]) != row["prompt_sha256"]:
            raise StudyError("Invalid selected prompt")
    if manifest.get("expected_generations") != len(rows) * len(core.ARM_IDS):
        raise StudyError("Generation count mismatch")
    arrays = load_control_arrays(root)
    metadata = json.loads((root / "control_metadata.json").read_text())
    for key, value in arrays.items():
        if hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() != metadata["arrays"][key]["sha256"]:
            raise StudyError(f"Control-array hash mismatch: {key}")
    from .cancellationjudging import frozen_settings
    if manifest["judging"] != frozen_settings(repo):
        raise StudyError("Judging settings differ from the frozen protocol")
    # Pin bytes to the original protocol as well as to this manifest.
    spec = manifest["judging"]["rubrics"]["safety"]
    for key, name in (("instructions_sha256", "safety_instructions.txt"),
                      ("user_template_sha256", "safety_user_template.txt"),
                      ("schema_sha256", "safety_schema.json")):
        if key in spec and core.sha256_file(root / "judge_rubrics" / name) != spec[key]:
            raise StudyError(f"Original rubric hash mismatch: {name}")
    return manifest


def load_control_arrays(study_root):
    with np.load(Path(study_root) / "controls.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    core.validate_control_bank(arrays)
    return arrays


def _is_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _finite_number(value, *, nonnegative=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and bool(np.isfinite(value)) and (not nonnegative or value >= 0))


def _validate_record(record, manifest, row_map):
    if not isinstance(record, dict):
        raise StudyError("Generation record must be an object")
    key = (record.get("row_id"), record.get("arm_id"))
    if not all(isinstance(value, str) for value in key) or key[0] not in row_map or key[1] not in core.ARM_IDS:
        raise StudyError("Unknown row or arm in generation record")
    row = row_map[key[0]]
    if record.get("status") != "ok" or record.get("study_sha256") != manifest["study_sha256"]:
        raise StudyError("Invalid status or study hash in generation record")
    if record.get("prompt_text") != row["prompt_text"] or record.get("prompt_sha256") != row["prompt_sha256"]:
        raise StudyError("Generation prompt differs from the selected prompt")
    response = record.get("response_text")
    if not isinstance(response, str) or record.get("response_sha256") != text_hash(response):
        raise StudyError("Response hash mismatch")
    if record.get("generation_fingerprint") != manifest["generation_fingerprint"]:
        raise StudyError("Generation fingerprint mismatch in record")
    for field in ("execution_sha256", "capture_sha256", "record_sha256"):
        if not _is_sha256(record.get(field)):
            raise StudyError(f"Missing or invalid {field} in generation record")
    try:
        expected_hash = digest({k: v for k, v in record.items() if k != "record_sha256"})
    except (TypeError, ValueError, OverflowError) as exc:
        raise StudyError("Generation record contains invalid JSON values") from exc
    if record["record_sha256"] != expected_hash:
        raise StudyError("Generation record hash mismatch")
    planned = key[1] != "baseline"
    expected_hooks = 3 if planned else 0
    for field in ("hooks_expected", "hooks_fired"):
        if type(record.get(field)) is not int or record[field] != expected_hooks:
            raise StudyError(f"Unexpected {field} for {key[1]}")
    diagnostics = record.get("layer_diagnostics")
    if not isinstance(diagnostics, dict) or set(diagnostics) != {str(state) for state in core.STATES}:
        raise StudyError("Layer diagnostics must cover exactly output states 32, 33 and 34")
    for state in core.STATES:
        diagnostic = diagnostics[str(state)]
        if not isinstance(diagnostic, dict):
            raise StudyError("Layer diagnostic must be an object")
        for field, expected in (("layer", state-1), ("output_state", state),
                                ("budget_reference_state", state-1)):
            if type(diagnostic.get(field)) is not int or diagnostic[field] != expected:
                raise StudyError(f"Incorrect {field} in diagnostic at output state {state}")
        if diagnostic.get("live_dtype") != "torch.bfloat16":
            raise StudyError("Live intervention dtype differs from frozen bfloat16 generation")
        if diagnostic.get("edit_planned") is not planned or diagnostic.get("plan_status") != ("ok" if planned else "no_intervention"):
            raise StudyError("Planned edit diagnostic disagrees with the declared arm")
        for field in ("frozen_unsteered_input_norm", "live_before_l2", "live_after_l2"):
            if not _finite_number(diagnostic.get(field), nonnegative=True):
                raise StudyError(f"Invalid {field} in layer diagnostic")
        for group in ("intended_delta", "cast_delta", "realised_delta"):
            values = diagnostic.get(group)
            if not isinstance(values, dict):
                raise StudyError(f"Missing {group} measurements")
            for field in ("l2", "repair_coefficient", "perpendicular_l2"):
                if not _finite_number(values.get(field), nonnegative=field != "repair_coefficient"):
                    raise StudyError(f"Invalid {group}.{field} measurement")
                if not planned and values[field] != 0:
                    raise StudyError("Baseline has a non-zero edit measurement")
        changed = diagnostic.get("changed_coordinates")
        if type(changed) is not int or not 0 <= changed <= 2048 or (not planned and changed != 0):
            raise StudyError("Invalid changed coordinate count")
        if diagnostic.get("planned_edit_rounded_to_identity") is not (planned and changed == 0):
            raise StudyError("Rounding diagnostic disagrees with planned edit and changed coordinates")
    return key


def _validate_record_links(root, record, manifest):
    """Verify committed provenance bytes without loading activation tensors."""
    root = Path(root)
    capture_path = root / "captures" / (text_hash(record["row_id"]) + ".npz")
    try:
        environment = json.loads((root / "execution_environment.json").read_text())
        metadata = json.loads(capture_path.with_suffix(".json").read_text())
        if not isinstance(environment, dict) or digest(environment) != record["execution_sha256"]:
            raise StudyError("Execution environment digest mismatch")
        if not isinstance(metadata, dict):
            raise StudyError("Capture metadata must be an object")
        for field, expected in (("row_id", record["row_id"]),
                                ("prompt_sha256", record["prompt_sha256"]),
                                ("study_sha256", manifest["study_sha256"]),
                                ("execution_sha256", record["execution_sha256"]),
                                ("sha256", record["capture_sha256"])):
            if metadata.get(field) != expected:
                raise StudyError(f"Capture metadata mismatch: {field}")
        if metadata.get("states") != [31, 32, 33, 34] or type(metadata.get("generated_tokens")) is not int or metadata["generated_tokens"] != 0:
            raise StudyError("Capture was not the declared unsteered prompt capture")
        count = metadata.get("prompt_token_count")
        if type(count) is not int or count <= 0 or type(record.get("prompt_token_count")) is not int or record["prompt_token_count"] != count:
            raise StudyError("Capture and generation prompt token counts disagree")
        if core.sha256_file(capture_path) != record["capture_sha256"]:
            raise StudyError("Capture byte hash mismatch")
    except (OSError, TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, StudyError):
            raise
        raise StudyError(f"Missing or invalid generation provenance: {exc}") from exc


def load_records(study_root, manifest=None):
    root = Path(study_root)
    manifest = manifest or verify_study(root)
    rows = {r["row_id"]: r for r in read_jsonl(root / "selected_rows.jsonl")}
    result = {}
    for record in read_jsonl(root / "generations/records.jsonl"):
        key = _validate_record(record, manifest, rows)
        if key in result:
            raise StudyError(f"Duplicate successful generation: {key}")
        _validate_record_links(root, record, manifest)
        result[key] = record
    return result


def save_success(study_root, record):
    """Caller must own study_lock; atomic replacement preserves all prior rows."""
    root = Path(study_root)
    manifest = verify_study(root)
    records = load_records(root, manifest)
    rows = {r["row_id"]: r for r in read_jsonl(root / "selected_rows.jsonl")}
    key = _validate_record(record, manifest, rows)
    if key in records:
        raise StudyError(f"Refusing to overwrite successful generation: {key}")
    _validate_record_links(root, record, manifest)
    records[key] = record
    write_jsonl(root / "generations/records.jsonl", list(records.values()))


def record_attempt(study_root, record):
    path = Path(study_root) / "generations/attempts.jsonl"
    attempts = read_jsonl(path)
    attempts.append(record)
    write_jsonl(path, attempts)


def progress(study_root, repo_root=None):
    manifest = verify_study(study_root, repo_root)
    records = load_records(study_root, manifest)
    attempts = read_jsonl(Path(study_root) / "generations/attempts.jsonl")
    failed = {(r.get("row_id"), r.get("arm_id")) for r in attempts if r.get("status") == "error"} - set(records)
    return {"stage": "progress", "study_id": core.PROTOCOL, "successful": len(records),
            "expected": 162, "remaining": 162-len(records), "unresolved_failed_generations": len(failed),
            "by_arm": {arm: sum(key[1] == arm for key in records) for arm in core.ARM_IDS},
            "capped_responses": sum(bool(r.get("hit_max_new_tokens")) for r in records.values()),
            "empty_responses": sum(not r["response_text"].strip() for r in records.values()),
            "complete": len(records) == 162}


def build_plans(arrays, arm_id, capture):
    from .dualsteer import build_dual_plan
    plans = {}
    for state in core.STATES:
        if state - 1 not in capture:
            raise StudyError(f"Missing unsteered input capture for layer {state-1}")
        repair, second, alpha_r, alpha_s = core.arm_components(arrays, arm_id, state)
        plans[state] = build_dual_plan(state, repair, second, alpha_r, alpha_s,
                                       capture[state-1], budget_reference="block_input")
    return plans


def make_controller(model, arrays):
    import torch
    from .hooks import PatchController, first_tensor

    class MeasuredController(PatchController):
        def reset_run(self, *args, **kwargs):
            self.diagnostics = {}
            return super().reset_run(*args, **kwargs)

        def _on_state(self, state, output):
            if self.run.mode != "patch" or self.run.forward_index != 0 or state not in self.run.plans:
                return super()._on_state(state, output)
            before_tensor = first_tensor(output)
            if before_tensor is None or before_tensor.ndim != 3 or before_tensor.shape[0] != 1:
                raise StudyError(f"Invalid live activation at layer {state-1}")
            before = before_tensor[0, -1].detach().float().cpu().clone()
            plan = self.run.plans[state]
            proposed = (torch.zeros_like(before) if plan.delta is None else plan.delta.detach().float().cpu())
            cast = proposed.to(dtype=before_tensor.dtype).float()
            result = super()._on_state(state, output)
            after = first_tensor(result)[0, -1].detach().float().cpu()
            actual = after - before
            repair = torch.tensor(arrays["repair"][core.STATES.index(state)], dtype=torch.float32)
            def measures(vector):
                coefficient = float(torch.dot(vector, repair) / torch.dot(repair, repair))
                perpendicular = vector - coefficient * repair
                return {"l2": float(torch.linalg.vector_norm(vector)), "repair_coefficient": coefficient,
                        "perpendicular_l2": float(torch.linalg.vector_norm(perpendicular))}
            if not torch.isfinite(actual).all() or not torch.isfinite(after).all():
                raise StudyError(f"Non-finite written activation at layer {state-1}")
            self.diagnostics[str(state)] = {
                "layer": state-1, "output_state": state, "budget_reference_state": state-1,
                "frozen_unsteered_input_norm": float(plan.reference_norm), "live_dtype": str(before_tensor.dtype),
                "live_before_l2": float(torch.linalg.vector_norm(before)), "live_after_l2": float(torch.linalg.vector_norm(after)),
                "intended_delta": measures(proposed), "cast_delta": measures(cast),
                "realised_delta": measures(actual), "changed_coordinates": int(torch.count_nonzero(actual)),
                "edit_planned": bool(plan.applies_anything), "plan_status": plan.status,
                "planned_edit_rounded_to_identity": bool(plan.applies_anything and not torch.count_nonzero(actual)),
            }
            return result

    return MeasuredController(model, core.STATES, budget_reference="block_input")


def _execution_environment():
    import torch
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"), "numpy": np.__version__,
            "cuda": torch.version.cuda, "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "seed": 0, "note": "Greedy settings are fixed; hardware-level bitwise reproducibility is not asserted"}


def _capture_for_row(root, row, model, tokenizer, controller, config, manifest, execution_hash):
    from .runtime import capture_prompt_states
    import torch
    path = root / "captures" / (text_hash(row["row_id"]) + ".npz")
    metadata_path = path.with_suffix(".json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["study_sha256"] != manifest["study_sha256"] or metadata["prompt_sha256"] != row["prompt_sha256"]:
            raise StudyError("Capture provenance mismatch")
        if metadata["execution_sha256"] != execution_hash or core.sha256_file(path) != metadata["sha256"]:
            raise StudyError("Capture bytes or execution settings changed")
        with np.load(path, allow_pickle=False) as archive:
            capture = {int(key): torch.from_numpy(archive[key].copy()) for key in archive.files}
        return capture, metadata
    if path.exists():
        raise StudyError(f"Uncommitted capture found: {path}; preserve it and inspect before resuming")
    count = capture_prompt_states(model, tokenizer, row["prompt_text"], controller,
                                 "unsteered", row["row_id"], config)
    capture = controller.capture_for("unsteered")
    if set(capture) != {31, 32, 33, 34}:
        raise StudyError("Capture did not cover all required input and output states")
    saved = {}
    for state, tensor in capture.items():
        if tuple(tensor.shape) != (2048,) or not torch.isfinite(tensor).all():
            raise StudyError(f"Invalid capture at state {state}")
        saved[str(state)] = tensor.numpy()
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **saved)
    atomic_bytes(path, buffer.getvalue())
    metadata = {"row_id": row["row_id"], "prompt_sha256": row["prompt_sha256"],
                "study_sha256": manifest["study_sha256"], "execution_sha256": execution_hash,
                "sha256": core.sha256_file(path), "prompt_token_count": count, "created_utc": now(),
                "states": [31,32,33,34], "generated_tokens": 0}
    write_json(metadata_path, metadata)
    return capture, metadata


def generate(study_root, repo_root=None, limit=None, retry_failures=False):
    import torch
    from .runtime import load_model_and_tokenizer, generate_response
    root = Path(study_root)
    if limit is not None and limit < 1:
        raise StudyError("limit must be positive")
    with study_lock(root):
        manifest = verify_study(root, repo_root)
        records = load_records(root, manifest)
        if len(records) == 162:
            return progress(root, repo_root)
        attempts = read_jsonl(root / "generations/attempts.jsonl")
        failures = {(r.get("row_id"), r.get("arm_id")) for r in attempts if r.get("status") == "error"} - set(records)
        if failures and not retry_failures:
            raise StudyError("A generation failed previously. Inspect attempts.jsonl, then use --retry-failures deliberately")
        if not torch.cuda.is_available():
            raise StudyError("Generation requires the allocated GPU node; CPU checks make no generations")
        config = GenerationConfig()
        environment = _execution_environment()
        execution_hash = digest(environment)
        execution_path = root / "execution_environment.json"
        if execution_path.exists():
            if json.loads(execution_path.read_text()) != environment:
                raise StudyError("Execution environment changed; do not mix library versions or GPU types in this study")
        else:
            write_json(execution_path, environment)
        arrays = load_control_arrays(root)
        model, tokenizer = load_model_and_tokenizer(config)
        controller = make_controller(model, arrays)
        attempted = 0
        try:
            for row in read_jsonl(root / "selected_rows.jsonl"):
                remaining = [arm for arm in core.ARM_IDS if (row["row_id"], arm) not in records]
                if not remaining:
                    continue
                torch.manual_seed(0)
                torch.cuda.manual_seed_all(0)
                capture, capture_metadata = _capture_for_row(root, row, model, tokenizer, controller,
                                                            config, manifest, execution_hash)
                for arm in remaining:
                    if limit is not None and attempted >= limit:
                        return progress(root, repo_root)
                    base = {**row, "arm_id": arm, "study_sha256": manifest["study_sha256"],
                            "execution_sha256": execution_hash, "capture_sha256": capture_metadata["sha256"],
                            "generation_fingerprint": config.fingerprint(), "started_utc": now()}
                    record_attempt(root, {"row_id": row["row_id"], "arm_id": arm, "status": "started",
                                          "study_sha256": manifest["study_sha256"], "created_utc": now()})
                    try:
                        plans = build_plans(arrays, arm, capture)
                        torch.manual_seed(0)
                        torch.cuda.manual_seed_all(0)
                        result = generate_response(model, tokenizer, row["prompt_text"], config,
                                                   controller, "patch", plans, row["row_id"])
                        if set(controller.diagnostics) != {"32", "33", "34"}:
                            raise StudyError("Missing measurements of the three intervention sites")
                        record = {**base, **result, "status": "ok", "completed_utc": now(),
                                  "hooks_expected": 0 if arm == "baseline" else 3,
                                  "response_sha256": text_hash(result["response_text"]),
                                  "layer_diagnostics": controller.diagnostics}
                        record["record_sha256"] = digest(record)
                        save_success(root, record)
                        records[(row["row_id"], arm)] = record
                        attempted += 1
                        print(f"[generation] {len(records)}/162 complete | {arm}", flush=True)
                    except Exception as exc:
                        record_attempt(root, {"row_id": row["row_id"], "arm_id": arm, "status": "error",
                                              "study_sha256": manifest["study_sha256"], "created_utc": now(),
                                              "error_type": type(exc).__name__, "error": str(exc)})
                        raise
                controller.captures.clear()
        finally:
            controller.close()
        return progress(root, repo_root)


def preflight(study_root, repo_root=None, check_model=False):
    manifest = verify_study(study_root, repo_root)
    summary = progress(study_root, repo_root)
    if check_model:
        from .runtime import resolve_snapshot
        revision, _ = resolve_snapshot()
        summary["local_model_revision_verified"] = revision
    summary.update({"stage": "preflight", "study_sha256": manifest["study_sha256"],
                    "model_loaded": False, "api_calls": 0, "new_generations": 0})
    return summary
