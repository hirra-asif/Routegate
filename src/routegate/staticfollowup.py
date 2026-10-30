"""Prospective, ungated counterpart of the released Safety-first static edit.

The original final protocol and outputs are read-only. This follow-up reuses
the released 380 prompts and the exact final-run arithmetic. It does not train
a model, select a strength, evaluate the gate, or extend the original protocol.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

from . import finaljudging as F
from .finalrunner import STATIC_ALPHA, STATIC_BLOCKS, STATIC_STATES, GATE_READ_STATE
from .validation import ValidationError

REPO = Path(__file__).resolve().parents[2]
NAME = "static_ungated_followup_v1"
CONDITION = "static_ungated"
CONFIG = "manifests/static_ungated_followup_v1.json"
HOOK_STATES = tuple(sorted(set(STATIC_STATES) | {GATE_READ_STATE}))
RECORD_SCHEMA = "routegate/static_ungated_record/1"


class FollowupError(ValidationError):
    pass


def require(ok, message):
    if not ok:
        raise FollowupError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def jbytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    result = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            require(line.endswith("\n") and bool(line.strip()), "Incomplete or blank JSONL record: " + str(path))
            result.append(json.loads(line))
    return result


def digest_record(value, field="record_sha256"):
    return sha(jbytes({k: v for k, v in value.items() if k != field}))


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def root_for(data_root):
    return Path(data_root).expanduser().resolve() / NAME


@contextlib.contextmanager
def lock(root, name="study", write=True, shared=False):
    if not write:
        yield
        return
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ("." + name + ".lock")).open("a") as handle:
        try:
            fcntl.flock(handle, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError:
            raise FollowupError("Another process is using this study (" + name + ").")
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def load_config(repo=REPO):
    config = read(Path(repo) / CONFIG)
    require(config.get("study_id") == NAME and config.get("record_sha256") == digest_record(config),
            "Follow-up configuration identity changed.")
    require(config["condition"] == CONDITION and config["strength"] == STATIC_ALPHA
            and config["layers_zero_based"] == list(STATIC_BLOCKS)
            and config["gate_used"] is False and config["expected_rows"] == 380,
            "This comparison fixes the original static edit and removes only selection.")
    for name, expected in config["source_sha256"].items():
        path = Path(repo) / name
        require(path.is_file() and sha(path.read_bytes()) == expected,
                "A pinned source differs: " + name)
    return config


def source_context(data_root, repo=REPO):
    """Authorise only the previously released v5 records, without changing guards."""
    config = load_config(repo)
    release = F.load_release(Path(repo))
    require(release["record_sha256"] == config["parent_release_sha256"], "Wrong parent judging release.")
    frame, originals = F.audit_generations(release, Path(data_root), repo=Path(repo))
    baseline, baseline_raw = F.read_adjudicated(Path(data_root), "baseline")
    require(sha(baseline_raw) == config["baseline_adjudication_sha256"],
            "The frozen baseline group assignment changed.")
    groups = F.derive_groups(frame, baseline)
    order = [p["row_id"] for p in frame]
    require(len(frame) == 380 and len(set(order)) == 380, "Expected the complete released 380-prompt frame.")
    return config, release, frame, originals, groups


def actuator_identity(actuator):
    require(actuator.variant == "consensus23_global_frozen", "Wrong static direction.")
    require(tuple(actuator.layers) == STATIC_BLOCKS and tuple(actuator.states) == STATIC_STATES
            and float(actuator.strength) == STATIC_ALPHA, "Frozen actuator settings differ.")
    return {"variant": actuator.variant, "fingerprint": actuator.fingerprint,
            "archive_sha256": actuator.archive_sha256, "file_sha256": actuator.file_sha256,
            "per_layer_sha256": actuator.per_layer_sha256}


def prepare(data_root, repo=REPO, write=False, check_actuator=True):
    config, release, frame, originals, groups = source_context(data_root, repo)
    root = root_for(data_root)
    if check_actuator:
        from .actuator import load_frozen_actuator, PRIMARY_VARIANT
        identity = actuator_identity(load_frozen_actuator(PRIMARY_VARIANT))
    else:
        identity = None  # Test/audit-only option; the CLI never disables this.
    raw_inputs = b"".join((json.dumps({"item_index": i, "item_id": p["row_id"],
                      "prompt": p["prompt_text"], "prompt_sha256": sha(p["prompt_text"].encode())},
                      ensure_ascii=False, sort_keys=True) + "\n").encode() for i, p in enumerate(frame))
    payloads = {"inputs/prompts.jsonl": raw_inputs, "inputs/groups.json": jbytes(groups),
                "inputs/baseline_adjudicated.jsonl": (F.phase_root(Path(data_root), "baseline") /
                        "judging/adjudicated/adjudicated_cells.jsonl").read_bytes()}
    for arm in ("baseline", "safety_first_routegate"):
        payloads["inputs/" + arm + ".json"] = jbytes(originals[arm])
    manifest = {"schema": "routegate/static_ungated_study/1", "study_id": NAME,
                "configuration_sha256": config["record_sha256"], "actuator": identity,
                "input_sha256": {n: sha(b) for n, b in payloads.items()},
                "expected_rows": 380,
                "gate_on_reference_ids": [p["row_id"] for p in frame
                    if originals["safety_first_routegate"][p["row_id"]]["gate_fires"]],
                "prior_external_use_disclosed": True}
    manifest["record_sha256"] = digest_record(manifest)
    payloads["study_manifest.json"] = jbytes(manifest)
    if root.exists() and not (root / "study_manifest.json").exists():
        leftovers = [p for p in root.iterdir() if not p.name.endswith(".lock")]
        require(not leftovers, "Output directory has unclaimed files; choose no overwrite or deletion.")
    for name, blob in payloads.items():
        if (root / name).exists():
            require((root / name).read_bytes() == blob, "Prepared study differs: " + name)
    if write:
        for name, blob in payloads.items():
            F.put_once(root / name, blob)
    return {"stage": "init" if write else "preflight", "prompts": 380,
            "gate_on_consistency_cases": len(manifest["gate_on_reference_ids"]),
            "strength": STATIC_ALPHA, "layers_zero_based": list(STATIC_BLOCKS),
            "max_new_tokens": 768, "actuator_checked": identity is not None,
            "new_generations": 0, "api_calls": 0, "files_written": write}


def study_context(data_root, repo=REPO):
    config = load_config(repo)
    root = root_for(data_root)
    require((root / "study_manifest.json").is_file(), "Initialise with run.sh init first.")
    study = read(root / "study_manifest.json")
    require(study["record_sha256"] == digest_record(study)
            and study["configuration_sha256"] == config["record_sha256"], "Study identity changed.")
    for name, expected in study["input_sha256"].items():
        require(sha((root / name).read_bytes()) == expected, "Study input changed: " + name)
    items = rows(root / "inputs/prompts.jsonl")
    require(len(items) == 380 and len({x["item_id"] for x in items}) == 380,
            "Incomplete prompt inventory.")
    return root, study, items


def make_plans(actuator, capture):
    """Same builder, directions, budgets and arithmetic as finalrunner.execute_item."""
    from .onepassbudget import block_input_state
    from .steering import build_steering_plan
    plans = {}
    for layer in STATIC_BLOCKS:
        state = layer + 1
        reference_state = block_input_state(layer)
        require(reference_state in capture, "Missing unsteered budget state " + str(reference_state))
        plans[state] = build_steering_plan(state=state,
            direction=actuator.direction_for_state(state), strength=STATIC_ALPHA,
            reference_vector=capture[reference_state], budget_reference="block_input")
    return plans


def execute_static(*, model, tokenizer, controller, actuator, item, generation):
    from .runtime import capture_prompt_states, generate_response
    require(controller is not None, "Static generation needs the three intervention hooks.")
    capture_prompt_states(model, tokenizer, item["prompt"], controller,
        capture_name="target", example_id=item["item_id"], config=generation, prompt_mode="chat")
    capture = dict(controller.capture_for("target"))
    require(GATE_READ_STATE in capture, "The diagnostic pass must retain the original hook coverage.")
    plans = make_plans(actuator, capture)
    produced = generate_response(model, tokenizer, item["prompt"], generation,
        controller=controller, mode="patch", plans=plans, example_id=item["item_id"],
        prompt_mode="chat", stop_strings=[])
    require(produced["hooks_fired"] == 3, "The static edit did not fire at all three layers.")
    require(set(map(int, produced["applied_by_state"])) == set(STATIC_STATES),
            "The applied state inventory differs from layers 31–33.")
    return produced


def make_record(item, produced, study, execution_sha256, original):
    require(isinstance(produced.get("response_text"), str) and bool(produced["response_text"].strip()),
            "The model returned no text; this is not a completed response.")
    out = {"schema": RECORD_SCHEMA, "condition": CONDITION, "status": "ok",
           "study_sha256": study["record_sha256"], "execution_sha256": execution_sha256,
           "item_id": item["item_id"], "item_index": item["item_index"],
           "prompt_sha256": item["prompt_sha256"], "response_sha256": sha(produced["response_text"].encode()),
           "created_utc": now(), "strength": STATIC_ALPHA, "layers_zero_based": list(STATIC_BLOCKS),
           "mode": "patch", "gate_used": False, "hooks_expected": 3,
           "max_new_tokens": 768, "original_gate_fires": bool(original["gate_fires"]),
           "same_text_as_original_routegate": produced["response_text"] == original["response_text"],
           **{k: produced[k] for k in ("response_text", "prompt_token_count", "completion_token_count",
               "eos_reached", "hit_max_new_tokens", "stop_fired", "stop_reason", "stop_strings",
               "prompt_mode", "hooks_fired", "applied_by_state")}}
    out["record_sha256"] = digest_record(out)
    validate_record(out, item, study, execution_sha256, original)
    return out


def validate_record(r, item, study, execution_sha256, original):
    require(r.get("record_sha256") == digest_record(r), "Generation record digest differs.")
    require(r.get("schema") == RECORD_SCHEMA and r.get("condition") == CONDITION
            and r.get("status") == "ok" and r.get("study_sha256") == study["record_sha256"]
            and r.get("execution_sha256") == execution_sha256, "Generation provenance differs.")
    require(r.get("item_id") == item["item_id"] and r.get("item_index") == item["item_index"]
            and r.get("prompt_sha256") == item["prompt_sha256"], "Generation belongs to a different prompt.")
    require(isinstance(r.get("response_text"), str) and bool(r["response_text"].strip())
            and r.get("response_sha256") == sha(r["response_text"].encode()), "Missing or changed response.")
    require(r.get("gate_used") is False and r.get("mode") == "patch"
            and r.get("strength") == STATIC_ALPHA and r.get("layers_zero_based") == list(STATIC_BLOCKS)
            and type(r.get("hooks_expected")) is int and r["hooks_expected"] == 3
            and type(r.get("hooks_fired")) is int and r["hooks_fired"] == 3
            and set(map(int, r.get("applied_by_state", {}))) == set(STATIC_STATES),
            "Wrong edit or missing intervention evidence.")
    require(all(type(v) in (int,float) and math.isfinite(v) and v > 0
                for v in r["applied_by_state"].values()), "Non-positive or invalid applied edit norms.")
    require(r.get("prompt_mode") == "chat" and r.get("max_new_tokens") == 768
            and r.get("stop_strings") == [] and r.get("stop_fired") is False,
            "Prompt or generation boundary changed.")
    require(type(r.get("completion_token_count")) is int and 0 < r["completion_token_count"] <= 768
            and type(r.get("prompt_token_count")) is int and r["prompt_token_count"] > 0,
            "Invalid token counts.")
    require(type(r.get("eos_reached")) is bool and type(r.get("hit_max_new_tokens")) is bool
            and r["hit_max_new_tokens"] == (r["completion_token_count"] >= 768), "Invalid stopping flags.")
    require(r.get("stop_reason") == ("eos" if r["eos_reached"] else "max_new_tokens")
            and (r["eos_reached"] or r["hit_max_new_tokens"]), "Invalid completion boundary.")
    require(r.get("original_gate_fires") is bool(original["gate_fires"])
            and r.get("same_text_as_original_routegate") is (r["response_text"] == original["response_text"]),
            "Original comparison metadata differs.")


def load_records(root, study, items):
    path = root / "generations/records.jsonl"
    if not path.exists():
        return {}
    require((root / "execution.json").is_file(), "Generation records have no execution manifest.")
    execution = read(root / "execution.json")
    require(execution["record_sha256"] == digest_record(execution), "Execution manifest changed.")
    originals = read(root / "inputs/safety_first_routegate.json")
    byid = {i["item_id"]: i for i in items}
    records = {}
    for r in rows(path):
        rid = r.get("item_id")
        require(rid in byid and rid not in records, "Duplicate or foreign generation row.")
        validate_record(r, byid[rid], study, execution["record_sha256"], originals[rid])
        records[rid] = r
    return records


def consistency(records, study):
    ids = study["gate_on_reference_ids"]
    done = [i for i in ids if i in records]
    mismatch = [i for i in done if not records[i]["same_text_as_original_routegate"]]
    return {"expected": len(ids), "checked": len(done), "identical": len(done) - len(mismatch),
            "mismatched_item_ids": mismatch, "complete": len(done) == len(ids),
            "passed": len(done) == len(ids) and not mismatch}


def status(data_root, repo=REPO):
    root = root_for(data_root)
    if not (root / "study_manifest.json").is_file():
        return {"stage": "status", "initialised": False, "completed": 0, "expected": 380}
    root, study, items = study_context(data_root, repo)
    records = load_records(root, study, items)
    return {"stage": "status", "initialised": True, "completed": len(records), "expected": 380,
            "remaining": 380-len(records), "complete": len(records) == 380,
            "capped_responses": sum(r["hit_max_new_tokens"] for r in records.values()),
            "gate_on_consistency": consistency(records, study), "api_calls": 0}


def finalise(root, study, records):
    if len(records) != 380:
        return
    check = consistency(records, study)
    require(check["passed"], "Original edit consistency check is incomplete.")
    F.put_once(root / "generation_complete.json", jbytes({"rows": 380,
        "records_sha256": sha((root / "generations/records.jsonl").read_bytes()),
        "study_sha256": study["record_sha256"], "gate_on_consistency": check}))


def generate(data_root, repo=REPO, limit=None):
    require(limit is None or limit > 0, "--limit must be positive.")
    root = root_for(data_root)
    with lock(root):
        prepare(data_root, repo, write=True)
        root, study, items = study_context(data_root, repo)
        records = load_records(root, study, items)
        require(not consistency(records, study)["mismatched_item_ids"],
                "An originally edited response differs. Inspect consistency before generating more; no result is overwritten.")
        if len(records) == 380:
            finalise(root, study, records)
            return status(data_root, repo)
        from .actuator import PRIMARY_VARIANT, load_frozen_actuator
        from .runtime import GenerationConfig, load_model_and_tokenizer
        from .hooks import PatchController
        from .livecontroller import assert_no_parameter_requires_grad
        import torch
        require(torch.cuda.is_available() and torch.cuda.device_count() == 4,
                "Use the same four-GPU allocation as the original final evaluation.")
        actuator = load_frozen_actuator(PRIMARY_VARIANT)
        require(actuator_identity(actuator) == study["actuator"], "Actuator bytes changed.")
        generation = GenerationConfig(max_new_tokens=768)
        generation.validate()
        model, tokenizer = load_model_and_tokenizer(generation)
        model.eval()
        assert_no_parameter_requires_grad([model])
        execution = {"schema": "routegate/static_ungated_execution/1", "study_sha256": study["record_sha256"],
                     "python": platform.python_version(), "torch": torch.__version__,
                     "cuda": torch.version.cuda, "gpu_names": [torch.cuda.get_device_name(i) for i in range(4)],
                     "generation": generation.as_dict(), "device_map": {k: str(v) for k,v in getattr(model,"hf_device_map",{}).items()}}
        execution["record_sha256"] = digest_record(execution)
        F.put_once(root / "execution.json", jbytes(execution))
        controller = PatchController(model, list(HOOK_STATES), budget_reference="block_input")
        original = read(root / "inputs/safety_first_routegate.json")
        known_on = set(study["gate_on_reference_ids"])
        # Technical identity cases first, then all others in original input order.
        ordered = sorted(items, key=lambda x: (x["item_id"] not in known_on, x["item_index"]))
        written = 0
        path = root / "generations/records.jsonl"
        path.parent.mkdir(exist_ok=True)
        try:
            for item in ordered:
                rid = item["item_id"]
                if rid in records:
                    continue
                produced = execute_static(model=model, tokenizer=tokenizer, controller=controller,
                    actuator=actuator, item=item, generation=generation)
                record = make_record(item, produced, study, execution["record_sha256"], original[rid])
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush(); os.fsync(handle.fileno())
                records[rid] = record; written += 1
                print("[static] {}/380 complete | {}".format(len(records), rid), flush=True)
                check = consistency(records, study)
                if rid in known_on:
                    F.put_once(root / "consistency" / (rid.replace(":", "_") + ".json"),
                               jbytes({"item_id": rid, "identical": record["same_text_as_original_routegate"]}))
                    require(not check["mismatched_item_ids"],
                            "An originally edited response differs. The new record is preserved; generation stopped before further cases. Inspect the runtime before interpreting a gate-only effect.")
                if limit is not None and written >= limit:
                    break
        finally:
            controller.close()
        current = status(data_root, repo)
        if current["complete"]:
            finalise(root, study, records)
        return current
