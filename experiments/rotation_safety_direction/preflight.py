#!/usr/bin/env python3
"""Everything the rotation screen can be refused for, without a GPU.

Offline. Loads no model, generates nothing, judges nothing, calls no service
and writes nothing unless asked. It runs the same gates the runner runs before
its model load, plus the ones only a preflight is in a position to run: the
shard partition's determinism and disjointness, the operator's mathematical
identities on synthetic vectors, and the sealed-set reachability sweep.

A preflight that passes does not mean the screen will produce anything
interesting. It means the screen that runs is the screen that was declared.

Usage
-----
    python experiments/rotation_safety_direction/preflight.py \\
        --direction-bank <bank>
    python experiments/rotation_safety_direction/preflight.py \\
        --direction-bank <bank> --write-report <path.json>
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate import __version__ as PACKAGE_VERSION                # noqa: E402
from routegate.data import load_input_manifest, load_local_paths     # noqa: E402
from routegate.directionbank import BANK_STATES, HISTORICAL_LAYERS  # noqa: E402
from routegate.dualsteer import (                                   # noqa: E402
    BUDGET_REFERENCE,
    load_validated_bank,
)
from routegate.phase2 import verify_phase2_selection                # noqa: E402
from routegate.phase2closure import assert_protocol_is_open         # noqa: E402
from routegate.rotation import (                                    # noqa: E402
    ROTATION_TOLERANCES,
    TOLERANCES_VERSION,
    RotationError,
    assert_rotation_invariants,
    rotate_toward,
)
from routegate.rotationfreeze import (                              # noqa: E402
    PREDECLARATION,
    SAFETY_DIRECTION_SHA256,
    verify_predeclaration,
    verify_preserved_artifacts,
)
from routegate.rotationscreen import (                              # noqa: E402
    ANALYSIS_TABLE_RECORDS,
    ARM_BY_ID_ROTATION,
    BETA_GRID,
    DEFAULT_CONFIG,
    INTERVENTION_ORDER,
    NEW_GENERATIONS,
    N_ROWS,
    PROTOCOL,
    REFERENCE_ARM_IDS,
    REFERENCE_RECORDS,
    SELECTED_ROWS,
    RotationScreenError,
    assert_config_matches,
    assert_inventory,
    assert_safety_direction_is_h_minus_e,
    build_inventory,
    safety_direction_digests,
    selection_digest_holds,
    shard,
)
from routegate.runrecord import environment_record, utc_now, write_json  # noqa: E402
from routegate.runtime import GenerationConfig                      # noqa: E402
from routegate.validation import (                                  # noqa: E402
    Checklist,
    ValidationError,
    assert_config_is_sealed_free,
    looks_sealed,
    sha256_text,
)

RUNNER = REPO_ROOT / "experiments" / "rotation_safety_direction"


def synthetic_rows() -> List[Dict[str, str]]:
    """The 18 bound rows, as inventory inputs, without opening the corpus.

    The preflight's job here is the PARTITION, not the prompts: whether the
    inventory is complete and whether the shards divide it exactly. The corpus
    is hash-gated separately below, and reading its prompt text would put
    prompt bytes into a report that says it holds none.
    """
    return [{"row_id": row_id, "row_class": "H",
             "prompt_sha256": sha256_text(f"preflight placeholder {row_id}")}
            for row_id in SELECTED_ROWS]


def check_shards(inventory: Sequence[Dict[str, Any]], checks: Checklist,
                 counts: Sequence[int] = (1, 2, 3, 4, 6, 8)) -> Dict[str, Any]:
    """The shards partition the inventory: exact cover, no overlap, stable."""
    results: Dict[str, Any] = {}
    for count in counts:
        pieces = [shard(inventory, index, count)
                  for index in range(1, count + 1)]
        again = [shard(inventory, index, count)
                 for index in range(1, count + 1)]
        keys = [[(item["row_id"], item["arm_id"]) for item in piece]
                for piece in pieces]
        flat = [key for piece in keys for key in piece]
        expected = [(item["row_id"], item["arm_id"]) for item in inventory]
        if pieces != again:
            raise RotationScreenError(
                f"the {count}-way shard split is not deterministic")
        if flat != expected:
            raise RotationScreenError(
                f"the {count}-way shard split is not an exact, ordered cover "
                f"of the inventory: {len(flat)} item(s) against "
                f"{len(expected)}")
        if len(set(flat)) != len(flat):
            raise RotationScreenError(
                f"the {count}-way shard split overlaps: "
                f"{len(flat) - len(set(flat))} duplicated cell(s)")
        results[str(count)] = [len(piece) for piece in pieces]
    checks.record("shards_partition_the_inventory", True,
                  f"{sorted(results)} way(s): deterministic, disjoint, and an "
                  f"exact cover of {len(inventory)} cell(s)")
    return results


def check_operator(checks: Checklist) -> Dict[str, Any]:
    """The operator's identities, on synthetic vectors, at this precision."""
    import torch

    generator = torch.Generator().manual_seed(20260830)
    worst = {"norm": 0.0, "chord": 0.0, "angle": 0.0, "cast_norm": 0.0}
    trials = 0
    for trial in range(8):
        scale = 10.0 ** (trial - 3)
        h = torch.randn(2048, generator=generator, dtype=torch.float32) * scale
        direction = torch.randn(2048, generator=generator, dtype=torch.float32)
        direction = direction / torch.linalg.vector_norm(direction)
        for beta in BETA_GRID:
            outcome = rotate_toward(h, direction, beta)
            assert_rotation_invariants(outcome.diagnostics, ROTATION_TOLERANCES,
                                       label=f"preflight scale {scale}")
            diagnostics = outcome.diagnostics
            worst["norm"] = max(
                worst["norm"],
                float(diagnostics["relative_norm_error_float32"]))
            worst["cast_norm"] = max(
                worst["cast_norm"],
                float(diagnostics["relative_norm_error_after_cast"]))
            worst["chord"] = max(worst["chord"],
                                 float(diagnostics["chord_absolute_error"]))
            worst["angle"] = max(
                worst["angle"],
                abs(float(diagnostics["remaining_angle"])
                    - float(diagnostics["expected_remaining_angle"])))
            trials += 1

    # beta = 0 is the exact identity, and beta = 1 lands on the direction.
    h = torch.randn(2048, generator=generator, dtype=torch.float32) * 3.0
    direction = torch.randn(2048, generator=generator, dtype=torch.float32)
    direction = direction / torch.linalg.vector_norm(direction)
    identity = rotate_toward(h, direction, 0.0)
    if not bool(torch.equal(identity.tensor, h)):
        raise RotationScreenError("beta = 0 is not the bitwise identity")
    landed = rotate_toward(h, direction, 1.0)
    if abs(float(landed.diagnostics["cosine_after_float32"]) - 1.0) > 1e-5:
        raise RotationScreenError(
            f"beta = 1 left cosine "
            f"{landed.diagnostics['cosine_after_float32']}, not 1")

    # The four fail-closed inputs are refused rather than approximated.
    refusals: Dict[str, str] = {}
    zero = torch.zeros(2048, dtype=torch.float32)
    for name, args in (
            ("zero_activation", (zero, direction, 0.15)),
            ("zero_direction", (h, zero, 0.15)),
            ("nan_activation",
             (h.clone().index_fill_(0, torch.tensor([7]), float("nan")),
              direction, 0.15)),
            ("inf_direction",
             (h, direction.clone().index_fill_(0, torch.tensor([11]),
                                               float("inf")), 0.15)),
            ("antiparallel", (direction * -2.0, direction, 0.15))):
        try:
            rotate_toward(*args)
        except RotationError as exc:
            refusals[name] = str(exc).split(".")[0]
        else:
            raise RotationScreenError(
                f"{name} was rotated instead of refused")

    checks.record(
        "rotation_operator_identities", True,
        f"{trials} (scale, beta) case(s) over 8 magnitude decades: worst "
        f"relative float32 norm error {worst['norm']:.3e} (tol "
        f"{ROTATION_TOLERANCES.norm_preservation_rtol_float32:.3e}), worst "
        f"chord error {worst['chord']:.3e}, worst remaining-angle error "
        f"{worst['angle']:.3e}")
    checks.record("rotation_operator_fails_closed", True,
                  f"{len(refusals)} degenerate input(s) refused: "
                  f"{sorted(refusals)}")
    return {"worst": worst, "trials": trials, "refusals": sorted(refusals)}


#: The two literals this preflight deliberately feeds to the guards, to prove
#: the refusal is installed rather than assumed. They are the only path-like
#: strings in this stage that may name held-out material, and they are never
#: opened: each is handed to a gate whose job is to reject it.
GUARD_NEGATIVE_TEST_LITERALS: Tuple[str, ...] = (
    "/tmp/sealed_benchmark_out",
    "/tmp/final_benchmark.csv",
)

#: Committed, prompt-free repository files whose NAMES match the guard's
#: pattern because naming the commitment is what they are for. Read as a
#: digest and never opened for content by this stage.
PERMITTED_PATH_LITERALS: Tuple[str, ...] = (
    "manifests/sealed_set_commitment.json",
)

#: Suffixes that make a bare string a data path even without a separator.
PATH_SUFFIXES: Tuple[str, ...] = (
    ".csv", ".json", ".jsonl", ".npy", ".npz", ".pt", ".txt", ".md")


def path_like_constants(path: Path) -> List[str]:
    """Every string constant in a module that could be a path.

    Prose is not swept, and deliberately so. The guard's own documentation has
    to be able to say what it refuses, and a sweep that failed on the word
    would force this stage to describe its protection in euphemism. What
    matters is that no *path* names held-out material, so the sweep parses the
    module and looks at string constants that carry a separator or a data
    suffix -- which is what a path is and what prose is not.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    # Docstrings are prose that happens to be a string constant, and a usage
    # example in one is documentation rather than a path the code can open.
    docstrings = {id(node.body[0].value)
                  for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.ClassDef,
                                       ast.FunctionDef, ast.AsyncFunctionDef))
                  and node.body
                  and isinstance(node.body[0], ast.Expr)
                  and isinstance(node.body[0].value, ast.Constant)
                  and isinstance(node.body[0].value.value, str)}
    found: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in docstrings:
            text = node.value
            if "/" in text or text.endswith(PATH_SUFFIXES):
                found.append(text)
    return found


def check_no_sealed_path(checks: Checklist, config: Dict[str, Any],
                         paths: Dict[str, Any]) -> Dict[str, Any]:
    """No path this screen can reach names held-out material.

    Four sweeps, because they fail differently. The configuration and the
    local path file could carry a held-out *value*. This stage's own source
    could carry a held-out *literal*, which would bypass a guard that only
    ever sees a variable. And the gates the runner relies on are exercised on
    a name that must be refused, so "it is guarded" is a checked statement
    rather than an assurance: a guard someone removed passes the first three
    sweeps and fails the fourth.
    """
    assert_config_is_sealed_free(config, "rotation screen configuration")
    offending = {key: value for key, value in paths.items()
                 if isinstance(value, str) and looks_sealed(value)}
    if offending:
        raise RotationScreenError(
            f"the local path file names held-out material at "
            f"{sorted(offending)}")

    sources = sorted(RUNNER.glob("*.py")) + [
        REPO_ROOT / "src" / "routegate" / "rotation.py",
        REPO_ROOT / "src" / "routegate" / "rotationscreen.py",
        REPO_ROOT / "src" / "routegate" / "rotationfreeze.py",
    ]
    hits: List[str] = []
    constants = 0
    for path in sources:
        for text in path_like_constants(path):
            constants += 1
            if not looks_sealed(text):
                continue
            if text in GUARD_NEGATIVE_TEST_LITERALS \
                    or text in PERMITTED_PATH_LITERALS:
                continue
            hits.append(f"{path.name}: {text!r}")
    if hits:
        raise RotationScreenError(
            f"{len(hits)} path literal(s) in this stage name held-out "
            f"material: {hits[:5]}")

    # The gates the runner actually applies, exercised on names that must be
    # refused. Neither path is opened; each is handed to a gate whose job is
    # to reject it, and a gate that accepts one is a gate that is not there.
    from routegate.rotationscreen import claim_output_dir
    from routegate.validation import SealedSetError, assert_not_sealed

    exercised: List[str] = []
    for label, call in (
            ("output_directory",
             lambda: claim_output_dir(Path(GUARD_NEGATIVE_TEST_LITERALS[0]),
                                      create=False)),
            ("dataset_path",
             lambda: assert_not_sealed(Path(GUARD_NEGATIVE_TEST_LITERALS[1]),
                                       "rotation screen dataset"))):
        try:
            call()
        except SealedSetError:
            exercised.append(label)
            continue
        raise RotationScreenError(
            f"the {label} gate accepted a held-out name; the refusal this "
            f"screen relies on is not installed")

    checks.record(
        "no_sealed_path_is_reachable", True,
        f"{len(sources)} source file(s) ({constants} path-like constant(s)), "
        f"the configuration and {len(paths)} local path(s) sweep clean; the "
        f"{' and '.join(exercised)} gate(s) refuse a held-out name when given "
        f"one")
    return {"sources_swept": len(sources),
            "path_constants_swept": constants,
            "paths_swept": len(paths),
            "gates_exercised": exercised}


def preflight(bank_dir: Path, config_path: Path, predeclaration: Path,
              paths_file: Optional[Path], report: Optional[Path]) -> int:
    checks = Checklist("rotation_screen_preflight")

    assert_protocol_is_open(PROTOCOL, checks=checks)
    preserved = verify_preserved_artifacts(checks=checks)
    frozen = verify_predeclaration(predeclaration, checks=checks)
    verify_phase2_selection(checks=checks)
    selection_digest_holds()

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    assert_config_matches(config, checks=checks)

    generation = GenerationConfig()
    generation.validate()
    checks.record("generation_config", True,
                  f"{generation.model} @ {generation.revision}, greedy, "
                  f"max_new_tokens {generation.max_new_tokens}, fingerprint "
                  f"{generation.fingerprint()[:16]}")

    input_manifest = load_input_manifest()
    local_paths = (load_local_paths(paths_file) if paths_file
                   else load_local_paths())
    dataset_sha = input_manifest["dataset"]["sha256"]

    bank = load_validated_bank(
        bank_dir, config["direction_bank"]["artifact_sha256"], checks)
    direction_metadata = assert_safety_direction_is_h_minus_e(bank, checks)
    digests = safety_direction_digests(bank)
    if digests != SAFETY_DIRECTION_SHA256:
        raise RotationScreenError(
            f"the per-layer H - E digests are {digests} and the package pins "
            f"{SAFETY_DIRECTION_SHA256}")
    checks.record("per_layer_h_minus_e_digests", True,
                  ", ".join(f"state {state} {digest[:12]}"
                            for state, digest in sorted(digests.items())))
    if len(set(digests.values())) != len(BANK_STATES):
        raise RotationScreenError(
            "two states carry the same H - E vector; each layer has its own")

    inventory = build_inventory(synthetic_rows())
    counts = assert_inventory(inventory, checks)
    shards = check_shards(inventory, checks)
    operator = check_operator(checks)
    sealed = check_no_sealed_path(checks, config, local_paths)

    summary = {
        "schema": "routegate/rotation_safety_direction_preflight/1",
        "created_utc": utc_now(),
        "package_version": PACKAGE_VERSION,
        "protocol": PROTOCOL,
        "is_a_candidate_not_a_result": True,
        "generation_performed": False,
        "model_loaded": False,
        "predeclaration": predeclaration.name,
        "predeclaration_sha256": frozen["sha256"],
        "frozen_input_sha256": {
            "dataset": dataset_sha,
            "direction_bank": bank.artifact_sha256,
            "direction_bank_manifest": bank.manifest_sha256,
            "safety_direction_by_state": digests,
            "repair_direction_by_state": {
                str(state): bank.direction_sha256("repair_direction", state)
                for state in bank.states},
            "generation_config": generation.fingerprint(),
            "selected_rows": selection_digest_holds(),
        },
        "direction_metadata": direction_metadata,
        "intervention": {
            "order": INTERVENTION_ORDER,
            "states": list(BANK_STATES),
            "historical_layers": list(HISTORICAL_LAYERS),
            "budget_reference": BUDGET_REFERENCE,
            "token_position": "final prompt token",
            "prefill_only": True,
        },
        "beta_grid": list(BETA_GRID),
        "tolerances_version": TOLERANCES_VERSION,
        "tolerances": ROTATION_TOLERANCES.as_dict(),
        "counts": {**counts,
                   "planned_generations": NEW_GENERATIONS,
                   "reference_records": REFERENCE_RECORDS,
                   "reference_arms": list(REFERENCE_ARM_IDS),
                   "analysis_table_records": ANALYSIS_TABLE_RECORDS},
        "shard_partitions": shards,
        "operator": operator,
        "sealed_set": {"accessible": False, **sealed},
        "preserved_artifacts": preserved,
        "environment": environment_record(PACKAGE_VERSION),
        "checklist": checks.as_dict(),
        "contains_prompt_or_response_text": False,
    }

    print(f"[preflight] {PROTOCOL} -- a CANDIDATE screen; not run, not judged, "
          f"not validated")
    print(f"[preflight] predeclaration {frozen['sha256']}")
    print(f"[preflight] frozen inputs:")
    for name, value in sorted(summary["frozen_input_sha256"].items()):
        if isinstance(value, dict):
            for state, digest in sorted(value.items()):
                print(f"[preflight]   {name}[{state}] {digest}")
        else:
            print(f"[preflight]   {name} {value}")
    print(f"[preflight] d_S = {direction_metadata['subtraction_order']}, "
          f"orthogonalisation {direction_metadata['orthogonalisation_applied']}")
    print(f"[preflight] plan {NEW_GENERATIONS} generation(s) = {N_ROWS} rows x "
          f"{len(ARM_BY_ID_ROTATION)} arms; {REFERENCE_RECORDS} inherited; "
          f"table {ANALYSIS_TABLE_RECORDS}")
    print(f"[preflight] operator worst float32 norm error "
          f"{operator['worst']['norm']:.3e}, chord {operator['worst']['chord']:.3e}, "
          f"angle {operator['worst']['angle']:.3e}")
    print(f"[preflight] {checks.as_dict()['checks_run']} check(s), all passed")
    for entry in checks.entries:
        print(f"[preflight]   [{entry['status']}] {entry['check']}: "
              f"{entry['detail']}")

    if report is not None:
        write_json(report, summary)
        print(f"[preflight] wrote {report}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction-bank", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--predeclaration", type=Path, default=PREDECLARATION)
    parser.add_argument("--paths", type=Path, default=None)
    parser.add_argument("--write-report", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return preflight(args.direction_bank, args.config,
                         args.predeclaration, args.paths, args.write_report)
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
