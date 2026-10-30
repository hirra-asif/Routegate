"""The migrated frozen RQ2 actuator: that it arrived intact, and loads safely.

Nothing here loads a model, allocates a GPU, opens a socket or reads a prompt.
The only real file these tests touch is the migrated freeze directory itself,
and every corruption case is built in ``tmp_path`` from a copy of it, so the
migrated artefact is never written to.

The tests are organised by what they establish:

    1. migration integrity   the copies are the frozen bytes
    2. safe loading          no pickle, no torch.load, no surprises
    3. identity              fingerprints, digests, norms, shape, dtype
    4. layer mapping         31/32/33 -> states 32/33/34, exactly
    5. arithmetic            the target-relative budget, on synthetic tensors
    6. fail-closed           every gate refuses rather than adapts
    7. isolation             no prompt text, no model, no network, no sealed set
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate import actuator as actuator_module              # noqa: E402
from routegate.actuator import (                               # noqa: E402
    ACTUATOR_STATES,
    APPLICATION_PATH_ANALYTICAL,
    APPLICATION_PATH_CANONICAL,
    APPLICATION_PATHS,
    BYTES_PER_STORAGE,
    COEFFICIENT_MODE_CONTROLLER,
    COEFFICIENT_MODE_HISTORICAL,
    COEFFICIENT_MODES,
    COMPARATOR_VARIANT,
    FROZEN_STRENGTH,
    HIDDEN_SIZE,
    HISTORICAL_LAYERS,
    N_LAYERS,
    PATHS_KEY,
    PRIMARY_VARIANT,
    ActuatorError,
    FrozenActuator,
    actuator_dir,
    direction_fingerprint,
    load_actuator_manifest,
    load_frozen_actuator,
    verify_actuator_files,
)
from routegate.intervention import (                           # noqa: E402
    historical_layer_to_state,
    state_to_site,
)
from routegate.validation import ValidationError, sha256_file  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "manifests" / "rq2_actuator.json"
PATHS_PATH = REPO_ROOT / "configs" / "paths.local.json"
SOURCE_MODULE = REPO_ROOT / "src" / "routegate" / "actuator.py"

#: The historical freeze directory, if this machine carries it. Supplied by
#: environment variable rather than hard-coded: an absolute path to one home
#: directory on one cluster does not belong in a committed test, and the
#: migrated copy carries the origin's own checksum file anyway.
ORIGIN_ENV = "ROUTEGATE_RQ2_FREEZE_ORIGIN"

HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")


def _actuator_available() -> bool:
    if not PATHS_PATH.exists():
        return False
    try:
        paths = json.loads(PATHS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return False
    directory = paths.get(PATHS_KEY)
    return bool(directory) and Path(directory).is_dir()


needs_actuator = pytest.mark.skipif(
    not _actuator_available(),
    reason=f"the migrated RQ2 actuator freeze is not reachable "
           f"(configs/paths.local.json key {PATHS_KEY!r})")

needs_origin = pytest.mark.skipif(
    not os.environ.get(ORIGIN_ENV),
    reason=f"the historical freeze run is not identified; set {ORIGIN_ENV}")


@pytest.fixture(scope="module")
def manifest():
    return load_actuator_manifest(MANIFEST_PATH)


@pytest.fixture(scope="module")
def freeze_dir():
    return actuator_dir(paths_path=PATHS_PATH)


@pytest.fixture(scope="module")
def primary(freeze_dir):
    return load_frozen_actuator(PRIMARY_VARIANT, directory=freeze_dir)


@pytest.fixture(scope="module")
def comparator(freeze_dir):
    return load_frozen_actuator(COMPARATOR_VARIANT, directory=freeze_dir)


@pytest.fixture
def sandbox(tmp_path, freeze_dir):
    """A writable copy of the freeze, for the corruption cases."""
    target = tmp_path / "rq2_actuator"
    shutil.copytree(freeze_dir, target)
    for path in target.iterdir():
        path.chmod(0o644)
    return target


def _manifest_for(directory: Path, base: dict) -> dict:
    """``base`` with the archive's recorded size and digest refreshed.

    A test that alters the archive has to get past the file-hash gate before it
    can exercise an archive gate. Refreshing only those two fields is the
    minimum that allows it, and it means the archive test is testing the archive
    checks rather than re-testing the hash check.
    """
    updated = json.loads(json.dumps(base))
    name = updated["archive"]["filename"]
    entry = updated["files"]["covered_by_historical_checksum"][name]
    entry["sha256"] = sha256_file(directory / name)
    entry["bytes"] = (directory / name).stat().st_size
    sums = directory / "FREEZE_SHA256SUMS.txt"
    lines = []
    for raw in sums.read_text(encoding="utf-8").splitlines():
        if raw.strip().endswith(name):
            lines.append(f"{entry['sha256']}  {raw.split(None, 1)[1]}")
        else:
            lines.append(raw)
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    checksum_entry = updated["files"]["the_checksum_file_itself"][
        "FREEZE_SHA256SUMS.txt"]
    checksum_entry["sha256"] = sha256_file(sums)
    checksum_entry["bytes"] = sums.stat().st_size
    return updated


def _rewrite_archive(directory: Path, mutate) -> None:
    """Rebuild ``frozen_directions.pt`` with ``mutate(members) -> members``."""
    path = directory / "frozen_directions.pt"
    with zipfile.ZipFile(path) as archive:
        members = {info.filename: archive.read(info.filename)
                   for info in archive.infolist()}
    members = mutate(members)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


# =============================================================== 1. migration
@needs_actuator
def test_the_migrated_directory_holds_exactly_the_freeze_files(freeze_dir, manifest):
    expected = set(manifest["files"]["covered_by_historical_checksum"])
    expected |= set(manifest["files"]["the_checksum_file_itself"])
    present = {p.name for p in freeze_dir.iterdir() if p.is_file()}
    assert present == expected
    assert len(expected) == 10


@needs_actuator
def test_every_migrated_file_matches_the_manifest(freeze_dir, manifest):
    observed, checks = verify_actuator_files(freeze_dir, manifest)
    assert checks.all_passed
    covered = manifest["files"]["covered_by_historical_checksum"]
    for name, spec in covered.items():
        assert observed[name] == spec["sha256"]
        assert (freeze_dir / name).stat().st_size == spec["bytes"]


@needs_actuator
def test_the_historical_checksum_file_independently_confirms_nine_of_them(freeze_dir):
    """The origin's own record, carried along with the copies.

    This is what makes "source and copied hashes match" provable from the
    migrated directory alone, on a machine that no longer has the origin.
    """
    sums = {}
    for line in (freeze_dir / "FREEZE_SHA256SUMS.txt").read_text(
            encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, path = line.split(None, 1)
        sums[Path(path).name] = digest
    assert len(sums) == 9
    assert "FREEZE_SHA256SUMS.txt" not in sums   # it does not list itself
    for name, digest in sums.items():
        assert sha256_file(freeze_dir / name) == digest, name


@needs_actuator
@needs_origin
def test_the_migrated_copies_are_byte_identical_to_the_origin(freeze_dir, manifest):
    origin = Path(os.environ[ORIGIN_ENV])
    expected = set(manifest["files"]["covered_by_historical_checksum"])
    expected |= set(manifest["files"]["the_checksum_file_itself"])
    for name in sorted(expected):
        source, copied = origin / name, freeze_dir / name
        assert source.is_file(), f"{name} is absent from the origin"
        assert source.stat().st_size == copied.stat().st_size, name
        assert sha256_file(source) == sha256_file(copied), name


# ============================================================ 2. safe loading
def test_the_loader_never_calls_pickle_or_torch_load():
    """A static read of the module's *code*, before any behavioural check.

    Deliberately AST-based rather than a text search: the module's docstring
    explains at length why ``torch.load`` is not used, and a substring test
    would be satisfied by deleting that explanation.
    """
    tree = ast.parse(SOURCE_MODULE.read_text(encoding="utf-8"))

    def imports_of(nodes):
        found = set()
        for node in nodes:
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module.split(".")[0])
        return found

    top_level = imports_of(tree.body)
    everywhere = imports_of(ast.walk(tree))
    assert "pickle" not in everywhere, "the loader must not import pickle at all"
    assert "torch" not in top_level, (
        "torch must stay a lazy, function-local import: reading and checking "
        "the actuator needs numpy only")
    assert "torch" in everywhere          # as_torch() imports it where it is used
    # pickletools is the decoder that *is* allowed, and it must be present.
    assert "pickletools" in top_level

    attributes = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            attributes.add(f"{node.value.id}.{node.attr}")
    forbidden = {"pickle.load", "pickle.loads", "pickle.Unpickler",
                 "torch.load", "torch.jit", "np.load"}
    assert not (attributes & forbidden), sorted(attributes & forbidden)
    assert "pickletools.genops" in attributes


@needs_actuator
def test_loading_succeeds_with_pickle_and_torch_load_disabled(freeze_dir,
                                                              monkeypatch):
    """The behavioural half: break the unsafe paths, then load anyway."""
    import pickle

    def refuse(*args, **kwargs):
        raise AssertionError("the frozen actuator loader used pickle")

    monkeypatch.setattr(pickle, "load", refuse)
    monkeypatch.setattr(pickle, "loads", refuse)
    monkeypatch.setattr(pickle, "Unpickler", refuse)
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None:
        monkeypatch.setattr(torch, "load", refuse)
        monkeypatch.setattr(torch.serialization, "load", refuse, raising=False)

    loaded = load_frozen_actuator(PRIMARY_VARIANT, directory=freeze_dir)
    assert loaded.directions.shape == (N_LAYERS, HIDDEN_SIZE)
    assert loaded.fingerprint


@needs_actuator
def test_the_archive_is_never_extracted_to_disk(freeze_dir, tmp_path,
                                                monkeypatch):
    """No member is written out, so no member can be executed."""
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.iterdir())
    load_frozen_actuator(PRIMARY_VARIANT, directory=freeze_dir)
    assert set(tmp_path.iterdir()) == before


# ================================================================ 3. identity
@needs_actuator
@pytest.mark.parametrize("variant", [PRIMARY_VARIANT, COMPARATOR_VARIANT])
def test_the_loaded_tensor_reproduces_its_recorded_fingerprint(freeze_dir,
                                                               manifest, variant):
    loaded = load_frozen_actuator(variant, directory=freeze_dir)
    recorded = manifest["variants"][variant]["fingerprint"]
    assert loaded.fingerprint == recorded
    assert direction_fingerprint(loaded.directions, loaded.layers) == recorded

    frozen = json.loads((freeze_dir / "frozen_direction_manifest.json")
                        .read_text(encoding="utf-8"))
    assert frozen["variants"][variant]["fingerprint"] == recorded


@needs_actuator
@pytest.mark.parametrize("variant", [PRIMARY_VARIANT, COMPARATOR_VARIANT])
def test_the_loaded_tensor_reproduces_its_recorded_per_layer_digests(
        freeze_dir, manifest, variant):
    loaded = load_frozen_actuator(variant, directory=freeze_dir)
    recorded = manifest["variants"][variant]["per_layer_sha256"]
    for position, layer in enumerate(loaded.layers):
        digest = hashlib.sha256(
            np.ascontiguousarray(loaded.directions[position]).tobytes()).hexdigest()
        assert digest == recorded[str(layer)], layer
        assert loaded.per_layer_sha256[str(layer)] == recorded[str(layer)]


@needs_actuator
@pytest.mark.parametrize("variant", [PRIMARY_VARIANT, COMPARATOR_VARIANT])
def test_every_layer_is_unit_norm_and_matches_the_recorded_norm(
        freeze_dir, manifest, variant):
    loaded = load_frozen_actuator(variant, directory=freeze_dir)
    tolerance = manifest["tensors"]["unit_norm_tolerance"]
    recorded = manifest["variants"][variant]["per_layer_norm"]
    for position, layer in enumerate(loaded.layers):
        # float64, as the freeze's own verify_unit_norm computed it
        norm = float(np.linalg.norm(
            np.asarray(loaded.directions[position], dtype=np.float64), ord=2))
        assert abs(norm - 1.0) <= tolerance, layer
        assert norm == pytest.approx(recorded[position], abs=1e-9), layer
        assert loaded.per_layer_norm[position] == pytest.approx(norm, abs=0.0)


@needs_actuator
@pytest.mark.parametrize("variant", [PRIMARY_VARIANT, COMPARATOR_VARIANT])
def test_the_directions_are_three_by_hidden_size_float32(freeze_dir, variant):
    loaded = load_frozen_actuator(variant, directory=freeze_dir)
    assert loaded.directions.shape == (3, 2048) == (N_LAYERS, HIDDEN_SIZE)
    assert loaded.directions.dtype == np.float32
    assert bool(np.isfinite(loaded.directions).all())
    assert BYTES_PER_STORAGE == 3 * 2048 * 4


@needs_actuator
def test_the_loaded_array_is_read_only(primary):
    assert primary.directions.flags.writeable is False
    with pytest.raises(ValueError):
        primary.directions[0, 0] = 1.0


@needs_actuator
def test_the_two_variants_are_different_directions(primary, comparator):
    assert primary.variant != comparator.variant
    assert primary.role == "primary"
    assert comparator.role == "comparator"
    assert primary.fingerprint != comparator.fingerprint
    assert not np.array_equal(primary.directions, comparator.directions)


@needs_actuator
def test_variants_are_bound_to_storages_by_content(primary, comparator, manifest):
    """The archive's entry order is a cross-check, never the source of truth."""
    entries = {e["check"]: e["detail"] for e in primary.checklist.entries}
    binding = entries["storages_bound_to_variants_by_content"]
    assert f"{PRIMARY_VARIANT} <- " in binding
    assert f"{COMPARATOR_VARIANT} <- " in binding
    for variant, loaded in ((PRIMARY_VARIANT, primary),
                            (COMPARATOR_VARIANT, comparator)):
        member = manifest["variants"][variant]["expected_archive_member"]
        assert f"{variant} <- {member}" in binding


@needs_actuator
def test_the_frozen_strength_is_zero_point_one_five_not_zero_point_two(primary):
    """0.2 is the population scan's strength. It is not this actuator's."""
    assert primary.strength == FROZEN_STRENGTH == 0.15
    assert primary.strength != 0.2


@needs_actuator
def test_the_actuator_carries_its_provenance(primary, manifest):
    record = primary.provenance()
    assert record["variant"] == PRIMARY_VARIANT
    assert record["fingerprint"] == manifest["variants"][PRIMARY_VARIANT]["fingerprint"]
    assert record["historical_layers"] == [31, 32, 33]
    assert record["routegate_states"] == [32, 33, 34]
    assert record["strength"] == 0.15
    assert record["budget_reference"] == "block_input"
    assert record["site"] == "residual_stream"
    assert record["site_kind"] == "block_output"
    assert record["window"] == 1 and record["slot"] == 0
    assert record["n_source_cases"] == 10
    assert primary.archive_sha256 == (
        manifest["files"]["covered_by_historical_checksum"]
        ["frozen_directions.pt"]["sha256"])


# =========================================================== 4. layer mapping
def test_the_historical_layers_are_thirty_one_to_thirty_three():
    assert HISTORICAL_LAYERS == (31, 32, 33)


def test_the_states_are_thirty_two_to_thirty_four():
    assert ACTUATOR_STATES == (32, 33, 34)


def test_the_state_mapping_is_the_one_intervention_defines():
    """Derived, never written out twice, so the two cannot drift apart."""
    assert ACTUATOR_STATES == tuple(
        historical_layer_to_state(layer) for layer in HISTORICAL_LAYERS)
    for layer, state in zip(HISTORICAL_LAYERS, ACTUATOR_STATES):
        assert state == layer + 1
        site = state_to_site(state, 48)
        assert site.block_index == layer
        assert site.kind == "block_output"
        assert site.tensor == "output"


@needs_actuator
def test_layer_and_state_lookups_return_the_same_row(primary):
    for layer, state in zip(primary.layers, primary.states):
        assert np.array_equal(primary.direction_for_layer(layer),
                              primary.direction_for_state(state))
    assert primary.state_of_layer == {31: 32, 32: 33, 33: 34}


@needs_actuator
def test_a_layer_or_state_outside_the_actuator_is_refused(primary):
    for block in (0, 30, 34, 47):
        with pytest.raises(ActuatorError, match="decoder blocks"):
            primary.direction_for_layer(block)
    for state in (0, 31, 35, 45, 48):
        with pytest.raises(ActuatorError, match="states"):
            primary.direction_for_state(state)


@needs_actuator
def test_layer_thirty_two_is_not_state_thirty_two(primary):
    """The off-by-one that the two numbering schemes exist to prevent."""
    assert not np.array_equal(primary.direction_for_layer(32),
                              primary.direction_for_state(32))
    assert np.array_equal(primary.direction_for_layer(32),
                          primary.direction_for_state(33))


# ============================================================== 5. arithmetic
#
# Two paths exist and only one of them may produce a generation. These tests
# establish, in order: that the canonical path *is* the historical arithmetic,
# bit for bit; that the analytical helper is labelled and gated as an
# approximation; and that no caller asking for the first can be handed the
# second. See docs/decision_log.md, 2026-08-21.

def _historical_reference(direction, target, coefficient, reference=None,
                          reference_norm=None):
    """A literal transcription of the historical code, in torch.

    Deliberately written out here rather than imported, so that the test is a
    statement about what RQ2 executed and not a restatement of what the loader
    does. Sources, in the order they run:

        rq2_eval_mean_direction.make_synthetic_donor : donor = target + direction
        pathway_probe.build_site_plan                : raw_delta = donor - target
        pathway_probe.build_site_plan                : budget = strength * ||residual||
        pathway_probe.vector_norm                    : float32 norms
        pathway_probe.unit_scaled                    : delta * (budget / ||delta||)
    """
    import torch
    donor = target.float() + direction.float()
    raw_delta = donor - target.float()
    if reference_norm is None:
        reference_norm = float(torch.linalg.vector_norm(reference.float()).item())
    budget = coefficient * float(reference_norm)
    norm = float(torch.linalg.vector_norm(raw_delta.float()).item())
    return raw_delta.float() * (budget / norm)


@needs_actuator
@pytest.mark.parametrize("coefficient",
                         [0.15, 0.05, 0.10, 0.20, 0.37, 1.0, 2.5])
def test_the_canonical_path_reproduces_the_historical_arithmetic_exactly(
        primary, coefficient):
    """Bit for bit, for any coefficient — not to a tolerance."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(20260821 + int(coefficient * 1000))
    target = torch.from_numpy((rng.normal(size=HIDDEN_SIZE) * 3.0).astype(np.float32))
    reference = torch.from_numpy((rng.normal(size=HIDDEN_SIZE) * 3.0).astype(np.float32))
    direction = torch.from_numpy(np.array(primary.direction_for_layer(31),
                                          dtype=np.float32))

    expected = _historical_reference(direction, target, coefficient,
                                     reference=reference)
    got = primary.applied_delta(31, target, reference, coefficient=coefficient)

    assert torch.equal(expected, got), "not bit-identical"
    assert got.dtype == torch.float32
    assert type(got) is type(target)          # the backend it was handed


@needs_actuator
@pytest.mark.parametrize("layer", [31, 32, 33])
def test_the_canonical_path_is_exact_at_every_frozen_layer(primary, layer):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(4242 + layer)
    target = torch.from_numpy((rng.normal(size=HIDDEN_SIZE) * 7.0).astype(np.float32))
    reference = torch.from_numpy((rng.normal(size=HIDDEN_SIZE) * 7.0).astype(np.float32))
    direction = torch.from_numpy(np.array(primary.direction_for_layer(layer),
                                          dtype=np.float32))
    expected = _historical_reference(direction, target, primary.strength,
                                     reference=reference)
    assert torch.equal(expected, primary.applied_delta(layer, target, reference))


@needs_actuator
def test_the_canonical_path_defaults_to_the_frozen_coefficient(primary):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(9)
    target = torch.from_numpy(rng.normal(size=HIDDEN_SIZE).astype(np.float32))
    reference = torch.from_numpy(rng.normal(size=HIDDEN_SIZE).astype(np.float32))
    assert torch.equal(primary.applied_delta(31, target, reference),
                       primary.applied_delta(31, target, reference,
                                             coefficient=0.15))
    assert not torch.equal(primary.applied_delta(31, target, reference),
                           primary.applied_delta(31, target, reference,
                                                 coefficient=0.2))


@needs_actuator
def test_a_precomputed_reference_norm_gives_the_same_result_as_the_vector(primary):
    """The form a torch hook will use: the norm the model's own tensor gave."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(13)
    target = torch.from_numpy(rng.normal(size=HIDDEN_SIZE).astype(np.float32))
    reference = torch.from_numpy(rng.normal(size=HIDDEN_SIZE).astype(np.float32))
    norm = float(torch.linalg.vector_norm(reference.float()).item())
    assert torch.equal(primary.applied_delta(31, target, reference),
                       primary.applied_delta(31, target, reference_norm=norm))
    expected = _historical_reference(
        torch.from_numpy(np.array(primary.direction_for_layer(31), dtype=np.float32)),
        target, primary.strength, reference_norm=norm)
    assert torch.equal(expected,
                       primary.applied_delta(31, target, reference_norm=norm))


@needs_actuator
def test_the_canonical_path_matches_intervention_build_plan_bit_for_bit(primary):
    """The package's own donor-patching primitive, driven the historical way."""
    torch = pytest.importorskip("torch")
    from routegate.intervention import build_plan

    rng = np.random.default_rng(11)
    target = torch.from_numpy(rng.normal(size=HIDDEN_SIZE).astype(np.float32))
    reference = torch.from_numpy(rng.normal(size=HIDDEN_SIZE).astype(np.float32))
    direction = torch.from_numpy(np.array(primary.direction_for_layer(31),
                                          dtype=np.float32))

    plan = build_plan(state=historical_layer_to_state(31),
                      intervention_type="donor_direction",
                      strength=primary.strength,
                      target_state_vector=target,
                      donor_state_vector=target + direction,
                      reference_vector=reference,
                      budget_reference="block_input")
    assert plan.status == "ok"
    assert plan.budget_reference_state == 31       # state 32's input is state 31
    assert torch.equal(plan.delta,
                       primary.applied_delta(31, target, reference))


@needs_actuator
def test_the_numpy_backend_holds_the_same_operation_order(primary):
    """numpy and torch differ only in how they sum, never in what they compute."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(17)
    target = (rng.normal(size=HIDDEN_SIZE) * 2.0).astype(np.float32)
    reference = (rng.normal(size=HIDDEN_SIZE) * 2.0).astype(np.float32)

    from_numpy = primary.applied_delta(31, target, reference)
    from_torch = primary.applied_delta(31, torch.from_numpy(target),
                                       torch.from_numpy(reference))
    assert isinstance(from_numpy, np.ndarray) and from_numpy.dtype == np.float32
    relative = float(np.linalg.norm(from_numpy - from_torch.numpy())
                     / np.linalg.norm(from_numpy))
    # a 2048-term float32 reduction is not associative; see the module docstring
    assert relative <= 1e-6


# -------------------------------------------- the analytical helper is labelled
def test_the_analytical_helper_is_named_and_documented_as_inexact():
    assert "approx" in FrozenActuator.analytical_delta_approx.__name__
    doc = FrozenActuator.analytical_delta_approx.__doc__
    assert "Not exact" in doc
    assert "never a generation" in doc
    assert APPLICATION_PATH_ANALYTICAL == "analytical_approximate"
    assert "Approximate" in APPLICATION_PATHS[APPLICATION_PATH_ANALYTICAL]
    assert "never a generation" in APPLICATION_PATHS[APPLICATION_PATH_ANALYTICAL]
    # and the canonical one says the opposite
    assert "canonical" in FrozenActuator.applied_delta.__doc__.lower()
    assert "float32 cancellation" in APPLICATION_PATHS[APPLICATION_PATH_CANONICAL]


def test_the_manifest_records_both_paths_and_which_one_is_canonical(manifest):
    spec = manifest["application_paths"]
    assert spec["canonical"] == APPLICATION_PATH_CANONICAL
    assert spec["canonical"] == "historical_exact"
    assert spec[APPLICATION_PATH_ANALYTICAL]["exact"] is False
    assert spec[APPLICATION_PATH_CANONICAL]["exact"] is True
    assert spec[APPLICATION_PATH_ANALYTICAL]["may_produce_a_generation"] is False
    assert spec[APPLICATION_PATH_CANONICAL]["may_produce_a_generation"] is True
    assert spec["measured_divergence"]["1e+3"] > spec["measured_divergence"]["1e+0"]


@needs_actuator
def test_the_analytical_helper_refuses_without_an_explicit_acknowledgement(primary):
    rng = np.random.default_rng(21)
    reference = rng.normal(size=HIDDEN_SIZE).astype(np.float32)
    with pytest.raises(ActuatorError, match="not the path any generation"):
        primary.analytical_delta_approx(31, reference)
    with pytest.raises(ActuatorError, match="not the path any generation"):
        primary.analytical_delta_approx(31, reference, acknowledge_not_exact=False)
    for truthy_but_not_true in (1, "yes", [1]):
        with pytest.raises(ActuatorError, match="not the path any generation"):
            primary.analytical_delta_approx(
                31, reference, acknowledge_not_exact=truthy_but_not_true)
    assert primary.analytical_delta_approx(
        31, reference, acknowledge_not_exact=True) is not None


# ------------------------- no caller can ask for exact and receive approximate
def test_the_canonical_method_cannot_be_switched_to_the_analytical_one():
    """Structural, not conventional: there is no argument that would do it."""
    canonical = inspect.signature(FrozenActuator.applied_delta)
    analytical = inspect.signature(FrozenActuator.analytical_delta_approx)

    # the canonical path has no way to ask for the approximation
    assert "acknowledge_not_exact" not in canonical.parameters
    assert not any("approx" in name for name in canonical.parameters)

    # the approximation cannot even accept a target, so it cannot impersonate
    # the canonical call: the cancellation it omits is a function of the target
    assert "target_vector" in canonical.parameters
    assert "target_vector" not in analytical.parameters

    # the target is required and positional, so an old two-argument call fails
    target = canonical.parameters["target_vector"]
    assert target.default is inspect.Parameter.empty
    assert target.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD

    # the acknowledgement is keyword-only, so it cannot be passed by accident
    ack = analytical.parameters["acknowledge_not_exact"]
    assert ack.kind is inspect.Parameter.KEYWORD_ONLY
    assert ack.default is False


def test_neither_arithmetic_method_delegates_to_the_other():
    """An AST read: no branch inside one can reach the other."""
    tree = ast.parse(SOURCE_MODULE.read_text(encoding="utf-8"))
    bodies = {node.name: node for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef)
              and node.name in ("applied_delta", "analytical_delta_approx")}
    assert set(bodies) == {"applied_delta", "analytical_delta_approx"}
    for name, node in bodies.items():
        called = {n.func.attr for n in ast.walk(node)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        other = ("analytical_delta_approx" if name == "applied_delta"
                 else "applied_delta")
        assert other not in called, f"{name} calls {other}"


@needs_actuator
def test_the_actuator_reports_the_canonical_path_and_only_that(primary):
    assert primary.application_path == APPLICATION_PATH_CANONICAL == "historical_exact"
    record = primary.provenance()
    assert record["application_path"] == APPLICATION_PATH_CANONICAL
    assert "historical operation order" in record["application_path_note"]
    assert APPLICATION_PATH_ANALYTICAL not in record.values()


@needs_actuator
def test_the_two_paths_differ_enough_that_a_substitution_would_be_visible(primary):
    """If they agreed, none of the guarding above would matter."""
    rng = np.random.default_rng(20260821)
    target = (rng.normal(size=HIDDEN_SIZE) * 100.0).astype(np.float32)
    reference = (rng.normal(size=HIDDEN_SIZE) * 100.0).astype(np.float32)
    canonical = primary.applied_delta(31, target, reference)
    analytical = primary.analytical_delta_approx(31, reference,
                                                 acknowledge_not_exact=True)
    assert not np.array_equal(canonical, analytical)
    relative = float(np.linalg.norm(canonical - analytical)
                     / np.linalg.norm(canonical))
    assert relative > 1e-5


@needs_actuator
@pytest.mark.parametrize("scale,floor,ceiling",
                         [(1e-1, 0.0, 1e-6), (1e0, 1e-7, 1e-5),
                          (1e1, 1e-6, 1e-4), (1e2, 1e-5, 1e-3),
                          (1e3, 1e-4, 1e-2)])
def test_the_divergence_grows_with_the_target_as_documented(primary, scale,
                                                            floor, ceiling):
    """The cancellation is proportional to ||target||; the docstring says so."""
    rng = np.random.default_rng(int(scale * 7) + 3)
    target = (rng.normal(size=HIDDEN_SIZE) * scale).astype(np.float32)
    reference = (rng.normal(size=HIDDEN_SIZE) * scale).astype(np.float32)
    canonical = primary.applied_delta(31, target, reference)
    analytical = primary.analytical_delta_approx(31, reference,
                                                 acknowledge_not_exact=True)
    relative = float(np.linalg.norm(canonical - analytical)
                     / np.linalg.norm(canonical))
    assert floor <= relative <= ceiling, relative

    # a small rotation, not a different edit, and the budget is unaffected
    left = np.asarray(canonical, dtype=np.float64)
    right = np.asarray(analytical, dtype=np.float64)
    cosine = float(np.dot(left, right)
                   / (np.linalg.norm(left) * np.linalg.norm(right)))
    assert cosine > 0.9999
    assert float(np.linalg.norm(canonical)) == pytest.approx(
        float(np.linalg.norm(analytical)), rel=1e-5)


# ------------------------------------------------------- the budget and inputs
@needs_actuator
def test_the_budget_is_target_relative_and_reads_the_block_input(primary):
    """Doubling the block-input norm doubles the edit; nothing else does."""
    rng = np.random.default_rng(7)
    target = rng.normal(size=HIDDEN_SIZE).astype(np.float32)
    reference = rng.normal(size=HIDDEN_SIZE).astype(np.float32)
    small = primary.applied_delta(31, target, reference)
    large = primary.applied_delta(31, target, reference * np.float32(2.0))
    assert float(np.linalg.norm(large)) == pytest.approx(
        2.0 * float(np.linalg.norm(small)), rel=1e-5)

    assert primary.budget(10.0) == pytest.approx(1.5)
    assert primary.budget(10.0, coefficient=0.2) == pytest.approx(2.0)
    assert primary.budget_reference == "block_input"


@needs_actuator
def test_the_two_reference_forms_are_mutually_exclusive(primary):
    rng = np.random.default_rng(29)
    target = rng.normal(size=HIDDEN_SIZE).astype(np.float32)
    reference = rng.normal(size=HIDDEN_SIZE).astype(np.float32)
    with pytest.raises(ActuatorError, match="exactly one of"):
        primary.applied_delta(31, target)
    with pytest.raises(ActuatorError, match="exactly one of"):
        primary.applied_delta(31, target, reference, reference_norm=1.0)
    with pytest.raises(ActuatorError, match="exactly one of"):
        primary.analytical_delta_approx(31, acknowledge_not_exact=True)


@needs_actuator
def test_bad_vectors_and_coefficients_are_refused(primary):
    rng = np.random.default_rng(31)
    good = rng.normal(size=HIDDEN_SIZE).astype(np.float32)
    with pytest.raises(ActuatorError, match="hidden unit"):
        primary.applied_delta(31, np.ones(7, dtype=np.float32), good)
    with pytest.raises(ActuatorError, match="one hidden-state vector"):
        primary.applied_delta(31, np.ones((2, HIDDEN_SIZE), dtype=np.float32), good)
    bad = good.copy()
    bad[0] = np.nan
    with pytest.raises(ActuatorError, match="non-finite"):
        primary.applied_delta(31, bad, good)
    with pytest.raises(ActuatorError, match="non-finite"):
        primary.applied_delta(31, good, bad)
    with pytest.raises(ActuatorError, match="finite and non-negative"):
        primary.budget(-1.0)
    for bad_coefficient in (0.0, -0.15, float("nan"), float("inf")):
        with pytest.raises(ActuatorError, match="coefficient"):
            primary.budget(1.0, coefficient=bad_coefficient)


@needs_actuator
def test_the_edit_is_additive_and_lands_on_the_final_prompt_token(primary):
    """Added, never substituted, and at row -1 of the prefill forward."""
    torch = pytest.importorskip("torch")
    from routegate.intervention import (apply_delta_at_position,
                                        final_prompt_token_index)

    sequence = 9
    activation = torch.zeros(1, sequence, HIDDEN_SIZE)
    activation[0, 4] = 3.0                      # an untouched neighbour
    rng = np.random.default_rng(37)
    target = torch.from_numpy(rng.normal(size=HIDDEN_SIZE).astype(np.float32))
    reference = torch.from_numpy(np.ones(HIDDEN_SIZE, dtype=np.float32))
    delta = primary.applied_delta(31, target, reference)

    position = final_prompt_token_index(sequence)
    assert position == sequence - 1
    edited = apply_delta_at_position(activation, position, delta)

    assert edited is not activation                     # never mutated in place
    np.testing.assert_array_equal(edited[0, :position].numpy(),
                                  activation[0, :position].numpy())
    np.testing.assert_allclose(edited[0, position].numpy(), delta.numpy(),
                               rtol=0, atol=0)


# =============================================== 5b. device, and the coefficient
#
# Two runtime defects were found after the migration landed and are fixed here.
#
#   device      _as_float32_like converted a numpy direction with
#               torch.from_numpy(), which always lands on the CPU. Inside a real
#               hook the reference activation is on CUDA, and the very next line
#               -- target + direction -- raised "Expected all tensors to be on
#               the same device". A torch value on another device was likewise
#               only .float()-ed, never moved.
#
#   coefficient the budget path called float(coefficient), which detaches a
#               torch coefficient. The delta came back with requires_grad False
#               and backward() raised, so the controller could never have been
#               trained through it.
#
# The device tests below use torch's "meta" device wherever CUDA is absent. meta
# is a real non-CPU device that exists on any host, so the *mismatch* half of the
# defect is reproducible without a GPU; the CUDA tests carry the half that needs
# real hardware and skip when there is none.

def _cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


needs_cuda = pytest.mark.skipif(
    not _cuda_available(),
    reason="no CUDA device is allocated to this host; run under a GPU job")


# ------------------------------------------------------------------- device
@needs_actuator
@pytest.mark.parametrize("layer", [31, 32, 33])
def test_a_cpu_target_gives_a_cpu_float32_delta(primary, layer):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(101 + layer)
    target = torch.from_numpy(rng.normal(size=HIDDEN_SIZE).astype(np.float32))
    reference = torch.from_numpy(rng.normal(size=HIDDEN_SIZE).astype(np.float32))

    delta = primary.applied_delta(layer, target, reference)
    assert delta.device == target.device == torch.device("cpu")
    assert delta.dtype == torch.float32
    assert delta.shape == target.shape


@needs_actuator
def test_the_direction_is_converted_onto_the_reference_device(primary):
    """The regression itself: a numpy direction must not stay on the CPU."""
    torch = pytest.importorskip("torch")
    from routegate.actuator import _as_float32_like

    for device in ("cpu", "meta"):
        reference = torch.zeros(HIDDEN_SIZE, device=device)
        converted = _as_float32_like(
            np.array(primary.direction_for_layer(31), dtype=np.float32), reference)
        assert converted.device == reference.device, device
        assert converted.dtype == torch.float32
        # the operation that used to raise
        assert (reference + converted).device == reference.device


@needs_actuator
def test_a_torch_value_on_another_device_is_moved_not_only_recast(primary):
    """Copying *into* meta is the direction a CUDA reference would exercise.

    Copying *out* of a meta tensor is impossible by design -- it carries no
    data -- so this is the half of the torch-to-torch path that is testable
    without a GPU. The CUDA test below covers the other half.
    """
    torch = pytest.importorskip("torch")
    from routegate.actuator import _as_float32_like

    value = torch.zeros(4, dtype=torch.float64)          # wrong device and dtype
    moved = _as_float32_like(value, torch.zeros(4, device="meta"))
    assert moved.device == torch.device("meta")
    assert moved.dtype == torch.float32


@needs_actuator
def test_the_cpu_path_is_not_copied_when_nothing_needs_changing(primary):
    """``.to()`` returns self, so the historical CPU arithmetic is untouched."""
    torch = pytest.importorskip("torch")
    from routegate.actuator import _as_float32_like

    already = torch.zeros(4, dtype=torch.float32)
    assert _as_float32_like(already, already) is already


@needs_actuator
@needs_cuda
@pytest.mark.parametrize("layer", [31, 32, 33])
def test_a_cuda_target_gives_a_cuda_float32_delta(primary, layer):
    """The defect this fix exists for. Skipped unless a GPU is allocated."""
    torch = pytest.importorskip("torch")
    device = torch.device("cuda", torch.cuda.current_device())
    rng = np.random.default_rng(202 + layer)
    target = torch.from_numpy(
        rng.normal(size=HIDDEN_SIZE).astype(np.float32)).to(device)
    reference = torch.from_numpy(
        rng.normal(size=HIDDEN_SIZE).astype(np.float32)).to(device)

    delta = primary.applied_delta(layer, target, reference)
    assert delta.is_cuda
    assert delta.device == target.device
    assert delta.dtype == torch.float32

    # and it is the same numbers the CPU produces, up to the float32 reduction
    cpu = primary.applied_delta(layer, target.cpu(), reference.cpu())
    relative = float(np.linalg.norm(delta.cpu().numpy() - cpu.numpy())
                     / np.linalg.norm(cpu.numpy()))
    assert relative <= 1e-5


@needs_actuator
@needs_cuda
def test_a_cuda_target_with_a_precomputed_norm_also_works(primary):
    torch = pytest.importorskip("torch")
    device = torch.device("cuda", torch.cuda.current_device())
    rng = np.random.default_rng(303)
    target = torch.from_numpy(
        rng.normal(size=HIDDEN_SIZE).astype(np.float32)).to(device)
    norm = 12.5
    delta = primary.applied_delta(31, target, reference_norm=norm)
    assert delta.is_cuda and delta.dtype == torch.float32


# -------------------------------------------------- the differentiable path
@needs_actuator
def test_a_tensor_coefficient_keeps_the_delta_in_the_autograd_graph(primary):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(11)
    target = torch.from_numpy(rng.normal(size=HIDDEN_SIZE).astype(np.float32))
    reference = torch.from_numpy(rng.normal(size=HIDDEN_SIZE).astype(np.float32))
    coefficient = torch.tensor(0.15, requires_grad=True)

    delta = primary.applied_delta(31, target, reference, coefficient=coefficient)
    assert delta.requires_grad
    assert delta.grad_fn is not None

    loss = (delta ** 2).sum()
    loss.backward()

    assert coefficient.grad is not None
    assert bool(torch.isfinite(coefficient.grad).all())
    assert float(coefficient.grad) != 0.0


@needs_actuator
def test_only_the_coefficient_receives_a_gradient(primary):
    """The frozen direction and the activations stay out of the graph."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(13)
    target = torch.from_numpy(rng.normal(size=HIDDEN_SIZE).astype(np.float32))
    reference = torch.from_numpy(rng.normal(size=HIDDEN_SIZE).astype(np.float32))
    coefficient = torch.tensor(0.2, requires_grad=True)

    assert not target.requires_grad and not reference.requires_grad
    delta = primary.applied_delta(31, target, reference, coefficient=coefficient)
    delta.sum().backward()

    assert coefficient.grad is not None
    assert target.grad is None and reference.grad is None
    assert primary.directions.flags.writeable is False       # still frozen


@needs_actuator
def test_a_zero_tensor_coefficient_gives_an_exact_zero_edit(primary):
    """How a controller declines to act — and it must still be trainable."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(17)
    target = torch.from_numpy(rng.normal(size=HIDDEN_SIZE).astype(np.float32))
    reference = torch.from_numpy(rng.normal(size=HIDDEN_SIZE).astype(np.float32))
    coefficient = torch.tensor(0.0, requires_grad=True)

    delta = primary.applied_delta(31, target, reference, coefficient=coefficient)
    assert torch.equal(delta, torch.zeros_like(delta))       # exact, not close
    assert float(delta.abs().max()) == 0.0

    delta.sum().backward()
    assert coefficient.grad is not None
    assert float(coefficient.grad) != 0.0      # zero output, live gradient


@needs_actuator
@pytest.mark.parametrize("bad,match", [
    (-0.1, "must not be negative"),
    (float("nan"), "must be finite"),
    (float("inf"), "must be finite"),
])
def test_an_invalid_tensor_coefficient_fails_closed(primary, bad, match):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(19)
    target = torch.from_numpy(rng.normal(size=HIDDEN_SIZE).astype(np.float32))
    reference = torch.from_numpy(rng.normal(size=HIDDEN_SIZE).astype(np.float32))
    with pytest.raises(ActuatorError, match=match):
        primary.applied_delta(31, target, reference,
                              coefficient=torch.tensor(bad))


@needs_actuator
def test_a_non_scalar_tensor_coefficient_fails_closed(primary):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(23)
    target = torch.from_numpy(rng.normal(size=HIDDEN_SIZE).astype(np.float32))
    reference = torch.from_numpy(rng.normal(size=HIDDEN_SIZE).astype(np.float32))
    for shape in ((2,), (2, 3)):
        with pytest.raises(ActuatorError, match="single scalar value"):
            primary.applied_delta(31, target, reference,
                                  coefficient=torch.ones(shape))
    # one element in a shaped tensor is still one value, and is accepted
    delta = primary.applied_delta(31, target, reference,
                                  coefficient=torch.tensor([[0.15]]))
    assert delta.shape == target.shape


@needs_actuator
def test_a_tensor_coefficient_needs_a_torch_target(primary):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(29)
    target = rng.normal(size=HIDDEN_SIZE).astype(np.float32)      # numpy
    reference = rng.normal(size=HIDDEN_SIZE).astype(np.float32)
    with pytest.raises(ActuatorError, match="needs a torch target"):
        primary.applied_delta(31, target, reference,
                              coefficient=torch.tensor(0.15))
    with pytest.raises(ActuatorError, match="needs a torch target"):
        primary.analytical_delta_approx(31, reference,
                                        coefficient=torch.tensor(0.15),
                                        acknowledge_not_exact=True)


@needs_actuator
def test_the_historical_budget_method_refuses_a_tensor_coefficient(primary):
    """`budget()` is the fixed-scalar form and says so rather than detaching."""
    torch = pytest.importorskip("torch")
    with pytest.raises(ActuatorError, match="historical fixed-scalar form"):
        primary.budget(10.0, coefficient=torch.tensor(0.15))


@needs_actuator
def test_validation_does_not_consume_the_gradient_path(primary):
    """Checks read `.detach()`; the multiply uses the caller's own tensor."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(31)
    target = torch.from_numpy(rng.normal(size=HIDDEN_SIZE).astype(np.float32))
    reference = torch.from_numpy(rng.normal(size=HIDDEN_SIZE).astype(np.float32))

    source = torch.tensor(0.3, requires_grad=True)
    coefficient = source * 2.0                 # a non-leaf, mid-graph
    delta = primary.applied_delta(31, target, reference, coefficient=coefficient)
    delta.sum().backward()
    assert source.grad is not None and float(source.grad) != 0.0


@needs_actuator
def test_the_two_coefficient_modes_are_named_and_distinguished(primary):
    torch = pytest.importorskip("torch")
    assert primary.coefficient_mode() == COEFFICIENT_MODE_HISTORICAL
    assert primary.coefficient_mode(0.15) == COEFFICIENT_MODE_HISTORICAL
    assert primary.coefficient_mode(torch.tensor(0.15)) == COEFFICIENT_MODE_CONTROLLER
    assert COEFFICIENT_MODE_HISTORICAL != COEFFICIENT_MODE_CONTROLLER

    historical = COEFFICIENT_MODES[COEFFICIENT_MODE_HISTORICAL]
    controller = COEFFICIENT_MODES[COEFFICIENT_MODE_CONTROLLER]
    assert "replays RQ2" in historical
    assert "bit for bit" in historical
    # the learned path must never be described as a historical generation
    assert "NOT a historical RQ2 generation" in controller
    assert "not even at" in controller


@needs_actuator
def test_a_tensor_coefficient_at_the_frozen_value_still_is_not_a_replay(primary):
    """Numerically equal, provenance-wise different — and the value agrees."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(37)
    target = torch.from_numpy(rng.normal(size=HIDDEN_SIZE).astype(np.float32))
    reference = torch.from_numpy(rng.normal(size=HIDDEN_SIZE).astype(np.float32))

    replay = primary.applied_delta(31, target, reference)                 # float
    learned = primary.applied_delta(31, target, reference,
                                    coefficient=torch.tensor(0.15))       # tensor
    np.testing.assert_allclose(learned.detach().numpy(), replay.numpy(),
                               rtol=1e-6, atol=0)
    assert primary.coefficient_mode() != primary.coefficient_mode(
        torch.tensor(0.15))


@needs_actuator
@needs_cuda
def test_the_coefficient_trains_on_cuda_and_may_arrive_on_another_device(primary):
    """Gradient and device together, on real hardware. Skipped without a GPU."""
    torch = pytest.importorskip("torch")
    device = torch.device("cuda", torch.cuda.current_device())
    rng = np.random.default_rng(41)
    target = torch.from_numpy(
        rng.normal(size=HIDDEN_SIZE).astype(np.float32)).to(device)
    reference = torch.from_numpy(
        rng.normal(size=HIDDEN_SIZE).astype(np.float32)).to(device)

    # deliberately emitted on the CPU, as a controller head might
    coefficient = torch.tensor(0.15, requires_grad=True)
    assert coefficient.device.type == "cpu"

    delta = primary.applied_delta(31, target, reference, coefficient=coefficient)
    assert delta.is_cuda and delta.dtype == torch.float32
    assert delta.requires_grad

    (delta ** 2).sum().backward()
    assert coefficient.grad is not None
    assert bool(torch.isfinite(coefficient.grad).all())
    assert float(coefficient.grad) != 0.0

    on_device = torch.tensor(0.15, device=device, requires_grad=True)
    same = primary.applied_delta(31, target, reference, coefficient=on_device)
    (same ** 2).sum().backward()
    assert on_device.grad is not None and float(on_device.grad) != 0.0


# ============================================================= 6. fail-closed
@needs_actuator
def test_a_missing_file_is_refused(sandbox, manifest):
    (sandbox / "freeze_summary.json").unlink()
    with pytest.raises(ActuatorError, match="missing"):
        load_frozen_actuator(PRIMARY_VARIANT, directory=sandbox,
                             manifest=manifest)


@needs_actuator
def test_a_single_flipped_byte_is_refused(sandbox, manifest):
    path = sandbox / "frozen_direction_manifest.json"
    payload = bytearray(path.read_bytes())
    payload[100] ^= 0x01
    path.write_bytes(bytes(payload))
    with pytest.raises(ActuatorError, match="hash mismatch"):
        load_frozen_actuator(PRIMARY_VARIANT, directory=sandbox,
                             manifest=manifest)


@needs_actuator
def test_a_truncated_file_is_refused_on_size_before_its_hash(sandbox, manifest):
    path = sandbox / "config.json"
    path.write_bytes(path.read_bytes()[:-16])
    with pytest.raises(ActuatorError, match="bytes; the freeze recorded"):
        load_frozen_actuator(PRIMARY_VARIANT, directory=sandbox,
                             manifest=manifest)


@needs_actuator
def test_a_flipped_byte_inside_the_tensor_is_refused(sandbox, manifest):
    path = sandbox / "frozen_directions.pt"
    payload = bytearray(path.read_bytes())
    payload[-32] ^= 0x01
    path.write_bytes(bytes(payload))
    with pytest.raises(ActuatorError, match="hash mismatch"):
        load_frozen_actuator(PRIMARY_VARIANT, directory=sandbox,
                             manifest=manifest)


@needs_actuator
def test_an_unexpected_file_in_the_directory_is_refused(sandbox, manifest):
    (sandbox / "notes.txt").write_text("stray", encoding="utf-8")
    with pytest.raises(ValidationError, match="unexpected"):
        load_frozen_actuator(PRIMARY_VARIANT, directory=sandbox,
                             manifest=manifest)


@needs_actuator
def test_a_disagreeing_historical_checksum_file_is_refused(sandbox, manifest):
    """Both records must agree; one silently rewritten is a refusal."""
    sums = sandbox / "FREEZE_SHA256SUMS.txt"
    text = sums.read_text(encoding="utf-8").replace(
        "ee23afd98c5b9c39cc864af0e475392a69deb5ed2ef33a30d9f10703b5cb2bea",
        "0" * 64)
    sums.write_text(text, encoding="utf-8")
    updated = json.loads(json.dumps(manifest))
    entry = updated["files"]["the_checksum_file_itself"]["FREEZE_SHA256SUMS.txt"]
    entry["sha256"] = sha256_file(sums)
    entry["bytes"] = sums.stat().st_size
    with pytest.raises(ValidationError, match="disagreement"):
        load_frozen_actuator(PRIMARY_VARIANT, directory=sandbox,
                             manifest=updated)


@needs_actuator
def test_an_extra_archive_member_is_refused(sandbox, manifest):
    def mutate(members):
        members["frozen_directions/extra"] = b"x"
        return members
    _rewrite_archive(sandbox, mutate)
    with pytest.raises(ActuatorError, match="unexpected"):
        load_frozen_actuator(PRIMARY_VARIANT, directory=sandbox,
                             manifest=_manifest_for(sandbox, manifest))


@needs_actuator
def test_a_missing_archive_member_is_refused(sandbox, manifest):
    def mutate(members):
        members.pop("frozen_directions/byteorder")
        return members
    _rewrite_archive(sandbox, mutate)
    with pytest.raises(ActuatorError, match="missing"):
        load_frozen_actuator(PRIMARY_VARIANT, directory=sandbox,
                             manifest=_manifest_for(sandbox, manifest))


@needs_actuator
def test_a_second_archive_root_is_refused(sandbox, manifest):
    def mutate(members):
        members["elsewhere/data/0"] = b"\x00" * BYTES_PER_STORAGE
        return members
    _rewrite_archive(sandbox, mutate)
    with pytest.raises(ActuatorError, match="top-level director"):
        load_frozen_actuator(PRIMARY_VARIANT, directory=sandbox,
                             manifest=_manifest_for(sandbox, manifest))


@needs_actuator
def test_a_big_endian_archive_is_refused(sandbox, manifest):
    def mutate(members):
        members["frozen_directions/byteorder"] = b"big"
        return members
    _rewrite_archive(sandbox, mutate)
    updated = _manifest_for(sandbox, manifest)
    updated["archive"]["expected_member_bytes"]["byteorder"] = 3
    with pytest.raises(ActuatorError, match="byte order"):
        load_frozen_actuator(PRIMARY_VARIANT, directory=sandbox,
                             manifest=updated)


@needs_actuator
def test_a_changed_format_marker_is_refused(sandbox, manifest):
    def mutate(members):
        members["frozen_directions/version"] = b"4\n"
        return members
    _rewrite_archive(sandbox, mutate)
    with pytest.raises(ActuatorError, match="version"):
        load_frozen_actuator(PRIMARY_VARIANT, directory=sandbox,
                             manifest=_manifest_for(sandbox, manifest))


@needs_actuator
def test_a_resized_storage_is_refused(sandbox, manifest):
    def mutate(members):
        members["frozen_directions/data/0"] += b"\x00" * 4
        return members
    _rewrite_archive(sandbox, mutate)
    with pytest.raises(ActuatorError, match="bytes"):
        load_frozen_actuator(PRIMARY_VARIANT, directory=sandbox,
                             manifest=_manifest_for(sandbox, manifest))


@needs_actuator
def test_an_altered_storage_matches_no_variant_and_is_refused(sandbox, manifest):
    def mutate(members):
        payload = bytearray(members["frozen_directions/data/0"])
        payload[0] ^= 0x01
        members["frozen_directions/data/0"] = bytes(payload)
        return members
    _rewrite_archive(sandbox, mutate)
    with pytest.raises(ActuatorError, match="matches 0 storage member"):
        load_frozen_actuator(PRIMARY_VARIANT, directory=sandbox,
                             manifest=_manifest_for(sandbox, manifest))


@needs_actuator
def test_a_non_finite_storage_is_refused(sandbox, manifest):
    def mutate(members):
        bad = np.full((N_LAYERS, HIDDEN_SIZE), np.nan, dtype="<f4")
        members["frozen_directions/data/0"] = bad.tobytes()
        return members
    _rewrite_archive(sandbox, mutate)
    with pytest.raises(ActuatorError, match="non-finite"):
        load_frozen_actuator(PRIMARY_VARIANT, directory=sandbox,
                             manifest=_manifest_for(sandbox, manifest))


@needs_actuator
def test_an_unexpected_pickle_global_is_refused(sandbox, manifest):
    """The archive claiming to import something else is a hard stop."""
    original = b"ccollections\nOrderedDict\n"     # the GLOBAL opcode, verbatim
    swapped = b"csubprocess\ncheck_output\n"      # same length, different import

    def mutate(members):
        payload = members["frozen_directions/data.pkl"]
        assert original in payload
        assert len(original) == len(swapped)
        members["frozen_directions/data.pkl"] = payload.replace(
            original, swapped, 1)
        return members
    _rewrite_archive(sandbox, mutate)
    with pytest.raises(ActuatorError, match="not one of the symbols"):
        load_frozen_actuator(PRIMARY_VARIANT, directory=sandbox,
                             manifest=_manifest_for(sandbox, manifest))


@needs_actuator
def test_changed_pickle_metadata_is_refused(sandbox, manifest):
    """A string added or altered inside the archive changes the atom digest."""
    def mutate(members):
        payload = members["frozen_directions/data.pkl"]
        members["frozen_directions/data.pkl"] = payload.replace(
            b"greedy", b"gr33dy", 1)
        return members
    _rewrite_archive(sandbox, mutate)
    with pytest.raises(ActuatorError, match="metadata digest"):
        load_frozen_actuator(PRIMARY_VARIANT, directory=sandbox,
                             manifest=_manifest_for(sandbox, manifest))


@needs_actuator
def test_a_freeze_recording_a_different_strength_is_refused(sandbox, manifest):
    path = sandbox / "frozen_direction_manifest.json"
    frozen = json.loads(path.read_text(encoding="utf-8"))
    frozen["strength"] = 0.2
    path.write_text(json.dumps(frozen, indent=1), encoding="utf-8")
    updated = json.loads(json.dumps(manifest))
    entry = updated["files"]["covered_by_historical_checksum"][
        "frozen_direction_manifest.json"]
    entry["sha256"] = sha256_file(path)
    entry["bytes"] = path.stat().st_size
    sums = sandbox / "FREEZE_SHA256SUMS.txt"
    sums.write_text("\n".join(
        f"{entry['sha256']}  {line.split(None, 1)[1]}"
        if line.strip().endswith("frozen_direction_manifest.json") else line
        for line in sums.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8")
    checksum_entry = updated["files"]["the_checksum_file_itself"][
        "FREEZE_SHA256SUMS.txt"]
    checksum_entry["sha256"] = sha256_file(sums)
    checksum_entry["bytes"] = sums.stat().st_size
    with pytest.raises(ActuatorError, match="population scan"):
        load_frozen_actuator(PRIMARY_VARIANT, directory=sandbox,
                             manifest=updated)


@needs_actuator
def test_the_manifest_cannot_widen_the_pickle_allowlist(sandbox, manifest):
    """The manifest names the allowlist; the code bounds it."""
    updated = json.loads(json.dumps(manifest))
    updated["archive"]["pickle_policy"]["allowed_globals"].append(
        "subprocess check_output")
    with pytest.raises(ActuatorError, match="bounded by the code"):
        load_frozen_actuator(PRIMARY_VARIANT, directory=sandbox,
                             manifest=updated)


@needs_actuator
def test_the_manifest_cannot_loosen_the_unit_norm_tolerance(sandbox, manifest):
    """It may tighten the gate; it may not relax it."""
    updated = json.loads(json.dumps(manifest))
    updated["tensors"]["unit_norm_tolerance"] = 1e-1
    with pytest.raises(ActuatorError, match="bounded by the code"):
        load_frozen_actuator(PRIMARY_VARIANT, directory=sandbox,
                             manifest=updated)
    tightened = json.loads(json.dumps(manifest))
    tightened["tensors"]["unit_norm_tolerance"] = 1e-6
    loaded = load_frozen_actuator(PRIMARY_VARIANT, directory=sandbox,
                                  manifest=tightened)
    assert loaded.fingerprint          # the real directions clear 1e-6 too


@needs_actuator
def test_an_unknown_variant_is_refused(freeze_dir):
    with pytest.raises(ActuatorError, match="unknown variant"):
        load_frozen_actuator("orthogonalized_population_repair",
                             directory=freeze_dir)


def test_a_missing_paths_key_is_refused(tmp_path):
    paths = tmp_path / "paths.json"
    paths.write_text(json.dumps({"dataset": "/nowhere.csv"}), encoding="utf-8")
    with pytest.raises(ActuatorError, match=PATHS_KEY):
        actuator_dir(paths_path=paths)


def test_an_absent_paths_file_is_refused(tmp_path):
    with pytest.raises(ActuatorError, match="does not exist"):
        actuator_dir(paths_path=tmp_path / "absent.json")


def test_a_directory_that_is_not_there_is_refused(tmp_path):
    paths = tmp_path / "paths.json"
    paths.write_text(json.dumps({PATHS_KEY: str(tmp_path / "absent")}),
                     encoding="utf-8")
    with pytest.raises(ActuatorError, match="not a directory"):
        actuator_dir(paths_path=paths)


def test_a_sealed_looking_directory_is_refused(tmp_path):
    paths = tmp_path / "paths.json"
    paths.write_text(json.dumps({PATHS_KEY: "/data/sealed/rq2_actuator"}),
                     encoding="utf-8")
    with pytest.raises(ValidationError, match="sealed"):
        actuator_dir(paths_path=paths)


# =============================================================== 7. isolation
def test_the_manifest_records_no_absolute_path_and_no_prompt():
    text = MANIFEST_PATH.read_text(encoding="utf-8")
    assert "/home/" not in text
    assert "/Users/" not in text
    manifest = json.loads(text)
    assert manifest["local_destination"]["outside_git"] is True
    # the taxonomy claim this migration turns on
    assert manifest["direction_semantics"]["orthogonalisation"].startswith("none")
    assert manifest["intervention"]["strength"] == 0.15
    assert manifest["intervention"]["budget_reference"] == "block_input"
    assert manifest["layer_mapping"]["historical_decoder_blocks"] == [31, 32, 33]
    assert manifest["layer_mapping"]["routegate_states"] == [32, 33, 34]


@needs_actuator
def test_no_migrated_json_carries_prompt_or_response_text(freeze_dir):
    """Every prompt-shaped field in the freeze holds a digest, never text."""
    suspicious = re.compile(r"prompt|response|completion|answer|text",
                            re.IGNORECASE)
    offenders = []

    def walk(node, trail):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{trail}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{trail}[{index}]")
        elif isinstance(node, str) and suspicious.search(trail):
            leaf = trail.rsplit(".", 1)[-1]
            if HEX64.match(node):
                return                       # a prompt digest, which is the point
            if node.endswith((".csv", ".json", ".md", ".txt")):
                return                       # a filename
            if node.startswith("results/"):
                return                       # an origin-relative run directory
            if leaf == "system_prompt":
                return                       # the recorded absence of one
            offenders.append((trail, node[:40], len(node)))

    for name in ("config.json", "freeze_summary.json",
                 "frozen_direction_manifest.json", "input_hash_manifest.json",
                 "preregistered_final_protocol.json"):
        walk(json.loads((freeze_dir / name).read_text(encoding="utf-8")), name)
    assert offenders == []


@needs_actuator
def test_the_source_case_manifest_is_hashes_only(freeze_dir):
    import csv
    with open(freeze_dir / "source_case_manifest.csv", newline="",
              encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    hash_columns = [c for c in rows[0] if c.endswith("_sha256")]
    assert len(hash_columns) == 5
    for row in rows:
        assert row["role"] == "development_source_case"
        assert row["capture_index"].isdigit()
        for column in hash_columns:
            assert HEX64.match(row[column]), (row["case_id"], column)
        # no column holds anything that could be prompt text
        assert set(row) == {"case_id", "capture_index", "role", *hash_columns}


@needs_actuator
def test_no_migrated_file_names_held_out_material(freeze_dir):
    from routegate.validation import SEALED_PATH_TOKENS
    for path in sorted(freeze_dir.iterdir()):
        if path.suffix == ".pt":
            continue
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for token in SEALED_PATH_TOKENS:
            assert token not in text, f"{path.name} names {token!r}"


@needs_actuator
def test_loading_touches_no_model_gpu_or_network(freeze_dir, monkeypatch):
    """The gate that keeps this phase honest: it is arithmetic and hashing."""
    import socket

    def no_network(*args, **kwargs):
        raise AssertionError("the frozen actuator loader opened a socket")

    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    loaded = load_frozen_actuator(PRIMARY_VARIANT, directory=freeze_dir)
    assert loaded.fingerprint
    assert "transformers" not in sys.modules
    source = SOURCE_MODULE.read_text(encoding="utf-8")
    for forbidden in ("transformers", "AutoModel", "requests", "urllib",
                      "http", "cuda", "device_map"):
        assert forbidden not in source, forbidden


def test_the_module_imports_without_torch(monkeypatch):
    """numpy is required to read the actuator; torch is not."""
    import importlib
    monkeypatch.setitem(sys.modules, "torch", None)
    reloaded = importlib.reload(actuator_module)
    assert reloaded.FROZEN_STRENGTH == 0.15
    importlib.reload(actuator_module)
