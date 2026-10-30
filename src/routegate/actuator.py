"""The frozen RQ2 repair actuator: reading it safely, and what it means.

This module loads one artefact -- the single global repair direction RQ2 froze
on 2026-07-21 -- and refuses to load anything else. It holds no model, opens no
network connection and registers no hook, so every claim it makes can be checked
on synthetic tensors.

What the actuator is
--------------------
Three unit vectors, one per historical decoder block 31, 32 and 33, each of
2048 hidden units. The direction points **from an over-refused safe prompt
towards a benign answered one**: each contributing delta is a benign donor's
residual minus the over-refused target's residual (B - E), unit-normalised per
pair before any averaging, then averaged across donors within a case and across
the ten source cases, renormalising at each step. No class mean is taken and no
H term enters it. There is no orthogonalisation anywhere in its construction.

Four things it is not, because three of them are also called "the direction"
somewhere in this project. See ``docs/artifact_taxonomy.md``; the short form is:

* the **population direction bank** (``outputs/causal_repair/…/direction_bank/``)
  is a different object -- class means over the 480-row corpus, H-orthogonalised,
  cross-fitted, scanned at strength 0.2. It is an ablation. It does not replace
  this actuator and must never be described as it.
* the **89-row E-B donor table** (``outputs/causal_repair/donor_pairs.csv``) is a
  metadata pairing of corpus rows, with no H arm and no bijection.
* the **480-row E/B/H corpus** is unpaired by design, and is not what this
  actuator was estimated from.
* the **original matched triplets** are the prompts it *was* estimated from.
  They are not migrated here, and their prompt hashes do not intersect the
  corpus at all.

Why the loader is written the way it is
---------------------------------------
The historical artefact is a ``.pt``, which is a zip containing a pickle. Loading
a pickle executes whatever the pickle contains, and ``torch.load`` is exactly
that. Neither is used here.

Instead:

1. every file in the migrated freeze directory is re-hashed against
   ``manifests/rq2_actuator.json`` **before the archive is opened**;
2. the archive's member set, per-member sizes, byte order and format markers are
   checked against the manifest;
3. ``data.pkl`` is decoded opcode by opcode with :func:`pickletools.genops`,
   which reads the byte stream and executes nothing, purely to confirm the
   archive references only the three symbols the historical file is known to
   reference. It is an integrity check, not a data path;
4. the tensors are read from the raw ``data/N`` members with an explicit
   little-endian float32 dtype, and are bound to their variant names **by
   content** -- each storage's per-layer SHA256 must match exactly one variant in
   the historical ``frozen_direction_manifest.json``, and the binding must be
   bijective. The order of entries inside the archive is checked afterwards as a
   cross-check, and is never the source of truth;
5. the recomputed fingerprint and per-layer norms must match the freeze's own.

Any failure raises. Nothing is repaired, defaulted, or warned about.

The intervention this actuator defines
--------------------------------------
Preserved exactly as RQ2 froze it, and stated here so that a future caller
cannot quietly change one term of it::

    site            the decoder block's OUTPUT hidden state
    layers          historical blocks 31, 32, 33  ->  states 32, 33, 34
    position        the final prompt token, during prefill only
    operation       additive; added to the hooked tensor, never substituted
    coefficient     0.15 by default; any positive value is accepted
    budget          coefficient * ||the block's INPUT hidden state||

The budget reference is **not** the tensor being edited. RQ1 read it from the
block's input -- the preceding state -- and RQ2 inherited that. It is preserved
because changing it would change what every recorded RQ2 and RQ3 number means.

Two ways to compute the edit, and only one of them is the default
-----------------------------------------------------------------
RQ2 did not add its direction directly. It built a *synthetic donor*, ``donor =
target + direction``, handed that to ``pathway_probe.build_site_plan``, and let
it recompute ``raw_delta = donor - target``. Algebraically that is the identity.
In float32 it is not: ``fl(target + direction) - target`` discards low-order bits
of the direction in proportion to ``‖target‖``.

So there are two different vectors, and they are not interchangeable:

``applied_delta``  -- **canonical, and the default.**
    The historical order, transcribed. Carries the cancellation, and therefore
    reproduces what RQ2 and the two static E/B/H transfers actually applied.
    Every generation must come from here.

``analytical_delta_approx`` -- **approximate, never a generation.**
    ``unit(direction) * coefficient * ||reference||``. The closed form. Requires
    ``acknowledge_not_exact=True``, so arriving here is always deliberate.

Measured divergence between them, float32 throughout, one layer, by the target's
element scale -- it grows linearly with ``‖target‖`` because that is what drives
the cancellation::

    target element ~     1e-1     1e+0     1e+1     1e+2     1e+3
    relative diff       1.2e-7   1.1e-6   1.2e-5   1.1e-4   1.1e-3
    cosine              1.0000   1.0000   0.99999  0.99999  0.999999

A second, smaller effect is the reduction library. A 2048-term float32 sum is not
associative and numpy and torch do not order it the same way, so ``‖reference‖``
can differ between them by up to about 3e-7 relative even though the raw delta
bytes are identical. The arithmetic here therefore dispatches on the type it is
given: pass torch tensors from inside a hook and it runs in torch, bit-identical
to the historical implementation. Pass a precomputed ``reference_norm`` measured
on the model's own tensor and even the budget matches exactly.

Devices, and two coefficient modes
----------------------------------
Everything converted into the torch backend follows the **reference's device as
well as its dtype**. The directions are stored as numpy, and a direction left on
the CPU while the activation is on CUDA makes ``target + direction`` raise; the
conversion moves it. ``.to()`` returns ``self`` when nothing needs changing, so
the CPU path is not copied and the historical arithmetic is untouched.

The coefficient has two modes, and **only one of them replays RQ2**:

*historical fixed scalar* -- a Python number, 0.15 by default. Detached by
construction, Python-float arithmetic, bit-identical to the historical
implementation. This is the mode that reproduces a recorded RQ2 generation.

*differentiable controller coefficient* -- a scalar torch tensor, with a torch
target. It stays in the autograd graph, so a later controller can learn it.
Gradients are wanted for the coefficient **only**: the frozen direction, the
reference norm and the base model are not in the graph. A coefficient on another
device is moved with a differentiable ``.to()``. Zero is accepted and gives an
exact zero edit; negative and non-finite are refused.

A delta produced in the second mode is **not a historical RQ2 generation** at any
value, not even at 0.15 — numerically equal there, but a new run of a learned
actuator rather than a replay of a frozen one. :meth:`FrozenActuator.coefficient_mode`
returns which mode a coefficient selects, so a run record can say which it was.

numpy is required; torch is not. ``load_frozen_actuator`` returns numpy arrays,
:meth:`FrozenActuator.as_torch` and the torch arithmetic branches import torch
where they use it, and nothing at module scope does.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickletools
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .intervention import historical_layer_to_state
from .validation import (
    Checklist,
    ValidationError,
    assert_config_is_sealed_free,
    assert_not_sealed,
    sha256_file,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ACTUATOR_MANIFEST = REPO_ROOT / "manifests" / "rq2_actuator.json"
DEFAULT_PATHS_FILE = REPO_ROOT / "configs" / "paths.local.json"

#: The key in ``configs/paths.local.json`` that names the migrated directory.
PATHS_KEY = "rq2_actuator_dir"

#: The historical zero-based decoder-block indices, in the archive's layer order.
HISTORICAL_LAYERS: Tuple[int, int, int] = (31, 32, 33)

#: The same three sites in this package's state numbering. Derived, not written
#: out by hand, so the two numbering schemes cannot drift apart in a constant.
ACTUATOR_STATES: Tuple[int, ...] = tuple(
    historical_layer_to_state(layer) for layer in HISTORICAL_LAYERS)

HIDDEN_SIZE = 2048
N_LAYERS = len(HISTORICAL_LAYERS)

#: The frozen strength. Not a tunable: RQ2 froze it, and the two RQ3 static
#: transfers carried it unchanged.
FROZEN_STRENGTH = 0.15

PRIMARY_VARIANT = "consensus23_global_frozen"
COMPARATOR_VARIANT = "historical_d1_global_frozen"

#: The historical site name, and what it is in this package's vocabulary.
HISTORICAL_SITE = "residual_stream"
SITE_KIND = "block_output"
BUDGET_REFERENCE = "block_input"
WINDOW = 1
SLOT = 0

#: Explicit little-endian float32. Never the platform-native dtype: the archive
#: records its own byte order and the reader must honour it rather than assume.
STORAGE_DTYPE = np.dtype("<f4")
BYTES_PER_STORAGE = N_LAYERS * HIDDEN_SIZE * STORAGE_DTYPE.itemsize

#: The freeze's own unit-norm gate (``rq2_freeze_final_method.UNIT_NORM_TOL``).
UNIT_NORM_TOLERANCE = 1e-5

#: Every symbol the historical pickle references. Anything else is refused.
ALLOWED_PICKLE_GLOBALS = frozenset({
    "torch._utils _rebuild_tensor_v2",
    "torch FloatStorage",
    "collections OrderedDict",
})

#: A hard ceiling on what the reader will decompress out of the archive, so a
#: crafted zip cannot be expanded into memory. The real archive is 53 KB stored.
MAX_ARCHIVE_MEMBER_BYTES = 1 << 20

#: ``pathway_probe.EPS``: below this a delta has no direction to scale, and the
#: historical code recorded ``zero_delta`` and applied nothing.
EPS = 1e-8

#: The two ways an edit can be computed, and what each one is for. The first is
#: the default and the only one a generation may be produced by.
APPLICATION_PATH_CANONICAL = "historical_exact"
APPLICATION_PATH_ANALYTICAL = "analytical_approximate"
APPLICATION_PATHS: Dict[str, str] = {
    APPLICATION_PATH_CANONICAL: (
        "the historical operation order: synthetic donor, then donor minus "
        "target, then rescale to the budget. Carries the float32 cancellation "
        "every recorded RQ2 result carries. FrozenActuator.applied_delta"),
    APPLICATION_PATH_ANALYTICAL: (
        "the closed form, unit(direction) * coefficient * ||reference||. "
        "Approximate: it omits the cancellation. Reasoning and plotting only, "
        "never a generation. FrozenActuator.analytical_delta_approx"),
}

#: How the coefficient was supplied. Orthogonal to the application path above,
#: and recorded separately, because only one of these two is a replay of RQ2.
COEFFICIENT_MODE_HISTORICAL = "historical_fixed_scalar"
COEFFICIENT_MODE_CONTROLLER = "differentiable_controller_coefficient"
COEFFICIENT_MODES: Dict[str, str] = {
    COEFFICIENT_MODE_HISTORICAL: (
        "a Python number, 0.15 by default. Detached by construction, and the "
        "only mode that replays RQ2: at 0.15, on the canonical application "
        "path, it reproduces the historical generation bit for bit."),
    COEFFICIENT_MODE_CONTROLLER: (
        "a scalar torch tensor, kept in the autograd graph so a controller can "
        "learn it. NOT a historical RQ2 generation at any value -- not even at "
        "0.15, where it is numerically equal but is a new run of a learned "
        "actuator rather than a replay of a frozen one. Zero is permitted and "
        "yields an exact zero edit; negative and non-finite are refused."),
}


class ActuatorError(ValidationError):
    """The frozen actuator was missing, altered, or asked for something wrong."""


# ------------------------------------------------- backend-neutral arithmetic
# The historical operation ran in torch on float32 tensors. Reproducing it means
# reproducing the *order* of the operations, and -- where a caller is already
# inside a torch hook -- running them in torch, because a 2048-term float32
# reduction is not associative and numpy and torch do not sum in the same order.
#
# These five helpers are the whole abstraction. They dispatch on the type of the
# value they are given, so one transcription of the historical order serves both
# backends and there is no second copy of the arithmetic to drift.
#
# torch is imported lazily, inside the branch that needs it: reading, checking
# and reasoning about the actuator requires numpy only.
def _is_torch_tensor(value: Any) -> bool:
    """True for a torch tensor, without importing torch to find out."""
    return type(value).__module__.split(".")[0] == "torch"


def _as_float32_like(value: Any, reference: Any) -> Any:
    """``value`` as float32, on ``reference``'s device, in its backend.

    **Device, not just dtype.** The stored directions are numpy arrays and the
    reference is whatever the caller holds. Inside a real hook that reference is
    a CUDA activation, and a direction left on the CPU makes the very next line
    -- ``target + direction`` -- raise ``Expected all tensors to be on the same
    device``. Matching only the dtype was the bug; matching both is the fix.

    ``.to(device=..., dtype=...)`` returns ``self`` when nothing needs changing,
    so the CPU float32 path is untouched: no copy, no new tensor, and the
    historical arithmetic downstream is bit-for-bit what it was. It is also
    differentiable, which matters for the coefficient path below.
    """
    if _is_torch_tensor(reference):
        import torch
        if _is_torch_tensor(value):
            return value.to(device=reference.device, dtype=torch.float32)
        # np.array, not asarray: the stored directions are deliberately
        # read-only, and torch.from_numpy warns on a non-writable buffer.
        return torch.from_numpy(np.array(value, dtype=np.float32)).to(
            device=reference.device, dtype=torch.float32)
    if _is_torch_tensor(value):
        return np.ascontiguousarray(value.detach().cpu().numpy(),
                                    dtype=np.float32)
    return np.ascontiguousarray(value, dtype=np.float32)


def _scale(vector: Any, scalar: float) -> Any:
    """Multiply by a scalar, in float32, as each backend's historical caller did.

    torch multiplies a float32 tensor by a Python float and keeps float32, which
    is what ``unit_scaled`` relied on. numpy would promote to float64 against a
    Python float, so the scalar is cast first -- the two then agree bit for bit
    on the multiply itself.
    """
    if _is_torch_tensor(vector):
        return vector * scalar
    return vector * np.float32(scalar)


def _vector_norm_f32(vector: Any) -> float:
    """L2 norm in float32, as ``pathway_probe.vector_norm`` computed it."""
    if _is_torch_tensor(vector):
        import torch
        return float(torch.linalg.vector_norm(vector.float()).item())
    return float(np.linalg.norm(np.asarray(vector, dtype=np.float32), ord=2))


def _checked_vector(value: Any, label: str, hidden_size: int) -> Any:
    """One finite hidden-state vector of the right width, in its own backend."""
    if value is None:
        raise ActuatorError(f"the {label} is required")
    vector = _as_float32_like(value, value)
    shape = tuple(vector.shape)
    if len(shape) != 1:
        raise ActuatorError(
            f"the {label} must be one hidden-state vector, got shape {shape}")
    if shape[0] != hidden_size:
        raise ActuatorError(
            f"the {label} has {shape[0]} hidden unit(s); the actuator has "
            f"{hidden_size}")
    finite = (bool(vector.isfinite().all()) if _is_torch_tensor(vector)
              else bool(np.isfinite(vector).all()))
    if not finite:
        raise ActuatorError(f"the {label} carries a non-finite value")
    return vector


# ------------------------------------------------------------------ locating
def load_actuator_manifest(path: Path | str = DEFAULT_ACTUATOR_MANIFEST
                           ) -> Dict[str, Any]:
    """The committed gate. Read before anything else is opened."""
    path = Path(path)
    if not path.exists():
        raise ActuatorError(
            f"{path} does not exist; it is the manifest that gates the frozen "
            f"actuator and there is no fallback")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert_config_is_sealed_free(manifest, "rq2 actuator manifest")
    return manifest


def actuator_dir(paths: Optional[Mapping[str, str]] = None,
                 paths_path: Path | str = DEFAULT_PATHS_FILE) -> Path:
    """The machine-local directory the freeze was migrated into.

    Resolved through ``configs/paths.local.json`` like every other frozen input:
    the tensor archive is not committed, so the repository cannot resolve it
    from itself.
    """
    if paths is None:
        paths_path = Path(paths_path)
        if not paths_path.exists():
            raise ActuatorError(
                f"{paths_path} does not exist. Copy "
                f"configs/paths.local.example.json to configs/paths.local.json "
                f"and set '{PATHS_KEY}'.")
        paths = json.loads(paths_path.read_text(encoding="utf-8"))
    if PATHS_KEY not in paths:
        raise ActuatorError(
            f"the local path file has no '{PATHS_KEY}' key. The frozen RQ2 "
            f"actuator lives outside the repository; see "
            f"manifests/rq2_actuator.json for what belongs there.")
    directory = Path(assert_not_sealed(paths[PATHS_KEY], "rq2 actuator directory"))
    if not directory.is_dir():
        raise ActuatorError(
            f"{directory} is not a directory. It must hold the ten files listed "
            f"in manifests/rq2_actuator.json, copied byte-for-byte from the "
            f"historical freeze run.")
    return directory


# ------------------------------------------------------------- file integrity
def _expected_file_digests(manifest: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    files: Dict[str, Dict[str, Any]] = {}
    section = manifest["files"]
    for group in ("covered_by_historical_checksum", "the_checksum_file_itself"):
        for name, spec in section[group].items():
            files[name] = spec
    return files


def verify_actuator_files(directory: Path | str,
                          manifest: Optional[Mapping[str, Any]] = None,
                          checklist: Optional[Checklist] = None
                          ) -> Tuple[Dict[str, str], Checklist]:
    """Re-hash every migrated file. Raises on the first disagreement.

    Returns the observed digests and the checklist, so a caller can write the
    evidence into a run record instead of asserting that it checked.
    """
    directory = Path(directory)
    manifest = manifest if manifest is not None else load_actuator_manifest()
    checks = checklist or Checklist("rq2_actuator_files")
    expected = _expected_file_digests(manifest)

    present = {p.name for p in directory.iterdir() if p.is_file()}
    missing = sorted(set(expected) - present)
    if missing:
        raise ActuatorError(
            f"{directory} is missing {len(missing)} of the migrated freeze "
            f"file(s): {missing}. An incomplete freeze is refused, not "
            f"partially loaded.")
    unexpected = sorted(present - set(expected))
    checks.require(
        "directory_holds_only_the_migrated_freeze_files",
        not unexpected,
        f"{len(expected)} expected file(s); {len(unexpected)} unexpected: "
        f"{unexpected}")

    observed: Dict[str, str] = {}
    for name in sorted(expected):
        path = directory / name
        spec = expected[name]
        size = path.stat().st_size
        if size != spec["bytes"]:
            raise ActuatorError(
                f"{name} is {size} bytes; the freeze recorded {spec['bytes']}")
        digest = sha256_file(path)
        if digest != spec["sha256"]:
            raise ActuatorError(
                f"{name} hash mismatch in {directory}\n"
                f"  expected {spec['sha256']}\n"
                f"  actual   {digest}\n"
                f"The migrated copy is not the frozen artefact. Re-copy it from "
                f"the historical freeze run; do not adapt the manifest to it.")
        observed[name] = digest
        checks.record(f"sha256_{name}", True, digest)

    # The historical checksum file is itself a second, independent statement of
    # nine of those digests. Cross-check rather than assume they agree.
    sums = _parse_checksum_file(directory / "FREEZE_SHA256SUMS.txt")
    disagreements = sorted(name for name, want in sums.items()
                           if observed.get(name) != want)
    checks.require(
        "historical_checksum_file_agrees_with_this_manifest",
        not disagreements,
        f"{len(sums)} file(s) listed by FREEZE_SHA256SUMS.txt agree with "
        f"manifests/rq2_actuator.json"
        + (f"; disagreements: {disagreements}" if disagreements else ""))
    return observed, checks


def _parse_checksum_file(path: Path) -> Dict[str, str]:
    """``<sha256>  <path>`` lines, keyed by basename."""
    digests: Dict[str, str] = {}
    for line_number, raw in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ActuatorError(
                f"{path}:{line_number} is not a '<sha256>  <path>' line")
        digests[Path(parts[1]).name] = parts[0]
    return digests


# ------------------------------------------------------------- archive safety
_STRING_OPCODES = frozenset({
    "BINUNICODE", "SHORT_BINUNICODE", "UNICODE", "BINUNICODE8",
    "BINSTRING", "SHORT_BINSTRING", "STRING",
})


def _scan_pickle_without_loading(payload: bytes,
                                 policy: Mapping[str, Any]) -> Dict[str, Any]:
    """Decode ``data.pkl``'s opcodes and refuse anything unexpected.

    ``pickletools.genops`` is a byte-stream decoder: it walks the opcode stream
    and yields what it finds. It constructs no object, imports no module and
    calls nothing the stream names. This function therefore establishes what the
    archive *claims* to be without ever giving it the chance to be anything.

    Two gates:

    * **globals** -- only the three symbols the historical archive references
      are permitted, and ``STACK_GLOBAL`` (whose target is named at run time,
      so it cannot be read statically) is refused outright;
    * **string atoms** -- the distinct strings in the stream must digest to the
      recorded value. That pins the metadata exactly, and is what rules out a
      prompt, a response, or any other unexpected text having been added to the
      archive.
    """
    # The manifest names the allowlist and the code bounds it. A manifest that
    # widened the set would otherwise widen what the archive is permitted to
    # import, which is the one thing this scan exists to constrain.
    allowed = set(policy["allowed_globals"])
    widened = sorted(allowed - ALLOWED_PICKLE_GLOBALS)
    if widened:
        raise ActuatorError(
            f"the manifest permits pickle global(s) {widened} that this loader "
            f"does not: {sorted(ALLOWED_PICKLE_GLOBALS)}. The allowlist is "
            f"bounded by the code, not extended by the manifest.")
    seen_globals: List[str] = []
    atoms: List[str] = []
    try:
        operations = list(pickletools.genops(payload))
    except Exception as exc:  # a malformed stream is a refusal, not a crash
        raise ActuatorError(
            f"the archive's data.pkl is not a decodable pickle stream: {exc}"
        ) from exc

    for opcode, argument, _position in operations:
        if opcode.name == "STACK_GLOBAL":
            raise ActuatorError(
                "the archive's data.pkl uses STACK_GLOBAL, which names its "
                "target at run time. The historical artefact does not, and a "
                "stream whose imports cannot be read statically is refused.")
        if opcode.name == "GLOBAL":
            name = str(argument)
            seen_globals.append(name)
            if name not in allowed:
                raise ActuatorError(
                    f"the archive's data.pkl references {name!r}, which is not "
                    f"one of the symbols the frozen actuator is known to "
                    f"reference ({sorted(allowed)}). Refused.")
        elif opcode.name in _STRING_OPCODES and isinstance(argument, str):
            atoms.append(argument)

    distinct = sorted(set(atoms))
    expected_count = policy.get("expected_string_atoms")
    if expected_count is not None and len(distinct) != int(expected_count):
        raise ActuatorError(
            f"the archive's data.pkl holds {len(distinct)} distinct string "
            f"atom(s); the freeze recorded {expected_count}. The archive's "
            f"metadata is not the frozen metadata.")
    expected_digest = policy.get("expected_string_atoms_sha256")
    if expected_digest is not None:
        digest = hashlib.sha256("\n".join(distinct).encode("utf-8")).hexdigest()
        if digest != expected_digest:
            raise ActuatorError(
                f"the archive's data.pkl metadata digest is {digest}; the "
                f"freeze recorded {expected_digest}. A string was added, "
                f"removed or changed inside the archive.")
    return {"globals": seen_globals, "n_string_atoms": len(distinct)}


def _check_archive_members(archive: zipfile.ZipFile,
                           spec: Mapping[str, Any],
                           checks: Checklist) -> str:
    """Validate the member set, sizes and layout. Returns the archive root."""
    infos = archive.infolist()
    for info in infos:
        name = info.filename
        if name.startswith("/") or ".." in Path(name).parts:
            raise ActuatorError(
                f"the archive member {name!r} escapes its own directory; refused")
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise ActuatorError(
                f"the archive member {name!r} is a symbolic link; refused")
        if info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
            raise ActuatorError(
                f"the archive member {name!r} is {info.file_size} bytes, over "
                f"the {MAX_ARCHIVE_MEMBER_BYTES}-byte ceiling this reader will "
                f"decompress")

    roots = {Path(info.filename).parts[0] for info in infos}
    if len(roots) != 1:
        raise ActuatorError(
            f"the archive has {len(roots)} top-level directories {sorted(roots)}; "
            f"the frozen actuator has exactly one")
    root = roots.pop()
    if root != spec["root"]:
        raise ActuatorError(
            f"the archive root is {root!r}; the freeze recorded {spec['root']!r}")

    observed = sorted(info.filename[len(root) + 1:] for info in infos)
    expected = sorted(spec["expected_members"])
    if observed != expected:
        extra = sorted(set(observed) - set(expected))
        absent = sorted(set(expected) - set(observed))
        raise ActuatorError(
            f"the archive's member set is not the frozen one.\n"
            f"  unexpected: {extra}\n"
            f"  missing:    {absent}")
    checks.record("archive_member_set_is_exact", True, f"{len(observed)} member(s)")

    sizes = {info.filename[len(root) + 1:]: info.file_size for info in infos}
    for member, want in spec["expected_member_bytes"].items():
        if sizes[member] != want:
            raise ActuatorError(
                f"the archive member {member!r} is {sizes[member]} bytes; the "
                f"freeze recorded {want}")
    checks.record("archive_member_sizes_match", True,
                  f"{len(spec['expected_member_bytes'])} member size(s)")
    return root


def _check_archive_metadata(archive: zipfile.ZipFile, root: str,
                            spec: Mapping[str, Any], checks: Checklist) -> None:
    def member(name: str) -> str:
        return archive.read(f"{root}/{name}").decode("ascii").strip()

    byteorder = member("byteorder")
    if byteorder != spec["byteorder"]:
        raise ActuatorError(
            f"the archive declares byte order {byteorder!r}; this reader parses "
            f"{spec['byteorder']!r} and will not silently byte-swap a frozen "
            f"tensor")
    checks.record("archive_byteorder", True, byteorder)

    for name, key in (("version", "torch_serialization_version"),
                      (".format_version", "format_version"),
                      (".storage_alignment", "storage_alignment")):
        value = member(name)
        if value != spec[key]:
            raise ActuatorError(
                f"the archive's {name} is {value!r}; the freeze recorded "
                f"{spec[key]!r}")
        checks.record(f"archive_{name.lstrip('.')}", True, value)


# --------------------------------------------------------------- the tensors
def _read_storages(archive: zipfile.ZipFile, root: str, spec: Mapping[str, Any],
                   tensors: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """Every ``data/N`` member as a read-only ``[3, 2048]`` float32 array."""
    shape = tuple(int(v) for v in tensors["shape"])
    if shape != (N_LAYERS, HIDDEN_SIZE):
        raise ActuatorError(
            f"the manifest declares shape {shape}; this loader requires "
            f"{(N_LAYERS, HIDDEN_SIZE)} -- exactly three layer directions")
    if tensors["numpy_dtype"] != STORAGE_DTYPE.str:
        raise ActuatorError(
            f"the manifest declares dtype {tensors['numpy_dtype']!r}; this "
            f"loader reads {STORAGE_DTYPE.str!r} only")

    storages: Dict[str, np.ndarray] = {}
    for member in sorted(m for m in spec["expected_members"]
                         if m.startswith("data/")):
        raw = archive.read(f"{root}/{member}")
        if len(raw) != BYTES_PER_STORAGE:
            raise ActuatorError(
                f"the archive member {member!r} holds {len(raw)} bytes; a "
                f"{shape} float32 direction is {BYTES_PER_STORAGE}")
        # The dtype is not re-checked afterwards: np.frombuffer with an
        # explicit dtype cannot return another one, the byte count above fixes
        # the element count, and the archive's own byteorder marker has already
        # been required to be little. A dtype-identity check here would be
        # unreachable on a little-endian host and wrong on a big-endian one,
        # where np.float32 is '>f4' and the correctly-read array is '<f4'.
        array = np.frombuffer(raw, dtype=STORAGE_DTYPE).reshape(shape).copy()
        if not bool(np.isfinite(array).all()):
            bad = int((~np.isfinite(array)).sum())
            raise ActuatorError(
                f"{member!r} carries {bad} non-finite value(s); a frozen "
                f"direction with a NaN in it is not usable")
        array.setflags(write=False)
        storages[member] = array
    if not storages:
        raise ActuatorError("the archive holds no data/N storage member")
    return storages


def _layer_digests(array: np.ndarray) -> List[str]:
    """Per-layer SHA256 over the raw float32 bytes, the freeze's own convention."""
    return [hashlib.sha256(np.ascontiguousarray(array[index]).tobytes()).hexdigest()
            for index in range(array.shape[0])]


def _verification_norms(array: np.ndarray) -> Tuple[float, ...]:
    """Per-layer L2 norms in **float64**, as the freeze's own gate computed them.

    Two different precisions are in play and conflating them is a real error, so
    they are separated here rather than left to whichever happens to be
    convenient:

    * **verification** -- ``rq2_freeze_final_method.verify_unit_norm`` cast to
      double before measuring, and the norms it recorded are float64 values. A
      float32 reduction over 2048 units disagrees with them in the seventh
      decimal, which is enough to fail a 1e-9 comparison against the record.
    * **application** -- ``pathway_probe.vector_norm`` measured in float32, and
      every applied delta the historical runs recorded was scaled by a float32
      norm. :meth:`FrozenActuator.applied_delta` therefore stays in float32.

    Using the wrong one in either place would be a silent numerical divergence
    from the historical artefact.
    """
    doubled = np.asarray(array, dtype=np.float64)
    return tuple(float(np.linalg.norm(doubled[index], ord=2))
                 for index in range(doubled.shape[0]))


def direction_fingerprint(array: np.ndarray,
                          layers: Sequence[int] = HISTORICAL_LAYERS) -> str:
    """The historical ``rq2_eval_mean_direction.direction_fingerprint``.

    sha256 over, for each layer in ascending order, ``str(layer)`` as ASCII
    followed by that layer's float32 bytes. Reimplemented rather than imported:
    the point of the migration is that no SteerMoE module is on the import path.
    """
    if array.shape[0] != len(layers):
        raise ActuatorError(
            f"a fingerprint over {len(layers)} layer(s) needs an array with "
            f"{len(layers)} row(s), got shape {array.shape}")
    digest = hashlib.sha256()
    for position, layer in enumerate(sorted(layers)):
        digest.update(str(layer).encode())
        digest.update(np.ascontiguousarray(
            array[position], dtype=STORAGE_DTYPE).tobytes())
    return digest.hexdigest()


def _bind_storages_to_variants(storages: Mapping[str, np.ndarray],
                               frozen_manifest: Mapping[str, Any],
                               layers: Sequence[int],
                               checks: Checklist) -> Dict[str, str]:
    """Match each storage to a variant by content. Must be bijective.

    The archive's entry order is *not* consulted. Each storage's per-layer
    SHA256 is compared against the historical ``frozen_direction_manifest.json``,
    and a variant is accepted only if exactly one storage matches it on all
    three layers.
    """
    digests = {member: _layer_digests(array)
               for member, array in storages.items()}
    binding: Dict[str, str] = {}
    for variant, spec in frozen_manifest["variants"].items():
        want = [spec["per_layer_sha256"][str(layer)] for layer in layers]
        matches = sorted(member for member, got in digests.items() if got == want)
        if len(matches) != 1:
            raise ActuatorError(
                f"{variant} matches {len(matches)} storage member(s) "
                f"{matches} by per-layer SHA256; exactly one is required. The "
                f"archive's contents are not the frozen directions.")
        binding[variant] = matches[0]
    if len(set(binding.values())) != len(binding):
        raise ActuatorError(
            f"two variants bound to the same storage member: {binding}")
    unbound = sorted(set(storages) - set(binding.values()))
    if unbound:
        raise ActuatorError(
            f"the archive holds storage member(s) {unbound} that match no "
            f"frozen variant; refused")
    checks.record("storages_bound_to_variants_by_content", True,
                  "; ".join(f"{variant} <- {member}"
                            for variant, member in sorted(binding.items())))
    return binding


# ----------------------------------------------------------------- the payload
@dataclass(frozen=True)
class FrozenActuator:
    """One frozen direction, with everything needed to apply it as RQ2 did."""

    variant: str
    role: str
    directions: np.ndarray                 # [3, 2048] float32, read-only
    layers: Tuple[int, ...]                # historical decoder blocks
    states: Tuple[int, ...]                # this package's state numbering
    strength: float
    fingerprint: str
    per_layer_sha256: Dict[str, str]
    per_layer_norm: Tuple[float, ...]
    donor_ids_used: Tuple[str, ...]
    n_source_cases: int
    construction: str

    site: str = HISTORICAL_SITE
    site_kind: str = SITE_KIND
    budget_reference: str = BUDGET_REFERENCE
    window: int = WINDOW
    slot: int = SLOT

    #: Which arithmetic :meth:`applied_delta` performs. Fixed at the canonical
    #: path and not a constructor knob: a per-instance switch is exactly how a
    #: caller would end up believing it had asked for the historical operation
    #: and receiving the approximation. The approximation has its own method,
    #: its own name and a mandatory acknowledgement argument.
    application_path: str = APPLICATION_PATH_CANONICAL

    source_directory: Optional[Path] = None
    file_sha256: Dict[str, str] = field(default_factory=dict)
    archive_sha256: str = ""
    checklist: Checklist = field(default_factory=lambda: Checklist("rq2_actuator"))

    # ------------------------------------------------------------ lookups
    @property
    def hidden_size(self) -> int:
        return int(self.directions.shape[1])

    @property
    def state_of_layer(self) -> Dict[int, int]:
        return dict(zip(self.layers, self.states))

    def direction_for_layer(self, block_index: int) -> np.ndarray:
        """The direction for a historical zero-based decoder block."""
        if block_index not in self.layers:
            raise ActuatorError(
                f"the frozen actuator covers decoder blocks {list(self.layers)}; "
                f"block {block_index} is not one of them. It is a three-layer "
                f"actuator, not a per-layer bank.")
        return self.directions[self.layers.index(block_index)]

    def direction_for_state(self, state: int) -> np.ndarray:
        """The direction for a state in this package's numbering."""
        if state not in self.states:
            raise ActuatorError(
                f"the frozen actuator covers states {list(self.states)} "
                f"(historical decoder blocks {list(self.layers)}); state "
                f"{state} is not one of them.")
        return self.directions[self.states.index(state)]

    def as_torch(self) -> Any:
        """A ``[3, 2048]`` float32 CPU tensor, if torch is installed.

        Deliberately a method and not the return type: reading, checking and
        reasoning about the actuator needs numpy only, and this package does not
        take a deep-learning runtime as a hard dependency.
        """
        try:
            import torch
        except Exception as exc:  # pragma: no cover - torch is present in-container
            raise ActuatorError(
                f"as_torch() needs torch, which failed to import: {exc}") from exc
        return torch.from_numpy(np.array(self.directions, dtype=np.float32))

    # -------------------------------------------------------- the arithmetic
    def budget(self, reference_norm: float,
               coefficient: Optional[float] = None) -> float:
        """``coefficient * ||reference||`` -- the target-relative L2 budget.

        ``reference_norm`` is the L2 norm, in float32, of the patched block's
        **input** hidden state at the final prompt token -- not of the block
        output being edited. Passing the wrong one silently rescales every edit.

        ``coefficient`` defaults to the frozen strength. Any positive value is
        accepted: a per-prompt coefficient is exactly what a later controller
        will supply, and the operation order below does not depend on its value.
        """
        if _is_torch_tensor(coefficient):
            raise ActuatorError(
                "budget() is the historical fixed-scalar form and takes a "
                "Python number. A torch coefficient belongs to the "
                "differentiable controller path; pass it to applied_delta(), "
                "which keeps it in the autograd graph.")
        coefficient = self.strength if coefficient is None else float(coefficient)
        if not math.isfinite(reference_norm) or reference_norm < 0.0:
            raise ActuatorError(
                f"the budget reference norm must be finite and non-negative, "
                f"got {reference_norm!r}")
        if not math.isfinite(coefficient) or coefficient <= 0.0:
            raise ActuatorError(
                f"the coefficient must be finite and positive, got "
                f"{coefficient!r}")
        return coefficient * float(reference_norm)

    def _tensor_coefficient(self, coefficient: Any, target: Any) -> Any:
        """A scalar torch coefficient, validated, on the target's device.

        **The returned tensor is still the caller's, in the autograd graph.**
        Every check reads ``coefficient.detach()`` -- a view that shares storage
        but is outside the graph -- so validation cannot become the thing the
        gradient flows through. Only ``.to()`` and ``.reshape()`` are applied to
        the original, and both are differentiable and both return ``self`` when
        nothing needs changing.

        **Device.** If the coefficient is on another device it is moved with a
        differentiable ``.to(device=...)``, whose backward pass moves the
        gradient back. A controller emitting on CPU while the model runs on CUDA
        therefore works, and still trains.

        **Sign.** Zero is permitted here and produces an exact zero edit, which
        is how a controller says "do not edit this prompt". Negative and
        non-finite are refused: a negative coefficient would apply the direction
        backwards, towards refusal, which is not a smaller edit but an opposite
        one. The historical fixed-scalar path in :meth:`budget` keeps its
        stricter rule and still refuses zero, because a zero-strength historical
        generation is the baseline rather than an edit.
        """
        import torch

        if not _is_torch_tensor(target):
            raise ActuatorError(
                "a torch coefficient needs a torch target. The differentiable "
                "controller path exists to be applied to a model activation, "
                "so it is offered by applied_delta() with a torch target and "
                "nowhere else; a numpy target, or the analytical helper, has "
                "no autograd graph for the coefficient to join.")
        probe = coefficient.detach()          # inspection only, never returned
        if probe.numel() != 1:
            raise ActuatorError(
                f"the coefficient must be a single scalar value, got a tensor "
                f"of shape {tuple(probe.shape)} with {probe.numel()} element(s)")
        if not bool(torch.isfinite(probe).all()):
            raise ActuatorError(
                f"the coefficient must be finite, got {probe.reshape(()).item()!r}")
        if float(probe.reshape(()).item()) < 0.0:
            raise ActuatorError(
                f"the coefficient must not be negative, got "
                f"{probe.reshape(()).item()!r}; a negative coefficient applies "
                f"the repair direction backwards rather than more weakly")
        moved = coefficient
        if moved.device != target.device:
            moved = moved.to(device=target.device)      # differentiable
        return moved.reshape(()).to(dtype=torch.float32)

    def _resolve_budget(self, reference_vector: Any,
                        reference_norm: Optional[float],
                        coefficient: Optional[Any],
                        target: Any = None) -> Any:
        """One of the two reference forms, never both, never neither.

        A caller inside a torch hook already holds the block-input tensor the
        model produced and should pass ``reference_norm`` measured on it, which
        is what makes the result bit-identical to a historical generation.
        Offline callers pass the vector and accept a float32 reduction that may
        differ from torch's in its last bits; see the module docstring.

        Returns a Python float for the historical fixed-scalar path, or a 0-dim
        torch tensor for the differentiable controller path. The reference norm
        is a Python float either way: **gradients are wanted for the coefficient
        only**, and the norm of a frozen model's own activation is not something
        this stage learns.
        """
        if (reference_vector is None) == (reference_norm is None):
            raise ActuatorError(
                "supply exactly one of reference_vector or reference_norm: the "
                "budget is read from the patched block's input hidden state, "
                "either as the vector or as its precomputed float32 norm")
        if reference_norm is None:
            reference_norm = _vector_norm_f32(
                _checked_vector(reference_vector, "budget reference",
                                self.hidden_size))
        reference_norm = float(reference_norm)
        if not math.isfinite(reference_norm) or reference_norm < 0.0:
            raise ActuatorError(
                f"the budget reference norm must be finite and non-negative, "
                f"got {reference_norm!r}")

        if _is_torch_tensor(coefficient):
            return self._tensor_coefficient(coefficient, target) * reference_norm
        return self.budget(reference_norm, coefficient)

    def coefficient_mode(self, coefficient: Optional[Any] = None) -> str:
        """Which of the two coefficient modes a given coefficient selects.

        A run record must say which one produced it, because only one of them is
        a replay of RQ2. Returns one of :data:`COEFFICIENT_MODES`.
        """
        if _is_torch_tensor(coefficient):
            return COEFFICIENT_MODE_CONTROLLER
        return COEFFICIENT_MODE_HISTORICAL

    def applied_delta(self, block_index: int, target_vector: Any,
                      reference_vector: Any = None, *,
                      reference_norm: Optional[float] = None,
                      coefficient: Optional[float] = None) -> Any:
        """**The canonical path.** The historical operation, in its own order.

        This is what RQ2 executed, transcribed rather than simplified::

            donor      = target + direction          # make_synthetic_donor
            raw_delta  = donor - target              # build_site_plan
            budget     = coefficient * ||reference||
            applied    = raw_delta * (budget / ||raw_delta||)   # unit_scaled

        Algebraically the first two lines are the identity and ``raw_delta`` is
        the direction. In float32 they are not: ``fl(target + direction) −
        target`` discards low-order bits of the direction in proportion to
        ``‖target‖``, and the result is the frozen direction perturbed by that
        cancellation. Every number RQ2 and the two static transfers recorded
        carries that perturbation, so **reproducing it is the point**, and this
        method is the default for applying the actuator.

        The order is preserved **for any coefficient**, not only the frozen
        0.15: the coefficient enters once, in the budget, exactly where the
        historical code put it.

        Accepts numpy arrays or torch tensors and returns the same kind, on the
        target's own device. Given torch tensors and a ``reference_norm``, the
        result is bit-identical to the historical implementation. See
        :meth:`analytical_delta_approx` for the closed-form alternative, which
        is **not** exact.

        **Two coefficient modes, and only one of them replays RQ2.**

        *Historical fixed scalar* -- ``coefficient`` is a Python number, or
        omitted for the frozen 0.15. Detached by construction, Python-float
        arithmetic throughout, bit-identical to the historical implementation.
        This is the mode that reproduces a recorded RQ2 generation.

        *Differentiable controller coefficient* -- ``coefficient`` is a scalar
        torch tensor and the target is a torch tensor. The coefficient stays in
        the autograd graph, so ``backward()`` reaches it and a controller can
        learn it. The frozen direction, the reference norm and the base model
        are **not** in the graph and are not learned by this.

        A delta produced in the second mode is **not a historical RQ2
        generation**, at any value of the coefficient — not even at 0.15, where
        it is numerically equal. It is a new run of a learned actuator, and a
        run record must say so: :meth:`coefficient_mode` returns which mode a
        given coefficient selects, and :data:`COEFFICIENT_MODES` says what each
        one means.

        Zero is accepted in the tensor mode and produces an exact zero edit —
        how a controller declines to act. Negative and non-finite coefficients
        are refused in both modes.
        """
        target = _checked_vector(target_vector, "target state vector",
                                 self.hidden_size)
        budget = self._resolve_budget(reference_vector, reference_norm,
                                      coefficient, target)
        direction = _as_float32_like(self.direction_for_layer(block_index),
                                     target)

        donor = target + direction                   # make_synthetic_donor
        raw_delta = donor - target                   # build_site_plan
        norm = _vector_norm_f32(raw_delta)           # pathway_probe.vector_norm
        if not math.isfinite(norm) or norm < EPS:
            raise ActuatorError(
                f"the recovered direction for block {block_index} has norm "
                f"{norm}; the synthetic donor cancelled it entirely, which "
                f"means the target vector's magnitude is far outside anything "
                f"this model produces. The historical code recorded this as "
                f"'zero_delta' and applied nothing.")
        return _scale(raw_delta, budget / norm)      # unit_scaled

    def analytical_delta_approx(self, block_index: int,
                                reference_vector: Any = None, *,
                                reference_norm: Optional[float] = None,
                                coefficient: Optional[float] = None,
                                acknowledge_not_exact: bool = False) -> Any:
        """**Not exact, and not the runtime path.** The closed form only.

        ``unit(direction) * coefficient * ||reference||`` -- the operation as
        documented, with the synthetic donor removed. It is the right thing for
        reasoning, plotting and sanity arithmetic, and **never a generation**:
        it does not carry the float32 cancellation that every recorded RQ2
        result carries, so a run produced this way would not be the historical
        operation however closely it resembled it.

        Measured divergence from :meth:`applied_delta`, one layer, float32
        throughout, as a function of the target's element scale:

        =================  ==================
        target element ~   relative difference
        =================  ==================
        1e-1               1.2e-07
        1e+0               1.1e-06
        1e+1               1.2e-05
        1e+2               1.1e-04
        1e+3               1.1e-03
        =================  ==================

        It grows linearly with ``‖target‖`` because that is what drives the
        cancellation. Direction is preserved almost exactly -- the cosine stays
        above 0.9999994 across that whole range -- so the difference is a small
        rotation, not a different edit. Small is not zero, and a run record that
        claims to reproduce a historical generation must not be produced this
        way.

        ``acknowledge_not_exact=True`` is mandatory. It exists so that reaching
        this method is always a deliberate act: no default, no positional
        argument and no signature it shares with :meth:`applied_delta` can
        route a caller here by accident.
        """
        if acknowledge_not_exact is not True:
            raise ActuatorError(
                "analytical_delta_approx() is an approximation and is not the "
                "path any generation may be produced by. It omits the float32 "
                "cancellation of the historical synthetic-donor route, which "
                "every recorded RQ2 result carries. Use applied_delta() -- the "
                "canonical path -- or pass acknowledge_not_exact=True if an "
                "approximation is genuinely what you want.")
        budget = self._resolve_budget(reference_vector, reference_norm,
                                      coefficient)
        direction = self.direction_for_layer(block_index)
        norm = _vector_norm_f32(direction)
        if not math.isfinite(norm) or norm < EPS:
            raise ActuatorError(
                f"the direction for block {block_index} has norm {norm}")
        return _scale(_as_float32_like(direction, direction),
                      budget / norm)

    def provenance(self) -> Dict[str, Any]:
        """What a run record must carry to identify which actuator it used."""
        return {
            "variant": self.variant,
            "role": self.role,
            "fingerprint": self.fingerprint,
            "per_layer_sha256": dict(self.per_layer_sha256),
            "archive_sha256": self.archive_sha256,
            "historical_layers": list(self.layers),
            "routegate_states": list(self.states),
            "strength": self.strength,
            "site": self.site,
            "site_kind": self.site_kind,
            "budget_reference": self.budget_reference,
            "window": self.window,
            "slot": self.slot,
            "n_source_cases": self.n_source_cases,
            "donor_ids_used": list(self.donor_ids_used),
            "application_path": self.application_path,
            "application_path_note": APPLICATION_PATHS[self.application_path],
            # The mode is a property of a *call*, not of the actuator, so this
            # is the default only. A runner must record
            # coefficient_mode(coefficient) for the coefficient it actually
            # used: a tensor coefficient is not a replay of RQ2.
            "coefficient_mode_default": COEFFICIENT_MODE_HISTORICAL,
            "coefficient_mode_default_note": (
                COEFFICIENT_MODES[COEFFICIENT_MODE_HISTORICAL]),
        }


# -------------------------------------------------------------------- loading
def load_frozen_actuator(variant: str = PRIMARY_VARIANT,
                         directory: Optional[Path | str] = None,
                         manifest: Optional[Mapping[str, Any]] = None,
                         paths: Optional[Mapping[str, str]] = None,
                         paths_path: Path | str = DEFAULT_PATHS_FILE,
                         manifest_path: Path | str = DEFAULT_ACTUATOR_MANIFEST,
                         ) -> FrozenActuator:
    """Verify the whole migrated freeze, then read one direction out of it.

    Neither ``pickle`` nor ``torch.load`` is called. See the module docstring
    for the order the gates run in; the short version is that every hash is
    checked before the archive is opened, and the archive is checked before a
    byte of tensor data is read.
    """
    manifest = (manifest if manifest is not None
                else load_actuator_manifest(manifest_path))
    directory = (Path(directory) if directory is not None
                 else actuator_dir(paths, paths_path))

    checks = Checklist("rq2_actuator")
    file_digests, checks = verify_actuator_files(directory, manifest, checks)

    frozen_manifest = json.loads(
        (directory / "frozen_direction_manifest.json").read_text(encoding="utf-8"))
    variants = manifest["variants"]
    if variant not in frozen_manifest["variants"]:
        raise ActuatorError(
            f"unknown variant {variant!r}; the freeze holds "
            f"{sorted(frozen_manifest['variants'])}")
    if variant not in variants:
        raise ActuatorError(
            f"{variant!r} is in the historical freeze but not in "
            f"manifests/rq2_actuator.json; the RouteGate manifest is the gate "
            f"and is not extended at run time")

    layers = tuple(int(layer) for layer in frozen_manifest["layers"])
    if layers != HISTORICAL_LAYERS:
        raise ActuatorError(
            f"the freeze covers decoder blocks {list(layers)}; this loader "
            f"preserves the historical {list(HISTORICAL_LAYERS)} and does not "
            f"remap them")
    strength = float(frozen_manifest["strength"])
    if strength != FROZEN_STRENGTH:
        raise ActuatorError(
            f"the freeze records strength {strength}; this loader preserves "
            f"{FROZEN_STRENGTH}. Strength 0.2 belongs to the separate "
            f"population scan and is not this actuator.")
    checks.record("frozen_strength", True, str(strength))
    checks.record("historical_layers", True, str(list(layers)))
    checks.record("routegate_states", True, str(list(ACTUATOR_STATES)))

    archive_spec = manifest["archive"]
    tensor_spec = manifest["tensors"]
    archive_path = directory / archive_spec["filename"]
    with zipfile.ZipFile(archive_path) as archive:
        root = _check_archive_members(archive, archive_spec, checks)
        _check_archive_metadata(archive, root, archive_spec, checks)
        scan = _scan_pickle_without_loading(
            archive.read(f"{root}/data.pkl"),
            archive_spec["pickle_policy"])
        checks.record("pickle_globals_are_the_expected_three", True,
                      "; ".join(sorted(set(scan["globals"]))))
        checks.record("pickle_metadata_atoms_match_the_freeze", True,
                      f"{scan['n_string_atoms']} distinct string atom(s), "
                      f"digest "
                      f"{archive_spec['pickle_policy']['expected_string_atoms_sha256']}")
        storages = _read_storages(archive, root, archive_spec, tensor_spec)
        binding = _bind_storages_to_variants(storages, frozen_manifest, layers,
                                             checks)

    expected_member = variants[variant].get("expected_archive_member")
    if expected_member is not None and binding[variant] != expected_member:
        raise ActuatorError(
            f"{variant} was bound by content to {binding[variant]!r}, but "
            f"manifests/rq2_actuator.json records {expected_member!r}. The "
            f"content binding is authoritative; this disagreement means the "
            f"manifest and the archive describe different artefacts.")

    array = storages[binding[variant]]
    frozen_spec = frozen_manifest["variants"][variant]
    local_spec = variants[variant]

    fingerprint = direction_fingerprint(array, layers)
    for label, want in (("historical freeze manifest", frozen_spec["fingerprint"]),
                        ("manifests/rq2_actuator.json", local_spec["fingerprint"])):
        if fingerprint != want:
            raise ActuatorError(
                f"{variant} fingerprint mismatch against the {label}\n"
                f"  expected {want}\n"
                f"  actual   {fingerprint}")
    checks.record(f"fingerprint_{variant}", True, fingerprint)

    per_layer = _layer_digests(array)
    for position, layer in enumerate(layers):
        want = local_spec["per_layer_sha256"][str(layer)]
        if per_layer[position] != want:
            raise ActuatorError(
                f"{variant} layer {layer} digest mismatch\n"
                f"  expected {want}\n  actual   {per_layer[position]}")

    norms = _verification_norms(array)
    # As with the pickle allowlist: the manifest states the tolerance and the
    # code bounds it, so an edited manifest can tighten the gate but never
    # loosen it below what the freeze itself gated on.
    tolerance = float(tensor_spec["unit_norm_tolerance"])
    if tolerance > UNIT_NORM_TOLERANCE:
        raise ActuatorError(
            f"the manifest allows a unit-norm tolerance of {tolerance}, looser "
            f"than the {UNIT_NORM_TOLERANCE} the freeze gated on. The tolerance "
            f"is bounded by the code, not relaxed by the manifest.")
    off = [f"layer {layer}: {norm!r}"
           for layer, norm in zip(layers, norms)
           if abs(norm - 1.0) > tolerance]
    if off:
        raise ActuatorError(
            f"{variant} is not unit norm within {tolerance}: {off}. The freeze "
            f"gated on exactly this tolerance before writing the file.")
    recorded_norms = [float(v) for v in local_spec["per_layer_norm"]]
    drift = [f"layer {layer}: recorded {want!r}, recomputed {got!r}"
             for layer, want, got in zip(layers, recorded_norms, norms)
             if abs(want - got) > 1e-9]
    if drift:
        raise ActuatorError(
            f"{variant} per-layer norms disagree with the freeze's own record: "
            f"{drift}")
    checks.record(f"unit_norm_{variant}", True,
                  ", ".join(f"{norm:.12f}" for norm in norms))

    return FrozenActuator(
        variant=variant,
        role=frozen_spec["role"],
        directions=array,
        layers=layers,
        states=tuple(historical_layer_to_state(layer) for layer in layers),
        strength=strength,
        fingerprint=fingerprint,
        per_layer_sha256={str(layer): per_layer[position]
                          for position, layer in enumerate(layers)},
        per_layer_norm=norms,
        donor_ids_used=tuple(frozen_spec["donor_ids_used"]),
        n_source_cases=len(frozen_spec["source_case_ids"]),
        construction=frozen_spec["construction"],
        source_directory=directory,
        file_sha256=file_digests,
        archive_sha256=file_digests[archive_spec["filename"]],
        checklist=checks,
    )
