"""Input gates.

Nothing in this package reads a byte of scientific data before passing through
here. The rules are deliberately blunt: an input is either the exact artefact
recorded in ``manifests/development_inputs.json`` or it is refused. There is no
"close enough", no warning-and-continue, and no automatic fallback to a
different file in the same directory.

Three separate guards live here.

``assert_not_stale``
    The historical workspace kept several generations of the labelled dataset
    side by side, and the superseded ones carry the more authoritative-sounding
    names. They are rejected by filename, before any hash is computed, so a
    typo cannot silently reintroduce a two-revision-old label set.

``assert_not_sealed``
    The held-out benchmark split is not available to development code. This
    guard matches on generic tokens ("sealed", "final_benchmark", ...) rather
    than on any specific historical path, so the guard itself does not encode
    the location of the material it protects.

``Checklist``
    An append-only record of every gate that was evaluated, so a run directory
    can be audited without rerunning it.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class ValidationError(RuntimeError):
    """An input failed a gate. The caller must stop, not adapt."""


class StaleInputError(ValidationError):
    """A superseded artefact was offered where the accepted one is required."""


class SealedSetError(ValidationError):
    """Development code touched something that belongs to the held-out split."""


# --------------------------------------------------------------- stale inputs
# Superseded dataset revisions from the historical workspace. These carry
# pre-re-audit labels (E=114/B=162/Excluded=20 or E=116/B=180) and must never
# reach an analysis.
STALE_FILENAMES: frozenset = frozenset({
    "rq3_dataset_final_before_e_axis2_reaudit.csv",
    "rq3_ebh_final_analysis_dataset.csv",
    "rq3_ebh_final_dataset_summary.json",
    "rq3_ebh_final_dataset_summary.md",
})

# Any artefact from the superseded geometry-preparation package: its frozen
# folds and balanced subset were built on the original, pre-adjudication
# labels.
STALE_DIRECTORY_TOKENS: tuple = (
    "rq3_geometry_prepare_20260803_215218",
)


def assert_not_stale(path: Path | str) -> Path:
    """Refuse a superseded input by name, before it is opened."""
    path = Path(path)
    if path.name in STALE_FILENAMES:
        raise StaleInputError(
            f"{path.name} is a superseded dataset revision and is refused. "
            f"The accepted dataset is the one recorded in "
            f"manifests/development_inputs.json.")
    lowered = str(path).lower()
    for token in STALE_DIRECTORY_TOKENS:
        if token in lowered:
            raise StaleInputError(
                f"{path} lies inside the superseded package '{token}', whose "
                f"labels and folds predate the accepted dataset. Refused.")
    return path


# --------------------------------------------------------------- sealed split
# Generic tokens only. This package must not know, or be able to reconstruct,
# where the held-out material lives.
SEALED_PATH_TOKENS: tuple = (
    "sealed",
    "final_benchmark",
    "benchmark_test",
    "holdout_benchmark",
    "held_out_benchmark",
)

_SEALED_PATTERN = re.compile("|".join(re.escape(t) for t in SEALED_PATH_TOKENS),
                             re.IGNORECASE)


def looks_sealed(value: Any) -> Optional[str]:
    """Return the offending token if ``value`` reads like held-out material."""
    match = _SEALED_PATTERN.search(str(value))
    return match.group(0) if match else None


def assert_not_sealed(value: Any, context: str = "input") -> Any:
    """Refuse anything whose name suggests the held-out benchmark split."""
    token = looks_sealed(value)
    if token is not None:
        raise SealedSetError(
            f"{context} refers to '{token}' ({value!r}). The held-out benchmark "
            f"split is not available to development code; see "
            f"manifests/sealed_set_commitment.json.")
    return value


def assert_config_is_sealed_free(config: Any, context: str = "configuration") -> None:
    """Walk a nested config and refuse any sealed-looking key or value."""
    def walk(node: Any, trail: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                assert_not_sealed(key, f"{context} key {trail}{key}")
                walk(value, f"{trail}{key}.")
        elif isinstance(node, (list, tuple)):
            for i, value in enumerate(node):
                walk(value, f"{trail}[{i}].")
        elif isinstance(node, (str, Path)):
            assert_not_sealed(node, f"{context} value at {trail.rstrip('.')}")

    walk(config, "")


# -------------------------------------------------------------------- hashing
def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assert_hash(path: Path | str, expected: str, label: str) -> str:
    """Refuse a file whose contents are not exactly what was recorded."""
    actual = sha256_file(path)
    if actual != expected:
        raise ValidationError(
            f"{label} hash mismatch for {path}\n"
            f"  expected {expected}\n"
            f"  actual   {actual}\n"
            f"The accepted artefact has changed or the wrong file was supplied.")
    return actual


# ----------------------------------------------------------------- checklist
@dataclass
class Checklist:
    """Append-only record of the gates a run evaluated."""

    name: str
    entries: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, check: str, passed: bool, detail: str) -> None:
        self.entries.append({"check": check,
                             "status": "PASS" if passed else "FAIL",
                             "detail": detail})

    def require(self, check: str, passed: bool, detail: str) -> None:
        """Record the gate and stop the run if it did not pass."""
        self.record(check, passed, detail)
        if not passed:
            raise ValidationError(f"{check}: {detail}")

    @property
    def all_passed(self) -> bool:
        return all(e["status"] == "PASS" for e in self.entries)

    @property
    def failures(self) -> List[Dict[str, Any]]:
        return [e for e in self.entries if e["status"] != "PASS"]

    def as_dict(self) -> Dict[str, Any]:
        return {"checklist": self.name,
                "checks_run": len(self.entries),
                "all_passed": self.all_passed,
                "checks": list(self.entries)}


def require_finite(array, label: str) -> None:
    """Refuse non-finite values anywhere in an array."""
    import numpy as np
    finite = np.isfinite(array)
    if not bool(finite.all()):
        bad = int((~finite).sum())
        raise ValidationError(f"{label} contains {bad} non-finite value(s)")


def safe_argmax(values: Sequence[float], label: str = "values") -> int:
    """``argmax`` that refuses to let a non-finite entry win.

    ``np.argmax`` on an array containing NaN returns the NaN position, which
    would silently make a degenerate state the "best" one. This raises instead.
    """
    import numpy as np
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValidationError(f"{label} is empty; no maximum exists")
    if not bool(np.isfinite(array).all()):
        bad = [int(i) for i in np.flatnonzero(~np.isfinite(array))]
        raise ValidationError(
            f"{label} has non-finite entries at positions {bad}; refusing to "
            f"take a maximum over them")
    return int(np.argmax(array))


def summarise_paths(paths: Iterable[Path | str]) -> str:
    return ", ".join(str(Path(p).name) for p in paths)


# ------------------------------------------------- extraction output state
def routing_output_path_state(actual: Path | str,
                              expected: Path | str,
                              outputs_root: Path | str) -> Tuple[bool, str]:
    """Whether one declared extraction output path is in a safe state.

    Phase-independent, and that is the whole point of it. The predicate is
    evaluated before the extraction has ever run and again after a legitimate
    run, and it must hold both times. Exactly two states are safe:

    * the path is **absent** -- nothing has been extracted yet;
    * the path is a **plain directory at its exact expected location** -- an
      extraction has completed and wrote where it said it would.

    Everything else is refused: a regular file where a directory belongs, a
    symlink of any kind (which could redirect a write out of the repository),
    and any location other than the expected one under the repository's
    ``outputs`` directory.

    This replaces an earlier ``neither_dir_exists_yet`` check, which asserted
    that both directories were absent. That condition was only ever satisfiable
    before the first extraction, so it failed permanently -- and uninformatively
    -- once the extraction had legitimately run.

    Returns ``(ok, description)``. The description always reports what was
    actually found, so a validation record can never assert a state that was not
    observed.
    """
    actual, expected = Path(actual), Path(expected)
    outputs_root = Path(outputs_root)
    name = actual.name

    # Checked before resolution: resolve() would silently follow the link and
    # report the state of its target instead.
    if actual.is_symlink():
        try:
            target = os.readlink(actual)
        except OSError:                                  # pragma: no cover
            target = "unreadable"
        return False, f"{name}: REFUSED, symlink -> {target}"

    # resolve() is non-strict, so an absent path still normalises. It is applied
    # to both sides so a symlinked *parent* of the repository root -- e.g. a
    # home directory reached through two mount points -- does not read as a
    # relocation of the output directory itself.
    here, want, root = actual.resolve(), expected.resolve(), outputs_root.resolve()

    if root not in here.parents:
        return False, f"{name}: REFUSED, {here} is outside {root}"
    if here != want:
        return False, f"{name}: REFUSED, at {here}, expected {want}"
    if not actual.exists():
        return True, f"{name}: absent"
    if actual.is_dir():
        n = sum(1 for _ in actual.iterdir())
        return True, f"{name}: directory, {n} entr{'y' if n == 1 else 'ies'}"
    return False, f"{name}: REFUSED, exists but is not a directory"
