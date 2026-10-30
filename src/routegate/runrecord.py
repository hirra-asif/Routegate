"""Writing a run record.

A run directory has to be auditable without rerunning it, which means the small
mechanical parts -- what the environment was, what the outputs hash to, how a
table is written -- should not be reinvented per experiment.

The pilot reproduction is deliberately *not* refactored onto this module. Its
outputs are a frozen reproduction record, and the code that produced them is
kept exactly as it ran; a shared helper introduced afterwards would make that
record depend on code it never executed.
"""
from __future__ import annotations

import csv
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from .validation import sha256_file


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_csv(path: Path | str, fieldnames: Sequence[str],
              rows: Iterable[Mapping[str, Any]]) -> int:
    """Write a table, and return how many rows it holds."""
    written = 0
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            written += 1
    return written


def write_json(path: Path | str, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


#: What an offline tool truthfully says about itself. The default, because
#: most callers here read arrays and write JSON.
OFFLINE_NOTE = ("offline analysis only; no language model, no generation, "
                "no held-out benchmark material")

#: What a run that actually loaded a model and decoded from it says. The
#: network claim is unchanged and still true: the weights are read from a
#: local snapshot and no external service is called.
GENERATION_NOTE = ("a language model was loaded locally and generation was "
                   "performed; no external service was called and no held-out "
                   "benchmark material was read")


def environment_record(package_version: str, *,
                       model_loaded: bool = False,
                       generation_performed: bool = False,
                       note: Optional[str] = None) -> Dict[str, Any]:
    """The numerical stack a run executed on, and what the run actually did.

    The defaults describe an offline analysis tool, which is what almost every
    caller in this package is. A runner that loads a model and decodes from it
    passes ``model_loaded=True, generation_performed=True`` -- otherwise its
    own record would state, in its own words, that it did neither. That is not
    a cosmetic difference: the environment block is what a reader consults to
    know whether a number came from arithmetic over stored activations or from
    a model that was actually run.

    ``network_calls`` stays 0 in both cases and is not a parameter. The weights
    are read from a local snapshot with ``local_files_only=True``; loading a
    model is not a network call, and no caller in this package makes one.
    """
    import numpy as np
    import scipy
    import sklearn

    if note is None:
        note = (GENERATION_NOTE if (model_loaded or generation_performed)
                else OFFLINE_NOTE)
    return {
        "package": "routegate",
        "package_version": package_version,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "executable": sys.executable,
        "model_loaded": bool(model_loaded),
        "generation_performed": bool(generation_performed),
        "network_calls": 0,
        "note": note,
    }


def write_output_hashes(output_dir: Path | str,
                        filename: str = "output_hashes.json",
                        skip: Optional[Sequence[str]] = None) -> Dict[str, str]:
    """SHA256 every file in a run directory, so the record can be verified."""
    output_dir = Path(output_dir)
    skip = set(skip or ()) | {filename}
    digests: Dict[str, str] = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name not in skip:
            digests[str(path.relative_to(output_dir))] = sha256_file(path)
    write_json(output_dir / filename,
               {"directory": str(output_dir), "files": digests})
    return digests
