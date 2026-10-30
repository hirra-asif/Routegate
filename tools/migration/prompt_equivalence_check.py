#!/usr/bin/env python3
"""Phase 2 step 9 -- the one-time prompt-equivalence test.

The gate on step 10. ``experiments/routegate_routing/extract_routing_v1.py``
still imports the historical generation module for its model, tokeniser and
chat-template configuration. Cutting that import over to ``routegate.runtime``
changes a GPU code path that cannot be validated by re-running it here, so this
is the offline proxy: render and tokenise all 480 development prompts through
both implementations with the real pinned tokeniser and require them to be
identical, byte for byte and token for token.

Rendering is where a silent drift would be invisible and fatal. A different
template, a stray BOS, a changed ``enable_thinking`` default -- and every
captured activation lands at a different token while everything still appears to
work. Comparing rendered text alone is not enough: two different token sequences
can decode to the same string, and it is the token sequence the model sees.

Seven checks, all of which must pass:

1. the pinned model identifier agrees;
2. the pinned revision agrees;
3. both ``resolve_snapshot()`` implementations resolve the same local snapshot
   at the exact pinned commit;
4. ``tokenizer_config.json`` hashes to the same expected SHA256 on both sides;
5. the chat-template string hashes to the same expected SHA256 on both sides;
6. all 480 rendered prompts are identical as UTF-8 bytes, with identical SHA256;
7. all 480 tokenised sequences are identical -- full ``input_ids``, attention
   mask and length, and the final prompt token.

480/480 on every per-prompt comparison. One mismatch is a hard failure, and the
correct response to a failure is to investigate it, never to adapt either
implementation until it passes.

What this program will not do: load a model, touch a GPU, generate a response,
open a socket, write anything into the historical workspace, or read anything
belonging to the held-out split. The tokeniser is loaded from the local cache
only, and ``AutoModelForCausalLM`` is never instantiated.

Run it with explicit paths -- no machine path is written down in this file:

    python tools/migration/prompt_equivalence_check.py \\
        --legacy-module /path/to/rq3_dev_generation.py \\
        --manifest /path/to/manifest.csv
"""
from __future__ import annotations

# Offline before anything can import huggingface_hub or transformers: these are
# read at import time, so setting them later would be decorative. Assigned, not
# setdefault -- an inherited HF_HUB_OFFLINE=0 must not be able to put this stage
# back on the network.
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
import platform
import re
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate.validation import (            # noqa: E402
    ValidationError,
    assert_not_sealed,
    sha256_file,
    sha256_text,
)

LEGACY_MODULE_NAME = "legacy_rq3_dev_generation"
DEFAULT_INPUT_MANIFEST = REPO_ROOT / "manifests" / "development_inputs.json"
DEFAULT_EVIDENCE = (REPO_ROOT / "artifacts" / "prompt_equivalence"
                    / "prompt_equivalence_v1.json")

EXPECTED_ROWS = 480
EXPECTED_COUNTS = {"E": 89, "B": 178, "H": 213}
EXPECTED_COLUMNS = ("row_index", "row_id", "final_ebh", "source_family",
                    "prompt_text", "prompt_sha256")
# A column whose name contains any of these would carry model output. The
# extraction manifest carries none; this refuses one that appeared later.
RESPONSE_COLUMN_TOKENS = ("response", "generation", "output", "completion",
                          "answer", "reply", "before_text", "after_text")


class EquivalenceError(RuntimeError):
    """A gate failed. Stop; do not adapt either implementation."""


# --------------------------------------------------------------- no network
class NetworkAccessAttempted(EquivalenceError):
    pass


def block_network() -> None:
    """Make any outbound connection raise.

    Offline env vars are a request; this is enforcement, and it is what lets the
    evidence record say no network access occurred rather than that none was
    intended. There is no flag to switch it off: a run that needed the network
    would not be the run this stage is meant to certify.
    """
    def refuse(*_args, **_kwargs):
        raise NetworkAccessAttempted(
            "this stage must not touch the network; the pinned snapshot is "
            "read from the local cache only")

    socket.socket.connect = refuse                     # type: ignore[assignment]
    socket.socket.connect_ex = refuse                  # type: ignore[assignment]
    socket.create_connection = refuse                  # type: ignore[assignment]


# Installed at import, unconditionally, before anything in this module can pull
# in huggingface_hub or transformers -- both of which are imported lazily inside
# the functions below, so this runs first in every code path.
block_network()


# ------------------------------------------------------------- digest helpers
def ordered_digest(items: Sequence[str]) -> str:
    """One digest over an ordered sequence of per-row digests.

    Newline-joined so a row boundary cannot be forged by concatenation, and
    order-sensitive so a permutation of the 480 rows is a different digest.
    This is what makes the evidence auditable without storing 480 per-row
    hashes, which would be a fingerprint of the prompt set.
    """
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


def token_sequence_digest(token_ids: Sequence[int]) -> str:
    return sha256_text(",".join(str(int(t)) for t in token_ids))


# -------------------------------------------------------------- tokenisation
def tokenize_like_extraction(tokenizer: Any, text: str) -> Dict[str, Any]:
    """Tokenise exactly as the frozen extraction does.

    ``experiments/routegate_routing/extract_routing_v1.py`` calls
    ``tokenizer(text, return_tensors="pt")`` and
    ``routegate.runtime.tokenize_prompt`` calls the same. Neither passes
    ``add_special_tokens``, so neither may this: supplying it -- with *any*
    value -- would compare a tokenisation the extraction never performs, and a
    clean result would then be evidence about the wrong call.

    ``return_tensors=None`` is the one deliberate difference. It returns plain
    Python lists instead of torch tensors, which keeps this stage off the tensor
    framework entirely and changes no token id.
    """
    return tokenizer(text, return_tensors=None)


# ---------------------------------------------------------------- input gates
@dataclass
class ManifestGate:
    path_name: str
    sha256: str
    n_rows: int
    counts: Dict[str, int]
    columns: List[str]
    prompts: List[str] = field(default_factory=list)
    checks: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c["ok"] for c in self.checks)


def gate_manifest(manifest_path: Path, expected_sha256: str,
                  expected_filename: str) -> ManifestGate:
    """Accept only the pinned development extraction manifest, or refuse.

    Every gate is mandatory and there is no permissive mode. The prompts are
    returned in frozen row order, because that order is what the activation
    array is indexed by.
    """
    manifest_path = Path(manifest_path)
    assert_not_sealed(manifest_path, "prompt-equivalence manifest")

    checks: List[Dict[str, Any]] = []

    def record(name: str, ok: bool, detail: str) -> bool:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    record("manifest_filename_is_the_pinned_one",
           manifest_path.name == expected_filename,
           f"{manifest_path.name} (expected {expected_filename})")
    if not manifest_path.is_file():
        record("manifest_exists", False, "not a file")
        return ManifestGate(manifest_path.name, "", 0, {}, [], [], checks)
    record("manifest_exists", True, "present")

    digest = sha256_file(manifest_path)
    record("manifest_sha256_matches_the_pinned_value",
           digest == expected_sha256, digest)

    with open(manifest_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)

    record("manifest_columns_are_exactly_the_expected_six",
           tuple(columns) == EXPECTED_COLUMNS, str(columns))
    offending = [c for c in columns
                 if any(t in c.lower() for t in RESPONSE_COLUMN_TOKENS)]
    record("manifest_carries_no_response_column", not offending,
           f"no response-bearing column; offending={offending or 'none'}")

    record("manifest_has_exactly_480_rows", len(rows) == EXPECTED_ROWS,
           f"{len(rows)} (expected {EXPECTED_ROWS})")

    indices = [int(r["row_index"]) for r in rows] if rows else []
    record("row_index_is_0_to_479_in_order",
           indices == list(range(EXPECTED_ROWS)),
           "ascending 0..479, so manifest line i is array row i")

    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["final_ebh"]] = counts.get(row["final_ebh"], 0) + 1
    record("class_counts_are_E89_B178_H213", counts == EXPECTED_COUNTS,
           f"E={counts.get('E', 0)} B={counts.get('B', 0)} H={counts.get('H', 0)}")

    bad_hash = [int(r["row_index"]) for r in rows
                if sha256_text(r["prompt_text"]) != r["prompt_sha256"]]
    record("every_prompt_sha256_recomputes_from_its_own_text",
           not bad_hash,
           f"{len(rows) - len(bad_hash)}/{len(rows)} rows recompute")

    return ManifestGate(manifest_path.name, digest, len(rows), counts, columns,
                        [r["prompt_text"] for r in rows], checks)


# ------------------------------------------------------------- legacy import
def import_legacy_module(module_path: Path,
                         module_name: str = LEGACY_MODULE_NAME):
    """Import the historical module read-only, under a distinct name.

    A distinct name so it can never shadow or be mistaken for anything in this
    package. Its own directory goes on ``sys.path`` because it imports two
    sibling modules; both are stdlib-only at module scope, so this pulls in no
    model framework and executes no I/O.
    """
    module_path = Path(module_path).resolve()
    assert_not_sealed(module_path, "legacy generation module")
    if not module_path.is_file():
        raise EquivalenceError(f"legacy module {module_path.name} not found")

    # The historical workspace is read-only to this stage. Importing from it
    # would otherwise write __pycache__/*.pyc there whenever the cached bytecode
    # happens to be stale -- a modification of a directory this program has no
    # business modifying, and one that would go unnoticed.
    sys.dont_write_bytecode = True

    parent = str(module_path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise EquivalenceError(f"cannot load {module_path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------- the comparisons
@dataclass
class PromptComparison:
    """Per-prompt results, counts only -- never the text or the ids."""

    n_compared: int = 0
    rendered_type_failures: List[int] = field(default_factory=list)
    rendered_byte_mismatches: List[int] = field(default_factory=list)
    rendered_sha_mismatches: List[int] = field(default_factory=list)
    token_id_mismatches: List[int] = field(default_factory=list)
    attention_mask_mismatches: List[int] = field(default_factory=list)
    attention_length_mismatches: List[int] = field(default_factory=list)
    final_token_mismatches: List[int] = field(default_factory=list)
    rendered_digest: str = ""
    token_digest: str = ""

    @property
    def rendered_ok(self) -> bool:
        return not (self.rendered_type_failures or self.rendered_byte_mismatches
                    or self.rendered_sha_mismatches)

    @property
    def tokens_ok(self) -> bool:
        return not (self.token_id_mismatches or self.attention_mask_mismatches
                    or self.attention_length_mismatches
                    or self.final_token_mismatches)

    @property
    def ok(self) -> bool:
        return self.rendered_ok and self.tokens_ok

    def mismatch_rows(self) -> List[int]:
        rows = set()
        for group in (self.rendered_type_failures, self.rendered_byte_mismatches,
                      self.rendered_sha_mismatches, self.token_id_mismatches,
                      self.attention_mask_mismatches,
                      self.attention_length_mismatches,
                      self.final_token_mismatches):
            rows.update(group)
        return sorted(rows)


def _as_list(value: Any) -> List[int]:
    """Flatten one batch row of ids/mask to a plain list of ints.

    Accepts a torch tensor, a numpy array, a nested list of one row, or a flat
    list, so the caller is not forced to bring a tensor framework in.
    """
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value) == 1 \
            and isinstance(value[0], (list, tuple)):
        value = value[0]
    return [int(v) for v in value]


def compare_prompts(prompts: Sequence[str],
                    legacy_build: Callable[[Any, str], str],
                    runtime_build: Callable[[Any, str], str],
                    tokenizer: Any,
                    tokenize: Callable[[Any, str], Dict[str, Any]],
                    ) -> PromptComparison:
    """Render and tokenise every prompt through both paths and compare.

    The builders and the tokenise callable are injected so this -- the part that
    decides pass or fail -- is exercised by unit tests with synthetic doubles,
    with no model, no tokeniser and no historical workspace present.

    Comparison is on UTF-8 bytes, not on ``str`` equality, so two strings that
    differ only by a normalisation form are caught rather than smoothed over.
    """
    result = PromptComparison()
    rendered_digests: List[str] = []
    token_digests: List[str] = []

    for position, prompt in enumerate(prompts):
        legacy_text = legacy_build(tokenizer, prompt)
        runtime_text = runtime_build(tokenizer, prompt)
        result.n_compared += 1

        if not isinstance(legacy_text, str) or not isinstance(runtime_text, str):
            result.rendered_type_failures.append(position)
            rendered_digests.append("NON-STRING")
            token_digests.append("NON-STRING")
            continue

        legacy_bytes = legacy_text.encode("utf-8")
        runtime_bytes = runtime_text.encode("utf-8")
        if legacy_bytes != runtime_bytes:
            result.rendered_byte_mismatches.append(position)

        legacy_sha = hashlib.sha256(legacy_bytes).hexdigest()
        runtime_sha = hashlib.sha256(runtime_bytes).hexdigest()
        if legacy_sha != runtime_sha:
            result.rendered_sha_mismatches.append(position)
        rendered_digests.append(runtime_sha)

        legacy_enc = tokenize(tokenizer, legacy_text)
        runtime_enc = tokenize(tokenizer, runtime_text)

        legacy_ids = _as_list(legacy_enc["input_ids"])
        runtime_ids = _as_list(runtime_enc["input_ids"])
        if legacy_ids != runtime_ids:
            result.token_id_mismatches.append(position)

        legacy_mask = _as_list(legacy_enc["attention_mask"])
        runtime_mask = _as_list(runtime_enc["attention_mask"])
        if len(legacy_mask) != len(runtime_mask):
            result.attention_length_mismatches.append(position)
        if legacy_mask != runtime_mask:
            result.attention_mask_mismatches.append(position)

        # Position -1 is where every activation in the frozen extraction was
        # captured; if it differs, nothing downstream is comparable.
        legacy_final = legacy_ids[-1] if legacy_ids else None
        runtime_final = runtime_ids[-1] if runtime_ids else None
        if legacy_final != runtime_final:
            result.final_token_mismatches.append(position)

        token_digests.append(token_sequence_digest(runtime_ids))

    result.rendered_digest = ordered_digest(rendered_digests)
    result.token_digest = ordered_digest(token_digests)
    return result


def compare_constants(legacy: Any, runtime: Any) -> List[Dict[str, Any]]:
    """Checks 1 and 2: the pinned model and revision agree."""
    return [
        {"check": "model_identifier_agrees",
         "ok": legacy.MODEL == runtime.FROZEN_MODEL,
         "detail": f"legacy MODEL vs runtime.FROZEN_MODEL: "
                   f"{legacy.MODEL!r} vs {runtime.FROZEN_MODEL!r}"},
        {"check": "frozen_revision_agrees",
         "ok": legacy.FROZEN_REVISION == runtime.FROZEN_REVISION,
         "detail": f"{legacy.FROZEN_REVISION} vs {runtime.FROZEN_REVISION}"},
    ]


def compare_snapshots(legacy_resolved: Tuple[str, Any],
                      runtime_resolved: Tuple[str, Any],
                      pinned_revision: str,
                      legacy_tokenizer_sha: str, runtime_tokenizer_sha: str,
                      legacy_template_sha: str, runtime_template_sha: str,
                      ) -> List[Dict[str, Any]]:
    """Checks 3, 4 and 5.

    Both implementations verify these hashes internally and raise if they do not
    match, so reaching here at all is already meaningful; they are compared and
    recorded anyway, because evidence beats a side effect.
    """
    legacy_commit, legacy_path = legacy_resolved
    runtime_commit, runtime_path = runtime_resolved
    same_dir = Path(legacy_path).resolve() == Path(runtime_path).resolve()
    return [
        {"check": "both_resolve_the_same_snapshot_at_the_pinned_commit",
         "ok": (legacy_commit == runtime_commit == pinned_revision and same_dir),
         # The snapshot directory is a machine path and is deliberately not
         # recorded; that both sides resolved the same one is.
         "detail": f"commit {runtime_commit}; both paths identical: {same_dir}"},
        {"check": "tokenizer_config_sha256_agrees",
         "ok": legacy_tokenizer_sha == runtime_tokenizer_sha,
         "detail": runtime_tokenizer_sha},
        {"check": "chat_template_sha256_agrees",
         "ok": legacy_template_sha == runtime_template_sha,
         "detail": runtime_template_sha},
    ]


def seven_checks(constant_checks: List[Dict[str, Any]],
                 snapshot_checks: List[Dict[str, Any]],
                 comparison: PromptComparison,
                 expected_rows: int = EXPECTED_ROWS) -> List[Dict[str, Any]]:
    """The seven step-9 checks, in the order the migration plan states them."""
    rendered_ok = comparison.rendered_ok and comparison.n_compared == expected_rows
    tokens_ok = comparison.tokens_ok and comparison.n_compared == expected_rows
    return constant_checks + snapshot_checks + [
        {"check": "all_prompts_render_to_identical_bytes_and_sha256",
         "ok": rendered_ok,
         "detail": f"{comparison.n_compared - len(set(comparison.rendered_byte_mismatches) | set(comparison.rendered_sha_mismatches) | set(comparison.rendered_type_failures))}"
                   f"/{expected_rows} identical"},
        {"check": "all_prompts_tokenise_to_identical_sequences",
         "ok": tokens_ok,
         "detail": f"{comparison.n_compared - len(set(comparison.token_id_mismatches) | set(comparison.attention_mask_mismatches) | set(comparison.attention_length_mismatches) | set(comparison.final_token_mismatches))}"
                   f"/{expected_rows} identical (input_ids, attention mask and "
                   f"length, final prompt token)"},
    ]


# -------------------------------------------------------------------- evidence
def package_versions() -> Dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("transformers", "tokenizers", "huggingface_hub"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:                                   # pragma: no cover
            versions[name] = "not installed"
    return versions


def build_evidence(*, checks: List[Dict[str, Any]], gate: ManifestGate,
                   comparison: PromptComparison, legacy_module_sha256: str,
                   legacy_module_filename: str, runtime_sha256: str,
                   model: str, revision: str, resolved_commit: str,
                   tokenizer_config_sha256: str, chat_template_sha256: str,
                   expected_rows: int = EXPECTED_ROWS) -> Dict[str, Any]:
    """The machine-readable record.

    Deliberately absent: prompt text, rendered text, token ids, per-row hashes,
    response text, absolute machine paths, and anything about the held-out
    split. What remains is enough to re-verify the claim and not enough to
    reconstruct the prompt set.
    """
    passed = all(c["ok"] for c in checks) and gate.ok
    return {
        "schema": "routegate/prompt_equivalence/1",
        "stage": "phase2_step9_prompt_equivalence",
        "purpose": "gate on replacing the historical generation import with "
                   "routegate.runtime (step 10)",
        "verdict": "PASS" if passed else "FAIL",
        "recorded_utc": datetime.datetime.now(datetime.timezone.utc)
                                 .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checks": checks,
        "n_checks": len(checks),
        "n_failed_checks": sum(1 for c in checks if not c["ok"]),
        "input_gates": gate.checks,
        "comparison": {
            "n_expected": expected_rows,
            "n_compared": comparison.n_compared,
            "rendered_identical": comparison.n_compared - len(
                set(comparison.rendered_byte_mismatches)
                | set(comparison.rendered_sha_mismatches)
                | set(comparison.rendered_type_failures)),
            "tokens_identical": comparison.n_compared - len(
                set(comparison.token_id_mismatches)
                | set(comparison.attention_mask_mismatches)
                | set(comparison.attention_length_mismatches)
                | set(comparison.final_token_mismatches)),
            "mismatch_counts": {
                "rendered_non_string": len(comparison.rendered_type_failures),
                "rendered_bytes": len(comparison.rendered_byte_mismatches),
                "rendered_sha256": len(comparison.rendered_sha_mismatches),
                "token_ids": len(comparison.token_id_mismatches),
                "attention_mask": len(comparison.attention_mask_mismatches),
                "attention_length": len(comparison.attention_length_mismatches),
                "final_prompt_token": len(comparison.final_token_mismatches),
            },
        },
        "sources": {
            "legacy_module_filename": legacy_module_filename,
            "legacy_module_sha256": legacy_module_sha256,
            "routegate_runtime_sha256": runtime_sha256,
            "extraction_manifest_filename": gate.path_name,
            "extraction_manifest_sha256": gate.sha256,
            "extraction_manifest_rows": gate.n_rows,
            "extraction_manifest_class_counts": gate.counts,
        },
        "pinned_runtime": {
            "model": model,
            "revision": revision,
            "resolved_snapshot_commit": resolved_commit,
            "tokenizer_config_sha256": tokenizer_config_sha256,
            "chat_template_sha256": chat_template_sha256,
        },
        "aggregate_digests": {
            "definition": "sha256 over the newline-joined, row-ordered list of "
                          "per-row digests; order-sensitive, and it stores no "
                          "individual row hash",
            "rendered_text_sha256_sequence": comparison.rendered_digest,
            "token_id_sequence": comparison.token_digest,
            "token_row_digest_definition": "sha256 of the comma-joined input_ids",
        },
        "versions": package_versions(),
        "not_done": ["model load", "GPU use", "generation", "API call",
                     "network access", "sealed-set access",
                     "modification of the legacy module or routegate.runtime"],
    }


# ------------------------------------------------------------------------ main
def load_pinned_manifest_spec() -> Dict[str, Any]:
    """The pinned extraction-manifest record, from the committed manifest only.

    Not overridable. A ``--input-manifest`` flag would let the caller supply the
    expectations the external file is checked against, which is a gate that can
    be argued out of; the point of this one is that it cannot be. The committed
    record is read through ``routegate.data.load_input_manifest``, so it passes
    the same sealed-free validation as every other consumer, and its own shape
    is checked before anything is compared against it.
    """
    from routegate.data import load_input_manifest

    manifest = load_input_manifest(DEFAULT_INPUT_MANIFEST)
    if "extraction_manifest" not in manifest:
        raise EquivalenceError(
            "manifests/development_inputs.json carries no extraction_manifest "
            "record; there is nothing to pin the external manifest against")
    spec = manifest["extraction_manifest"]
    for key in ("filename", "sha256"):
        if not isinstance(spec.get(key), str) or not spec[key]:
            raise EquivalenceError(
                f"the committed extraction_manifest record has no usable "
                f"{key!r}")
    if not re.fullmatch(r"[0-9a-f]{64}", spec["sha256"]):
        raise EquivalenceError(
            "the committed extraction_manifest sha256 is not a SHA256 digest")
    return spec


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--legacy-module", required=True, type=Path,
                        help="path to the historical rq3_dev_generation.py")
    parser.add_argument("--manifest", required=True, type=Path,
                        help="path to the development extraction manifest.csv")
    parser.add_argument("--evidence-out", type=Path, default=DEFAULT_EVIDENCE,
                        help="where the evidence artifact is written")
    parser.add_argument("--failure-log", type=Path, default=None,
                        help="private path OUTSIDE the repository for mismatch "
                             "row positions; only written on failure")
    return parser.parse_args(argv)


def resolve_failure_log(path: Optional[Path],
                        repo_root: Path = REPO_ROOT) -> Optional[Path]:
    """Resolve the failure-log path, refusing anything inside the repository.

    A per-row mismatch list is a fingerprint of the prompt set, so the promise
    that it stays private has to be enforced rather than documented. Checked
    up front, before the comparison runs, so a bad path fails in a second
    instead of after the whole 480-prompt pass.
    """
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    root = Path(repo_root).resolve()
    if resolved == root or root in resolved.parents:
        raise EquivalenceError(
            f"--failure-log {resolved.name} resolves inside the RouteGate "
            f"repository; the mismatch list must be written to a private "
            f"location outside it")
    return resolved


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    say = print
    # Refused before the comparison, not after it.
    failure_log = resolve_failure_log(args.failure_log)
    say("=" * 78)
    say("PHASE 2 STEP 9 -- PROMPT EQUIVALENCE (no model is loaded)")
    say("=" * 78)

    spec = load_pinned_manifest_spec()
    gate = gate_manifest(args.manifest, spec["sha256"], spec["filename"])
    say("\n-- input gates --")
    for check in gate.checks:
        say(f"  [{'PASS' if check['ok'] else 'FAIL'}] {check['check']:<52} {check['detail']}")
    if not gate.ok:
        say("\n[STOP] the manifest did not pass its gates; nothing was compared.")
        return 1

    legacy = import_legacy_module(args.legacy_module)
    from routegate import runtime

    say("\n-- pinned configuration --")
    constant_checks = compare_constants(legacy, runtime)
    for check in constant_checks:
        say(f"  [{'PASS' if check['ok'] else 'FAIL'}] {check['check']:<52} {check['detail']}")

    legacy_resolved = legacy.resolve_snapshot()
    runtime_resolved = runtime.resolve_snapshot()
    snapshot_path = Path(runtime_resolved[1])
    tokenizer_config = snapshot_path / "tokenizer_config.json"
    tokenizer_config_sha = sha256_file(tokenizer_config)
    chat_template = json.loads(tokenizer_config.read_text(encoding="utf-8"))["chat_template"]
    chat_template_sha = sha256_text(chat_template)

    snapshot_checks = compare_snapshots(
        legacy_resolved, runtime_resolved, runtime.FROZEN_REVISION,
        legacy.FROZEN_TOKENIZER_CONFIG_SHA256, runtime.FROZEN_TOKENIZER_CONFIG_SHA256,
        legacy.FROZEN_CHAT_TEMPLATE_SHA256, runtime.FROZEN_CHAT_TEMPLATE_SHA256)
    for check in snapshot_checks:
        say(f"  [{'PASS' if check['ok'] else 'FAIL'}] {check['check']:<52} {check['detail']}")
    if tokenizer_config_sha != runtime.FROZEN_TOKENIZER_CONFIG_SHA256:
        raise EquivalenceError("snapshot tokenizer_config.json is not the frozen one")
    if chat_template_sha != runtime.FROZEN_CHAT_TEMPLATE_SHA256:
        raise EquivalenceError("snapshot chat_template is not the frozen one")

    # The tokeniser only. AutoModelForCausalLM is never instantiated, so no
    # weight is read and no accelerator is touched.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        runtime.FROZEN_MODEL, revision=runtime.FROZEN_REVISION,
        trust_remote_code=True, local_files_only=True)

    say(f"\n-- rendering and tokenising {gate.n_rows} prompts through both paths --")
    comparison = compare_prompts(gate.prompts, legacy.build_chat_strict,
                                 runtime.build_chat_strict, tokenizer,
                                 tokenize_like_extraction)

    checks = seven_checks(constant_checks, snapshot_checks, comparison)
    say("\n-- the seven checks --")
    for i, check in enumerate(checks, 1):
        say(f"  [{'PASS' if check['ok'] else 'FAIL'}] {i}. {check['check']:<50} {check['detail']}")

    evidence = build_evidence(
        checks=checks, gate=gate, comparison=comparison,
        legacy_module_sha256=sha256_file(args.legacy_module),
        legacy_module_filename=Path(args.legacy_module).name,
        runtime_sha256=sha256_file(REPO_ROOT / "src" / "routegate" / "runtime.py"),
        model=runtime.FROZEN_MODEL, revision=runtime.FROZEN_REVISION,
        resolved_commit=runtime_resolved[0],
        tokenizer_config_sha256=tokenizer_config_sha,
        chat_template_sha256=chat_template_sha)

    args.evidence_out.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_out.write_text(
        json.dumps(evidence, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    if not comparison.ok and failure_log is not None:
        # Row positions only, and outside the repository: a per-row failure list
        # is still a fingerprint of the prompt set.
        failure_log.parent.mkdir(parents=True, exist_ok=True)
        failure_log.write_text(json.dumps({
            "mismatch_row_positions": comparison.mismatch_rows(),
            "by_kind": {
                "rendered_non_string": comparison.rendered_type_failures,
                "rendered_bytes": comparison.rendered_byte_mismatches,
                "rendered_sha256": comparison.rendered_sha_mismatches,
                "token_ids": comparison.token_id_mismatches,
                "attention_mask": comparison.attention_mask_mismatches,
                "attention_length": comparison.attention_length_mismatches,
                "final_prompt_token": comparison.final_token_mismatches,
            }}, indent=2) + "\n", encoding="utf-8")
        say(f"\n  mismatch row positions written to the private failure log")

    say("\n" + "=" * 78)
    say(f"VERDICT: {evidence['verdict']}   "
        f"rendered {evidence['comparison']['rendered_identical']}/{EXPECTED_ROWS}   "
        f"tokens {evidence['comparison']['tokens_identical']}/{EXPECTED_ROWS}   "
        f"checks {len(checks) - evidence['n_failed_checks']}/{len(checks)}")
    say("=" * 78)
    return 0 if evidence["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
