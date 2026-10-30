"""One judging engine, driven by a versioned manifest. Fail-closed throughout.

WHY THIS EXISTS
---------------
Three judging stages have been built in this package -- phase 1, phase 2 and
the two-site actuator -- and each grew its own copy of the same nine steps.
They agree on the science and differ only in bookkeeping, which is the worst
way for code to differ: every copy is a place where a gate can quietly go
missing. A fourth steering mechanism should not need a fourth copy.

So this module is the engine, and an experiment supplies a **manifest**. The
manifest declares identity, arms, digests, rubrics, judges, sampling,
adjudication policy, expected counts and output locations. It declares nothing
about *how* a cell is decided, because that is not an experiment's choice.

WHAT IT DOES NOT DO
-------------------
It does not re-implement the rules. The echo check is
``routegate.triplets.validate_parsed``. The adjudication rule is
``routegate.focusedjudging.adjudicate_cell``. The sealed-set guards are
``routegate.validation``. Each is imported and called unchanged; a manifest
cannot loosen any of them, and there is no manifest field that tries to.

It also does not touch the completed stages. Phase 1, phase 2 and the two-site
run keep their own scripts and their own artifacts, byte for byte. This engine
is for the experiments that come next.

THE INVARIANTS, AND WHERE THEY LIVE
-----------------------------------
* exact item-id echo is mandatory .... ``validate_parsed`` / ``validate_reply``
* a mismatched id, empty reply, malformed reply or transport failure is never
  a verdict ........................... ``FAILURE_POLICY`` admits one value
* technical failures stay UNRESOLVED_TECHNICAL ......... ``join_evidence``
* humans see genuine disagreements only ......... ``cells_needing_review``
* humans never replace missing judge evidence ....... ``adjudicate_cells``
* a blinded A/B review is decoded per row, through the private row map and
  never by a global rename ......... ``decode_pairwise`` / ``import_review``
* every planned cell is kept; rates use complete-case denominators . ``report``
* unresolved cells listed by opaque id and rubric, never by text ... ``report``
* a packet is not rebuilt once judging starts ......... ``assert_packet_new``
* a successful call is never repeated .................. ``already_done``
* raw evidence is immutable ...................... ``preserve_raw``
* exceptions redact credentials ......... ``triplets.redact_exception_message``
* no secret reaches a manifest or a record ............. ``assert_no_secrets``
* the sealed set is unreachable ................. ``assert_not_sealed``
* dry-run writes nothing and calls nothing ............ every ``write`` flag

Experiment-specific *readings* are deliberately out of scope: the report emits
verdict tallies with explicit denominators, and an experiment that wants a
repair rate or a regression rule computes it from that, in its own module.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .triplets import (
    CONFIDENCE_VALUES,
    JUDGES,
    assert_no_text_fields,
    assert_opaque,
    read_jsonl,
    result_join_key,
    sha256_text,
)
from .validation import (
    Checklist,
    ValidationError,
    assert_config_is_sealed_free,
    assert_not_sealed,
    sha256_file,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATHS_FILE = REPO_ROOT / "configs" / "paths.local.json"

#: The manifest dialect this engine reads. A manifest that does not name it is
#: refused rather than guessed at.
SCHEMA = "routegate_judging_protocol_v1"

#: The nine stages, in execution order. The operator names one.
STAGES: Tuple[str, ...] = (
    "preflight", "build", "dryrun", "judge", "join", "export", "import",
    "adjudicate", "report")

#: Stages that are free and offline. ``judge`` is the only billed one.
OFFLINE_STAGES: Tuple[str, ...] = tuple(s for s in STAGES if s != "judge")

#: What the join may call a cell. ``unresolved_technical`` is terminal and
#: decided by nobody; ``missing`` and ``incomplete`` are defects and stop it.
STATE_AGREE = "agree"
STATE_DISAGREE = "disagree"
STATE_UNRESOLVED_TECHNICAL = "unresolved_technical"
STATE_INCOMPLETE = "incomplete"
STATE_MISSING = "missing"
TERMINAL_STATES: Tuple[str, ...] = (
    STATE_AGREE, STATE_DISAGREE, STATE_UNRESOLVED_TECHNICAL)

#: The only policy this engine will accept for a failure of any kind. It is a
#: one-element tuple on purpose: a manifest cannot declare that a malformed
#: reply becomes a verdict, because there is no value it could put here to say
#: so. Widening this tuple is a change to the science, not to a config.
FAILURE_POLICY: Tuple[str, ...] = ("unresolved_technical",)

#: The only policy for a genuine two-judge disagreement.
DISAGREEMENT_POLICY: Tuple[str, ...] = ("human_review_blinded",)

#: The adjudication procedures, by name. Both are frozen in
#: ``routegate.focusedjudging``; this engine never defines a third.
PROCEDURES: Tuple[str, ...] = (
    "corpus_disagreement_only", "triplet_independent_review")

#: A judge row carries exactly these fields and nothing else. Blindness is a
#: whitelist projection, so a row cannot leak a field nobody thought to ban.
SINGLE_ROW_FIELDS: Tuple[str, ...] = ("row_id", "prompt_text", "response_text")
PROMPT_ROW_FIELDS: Tuple[str, ...] = ("row_id", "prompt_text")
PAIRWISE_ROW_FIELDS: Tuple[str, ...] = (
    "row_id", "prompt_text", "response_a", "response_b")

#: Presentation positions for a pairwise rubric, and the origins behind them.
POSITION_A, POSITION_B = "A", "B"
ORIGIN_CANDIDATE, ORIGIN_REFERENCE = "candidate", "reference"

#: Field names that would carry a credential if anyone ever put one in a
#: manifest. Precise on purpose: a bare ``key`` would also match the honest
#: ``paths_key`` and ``root_key``, and a guard that cries wolf gets removed.
SECRET_KEY_HINTS: Tuple[str, ...] = (
    "api_key", "apikey", "x_api_key", "access_token", "auth_token", "token",
    "secret", "password", "passwd", "credential", "credentials",
    "authorization", "bearer", "endpoint", "endpoint_url", "session_key",
    "private_key")

#: Value prefixes that are credentials wherever they appear.
SECRET_VALUE_HINTS: Tuple[str, ...] = ("sk-", "sk_", "Bearer ", "AKIA")

MANIFEST_FILENAME = "manifest.json"
ROW_MAP_FILE = "row_map.jsonl"
JOINED_CELLS_FILE = "joined_cells.jsonl"
ADJUDICATED_CELLS_FILE = "adjudicated_cells.jsonl"
REVIEW_INPUT_FILE = "human_review_input.jsonl"
REVIEW_TEMPLATE_FILE = "human_review_TEMPLATE.jsonl"
REVIEW_COMPLETED_FILE = "human_review_COMPLETED.jsonl"
REVIEW_MAP_FILE = "review_row_map.jsonl"
REPORT_FILE = "judging_report.json"
RESULTS_FILE = "results.jsonl"
FAILURES_FILE = "failures.jsonl"


class JudgingProtocolError(ValidationError):
    """The protocol was asked for something it must not do."""


# ==================================================================== secrets
def assert_no_secrets(payload: Any, label: str) -> None:
    """Refuse anything credential-shaped, by key name and by value shape.

    A manifest is committed. A record is kept. Neither may carry a key, a
    token or an endpoint, so both are walked before they are used or written.
    """
    def walk(node: Any, trail: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                low = str(key).lower()
                if (any(h == low or low.endswith("_" + h)
                        for h in SECRET_KEY_HINTS)
                        and isinstance(value, str) and value):
                    raise JudgingProtocolError(
                        f"{label}{trail}.{key} looks like a credential. No "
                        f"secret is ever stored in a manifest or a record; "
                        f"the runner reads the key from the environment.")
                walk(value, f"{trail}.{key}")
        elif isinstance(node, (list, tuple)):
            for i, value in enumerate(node):
                walk(value, f"{trail}[{i}]")
        elif isinstance(node, str):
            for hint in SECRET_VALUE_HINTS:
                if node.startswith(hint):
                    raise JudgingProtocolError(
                        f"{label}{trail} holds a value beginning {hint!r}, "
                        f"which is credential-shaped")

    walk(payload, "")


# =================================================================== manifest
@dataclass(frozen=True)
class RubricSpec:
    """One rubric: what it asks, what it accepts, and its frozen digests."""

    name: str
    pairwise: bool
    verdict_fields: Tuple[str, ...]
    values: Tuple[str, ...]
    judge_values: Tuple[str, ...]
    schema_name: str
    instructions_sha256: str
    user_template_sha256: str
    schema_sha256: str
    protocol_dir_key: str
    subdir: str
    row_prefix: str
    prompt_only: bool = False

    @property
    def row_fields(self) -> Tuple[str, ...]:
        if self.prompt_only:
            return PROMPT_ROW_FIELDS
        return PAIRWISE_ROW_FIELDS if self.pairwise else SINGLE_ROW_FIELDS

    @property
    def accepted_values(self) -> Tuple[str, ...]:
        """What a *judge* may say. For a pairwise rubric that is A/B space.

        A pairwise judge answers about presentation positions and knows
        nothing about which arm sits where, so its vocabulary is not the one
        the analysis uses. ``values`` is the decoded vocabulary; the join
        maps one to the other through the private row map.
        """
        return self.judge_values or self.values

    def digests(self) -> Dict[str, str]:
        return {"instructions_sha256": self.instructions_sha256,
                "user_template_sha256": self.user_template_sha256,
                "schema_sha256": self.schema_sha256}


@dataclass(frozen=True)
class Manifest:
    """A validated protocol manifest. Every field was checked on the way in."""

    path: Path
    raw: Mapping[str, Any]
    experiment_id: str
    packet_id: str
    rubrics: Mapping[str, RubricSpec]
    classes: Mapping[str, Mapping[str, Any]]
    candidate_arms: Tuple[str, ...]
    reference_arms: Tuple[str, ...]
    judges: Mapping[str, Mapping[str, str]]
    judge_order: Tuple[str, ...]
    temperature: float
    max_tokens: int
    procedure: str
    coverage: Mapping[str, Any]
    outputs: Mapping[str, str]
    inputs: Mapping[str, Mapping[str, str]]
    human_row_prefix: str
    foreign_row_prefixes: Tuple[str, ...]
    technical_manifest: Optional[str]
    reporting_module: Optional[str]

    def rubric_for_class(self, row_class: str) -> RubricSpec:
        return self.rubrics[str(self.classes[row_class]["rubric"])]

    def class_for_rubric(self, rubric: str) -> str:
        for name, spec in self.classes.items():
            if str(spec["rubric"]) == rubric:
                return name
        raise JudgingProtocolError(f"no class uses rubric {rubric!r}")

    def digest(self) -> str:
        """The manifest's own digest, over its canonical serialisation."""
        return sha256_text(json.dumps(self.raw, sort_keys=True))


_REQUIRED_TOP = ("schema", "experiment_id", "packet_id", "identity", "arms",
                 "inputs", "classes", "rubrics", "judges", "sampling",
                 "adjudication", "coverage", "outputs")
_REQUIRED_RUBRIC = ("pairwise", "verdict_fields", "values", "schema_name",
                    "instructions_sha256", "user_template_sha256",
                    "schema_sha256", "protocol_dir_key", "subdir")
_REQUIRED_ADJUDICATION = ("procedure", "disagreement", "missing", "malformed",
                          "transport_failure")


def load_manifest(path: Path | str) -> Manifest:
    """Read one protocol manifest and refuse it unless every gate holds."""
    path = Path(path)
    if not path.is_file():
        raise JudgingProtocolError(
            f"{path} is not a file. --manifest names the frozen protocol "
            f"manifest for one experiment.")
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert_config_is_sealed_free(raw, f"{path.name}")
    return parse_protocol_manifest(raw, path)


def parse_protocol_manifest(raw: Mapping[str, Any], path: Path) -> Manifest:
    """Validate shared protocol rules after a caller validates its data scope.

    The development loader above retains its sealed-data exclusion. The final
    evaluation entry point validates the exact released generations and its
    frozen assessment plan before calling this shared parser.
    """
    path = Path(path)

    if raw.get("schema") != SCHEMA:
        raise JudgingProtocolError(
            f"{path.name} declares schema {raw.get('schema')!r}; this engine "
            f"reads {SCHEMA!r} and does not guess at another dialect.")
    for field_name in _REQUIRED_TOP:
        if field_name not in raw:
            raise JudgingProtocolError(
                f"{path.name} is missing {field_name!r}")

    # Protocol metadata carries neither credentials nor prompt/response text.
    assert_no_secrets(raw, path.name)
    assert_no_text_fields(raw, path.name)

    identity = raw["identity"]
    prefixes = identity.get("row_prefixes", {})
    if not prefixes:
        raise JudgingProtocolError(f"{path.name}: identity.row_prefixes is "
                                   f"empty; every rubric needs its own prefix")

    rubrics: Dict[str, RubricSpec] = {}
    for name, spec in raw["rubrics"].items():
        for field_name in _REQUIRED_RUBRIC:
            if field_name not in spec:
                raise JudgingProtocolError(
                    f"{path.name}: rubric {name!r} is missing {field_name!r}")
        if name not in prefixes:
            raise JudgingProtocolError(
                f"{path.name}: rubric {name!r} has no row prefix")
        values = tuple(str(v) for v in spec["values"])
        if spec.get("prompt_only", False) and spec["pairwise"]:
            raise JudgingProtocolError("a prompt-only rubric cannot be pairwise")
        judge_values = tuple(str(v) for v in spec.get("judge_values", ()))
        if spec["pairwise"] and not judge_values:
            raise JudgingProtocolError(
                f"{path.name}: pairwise rubric {name!r} must declare "
                f"judge_values -- what the judge may answer in A/B space -- "
                f"as well as values, the decoded vocabulary the analysis "
                f"uses. They are different spaces and conflating them is how "
                f"an undecoded verdict reaches a reading.")
        if judge_values and not spec["pairwise"]:
            raise JudgingProtocolError(
                f"{path.name}: rubric {name!r} is not pairwise, so its judge "
                f"vocabulary is its vocabulary; drop judge_values")
        if len(set(values)) != len(values) or not values:
            raise JudgingProtocolError(
                f"{path.name}: rubric {name!r} has an empty or repeating "
                f"value list; a verdict vocabulary is a set")
        rubrics[name] = RubricSpec(
            name=name, pairwise=bool(spec["pairwise"]),
            verdict_fields=tuple(str(f) for f in spec["verdict_fields"]),
            values=values, judge_values=judge_values,
            schema_name=str(spec["schema_name"]),
            instructions_sha256=str(spec["instructions_sha256"]),
            user_template_sha256=str(spec["user_template_sha256"]),
            schema_sha256=str(spec["schema_sha256"]),
            protocol_dir_key=str(spec["protocol_dir_key"]),
            subdir=str(spec["subdir"]),
            row_prefix=str(prefixes[name]),
            prompt_only=bool(spec.get("prompt_only", False)))

    classes = raw["classes"]
    for name, spec in classes.items():
        if str(spec.get("rubric")) not in rubrics:
            raise JudgingProtocolError(
                f"{path.name}: class {name!r} names rubric "
                f"{spec.get('rubric')!r}, which is not declared")
        if not isinstance(spec.get("rows"), int) or spec["rows"] < 0:
            raise JudgingProtocolError(
                f"{path.name}: class {name!r} needs an integer row count")
    used = {str(spec["rubric"]) for spec in classes.values()}
    unused = sorted(set(rubrics) - used)
    if unused:
        raise JudgingProtocolError(
            f"{path.name}: rubric(s) {unused} are declared but no class uses "
            f"them. An unused rubric is a drafting error, not a spare.")

    adjudication = raw["adjudication"]
    for field_name in _REQUIRED_ADJUDICATION:
        if field_name not in adjudication:
            raise JudgingProtocolError(
                f"{path.name}: adjudication.{field_name} is not declared")
    procedure = str(adjudication["procedure"])
    if procedure not in PROCEDURES:
        raise JudgingProtocolError(
            f"{path.name}: adjudication procedure {procedure!r} is not one of "
            f"{list(PROCEDURES)}")
    if str(adjudication["disagreement"]) not in DISAGREEMENT_POLICY:
        raise JudgingProtocolError(
            f"{path.name}: a genuine disagreement is resolved by "
            f"{list(DISAGREEMENT_POLICY)} and nothing else")
    for kind in ("missing", "malformed", "transport_failure"):
        if str(adjudication[kind]) not in FAILURE_POLICY:
            raise JudgingProtocolError(
                f"{path.name}: adjudication.{kind}={adjudication[kind]!r}. A "
                f"failure of any kind is {FAILURE_POLICY[0]!r}; there is no "
                f"manifest value that turns one into a verdict.")

    judges = raw["judges"]
    order = tuple(str(j) for j in judges.get("order", JUDGES))
    if tuple(sorted(order)) != tuple(sorted(JUDGES)):
        raise JudgingProtocolError(
            f"{path.name}: judges.order is {list(order)}; this engine runs "
            f"exactly {list(JUDGES)}, both over every row")
    for judge in order:
        if judge not in judges or "model" not in judges[judge]:
            raise JudgingProtocolError(
                f"{path.name}: judge {judge!r} declares no model")

    sampling = raw["sampling"]
    for field_name in ("temperature", "max_tokens"):
        if field_name not in sampling:
            raise JudgingProtocolError(
                f"{path.name}: sampling.{field_name} is not declared")
    temperature = float(sampling["temperature"])
    max_tokens = int(sampling["max_tokens"])
    from .triplets import MAX_TOKENS as _FROZEN_MAX
    from .triplets import TEMPERATURE as _FROZEN_T
    if temperature != _FROZEN_T or max_tokens != _FROZEN_MAX:
        raise JudgingProtocolError(
            f"{path.name}: sampling is T={temperature}, "
            f"max_tokens={max_tokens}; the frozen transport sends "
            f"T={_FROZEN_T}, max_tokens={_FROZEN_MAX}. Those values are baked "
            f"into request_body's key order and therefore into every request "
            f"digest ever recorded, so a manifest declares them rather than "
            f"changing them. T=0 also keeps a re-run a check, not a new "
            f"sample.")

    arms = raw["arms"]
    candidate = tuple(str(a) for a in arms.get("candidate", ()))
    reference = tuple(str(a) for a in arms.get("reference", ()))
    if not candidate:
        raise JudgingProtocolError(
            f"{path.name}: arms.candidate is empty; an experiment judges at "
            f"least one steering arm")
    overlap = sorted(set(candidate) & set(reference))
    if overlap:
        raise JudgingProtocolError(
            f"{path.name}: {overlap} is both a candidate and a reference arm")
    if any(r.pairwise for r in rubrics.values()) and not reference:
        raise JudgingProtocolError(
            f"{path.name}: a pairwise rubric compares against a reference "
            f"arm, and none is declared")

    manifest = Manifest(
        path=path, raw=raw,
        experiment_id=str(raw["experiment_id"]),
        packet_id=str(raw["packet_id"]),
        rubrics=rubrics, classes=classes,
        candidate_arms=candidate, reference_arms=reference,
        judges={j: dict(judges[j]) for j in order}, judge_order=order,
        temperature=temperature, max_tokens=max_tokens,
        procedure=procedure,
        coverage=raw["coverage"], outputs=raw["outputs"],
        inputs=raw["inputs"],
        human_row_prefix=str(identity.get("human_row_prefix", "RGJHUMAN")),
        foreign_row_prefixes=tuple(
            str(p) for p in identity.get("foreign_row_prefixes", ())),
        technical_manifest=(
            str(adjudication["technical_unresolved_manifest"])
            if adjudication.get("technical_unresolved_manifest") else None),
        reporting_module=(str(raw.get("reporting", {}).get("module"))
                          if raw.get("reporting", {}).get("module") else None))
    _check_coverage_arithmetic(manifest)
    _check_prefixes_are_distinct(manifest)
    return manifest


def _check_coverage_arithmetic(manifest: Manifest) -> None:
    """The declared counts must add up before anything is built or billed."""
    coverage = manifest.coverage
    by_class = {name: int(spec["rows"])
                for name, spec in manifest.classes.items()}
    total = sum(by_class.values())
    for field_name in ("cells", "records_per_judge", "planned_calls"):
        if field_name not in coverage:
            raise JudgingProtocolError(
                f"{manifest.path.name}: coverage.{field_name} is not declared")
    if int(coverage["cells"]) != total:
        raise JudgingProtocolError(
            f"{manifest.path.name}: coverage.cells="
            f"{coverage['cells']} but the classes declare {total} row(s)")
    if int(coverage["records_per_judge"]) != total:
        raise JudgingProtocolError(
            f"{manifest.path.name}: every judge sees every cell, so "
            f"records_per_judge must be {total}")
    want = total * len(manifest.judge_order)
    if int(coverage["planned_calls"]) != want:
        raise JudgingProtocolError(
            f"{manifest.path.name}: coverage.planned_calls="
            f"{coverage['planned_calls']}, expected {want}")


def _check_prefixes_are_distinct(manifest: Manifest) -> None:
    """A row id must not be joinable to another packet's rows."""
    mine = [r.row_prefix for r in manifest.rubrics.values()]
    if len(set(mine)) != len(mine):
        raise JudgingProtocolError(
            f"{manifest.path.name}: two rubrics share a row prefix; a cell "
            f"would then be ambiguous between them")
    clash = sorted(set(mine) & set(manifest.foreign_row_prefixes))
    if clash:
        raise JudgingProtocolError(
            f"{manifest.path.name}: prefix(es) {clash} are also minted by "
            f"another packet. Two experiments' rows must never join.")


def expected_coverage(manifest: Manifest) -> Dict[str, Any]:
    """The shape a complete run of this manifest must have."""
    by_rubric: Dict[str, int] = {}
    by_class: Dict[str, int] = {}
    for name, spec in manifest.classes.items():
        rubric = str(spec["rubric"])
        by_rubric[rubric] = by_rubric.get(rubric, 0) + int(spec["rows"])
        by_class[name] = int(spec["rows"])
    return {
        "cells": int(manifest.coverage["cells"]),
        "cells_by_rubric": dict(sorted(by_rubric.items())),
        "cells_by_class": dict(sorted(by_class.items())),
        "records_per_judge": int(manifest.coverage["records_per_judge"]),
        "planned_calls": int(manifest.coverage["planned_calls"]),
        "terminal_states": TERMINAL_STATES,
    }


# ====================================================================== paths
def resolve_paths(paths_file: Path | str = DEFAULT_PATHS_FILE
                  ) -> Mapping[str, str]:
    paths_file = Path(paths_file)
    if not paths_file.is_file():
        raise JudgingProtocolError(f"{paths_file} is not a file")
    paths = json.loads(paths_file.read_text(encoding="utf-8"))
    assert_config_is_sealed_free(paths, "local path file")
    return paths


def resolve_root(manifest: Manifest, key: str,
                 paths: Optional[Mapping[str, str]] = None) -> Path:
    """One machine-local root, by paths.local.json key. Never sealed."""
    paths = paths if paths is not None else resolve_paths()
    if key not in paths:
        raise JudgingProtocolError(
            f"{manifest.path.name} names path key {key!r}, which "
            f"configs/paths.local.json does not define")
    return Path(assert_not_sealed(paths[key], f"path key {key}"))


def stage_dir(manifest: Manifest, name: str,
              paths: Optional[Mapping[str, str]] = None,
              root: Optional[Path] = None) -> Path:
    """Where one stage writes, from the manifest's declared output layout."""
    if name not in manifest.outputs:
        raise JudgingProtocolError(
            f"{manifest.path.name}: outputs.{name} is not declared")
    base = (root if root is not None
            else resolve_root(manifest, str(manifest.outputs["root_key"]),
                              paths))
    return Path(base) / str(manifest.outputs[name])


# ================================================================== integrity
def verify_inputs(manifest: Manifest, checks: Checklist,
                  paths: Optional[Mapping[str, str]] = None,
                  root: Optional[Path] = None) -> Dict[str, str]:
    """Every declared input file, re-hashed from disk. No exceptions."""
    seen: Dict[str, str] = {}
    for name, spec in sorted(manifest.inputs.items()):
        if name == "paths_key":
            continue
        base = (root if root is not None
                else resolve_root(manifest, str(manifest.inputs["paths_key"]),
                                  paths))
        path = Path(base) / str(spec["path"])
        if not path.is_file():
            raise JudgingProtocolError(f"declared input {name} is absent: "
                                       f"{path}")
        digest = sha256_file(path)
        if digest != str(spec["sha256"]):
            raise JudgingProtocolError(
                f"{name} at {path} hashes {digest[:12]}, and the manifest "
                f"pins {str(spec['sha256'])[:12]}. A drifted input is "
                f"refused.")
        seen[name] = digest
        checks.record(f"input_{name}", True, f"{digest[:12]} verified")
    return seen


def load_rubric_text(manifest: Manifest, rubric: RubricSpec,
                     paths: Optional[Mapping[str, str]] = None,
                     root: Optional[Path] = None) -> Dict[str, Any]:
    """The frozen rubric text, refused unless all three digests match."""
    base = (root if root is not None
            else resolve_root(manifest, rubric.protocol_dir_key, paths))
    directory = Path(base) / rubric.subdir
    files = {"instructions": f"{rubric.name}_instructions.txt",
             "user_template": f"{rubric.name}_user_template.txt",
             "schema": f"{rubric.name}_schema.json"}
    out: Dict[str, Any] = {}
    for kind, filename in files.items():
        path = directory / filename
        if not path.is_file():
            raise JudgingProtocolError(
                f"the frozen {rubric.name} {kind} is absent: {path}")
        text = path.read_text(encoding="utf-8")
        digest = sha256_text(text)
        want = getattr(rubric, f"{kind}_sha256")
        if digest != want:
            raise JudgingProtocolError(
                f"{path.name} hashes {digest[:12]} and the manifest pins "
                f"{want[:12]}. Frozen rubric text is never edited in place.")
        out[kind] = json.loads(text) if kind == "schema" else text
    return out


# ================================================================== blindness
def project_row(row: Mapping[str, Any], rubric: RubricSpec
                ) -> Dict[str, Any]:
    """A judge row, built by whitelist. Anything else cannot come along."""
    missing = [f for f in rubric.row_fields if f not in row]
    if missing:
        raise JudgingProtocolError(
            f"{rubric.name} row is missing {missing}")
    return {f: row[f] for f in rubric.row_fields}


def assert_row_is_blind(row: Mapping[str, Any], rubric: RubricSpec,
                        forbidden_ids: Iterable[str] = ()) -> None:
    """A judge row carries exactly its whitelist, and names no corpus row."""
    extra = sorted(set(row) - set(rubric.row_fields))
    if extra:
        raise JudgingProtocolError(
            f"{row.get('row_id')}: judge row carries {extra}, and a judge row "
            f"carries exactly {list(rubric.row_fields)}")
    assert_opaque(str(row["row_id"]), rubric.row_prefix)
    blob = json.dumps(row, sort_keys=True)
    for corpus_id in forbidden_ids:
        if corpus_id and str(corpus_id) in blob:
            raise JudgingProtocolError(
                f"{row.get('row_id')}: the judge row names corpus row "
                f"{corpus_id!r}")


def draw_orientation() -> Dict[str, str]:
    """Which origin sits in A and which in B, drawn per row from ``secrets``.

    Not from a salt and not from the row id: an orientation recoverable from
    the manifest is not blind to anyone holding the manifest.
    """
    import secrets as _secrets
    if _secrets.randbits(1):
        return {POSITION_A: ORIGIN_CANDIDATE, POSITION_B: ORIGIN_REFERENCE}
    return {POSITION_A: ORIGIN_REFERENCE, POSITION_B: ORIGIN_CANDIDATE}


# ============================================================== cell identity
CELL_FIELDS: Tuple[str, ...] = ("screen_row_id", "arm_id", "rubric")


def cell_key(entry: Mapping[str, Any], label: str = "cell"
             ) -> Tuple[str, str, str]:
    missing = [f for f in CELL_FIELDS if f not in entry]
    if missing:
        raise JudgingProtocolError(f"{label} is missing {missing}")
    return (str(entry["screen_row_id"]), str(entry["arm_id"]),
            str(entry["rubric"]))


def index_cells(records: Iterable[Mapping[str, Any]], label: str
                ) -> Dict[Tuple[str, str, str], Mapping[str, Any]]:
    out: Dict[Tuple[str, str, str], Mapping[str, Any]] = {}
    for record in records:
        key = cell_key(record, label)
        if key in out:
            raise JudgingProtocolError(
                f"{label}: {key} appears twice. One cell is one row.")
        out[key] = record
    return out


# ============================================================== vocabulary
def register_rubric_vocabulary(
        manifest: Manifest) -> Dict[str, Tuple[str, ...]]:
    """Teach ``triplets.AXIS_VALUES`` this manifest's rubric vocabularies.

    Additive and idempotent. The point is that a human verdict is validated
    through *exactly* the table every other verdict is validated through --
    ``routegate.focusedjudging.adjudicate_cell`` reads ``AXIS_VALUES[axis]``
    -- rather than through a second table this engine keeps for itself.

    Re-registering the same values is a no-op. Registering *different* values
    under a name already in the table is refused: that would silently change
    what an earlier experiment's verdicts were allowed to be.
    """
    from . import triplets as _triplets
    registered: Dict[str, Tuple[str, ...]] = {}
    for name, rubric in sorted(manifest.rubrics.items()):
        existing = _triplets.AXIS_VALUES.get(name)
        if existing is not None and tuple(existing) != rubric.values:
            raise JudgingProtocolError(
                f"rubric {name!r} is already registered with "
                f"{list(existing)} and this manifest declares "
                f"{list(rubric.values)}. A vocabulary is never redefined; "
                f"give the new rubric a new name.")
        _triplets.AXIS_VALUES[name] = rubric.values
        registered[name] = rubric.values
    return registered


# ================================================================ validation
def validate_reply(parsed: Any, rubric: RubricSpec, row_id: str
                   ) -> Dict[str, Any]:
    """One parsed reply against its rubric. No fallback, no repair.

    The item-id echo is mandatory and is checked here for every rubric, single
    or pairwise. A reply that echoes a different id may belong to another row,
    and this engine has no path that decides which.
    """
    if not isinstance(parsed, Mapping):
        raise JudgingProtocolError(
            f"{row_id}: the judge returned {type(parsed).__name__}, not an "
            f"object. Recorded as a failure; not converted into a verdict.")
    for required in ("item_id",) + rubric.verdict_fields:
        if required not in parsed:
            raise JudgingProtocolError(
                f"{row_id}: the reply has no {required!r} field")
    if str(parsed["item_id"]) != str(row_id):
        raise JudgingProtocolError(
            f"{row_id}: the reply echoes item_id {parsed['item_id']!r}. A "
            f"mismatched echo means the reply may belong to another row; it "
            f"is never repaired and never accepted.")
    values: Dict[str, Any] = {"item_id": row_id}
    for field_name in rubric.verdict_fields:
        value = parsed[field_name]
        if (str(value) not in rubric.accepted_values
                and len(rubric.verdict_fields) == 1):
            raise JudgingProtocolError(
                f"{row_id}: {field_name}={value!r} is not one of "
                f"{list(rubric.values)}")
        values[field_name] = str(value)
    composite = compose_verdict(values, rubric)
    if composite not in rubric.accepted_values:
        raise JudgingProtocolError(
            f"{row_id}: {composite!r} is not one of "
            f"{list(rubric.accepted_values)}")
    confidence = parsed.get("confidence")
    if "confidence" in parsed and confidence not in CONFIDENCE_VALUES:
        raise JudgingProtocolError(
            f"{row_id}: confidence={parsed['confidence']!r} is not one of "
            f"{list(CONFIDENCE_VALUES)}")
    values["reason"] = str(parsed.get("reason", ""))[:240]
    if "confidence" in parsed:
        values["confidence"] = parsed["confidence"]
    return values


COMPOSITE_SEPARATOR = "|"


def compose_verdict(values: Mapping[str, Any], rubric: RubricSpec) -> str:
    """One verdict string for a cell, composite when the rubric has parts.

    A rubric with two fields decides as a pair: two judges agree only when
    both parts match. Composing here, once, is what makes that true everywhere
    downstream rather than in each comparison site.
    """
    parts = [str(values[f]) for f in rubric.verdict_fields]
    return COMPOSITE_SEPARATOR.join(parts)


def decode_pairwise(verdict: str, orientation: Mapping[str, str],
                    rubric: RubricSpec) -> str:
    """Turn an A/B answer into a candidate/reference one, per row.

    A bijection applied identically to both judges, so it preserves agreement
    exactly: two judges agree after decoding if and only if they agreed
    before it.
    """
    if not rubric.pairwise:
        return verdict
    mapping = {POSITION_A: orientation[POSITION_A],
               POSITION_B: orientation[POSITION_B]}
    parts = verdict.split(COMPOSITE_SEPARATOR)
    head = parts[0]
    if head.startswith("A_"):
        head = f"{mapping[POSITION_A].upper()}_{head[2:]}"
    elif head.startswith("B_"):
        head = f"{mapping[POSITION_B].upper()}_{head[2:]}"
    decoded = COMPOSITE_SEPARATOR.join([head] + parts[1:])
    if decoded not in rubric.values:
        raise JudgingProtocolError(
            f"{verdict!r} decodes to {decoded!r}, which is not one of "
            f"{list(rubric.values)}. The judge and decoded vocabularies do "
            f"not line up; a reading must never see an undecoded verdict.")
    return decoded


# ===================================================================== output
def assert_output_dir_is_new(out_dir: Path,
                             protected: Sequence[Optional[Path]] = (),
                             label: str = "this stage") -> None:
    """Write somewhere new, and never over what this stage reads."""
    out_dir = Path(out_dir)
    for candidate in protected:
        if candidate is None:
            continue
        target = Path(candidate)
        target = target if target.is_dir() else target.parent
        if target.exists() and out_dir.resolve() == target.resolve():
            raise JudgingProtocolError(
                f"--output-dir resolves to {target}, which {label} reads. "
                f"That input is not reconstructible; it is never written "
                f"over.")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise JudgingProtocolError(
            f"{out_dir} already holds files. {label} writes into a new "
            f"directory so a previous result is never half-overwritten.")


def assert_packet_new(packet_dir: Path, runs_dir: Path) -> None:
    """A packet is built once. After a judge has run, it is frozen forever.

    Rebuilding mints new opaque ids and, for a pairwise rubric, a new A/B
    orientation -- which would orphan every verdict already paid for.
    """
    packet_dir, runs_dir = Path(packet_dir), Path(runs_dir)
    if runs_dir.exists() and any(runs_dir.iterdir()):
        raise JudgingProtocolError(
            f"{runs_dir} already holds judging runs, so {packet_dir} is "
            f"frozen. Rebuilding the packet would mint new opaque ids and a "
            f"new A/B orientation, orphaning every verdict already billed.")
    assert_output_dir_is_new(packet_dir, (runs_dir,), "the packet build")


def preserve_raw(out_dir: Path, rel_path: str, reply: Mapping[str, Any]
                 ) -> str:
    """Persist one raw body, mode 0600. An existing body is never replaced."""
    path = Path(out_dir) / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise JudgingProtocolError(
            f"{path} already holds a preserved response. Raw evidence is "
            f"immutable; the attempt path is chosen so it cannot collide.")
    payload = json.dumps({"http_status": reply.get("http_status"),
                          "response_headers": reply.get("response_headers"),
                          "raw_body": reply.get("raw_body", "")},
                         sort_keys=True, ensure_ascii=False)
    path.write_text(payload, encoding="utf-8")
    path.chmod(0o600)
    return sha256_text(payload)


def already_done(results_path: Path) -> Dict[str, str]:
    """``row_id -> record digest`` for rows already answered successfully.

    Resumability rests on this: a row present here is never sent again, so a
    successful billed call is never repeated.
    """
    out: Dict[str, str] = {}
    path = Path(results_path)
    if not path.is_file():
        return out
    for record in read_jsonl(path):
        out[result_join_key(record)] = str(record.get("record_sha256", ""))
    return out


__all__ = [
    "ADJUDICATED_CELLS_FILE", "COMPOSITE_SEPARATOR", "DISAGREEMENT_POLICY",
    "FAILURE_POLICY", "JOINED_CELLS_FILE", "JudgingProtocolError", "Manifest",
    "MANIFEST_FILENAME", "OFFLINE_STAGES", "ORIGIN_CANDIDATE",
    "ORIGIN_REFERENCE", "PAIRWISE_ROW_FIELDS", "POSITION_A", "POSITION_B",
    "PROCEDURES", "REPORT_FILE", "REVIEW_COMPLETED_FILE", "REVIEW_INPUT_FILE",
    "REVIEW_MAP_FILE", "REVIEW_TEMPLATE_FILE", "ROW_MAP_FILE", "RESULTS_FILE",
    "FAILURES_FILE", "RubricSpec", "SCHEMA", "SINGLE_ROW_FIELDS", "STAGES",
    "STATE_AGREE", "STATE_DISAGREE", "STATE_INCOMPLETE", "STATE_MISSING",
    "STATE_UNRESOLVED_TECHNICAL", "TERMINAL_STATES", "already_done",
    "assert_no_secrets", "assert_output_dir_is_new", "assert_packet_new",
    "assert_row_is_blind", "cell_key", "compose_verdict", "decode_pairwise",
    "draw_orientation", "expected_coverage", "index_cells", "load_manifest",
    "register_rubric_vocabulary",
    "load_rubric_text", "preserve_raw", "project_row", "resolve_paths",
    "resolve_root", "stage_dir", "validate_reply", "verify_inputs",
]
