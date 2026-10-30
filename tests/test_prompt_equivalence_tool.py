"""The step-9 prompt-equivalence logic, exercised without the real inputs.

The one-time run needs the historical workspace, the pinned tokeniser and the
frozen manifest. None of those belong in the normal test suite, so everything
here uses synthetic doubles: a fabricated 480-row manifest built in ``tmp_path``
with invented prompt text, stub chat builders, and a stub tokeniser.

What that buys is the part worth testing anyway -- the code that decides PASS or
FAIL. A comparison harness that cannot fail is worthless, so most of these tests
inject a deliberate drift and require it to be caught: a changed byte, a changed
token, a longer attention mask, a different final token.

CPU-only, offline, no model, no GPU, no network, no ``~/SteerMoE-main``, and no
Hugging Face cache.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "migration" / "prompt_equivalence_check.py"


def _socket_state():
    import socket
    return (socket.socket.connect, socket.socket.connect_ex, socket.create_connection)


def _restore_sockets(state):
    import socket
    (socket.socket.connect, socket.socket.connect_ex,
     socket.create_connection) = state


def _tool(restore_sockets=True):
    """Load the tool by path. It is not an installed package.

    Registered in ``sys.modules`` before execution because the module uses
    ``from __future__ import annotations`` with dataclasses, and the dataclass
    machinery resolves annotations through the module entry.

    Importing it installs the outbound-connection guard process-wide, which is
    the intended behaviour for the one-time run but would leak into the rest of
    the suite, so the sockets are restored afterwards by default. One test loads
    it with ``restore_sockets=False`` to prove the guard really is installed at
    import time.
    """
    saved = _socket_state()
    spec = importlib.util.spec_from_file_location("prompt_equivalence_check", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)     # stdlib + routegate only; nothing heavy
    finally:
        if restore_sockets:
            _restore_sockets(saved)
    return mod


T = _tool()


# ------------------------------------------------------------------ doubles
class StubTokenizer:
    """Renders a template and tokenises by codepoint. Deterministic, tiny."""

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, enable_thinking=False):
        if tokenize:
            raise AssertionError("step 9 renders to text, never to ids here")
        body = messages[0]["content"]
        return f"<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n"

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        ids = [ord(c) % 1000 for c in text]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def build_ok(tokenizer, prompt):
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}])


def tokenize(tokenizer, text):
    return tokenizer(text)


@pytest.fixture
def prompts():
    return [f"synthetic development prompt number {i}" for i in range(12)]


@pytest.fixture
def tok():
    return StubTokenizer()


# ------------------------------------------------------- digests are honest
def test_ordered_digest_is_order_sensitive():
    assert T.ordered_digest(["a", "b"]) != T.ordered_digest(["b", "a"])


def test_ordered_digest_cannot_be_forged_by_concatenation():
    """"ab" + "c" must not collide with "a" + "bc"."""
    assert T.ordered_digest(["ab", "c"]) != T.ordered_digest(["a", "bc"])


def test_token_sequence_digest_distinguishes_sequences():
    assert T.token_sequence_digest([1, 2, 3]) != T.token_sequence_digest([1, 3, 2])
    assert T.token_sequence_digest([1, 2]) != T.token_sequence_digest([12])


def test_as_list_accepts_flat_nested_and_tolist_forms():
    class Tensorish:
        def tolist(self):
            return [[7, 8, 9]]
    assert T._as_list([1, 2]) == [1, 2]
    assert T._as_list([[1, 2]]) == [1, 2]
    assert T._as_list(Tensorish()) == [7, 8, 9]


# --------------------------------------------------- the comparison harness
def test_identical_builders_compare_clean(prompts, tok):
    r = T.compare_prompts(prompts, build_ok, build_ok, tok, tokenize)
    assert r.ok and r.rendered_ok and r.tokens_ok
    assert r.n_compared == len(prompts)
    assert r.mismatch_rows() == []
    assert len(r.rendered_digest) == 64 and len(r.token_digest) == 64


def test_a_single_changed_byte_is_caught(prompts, tok):
    def drifted(tokenizer, prompt):
        text = build_ok(tokenizer, prompt)
        return text.replace("user", "user ", 1) if prompt.endswith("3") else text

    r = T.compare_prompts(prompts, build_ok, drifted, tok, tokenize)
    assert not r.ok
    assert r.rendered_byte_mismatches == [3]
    assert r.rendered_sha_mismatches == [3]
    assert r.token_id_mismatches == [3]       # and it moves the tokens too
    assert r.mismatch_rows() == [3]


def test_a_missing_generation_prompt_is_caught(prompts, tok):
    """The classic drift: add_generation_prompt quietly off.

    Worth reading closely. Both renderings still end in a newline, so the
    *final token is identical* and a check that looked only at position -1
    would pass this. It is caught because the full sequence is compared, which
    is precisely why check 7 is not just "the last id agrees".
    """
    def no_gen_prompt(tokenizer, prompt):
        return build_ok(tokenizer, prompt).replace("<|im_start|>assistant\n", "")

    r = T.compare_prompts(prompts, build_ok, no_gen_prompt, tok, tokenize)
    assert not r.ok
    assert len(r.rendered_byte_mismatches) == len(prompts)
    assert len(r.token_id_mismatches) == len(prompts)
    assert r.final_token_mismatches == []        # the last id coincides


def test_identical_text_but_different_tokens_is_caught(prompts, tok):
    """Check 1 cannot see this; checks 3-5 exist for exactly this case."""
    class ShiftedTokenizer(StubTokenizer):
        def __call__(self, text, return_tensors=None, add_special_tokens=False):
            ids = [ord(c) % 1000 for c in text]
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    calls = {"n": 0}

    def alternating_tokenize(tokenizer, text):
        calls["n"] += 1
        enc = tokenizer(text)
        if calls["n"] % 2 == 0:                     # the runtime side only
            enc = {"input_ids": enc["input_ids"][:-1] + [999999],
                   "attention_mask": enc["attention_mask"]}
        return enc

    r = T.compare_prompts(prompts, build_ok, build_ok, ShiftedTokenizer(),
                          alternating_tokenize)
    assert r.rendered_ok                            # text is byte-identical
    assert not r.tokens_ok                          # but the ids are not
    assert len(r.token_id_mismatches) == len(prompts)
    assert len(r.final_token_mismatches) == len(prompts)


def test_a_padded_attention_mask_is_caught(prompts, tok):
    def padding_tokenize(tokenizer, text):
        enc = tokenizer(text)
        if text.endswith("assistant\n"):
            return {"input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"] + [0]}
        return enc

    calls = {"n": 0}

    def one_side_pads(tokenizer, text):
        calls["n"] += 1
        return padding_tokenize(tokenizer, text) if calls["n"] % 2 == 0 \
            else tokenizer(text)

    r = T.compare_prompts(prompts, build_ok, build_ok, tok, one_side_pads)
    assert not r.tokens_ok
    assert len(r.attention_length_mismatches) == len(prompts)
    assert len(r.attention_mask_mismatches) == len(prompts)


def test_a_non_string_render_is_a_failure_not_a_crash(prompts, tok):
    def returns_none(tokenizer, prompt):
        return None

    r = T.compare_prompts(prompts, build_ok, returns_none, tok, tokenize)
    assert not r.ok
    assert r.rendered_type_failures == list(range(len(prompts)))


def test_comparison_carries_no_prompt_or_token_material(prompts, tok):
    """Positions and digests only -- never the text or the ids."""
    r = T.compare_prompts(prompts, build_ok, build_ok, tok, tokenize)
    blob = json.dumps(r.__dict__, default=str)
    assert "synthetic development prompt" not in blob
    assert "im_start" not in blob


# ------------------------------------------------------------- input gates
def _write_manifest(path, *, rows=480, counts=(89, 178, 213),
                    break_hash_at=None, break_order=False, extra_column=None):
    labels = ["E"] * counts[0] + ["B"] * counts[1] + ["H"] * counts[2]
    labels = (labels + ["H"] * rows)[:rows]
    columns = list(T.EXPECTED_COLUMNS) + ([extra_column] if extra_column else [])
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for i in range(rows):
            text = f"synthetic prompt {i}"
            digest = hashlib.sha256(text.encode()).hexdigest()
            if break_hash_at == i:
                digest = "0" * 64
            row = {"row_index": rows - 1 - i if break_order else i,
                   "row_id": f"row_{i:04d}", "final_ebh": labels[i],
                   "source_family": "synthetic", "prompt_text": text,
                   "prompt_sha256": digest}
            if extra_column:
                row[extra_column] = "x"
            writer.writerow(row)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def manifest(tmp_path):
    path = tmp_path / "manifest.csv"
    return path, _write_manifest(path)


def test_a_correct_manifest_passes_every_gate(manifest):
    path, sha = manifest
    gate = T.gate_manifest(path, sha, "manifest.csv")
    assert gate.ok, [c for c in gate.checks if not c["ok"]]
    assert gate.n_rows == 480
    assert gate.counts == {"E": 89, "B": 178, "H": 213}
    assert len(gate.prompts) == 480


def test_a_wrong_hash_is_refused(manifest):
    path, _ = manifest
    gate = T.gate_manifest(path, "0" * 64, "manifest.csv")
    assert not gate.ok
    assert not _named(gate, "manifest_sha256_matches_the_pinned_value")


def test_a_wrong_filename_is_refused(tmp_path):
    path = tmp_path / "not_the_manifest.csv"
    sha = _write_manifest(path)
    gate = T.gate_manifest(path, sha, "manifest.csv")
    assert not _named(gate, "manifest_filename_is_the_pinned_one")


def test_the_wrong_row_count_is_refused(tmp_path):
    path = tmp_path / "manifest.csv"
    sha = _write_manifest(path, rows=479)
    gate = T.gate_manifest(path, sha, "manifest.csv")
    assert not _named(gate, "manifest_has_exactly_480_rows")


def test_out_of_order_row_index_is_refused(tmp_path):
    path = tmp_path / "manifest.csv"
    sha = _write_manifest(path, break_order=True)
    gate = T.gate_manifest(path, sha, "manifest.csv")
    assert not _named(gate, "row_index_is_0_to_479_in_order")


def test_wrong_class_counts_are_refused(tmp_path):
    path = tmp_path / "manifest.csv"
    sha = _write_manifest(path, counts=(90, 177, 213))
    gate = T.gate_manifest(path, sha, "manifest.csv")
    assert not _named(gate, "class_counts_are_E89_B178_H213")


def test_a_single_wrong_prompt_hash_is_refused(tmp_path):
    path = tmp_path / "manifest.csv"
    sha = _write_manifest(path, break_hash_at=317)
    gate = T.gate_manifest(path, sha, "manifest.csv")
    assert not _named(gate,
                      "every_prompt_sha256_recomputes_from_its_own_text")


@pytest.mark.parametrize("column", ["response", "response_text", "generation",
                                    "model_output", "after_text"])
def test_a_response_column_is_refused(tmp_path, column):
    path = tmp_path / "manifest.csv"
    sha = _write_manifest(path, extra_column=column)
    gate = T.gate_manifest(path, sha, "manifest.csv")
    assert not _named(gate, "manifest_carries_no_response_column")
    assert not _named(gate, "manifest_columns_are_exactly_the_expected_six")


def test_a_sealed_looking_path_is_refused(tmp_path):
    """A generic sealed-looking name, deliberately not the real one.

    The guard matches generic tokens precisely so that neither it nor its tests
    have to record where the held-out material actually lives.
    """
    from routegate.validation import SealedSetError
    sealed = tmp_path / "sealed_holdout_manifest.csv"
    _write_manifest(sealed)
    with pytest.raises(SealedSetError):
        T.gate_manifest(sealed, "0" * 64, "manifest.csv")


def _named(gate, name):
    return next(c["ok"] for c in gate.checks if c["check"] == name)


# ------------------------------------------------------- constants + evidence
class _Legacy:
    MODEL = "Qwen/Qwen3-30B-A3B"
    FROZEN_REVISION = "ad44e777"
    FROZEN_TOKENIZER_CONFIG_SHA256 = "aa"
    FROZEN_CHAT_TEMPLATE_SHA256 = "bb"


class _Runtime:
    FROZEN_MODEL = "Qwen/Qwen3-30B-A3B"
    FROZEN_REVISION = "ad44e777"
    FROZEN_TOKENIZER_CONFIG_SHA256 = "aa"
    FROZEN_CHAT_TEMPLATE_SHA256 = "bb"


def test_matching_constants_pass():
    assert all(c["ok"] for c in T.compare_constants(_Legacy(), _Runtime()))


def test_a_drifted_revision_is_caught():
    class Drifted(_Runtime):
        FROZEN_REVISION = "deadbeef"
    checks = T.compare_constants(_Legacy(), Drifted())
    assert not _check(checks, "frozen_revision_agrees")


def test_a_drifted_model_is_caught():
    class Drifted(_Runtime):
        FROZEN_MODEL = "Qwen/Qwen3-8B"
    checks = T.compare_constants(_Legacy(), Drifted())
    assert not _check(checks, "model_identifier_agrees")


def test_snapshot_checks_require_the_same_dir_and_pinned_commit(tmp_path):
    snap, other = tmp_path / "s", tmp_path / "o"
    snap.mkdir(); other.mkdir()
    ok = T.compare_snapshots(("rev1", snap), ("rev1", snap), "rev1",
                             "aa", "aa", "bb", "bb")
    assert all(c["ok"] for c in ok)
    diff_dir = T.compare_snapshots(("rev1", snap), ("rev1", other), "rev1",
                                   "aa", "aa", "bb", "bb")
    assert not _check(diff_dir, "both_resolve_the_same_snapshot_at_the_pinned_commit")
    unpinned = T.compare_snapshots(("rev2", snap), ("rev2", snap), "rev1",
                                   "aa", "aa", "bb", "bb")
    assert not _check(unpinned, "both_resolve_the_same_snapshot_at_the_pinned_commit")
    tok_drift = T.compare_snapshots(("rev1", snap), ("rev1", snap), "rev1",
                                    "aa", "zz", "bb", "bb")
    assert not _check(tok_drift, "tokenizer_config_sha256_agrees")
    tpl_drift = T.compare_snapshots(("rev1", snap), ("rev1", snap), "rev1",
                                    "aa", "aa", "bb", "zz")
    assert not _check(tpl_drift, "chat_template_sha256_agrees")


def _check(checks, name):
    return next(c["ok"] for c in checks if c["check"] == name)


def test_seven_checks_are_exactly_seven(prompts, tok):
    r = T.compare_prompts(prompts, build_ok, build_ok, tok, tokenize)
    checks = T.seven_checks(T.compare_constants(_Legacy(), _Runtime()),
                            T.compare_snapshots(("r", Path(".")), ("r", Path(".")),
                                                "r", "a", "a", "b", "b"),
                            r, expected_rows=len(prompts))
    assert len(checks) == 7
    assert all(c["ok"] for c in checks)


def test_a_short_run_cannot_pass(prompts, tok):
    """479 of 480 is a failure. There is no 'almost' outcome."""
    r = T.compare_prompts(prompts[:-1], build_ok, build_ok, tok, tokenize)
    checks = T.seven_checks(T.compare_constants(_Legacy(), _Runtime()),
                            T.compare_snapshots(("r", Path(".")), ("r", Path(".")),
                                                "r", "a", "a", "b", "b"),
                            r, expected_rows=len(prompts))
    assert not _check(checks, "all_prompts_render_to_identical_bytes_and_sha256")
    assert not _check(checks, "all_prompts_tokenise_to_identical_sequences")


@pytest.fixture
def evidence(prompts, tok, manifest):
    path, sha = manifest
    gate = T.gate_manifest(path, sha, "manifest.csv")
    r = T.compare_prompts(prompts, build_ok, build_ok, tok, tokenize)
    checks = T.seven_checks(T.compare_constants(_Legacy(), _Runtime()),
                            T.compare_snapshots(("r", Path(".")), ("r", Path(".")),
                                                "r", "a", "a", "b", "b"),
                            r, expected_rows=len(prompts))
    return T.build_evidence(
        checks=checks, gate=gate, comparison=r,
        legacy_module_sha256="c" * 64, legacy_module_filename="rq3_dev_generation.py",
        runtime_sha256="d" * 64, model="Qwen/Qwen3-30B-A3B", revision="ad44e777",
        resolved_commit="ad44e777", tokenizer_config_sha256="a" * 64,
        chat_template_sha256="b" * 64, expected_rows=len(prompts))


def test_evidence_reports_pass_and_every_required_field(evidence):
    assert evidence["verdict"] == "PASS"
    assert evidence["n_checks"] == 7 and evidence["n_failed_checks"] == 0
    src, pin, agg = evidence["sources"], evidence["pinned_runtime"], evidence["aggregate_digests"]
    for key in ("legacy_module_sha256", "routegate_runtime_sha256",
                "extraction_manifest_sha256"):
        assert src[key]
    for key in ("model", "revision", "resolved_snapshot_commit",
                "tokenizer_config_sha256", "chat_template_sha256"):
        assert pin[key]
    assert agg["rendered_text_sha256_sequence"] and agg["token_id_sequence"]
    assert evidence["versions"]["python"]
    assert evidence["comparison"]["n_compared"] == evidence["comparison"]["rendered_identical"]


def test_evidence_leaks_no_prompt_token_or_path_material(evidence, prompts):
    blob = json.dumps(evidence)
    for forbidden in ("synthetic prompt", "synthetic development prompt",
                      "im_start", "/home/", "SteerMoE"):
        assert forbidden not in blob, forbidden
    # Sealed-set *locators*. The literal word "sealed" is allowed, and appears
    # only in the not_done list asserting the split was never accessed.
    for locator in ("rq3_sealed", "sealed_final_manifest", "final_benchmark",
                    "rq3_prompt_splits"):
        assert locator not in blob, locator
    # Column *names* are schema, already documented in docs/standalone_migration.md,
    # and recording them is how the no-response-column gate shows its work.
    # Column *values* are the thing that must never appear.
    assert "row_0000" not in blob and "row_0479" not in blob
    # No per-row hashes: the record carries a bounded handful of digests (the
    # two aggregates plus the source and snapshot hashes), never 480 of them.
    digests = set(re.findall(r"\b[0-9a-f]{64}\b", blob))
    assert len(digests) < 20, f"{len(digests)} digests looks like per-row hashes"


def test_evidence_records_fail_when_a_check_fails(prompts, tok, manifest):
    path, sha = manifest
    gate = T.gate_manifest(path, sha, "manifest.csv")
    r = T.compare_prompts(prompts, build_ok, lambda t, p: build_ok(t, p) + "!",
                          tok, tokenize)
    checks = T.seven_checks(T.compare_constants(_Legacy(), _Runtime()),
                            T.compare_snapshots(("r", Path(".")), ("r", Path(".")),
                                                "r", "a", "a", "b", "b"),
                            r, expected_rows=len(prompts))
    ev = T.build_evidence(
        checks=checks, gate=gate, comparison=r, legacy_module_sha256="c" * 64,
        legacy_module_filename="rq3_dev_generation.py", runtime_sha256="d" * 64,
        model="m", revision="r", resolved_commit="r",
        tokenizer_config_sha256="a", chat_template_sha256="b",
        expected_rows=len(prompts))
    assert ev["verdict"] == "FAIL"
    assert ev["comparison"]["mismatch_counts"]["rendered_bytes"] == len(prompts)


# ------------------------------------------------------------ the no-net guard
def test_the_network_guard_refuses_outbound_connections():
    import socket as socket_module
    saved = _socket_state()
    try:
        T.block_network()
        with pytest.raises(T.NetworkAccessAttempted):
            socket_module.create_connection(("example.invalid", 80))
    finally:
        _restore_sockets(saved)


def test_importing_the_tool_installs_the_guard_unconditionally():
    """Not opt-in and not deferred to main(): installed at import.

    The re-executed module defines its own exception class, so the assertion is
    made against *that* module's class rather than the one already imported.
    """
    import socket as socket_module
    saved = _socket_state()
    try:
        fresh = _tool(restore_sockets=False)
        with pytest.raises(fresh.NetworkAccessAttempted):
            socket_module.create_connection(("example.invalid", 80))
    finally:
        _restore_sockets(saved)


def test_there_is_no_flag_to_re_enable_the_network():
    text = TOOL.read_text(encoding="utf-8")
    assert "--allow-network" not in text
    assert "allow_network" not in text
    with pytest.raises(SystemExit):
        T.parse_args(["--legacy-module", "/x/m.py", "--manifest", "/y/manifest.csv",
                      "--allow-network"])


def test_offline_env_vars_are_assigned_not_defaulted():
    """setdefault would let an inherited HF_HUB_OFFLINE=0 survive."""
    text = TOOL.read_text(encoding="utf-8")
    assert 'os.environ["HF_HUB_OFFLINE"] = "1"' in text
    assert 'os.environ["TRANSFORMERS_OFFLINE"] = "1"' in text
    # The word survives in the comment explaining why it is not used; what must
    # not survive is the call.
    assert "os.environ.setdefault" not in text
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr != "setdefault"


# ------------------------------------------------------- committed-code hygiene
def test_the_tool_hard_codes_no_machine_path():
    text = TOOL.read_text(encoding="utf-8")
    for needle in ("SteerMoE-main", "/home/ucabh21", "/myriadfs", "routegate-data"):
        assert needle not in text, needle


def test_the_tool_never_instantiates_a_causal_lm():
    """Structural, not textual: the docstring may name what it refuses to do."""
    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    imported, called = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                called.add(func.id)
            elif isinstance(func, ast.Attribute):
                called.add(func.attr)
    assert "AutoModelForCausalLM" not in imported
    assert "AutoModelForCausalLM" not in called
    assert "load_model_and_tokenizer" not in called
    assert "torch" not in imported
    # the tokeniser is the one thing it may pull from transformers
    assert "AutoTokenizer" in imported


def test_the_tool_requires_both_paths_as_arguments():
    with pytest.raises(SystemExit):
        T.parse_args([])
    args = T.parse_args(["--legacy-module", "/x/m.py", "--manifest", "/y/manifest.csv"])
    assert args.legacy_module == Path("/x/m.py")
    assert args.manifest == Path("/y/manifest.csv")


def test_the_tool_forces_offline_environment_variables():
    import os
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def test_the_tool_refuses_to_write_bytecode_into_the_legacy_workspace():
    """Importing from a read-only workspace must not leave .pyc files behind."""
    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    assigns = [n for n in ast.walk(tree)
               if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Attribute) and t.attr == "dont_write_bytecode"
                       for t in n.targets)]
    assert assigns, "sys.dont_write_bytecode is never set"
    assert all(a.value.value is True for a in assigns)


def test_importing_the_legacy_module_leaves_no_pycache(tmp_path):
    """End-to-end on a synthetic stand-in, so no real workspace is involved."""
    workspace = tmp_path / "legacy_workspace"
    workspace.mkdir()
    (workspace / "sibling_helper.py").write_text("VALUE = 7\n")
    (workspace / "fake_generation.py").write_text(
        "from sibling_helper import VALUE\n"
        "MODEL = 'Qwen/Qwen3-30B-A3B'\n"
        "FROZEN_REVISION = 'ad44e777'\n"
        "def build_chat_strict(tok, prompt):\n"
        "    return prompt\n")
    mod = T.import_legacy_module(workspace / "fake_generation.py",
                                 "legacy_under_test")
    assert mod.MODEL == "Qwen/Qwen3-30B-A3B"
    assert not list(workspace.rglob("__pycache__"))
    assert not list(workspace.rglob("*.pyc"))


# ------------------------------------- extraction-equivalent tokenisation
class RecordingTokenizer(StubTokenizer):
    """Renders like the stub, but records exactly how it was called."""

    def __init__(self):
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append({"text": text, "kwargs": dict(kwargs)})
        ids = [ord(c) % 1000 for c in text]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_tokenize_like_extraction_supplies_no_add_special_tokens_override():
    """The frozen extraction never passes it, so neither may this stage.

    Passing it -- even as the value that happens to be the default -- would
    compare a call the extraction does not make.
    """
    rec = RecordingTokenizer()
    T.tokenize_like_extraction(rec, "hello")
    assert len(rec.calls) == 1
    assert "add_special_tokens" not in rec.calls[0]["kwargs"]


def test_tokenize_like_extraction_passes_return_tensors_none():
    rec = RecordingTokenizer()
    T.tokenize_like_extraction(rec, "hello")
    assert rec.calls[0]["kwargs"] == {"return_tensors": None}


def test_the_comparison_uses_the_extraction_equivalent_helper(prompts):
    """End-to-end: no override reaches the tokeniser during a real comparison."""
    rec = RecordingTokenizer()
    T.compare_prompts(prompts, build_ok, build_ok, rec, T.tokenize_like_extraction)
    assert rec.calls, "the tokeniser was never called"
    assert all("add_special_tokens" not in c["kwargs"] for c in rec.calls)


def test_the_tool_never_writes_add_special_tokens_in_a_call():
    """Structural: no call anywhere in the tool passes the keyword."""
    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            assert not any(k.arg == "add_special_tokens" for k in node.keywords)


def test_the_extraction_and_runtime_really_do_omit_the_override():
    """Guards the premise this stage rests on, in the code being migrated."""
    extraction = (REPO_ROOT / "experiments/routegate_routing/extract_routing_v1.py"
                  ).read_text(encoding="utf-8")
    runtime = (REPO_ROOT / "src/routegate/runtime.py").read_text(encoding="utf-8")
    assert 'tokenizer(text, return_tensors="pt")' in extraction
    assert 'tokenizer(text, return_tensors="pt")' in runtime
    assert "add_special_tokens" not in extraction
    assert "add_special_tokens" not in runtime


# ------------------------------------------------- the pin source is fixed
def test_there_is_no_input_manifest_override():
    with pytest.raises(SystemExit):
        T.parse_args(["--legacy-module", "/x/m.py", "--manifest", "/y/manifest.csv",
                      "--input-manifest", "/z/other.json"])


def test_the_pin_comes_from_the_committed_manifest():
    spec = T.load_pinned_manifest_spec()
    assert spec["filename"] == "manifest.csv"
    assert re.fullmatch(r"[0-9a-f]{64}", spec["sha256"])
    assert T.DEFAULT_INPUT_MANIFEST == REPO_ROOT / "manifests" / "development_inputs.json"


def test_load_pinned_manifest_spec_takes_no_path_argument():
    import inspect
    assert list(inspect.signature(T.load_pinned_manifest_spec).parameters) == []


# ------------------------------------------- the failure log stays private
def test_a_failure_log_inside_the_repository_is_refused():
    with pytest.raises(T.EquivalenceError):
        T.resolve_failure_log(REPO_ROOT / "artifacts" / "mismatches.json")


def test_the_repository_root_itself_is_refused():
    with pytest.raises(T.EquivalenceError):
        T.resolve_failure_log(REPO_ROOT)


def test_a_failure_log_outside_the_repository_is_accepted(tmp_path):
    target = tmp_path / "mismatches.private.json"
    resolved = T.resolve_failure_log(target)
    assert resolved == target.resolve()


def test_no_failure_log_is_fine():
    assert T.resolve_failure_log(None) is None


def test_a_symlink_pointing_into_the_repository_is_refused(tmp_path):
    """Resolved before the check, so a link cannot smuggle it back inside."""
    link = tmp_path / "looks_outside.json"
    link.symlink_to(REPO_ROOT / "artifacts" / "sneaked.json")
    with pytest.raises(T.EquivalenceError):
        T.resolve_failure_log(link)
