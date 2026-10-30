"""Phase 2 step 10A -- the extraction no longer depends on SteerMoE.

``experiments/routegate_routing/extract_routing_v1.py`` was the last script
still importing ``rq3_dev_generation`` from the historical workspace. Step 10
replaced it with ``routegate.runtime``, gated on the step-9 prompt-equivalence
evidence.

Two things are tested here. First, that the dependency is genuinely gone --
structurally, so a docstring recording the provenance of the change does not
count as a dependency and does not have to be deleted to make a test pass.
Second, that the fresh-smoke guard and the comparison tool behave: the guard
must refuse anything that could touch a historical output, and the comparator
must accept exactly the documented volatile differences and nothing else.

CPU-only, offline. No model, no GPU, no API, no sealed data, no
``~/SteerMoE-main``, no Hugging Face cache. The extraction module is read as
text and parsed, never imported -- importing it would pull in torch.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACTION = REPO_ROOT / "experiments/routegate_routing/extract_routing_v1.py"
COMPARATOR = REPO_ROOT / "tools/migration/compare_smoke_extractions.py"
QSUB = REPO_ROOT / "experiments/routegate_routing/submit_fresh_smoke_equivalence.sh"


def extraction_source() -> str:
    return EXTRACTION.read_text(encoding="utf-8")


def _comparator():
    spec = importlib.util.spec_from_file_location("compare_smoke_extractions", COMPARATOR)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _comparator()


# ------------------------------------------- the SteerMoE dependency is gone
def test_the_extraction_names_no_machine_path():
    text = extraction_source()
    for needle in ("SteerMoE-main", "/home/ucabh21", "/myriadfs", "routegate-data"):
        assert needle not in text, needle


def test_the_extraction_does_not_import_the_historical_module():
    """Structural: no import statement, however the name is spelled."""
    tree = ast.parse(extraction_source())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all("rq3_dev_generation" not in a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert "rq3_dev_generation" not in (node.module or "")


def test_the_extraction_inserts_no_foreign_path_on_sys_path():
    """Only RouteGate's own src/ may be added."""
    tree = ast.parse(extraction_source())
    inserts = [n for n in ast.walk(tree)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "insert"
               and isinstance(n.func.value, ast.Attribute)
               and n.func.value.attr == "path"]
    assert len(inserts) == 1, "exactly one sys.path insertion is expected"
    assert "RG / \"src\"" in ast.unparse(inserts[0]) or "src" in ast.unparse(inserts[0])


def test_the_extraction_defines_no_SM_constant():
    tree = ast.parse(extraction_source())
    assigned = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            assigned.update(t.id for t in node.targets if isinstance(t, ast.Name))
    assert "SM" not in assigned


def test_the_extraction_derives_the_repository_from_dunder_file():
    assert 'RG = Path(__file__).resolve().parents[2]' in extraction_source()


def test_the_extraction_uses_routegate_runtime():
    text = extraction_source()
    assert "from routegate import runtime" in text
    for call in ("runtime.resolve_snapshot()", "runtime.load_model_and_tokenizer()",
                 "runtime.build_chat_strict(tokenizer,"):
        assert call in text, call
    assert "runtime.FROZEN_MODEL" in text
    assert "G.MODEL" not in text


def test_the_extraction_uses_the_shared_input_loaders():
    text = extraction_source()
    assert "from routegate.data import load_input_manifest, load_local_paths" in text
    assert "_PATHS[\"extraction_dir\"]" in text
    assert "_PATHS[\"dataset\"]" in text
    assert '_MANIFEST_SPEC["extraction_manifest"]["filename"]' in text


def test_the_provenance_change_is_recorded_not_disguised():
    """The config must name the module that actually renders the prompt."""
    text = extraction_source()
    assert 'prompt_config="routegate.runtime.build_chat_strict' in text
    assert 'prompt_config="rq3_dev_generation' not in text


# --------------------------------------------- the scientific settings hold
@pytest.mark.parametrize("expected", [
    'TOKEN_POSITION = -1',
    'ROWS = 480',
    'EBH_COUNTS = {"E": 89, "B": 178, "H": 213}',
    'dtype="bfloat16", attn_implementation="eager", device_map="auto"',
    'use_cache=False, inference_mode=True, model_eval=True',
    'generation="none (model.generate is disarmed)"',
    'steering="none", router_modification="none", expert_suppression="none"',
    'next(r for r in rows if r["final_ebh"] == "E")',
    'next(r for r in rows if r["final_ebh"] == "H")',
])
def test_scientific_settings_are_unchanged(expected):
    assert expected in extraction_source()


def test_the_full_mode_still_writes_to_the_frozen_full_directory():
    text = extraction_source()
    assert 'return run("full", OUT_FULL, rows, a.resume)' in text
    assert 'OUT_FULL = RG / "outputs/routing_extraction_v1"' in text


# ---------------------------------------------------- the fresh-smoke guard
@pytest.fixture
def resolve(tmp_path):
    """The guard, lifted out of the module without importing torch."""
    tree = ast.parse(extraction_source())
    wanted = {"resolve_smoke_output_dir", "fail"}
    nodes = [n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert len(nodes) == 2
    ns = {"Path": Path, "Any": object, "Tuple": tuple,
          "PROTECTED_OUTPUT_DIRS": (),      # supplied per test
          "ExtractionError": type("ExtractionError", (RuntimeError,), {})}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<guard>", "exec"), ns)
    return ns


def test_smoke_output_dir_is_rejected_without_smoke(resolve, tmp_path):
    with pytest.raises(RuntimeError, match="only valid with --smoke"):
        resolve["resolve_smoke_output_dir"](tmp_path / "fresh", False, ())


def test_an_existing_directory_is_rejected(resolve, tmp_path):
    existing = tmp_path / "already_here"
    existing.mkdir()
    with pytest.raises(RuntimeError, match="already exists"):
        resolve["resolve_smoke_output_dir"](existing, True, ())


def test_an_existing_file_is_also_rejected(resolve, tmp_path):
    existing = tmp_path / "a_file"
    existing.write_text("x")
    with pytest.raises(RuntimeError, match="already exists"):
        resolve["resolve_smoke_output_dir"](existing, True, ())


def test_a_fresh_directory_is_accepted_and_resolved(resolve, tmp_path):
    fresh = tmp_path / "fresh_candidate"
    got = resolve["resolve_smoke_output_dir"](fresh, True, ())
    assert got == fresh.resolve()
    assert not got.exists(), "the guard must not create it"


@pytest.mark.parametrize("which", [0, 1])
def test_the_historical_output_directories_are_refused(resolve, tmp_path, which):
    protected = (tmp_path / "routing_extraction_v1", tmp_path / "routing_extraction_v1_smoke")
    with pytest.raises(RuntimeError, match="historical output"):
        resolve["resolve_smoke_output_dir"](protected[which], True, protected)


@pytest.mark.parametrize("which", [0, 1])
def test_a_child_of_a_protected_directory_is_refused(resolve, tmp_path, which):
    """Audit finding: a fresh child inside a protected directory was accepted.

    The path does not exist and is not equal to the protected directory, so the
    two earlier tests both passed while the run would still have written inside
    the very directory the comparison treats as read-only.
    """
    protected = (tmp_path / "routing_extraction_v1", tmp_path / "routing_extraction_v1_smoke")
    for d in protected:
        d.mkdir()
    child = protected[which] / "fresh_inside"
    with pytest.raises(RuntimeError, match="may not be inside"):
        resolve["resolve_smoke_output_dir"](child, True, protected)


@pytest.mark.parametrize("which", [0, 1])
def test_a_deep_descendant_of_a_protected_directory_is_refused(resolve, tmp_path, which):
    protected = (tmp_path / "routing_extraction_v1", tmp_path / "routing_extraction_v1_smoke")
    for d in protected:
        d.mkdir()
    with pytest.raises(RuntimeError, match="may not be inside"):
        resolve["resolve_smoke_output_dir"](protected[which] / "a" / "b" / "c", True, protected)


def test_a_symlinked_parent_leading_into_a_protected_directory_is_refused(resolve, tmp_path):
    """The candidate itself is new; its resolved parent is protected."""
    protected = (tmp_path / "routing_extraction_v1_smoke",)
    protected[0].mkdir()
    link = tmp_path / "innocent_looking"
    link.symlink_to(protected[0], target_is_directory=True)
    with pytest.raises(RuntimeError, match="may not be inside"):
        resolve["resolve_smoke_output_dir"](link / "fresh", True, protected)


def test_a_broken_symlink_in_the_way_is_refused(resolve, tmp_path):
    link = tmp_path / "dangling"
    link.symlink_to(tmp_path / "does_not_exist", target_is_directory=True)
    with pytest.raises(RuntimeError, match="already exists"):
        resolve["resolve_smoke_output_dir"](link, True, ())


def test_a_relative_spelling_of_a_historical_directory_is_refused(resolve, tmp_path, monkeypatch):
    protected = (tmp_path / "routing_extraction_v1_smoke",)
    protected[0].mkdir()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="historical output"):
        resolve["resolve_smoke_output_dir"]("routing_extraction_v1_smoke", True, protected)


def test_a_symlink_to_a_historical_directory_is_refused(resolve, tmp_path):
    real = tmp_path / "routing_extraction_v1"
    real.mkdir()
    link = tmp_path / "looks_new"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(RuntimeError):
        resolve["resolve_smoke_output_dir"](link, True, (real,))


def test_no_overwrite_or_delete_option_exists():
    text = extraction_source()
    for forbidden in ("--overwrite", "--force", "shutil.rmtree", "--clean"):
        assert forbidden not in text, forbidden


# ============================================================== the comparator
def _write_smoke(directory: Path, *, logits=b"LOGITS-A", prompt_config=None,
                 completed_utc="2026-08-12T02:04:00Z", seq_len=41,
                 logits_sha="a" * 64, extra_config=None, failures=0, checks=20,
                 corrupt_hash_for=None, drop=(), packages=None,
                 source_manifest=None, fail_line=False, validation_text=None,
                 extra_files=(), extra_dirs=(), extra_symlinks=()):
    """A minimal but structurally faithful smoke directory."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "router_logits.npy": logits,
        "selected_experts.npy": b"EXPERTS",
        "selected_probs.npy": b"PROBS",
        "routing_entropy.npy": b"ENTROPY",
        "routing_margin.npy": b"MARGIN",
        "row_ids.json": b'["row_a","row_b"]',
        "router_metadata.json": b'{"num_moe_blocks": 48}',
    }
    for name, blob in payload.items():
        if name not in drop:
            (directory / name).write_bytes(blob)

    ledger = [{"row_id": f"row_{c}", "row_index": i, "seq_len": seq_len,
               "logits_sha256": logits_sha, "prompt_sha256": "b" * 64,
               "completed_utc": completed_utc}
              for i, c in enumerate(("a", "b"))]
    (directory / "completed_rows.jsonl").write_text(
        "\n".join(json.dumps(r) for r in ledger) + "\n", encoding="utf-8")

    config = {
        "stage": "routegate_routing_extraction_v1", "mode": "smoke", "rows": 2,
        "token_position": -1, "architecture_signature": "sig", "revision": "ad44e777",
        "frozen_revision": "ad44e777", "manifest_sha256": "c" * 64,
        "dataset_sha256": "d" * 64, "residual_npy_sha256": "e" * 64,
        "dtype": "bfloat16", "attn_implementation": "eager", "device_map": "auto",
        "generation": "none (model.generate is disarmed)", "steering": "none",
        "router_modification": "none", "array_shapes": {"router_logits.npy": [2, 48, 128]},
        "prompt_config": prompt_config or C.PROMPT_CONFIG_BEFORE,
        "packages": packages or {"python": "3.12.11", "torch": "2.7.1"},
        "source_manifest": source_manifest
            or "results/rq3_residual_extraction_20260805_001100/manifest.csv",
    }
    config.update(extra_config or {})
    (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
              for name in payload if (directory / name).is_file()}
    if corrupt_hash_for:
        hashes[corrupt_hash_for] = "f" * 64
    (directory / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    body = f"completed_utc: {completed_utc}\nelapsed_seconds: 12.3\n"
    if fail_line:
        body += "  [FAIL] something_broke                       detail\n"
    (directory / "extraction_validation.txt").write_text(
        validation_text if validation_text is not None
        else body + f"\nCHECKS: {checks}   FAILURES: {failures}\n", encoding="utf-8")
    for name in extra_files:
        (directory / name).write_bytes(b"SNEAKED IN")
    for name in extra_dirs:
        (directory / name).mkdir()
    for name, target in extra_symlinks:
        (directory / name).symlink_to(target)
    return directory


CUTOVER_CONFIG = C.PROMPT_CONFIG_AFTER


@pytest.fixture
def reference(tmp_path):
    return _write_smoke(tmp_path / "reference")


@pytest.fixture
def candidate(tmp_path):
    """A faithful post-cutover rerun: new timestamps, new prompt_config."""
    return _write_smoke(tmp_path / "candidate", prompt_config=CUTOVER_CONFIG,
                        completed_utc="2026-08-19T21:00:00Z")


def test_a_faithful_rerun_passes(reference, candidate):
    result = C.compare(reference, candidate)
    assert result.ok, result.failures


def test_only_the_documented_differences_are_tolerated(reference, candidate):
    """The three derived files differ; every stable artifact is identical."""
    for name in C.STABLE_ARTIFACTS:
        assert (reference / name).read_bytes() == (candidate / name).read_bytes()
    assert (reference / "config.json").read_bytes() != (candidate / "config.json").read_bytes()
    assert (reference / "completed_rows.jsonl").read_bytes() \
        != (candidate / "completed_rows.jsonl").read_bytes()
    assert C.compare(reference, candidate).ok


@pytest.mark.parametrize("artifact", list(C.STABLE_ARTIFACTS))
def test_any_changed_stable_artifact_fails(reference, tmp_path, artifact):
    cand = _write_smoke(tmp_path / "cand", prompt_config=CUTOVER_CONFIG)
    (cand / artifact).write_bytes((cand / artifact).read_bytes() + b"X")
    result = C.compare(reference, cand)
    assert not result.ok
    assert f"identical:{artifact}" in result.failures


def test_changed_logits_report_a_max_absolute_difference(reference, tmp_path):
    """Diagnosis is offered; the verdict is still FAIL."""
    np = pytest.importorskip("numpy")
    ref = _write_smoke(tmp_path / "r")
    cand = _write_smoke(tmp_path / "c", prompt_config=CUTOVER_CONFIG)
    np.save(ref / "router_logits.npy", np.zeros((2, 3), dtype="float32"))
    np.save(cand / "router_logits.npy", np.full((2, 3), 0.25, dtype="float32"))
    result = C.compare(ref, cand)
    assert not result.ok
    assert "identical:router_logits.npy" in result.failures
    assert any("max |difference| = 0.25" in d for d in result.diagnostics)


def test_no_tolerance_option_exists():
    text = COMPARATOR.read_text(encoding="utf-8")
    for forbidden in ("--tolerance", "--atol", "--rtol", "allclose", "--ignore"):
        assert forbidden not in text, forbidden


def test_a_missing_stable_artifact_fails(reference, tmp_path):
    cand = _write_smoke(tmp_path / "cand", prompt_config=CUTOVER_CONFIG,
                        drop=("routing_margin.npy",))
    result = C.compare(reference, cand)
    assert not result.ok
    assert "identical:routing_margin.npy" in result.failures


# ------------------------------------------------------------------- ledger
def test_a_changed_logits_sha256_in_the_ledger_fails(reference, tmp_path):
    cand = _write_smoke(tmp_path / "cand", prompt_config=CUTOVER_CONFIG,
                        logits_sha="9" * 64)
    result = C.compare(reference, cand)
    assert not result.ok
    assert "ledger_rows_match_excluding_only_completed_utc" in result.failures


def test_a_changed_seq_len_fails(reference, tmp_path):
    cand = _write_smoke(tmp_path / "cand", prompt_config=CUTOVER_CONFIG, seq_len=42)
    result = C.compare(reference, cand)
    assert not result.ok
    assert "ledger_rows_match_excluding_only_completed_utc" in result.failures


def test_only_completed_utc_is_excluded_from_the_ledger():
    assert C.LEDGER_VOLATILE_FIELDS == ("completed_utc",)
    row = {"row_id": "a", "seq_len": 41, "logits_sha256": "x", "completed_utc": "t"}
    assert C.strip_volatile(row) == {"row_id": "a", "seq_len": 41, "logits_sha256": "x"}


def test_a_differing_row_count_fails(reference, tmp_path):
    cand = _write_smoke(tmp_path / "cand", prompt_config=CUTOVER_CONFIG)
    (cand / "completed_rows.jsonl").write_text(
        json.dumps({"row_id": "row_a"}) + "\n", encoding="utf-8")
    result = C.compare(reference, cand)
    assert not result.ok and "ledger_row_count" in result.failures


# ------------------------------------------------------------------- config
@pytest.mark.parametrize("key,value", [
    ("token_position", 0), ("dtype", "float16"), ("attn_implementation", "sdpa"),
    ("steering", "applied"), ("router_modification", "yes"),
    ("architecture_signature", "different"), ("revision", "deadbeef"),
    ("rows", 3), ("mode", "full"), ("manifest_sha256", "0" * 64),
])
def test_any_other_config_value_change_fails(reference, tmp_path, key, value):
    cand = _write_smoke(tmp_path / "cand", prompt_config=CUTOVER_CONFIG,
                        extra_config={key: value})
    result = C.compare(reference, cand)
    assert not result.ok
    assert "every_other_config_value_is_identical" in result.failures


def test_an_undocumented_prompt_config_change_fails(reference, tmp_path):
    cand = _write_smoke(tmp_path / "cand",
                        prompt_config="something_else.build_chat: single user message")
    result = C.compare(reference, cand)
    assert not result.ok
    assert "candidate_prompt_config_is_exactly_the_post_cutover_string" in result.failures


def test_a_new_or_missing_config_key_fails(reference, tmp_path):
    cand = _write_smoke(tmp_path / "cand", prompt_config=CUTOVER_CONFIG,
                        extra_config={"sneaked_in": 1})
    result = C.compare(reference, cand)
    assert not result.ok and "config_key_set_identical" in result.failures


# ---- the audit's fail-open cases, each now a standing regression test ----
def test_a_changed_python_version_fails(reference, tmp_path):
    """Audit finding: PASS 36/36 with python 3.12.11 -> 9.9.9."""
    cand = _write_smoke(tmp_path / "cand", prompt_config=CUTOVER_CONFIG,
                        packages={"python": "9.9.9", "torch": "2.7.1"})
    result = C.compare(reference, cand)
    assert not result.ok
    assert "every_other_config_value_is_identical" in result.failures


def test_a_changed_torch_version_fails(reference, tmp_path):
    cand = _write_smoke(tmp_path / "cand", prompt_config=CUTOVER_CONFIG,
                        packages={"python": "3.12.11", "torch": "9.9.9"})
    assert not C.compare(reference, cand).ok


def test_a_changed_source_manifest_fails(reference, tmp_path):
    """Audit finding: an unexpected source_manifest was tolerated."""
    cand = _write_smoke(tmp_path / "cand", prompt_config=CUTOVER_CONFIG,
                        source_manifest="/somewhere/else/manifest.csv")
    result = C.compare(reference, cand)
    assert not result.ok
    assert "every_other_config_value_is_identical" in result.failures


def test_an_appended_prompt_config_suffix_fails(reference, tmp_path):
    """Audit finding: contradictory text appended to prompt_config passed.

    The suffix reverses the meaning of the string it is appended to, which is
    precisely why substring matching is unsafe here.
    """
    cand = _write_smoke(
        tmp_path / "cand",
        prompt_config=C.PROMPT_CONFIG_AFTER + "  AND ALSO enable_thinking=True")
    result = C.compare(reference, cand)
    assert not result.ok
    assert "candidate_prompt_config_is_exactly_the_post_cutover_string" in result.failures


def test_an_unchanged_prompt_config_in_the_candidate_fails(reference, tmp_path):
    """The cutover must be visible; an unchanged value means it did not happen."""
    cand = _write_smoke(tmp_path / "cand", prompt_config=C.PROMPT_CONFIG_BEFORE)
    result = C.compare(reference, cand)
    assert not result.ok
    assert "candidate_prompt_config_is_exactly_the_post_cutover_string" in result.failures


def test_a_reference_without_the_pre_cutover_string_fails(tmp_path):
    ref = _write_smoke(tmp_path / "r", prompt_config="unexpected value")
    cand = _write_smoke(tmp_path / "c", prompt_config=CUTOVER_CONFIG)
    result = C.compare(ref, cand)
    assert not result.ok
    assert "reference_prompt_config_is_exactly_the_pre_cutover_string" in result.failures


def test_no_environment_tolerance_list_exists():
    text = COMPARATOR.read_text(encoding="utf-8")
    assert "CONFIG_ENVIRONMENT_FIELDS" not in text
    assert "PROMPT_CONFIG_BEFORE" in text and "PROMPT_CONFIG_AFTER" in text


# ------------------------------------------------ validation summary rules
def test_zero_checks_zero_failures_is_rejected(reference, tmp_path):
    """Audit finding: an empty report trivially has no failures."""
    cand = _write_smoke(tmp_path / "cand", prompt_config=CUTOVER_CONFIG, checks=0)
    result = C.compare(reference, cand)
    assert not result.ok
    assert "candidate_validation_is_not_empty" in result.failures


def test_a_reduced_validation_check_count_is_rejected(reference, tmp_path):
    cand = _write_smoke(tmp_path / "cand", prompt_config=CUTOVER_CONFIG, checks=19)
    result = C.compare(reference, cand)
    assert not result.ok
    assert "candidate_validation_check_count_matches_reference" in result.failures


def test_an_explicit_fail_line_is_rejected_even_with_zero_in_the_summary(reference, tmp_path):
    cand = _write_smoke(tmp_path / "cand", prompt_config=CUTOVER_CONFIG, fail_line=True)
    result = C.compare(reference, cand)
    assert not result.ok
    assert "candidate_validation_has_no_fail_line" in result.failures


def test_a_reference_with_validation_failures_is_rejected(tmp_path):
    ref = _write_smoke(tmp_path / "r", failures=2)
    cand = _write_smoke(tmp_path / "c", prompt_config=CUTOVER_CONFIG)
    result = C.compare(ref, cand)
    assert not result.ok and "reference_validation_zero_failures" in result.failures


# --------------------------------------------------------- artifact hashes
def test_artifact_hashes_stable_entries_must_match(reference, tmp_path):
    cand = _write_smoke(tmp_path / "cand", prompt_config=CUTOVER_CONFIG,
                        corrupt_hash_for="router_logits.npy")
    result = C.compare(reference, cand)
    assert not result.ok
    assert "artifact_hashes_stable_entries_match" in result.failures
    assert "artifact_hashes_are_self_consistent" in result.failures


def test_the_whole_artifact_hashes_file_is_not_required_to_match(reference, candidate):
    """It hashes the ledger and config, so it cannot be byte-identical."""
    ref = json.loads((reference / "artifact_hashes.json").read_text())
    cand = json.loads((candidate / "artifact_hashes.json").read_text())
    assert set(ref) == set(cand)
    assert C.compare(reference, candidate).ok


def test_the_comparator_documents_why_three_files_differ():
    text = COMPARATOR.read_text(encoding="utf-8")
    for name in ("completed_rows.jsonl", "config.json", "artifact_hashes.json"):
        assert name in text
    assert "completed_utc" in text and "prompt_config" in text


# --------------------------------------------------------------- validation
def test_a_candidate_with_validation_failures_is_rejected(reference, tmp_path):
    cand = _write_smoke(tmp_path / "cand", prompt_config=CUTOVER_CONFIG, failures=1)
    result = C.compare(reference, cand)
    assert not result.ok and "candidate_validation_zero_failures" in result.failures


def test_comparing_a_directory_against_itself_is_refused(reference):
    result = C.compare(reference, reference)
    assert not result.ok and "reference_and_candidate_differ" in result.failures


def test_a_missing_directory_is_refused(reference, tmp_path):
    result = C.compare(reference, tmp_path / "nope")
    assert not result.ok and "candidate_directory_exists" in result.failures


def test_the_cli_requires_both_directories():
    with pytest.raises(SystemExit):
        C.parse_args([])
    args = C.parse_args(["--reference", "/a", "--candidate", "/b"])
    assert args.reference == Path("/a") and args.candidate == Path("/b")


def test_main_returns_nonzero_on_mismatch(reference, tmp_path, capsys):
    cand = _write_smoke(tmp_path / "cand", prompt_config=CUTOVER_CONFIG)
    (cand / "router_logits.npy").write_bytes(b"DIFFERENT")
    code = C.main(["--reference", str(reference), "--candidate", str(cand)])
    assert code == 1
    assert "FAIL" in capsys.readouterr().out


def test_main_returns_zero_on_a_faithful_rerun(reference, candidate, capsys):
    assert C.main(["--reference", str(reference), "--candidate", str(candidate)]) == 0
    assert "PASS" in capsys.readouterr().out


# ------------------------------------------------------------- the qsub job
def test_the_qsub_script_never_runs_the_full_extraction():
    text = QSUB.read_text(encoding="utf-8")
    assert "--full" not in text
    assert "--smoke --smoke-output-dir" in text


def test_the_qsub_script_uses_cwd_not_a_hard_coded_repository():
    text = QSUB.read_text(encoding="utf-8")
    assert "#$ -cwd" in text
    assert "-wd /home/ucabh21" not in text
    assert 'REPO="/home/ucabh21/routegate"' not in text


@pytest.mark.parametrize("resource", ["-l gpu=4", "-l mem=96G", "-l tmpfs=30G",
                                      "-ac allow=L", "-l h_rt="])
def test_the_qsub_script_requests_the_house_gpu_resources(resource):
    assert resource in QSUB.read_text(encoding="utf-8")


def test_the_qsub_script_is_offline_and_job_id_keyed():
    text = QSUB.read_text(encoding="utf-8")
    assert "HF_HUB_OFFLINE=1" in text and "TRANSFORMERS_OFFLINE=1" in text
    assert "JOB_ID" in text
    assert "compare_smoke_extractions.py" in text
    assert "set -euo pipefail" in text


# --------------------------------------------------- smoke and inspect modes
def test_smoke_requires_an_explicit_output_directory():
    """Audit finding: plain --smoke fell back to the historical directory."""
    text = extraction_source()
    assert "if a.smoke_output_dir is None:" in text
    assert "--smoke requires --smoke-output-dir" in text
    assert 'run("smoke", OUT_SMOKE' not in text


def test_no_branch_directs_a_smoke_or_inspect_run_at_the_historical_directory():
    """Structural over the whole call graph, not just the branch we edited."""
    import re as _re
    tree = ast.parse(extraction_source())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "run" and node.args:
            mode = node.args[0]
            target = ast.unparse(node.args[1]) if len(node.args) > 1 else ""
            if isinstance(mode, ast.Constant) and mode.value in ("smoke", "inspect"):
                assert "OUT_SMOKE" not in target, ast.unparse(node)


def test_inspect_only_writes_nothing_at_all():
    text = extraction_source()
    assert 'return run("inspect", None, rows, False)' in text
    assert "router_metadata_inspect_only.json" not in text
    assert 'if mode != "inspect":' in text          # no directory is created


def test_the_smoke_candidate_is_created_atomically_before_the_model_loads():
    """exist_ok=False, so two jobs cannot race into one directory."""
    text = extraction_source()
    assert 'out_dir.mkdir(parents=True, exist_ok=(mode != "smoke"))' in text
    assert text.index("out_dir.mkdir") < text.index("runtime.load_model_and_tokenizer()")


def test_full_mode_still_creates_its_directory_permissively():
    """--full is resumable and must keep exist_ok=True."""
    assert 'exist_ok=(mode != "smoke")' in extraction_source()


def test_the_flag_is_rejected_with_full_and_inspect_before_any_model_load():
    text = extraction_source()
    assert "if a.smoke_output_dir is not None and not a.smoke:" in text
    # the rejection precedes every run() dispatch
    assert text.index("if a.smoke_output_dir is not None and not a.smoke:") \
        < text.index('if a.inspect_only:')


def test_the_protected_comment_does_not_overclaim():
    """--full does write into OUT_FULL; the guard is about candidate selection."""
    text = extraction_source()
    assert "may not write into either of them under any flag" not in text
    assert "protects them from being chosen as a *candidate directory*" in text


# ----------------------------------------------------- the qsub boundary
def test_the_qsub_script_requires_an_expected_commit():
    text = QSUB.read_text(encoding="utf-8")
    assert 'EXPECTED_COMMIT:?' in text
    assert 'qsub -v EXPECTED_COMMIT=' in text


def test_the_qsub_script_requires_a_clean_tree_at_the_expected_commit():
    text = QSUB.read_text(encoding="utf-8")
    assert "git status --porcelain" in text
    assert 'ACTUAL_COMMIT="$(git rev-parse HEAD)"' in text
    assert '"$ACTUAL_COMMIT" != "$EXPECTED_COMMIT"' in text


def test_the_qsub_script_prints_the_three_component_hashes():
    text = QSUB.read_text(encoding="utf-8")
    for component in ("experiments/routegate_routing/extract_routing_v1.py",
                      "src/routegate/runtime.py",
                      "tools/migration/compare_smoke_extractions.py"):
        assert f"sha256sum {component}" in text, component


def test_the_qsub_script_hashes_the_reference_before_and_after():
    text = QSUB.read_text(encoding="utf-8")
    assert "REF_HASHES_BEFORE" in text and "REF_HASHES_AFTER" in text
    assert 'diff -q "$REF_HASHES_BEFORE" "$REF_HASHES_AFTER"' in text


def test_the_qsub_report_is_written_beside_the_candidate_not_inside_it():
    text = QSUB.read_text(encoding="utf-8")
    assert 'REPORT="$CANDIDATE_ROOT/equivalence_report' in text
    assert '$CANDIDATE/equivalence_report' not in text


# ================================================== the output inventory
def test_an_unexpected_candidate_file_fails(reference, tmp_path):
    """Audit finding: an extra array in the candidate passed unexamined.

    Nothing had ever asked what else was in the directory, so a file that no
    check named was a file no check could object to.
    """
    cand = _write_smoke(tmp_path / "cand", prompt_config=CUTOVER_CONFIG,
                        extra_files=("unexpected_scientific_output.npy",))
    result = C.compare(reference, cand)
    assert not result.ok
    assert any("candidate_inventory" in f for f in result.failures)


def test_an_unexpected_candidate_directory_fails(reference, tmp_path):
    cand = _write_smoke(tmp_path / "cand", prompt_config=CUTOVER_CONFIG,
                        extra_dirs=("sneaky_subdir",))
    result = C.compare(reference, cand)
    assert not result.ok
    assert any("candidate_inventory" in f for f in result.failures)


def test_a_symlink_in_the_candidate_fails(reference, tmp_path):
    """A symlink named like an artifact could point anywhere at all."""
    outside = tmp_path / "elsewhere.npy"
    outside.write_bytes(b"OUTSIDE")
    cand = _write_smoke(tmp_path / "cand", prompt_config=CUTOVER_CONFIG,
                        extra_symlinks=(("extra_link.npy", outside),))
    result = C.compare(reference, cand)
    assert not result.ok
    assert any("candidate_inventory" in f for f in result.failures)


def test_an_unexpected_reference_file_fails(tmp_path):
    """Recorded separately from the candidate: a reference that is not what this
    tool was written against is a different problem needing a different fix."""
    ref = _write_smoke(tmp_path / "r", extra_files=("stray.npy",))
    cand = _write_smoke(tmp_path / "c", prompt_config=CUTOVER_CONFIG)
    result = C.compare(ref, cand)
    assert not result.ok
    assert any("reference_inventory" in f for f in result.failures)


def test_a_missing_derived_file_fails(reference, tmp_path):
    cand = _write_smoke(tmp_path / "cand", prompt_config=CUTOVER_CONFIG)
    (cand / "config.json").unlink()
    result = C.compare(reference, cand)
    assert not result.ok
    assert any("candidate_inventory" in f for f in result.failures)


def test_the_expected_inventory_is_the_eleven_known_files():
    assert len(C.EXPECTED_FILES) == 11
    assert set(C.EXPECTED_FILES) == set(C.STABLE_ARTIFACTS) | set(C.DERIVED_FILES)


def test_a_faithful_candidate_has_a_clean_inventory(reference, candidate):
    assert C.inventory_problems(reference) == []
    assert C.inventory_problems(candidate) == []


# ============================================ strict validation parsing
DUP_WITH_FAILURES = ("completed_utc: T\nelapsed_seconds: 1.0\n\n"
                     "CHECKS: 21   FAILURES: 0\n"
                     "CHECKS: 21   FAILURES: 7\n")
DUP_BOTH_ZERO = ("completed_utc: T\nelapsed_seconds: 1.0\n\n"
                 "CHECKS: 21   FAILURES: 0\n"
                 "CHECKS: 21   FAILURES: 0\n")


def test_duplicate_candidate_summaries_fail_when_the_later_one_reports_failures(
        reference, tmp_path):
    """Audit finding: the parser stopped at the first summary."""
    cand = _write_smoke(tmp_path / "cand", prompt_config=CUTOVER_CONFIG,
                        validation_text=DUP_WITH_FAILURES)
    result = C.compare(reference, cand)
    assert not result.ok
    assert "candidate_validation_has_exactly_one_summary_line" in result.failures


def test_duplicate_candidate_summaries_fail_even_when_both_claim_zero(
        reference, tmp_path):
    """Which summary is the real one is unanswerable, so neither is used."""
    cand = _write_smoke(tmp_path / "cand", prompt_config=CUTOVER_CONFIG,
                        validation_text=DUP_BOTH_ZERO)
    result = C.compare(reference, cand)
    assert not result.ok
    assert "candidate_validation_has_exactly_one_summary_line" in result.failures


def test_a_contradictory_reference_validation_report_fails(tmp_path):
    ref = _write_smoke(tmp_path / "r", validation_text=DUP_WITH_FAILURES)
    cand = _write_smoke(tmp_path / "c", prompt_config=CUTOVER_CONFIG)
    result = C.compare(ref, cand)
    assert not result.ok
    assert "reference_validation_has_exactly_one_summary_line" in result.failures


def test_a_fail_line_in_the_reference_is_rejected(tmp_path):
    ref = _write_smoke(tmp_path / "r", fail_line=True)
    cand = _write_smoke(tmp_path / "c", prompt_config=CUTOVER_CONFIG)
    result = C.compare(ref, cand)
    assert not result.ok
    assert "reference_validation_has_no_fail_line" in result.failures


def test_a_report_with_no_summary_line_fails(reference, tmp_path):
    cand = _write_smoke(tmp_path / "cand", prompt_config=CUTOVER_CONFIG,
                        validation_text="completed_utc: T\nno summary here\n")
    result = C.compare(reference, cand)
    assert not result.ok
    assert "candidate_validation_has_exactly_one_summary_line" in result.failures


def test_a_summary_embedded_in_prose_is_not_accepted(reference, tmp_path):
    """Anchored: the whole line must be the summary."""
    cand = _write_smoke(
        tmp_path / "cand", prompt_config=CUTOVER_CONFIG,
        validation_text="note: we expect CHECKS: 21   FAILURES: 0 eventually\n")
    result = C.compare(reference, cand)
    assert not result.ok


def test_the_parser_returns_every_summary_not_just_the_first(tmp_path):
    d = _write_smoke(tmp_path / "d", validation_text=DUP_WITH_FAILURES)
    assert C.validation_summaries(d) == [(21, 0), (21, 7)]


def test_a_single_clean_summary_parses_to_one_entry(reference):
    assert C.validation_summaries(reference) == [(20, 0)]


# ------------------------------------------------------------- the control
def test_the_faithful_post_cutover_control_still_passes(reference, candidate):
    """Everything above tightens the harness; this proves it did not seize up."""
    result = C.compare(reference, candidate)
    assert result.ok, result.failures
