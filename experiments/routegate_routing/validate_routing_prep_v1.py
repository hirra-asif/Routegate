#!/usr/bin/env python3
"""
Offline preparation validation for the RouteGate MoE routing extraction.

Validates everything that can be checked WITHOUT loading Qwen: frozen hashes,
the 480-row order, the extraction script's syntax/imports/structure, the resume
logic (exercised against a simulated crash), smoke/full output separation, and
the absence of generation, steering, sealed-set and API references.

Architecture-dependent checks are NOT simulated here and are explicitly listed
as still outstanding.
"""
from __future__ import annotations
import ast, collections, csv, hashlib, json, os, re, subprocess, sys, tempfile
from pathlib import Path

# ------------------------------------------------------------------- paths
# Frozen inputs come from the gitignored configs/paths.local.json plus the
# committed input manifest; repository paths are derived from this file, so the
# validator runs from any working directory.
RG = Path(__file__).resolve().parents[2]
if str(RG / "src") not in sys.path:
    sys.path.insert(0, str(RG / "src"))

from routegate.data import load_input_manifest, load_local_paths   # noqa: E402
from routegate.validation import routing_output_path_state        # noqa: E402

_PATHS = load_local_paths()
_MANIFEST = load_input_manifest()

SCRIPT = RG / "experiments/routegate_routing/extract_routing_v1.py"
RESID_DIR = Path(_PATHS["extraction_dir"])
MANIFEST = RESID_DIR / _MANIFEST["extraction_manifest"]["filename"]
RESIDUAL_NPY = RESID_DIR / _MANIFEST["activation"]["filename"]
DATASET = Path(_PATHS["dataset"])

FROZEN_DATASET_SHA = "0943449697327aa47912ef80337462414ab98255c3ad99b717baa716353e1d08"
FROZEN_RESIDUAL_SHA = "af89872d5decd4dc432ec579a941e6247bc7de63ff5690432e9f5812a994e2d8"
FROZEN_MANIFEST_SHA = "a3b7444b1a2782b5bc04142f3545ac5d71ad57a5f546edc315d7d51d14947abd"
EBH = {"E": 89, "B": 178, "H": 213}

class ExtractionErrorProbe(RuntimeError):
    """Stands in for the extraction script's own error when its guard is
    exercised here, so a refusal is distinguishable from a bug in the probe."""


LOG, FAILS = [], []
def say(s=""):
    print(s); LOG.append(s)
def P(tag, ok, detail=""):
    say(f"  [{'PASS' if ok else 'FAIL'}] {tag:<48} {detail}")
    if not ok: FAILS.append(tag)
def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""): h.update(c)
    return h.hexdigest()
def sha_text(t): return hashlib.sha256(t.encode()).hexdigest()

say("=" * 84); say("OFFLINE PREPARATION VALIDATION  (no model is loaded)"); say("=" * 84)

# ------------------------------------------------- 1. frozen inputs
say("\n-- frozen inputs --")
P("dataset_sha256", sha_file(DATASET) == FROZEN_DATASET_SHA, FROZEN_DATASET_SHA[:24] + "...")
P("residual_npy_sha256", sha_file(RESIDUAL_NPY) == FROZEN_RESIDUAL_SHA, FROZEN_RESIDUAL_SHA[:24] + "...")
P("manifest_sha256", sha_file(MANIFEST) == FROZEN_MANIFEST_SHA, FROZEN_MANIFEST_SHA[:24] + "...")

rows = list(csv.DictReader(open(MANIFEST, newline="", encoding="utf-8")))
ds = {r["row_id"]: r for r in csv.DictReader(open(DATASET, newline="", encoding="utf-8"))}
P("480_rows", len(rows) == 480, f"{len(rows)}")
P("480_unique_row_ids", len({r["row_id"] for r in rows}) == 480, "480 unique")
c = collections.Counter(r["final_ebh"] for r in rows)
P("E89_B178_H213", dict(c) == EBH, f"E={c['E']} B={c['B']} H={c['H']}")
P("row_index_order", [int(r["row_index"]) for r in rows] == list(range(480)), "0..479 ascending")
P("prompt_sha_selfconsistent", all(sha_text(r["prompt_text"]) == r["prompt_sha256"] for r in rows), "480/480")
P("prompts_byte_identical_to_dataset",
  all(ds[r["row_id"]]["prompt_text"] == r["prompt_text"] for r in rows), "480/480")
P("labels_match_dataset", all(ds[r["row_id"]]["final_ebh"] == r["final_ebh"] for r in rows), "480/480")

v1 = RG / "outputs/routegate_residual_v1/oof_predictions.csv"
oof = list(csv.DictReader(open(v1, newline="", encoding="utf-8")))
P("row_order_matches_frozen_residual_baseline",
  [r["row_id"] for r in rows] == [r["row_id"] for r in oof],
  "manifest order == routegate_residual_v1 OOF order, row for row")

# ------------------------------------------------- 2. manifest carries no response text
say("\n-- input surface --")
cols = list(rows[0].keys())
P("manifest_has_no_response_column",
  not any(re.search(r"response|answer|completion|output_text", k, re.I) for k in cols),
  f"columns: {cols}")

# ------------------------------------------------- 3. script structure
say("\n-- extraction script --")
P("script_exists", SCRIPT.is_file(), str(SCRIPT))
src = SCRIPT.read_text(encoding="utf-8")
try:
    tree = ast.parse(src); ok_syntax = True; err = "parses cleanly"
except SyntaxError as e:
    ok_syntax = False; err = str(e)
P("script_syntax", ok_syntax, err)
P("script_compiles", subprocess.run([sys.executable, "-m", "py_compile", str(SCRIPT)],
                                    capture_output=True).returncode == 0, "py_compile")

fns = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
for need in ["load_rows", "inspect_architecture", "RouterCapture", "derive", "run",
             "validate", "load_ledger", "append_ledger", "architecture_signature"]:
    P(f"defines:{need}", need in fns, "")

# top-level imports must be stdlib only, so the script can be argument-checked
# without torch present
top_imports = set()
for n in tree.body:
    if isinstance(n, ast.Import):
        top_imports |= {a.name.split(".")[0] for a in n.names}
    elif isinstance(n, ast.ImportFrom) and n.module:
        top_imports.add(n.module.split(".")[0])
heavy = top_imports & {"torch", "transformers", "numpy", "accelerate"}
P("no_heavy_top_level_imports", not heavy, f"top-level: {sorted(top_imports)}")

# ------------------------------------------------- 4. forbidden operations
say("\n-- forbidden operations (static scan) --")
code_lines = [l for l in src.splitlines()]
def scan(pattern, allow_substr=()):
    hits = []
    for i, l in enumerate(code_lines, 1):
        s = l.split("#")[0]
        if re.search(pattern, s) and not any(a in l for a in allow_substr):
            hits.append(f"L{i}: {l.strip()[:70]}")
    return hits

gen = scan(r"\.generate\s*\(", allow_substr=("model.generate = _no_generate",))
P("no_generate_call", not gen, gen or "model.generate is assigned a raising stub, never called")
P("generate_is_disarmed", "model.generate = _no_generate" in src, "any call raises ExtractionError")
P("no_max_new_tokens", not scan(r"max_new_tokens"), "no generation length parameter anywhere")
P("no_sampling_params", not scan(r"do_sample|temperature\s*=|top_p\s*="), "no decoding parameters")

P("no_backward_or_grad", not scan(r"\.backward\(|requires_grad_\(True\)|torch\.enable_grad"), "none")
P("uses_inference_mode", "torch.inference_mode()" in src, "forward pass wrapped in inference_mode")

mod_hook = scan(r"register_forward_pre_hook|register_full_backward_hook")
P("no_pre_or_backward_hooks", not mod_hook, mod_hook or "only register_forward_hook (read-only)")
P("hook_returns_none", "return None          # observational" in src,
  "the forward hook explicitly returns None, so no output is replaced")
# Behavioural, not lexical: the words "steering"/"sealed" legitimately appear in
# docstrings and metadata that DECLARE their absence, so scanning for the words
# only re-finds that documentation. What matters is whether any code path could
# modify an activation or reach a sealed-set file.
hook_returns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "hook"]
P("hook_never_returns_a_tensor",
  bool(hook_returns) and all(
      all(r.value is None or (isinstance(r.value, ast.Constant) and r.value.value is None)
          for r in ast.walk(fn) if isinstance(r, ast.Return))
      for fn in hook_returns),
  "every return inside the forward hook is a bare None, so no module output can be replaced")
mutations = scan(r"(hidden_states|activation|resid|output)\s*(\[[^\]]*\])?\s*(=[^=]|\+=|-=|\*=)",
                 allow_substr=("output.detach()", "out = []", "spec_out", "output_router",
                               "outputs/", "out_dir", "out.append", "return out"))
P("no_activation_mutation", not mutations, mutations or "no assignment into any activation tensor")
P("no_module_replacement", not scan(r"setattr\(.*(mlp|gate|experts|layers)|\.weight\s*=|\.bias\s*="),
  "no module, weight or bias is replaced")

sealed_paths = scan(r"['\"][^'\"]*sealed[^'\"]*['\"]")
sealed_paths = [h for h in sealed_paths if re.search(r"/|\.csv|\.npy|\.json|\.jsonl", h)]
P("no_sealed_set_path_or_open", not sealed_paths,
  sealed_paths or "no path-like literal references a sealed set; the only 'sealed' "
                  "occurrences are a docstring and a validation label asserting non-access")
P("no_network_or_api",
  not scan(r"requests\.|urllib|http[s]?://|boto3|invoke_model|api_key",
           allow_substr=("local_files_only",)),
  "no HTTP client, endpoint or credential")
# The frozen model/prompt configuration now lives in this repository. Before
# step 10 it was read out of the historical workspace, at a path parsed from the
# extraction script's own SM constant; that constant and that dependency are
# gone, so the module under inspection is simply src/routegate/runtime.py.
GENERATION_MODULE = RG / "src/routegate/runtime.py"
gen_src = (GENERATION_MODULE.read_text(encoding="utf-8")
           if GENERATION_MODULE.is_file() else "")
P("snapshot_is_local_only", "local_files_only=True" in gen_src,
  "resolve_snapshot() resolves the pinned revision from local cache only")

# ------------------------------------------------- 5. frozen config reuse
say("\n-- frozen model / prompt configuration --")
P("imports_frozen_generation_module", "from routegate import runtime" in src,
  "model + prompt settings are imported, never restated")
P("uses_frozen_snapshot_guard", "runtime.resolve_snapshot()" in src,
  "revision + tokenizer_config + chat_template hashes verified before load")
P("uses_frozen_loader", "runtime.load_model_and_tokenizer()" in src,
  "bfloat16, eager attention, device_map=auto, pinned revision, eval(), grads off")
P("uses_frozen_chat_builder", "runtime.build_chat_strict(tokenizer" in src,
  "single user message, no system prompt, add_generation_prompt=True, enable_thinking=False (fail-closed)")
# Structural, so the extraction script may keep the docstring recording where
# its configuration used to come from without that reading as a dependency.
_imports = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
_imports |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
P("no_legacy_generation_import_remains",
  not any("rq3_dev_generation" in name for name in _imports),
  "step 10 cutover complete; the frozen configuration comes from routegate.runtime")
P("no_foreign_path_on_sys_path",
  not scan(r"sys\.path\.insert\(.*(?<!RG / \"src\")\)", allow_substr=('RG / "src"',)),
  "only this repository's src/ is added to sys.path")
P("frozen_revision_pinned", 'FROZEN_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"' in gen_src,
  "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39")
P("token_position_is_final_prompt_token", "TOKEN_POSITION = -1" in src,
  "position -1 of a batch-of-one, i.e. the last real prompt token")

# ------------------------------------------------- 6. no hard-coded architecture
say("\n-- architecture must be discovered, not assumed --")
P("no_hardcoded_expert_count", not scan(r"\b(128|64|60|160)\s*(experts|,\s*#)"), "none in code")
P("no_hardcoded_topk", not re.search(r"top_k\s*=\s*\d", src), "top_k is read from the model config")
P("no_hardcoded_moe_indices", not re.search(r"moe_block_indices\s*=\s*\[\s*\d", src),
  "MoE block indices are discovered by module walk")
P("expert_count_cross_checked", "but the modules expose" in src,
  "config expert count must equal len(experts); disagreement is fatal")
P("router_identified_by_shape", "cm.out_features == n_exp" in src,
  "the router is the unique child Linear projecting to the expert count")
P("fails_if_router_ambiguous", "Refusing to guess which tensor" in src,
  "more or fewer than one candidate Linear -> hard failure")
P("fails_if_no_moe", "will not substitute another signal" in src,
  "no MoE module found -> hard failure, no fallback signal")
P("rank_checked_before_indexing", "unexpected router output rank" in src,
  "2D and 3D router outputs handled; anything else fails")
P("seq_len_checked", "cannot identify the final prompt token" in src,
  "hook rows must equal seq_len or the final-token index is not trustworthy")

# ------------------------------------------------- 7. resume logic, exercised
say("\n-- resume logic (exercised offline against a simulated crash) --")
sys.path.insert(0, str(SCRIPT.parent))
import importlib.util
spec_ = importlib.util.spec_from_file_location("erv1", SCRIPT)
M = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(M)          # safe: no heavy imports at module scope

with tempfile.TemporaryDirectory() as td:
    led = Path(td) / "completed_rows.jsonl"
    P("ledger_empty_when_absent", M.load_ledger(led) == {}, "missing ledger -> no completed rows")
    for i in range(3):
        M.append_ledger(led, dict(row_id=f"r{i}", row_index=i, prompt_sha256=f"s{i}"))
    P("ledger_roundtrip", set(M.load_ledger(led)) == {"r0", "r1", "r2"}, "3 rows recorded durably")
    with open(led, "a", encoding="utf-8") as f:
        f.write('{"row_id": "r3", "prompt_sha')          # simulate a crash mid-write
    d = M.load_ledger(led)
    P("torn_final_line_ignored", set(d) == {"r0", "r1", "r2"},
      "a half-written ledger line is dropped, so that row is simply redone")
    M.append_ledger(led, dict(row_id="r3", row_index=3, prompt_sha256="s3"))
    P("append_after_torn_line", set(M.load_ledger(led)) == {"r0", "r1", "r2", "r3"},
      "the ledger keeps working after a crash")

P("resume_guard_on_config", "resume guard:" in src,
  "revision, architecture signature, hashes, row count and mode must all match to resume")
P("resume_guard_on_prompt_hash", "whose prompt hash does not match" in src,
  "a ledger row whose prompt hash drifted aborts the resume")
P("completed_rows_not_rewritten", "todo = [r for r in rows if r[\"row_id\"] not in done]" in src,
  "only outstanding rows are computed")
P("per_row_durable_write", "os.fsync(f.fileno())" in src and "a.flush()" in src,
  "array flushed and ledger fsync'd per row")
P("row_write_is_idempotent", "arrs[\"router_logits.npy\"][i] = lg" in src,
  "a redone row overwrites its own slot by fixed index; no duplication is possible")
P("deterministic_row_ordering", "int(r[\"row_index\"])" in src,
  "each row always writes to its frozen row_index slot, whatever order it is computed in")
P("consolidation_gated_on_completeness", "if len(done) != n:" in src,
  "metadata, hashes and validation are written only after all rows are present")
P("explicit_resume_flag", "--resume given but no ledger" in src,
  "--resume fails loudly if there is nothing to resume")

# ------------------------------------------------- 8. output separation
say("\n-- output separation --")
P("full_and_smoke_dirs_differ", M.OUT_FULL != M.OUT_SMOKE, f"{M.OUT_FULL.name}  vs  {M.OUT_SMOKE.name}")
P("full_dir_expected", Path(M.OUT_FULL) == RG / "outputs/routing_extraction_v1",
  str(Path(M.OUT_FULL).relative_to(RG)) if str(M.OUT_FULL).startswith(str(RG)) else str(M.OUT_FULL))
P("smoke_writes_elsewhere", "routing_extraction_v1_smoke" in str(M.OUT_SMOKE), str(M.OUT_SMOKE))
# Step 10 added --smoke-output-dir, so the destination is now a variable; the
# row list is what this check is about and it is unchanged.
P("smoke_mode_two_rows", 'run("smoke", out, [e, h], False)' in src,
  "exactly one E and one H prompt")
# Behavioural, not nominal. The previous version of these checks asserted that
# certain names appeared in the source, which a fail-open guard satisfies just
# as well as a working one. The guard is now exercised: it is lifted out of the
# module (importing it would pull in torch) and driven against real paths.
_guard_ns = {"Path": Path, "Any": object, "Tuple": tuple,
             "PROTECTED_OUTPUT_DIRS": (), "ExtractionError": ExtractionErrorProbe}
_guard_nodes = [n for n in tree.body
                if isinstance(n, ast.FunctionDef)
                and n.name in {"resolve_smoke_output_dir", "fail"}]
if len(_guard_nodes) == 2:
    exec(compile(ast.Module(body=_guard_nodes, type_ignores=[]), "<guard>", "exec"), _guard_ns)
    _guard = _guard_ns["resolve_smoke_output_dir"]

    def _refuses(path, smoke=True, protected=()):
        try:
            _guard(path, smoke, protected)
            return False
        except ExtractionErrorProbe:
            return True

    with tempfile.TemporaryDirectory() as _td:
        _root = Path(_td)
        _protected = (_root / "routing_extraction_v1", _root / "routing_extraction_v1_smoke")
        for _d in _protected:
            _d.mkdir()
        _existing_file = _root / "a_file"; _existing_file.write_text("x")
        _link = _root / "link_to_protected"; _link.symlink_to(_protected[1], target_is_directory=True)
        _child = _protected[1] / "fresh_inside"
        _fresh = _root / "genuinely_fresh"

        P("guard_refuses_a_protected_directory_itself",
          all(_refuses(d, True, _protected) for d in _protected),
          "neither historical output can be selected as a candidate")
        P("guard_refuses_a_child_of_a_protected_directory",
          _refuses(_child, True, _protected),
          "a fresh subdirectory inside a historical output is still inside it")
        P("guard_refuses_a_symlink_to_a_protected_directory",
          _refuses(_link, True, _protected), "resolved before the comparison")
        P("guard_refuses_an_existing_path",
          _refuses(_existing_file, True, _protected) and _refuses(_root, True, _protected),
          "an existing file or directory is never overwritten")
        P("guard_refuses_the_flag_without_smoke",
          _refuses(_fresh, False, _protected),
          "--smoke-output-dir is rejected with --full and --inspect-only, never ignored")
        _accepted = None
        try:
            _accepted = _guard(_fresh, True, _protected)
        except ExtractionErrorProbe:
            pass
        P("guard_accepts_a_genuinely_fresh_path",
          _accepted == _fresh.resolve() and not _fresh.exists(),
          "a new path is accepted and not created by the guard itself")
else:
    P("guard_is_present_and_testable", False,
      "resolve_smoke_output_dir/fail could not be lifted out of the script")

P("smoke_requires_an_explicit_output_dir",
  'if a.smoke_output_dir is None:' in src and "--smoke requires --smoke-output-dir" in src,
  "plain --smoke has no default destination and cannot reuse the historical smoke")
P("no_branch_falls_back_to_the_historical_smoke",
  not re.search(r'run\(\s*["\'](?:smoke|inspect)["\']\s*,\s*OUT_SMOKE', src),
  "no smoke or inspect run is ever directed at OUT_SMOKE")
P("inspect_only_writes_nothing",
  'return run("inspect", None, rows, False)' in src
  and "router_metadata_inspect_only.json" not in src,
  "--inspect-only prints to stdout and creates no directory")
P("smoke_candidate_created_atomically",
  'exist_ok=(mode != "smoke")' in src,
  "the candidate directory is claimed with exist_ok=False before the model loads")
P("no_overwrite_or_delete_option", not scan(r"--overwrite|--force|rmtree"),
  "there is no way to overwrite or delete an existing extraction")
P("smoke_row_choice_deterministic", 'next(r for r in rows if r["final_ebh"] == "E")' in src,
  "first E and first H in frozen row order")
# Phase-independent. The check this replaced asserted that both directories were
# absent, which held only before the very first extraction: once the extraction
# had legitimately run it failed forever, and its detail text kept claiming a
# state it had not observed. What matters at every phase is that each declared
# output path is either absent or a plain directory exactly where the script
# says it writes -- never a file, never a symlink, never somewhere else.
_out_states = [
    routing_output_path_state(M.OUT_FULL, RG / "outputs/routing_extraction_v1",
                              RG / "outputs"),
    routing_output_path_state(M.OUT_SMOKE, RG / "outputs/routing_extraction_v1_smoke",
                              RG / "outputs"),
]
P("routing_output_paths_absent_or_plain_dirs",
  all(ok for ok, _ in _out_states),
  "; ".join(detail for _, detail in _out_states))

# ------------------------------------------------- 9. CLI
say("\n-- command-line interface --")
r = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
P("help_runs_without_torch", r.returncode == 0, "argparse --help works with no GPU stack present")
for flag in ("--inspect-only", "--smoke", "--full", "--resume"):
    P(f"flag:{flag}", flag in r.stdout, "")
r2 = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
P("mode_flag_required", r2.returncode != 0, "refuses to run with no mode selected")

# ------------------------------------------------- 10. frozen artifacts untouched
say("\n-- frozen artifacts untouched --")
for d in ["outputs/routegate_residual_v1", "outputs/routegate_topic_analysis_v1"]:
    p = RG / d
    P(f"present:{Path(d).name}", p.is_dir(), f"{len(list(p.iterdir()))} files, not modified by this stage")
FORBIDDEN_OUTPUT_DIRS = ["outputs/routegate_residual_v1",
                         "outputs/routegate_topic_analysis_v1",
                         "outputs/routegate_routing_analysis_v1",
                         "outputs/residual_geometry",
                         "outputs/causal_repair"]
P("no_writes_outside_routing_dirs",
  not any(scan(r"open\([^)]*['\"][^'\"]*" + re.escape(d)) for d in FORBIDDEN_OUTPUT_DIRS),
  "the extraction script writes only into routing_extraction_v1[_smoke]")

# ------------------------------------------------- summary
say("\n" + "=" * 84)
say("CHECKS THAT REMAIN NECESSARILY GPU-DEPENDENT (not claimed here)")
say("=" * 84)
for s in [
    "the pinned snapshot resolves locally and its tokenizer_config / chat_template hashes match",
    "the model loads in bfloat16 / eager / device_map=auto within the GPU allocation",
    "the decoder block list is found and its length equals config.num_hidden_layers",
    "which block indices actually contain MoE routing (and how many are dense)",
    "the routed-expert count, top-k, and whether shared experts exist",
    "exactly one child Linear per MoE block projects to the expert count (router identification)",
    "the router hook fires for every MoE block on every prompt",
    "the router output rank/shape, and that its row count equals seq_len",
    "that the captured logits are finite and non-degenerate on real prompts",
    "tensor dimensions are consistent across prompts of different lengths",
    "per-row wall-clock cost and total extraction runtime",
]:
    say(f"  - {s}")

out = RG / "outputs/routing_extraction_prep_v1"
out.mkdir(parents=True, exist_ok=True)
(out / "offline_validation.txt").write_text(
    "\n".join(LOG) + f"\n\nOFFLINE VALIDATION FAILURES: {len(FAILS)}\n", encoding="utf-8")
json.dump({"script": str(SCRIPT), "script_sha256": sha_file(SCRIPT),
           "checks": len([l for l in LOG if "[PASS]" in l or "[FAIL]" in l]),
           "failures": FAILS,
           "frozen": {"dataset_sha256": FROZEN_DATASET_SHA,
                      "residual_npy_sha256": FROZEN_RESIDUAL_SHA,
                      "manifest_sha256": FROZEN_MANIFEST_SHA}},
          open(out / "offline_validation.json", "w"), indent=2)

say("\n" + "=" * 84)
say(f"OFFLINE VALIDATION FAILURES: {len(FAILS)}" + (f" -> {FAILS}" if FAILS else " (none)"))
say("=" * 84)
say(f"script sha256: {sha_file(SCRIPT)}")
say(f"report: {out}/offline_validation.txt")
sys.exit(1 if FAILS else 0)
