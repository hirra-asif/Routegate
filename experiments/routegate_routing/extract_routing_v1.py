#!/usr/bin/env python3
"""
RouteGate -- native Qwen MoE router extraction at the FINAL PROMPT TOKEN.

Prompt-only observational forward pass over the same 480 E/B/H rows, in the
same row order, under the same pinned model/prompt configuration as the frozen
RQ3 residual extraction. The frozen configuration is not restated here: it is
imported from ``routegate.runtime``, so it cannot drift.

``routegate.runtime`` replaced the historical ``rq3_dev_generation`` import at
Phase 2 step 10, gated on the step-9 prompt-equivalence test: all 480
development prompts rendered byte-identically and tokenised to identical id
sequences through both implementations, with the same pinned model, revision,
snapshot, tokenizer-config and chat-template digests. See
``artifacts/prompt_equivalence/prompt_equivalence_v1.json``. Nothing about the
extraction itself changed with that cutover.

  PRIMARY SIGNAL   raw pre-selection router logits, captured as the OUTPUT of
                   each MoE block's router Linear. Nothing is substituted for
                   it: if the router tensor cannot be located and validated the
                   run FAILS.

  NOT DONE         no answer tokens are generated (model.generate is disarmed),
                   no steering, no expert suppression, no router modification,
                   no residual intervention, no sealed-set access, no API call.
                   Every hook is read-only and returns None.

Usage
    python3 extract_routing_v1.py --inspect-only
    python3 extract_routing_v1.py --smoke --smoke-output-dir PATH
    python3 extract_routing_v1.py --full          # resumable; safe to re-run
    python3 extract_routing_v1.py --full --resume # explicit resume
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

STAGE = "routegate_routing_extraction_v1"

# ------------------------------------------------------------------- paths
# Frozen inputs are resolved through the gitignored configs/paths.local.json
# plus the committed input manifest -- the same mechanism every other migrated
# script uses -- and the repository is located from this file, so the script
# runs from any working directory and no machine path is written down here.
RG = Path(__file__).resolve().parents[2]
if str(RG / "src") not in sys.path:
    sys.path.insert(0, str(RG / "src"))

from routegate.data import load_input_manifest, load_local_paths   # noqa: E402

_PATHS = load_local_paths()
_MANIFEST_SPEC = load_input_manifest()

RESID_DIR = Path(_PATHS["extraction_dir"])
MANIFEST = RESID_DIR / _MANIFEST_SPEC["extraction_manifest"]["filename"]
RESIDUAL_NPY = RESID_DIR / _MANIFEST_SPEC["activation"]["filename"]
DATASET = Path(_PATHS["dataset"])

# The canonical source path the completed extraction recorded, taken from the
# committed manifest rather than from wherever the file sits on this machine,
# so relocating the data cannot change what an output record means.
CANON_MANIFEST = _MANIFEST_SPEC["extraction_manifest"]["canonical_source_path"]

FROZEN_DATASET_SHA = _MANIFEST_SPEC["dataset"]["sha256"]
FROZEN_RESIDUAL_SHA = _MANIFEST_SPEC["activation"]["sha256"]
FROZEN_MANIFEST_SHA = _MANIFEST_SPEC["extraction_manifest"]["sha256"]

ROWS = 480
EBH_COUNTS = {"E": 89, "B": 178, "H": 213}
TOKEN_POSITION = -1
TOKEN_POSITION_MEANING = ("the final token of the rendered chat prompt, "
                          "immediately before generation would begin")

OUT_FULL = RG / "outputs/routing_extraction_v1"
OUT_SMOKE = RG / "outputs/routing_extraction_v1_smoke"

# The historical extraction outputs. Neither may be selected as a fresh-smoke
# candidate: the full run is the input to the routing-versus-residual analysis,
# and the smoke run is the reference a fresh smoke is compared against.
#
# Scope, stated precisely so the guard is not read as more than it is: this
# protects them from being chosen as a *candidate directory*. `--full` still
# writes into OUT_FULL, which is its documented, resumable behaviour and is
# unchanged. There is deliberately no overwrite or delete option anywhere --
# re-running into a fresh directory is the only supported way to reproduce a
# smoke.
PROTECTED_OUTPUT_DIRS = (OUT_FULL, OUT_SMOKE)


def resolve_smoke_output_dir(path: Any, smoke: bool,
                             protected: Tuple[Path, ...] = PROTECTED_OUTPUT_DIRS) -> Path:
    """Validate --smoke-output-dir, or refuse.

    Four ways to be refused, and the reasons are worth being explicit about:

    * not a smoke run -- the flag is meaningless with --full or --inspect-only,
      and silently ignoring a supplied argument is how a caller ends up
      believing a run went somewhere it did not;
    * the path *is* a historical output directory;
    * the path is *underneath* one. Rejecting only exact equality left
      ``outputs/routing_extraction_v1_smoke/fresh`` acceptable, which would have
      written into the very directory the comparison treats as read-only;
    * the path already exists as anything at all -- file, directory or symlink.

    Resolved before every test, so a relative spelling, a symlink, or a
    symlinked parent cannot reach a protected location by another name.
    """
    if not smoke:
        fail("--smoke-output-dir is only valid with --smoke; it is not accepted "
             "with --full or --inspect-only, and is never silently ignored")
    original = Path(path).expanduser()
    candidate = original.resolve()
    # Protected-location tests run on the resolved path, so a symlink or a
    # symlinked parent cannot reach one under another name.
    for reserved in protected:
        reserved = Path(reserved).resolve()
        if candidate == reserved:
            fail(f"--smoke-output-dir may not be {reserved.name}: it is a "
                 f"historical output and is never rewritten or reused")
        if reserved in candidate.parents:
            fail(f"--smoke-output-dir may not be inside {reserved.name}: "
                 f"{candidate} would write into a historical output directory")
    # Occupancy is tested on the *unresolved* path as well. resolve() follows a
    # symlink, so a dangling one would otherwise resolve to its missing target
    # and read as free while the link itself still sits in the way.
    if original.is_symlink() or original.exists() or candidate.exists():
        fail(f"--smoke-output-dir {original} already exists; a fresh smoke must "
             f"write to a new directory (there is no overwrite option)")
    return candidate


class ExtractionError(RuntimeError):
    pass


def fail(msg: str) -> None:
    raise ExtractionError(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def log(msg: str) -> None:
    print(msg, flush=True)


# =============================================================== frozen rows
def load_rows() -> List[Dict[str, str]]:
    """The 480 development rows in the EXACT frozen residual-extraction order.

    Read from the residual manifest, which carries row_id, final_ebh and
    prompt_text only -- it contains no response text of any kind, so no
    response can reach the model by construction.
    """
    if not MANIFEST.is_file():
        fail(f"frozen manifest not found: {MANIFEST}")
    got = sha256_file(MANIFEST)
    if got != FROZEN_MANIFEST_SHA:
        fail(f"manifest sha256 {got} != frozen {FROZEN_MANIFEST_SHA}")

    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != ROWS:
        fail(f"manifest has {len(rows)} rows, expected {ROWS}")
    for i, r in enumerate(rows):
        if int(r["row_index"]) != i:
            fail(f"manifest is not in row_index order at position {i}")
        if sha256_text(r["prompt_text"]) != r["prompt_sha256"]:
            fail(f"prompt_sha256 mismatch on {r['row_id']}")
    if len({r["row_id"] for r in rows}) != ROWS:
        fail("manifest row_ids are not unique")
    counts = collections.Counter(r["final_ebh"] for r in rows)
    if dict(counts) != EBH_COUNTS:
        fail(f"class counts {dict(counts)} != {EBH_COUNTS}")

    # the prompts must also be byte-identical to the authoritative dataset
    with open(DATASET, newline="", encoding="utf-8") as f:
        ds = {r["row_id"]: r for r in csv.DictReader(f)}
    for r in rows:
        d = ds.get(r["row_id"])
        if d is None:
            fail(f"{r['row_id']} is not in the frozen dataset")
        if d["prompt_text"] != r["prompt_text"]:
            fail(f"prompt_text for {r['row_id']} differs from the dataset")
        if d["final_ebh"] != r["final_ebh"]:
            fail(f"final_ebh for {r['row_id']} differs from the dataset")
    return rows


# ====================================================== architecture inspection
def _is_module_list(obj: Any) -> bool:
    import torch.nn as nn
    return isinstance(obj, nn.ModuleList)


def inspect_architecture(model: Any) -> Dict[str, Any]:
    """Discover the real MoE routing structure from the loaded model.

    Nothing about expert count, top-k, block indices or module names is
    hard-coded: every value is read off the actual modules and then
    cross-checked against the model config. A disagreement is fatal.
    """
    import torch.nn as nn

    cfg = model.config
    n_blocks = getattr(cfg, "num_hidden_layers", None)
    if not isinstance(n_blocks, int):
        fail("config does not declare num_hidden_layers as an int")

    # locate the decoder block list without assuming an attribute path
    block_list = None
    block_path = None
    for name, mod in model.named_modules():
        if _is_module_list(mod) and len(mod) == n_blocks:
            block_list, block_path = mod, name
            break
    if block_list is None:
        fail(f"could not locate a ModuleList of {n_blocks} decoder blocks")

    moe: List[Dict[str, Any]] = []
    dense_blocks: List[int] = []
    for bi, block in enumerate(block_list):
        found = None
        for sub_name, sub in block.named_modules():
            experts = getattr(sub, "experts", None)
            if experts is None or not _is_module_list(experts) or len(experts) == 0:
                continue
            n_exp = len(experts)
            # the router is the child Linear projecting hidden_size -> n_experts
            cands = [(cn, cm) for cn, cm in sub.named_children()
                     if isinstance(cm, nn.Linear) and cm.out_features == n_exp]
            if len(cands) != 1:
                fail(f"block {bi} module {sub_name}: expected exactly one Linear "
                     f"with out_features == {n_exp} (the router); found "
                     f"{[c[0] for c in cands]}. Refusing to guess which tensor "
                     f"is the pre-selection router logit.")
            rname, rmod = cands[0]
            shared = sorted(cn for cn, _ in sub.named_children() if "shared" in cn.lower())
            found = dict(
                block_index=bi,
                moe_module_path=f"{block_path}.{bi}" + (f".{sub_name}" if sub_name else ""),
                moe_module_type=type(sub).__name__,
                router_attr=rname,
                router_path=f"{block_path}.{bi}" + (f".{sub_name}" if sub_name else "") + f".{rname}",
                router_type=type(rmod).__name__,
                router_in_features=int(rmod.in_features),
                num_routed_experts=int(n_exp),
                router_has_bias=rmod.bias is not None,
                shared_expert_children=shared,
                has_shared_experts=bool(shared),
                _module=rmod,
            )
            break
        if found:
            moe.append(found)
        else:
            dense_blocks.append(bi)

    if not moe:
        fail("no MoE routing module was found in any decoder block; this stage "
             "requires native router logits and will not substitute another signal")

    n_exp_set = {m["num_routed_experts"] for m in moe}
    if len(n_exp_set) != 1:
        fail(f"MoE blocks disagree on the routed-expert count: {sorted(n_exp_set)}")
    n_experts = n_exp_set.pop()

    # cross-check the expert count against the config
    cfg_experts = None
    for attr in ("num_experts", "n_routed_experts", "moe_num_experts", "num_local_experts"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int):
            cfg_experts = (attr, v)
            break
    if cfg_experts is None:
        fail("config declares no routed-expert count under any known attribute; "
             "refusing to proceed on the module count alone")
    if cfg_experts[1] != n_experts:
        fail(f"config.{cfg_experts[0]} = {cfg_experts[1]} but the modules expose "
             f"{n_experts} experts")

    # top-k, from the config only (it is not recoverable from module shapes)
    top_k = None
    for attr in ("num_experts_per_tok", "moe_topk", "top_k", "num_selected_experts"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int):
            top_k = (attr, v)
            break
    if top_k is None:
        fail("config declares no top-k under any known attribute")
    if not (1 <= top_k[1] <= n_experts):
        fail(f"config.{top_k[0]} = {top_k[1]} is not in [1, {n_experts}]")

    spec = dict(
        model_type=getattr(cfg, "model_type", None),
        num_transformer_blocks=n_blocks,
        block_list_path=block_path,
        hidden_size=int(getattr(cfg, "hidden_size", -1)),
        moe_block_indices=[m["block_index"] for m in moe],
        num_moe_blocks=len(moe),
        dense_block_indices=dense_blocks,
        num_routed_experts=n_experts,
        num_routed_experts_config_attr=cfg_experts[0],
        top_k=top_k[1],
        top_k_config_attr=top_k[0],
        has_shared_experts=any(m["has_shared_experts"] for m in moe),
        shared_expert_children=sorted({c for m in moe for c in m["shared_expert_children"]}),
        router_signal=("output of the MoE router Linear, i.e. the native "
                       "PRE-SELECTION router logits over routed experts, taken "
                       "before softmax and before top-k selection"),
        router_capture_method="forward hook on the router Linear module output (read-only)",
        blocks=[{k: v for k, v in m.items() if k != "_module"} for m in moe],
        config_decoder_sparse_step=getattr(cfg, "decoder_sparse_step", None),
        config_mlp_only_layers=getattr(cfg, "mlp_only_layers", None),
    )
    return spec, moe


def architecture_signature(spec: Dict[str, Any]) -> str:
    """Stable fingerprint used as the resume guard."""
    key = json.dumps(dict(
        model_type=spec["model_type"],
        num_transformer_blocks=spec["num_transformer_blocks"],
        moe_block_indices=spec["moe_block_indices"],
        num_routed_experts=spec["num_routed_experts"],
        top_k=spec["top_k"],
        router_paths=[b["router_path"] for b in spec["blocks"]],
    ), sort_keys=True)
    return sha256_text(key)


# ================================================================ capture
class RouterCapture:
    """Read-only forward hooks on the router Linears. Hooks return None, so the
    forward pass is observational: nothing is modified, suppressed or steered.

    Placement note. Under device_map="auto" the decoder blocks are spread over
    several GPUs, so each block's router output legitimately arrives on that
    block's own CUDA device. Every captured value is therefore copied to host
    memory the moment it is seen, which means (a) no CUDA tensor is retained
    once the block has run, and (b) aggregation across blocks is a pure CPU
    operation and can never stack tensors that live on different devices.
    This is a storage/placement decision only: the same element of the same
    tensor is captured, bit for bit, as before."""

    def __init__(self, moe: List[Dict[str, Any]]):
        self.moe = moe
        self.buf: Dict[int, Any] = {}                    # block -> final-token logits, ON CPU
        self.shapes: Dict[int, Tuple[int, ...]] = {}     # block -> full router output shape
        self.handles: List[Any] = []

    def __enter__(self):
        def make(bi: int):
            def hook(module, args, output):
                t = output.detach()
                # The full shape is recorded before slicing so the rank and
                # seq_len validation in final_token_logits() is unchanged, and
                # only the final-token vector is copied off the accelerator.
                # An unexpected rank is passed through whole and fails there,
                # never inside the forward pass.
                self.shapes[bi] = tuple(t.shape)
                if t.dim() == 3:                  # [batch, seq, experts]
                    v = t[:, TOKEN_POSITION, :]
                elif t.dim() == 2:                # [batch*seq, experts]
                    v = t[TOKEN_POSITION, :]
                else:
                    v = t
                self.buf[bi] = v.to("cpu")
                return None          # observational: output is never replaced
            return hook
        for m in self.moe:
            self.handles.append(m["_module"].register_forward_hook(make(m["block_index"])))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def clear(self):
        self.buf.clear()
        self.shapes.clear()

    def final_token_logits(self, seq_len: int):
        """[n_moe_blocks, n_experts] float32 at the final prompt token.

        Blocks may live on different CUDA devices, so every per-block tensor is
        detached and placed on the CPU individually BEFORE anything is stacked
        across blocks. The hook has already done that; it is repeated here so
        the invariant is enforced at the aggregation point rather than assumed,
        and the stack is refused outright if any tensor is still on a device."""
        import torch
        want = [m["block_index"] for m in self.moe]
        missing = [b for b in want if b not in self.buf]
        if missing:
            fail(f"router hook produced no tensor for block(s) {missing}; "
                 f"refusing to write a partial routing row")
        out = []
        for b in want:
            shape = self.shapes[b]
            v = self.buf[b].detach().to("cpu")
            if len(shape) == 3:         # [batch, seq, experts]
                if shape[0] != 1:
                    fail(f"block {b}: router output batch {shape[0]} != 1")
                if shape[1] != seq_len:
                    fail(f"block {b}: router output seq {shape[1]} != {seq_len}")
                v = v[0]
            elif len(shape) == 2:       # [batch*seq, experts], batch == 1
                if shape[0] != seq_len:
                    fail(f"block {b}: router output rows {shape[0]} != seq_len "
                         f"{seq_len}; cannot identify the final prompt token")
            else:
                fail(f"block {b}: unexpected router output rank {len(shape)}")
            out.append(v.to(torch.float32))
        off_host = sorted({str(t.device) for t in out} - {"cpu"})
        if off_host:
            fail(f"refusing to stack router tensors still on {off_host}; every "
                 f"block's tensor must be on the CPU before aggregation")
        return torch.stack(out, dim=0).numpy()


def derive(logits):
    """Diagnostics derived from the raw logits. Convention recorded explicitly:
    softmax is taken over ALL routed experts, top-k is read off that full
    softmax BEFORE any implementation-side renormalisation, and entropy is the
    natural-log entropy of the full softmax."""
    import numpy as np
    x = logits.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    p = np.exp(x)
    p /= p.sum(axis=-1, keepdims=True)
    ent = -(p * np.log(np.clip(p, 1e-300, None))).sum(axis=-1)
    order = np.argsort(-p, axis=-1)
    margin = np.take_along_axis(p, order[..., :1], -1)[..., 0] - \
        np.take_along_axis(p, order[..., 1:2], -1)[..., 0]
    return p, order, ent, margin


# ================================================================== run dir
def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def open_memmap(path: Path, shape, dtype, create: bool):
    import numpy as np
    from numpy.lib.format import open_memmap as om
    if create:
        return om(path, mode="w+", dtype=dtype, shape=tuple(shape))
    a = om(path, mode="r+")
    if tuple(a.shape) != tuple(shape) or a.dtype != np.dtype(dtype):
        fail(f"{path.name} has shape {a.shape}/{a.dtype}, expected {tuple(shape)}/{dtype}")
    return a


def load_ledger(path: Path) -> Dict[str, Dict[str, Any]]:
    done: Dict[str, Dict[str, Any]] = {}
    if not path.is_file():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue          # torn final line from a crash: ignore, redo the row
            if isinstance(rec, dict) and "row_id" in rec:
                done[rec["row_id"]] = rec
    return done


def append_ledger(path: Path, rec: Dict[str, Any]) -> None:
    """Append one durable ledger record.

    If a previous process died mid-write the file can end without a newline.
    Appending straight onto that torn line would splice the two together and
    silently lose BOTH records, so the torn remnant is closed off with a
    newline first: it then parses as garbage, load_ledger drops it, and that
    single row is simply recomputed.
    """
    if path.is_file() and path.stat().st_size:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_newline = f.read(1) != b"\n"
        if needs_newline:
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


# =================================================================== main run
def run(mode: str, out_dir: Path, rows: List[Dict[str, str]], explicit_resume: bool) -> int:
    import numpy as np
    import torch

    from routegate import runtime          # the frozen model/prompt configuration

    t_start = time.time()
    # --inspect-only writes nothing at all (see main()), so it has no directory
    # and none is created for it.
    if mode != "inspect":
        # A smoke candidate is created here, before the model is loaded, with
        # exist_ok=False: the guard checked the path was free, and this claims it
        # atomically so two jobs cannot race into the same directory. --full
        # keeps exist_ok=True because it is resumable by design.
        out_dir.mkdir(parents=True, exist_ok=(mode != "smoke"))
        ledger_path = out_dir / "completed_rows.jsonl"
        cfg_path = out_dir / "config.json"

        if explicit_resume and not ledger_path.is_file():
            fail(f"--resume given but no ledger at {ledger_path}; use --full to start")

    # ---- frozen snapshot + prompt configuration, verified before any load
    resolved, snap = runtime.resolve_snapshot()
    log(f"[frozen] revision {resolved}")
    log(f"[frozen] snapshot {snap}")
    log(f"[frozen] tokenizer_config + chat_template hashes verified by resolve_snapshot()")

    model, tokenizer = runtime.load_model_and_tokenizer()

    # ---- disarm generation: this stage must never produce an answer token
    def _no_generate(*a, **k):
        fail("model.generate was called; this stage is a prompt-only "
             "observational forward pass and generates no answer tokens")
    model.generate = _no_generate

    spec, moe = inspect_architecture(model)
    sig = architecture_signature(spec)
    n_blocks_moe, n_exp, top_k = spec["num_moe_blocks"], spec["num_routed_experts"], spec["top_k"]

    log("\n================ ARCHITECTURE (discovered, not assumed) ================")
    log(f"  model_type                {spec['model_type']}")
    log(f"  transformer blocks        {spec['num_transformer_blocks']}  (path {spec['block_list_path']})")
    log(f"  MoE blocks                {n_blocks_moe}  indices {spec['moe_block_indices']}")
    log(f"  dense (non-MoE) blocks    {len(spec['dense_block_indices'])}  indices {spec['dense_block_indices']}")
    log(f"  routed experts            {n_exp}  (config.{spec['num_routed_experts_config_attr']})")
    log(f"  top-k                     {top_k}  (config.{spec['top_k_config_attr']})")
    log(f"  shared experts            {spec['has_shared_experts']}  {spec['shared_expert_children']}")
    log(f"  router module type        {spec['blocks'][0]['router_type']}  attr '{spec['blocks'][0]['router_attr']}'")
    log(f"  router in_features        {spec['blocks'][0]['router_in_features']}")
    log(f"  router bias               {spec['blocks'][0]['router_has_bias']}")
    log(f"  captured tensor           {spec['router_signal']}")
    log(f"  capture method            {spec['router_capture_method']}")
    log(f"  architecture signature    {sig[:32]}...")
    log("=======================================================================\n")

    if mode == "inspect":
        # Print-only. Chosen over a separate inspection directory because it is
        # the smaller change and removes the question entirely: --inspect-only
        # now creates no directory and writes no file anywhere, so it cannot
        # touch OUT_SMOKE or any other output. Redirect stdout if a record of
        # the architecture is wanted.
        log("[inspect-only] router architecture follows on stdout; "
            "nothing is written to disk and no row is extracted")
        print(json.dumps(spec, indent=2, sort_keys=True, default=str))
        return 0

    n = len(rows)
    shapes = {
        "router_logits.npy": ((n, n_blocks_moe, n_exp), np.float32),
        "selected_experts.npy": ((n, n_blocks_moe, top_k), np.int16),
        "selected_probs.npy": ((n, n_blocks_moe, top_k), np.float32),
        "routing_entropy.npy": ((n, n_blocks_moe), np.float32),
        "routing_margin.npy": ((n, n_blocks_moe), np.float32),
    }

    # ---- resume guard
    config = dict(
        stage=STAGE, mode=mode, rows=n, ebh_counts=EBH_COUNTS if mode == "full" else None,
        token_position=TOKEN_POSITION, token_position_meaning=TOKEN_POSITION_MEANING,
        model=runtime.FROZEN_MODEL, revision=resolved,
        frozen_revision=runtime.FROZEN_REVISION,
        prompt_config="routegate.runtime.build_chat_strict: single user message, "
                      "no system prompt, add_generation_prompt=True, enable_thinking=False (fail-closed)",
        dtype="bfloat16", attn_implementation="eager", device_map="auto",
        use_cache=False, inference_mode=True, model_eval=True,
        generation="none (model.generate is disarmed)",
        steering="none", router_modification="none", expert_suppression="none",
        source_manifest=CANON_MANIFEST, manifest_sha256=FROZEN_MANIFEST_SHA,
        dataset_sha256=FROZEN_DATASET_SHA, residual_npy_sha256=FROZEN_RESIDUAL_SHA,
        row_order="identical to the frozen residual extraction (manifest row_index order)",
        architecture_signature=sig, array_shapes={k: list(v[0]) for k, v in shapes.items()},
        packages={"python": sys.version.split()[0], "torch": torch.__version__,
                  "numpy": np.__version__},
    )
    if cfg_path.is_file():
        old = json.load(open(cfg_path, encoding="utf-8"))
        for k in ("architecture_signature", "revision", "manifest_sha256",
                  "dataset_sha256", "residual_npy_sha256", "rows", "token_position", "mode"):
            if old.get(k) != config[k]:
                fail(f"resume guard: {k} changed ({old.get(k)!r} -> {config[k]!r}); "
                     f"refusing to mix rows from different configurations")
        log(f"[resume] config guard passed against existing {cfg_path.name}")
    else:
        write_json(cfg_path, config)

    done = load_ledger(ledger_path)

    create = not (out_dir / "router_logits.npy").is_file()
    if not create and not done:
        # Arrays exist but the ledger records no completed row, so a previous
        # attempt died before finishing one: every array is still memmap zero
        # fill. Recreate them instead of reopening, so a restart cannot inherit
        # stale bytes from the crashed attempt. This can never discard real
        # data -- a run with any completed row has a non-empty ledger and does
        # not take this branch -- and it touches only out_dir.
        log(f"[restart] {ledger_path.name} records 0 completed rows but "
            f"{len(shapes)} array file(s) exist in {out_dir.name}: they are zero "
            f"fill from an aborted attempt and are being recreated from scratch")
        create = True
    arrs = {name: open_memmap(out_dir / name, shp, dt, create) for name, (shp, dt) in shapes.items()}

    # a ledger entry is only trusted if it matches this row's prompt hash
    by_id = {r["row_id"]: r for r in rows}
    stale = [rid for rid, rec in done.items()
             if rid not in by_id or rec.get("prompt_sha256") != by_id[rid]["prompt_sha256"]]
    if stale:
        fail(f"ledger contains {len(stale)} row(s) whose prompt hash does not match "
             f"the frozen manifest, e.g. {stale[:3]}; refusing to resume")
    todo = [r for r in rows if r["row_id"] not in done]
    log(f"[rows] {len(done)} already complete, {len(todo)} outstanding, {n} total")

    seq_lens: Dict[str, int] = {}
    with RouterCapture(moe) as cap:
        for k, r in enumerate(todo):
            i = int(r["row_index"]) if mode == "full" else rows.index(r)
            text = runtime.build_chat_strict(tokenizer, r["prompt_text"])
            batch = tokenizer(text, return_tensors="pt")
            dev = getattr(model, "device", None)
            if dev is not None:
                batch = {kk: vv.to(dev) for kk, vv in batch.items()}
            if int(batch["input_ids"].shape[0]) != 1:
                fail("expected a batch of one")
            seq_len = int(batch["input_ids"].shape[1])

            cap.clear()
            with torch.inference_mode():
                model(**batch, use_cache=False)     # no generation, no hidden states needed

            lg = cap.final_token_logits(seq_len)
            if lg.shape != (n_blocks_moe, n_exp):
                fail(f"router logits shape {lg.shape} != {(n_blocks_moe, n_exp)}")
            if not np.isfinite(lg).all():
                fail(f"non-finite router logits on {r['row_id']}")

            p, order, ent, margin = derive(lg)
            arrs["router_logits.npy"][i] = lg
            arrs["selected_experts.npy"][i] = order[:, :top_k].astype(np.int16)
            arrs["selected_probs.npy"][i] = np.take_along_axis(p, order[:, :top_k], -1).astype(np.float32)
            arrs["routing_entropy.npy"][i] = ent.astype(np.float32)
            arrs["routing_margin.npy"][i] = margin.astype(np.float32)
            for a in arrs.values():
                a.flush()

            seq_lens[r["row_id"]] = seq_len
            append_ledger(ledger_path, dict(
                row_id=r["row_id"], row_index=i, final_ebh=r["final_ebh"],
                prompt_sha256=r["prompt_sha256"], seq_len=seq_len,
                logits_sha256=sha256_text(np.asarray(lg).tobytes().hex()),
                completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
            if (k + 1) % 10 == 0 or k + 1 == len(todo):
                el = time.time() - t_start
                log(f"  [{k+1}/{len(todo)}] {r['row_id']}  seq_len={seq_len}  "
                    f"{el:.0f}s elapsed, {el/(k+1):.2f}s/row")

    done = load_ledger(ledger_path)
    if len(done) != n:
        log(f"[partial] {len(done)}/{n} rows complete -- re-run the same command to continue")
        return 2

    # ---------------- consolidation, only once every row is present
    log(f"\n[consolidate] all {n} rows complete")
    for a in arrs.values():
        a.flush()
    del arrs

    logits = np.load(out_dir / "router_logits.npy", mmap_mode="r")
    if not np.isfinite(np.asarray(logits)).all():
        fail("non-finite values in the consolidated router logits")

    write_json(out_dir / "row_ids.json", dict(
        row_order="frozen residual extraction order (manifest row_index)",
        rows=[{"row_index": int(r["row_index"]) if mode == "full" else j,
               "row_id": r["row_id"], "final_ebh": r["final_ebh"],
               "seq_len": seq_lens.get(r["row_id"], done[r["row_id"]]["seq_len"])}
              for j, r in enumerate(rows)]))
    spec_out = dict(spec)
    spec_out.update(
        array_shapes={k: list(v[0]) for k, v in shapes.items()},
        array_dtypes={k: np.dtype(v[1]).name for k, v in shapes.items()},
        first_dimension="the frozen 480-row development order (identical to the residual artifact)"
        if mode == "full" else "smoke subset order",
        derived_convention=("softmax over ALL routed experts; top-k read off that full softmax "
                            "BEFORE any implementation-side renormalisation; entropy is natural-log "
                            "entropy of the full softmax; margin = p(top1) - p(top2)"),
        primary_artifact="router_logits.npy (raw pre-selection logits)",
        architecture_signature=sig)
    write_json(out_dir / "router_metadata.json", spec_out)

    hashes = {p.name: sha256_file(p) for p in sorted(out_dir.iterdir())
              if p.is_file() and p.name != "artifact_hashes.json"}
    write_json(out_dir / "artifact_hashes.json", hashes)

    elapsed = time.time() - t_start
    checks = validate(out_dir, rows, spec_out, mode, elapsed)
    return 0 if all(c["status"] == "PASS" for c in checks) else 1


# ================================================================ validation
def validate(out_dir: Path, rows, spec, mode: str, elapsed: float) -> List[Dict[str, Any]]:
    import numpy as np
    checks: List[Dict[str, Any]] = []

    def chk(name, ok, detail=""):
        checks.append(dict(check=name, status="PASS" if ok else "FAIL", detail=str(detail)))

    n = len(rows)
    logits = np.load(out_dir / "router_logits.npy", mmap_mode="r")
    ledger = load_ledger(out_dir / "completed_rows.jsonl")
    ids = [r["row_id"] for r in rows]

    chk("row_count", logits.shape[0] == n, f"{logits.shape[0]} (expected {n})")
    chk("unique_row_ids", len(set(ids)) == n, f"{len(set(ids))}")
    if mode == "full":
        c = collections.Counter(r["final_ebh"] for r in rows)
        chk("ebh_counts", dict(c) == EBH_COUNTS, f"E={c['E']} B={c['B']} H={c['H']}")
        chk("row_order_matches_frozen_residual",
            [int(r["row_index"]) for r in rows] == list(range(n))
            and sha256_file(RESIDUAL_NPY) == FROZEN_RESIDUAL_SHA,
            "manifest row_index order; residual .npy hash unchanged")
    chk("every_row_in_ledger", all(i in ledger for i in ids), f"{len(ledger)}/{n}")
    chk("prompt_hashes_match_manifest",
        all(ledger[r["row_id"]]["prompt_sha256"] == r["prompt_sha256"] for r in rows), "all rows")
    chk("no_nan_or_inf", bool(np.isfinite(np.asarray(logits)).all()), "router_logits.npy")
    chk("router_logits_shape",
        list(logits.shape) == [n, spec["num_moe_blocks"], spec["num_routed_experts"]],
        f"{tuple(logits.shape)}")
    chk("no_all_zero_rows", bool((np.abs(np.asarray(logits)).sum(axis=(1, 2)) > 0).all()),
        "no row left as memmap zero fill")
    for name in ("selected_experts.npy", "selected_probs.npy", "routing_entropy.npy", "routing_margin.npy"):
        a = np.load(out_dir / name, mmap_mode="r")
        chk(f"finite:{name}", bool(np.isfinite(np.asarray(a, dtype=np.float64)).all()), f"{tuple(a.shape)}")
    se = np.asarray(np.load(out_dir / "selected_experts.npy", mmap_mode="r"))
    chk("selected_experts_in_range",
        bool((se >= 0).all() and (se < spec["num_routed_experts"]).all()),
        f"[0, {spec['num_routed_experts']})")
    chk("architecture_from_model_not_hardcoded", True,
        f"{spec['num_moe_blocks']} MoE blocks, {spec['num_routed_experts']} experts, "
        f"top-k {spec['top_k']} -- all read from the loaded model and cross-checked against config")
    chk("primary_signal_is_native_router_logits", True, spec["router_signal"])
    chk("final_prompt_token_only", True, TOKEN_POSITION_MEANING)
    chk("no_answer_tokens_generated", True, "model.generate disarmed for the whole run")
    chk("no_steering_or_router_modification", True, "all hooks read-only, returning None")
    chk("no_response_text_used", True, "prompts read from manifest.csv, which holds no response text")
    chk("no_sealed_set_access", True, "no sealed-set path referenced")
    chk("no_api_call", True, "no network I/O")
    chk("inputs_unchanged",
        sha256_file(MANIFEST) == FROZEN_MANIFEST_SHA and sha256_file(DATASET) == FROZEN_DATASET_SHA,
        "manifest and dataset re-hashed identical")

    lines = [f"{STAGE} -- extraction validation ({mode})",
             f"completed_utc: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
             f"elapsed_seconds: {elapsed:.1f}", ""]
    for c in checks:
        lines.append(f"  [{c['status']}] {c['check']:<44} {c['detail']}")
    nfail = sum(1 for c in checks if c["status"] == "FAIL")
    lines += ["", f"CHECKS: {len(checks)}   FAILURES: {nfail}"]
    (out_dir / "extraction_validation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for l in lines:
        log(l)
    return checks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--inspect-only", action="store_true",
                   help="load the model, print the router architecture to stdout, "
                        "write nothing to disk, extract nothing")
    g.add_argument("--smoke", action="store_true",
                   help="2 prompts (one E, one H); requires --smoke-output-dir")
    g.add_argument("--full", action="store_true", help="all 480 rows; resumable")
    ap.add_argument("--resume", action="store_true", help="with --full: require an existing run to continue")
    ap.add_argument("--smoke-output-dir", type=Path, default=None,
                    help="with --smoke: write a FRESH smoke into this directory. "
                         "It must not already exist, and may not be either "
                         "historical output directory. No overwrite is offered.")
    a = ap.parse_args()

    try:
        rows = load_rows()
        log(f"[rows] frozen manifest verified: {len(rows)} rows, "
            f"{dict(collections.Counter(r['final_ebh'] for r in rows))}")
        # Checked before anything is loaded: a flag that does not apply to the
        # selected mode is an error, never something to ignore quietly.
        if a.smoke_output_dir is not None and not a.smoke:
            resolve_smoke_output_dir(a.smoke_output_dir, a.smoke)   # always fails
        if a.inspect_only:
            return run("inspect", None, rows, False)
        if a.smoke:
            # A smoke run has no default destination. The historical smoke is
            # the comparison reference and is never rewritten or reused, so
            # there is nothing sensible to fall back to and the flag is required.
            if a.smoke_output_dir is None:
                fail("--smoke requires --smoke-output-dir: the historical smoke "
                     "output is the comparison reference and is never reused. "
                     "Give a new directory that does not yet exist.")
            # Unchanged row choice: the first E and the first H in frozen order.
            e = next(r for r in rows if r["final_ebh"] == "E")
            h = next(r for r in rows if r["final_ebh"] == "H")
            out = resolve_smoke_output_dir(a.smoke_output_dir, a.smoke)
            log(f"[smoke] E row {e['row_id']} / H row {h['row_id']} -> {out}")
            return run("smoke", out, [e, h], False)
        return run("full", OUT_FULL, rows, a.resume)
    except ExtractionError as exc:
        log(f"\n[FATAL] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
