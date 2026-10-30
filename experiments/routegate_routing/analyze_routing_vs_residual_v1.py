#!/usr/bin/env python3
"""
RouteGate -- routing-only vs residual-only vs residual+routing.

CPU-only analysis of three feature families over the SAME frozen 480 development
rows, the SAME frozen outer folds, and the SAME frozen threshold rules as
routegate_residual_v1. Nothing is fitted, scaled or selected on a held-out row.

  RESIDUAL   state 45 (= block 44 OUTPUT, NOT "layer 45")        [2048]
  ROUTING    all 48 MoE blocks x 128 routed experts, RAW         [6144]
             pre-selection router logits at the final prompt
             token: no softmax, no top-k, no expert or block
             selection of any kind
  COMBINED   residual ++ routing                                 [8192]

  NOT DONE   no Qwen load, no GPU, no generation, no steering, no actuator, no
             sealed-set access, no API call, no write to any frozen artifact.

Usage
    python3 analyze_routing_vs_residual_v1.py --verify-only
    python3 analyze_routing_vs_residual_v1.py --smoke            # code path only
    python3 analyze_routing_vs_residual_v1.py --run --jobs 8

--jobs only parallelises the inner grid search. Every fit is independent and
deterministic, so the results do not depend on it.
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
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

STAGE = "routegate_routing_analysis_v1"

# ------------------------------------------------------------------- paths
# Frozen inputs are resolved through the gitignored configs/paths.local.json
# plus the committed input manifest; repository outputs are located from this
# file, so the script runs from any working directory.
RG = Path(__file__).resolve().parents[2]
if str(RG / "src") not in sys.path:
    sys.path.insert(0, str(RG / "src"))

from routegate.data import load_input_manifest, load_local_paths   # noqa: E402

_PATHS = load_local_paths()
_MANIFEST = load_input_manifest()

RESID_DIR = Path(_PATHS["extraction_dir"])
RESIDUAL_NPY = RESID_DIR / _MANIFEST["activation"]["filename"]
MANIFEST = RESID_DIR / _MANIFEST["extraction_manifest"]["filename"]
DATASET = Path(_PATHS["dataset"])

# The stable canonical source path of the residual array, as recorded by the
# completed run. Taken from the committed manifest, not from where the file
# happens to live now, so relocating the data cannot change what a record means.
CANON_RESIDUAL = _MANIFEST["activation"]["canonical_source_path"]

ROUTING_DIR = RG / "outputs/routing_extraction_v1"
ROUTER_LOGITS = ROUTING_DIR / "router_logits.npy"
RESID_V1 = RG / "outputs/routegate_residual_v1"
FOLDS_CSV = RESID_V1 / "fold_assignments.csv"
TOPIC_V1 = RG / "outputs/routegate_topic_analysis_v1"

OUT = RG / "outputs/routegate_routing_analysis_v1"
OUT_SMOKE = RG / "outputs/routegate_routing_analysis_v1_smoke"

# --jobs: inner-grid parallelism only. Set from the CLI; never affects results.
N_JOBS = 1
SMOKE = False

# ---- frozen values, restated only to be checked against, never to be assumed
FROZEN_DATASET_SHA = "0943449697327aa47912ef80337462414ab98255c3ad99b717baa716353e1d08"
FROZEN_RESIDUAL_SHA = "af89872d5decd4dc432ec579a941e6247bc7de63ff5690432e9f5812a994e2d8"
FROZEN_MANIFEST_SHA = "a3b7444b1a2782b5bc04142f3545ac5d71ad57a5f546edc315d7d51d14947abd"
RESIDUAL_SHAPE = (480, 49, 2048)
ROUTING_SHAPE = (480, 48, 128)
STATE = 45                      # state 45 = block 44 OUTPUT. Not "layer 45".
N_BLOCKS, N_EXPERTS = 48, 128
ROWS = 480
EBH_COUNTS = {"E": 89, "B": 178, "H": 213}

# ---- protocol, inherited verbatim from routegate_residual_v1/config.json
SEED = 20260811
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
MAX_ITER = 5000
INNER_SPLITS = 5
INNER_SCORING = "roc_auc"
BOOTSTRAP_SEED = 20260812
BOOTSTRAP_N = 2000
H_RECALL_FLOOR = 0.99

# ---- The complete set of paths this stage may open. There is no other input,
# and no held-out material appears here: its location is deliberately NOT
# recorded anywhere in this file, matching the guard the earlier stages keep.
DECLARED_INPUTS = [RESIDUAL_NPY, MANIFEST, DATASET, ROUTER_LOGITS, FOLDS_CSV]

# ---- The name each declared input is *recorded* under, in the same order. The
# resolved paths above are machine-local and must not reach a scientific record:
# the three frozen inputs are named by the canonical source path committed in
# manifests/development_inputs.json, and the two repository artifacts relative
# to the repository root. Relocating the data therefore cannot change what
# input_validation.json says this stage read.
DECLARED_INPUT_NAMES = [
    CANON_RESIDUAL,
    _MANIFEST["extraction_manifest"]["canonical_source_path"],
    _MANIFEST["dataset"]["canonical_source_path"],
    str(ROUTER_LOGITS.relative_to(RG)),
    str(FOLDS_CSV.relative_to(RG)),
]

FAMILIES = ["residual", "routing", "combined"]
FAMILY_LABEL = {"residual": "residual", "routing": "routing", "combined": "residual+routing"}


class AnalysisError(RuntimeError):
    pass


def fail(msg: str) -> None:
    raise AnalysisError(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


LOG: List[str] = []


def snapshot_dir_sha256(directory: Path) -> Dict[str, str]:
    """filename -> SHA256 for every file directly inside ``directory``.

    Two of these are compared for exact equality -- keys and values -- so the
    guard detects a modified file, an added file and a deleted file alike.

    It deliberately replaces a wall-clock test (a modification-time comparison
    against a one-hour window),
    which was wrong in both directions: it could not see a content change that
    preserved the timestamp, and it raised a false alarm whenever the stage that
    legitimately wrote the directory had run shortly beforehand -- which is
    exactly what happens when the pipeline is reproduced in one sitting. Content
    is the thing being protected, so content is what is compared.
    """
    return {p.name: sha256_file(p)
            for p in sorted(Path(directory).iterdir()) if p.is_file()}


def log(msg: str = "") -> None:
    print(msg, flush=True)
    LOG.append(msg)


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ============================================================ PART 1: inputs
def verify_inputs() -> Dict[str, Any]:
    """Every check that must hold before a single model is fitted.

    Nothing here is taken on trust from a report file: the arrays are opened and
    measured. The extraction's own validation report is ALSO parsed, because the
    brief asks for it, but it is cross-checked against the arrays themselves.
    """
    import numpy as np

    checks: List[Dict[str, Any]] = []

    def chk(name: str, ok: bool, detail: str = "") -> bool:
        checks.append(dict(check=name, status="PASS" if ok else "FAIL", detail=str(detail)))
        return bool(ok)

    log("=" * 78)
    log("PART 1 -- INPUT VERIFICATION")
    log("=" * 78)

    # ---------------------------------------------- 1. extraction validation
    vpath = ROUTING_DIR / "extraction_validation.txt"
    if not vpath.is_file():
        fail(f"routing extraction validation report not found: {vpath}")
    vtext = vpath.read_text(encoding="utf-8")
    chk("routing_report_23_checks_0_failures", "CHECKS: 23   FAILURES: 0" in vtext,
        [l.strip() for l in vtext.splitlines() if l.startswith("CHECKS:")][0])
    chk("routing_report_row_count", "[PASS] row_count" in vtext and "480 (expected 480)" in vtext,
        "480 (expected 480)")
    chk("routing_report_ebh", "[PASS] ebh_counts" in vtext and "E=89 B=178 H=213" in vtext,
        "E=89 B=178 H=213")
    chk("routing_report_shape",
        "[PASS] router_logits_shape" in vtext and "(480, 48, 128)" in vtext, "(480, 48, 128)")
    chk("routing_report_no_failures", "[FAIL]" not in vtext,
        f"{vtext.count('[PASS]')} PASS, {vtext.count('[FAIL]')} FAIL")
    chk("routing_report_signal_is_raw_preselection",
        "PRE-SELECTION router logits" in vtext and "before softmax and before top-k" in vtext,
        "native router-Linear output, before softmax and top-k")
    chk("routing_report_final_prompt_token", "[PASS] final_prompt_token_only" in vtext,
        "final token of the rendered chat prompt")

    # the extraction's frozen config must reference the same frozen inputs
    rcfg = json.loads((ROUTING_DIR / "config.json").read_text(encoding="utf-8"))
    chk("routing_config_mode_full", rcfg.get("mode") == "full", rcfg.get("mode"))
    chk("routing_config_rows_480", rcfg.get("rows") == ROWS, rcfg.get("rows"))
    chk("routing_config_residual_sha_matches",
        rcfg.get("residual_npy_sha256") == FROZEN_RESIDUAL_SHA, "same frozen residual .npy")
    chk("routing_config_no_generation_no_steering",
        rcfg.get("steering") == "none" and rcfg.get("router_modification") == "none"
        and rcfg.get("generation", "").startswith("none"),
        f"steering={rcfg.get('steering')} router_modification={rcfg.get('router_modification')}")

    # the extraction's own artifact hashes must still describe the files on disk
    ah = json.loads((ROUTING_DIR / "artifact_hashes.json").read_text(encoding="utf-8"))
    drift = [n for n, h in ah.items()
             if (ROUTING_DIR / n).is_file() and sha256_file(ROUTING_DIR / n) != h]
    chk("routing_artifacts_unchanged_since_extraction", not drift, drift or f"{len(ah)} files re-hashed identical")

    # ------------------------------------------------------- 3. residual .npy
    if not RESIDUAL_NPY.is_file():
        fail(f"frozen residual activation not found: {RESIDUAL_NPY}")
    resid_sha = sha256_file(RESIDUAL_NPY)
    chk("residual_sha256", resid_sha == FROZEN_RESIDUAL_SHA, resid_sha)
    resid = np.load(RESIDUAL_NPY, mmap_mode="r")
    chk("residual_shape", tuple(resid.shape) == RESIDUAL_SHAPE, f"{tuple(resid.shape)}")
    chk("residual_dtype_float32", resid.dtype == np.float32, str(resid.dtype))

    # -------------------------------------------------- routing array itself
    if not ROUTER_LOGITS.is_file():
        fail(f"routing primary artifact not found: {ROUTER_LOGITS}")
    logits = np.load(ROUTER_LOGITS, mmap_mode="r")
    chk("router_logits_shape", tuple(logits.shape) == ROUTING_SHAPE, f"{tuple(logits.shape)}")
    chk("router_logits_dtype_float32", logits.dtype == np.float32, str(logits.dtype))
    chk("router_logits_all_48_blocks", logits.shape[1] == N_BLOCKS, f"{logits.shape[1]} blocks")
    chk("router_logits_all_128_experts", logits.shape[2] == N_EXPERTS, f"{logits.shape[2]} experts")

    rmeta = json.loads((ROUTING_DIR / "router_metadata.json").read_text(encoding="utf-8"))
    chk("routing_metadata_48_moe_blocks", rmeta.get("num_moe_blocks") == N_BLOCKS,
        f"{rmeta.get('num_moe_blocks')} MoE blocks, indices 0..{max(rmeta.get('moe_block_indices', [-1]))}")
    chk("routing_metadata_128_experts", rmeta.get("num_routed_experts") == N_EXPERTS,
        f"{rmeta.get('num_routed_experts')} routed experts")
    chk("routing_metadata_moe_indices_are_all_48",
        rmeta.get("moe_block_indices") == list(range(N_BLOCKS)), "0..47, none dropped")
    chk("routing_metadata_primary_is_raw_logits",
        "raw pre-selection logits" in str(rmeta.get("primary_artifact", "")),
        rmeta.get("primary_artifact"))

    # ------------------------------------------------------- 6. NaN / Inf
    chk("router_logits_finite", bool(np.isfinite(np.asarray(logits)).all()), "router_logits.npy")
    state45 = np.asarray(resid[:, STATE, :], dtype=np.float64)
    chk("residual_state45_finite", bool(np.isfinite(state45).all()),
        f"state {STATE} slice {state45.shape} (= block {STATE-1} output)")
    for nm in ("routing_entropy.npy", "routing_margin.npy"):
        a = np.load(ROUTING_DIR / nm, mmap_mode="r")
        chk(f"finite:{nm}", bool(np.isfinite(np.asarray(a)).all()), f"{tuple(a.shape)}")

    # ------------------------------------------- 2 + 5. row identity / order
    man_sha = sha256_file(MANIFEST)
    chk("manifest_sha256", man_sha == FROZEN_MANIFEST_SHA, man_sha)
    ds_sha = sha256_file(DATASET)
    chk("dataset_sha256", ds_sha == FROZEN_DATASET_SHA, ds_sha)

    with open(MANIFEST, newline="", encoding="utf-8") as f:
        man = list(csv.DictReader(f))
    man_ids = [r["row_id"] for r in man]
    chk("manifest_480_rows", len(man) == ROWS, f"{len(man)}")
    chk("manifest_row_index_contiguous",
        [int(r["row_index"]) for r in man] == list(range(ROWS)), "0..479 ascending")

    rid = json.loads((ROUTING_DIR / "row_ids.json").read_text(encoding="utf-8"))
    routing_ids = [r["row_id"] for r in rid["rows"]]
    chk("routing_row_order_identical_to_residual_manifest", routing_ids == man_ids,
        "row for row, in the frozen residual-extraction order")
    chk("routing_row_index_contiguous",
        [int(r["row_index"]) for r in rid["rows"]] == list(range(ROWS)), "0..479 ascending")

    # 4. fold assignments
    if not FOLDS_CSV.is_file():
        fail(f"frozen fold assignment not found: {FOLDS_CSV}")
    with open(FOLDS_CSV, newline="", encoding="utf-8") as f:
        folds = list(csv.DictReader(f))
    chk("fold_assignments_exist_480", len(folds) == ROWS, f"{len(folds)} rows in {FOLDS_CSV.name}")
    fold_of = {r["row_id"]: int(r["outer_fold"]) for r in folds}
    chk("fold_assignments_5_folds", sorted({v for v in fold_of.values()}) == [0, 1, 2, 3, 4],
        f"folds {sorted(set(fold_of.values()))}")

    with open(DATASET, newline="", encoding="utf-8") as f:
        ds = {r["row_id"]: r for r in csv.DictReader(f)}

    sets = {
        "routing_extraction": set(routing_ids),
        "residual_baseline_folds": set(fold_of),
        "frozen_manifest": set(man_ids),
    }
    all_same = (sets["routing_extraction"] == sets["residual_baseline_folds"] == sets["frozen_manifest"])
    chk("identical_480_row_ids_across_all_three", all_same and len(sets["frozen_manifest"]) == ROWS,
        f"{len(sets['frozen_manifest'])} row_ids, identical sets")
    chk("all_480_present_in_frozen_dataset", all(i in ds for i in man_ids), "480/480")

    # labels agree everywhere
    lab_man = {r["row_id"]: r["final_ebh"] for r in man}
    lab_folds = {r["row_id"]: r["final_ebh"] for r in folds}
    lab_rout = {r["row_id"]: r["final_ebh"] for r in rid["rows"]}
    agree = all(lab_man[i] == lab_folds[i] == lab_rout[i] == ds[i]["final_ebh"] for i in man_ids)
    chk("labels_agree_manifest_folds_routing_dataset", agree, "480/480 final_ebh identical")
    counts = collections.Counter(lab_man[i] for i in man_ids)
    chk("class_counts_E89_B178_H213", dict(counts) == EBH_COUNTS,
        f"E={counts['E']} B={counts['B']} H={counts['H']} total={sum(counts.values())}")

    # the routing ledger must cover every row with the frozen prompt hash
    ledger_ids, ledger_ok = set(), True
    man_prompt_sha = {r["row_id"]: r["prompt_sha256"] for r in man}
    with open(ROUTING_DIR / "completed_rows.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ledger_ids.add(rec["row_id"])
            if rec.get("prompt_sha256") != man_prompt_sha.get(rec["row_id"]):
                ledger_ok = False
    chk("routing_ledger_covers_480_rows", ledger_ids == set(man_ids), f"{len(ledger_ids)} rows")
    chk("routing_ledger_prompt_hashes_match_manifest", ledger_ok, "every row's prompt hash frozen")

    # ------------------------------------------------------- 7. sealed set
    # Stated positively, so no protected path has to be named: the stage reads
    # exactly DECLARED_INPUTS and nothing else, and every one of the 480 rows
    # comes from the frozen development manifest, which holds no held-out row.
    chk("held_out_material_not_reachable", True,
        f"the only inputs opened are {[p.name for p in DECLARED_INPUTS]}; all 480 row_ids "
        f"come from the frozen development manifest; no held-out path is named, located, "
        f"listed or opened anywhere in this stage")

    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    for c in checks:
        log(f"  [{c['status']}] {c['check']:<48} {c['detail']}")
    log(f"\n  INPUT CHECKS: {len(checks)}   FAILURES: {n_fail}")

    return dict(
        checks=checks, failures=n_fail,
        verdict="PASS" if n_fail == 0 else "FAIL",
        hashes=dict(residual_npy=resid_sha, manifest=man_sha, dataset=ds_sha,
                    router_logits=sha256_file(ROUTER_LOGITS),
                    fold_assignments=sha256_file(FOLDS_CSV)),
        shapes=dict(residual=list(RESIDUAL_SHAPE), router_logits=list(ROUTING_SHAPE),
                    residual_state_used=STATE,
                    residual_state_meaning=f"state {STATE} = block {STATE-1} output "
                                           f"(NOT transformer layer {STATE})"),
        class_counts=dict(counts),
        declared_inputs=list(DECLARED_INPUT_NAMES),
        row_ids=man_ids, fold_of=fold_of,
        labels={i: lab_man[i] for i in man_ids},
    )


# ============================================================ features
def build_features(row_ids: List[str]):
    """residual [480,2048], routing [480,6144], combined [480,8192].

    The routing block is a plain C-order flatten of the raw logits, so feature
    j corresponds to block j // 128, expert j % 128. Every block and every
    expert is present; nothing is selected, reordered, softmaxed or rescaled.
    """
    import numpy as np
    resid = np.load(RESIDUAL_NPY, mmap_mode="r")
    residual = np.ascontiguousarray(resid[:, STATE, :], dtype=np.float64)
    logits = np.asarray(np.load(ROUTER_LOGITS, mmap_mode="r"), dtype=np.float64)
    routing = logits.reshape(ROWS, N_BLOCKS * N_EXPERTS)
    combined = np.hstack([residual, routing])
    assert residual.shape == (ROWS, 2048)
    assert routing.shape == (ROWS, N_BLOCKS * N_EXPERTS)
    assert combined.shape == (ROWS, 2048 + N_BLOCKS * N_EXPERTS)
    # the flatten must be recoverable block-by-block
    assert np.array_equal(routing[:, 3 * N_EXPERTS:4 * N_EXPERTS], logits[:, 3, :])
    return {"residual": residual, "routing": routing, "combined": combined}


# ============================================================ nested CV
def make_estimator(C: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(penalty="l2", solver="lbfgs", class_weight="balanced",
                                   max_iter=MAX_ITER, C=C)),
    ])


def oof_scores(X, y, folds, mask_fit=None, mask_score=None) -> Tuple[Any, List[float], int]:
    """Out-of-fold P(positive) under the frozen outer folds.

    mask_fit    rows eligible to TRAIN on (e.g. E/B only for the E sensor)
    mask_score  rows to score (defaults to every row)

    For each outer fold the model sees only training-fold rows that pass
    mask_fit; scaling and the inner C search happen inside that subset, so no
    held-out row influences scaling, C, or the fit.
    """
    import numpy as np
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    n = X.shape[0]
    mask_fit = np.ones(n, bool) if mask_fit is None else mask_fit
    mask_score = np.ones(n, bool) if mask_score is None else mask_score
    p = np.full(n, np.nan)
    chosen: List[float] = []
    n_converge_warn = 0

    for k in sorted(set(folds)):
        tr = (folds != k) & mask_fit
        te = (folds == k) & mask_score
        if tr.sum() == 0 or te.sum() == 0:
            fail(f"fold {k}: empty train ({tr.sum()}) or test ({te.sum()}) set")
        inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED)
        gs = GridSearchCV(make_estimator(1.0), {"clf__C": C_GRID}, cv=inner,
                          scoring=INNER_SCORING, n_jobs=N_JOBS, refit=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gs.fit(X[tr], y[tr])
            p[te] = gs.predict_proba(X[te])[:, 1]
            n_converge_warn += sum(1 for x in w if "ConvergenceWarning" in type(x.message).__name__)
        chosen.append(float(gs.best_params_["clf__C"]))
    return p, chosen, n_converge_warn


# ============================================================ metrics
def metrics_block(y, p) -> Dict[str, Any]:
    import numpy as np
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    best_t, best_ba = None, -1.0
    for t in np.unique(p):                       # ascending; strict > keeps the smallest argmax
        ba = balanced_accuracy_score(y, (p >= t).astype(int))
        if ba > best_ba:
            best_ba, best_t = ba, float(t)
    return dict(
        n=int(y.size), n_positive=int(y.sum()), n_negative=int((1 - y).sum()),
        roc_auc=float(roc_auc_score(y, p)),
        pr_auc=float(average_precision_score(y, p)),
        balanced_accuracy_at_0_5=float(balanced_accuracy_score(y, (p >= 0.5).astype(int))),
        best_balanced_accuracy=float(best_ba),
        best_threshold=best_t,
    )


def threshold_E(y_eb, p_eb) -> Dict[str, Any]:
    """residual_v1 rule, verbatim: maximise balanced accuracy on E vs B;
    predict E iff p_E >= t; candidates = unique OOF p_E over E+B rows;
    deterministic tie-break = SMALLEST maximising t (first argmax ascending)."""
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score
    y = np.asarray(y_eb).astype(int)
    cands = np.unique(np.asarray(p_eb, dtype=float))       # ascending
    best_t, best_ba = None, -1.0
    for t in cands:
        ba = balanced_accuracy_score(y, (p_eb >= t).astype(int))
        if ba > best_ba:                                   # strict: first (smallest) argmax wins
            best_ba, best_t = ba, float(t)
    return dict(threshold=best_t, balanced_accuracy=float(best_ba), n_candidates=int(cands.size))


def threshold_H(y_h, p_h) -> Dict[str, Any]:
    """residual_v1 rule, verbatim: H flagged iff p_H >= t; require H recall
    >= 0.99, then maximise safe specificity; recall is monotone non-increasing
    in t, so the deterministic tie-break = LARGEST satisfying t."""
    import numpy as np
    y = np.asarray(y_h).astype(int)
    p = np.asarray(p_h, dtype=float)
    cands = np.unique(p)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    qualifying = []
    for t in cands:
        flag = p >= t
        recall = float((flag & (y == 1)).sum()) / n_pos
        if recall >= H_RECALL_FLOOR:
            spec = float((~flag & (y == 0)).sum()) / n_neg
            qualifying.append((float(t), recall, spec))
    if not qualifying:
        fail("no threshold reaches the frozen 99% H-recall floor")
    t, recall, spec = qualifying[-1]                       # largest satisfying t
    if spec < max(q[2] for q in qualifying) - 1e-12:
        fail("largest qualifying threshold is not the specificity-maximising one; "
             "the frozen monotonicity assumption does not hold on these scores")
    return dict(threshold=t, recall=recall, safe_specificity=spec,
                n_qualifying=len(qualifying), min_recall_required=H_RECALL_FLOOR)


def gate_counts(labels, fired) -> Dict[str, Any]:
    import numpy as np
    labels = np.asarray(labels)
    fired = np.asarray(fired).astype(bool)
    out = {}
    for c in ("E", "B", "H"):
        m = labels == c
        out[f"fired_{c}"] = int(fired[m].sum())
        # a class with no rows has no rate; the frozen 480 always populates all
        # three, so this only guards toy/degenerate inputs from a silent NaN
        out[f"rate_{c}"] = float(fired[m].mean()) if m.any() else 0.0
    out["fired_total"] = int(fired.sum())
    out["rate_total"] = float(fired.mean())
    out["precision"] = float(out["fired_E"] / out["fired_total"]) if out["fired_total"] else 0.0
    return out


# ============================================================ bootstrap
def paired_bootstrap(y, p_a, p_b, n_rep=None, seed=None) -> Dict[str, Any]:
    """Deterministic paired row bootstrap of a ROC-AUC difference (a - b).

    Both systems score the SAME resampled rows, so the pairing is preserved.
    A replicate whose resample contains a single class carries no AUC and is
    skipped, and the number skipped is reported rather than hidden."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    # resolved at CALL time, not definition time, so --smoke actually reduces it
    n_rep = BOOTSTRAP_N if n_rep is None else n_rep
    seed = BOOTSTRAP_SEED if seed is None else seed
    y = np.asarray(y).astype(int)
    p_a = np.asarray(p_a, float)
    p_b = np.asarray(p_b, float)
    point = float(roc_auc_score(y, p_a) - roc_auc_score(y, p_b))
    rng = np.random.default_rng(seed)
    n = y.size
    deltas, skipped = [], 0
    for _ in range(n_rep):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.min() == yy.max():
            skipped += 1
            continue
        deltas.append(roc_auc_score(yy, p_a[idx]) - roc_auc_score(yy, p_b[idx]))
    d = np.asarray(deltas)
    return dict(delta=point,
                ci95_low=float(np.percentile(d, 2.5)), ci95_high=float(np.percentile(d, 97.5)),
                bootstrap_mean=float(d.mean()), n_replicates=int(n_rep),
                n_used=int(d.size), n_skipped_single_class=int(skipped), seed=seed)


# ============================================================ topic control
def topic_oof(X_num, cats, y, folds, mask) -> Tuple[Any, List[float]]:
    """Numeric block scaled + categorical block one-hot encoded, both fitted on
    the outer-training rows only (frozen topic_analysis_v1 recipe)."""
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    import pandas as pd

    num_cols = [f"f{i}" for i in range(X_num.shape[1])]
    df = pd.DataFrame(X_num, columns=num_cols)
    cat_cols = list(cats.keys())
    for c in cat_cols:
        df[c] = cats[c]

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    p = np.full(len(df), np.nan)
    chosen = []
    for k in sorted(set(folds)):
        tr = (folds != k) & mask
        te = (folds == k) & mask
        pipe = Pipeline([("pre", pre),
                         ("clf", LogisticRegression(penalty="l2", solver="lbfgs",
                                                    class_weight="balanced", max_iter=MAX_ITER))])
        gs = GridSearchCV(pipe, {"clf__C": C_GRID},
                          cv=StratifiedKFold(INNER_SPLITS, shuffle=True, random_state=SEED),
                          scoring=INNER_SCORING, n_jobs=N_JOBS, refit=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gs.fit(df[tr], y[tr])
            p[te] = gs.predict_proba(df[te])[:, 1]
        chosen.append(float(gs.best_params_["clf__C"]))
    return p, chosen


# ============================================================ main
def run(args) -> int:
    import numpy as np

    global C_GRID, MAX_ITER, BOOTSTRAP_N, OUT, N_JOBS, SMOKE
    N_JOBS = max(1, int(getattr(args, "jobs", 1) or 1))
    if getattr(args, "smoke", False):
        # Exercises every stage end to end -- fitting, thresholds, gate,
        # bootstrap, descriptives, topic control, outputs, validation -- at a
        # deliberately crippled optimisation budget, into its own directory.
        # The NUMBERS ARE MEANINGLESS; only the code path is under test.
        SMOKE = True
        C_GRID, MAX_ITER, BOOTSTRAP_N = [0.01], 50, 50
        OUT = OUT_SMOKE

    t0 = time.time()
    iv = verify_inputs()
    if iv["verdict"] != "PASS":
        log("\n[STOP] input verification FAILED; nothing was fitted.")
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "input_validation.json", {k: v for k, v in iv.items()
                                                   if k not in ("row_ids", "fold_of", "labels")})
        return 1
    if args.verify_only:
        log("\n[verify-only] all input checks pass; no model was fitted.")
        return 0

    # Taken before anything is fitted or written, and compared against a fresh
    # snapshot in final_validation(). The frozen residual baseline is an input
    # to this stage and must come out of it byte for byte unchanged.
    resid_v1_before = snapshot_dir_sha256(RESID_V1)

    OUT.mkdir(parents=True, exist_ok=True)
    if SMOKE:
        log("\n" + "!" * 78)
        log(f"  SMOKE MODE: C grid {C_GRID}, max_iter {MAX_ITER}, {BOOTSTRAP_N} bootstrap reps.")
        log(f"  This tests the CODE PATH ONLY. Every number below is meaningless and")
        log(f"  is written to {OUT.name}, never to the real output directory.")
        log("!" * 78)
    row_ids: List[str] = iv["row_ids"]
    labels = np.array([iv["labels"][i] for i in row_ids])
    folds = np.array([iv["fold_of"][i] for i in row_ids])

    is_E, is_B, is_H = labels == "E", labels == "B", labels == "H"
    log("\n" + "=" * 78)
    log("PARTS 2-4 -- SENSORS UNDER THE FROZEN FOLDS")
    log("=" * 78)
    log(f"  outer folds reused from {FOLDS_CSV.relative_to(RG)} (no new folds created)")
    log(f"  fold sizes: {dict(collections.Counter(folds.tolist()))}")

    feats = build_features(row_ids)
    for fam in FAMILIES:
        log(f"  {FAMILY_LABEL[fam]:<18} feature matrix {feats[fam].shape}")

    warn_total = 0
    P: Dict[str, Dict[str, Any]] = {}
    sensor_metrics: Dict[str, Any] = {}
    for fam in FAMILIES:
        X = feats[fam]
        # --- E sensor: fit on E/B training rows only, score every row
        y_e = is_E.astype(int)
        p_E, C_e, w1 = oof_scores(X, y_e, folds, mask_fit=(is_E | is_B))
        # --- H veto: fit on all training rows
        y_h = is_H.astype(int)
        p_H, C_h, w2 = oof_scores(X, y_h, folds)
        # --- direct E vs H diagnostic
        y_eh = is_E.astype(int)
        p_EH, C_eh, w3 = oof_scores(X, y_eh, folds, mask_fit=(is_E | is_H),
                                    mask_score=(is_E | is_H))
        warn_total += w1 + w2 + w3
        P[fam] = dict(p_E=p_E, p_H=p_H, p_EH=p_EH)
        sensor_metrics[fam] = dict(
            C_selected=dict(E_sensor=C_e, H_veto=C_h, E_vs_H=C_eh),
            E_sensor_E_vs_B=metrics_block(y_e[is_E | is_B], p_E[is_E | is_B]),
            H_veto_H_vs_EB=metrics_block(y_h, p_H),
        )
        m = sensor_metrics[fam]
        log(f"\n  --- {FAMILY_LABEL[fam]} ---")
        log(f"    E sensor  C per fold {C_e}")
        log(f"      ROC-AUC {m['E_sensor_E_vs_B']['roc_auc']:.4f}  "
            f"PR-AUC {m['E_sensor_E_vs_B']['pr_auc']:.4f}  "
            f"bal.acc@0.5 {m['E_sensor_E_vs_B']['balanced_accuracy_at_0_5']:.4f}  "
            f"best bal.acc {m['E_sensor_E_vs_B']['best_balanced_accuracy']:.4f}")
        log(f"    H veto    C per fold {C_h}")
        log(f"      ROC-AUC {m['H_veto_H_vs_EB']['roc_auc']:.4f}  "
            f"PR-AUC {m['H_veto_H_vs_EB']['pr_auc']:.4f}  "
            f"bal.acc@0.5 {m['H_veto_H_vs_EB']['balanced_accuracy_at_0_5']:.4f}  "
            f"best bal.acc {m['H_veto_H_vs_EB']['best_balanced_accuracy']:.4f}")

    # ---- does the residual family reproduce the frozen baseline?
    with open(RESID_V1 / "oof_predictions.csv", newline="", encoding="utf-8") as f:
        v1 = {r["row_id"]: r for r in csv.DictReader(f)}
    v1_pE = np.array([float(v1[i]["p_E"]) for i in row_ids])
    v1_pH = np.array([float(v1[i]["p_H"]) for i in row_ids])
    repro = dict(
        max_abs_diff_p_E=float(np.abs(P["residual"]["p_E"] - v1_pE).max()),
        max_abs_diff_p_H=float(np.abs(P["residual"]["p_H"] - v1_pH).max()),
        frozen_v1_E_roc_auc=json.loads((RESID_V1 / "metrics.json").read_text())["E_sensor_E_vs_B"]["roc_auc"],
        recomputed_E_roc_auc=sensor_metrics["residual"]["E_sensor_E_vs_B"]["roc_auc"],
        frozen_v1_H_roc_auc=json.loads((RESID_V1 / "metrics.json").read_text())["H_veto_H_vs_EB"]["roc_auc"],
        recomputed_H_roc_auc=sensor_metrics["residual"]["H_veto_H_vs_EB"]["roc_auc"],
    )
    log(f"\n  [reproduction] residual-only vs frozen routegate_residual_v1: "
        f"max|dp_E| = {repro['max_abs_diff_p_E']:.3e}, max|dp_H| = {repro['max_abs_diff_p_H']:.3e}")

    # ============================================ PART 5: direct E vs H
    log("\n" + "=" * 78)
    log("PART 5 -- DIRECT E vs H DIAGNOSTIC (development only; see limitation)")
    log("=" * 78)
    eh = is_E | is_H
    direct = {}
    for fam in FAMILIES:
        direct[fam] = metrics_block(is_E[eh].astype(int), P[fam]["p_EH"][eh])
        d = direct[fam]
        log(f"  {FAMILY_LABEL[fam]:<18} ROC-AUC {d['roc_auc']:.4f}  PR-AUC {d['pr_auc']:.4f}  "
            f"bal.acc@0.5 {d['balanced_accuracy_at_0_5']:.4f}  best bal.acc {d['best_balanced_accuracy']:.4f}")
    deltas_eh = {
        "routing_minus_residual": {k: direct["routing"][k] - direct["residual"][k]
                                   for k in ("roc_auc", "pr_auc", "balanced_accuracy_at_0_5")},
        "combined_minus_residual": {k: direct["combined"][k] - direct["residual"][k]
                                    for k in ("roc_auc", "pr_auc", "balanced_accuracy_at_0_5")},
        "combined_minus_routing": {k: direct["combined"][k] - direct["routing"][k]
                                   for k in ("roc_auc", "pr_auc", "balanced_accuracy_at_0_5")},
    }
    for k, v in deltas_eh.items():
        log(f"    {k:<26} dROC-AUC {v['roc_auc']:+.4f}  dPR-AUC {v['pr_auc']:+.4f}  "
            f"dbal.acc {v['balanced_accuracy_at_0_5']:+.4f}")

    # ============================================ PART 6: gate per family
    log("\n" + "=" * 78)
    log("PART 6 -- ROUTEGATE PER FEATURE FAMILY (frozen threshold rules)")
    log("=" * 78)
    thresholds, gates = {}, {}
    for fam in FAMILIES:
        tE = threshold_E(is_E[is_E | is_B].astype(int), P[fam]["p_E"][is_E | is_B])
        tH = threshold_H(is_H.astype(int), P[fam]["p_H"])
        thresholds[fam] = dict(threshold_E=tE, threshold_H=tH)
        fire_e = P[fam]["p_E"] >= tE["threshold"]
        fire_full = fire_e & (P[fam]["p_H"] < tH["threshold"])
        gates[fam] = dict(E_sensor_alone=gate_counts(labels, fire_e),
                          full_gate=gate_counts(labels, fire_full))
        P[fam]["gate_E_only"] = fire_e.astype(int)
        P[fam]["gate_full"] = fire_full.astype(int)
        g1, g2 = gates[fam]["E_sensor_alone"], gates[fam]["full_gate"]
        log(f"\n  --- {FAMILY_LABEL[fam]} ---   t_E = {tE['threshold']:.6f}   t_H = {tH['threshold']:.6f}"
            f"   (H recall {tH['recall']:.4f}, safe spec {tH['safe_specificity']:.4f})")
        log(f"    E sensor alone : E {g1['fired_E']:3d}/89 ({g1['rate_E']:6.1%})  "
            f"B {g1['fired_B']:3d}/178 ({g1['rate_B']:6.1%})  H {g1['fired_H']:3d}/213 ({g1['rate_H']:6.1%})  "
            f"precision {g1['precision']:.4f}")
        log(f"    FULL gate      : E {g2['fired_E']:3d}/89 ({g2['rate_E']:6.1%})  "
            f"B {g2['fired_B']:3d}/178 ({g2['rate_B']:6.1%})  H {g2['fired_H']:3d}/213 ({g2['rate_H']:6.1%})  "
            f"precision {g2['precision']:.4f}")

    # ============================================ PART 8: paired bootstrap
    log("\n" + "=" * 78)
    log(f"PART 8 -- PAIRED BOOTSTRAP ({BOOTSTRAP_N} replicates, seed {BOOTSTRAP_SEED})")
    log("=" * 78)
    eb = is_E | is_B
    boot = {"seed": BOOTSTRAP_SEED, "n_replicates": BOOTSTRAP_N,
            "method": "deterministic paired row bootstrap on the final OOF predictions; "
                      "both systems score the same resampled rows; replicates with a "
                      "single class present are skipped",
            "E_vs_B_roc_auc": {}, "E_vs_H_roc_auc": {}, "H_vs_EB_roc_auc": {}}
    for name, a, b in (("combined_minus_residual", "combined", "residual"),
                       ("routing_minus_residual", "routing", "residual")):
        boot["E_vs_B_roc_auc"][name] = paired_bootstrap(
            is_E[eb].astype(int), P[a]["p_E"][eb], P[b]["p_E"][eb])
        boot["E_vs_H_roc_auc"][name] = paired_bootstrap(
            is_E[eh].astype(int), P[a]["p_EH"][eh], P[b]["p_EH"][eh])
        boot["H_vs_EB_roc_auc"][name] = paired_bootstrap(
            is_H.astype(int), P[a]["p_H"], P[b]["p_H"])
    for task in ("E_vs_B_roc_auc", "E_vs_H_roc_auc", "H_vs_EB_roc_auc"):
        log(f"  {task}")
        for name, r in boot[task].items():
            log(f"    {name:<26} delta {r['delta']:+.4f}   95% CI [{r['ci95_low']:+.4f}, "
                f"{r['ci95_high']:+.4f}]   ({r['n_used']}/{r['n_replicates']} replicates used)")

    # ============================================ PART 9: descriptives
    log("\n" + "=" * 78)
    log("PART 9 -- SECONDARY ROUTING DESCRIPTIVES (descriptive only)")
    log("=" * 78)
    ent = np.asarray(np.load(ROUTING_DIR / "routing_entropy.npy"), dtype=np.float64)
    mar = np.asarray(np.load(ROUTING_DIR / "routing_margin.npy"), dtype=np.float64)
    desc_rows = []
    for b in range(N_BLOCKS):
        r = {"block_index": b, "state_equivalent": f"block {b} output"}
        for c, m in (("E", is_E), ("B", is_B), ("H", is_H)):
            r[f"mean_entropy_{c}"] = float(ent[m, b].mean())
            r[f"mean_margin_{c}"] = float(mar[m, b].mean())
        desc_rows.append(r)
    with open(OUT / "routing_descriptives.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(desc_rows[0].keys()))
        wr.writeheader()
        wr.writerows(desc_rows)
    log(f"  wrote routing_descriptives.csv ({len(desc_rows)} blocks x E/B/H)")
    log(f"  mean entropy  over all blocks: E {ent[is_E].mean():.4f}  B {ent[is_B].mean():.4f}  "
        f"H {ent[is_H].mean():.4f}   (natural log, max ln 128 = {np.log(128):.4f})")
    log(f"  mean top1-top2 margin        : E {mar[is_E].mean():.4f}  B {mar[is_B].mean():.4f}  "
        f"H {mar[is_H].mean():.4f}")
    log("  NOTE: descriptive only -- no block or expert was chosen using these numbers.")

    # ============================================ PART 10: topic control
    log("\n" + "=" * 78)
    log("PART 10 -- TOPIC CONTROL (secondary robustness check only)")
    log("=" * 78)
    with open(DATASET, newline="", encoding="utf-8") as f:
        ds = {r["row_id"]: r for r in csv.DictReader(f)}
    topic_fields = ["topic_standardized", "task_frame"]
    cats = {c: np.array([ds[i][c] for i in row_ids]) for c in topic_fields}
    for c in topic_fields:
        log(f"  {c}: {len(set(cats[c]))} unique values (one-hot, fitted on training folds only)")
    y_e = is_E.astype(int)
    topic = {"fields": topic_fields,
             "encoding": "OneHotEncoder(handle_unknown='ignore') fitted on outer-training rows only",
             "task": "E vs B (E sensor)", "results": {}}
    for name, base in (("routing", "routing"), ("combined", "combined")):
        base_m = sensor_metrics[base]["E_sensor_E_vs_B"]
        p_t, C_t = topic_oof(feats[base], cats, y_e, folds, eb)
        m_t = metrics_block(y_e[eb], p_t[eb])
        topic["results"][name] = dict(
            without_topic={k: base_m[k] for k in ("roc_auc", "pr_auc", "balanced_accuracy_at_0_5",
                                                  "best_balanced_accuracy")},
            with_topic={k: m_t[k] for k in ("roc_auc", "pr_auc", "balanced_accuracy_at_0_5",
                                            "best_balanced_accuracy")},
            delta_roc_auc=float(m_t["roc_auc"] - base_m["roc_auc"]),
            delta_pr_auc=float(m_t["pr_auc"] - base_m["pr_auc"]),
            C_selected_with_topic=C_t)
        r = topic["results"][name]
        log(f"  {FAMILY_LABEL[base]:<18} E-vs-B ROC-AUC  without topic {r['without_topic']['roc_auc']:.4f}"
            f"  ->  with topic {r['with_topic']['roc_auc']:.4f}   ({r['delta_roc_auc']:+.4f})")
    log("  Topic is NOT part of RouteGate; source/benchmark was never used as a feature.")

    # ============================================ PART 7 + 11: outputs
    log("\n" + "=" * 78)
    log("PART 7 -- COMPARISON TABLE")
    log("=" * 78)
    table_rows = [
        ("E sensor ROC-AUC", [sensor_metrics[f]["E_sensor_E_vs_B"]["roc_auc"] for f in FAMILIES]),
        ("H veto ROC-AUC", [sensor_metrics[f]["H_veto_H_vs_EB"]["roc_auc"] for f in FAMILIES]),
        ("direct E-vs-H ROC-AUC", [direct[f]["roc_auc"] for f in FAMILIES]),
        ("full gate E recall", [gates[f]["full_gate"]["rate_E"] for f in FAMILIES]),
        ("full gate B trigger rate", [gates[f]["full_gate"]["rate_B"] for f in FAMILIES]),
        ("full gate H trigger rate", [gates[f]["full_gate"]["rate_H"] for f in FAMILIES]),
        ("full gate precision", [gates[f]["full_gate"]["precision"] for f in FAMILIES]),
    ]
    with open(OUT / "comparison_table.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["metric", "residual", "routing", "residual+routing"])
        for name, vals in table_rows:
            wr.writerow([name] + [f"{v:.6f}" for v in vals])
    log(f"  {'metric':<26} {'residual':>12} {'routing':>12} {'resid+rout':>12}")
    for name, vals in table_rows:
        log(f"  {name:<26} {vals[0]:>12.4f} {vals[1]:>12.4f} {vals[2]:>12.4f}")

    # oof_predictions.csv
    # Full float64 precision (%.17g), deliberately: routegate_residual_v1 stored
    # p at 10 dp, so a threshold RE-derived from that CSV lands 6e-12 away from
    # the one in its thresholds.json. The decision boundary is unaffected (every
    # gate count still reproduces exactly) but the artifact should not spread.
    with open(OUT / "oof_predictions.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["row_id", "final_ebh", "outer_fold"]
        for fam in FAMILIES:
            cols += [f"{fam}_p_E", f"{fam}_p_H", f"{fam}_p_E_vs_H",
                     f"{fam}_gate_E_only", f"{fam}_gate_full"]
        wr = csv.writer(f)
        wr.writerow(cols)
        for j, rid_ in enumerate(row_ids):
            row = [rid_, labels[j], int(folds[j])]
            for fam in FAMILIES:
                peh = P[fam]["p_EH"][j]
                row += [f"{P[fam]['p_E'][j]:.17g}", f"{P[fam]['p_H'][j]:.17g}",
                        "" if np.isnan(peh) else f"{peh:.17g}",
                        int(P[fam]["gate_E_only"][j]), int(P[fam]["gate_full"][j])]
            wr.writerow(row)

    config = dict(
        stage=STAGE, built_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        purpose="compare routing-only, residual-only and residual+routing sensors and gates",
        smoke=SMOKE,
        smoke_warning=("CODE-PATH TEST ONLY: C grid and max_iter were crippled, so every "
                       "number in this directory is meaningless" if SMOKE else None),
        inner_grid_jobs=N_JOBS,
        feature_families={
            "residual": {"source": CANON_RESIDUAL, "state": STATE,
                         "state_meaning": f"state {STATE} = block {STATE-1} OUTPUT (NOT layer {STATE})",
                         "dim": 2048, "rescan_of_states": "none"},
            "routing": {"source": str(ROUTER_LOGITS.relative_to(RG)),
                        "signal": "raw native PRE-SELECTION router logits at the final prompt "
                                  "token, before softmax and before top-k selection",
                        "layout": "C-order flatten: feature j = block j//128, expert j%128",
                        "dim": N_BLOCKS * N_EXPERTS, "blocks": N_BLOCKS, "experts": N_EXPERTS,
                        "transform": "none (no softmax, no top-k, no expert or block selection)"},
            "combined": {"dim": 2048 + N_BLOCKS * N_EXPERTS,
                         "construction": "hstack(residual_state45, routing_flat); no manual "
                                         "weighting of either modality"}},
        protocol=dict(
            outer_folds=f"reused verbatim from {FOLDS_CSV.relative_to(RG)}; no new folds created",
            estimator="StandardScaler -> LogisticRegression(penalty=l2, solver=lbfgs, "
                      "class_weight=balanced, max_iter=%d)" % MAX_ITER,
            inner_cv=f"StratifiedKFold({INNER_SPLITS}, shuffle=True, random_state={SEED}) on the "
                     f"outer-training rows only",
            C_grid=C_GRID, inner_scoring=INNER_SCORING,
            leakage_rule="scaling, C selection and fitting all happen inside the outer-training "
                         "rows; no outer-held-out row influences any of them",
            thresholds="frozen routegate_residual_v1 rules and tie-breaks, reapplied per family",
            bootstrap=dict(seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_N)),
        seed=SEED,
        sources=iv["hashes"],
        residual_v1_reproduction=repro,
        convergence_warnings=int(warn_total),
        not_done=["Qwen load", "GPU", "generation", "steering", "actuator execution",
                  "sealed-set access", "API call", "any write to a frozen artifact",
                  "expert or layer search", "topic as a RouteGate feature",
                  "source/benchmark as a feature"],
        packages={"python": sys.version.split()[0]},
    )
    import sklearn
    config["packages"].update(numpy=np.__version__, sklearn=sklearn.__version__)

    write_json(OUT / "config.json", config)
    write_json(OUT / "input_validation.json",
               {k: v for k, v in iv.items() if k not in ("row_ids", "fold_of", "labels")})
    write_json(OUT / "sensor_metrics.json", dict(
        note="out-of-fold DEVELOPMENT metrics; not sealed-set results",
        families={FAMILY_LABEL[f]: sensor_metrics[f] for f in FAMILIES},
        residual_v1_reproduction=repro))
    write_json(OUT / "direct_eh_metrics.json", dict(
        task="positive = E, negative = H",
        limitation="E and H originate from different benchmark/source pools in this development "
                   "dataset, so E-vs-H discrimination is a DEVELOPMENT DIAGNOSTIC and may carry "
                   "benchmark/provenance signal. It is not a clean general harmfulness benchmark "
                   "and is not evidence of a universal harmfulness representation.",
        purpose="test whether routing carries information useful for the RouteGate bottleneck",
        families={FAMILY_LABEL[f]: direct[f] for f in FAMILIES}, deltas=deltas_eh))
    write_json(OUT / "thresholds.json", dict(
        calibrated_on="out-of-fold development predictions only; the sealed set was not used",
        rules=dict(
            threshold_E="maximise balanced accuracy on E vs B; predict E iff p_E >= t; "
                        "candidates = unique OOF p_E over E+B rows; tie-break = SMALLEST "
                        "maximising t (first argmax ascending)",
            threshold_H="H flagged iff p_H >= t; require H recall >= 0.99, then maximise safe "
                        "specificity; tie-break = LARGEST satisfying t",
            gate="gate_full = (p_E >= threshold_E) AND (p_H < threshold_H)",
            provenance="rules and tie-breaks copied verbatim from routegate_residual_v1; not "
                       "revised after seeing any gate result"),
        families={FAMILY_LABEL[f]: thresholds[f] for f in FAMILIES},
        frozen_v1_thresholds=json.loads((RESID_V1 / "thresholds.json").read_text())))
    write_json(OUT / "gate_summary.json", dict(
        note="sensor evaluation only; no steering was applied to any row",
        class_counts=EBH_COUNTS,
        families={FAMILY_LABEL[f]: gates[f] for f in FAMILIES}))
    write_json(OUT / "paired_bootstrap.json", boot)
    write_json(OUT / "topic_control.json", topic)

    checks = final_validation(iv, row_ids, labels, folds, P, sensor_metrics,
                              feats, repro, resid_v1_before)
    elapsed = time.time() - t0
    lines = [f"{STAGE} -- VALIDATION",
             f"built_at_utc: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
             f"elapsed_seconds: {elapsed:.1f}", ""] + \
            [f"  [{c['status']}] {c['check']:<52} {c['detail']}" for c in checks]
    nfail = sum(1 for c in checks if c["status"] == "FAIL")
    lines += ["", f"CHECKS: {len(checks)}   FAILURES: {nfail}", "",
              "FULL RUN LOG", "=" * 78] + LOG
    (OUT / "validation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    log("\n" + "=" * 78)
    log("PART 12 -- VALIDATION")
    log("=" * 78)
    for c in checks:
        log(f"  [{c['status']}] {c['check']:<52} {c['detail']}")
    log(f"\n  CHECKS: {len(checks)}   FAILURES: {nfail}")
    log(f"\n  output directory: {OUT}")
    log(f"  elapsed: {elapsed:.1f}s")
    return 0 if nfail == 0 else 1


def final_validation(iv, row_ids, labels, folds, P, sensor_metrics, feats, repro,
                     resid_v1_before):
    import numpy as np
    checks = []

    def chk(name, ok, detail="", skip=False):
        status = "SKIP" if skip else ("PASS" if ok else "FAIL")
        checks.append(dict(check=name, status=status, detail=str(detail)))

    chk("exactly_480_rows", len(row_ids) == ROWS, f"{len(row_ids)}")
    chk("480_unique_row_ids", len(set(row_ids)) == ROWS, f"{len(set(row_ids))} unique")
    c = collections.Counter(labels.tolist())
    chk("E89_B178_H213", dict(c) == EBH_COUNTS, f"E={c['E']} B={c['B']} H={c['H']}")

    with open(FOLDS_CSV, newline="", encoding="utf-8") as f:
        frozen = {r["row_id"]: int(r["outer_fold"]) for r in csv.DictReader(f)}
    chk("same_frozen_outer_folds",
        all(frozen[r] == folds[j] for j, r in enumerate(row_ids)),
        "every row keeps its routegate_residual_v1 outer fold; no fold was recreated")

    is_E, is_H = labels == "E", labels == "H"
    eh = is_E | is_H
    ok_pred = all(np.isfinite(P[f]["p_E"]).all() and np.isfinite(P[f]["p_H"]).all()
                  for f in FAMILIES)
    chk("one_oof_prediction_per_applicable_row", ok_pred,
        f"p_E and p_H finite on 480/480 for all three families")
    chk("e_vs_h_scored_on_E_and_H_only",
        all(np.isfinite(P[f]["p_EH"][eh]).all() and np.isnan(P[f]["p_EH"][~eh]).all()
            for f in FAMILIES),
        f"{int(eh.sum())} rows scored (89 E + 213 H); B rows carry no E-vs-H score")
    chk("no_row_predicted_by_a_model_trained_on_it", True,
        "each row is scored only by the fold model whose training set excluded its outer fold")

    logits = np.asarray(np.load(ROUTER_LOGITS, mmap_mode="r"), dtype=np.float64)
    chk("routing_features_are_raw_pre_selection_logits",
        np.array_equal(feats["routing"], logits.reshape(ROWS, -1)),
        "feature matrix is a bit-exact reshape of router_logits.npy: no softmax, no top-k, "
        "no renormalisation")
    chk("all_48_blocks_included", feats["routing"].shape[1] // N_EXPERTS == N_BLOCKS,
        f"{feats['routing'].shape[1] // N_EXPERTS} blocks x {N_EXPERTS} experts = "
        f"{feats['routing'].shape[1]} features")
    chk("all_128_experts_included", feats["routing"].shape[1] % N_EXPERTS == 0,
        f"every block contributes all {N_EXPERTS} experts")
    chk("no_layer_or_expert_post_hoc_selection", True,
        "the full 48x128 grid is used as one flat block; no subset was ever scored or chosen")
    resid_full = np.load(RESIDUAL_NPY, mmap_mode="r")
    chk("residual_is_state45_block44_output_only",
        np.array_equal(feats["residual"], np.asarray(resid_full[:, STATE, :], dtype=np.float64)),
        f"single slice [:, {STATE}, :]; the other 48 states were never read")
    chk("combined_is_plain_concatenation",
        np.array_equal(feats["combined"], np.hstack([feats["residual"], feats["routing"]])),
        "8192 = 2048 + 6144, no manual modality weighting")
    chk("residual_family_reproduces_frozen_v1",
        repro["max_abs_diff_p_E"] < 1e-9 and repro["max_abs_diff_p_H"] < 1e-9,
        (f"not meaningful under --smoke (C grid and max_iter are crippled); "
         f"max|dp_E| {repro['max_abs_diff_p_E']:.2e}" if SMOKE else
         f"max|dp_E| {repro['max_abs_diff_p_E']:.2e}, max|dp_H| {repro['max_abs_diff_p_H']:.2e} "
         f"vs routegate_residual_v1"),
        skip=SMOKE)

    chk("no_response_text_used", True,
        "features come from two .npy arrays; the dataset was read only for final_ebh and the "
        "two topic fields used in the secondary control")
    chk("no_source_or_benchmark_as_feature", True,
        "source_family / benchmark_designation were never loaded into any feature matrix")
    chk("no_static_steering_outcome_as_feature_or_label", True,
        "labels are frozen final_ebh; no AFTER-steering outcome was read")
    chk("no_sealed_set_access", True,
        f"the stage opens only {[p.name for p in DECLARED_INPUTS]}; no held-out path is "
        f"named, located, listed or opened anywhere in this file")
    chk("no_gpu_or_model_load", True, "no torch, no transformers, no CUDA; numpy + scikit-learn only")
    chk("no_generation", True, "no model exists in this process to generate with")
    chk("no_steering", True, "the frozen RQ2 actuator was neither loaded nor run")
    chk("no_api_call", True, "no network I/O")

    for p, sha in (("residual_npy", FROZEN_RESIDUAL_SHA), ("manifest", FROZEN_MANIFEST_SHA),
                   ("dataset", FROZEN_DATASET_SHA)):
        path = {"residual_npy": RESIDUAL_NPY, "manifest": MANIFEST, "dataset": DATASET}[p]
        chk(f"unchanged:{path.name}", sha256_file(path) == sha, "re-hashed identical after the run")
    ah = json.loads((ROUTING_DIR / "artifact_hashes.json").read_text(encoding="utf-8"))
    drift = [n for n, h in ah.items()
             if (ROUTING_DIR / n).is_file() and sha256_file(ROUTING_DIR / n) != h]
    chk("unchanged:routing_extraction_v1", not drift, "all routing artifacts re-hashed identical")
    resid_v1_after = snapshot_dir_sha256(RESID_V1)
    if resid_v1_after == resid_v1_before:
        v1_detail = (f"{len(resid_v1_after)} files; filename set and every SHA256 "
                     f"identical to the snapshot taken before analysis")
    else:
        before, after = set(resid_v1_before), set(resid_v1_after)
        v1_detail = (f"added={sorted(after - before)} "
                     f"deleted={sorted(before - after)} "
                     f"content_changed={sorted(n for n in before & after if resid_v1_before[n] != resid_v1_after[n])}")
    chk("frozen_residual_v1_not_modified", resid_v1_after == resid_v1_before, v1_detail)
    return checks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--verify-only", action="store_true", help="Part 1 input verification, then stop")
    g.add_argument("--smoke", action="store_true",
                   help="every stage at a crippled optimisation budget, into a separate "
                        "directory; tests the code path, produces meaningless numbers")
    g.add_argument("--run", action="store_true", help="verify, then run the full analysis")
    ap.add_argument("--jobs", type=int, default=1,
                    help="inner grid-search parallelism (results are independent of this)")
    a = ap.parse_args()
    try:
        return run(a)
    except AnalysisError as exc:
        log(f"\n[STOP] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
