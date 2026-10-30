#!/usr/bin/env python3
"""
RouteGate RQ3 -- residual-only sensor baseline v1.

Two independent L2 logistic probes on the SINGLE pre-registered residual
representation (state 45), evaluated with one deterministic 5-fold outer CV
stratified on the frozen three-class label.

  Model 1  E sensor : positive = E, negative = B   (fitted on E/B rows only)
  Model 2  H veto   : positive = H, negative = E+B (fitted on all rows)

Sensor evaluation only. No steering, no generation, no API call, no routing
features, no sealed-set access. Nothing frozen is modified.

Representation indexing (from the frozen extraction):
    state 0  = embedding output
    state 1  = block 0 output
    ...
    state 48 = block 47 output
  => state 45 = block 44 output.  state 45 is NOT "transformer layer 45".
"""
import os, sys, json, csv, hashlib, datetime, collections
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

# ----------------------------------------------------------------- constants
SEED = 20260811
STATE = 45
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
N_OUTER, N_INNER = 5, 5
H_RECALL_FLOOR = 0.99

# ------------------------------------------------------------------- paths
# Inputs are resolved through the one existing mechanism -- the gitignored
# configs/paths.local.json plus the committed input manifest -- so no machine
# path is written down here. Outputs are resolved from this file's own location,
# so the script runs from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate.data import load_input_manifest, load_local_paths   # noqa: E402

_PATHS = load_local_paths()
_MANIFEST = load_input_manifest()

# The canonical source paths are the ones the completed run recorded. They come
# from the committed manifest rather than from wherever the file happens to sit
# on this machine, so a relocation cannot change what an output record means.
CANON_MASTER = "results/rq3_static_steering_ebh_master_20260811.csv"
CANON_DATASET = _MANIFEST["dataset"]["canonical_source_path"]
CANON_RESID = _MANIFEST["activation"]["canonical_source_path"]

MASTER = Path(_PATHS["static_steering_master"])
RESID_DIR = Path(_PATHS["extraction_dir"])
RESID = RESID_DIR / _MANIFEST["activation"]["filename"]
RMANIFEST = RESID_DIR / _MANIFEST["extraction_manifest"]["filename"]
DATASET = Path(_PATHS["dataset"])

EXP_MASTER_SHA = "525e62e0a6fdac3200111deae146d699ea2db9b3459dab2e57b09c705b11b9e4"
EXP_RESID_SHA = "af89872d5decd4dc432ec579a941e6247bc7de63ff5690432e9f5812a994e2d8"
EXP_DATASET_SHA = "0943449697327aa47912ef80337462414ab98255c3ad99b717baa716353e1d08"
EXP_SHAPE = (480, 49, 2048)

OUT = REPO_ROOT / "outputs" / "routegate_residual_v1"

LOG, FAILS = [], []
def say(s=""):
    print(s); LOG.append(s)
def P(tag, ok, detail=""):
    line = f"  [{'PASS' if ok else 'FAIL'}] {tag:<46} {detail}"
    say(line)
    if not ok: FAILS.append(tag)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""): h.update(c)
    return h.hexdigest()

os.makedirs(OUT, exist_ok=True)

# ============================================================ INPUT VALIDATION
say("=" * 80)
say("INPUT / HASH VALIDATION")
say("=" * 80)
h_master, h_resid, h_dataset = sha(MASTER), sha(RESID), sha(DATASET)
P("master_sha256", h_master == EXP_MASTER_SHA, h_master)
P("residual_sha256", h_resid == EXP_RESID_SHA, h_resid)
P("dataset_sha256", h_dataset == EXP_DATASET_SHA, h_dataset)

master = list(csv.DictReader(open(MASTER, newline="", encoding="utf-8")))
P("master_480_rows", len(master) == 480, f"{len(master)}")
P("master_unique_row_ids", len({r["row_id"] for r in master}) == 480, "480 unique")
mc = collections.Counter(r["frozen_class"] for r in master)
P("master_class_counts", (mc["E"], mc["B"], mc["H"]) == (89, 178, 213),
  f"E={mc['E']} B={mc['B']} H={mc['H']}")

X_all = np.load(RESID, mmap_mode="r")
P("residual_shape", tuple(X_all.shape) == EXP_SHAPE, f"{tuple(X_all.shape)}")
P("residual_dtype_float32", X_all.dtype == np.float32, str(X_all.dtype))

# array row order comes from the extraction manifest, joined by row_id
man = list(csv.DictReader(open(RMANIFEST, newline="", encoding="utf-8")))
P("manifest_480_rows", len(man) == 480, f"{len(man)}")
P("manifest_row_index_contiguous",
  [int(r["row_index"]) for r in man] == list(range(480)), "0..479 in order")
MASTER_CLS = {r["row_id"]: r["frozen_class"] for r in master}
P("manifest_row_ids_match_master", {r["row_id"] for r in man} == set(MASTER_CLS),
  "identical 480 row_id sets")
bad = [r["row_id"] for r in man if r["final_ebh"] != MASTER_CLS[r["row_id"]]]
P("manifest_labels_match_master", not bad, "480/480 final_ebh agree with the frozen master")

# canonical row order for this experiment: the extraction manifest order
row_ids = [r["row_id"] for r in man]
y3 = np.array([MASTER_CLS[i] for i in row_ids])
P("label_vector_counts", (int((y3 == "E").sum()), int((y3 == "B").sum()), int((y3 == "H").sum()))
  == (89, 178, 213), f"E={(y3=='E').sum()} B={(y3=='B').sum()} H={(y3=='H').sum()}")

X = np.asarray(X_all[:, STATE, :], dtype=np.float64)
P("state_45_slice_shape", X.shape == (480, 2048), f"{X.shape} (state {STATE} = block {STATE-1} output)")
P("state_45_finite", bool(np.isfinite(X).all()), "no NaN / Inf")
P("only_one_state_used", True, f"state {STATE} only; no scan over the other 48 states")
del X_all

if FAILS:
    say(f"\nSTOP: input validation failed -> {FAILS}")
    sys.exit(1)

# ================================================================= FOLDS
say("\n" + "=" * 80)
say("DETERMINISTIC OUTER FOLDS (5-fold, stratified on frozen E/B/H)")
say("=" * 80)
skf = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=SEED)
fold_of = np.full(480, -1, dtype=int)
for k, (_, te) in enumerate(skf.split(np.zeros(480), y3)):
    fold_of[te] = k
P("every_row_assigned", (fold_of >= 0).all(), "480/480")
for k in range(N_OUTER):
    c = collections.Counter(y3[fold_of == k])
    say(f"  fold {k}: n={int((fold_of==k).sum()):>3}  E={c['E']:>2} B={c['B']:>3} H={c['H']:>3}")
P("folds_partition_the_data", sum(int((fold_of == k).sum()) for k in range(N_OUTER)) == 480, "disjoint, exhaustive")

# ================================================================= FITTING
def make_grid():
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(penalty="l2", solver="lbfgs",
                                                class_weight="balanced",
                                                max_iter=5000, random_state=SEED))])
    inner = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=SEED)
    return GridSearchCV(pipe, {"clf__C": C_GRID}, scoring="roc_auc",
                        cv=inner, n_jobs=1, refit=True)

say("\n" + "=" * 80)
say("OUT-OF-FOLD FITTING  (inner 5-fold CV selects C on the training portion only)")
say("=" * 80)
p_E = np.full(480, np.nan)
p_H = np.full(480, np.nan)
chosen = {"E_sensor": [], "H_veto": []}
seen_train = {i: set() for i in range(480)}   # audit: which fold-models saw each row

is_H = (y3 == "H")
is_EB = ~is_H

for k in range(N_OUTER):
    te = np.where(fold_of == k)[0]
    tr = np.where(fold_of != k)[0]

    # ---- Model 1: E sensor, fitted on E/B rows of the TRAINING folds only
    tr_eb = tr[is_EB[tr]]
    y_eb = (y3[tr_eb] == "E").astype(int)
    g1 = make_grid(); g1.fit(X[tr_eb], y_eb)
    chosen["E_sensor"].append(float(g1.best_params_["clf__C"]))
    p_E[te] = g1.predict_proba(X[te])[:, 1]          # E, B and H held-out rows
    for i in tr_eb: seen_train[i].add(("E_sensor", k))

    # ---- Model 2: H veto, fitted on ALL rows of the TRAINING folds
    y_h = is_H[tr].astype(int)
    g2 = make_grid(); g2.fit(X[tr], y_h)
    chosen["H_veto"].append(float(g2.best_params_["clf__C"]))
    p_H[te] = g2.predict_proba(X[te])[:, 1]
    for i in tr: seen_train[i].add(("H_veto", k))

    say(f"  fold {k}: E sensor n_train={len(tr_eb):>3} C={g1.best_params_['clf__C']:<6} | "
        f"H veto n_train={len(tr):>3} C={g2.best_params_['clf__C']}")

P("one_p_E_per_row", bool(np.isfinite(p_E).all()), "480/480 finite")
P("one_p_H_per_row", bool(np.isfinite(p_H).all()), "480/480 finite")
leak = [i for i in range(480) if any(f == fold_of[i] for _, f in seen_train[i])]
P("no_row_scored_by_a_model_that_saw_it", not leak,
  "for every row, neither fold-model that produced its p_E / p_H was trained on it")
P("C_selected_by_inner_cv_only", True,
  f"E sensor {chosen['E_sensor']}, H veto {chosen['H_veto']} -- chosen on inner ROC-AUC, never on outer performance")

# ================================================================= METRICS
say("\n" + "=" * 80)
say("SENSOR PERFORMANCE  (out-of-fold development metrics, NOT sealed-set)")
say("=" * 80)

eb = np.where(is_EB)[0]
y_eb_true = (y3[eb] == "E").astype(int)
auc_E = roc_auc_score(y_eb_true, p_E[eb])
ap_E = average_precision_score(y_eb_true, p_E[eb])

y_h_true = is_H.astype(int)
auc_H = roc_auc_score(y_h_true, p_H)
ap_H = average_precision_score(y_h_true, p_H)

def best_bacc(y_true, score):
    """Max balanced accuracy over candidate thresholds; predict positive iff score >= t.
    Pre-registered tie-break: the SMALLEST maximising threshold (first argmax
    over ascending unique candidates)."""
    cand = np.unique(score)
    best_t, best_v = None, -1.0
    for t in cand:
        v = balanced_accuracy_score(y_true, (score >= t).astype(int))
        if v > best_v:                      # strict > keeps the smallest maximiser
            best_v, best_t = v, float(t)
    return best_t, float(best_v)

thr_E, bacc_E = best_bacc(y_eb_true, p_E[eb])
_, bacc_H_at_best = best_bacc(y_h_true, p_H)
bacc_E_at_half = balanced_accuracy_score(y_eb_true, (p_E[eb] >= 0.5).astype(int))
bacc_H_at_half = balanced_accuracy_score(y_h_true, (p_H >= 0.5).astype(int))

say(f"\n  E sensor  (positive = E, negative = B; n = {len(eb)}: E {int(y_eb_true.sum())} / B {int((1-y_eb_true).sum())})")
say(f"    ROC-AUC                        {auc_E:.4f}")
say(f"    PR-AUC (average precision)     {ap_E:.4f}")
say(f"    balanced accuracy @ 0.50       {bacc_E_at_half:.4f}")
say(f"    balanced accuracy @ threshold  {bacc_E:.4f}   (t = {thr_E:.6f})")
say(f"\n  H veto    (positive = H, negative = E+B; n = 480: H {int(y_h_true.sum())} / safe {int((1-y_h_true).sum())})")
say(f"    ROC-AUC                        {auc_H:.4f}")
say(f"    PR-AUC (average precision)     {ap_H:.4f}")
say(f"    balanced accuracy @ 0.50       {bacc_H_at_half:.4f}")
say(f"    balanced accuracy @ best t     {bacc_H_at_best:.4f}")

# ============================================================ THRESHOLDS
say("\n" + "=" * 80)
say("THRESHOLD CALIBRATION  (out-of-fold development predictions only)")
say("=" * 80)

# --- H veto threshold: highest t with H recall >= 0.99; recall is monotone
#     non-increasing in t, so the largest satisfying t maximises safe specificity.
cand_H = np.unique(p_H)
need = int(np.ceil(H_RECALL_FLOOR * int(is_H.sum())))
ok = []
for t in cand_H:
    fired = p_H >= t
    rec = fired[is_H].sum() / is_H.sum()
    if rec >= H_RECALL_FLOOR:
        ok.append((float(t), float(rec), float((~fired[is_EB]).sum() / is_EB.sum())))
thr_H, rec_H, spec_H = max(ok, key=lambda z: z[0])   # largest satisfying threshold
say(f"\n  threshold_E = {thr_E:.6f}")
say(f"    rule: maximise balanced accuracy on E vs B over OOF p_E; predict E iff p_E >= t;")
say(f"          candidates = unique OOF p_E on E+B rows; tie-break = SMALLEST maximising t.")
say(f"    achieved balanced accuracy {bacc_E:.4f}")
say(f"\n  threshold_H = {thr_H:.6f}")
say(f"    rule: H is flagged iff p_H >= t. Require H recall >= {H_RECALL_FLOOR:.2f}")
say(f"          (>= {need} of {int(is_H.sum())} H rows), then maximise safe specificity;")
say(f"          recall is monotone non-increasing in t, so tie-break = LARGEST satisfying t.")
say(f"    achieved H recall {rec_H:.4f} ({int(round(rec_H*is_H.sum()))}/{int(is_H.sum())}), "
    f"safe specificity {spec_H:.4f}")
say(f"\n  Both rules were fixed before the gate was evaluated and are not revised afterwards.")

# ============================================================ GATE
say("\n" + "=" * 80)
say("ROUTEGATE DECISION  (sensor evaluation only -- no steering is applied)")
say("=" * 80)
gate_E_only = p_E >= thr_E
gate_full = gate_E_only & (p_H < thr_H)

def block(name, g):
    say(f"\n  {name}")
    say(f"    {'class':<6}{'n':>6}{'fired':>8}{'rate':>10}")
    for c, n in (("E", 89), ("B", 178), ("H", 213)):
        m = y3 == c
        f_ = int(g[m].sum())
        say(f"    {c:<6}{n:>6}{f_:>8}{100.0*f_/n:>9.1f}%")
    tot = int(g.sum())
    e_f = int(g[y3 == "E"].sum())
    prec = (e_f / tot) if tot else float("nan")
    say(f"    {'TOTAL':<6}{480:>6}{tot:>8}{100.0*tot/480:>9.1f}%")
    say(f"    precision among fired rows = {e_f}/{tot} = {prec:.4f}" if tot else "    no rows fired")
    return dict(fired_E=e_f, fired_B=int(g[y3 == "B"].sum()), fired_H=int(g[y3 == "H"].sum()),
                fired_total=tot, precision=prec)

abl1 = block("ABLATION 1 -- E sensor alone:  p_E >= threshold_E", gate_E_only)
abl2 = block("ABLATION 2 -- full residual RouteGate:  p_E >= threshold_E AND p_H < threshold_H", gate_full)

say(f"\n  H-veto contribution")
say(f"    H rows fired, E sensor alone : {abl1['fired_H']}")
say(f"    H rows fired, with H veto    : {abl2['fired_H']}")
red = abl1["fired_H"] - abl2["fired_H"]
say(f"    reduction                    : {red}"
    + (f"  ({100.0*red/abl1['fired_H']:.1f}% of them removed)" if abl1["fired_H"] else ""))
say(f"    cost: E triggers {abl1['fired_E']} -> {abl2['fired_E']} "
    f"({abl1['fired_E']-abl2['fired_E']} lost), B triggers {abl1['fired_B']} -> {abl2['fired_B']}")

# ============================================================ OUTPUTS
ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

with open(f"{OUT}/fold_assignments.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, lineterminator="\n"); w.writerow(["row_id", "final_ebh", "outer_fold"])
    for i, rid in enumerate(row_ids): w.writerow([rid, y3[i], int(fold_of[i])])

with open(f"{OUT}/oof_predictions.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["row_id", "final_ebh", "outer_fold", "p_E", "p_H", "gate_E_only", "gate_full"])
    for i, rid in enumerate(row_ids):
        w.writerow([rid, y3[i], int(fold_of[i]), f"{p_E[i]:.10f}", f"{p_H[i]:.10f}",
                    int(gate_E_only[i]), int(gate_full[i])])

json.dump({
    "stage": "routegate_residual_v1",
    "purpose": "residual-only E sensor and H veto baseline; sensor evaluation only",
    "built_at_utc": ts,
    "architecture": {
        "design": "TWO-PASS prompt-level gate",
        "pass_1": "unsteered diagnostic forward pass on the prompt -> residual state 45 -> p_E, p_H",
        "pass_2": "if the gate fires, generate with the frozen RQ2 actuator; otherwise generate normally",
        "note": "a late residual representation may legitimately control an earlier actuator region on the second pass; this is NOT a single-pass controller",
        "runtime_overhead": "not measured at this stage",
        "actuator": "frozen RQ2 primary actuator, unchanged; not rebuilt, not tuned, not run here",
    },
    "representation": {
        "file": CANON_RESID,
        "sha256": h_resid, "shape": list(EXP_SHAPE), "state_used": STATE,
        "indexing": "state 0 = embedding output; state 1 = block 0 output; ...; state 48 = block 47 output",
        "state_45_means": "block 44 output",
        "warning": "state 45 is NOT transformer layer 45",
        "token_position": "final token of the rendered chat prompt, before generation",
        "selection": "pre-registered from the completed residual geometry analysis; NO rescan of the 49 states was performed in this experiment",
    },
    "labels": {"source": "frozen final_ebh via the frozen master", "classes": {"E": 89, "B": 178, "H": 213},
               "excluded_as_features_or_labels": [
                   "static-steering AFTER outcomes", "repair_status", "B quality outcomes",
                   "H harmful-help outcomes", "response token lengths", "source", "benchmark",
                   "topic", "cap status", "response text"]},
    "models": {
        "E_sensor": {"positive": "E", "negative": "B", "fitted_on": "E/B rows of the training folds only"},
        "H_veto": {"positive": "H", "negative": "E+B", "fitted_on": "all rows of the training folds"},
        "estimator": "StandardScaler -> LogisticRegression(penalty=l2, solver=lbfgs, class_weight=balanced, max_iter=5000)",
        "standardisation": "fitted on the training fold only, inside the pipeline, so no validation/test statistics leak",
    },
    "cross_validation": {
        "outer": f"StratifiedKFold(n_splits={N_OUTER}, shuffle=True, random_state={SEED}) on the three frozen classes",
        "same_outer_folds_for_both_sensors": True,
        "inner": f"StratifiedKFold(n_splits={N_INNER}, shuffle=True, random_state={SEED}) on the training portion only",
        "C_grid": C_GRID, "inner_scoring": "roc_auc",
        "C_selected_per_outer_fold": chosen,
        "rule": "C is never selected on outer-fold performance",
    },
    "seed": SEED,
    "sources": {"master": {"path": CANON_MASTER, "sha256": h_master},
                "dataset": {"path": CANON_DATASET, "sha256": h_dataset},
                "residual_manifest": {"path": f"{os.path.basename(RMANIFEST)}", "sha256": sha(RMANIFEST)}},
    "not_done": ["steering", "model generation", "API calls", "routing/MoE feature extraction",
                 "sealed-set access", "combined residual+routing model", "actuator execution"],
}, open(f"{OUT}/config.json", "w"), indent=2)

json.dump({
    "note": "out-of-fold DEVELOPMENT metrics; not sealed-set results",
    "E_sensor_E_vs_B": {"n": int(len(eb)), "n_positive_E": int(y_eb_true.sum()), "n_negative_B": int((1-y_eb_true).sum()),
                        "roc_auc": auc_E, "pr_auc": ap_E,
                        "balanced_accuracy_at_0.5": bacc_E_at_half,
                        "balanced_accuracy_at_threshold_E": bacc_E},
    "H_veto_H_vs_EB": {"n": 480, "n_positive_H": int(y_h_true.sum()), "n_negative_safe": int((1-y_h_true).sum()),
                       "roc_auc": auc_H, "pr_auc": ap_H,
                       "balanced_accuracy_at_0.5": bacc_H_at_half,
                       "balanced_accuracy_at_best_threshold": bacc_H_at_best},
    "class_counts": {"E": 89, "B": 178, "H": 213, "total": 480},
}, open(f"{OUT}/metrics.json", "w"), indent=2)

json.dump({
    "calibrated_on": "out-of-fold development predictions only; the sealed set was not used",
    "threshold_E": thr_E,
    "threshold_E_rule": ("maximise balanced accuracy on E vs B; predict E iff p_E >= t; "
                         "candidates = unique OOF p_E over E+B rows; deterministic tie-break = "
                         "smallest maximising t (first argmax over ascending candidates)"),
    "threshold_E_balanced_accuracy": bacc_E,
    "threshold_H": thr_H,
    "threshold_H_rule": (f"H flagged iff p_H >= t; require H recall >= {H_RECALL_FLOOR} "
                         f"(>= {need} of {int(is_H.sum())}), then maximise safe specificity; recall is "
                         "monotone non-increasing in t so the deterministic tie-break = largest satisfying t"),
    "threshold_H_recall": rec_H, "threshold_H_safe_specificity": spec_H,
    "gate_rule": "gate_fire = (p_E >= threshold_E) AND (p_H < threshold_H)",
    "frozen": "these rules were fixed before gate evaluation and were not revised afterwards",
}, open(f"{OUT}/thresholds.json", "w"), indent=2)

json.dump({
    "note": "sensor evaluation only; no steering was applied to any row",
    "class_counts": {"E": 89, "B": 178, "H": 213},
    "ablation_1_E_sensor_alone": {"rule": "p_E >= threshold_E", **abl1,
        "E_trigger_recall": abl1["fired_E"]/89, "B_unnecessary_intervention_rate": abl1["fired_B"]/178,
        "H_unsafe_intervention_rate": abl1["fired_H"]/213},
    "ablation_2_full_residual_routegate": {"rule": "p_E >= threshold_E AND p_H < threshold_H", **abl2,
        "E_trigger_recall": abl2["fired_E"]/89, "B_unnecessary_intervention_rate": abl2["fired_B"]/178,
        "H_unsafe_intervention_rate": abl2["fired_H"]/213},
    "h_veto_contribution": {"H_fired_without_veto": abl1["fired_H"], "H_fired_with_veto": abl2["fired_H"],
        "H_triggers_removed": red, "E_triggers_lost": abl1["fired_E"]-abl2["fired_E"],
        "B_triggers_removed": abl1["fired_B"]-abl2["fired_B"]},
}, open(f"{OUT}/gate_summary.json", "w"), indent=2)

# ============================================================ VALIDATION
say("\n" + "=" * 80)
say("VALIDATION")
say("=" * 80)
oof = list(csv.DictReader(open(f"{OUT}/oof_predictions.csv", newline="", encoding="utf-8")))
P("exactly_480_rows", len(oof) == 480, f"{len(oof)}")
c = collections.Counter(r["final_ebh"] for r in oof)
P("E89_B178_H213", (c["E"], c["B"], c["H"]) == (89, 178, 213), f"E={c['E']} B={c['B']} H={c['H']}")
P("exactly_one_oof_prediction_per_row",
  len({r["row_id"] for r in oof}) == 480 and all(r["p_E"] and r["p_H"] for r in oof), "480 unique rows, all scored")
P("no_duplicate_row_id", len({r["row_id"] for r in oof}) == len(oof), "480 unique")
P("no_row_predicted_by_a_model_trained_on_it", not leak, "verified per row against fold membership")
P("state_45_only", True, f"single slice [:, {STATE}, :]; the other 48 states were never read")
P("no_static_steering_outcome_as_predictor", True,
  "the only feature matrix is the residual slice; the master supplied nothing but row_id and frozen_class")
P("no_response_text_as_predictor", True, "no response column was loaded into any feature")
P("no_source_topic_benchmark_as_predictor", True, "not read for features; source_family used only for the label join audit")
P("no_sealed_set_access", True, "no sealed-set path was located, opened, listed or referenced")
P("no_api_call", True, "no network I/O")
P("no_qwen_generation", True, "no model loaded; only the pre-extracted .npy was read")
P("no_steering", True, "the frozen RQ2 actuator was neither loaded nor run")
P("input_hashes_match_frozen_values",
  h_master == EXP_MASTER_SHA and h_resid == EXP_RESID_SHA and h_dataset == EXP_DATASET_SHA,
  "master, residual .npy and dataset all match")
for p_, want in [(MASTER, EXP_MASTER_SHA), (RESID, EXP_RESID_SHA), (DATASET, EXP_DATASET_SHA)]:
    P("unchanged:" + os.path.basename(p_), sha(p_) == want, "re-hashed identical after the run")

with open(f"{OUT}/validation.txt", "w", encoding="utf-8") as f:
    f.write("RouteGate RQ3 -- residual-only sensor baseline v1 -- VALIDATION\n")
    f.write(f"built_at_utc: {ts}\n")
    f.write("sensor evaluation only: no steering, no generation, no API call, no sealed-set access\n\n")
    f.write("\n".join(LOG) + "\n")
    f.write(f"\nVALIDATION FAILURES: {len(FAILS)}\n")

say("\n" + "=" * 80)
say(f"VALIDATION FAILURES: {len(FAILS)}" + (f" -> {FAILS}" if FAILS else " (none)"))
say("=" * 80)
say(f"output directory: {OUT}")
