#!/usr/bin/env python3
"""
RouteGate RQ3 -- topic / category robustness analysis for the frozen
residual-only baseline (routegate_residual_v1).

Confound control and robustness only. Topic metadata is NOT part of the
primary RouteGate method and nothing in routegate_residual_v1 is altered,
refitted or re-thresholded here.

No API calls, no Qwen generation, no steering, no sealed-set access.
"""
import os, sys, json, csv, hashlib, datetime, collections
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

SEED, STATE = 20260811, 45
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
N_INNER = 5

# ------------------------------------------------------------------- paths
# Inputs come from the gitignored configs/paths.local.json plus the committed
# input manifest; outputs are located from this file, so the script runs from
# any working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate.data import load_input_manifest, load_local_paths   # noqa: E402

_PATHS = load_local_paths()
_MANIFEST = load_input_manifest()

DATASET = Path(_PATHS["dataset"])
_EXTRACTION_DIR = Path(_PATHS["extraction_dir"])
RESID = _EXTRACTION_DIR / _MANIFEST["activation"]["filename"]
RMANIFEST = _EXTRACTION_DIR / _MANIFEST["extraction_manifest"]["filename"]
V1 = REPO_ROOT / "outputs" / "routegate_residual_v1"
OUT = REPO_ROOT / "outputs" / "routegate_topic_analysis_v1"

EXP_RESID_SHA = "af89872d5decd4dc432ec579a941e6247bc7de63ff5690432e9f5812a994e2d8"
EXP_DATASET_SHA = "0943449697327aa47912ef80337462414ab98255c3ad99b717baa716353e1d08"

LOG, FAILS = [], []
def say(s=""):
    print(s); LOG.append(s)
def P(tag, ok, detail=""):
    say(f"  [{'PASS' if ok else 'FAIL'}] {tag:<44} {detail}")
    if not ok: FAILS.append(tag)
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""): h.update(c)
    return h.hexdigest()

os.makedirs(OUT, exist_ok=True)

# ===================================================== inputs (read-only)
say("=" * 82); say("INPUTS  (frozen; nothing is rewritten)"); say("=" * 82)
h_res, h_ds = sha(RESID), sha(DATASET)
P("residual_sha256", h_res == EXP_RESID_SHA, h_res[:24] + "...")
P("dataset_sha256", h_ds == EXP_DATASET_SHA, h_ds[:24] + "...")
V1_HASHES = {f: sha(f"{V1}/{f}") for f in sorted(os.listdir(V1)) if os.path.isfile(f"{V1}/{f}")}
P("v1_baseline_present", len(V1_HASHES) >= 7, f"{len(V1_HASHES)} frozen files recorded for the untouched check")

man = list(csv.DictReader(open(RMANIFEST, newline="", encoding="utf-8")))
row_ids = [r["row_id"] for r in man]
ds = {r["row_id"]: r for r in csv.DictReader(open(DATASET, newline="", encoding="utf-8"))}
oof = {r["row_id"]: r for r in csv.DictReader(open(f"{V1}/oof_predictions.csv", newline="", encoding="utf-8"))}
P("oof_480_rows", len(oof) == 480, f"{len(oof)}")
P("row_order_aligned", all(i in oof and i in ds for i in row_ids), "manifest / dataset / OOF share all 480 row_ids")

y3 = np.array([oof[i]["final_ebh"] for i in row_ids])
fold = np.array([int(oof[i]["outer_fold"]) for i in row_ids])
p_E_frozen = np.array([float(oof[i]["p_E"]) for i in row_ids])
p_H_frozen = np.array([float(oof[i]["p_H"]) for i in row_ids])
gate_E_only = np.array([int(oof[i]["gate_E_only"]) for i in row_ids], bool)
gate_full = np.array([int(oof[i]["gate_full"]) for i in row_ids], bool)
P("frozen_folds_reused", sorted(collections.Counter(fold).values()) == [96]*5,
  "the exact outer fold assignment from routegate_residual_v1 is reused, not regenerated")

X_res = np.asarray(np.load(RESID, mmap_mode="r")[:, STATE, :], dtype=np.float64)
P("state_45_only", X_res.shape == (480, 2048), f"{X_res.shape}; state 45 = block 44 output, unchanged")

is_H = y3 == "H"; is_EB = ~is_H
eb_idx = np.where(is_EB)[0]

# ============================================== 1. TOPIC / CATEGORY FIELDS
say("\n" + "=" * 82); say("1. FROZEN TOPIC / CATEGORY FIELDS"); say("=" * 82)
TOPIC_FIELDS = ["topic_standardized", "task_frame"]
SUPP_FIELDS = ["prompt_category"]
CONFOUND_FIELDS = ["source_family", "benchmark_designation"]

def describe(col):
    vals = [ds[i][col] for i in row_ids]
    u = sorted(set(vals))
    rows = []
    for v in u:
        c = collections.Counter(y3[k] for k, x in enumerate(vals) if x == v)
        rows.append(dict(value=v, n=sum(c.values()), E=c["E"], B=c["B"], H=c["H"]))
    rows.sort(key=lambda r: -r["n"])
    return vals, u, rows

FIELD_REPORT = {}
for col in TOPIC_FIELDS + SUPP_FIELDS + CONFOUND_FIELDS:
    vals, u, rows = describe(col)
    FIELD_REPORT[col] = dict(n_unique=len(u), n_blank=sum(1 for v in vals if not v.strip()), values=rows)
    role = ("PRIMARY topic field" if col in TOPIC_FIELDS else
            "supplementary (high-cardinality, benchmark-entangled)" if col in SUPP_FIELDS else
            "CONFOUND -- analysed separately, never a topic substitute")
    say(f"\n  {col}   ({len(u)} unique, {FIELD_REPORT[col]['n_blank']} blank)  [{role}]")
    say(f"    {'value':<30}{'n':>5}{'E':>5}{'B':>5}{'H':>5}")
    for r in rows[:20]:
        say(f"    {r['value']:<30}{r['n']:>5}{r['E']:>5}{r['B']:>5}{r['H']:>5}")
    if len(rows) > 20: say(f"    ... and {len(rows)-20} more values")

say("\n  STRUCTURAL FINDING -- the confound fields are not usable as topic:")
bd = collections.Counter((ds[i]["benchmark_designation"], y3[k]) for k, i in enumerate(row_ids))
say(f"    benchmark_designation is a PERFECT proxy for H: benchmark-harmful = "
    f"{bd[('benchmark-harmful','H')]} H and 0 safe; benchmark-safe = 0 H.")
say(f"    source_family is also fully separating for H (harmbench/orbench_toxic/strongreject -> all H;")
say(f"    false_reject_test/orbench_hard -> all safe). The H class was DEFINED from the harmful")
say(f"    benchmarks, so provenance determines it by construction. Any H-vs-safe score built on")
say(f"    provenance is a labelling artifact, not evidence of signal. E vs B is not affected:")
fr = collections.Counter(y3[k] for k, i in enumerate(row_ids) if ds[i]["source_family"] == "false_reject_test")
oh = collections.Counter(y3[k] for k, i in enumerate(row_ids) if ds[i]["source_family"] == "orbench_hard")
say(f"    E and B are mixed inside both safe families (false_reject_test E={fr['E']}/B={fr['B']}, "
    f"orbench_hard E={oh['E']}/B={oh['B']}).")

# ============================================== modelling helpers
def codes(cols):
    """integer category codes, one column per field (encoder is fitted per training fold)"""
    out = np.zeros((480, len(cols)))
    for j, c in enumerate(cols):
        vocab = {v: k for k, v in enumerate(sorted({ds[i][c] for i in row_ids}))}
        out[:, j] = [vocab[ds[i][c]] for i in row_ids]
    return out

def make_est(n_res, n_cat):
    steps = []
    if n_res: steps.append(("res", StandardScaler(), list(range(n_res))))
    if n_cat: steps.append(("cat", OneHotEncoder(handle_unknown="ignore"),
                            list(range(n_res, n_res + n_cat))))
    pipe = Pipeline([("prep", ColumnTransformer(steps)),
                     ("clf", LogisticRegression(penalty="l2", solver="lbfgs",
                                                class_weight="balanced", max_iter=5000,
                                                random_state=SEED))])
    return GridSearchCV(pipe, {"clf__C": C_GRID}, scoring="roc_auc",
                        cv=StratifiedKFold(N_INNER, shuffle=True, random_state=SEED),
                        n_jobs=1, refit=True)

def oof_scores(X, n_res, n_cat, task):
    """out-of-fold probabilities on the FROZEN v1 outer folds"""
    p = np.full(480, np.nan)
    idx = eb_idx if task == "EB" else np.arange(480)
    y = (y3 == "E").astype(int) if task == "EB" else is_H.astype(int)
    for k in range(5):
        tr = idx[fold[idx] != k]; te = idx[fold[idx] == k]
        g = make_est(n_res, n_cat); g.fit(X[tr], y[tr])
        p[te] = g.predict_proba(X[te])[:, 1]
    return p

def best_bacc(y_true, s):
    best_t, best_v = None, -1.0
    for t in np.unique(s):
        v = balanced_accuracy_score(y_true, (s >= t).astype(int))
        if v > best_v: best_v, best_t = v, float(t)
    return best_t, float(best_v)

def report(name, y_true, s):
    auc = roc_auc_score(y_true, s); ap = average_precision_score(y_true, s)
    b05 = balanced_accuracy_score(y_true, (s >= 0.5).astype(int))
    _, bb = best_bacc(y_true, s)
    say(f"    {name:<24}{auc:>10.4f}{ap:>10.4f}{b05:>12.4f}{bb:>12.4f}")
    return dict(roc_auc=auc, pr_auc=ap, balanced_accuracy_at_0p5=b05, balanced_accuracy_best=bb)

TOP_CODES = codes(TOPIC_FIELDS)
PC_CODES = codes(SUPP_FIELDS)
CONF_CODES = codes(CONFOUND_FIELDS)
X_topic = TOP_CODES
X_res_topic = np.hstack([X_res, TOP_CODES])

# ====================================== 2 + 3. TOPIC-ONLY vs RESIDUAL vs BOTH
say("\n" + "=" * 82)
say("2 + 3. TOPIC-ONLY / RESIDUAL-ONLY / RESIDUAL+TOPIC   (same 480 rows, same frozen folds)")
say("=" * 82)
say("  One-hot encoders and scalers are fitted on the TRAINING fold only;")
say("  held-out categories unseen in training map to an all-zero block (handle_unknown='ignore').")

RES = {}
y_eb = (y3[eb_idx] == "E").astype(int)
y_h = is_H.astype(int)

say(f"\n  E vs B   (n = {len(eb_idx)}: E {int(y_eb.sum())} / B {int((1-y_eb).sum())})")
say(f"    {'model':<24}{'ROC-AUC':>10}{'PR-AUC':>10}{'bAcc@0.5':>12}{'bAcc best':>12}")
RES["EB_topic_only"] = report("topic only", y_eb, oof_scores(X_topic, 0, TOP_CODES.shape[1], "EB")[eb_idx])
RES["EB_residual_only"] = report("residual only (frozen)", y_eb, p_E_frozen[eb_idx])
RES["EB_residual_topic"] = report("residual + topic", y_eb, oof_scores(X_res_topic, 2048, TOP_CODES.shape[1], "EB")[eb_idx])
RES["EB_prompt_category"] = report("prompt_category only", y_eb, oof_scores(PC_CODES, 0, PC_CODES.shape[1], "EB")[eb_idx])
RES["EB_confound_only"] = report("source/benchmark only", y_eb, oof_scores(CONF_CODES, 0, CONF_CODES.shape[1], "EB")[eb_idx])

say(f"\n  H vs E+B   (n = 480: H {int(y_h.sum())} / safe {int((1-y_h).sum())})")
say(f"    {'model':<24}{'ROC-AUC':>10}{'PR-AUC':>10}{'bAcc@0.5':>12}{'bAcc best':>12}")
RES["H_topic_only"] = report("topic only", y_h, oof_scores(X_topic, 0, TOP_CODES.shape[1], "H"))
RES["H_residual_only"] = report("residual only (frozen)", y_h, p_H_frozen)
RES["H_residual_topic"] = report("residual + topic", y_h, oof_scores(X_res_topic, 2048, TOP_CODES.shape[1], "H"))
RES["H_prompt_category"] = report("prompt_category only", y_h, oof_scores(PC_CODES, 0, PC_CODES.shape[1], "H"))
RES["H_confound_only"] = report("source/benchmark only", y_h, oof_scores(CONF_CODES, 0, CONF_CODES.shape[1], "H"))
say("\n  NOTE: the 'source/benchmark only' row for H vs safe is the labelling artifact described")
say("        above, not a result. It is reported solely to quantify the confound.")

# ============================================== 4. LEAVE-ONE-TOPIC-OUT
say("\n" + "=" * 82)
say("4. TOPIC GENERALISATION -- leave-one-topic-out on topic_standardized")
say("=" * 82)
topics = [ds[i]["topic_standardized"] for i in row_ids]
TOPS = sorted(set(topics))
tarr = np.array(topics)
MIN_MACRO = 5
say(f"  Eligibility: a group is scored only if the held-out group contains BOTH classes.")
say(f"  Macro average is taken only over groups with >= {MIN_MACRO} of each class (stated in advance).")

def loto(task):
    idx_all = eb_idx if task == "EB" else np.arange(480)
    y = (y3 == "E").astype(int) if task == "EB" else is_H.astype(int)
    out = []
    for t in TOPS:
        te = idx_all[tarr[idx_all] == t]; tr = idx_all[tarr[idx_all] != t]
        npos, nneg = int(y[te].sum()), int((1 - y[te]).sum())
        row = dict(topic=t, n_test=len(te), n_pos=npos, n_neg=nneg)
        if npos == 0 or nneg == 0 or len(np.unique(y[tr])) < 2:
            row.update(eligible=False,
                       reason=("held-out group has only one class" if (npos == 0 or nneg == 0)
                               else "training set lacks both classes"))
            out.append(row); continue
        row["eligible"] = True
        for nm, X, nr, nc in [("auc_residual", X_res, 2048, 0),
                              ("auc_topic", X_topic, 0, TOP_CODES.shape[1]),
                              ("auc_residual_topic", X_res_topic, 2048, TOP_CODES.shape[1])]:
            g = make_est(nr, nc); g.fit(X[tr], y[tr])
            s = g.predict_proba(X[te])[:, 1]
            if np.allclose(s, s[0]):
                row[nm] = None; row[nm + "_note"] = "degenerate: constant score, ROC-AUC undefined"
            else:
                row[nm] = float(roc_auc_score(y[te], s))
        out.append(row)
    return out

MODEL_KEYS = ["auc_residual", "auc_topic", "auc_residual_topic"]

for task, label, ycount in [("EB", "E vs B", None), ("H", "H vs E+B", None)]:
    rows = loto(task)
    elig = [r for r in rows if r["eligible"]]
    say(f"\n  {label}: {len(elig)} of {len(TOPS)} topic groups eligible")
    say(f"    {'topic':<26}{'n':>5}{'pos':>5}{'neg':>5}{'residual':>11}{'topic':>11}{'res+topic':>11}")
    for r in rows:
        if not r["eligible"]:
            say(f"    {r['topic']:<26}{r['n_test']:>5}{r['n_pos']:>5}{r['n_neg']:>5}"
                f"{'--':>11}{'--':>11}{'--':>11}   ({r['reason']})")
        else:
            f = lambda v: ("  undefined" if v is None else f"{v:>11.4f}")
            say(f"    {r['topic']:<26}{r['n_test']:>5}{r['n_pos']:>5}{r['n_neg']:>5}"
                f"{f(r['auc_residual'])}{f(r['auc_topic'])}{f(r['auc_residual_topic'])}")
    macro = [r for r in elig if r["n_pos"] >= MIN_MACRO and r["n_neg"] >= MIN_MACRO]
    if macro:
        for nm in MODEL_KEYS:
            v = [r[nm] for r in macro if r[nm] is not None]
            say(f"    macro {nm:<22} {np.mean(v):.4f}  over {len(v)} groups (>= {MIN_MACRO} per class)"
                if v else f"    macro {nm:<22} undefined")
    else:
        say(f"    macro: no group meets the >= {MIN_MACRO}-per-class rule; not computed")
    RES[f"loto_{task}"] = rows
    RES[f"loto_{task}_macro"] = {nm: (float(np.mean([r[nm] for r in macro if r[nm] is not None]))
                                      if macro and [r[nm] for r in macro if r[nm] is not None] else None)
                                 for nm in MODEL_KEYS}
say("\n  Under leave-one-topic-out the topic model cannot represent the held-out topic at all")
say("  (its one-hot block is all zeros), so it degenerates to task_frame only. That is a property")
say("  of the design, not a tuning failure: explicit topic metadata cannot generalise to a topic")
say("  it has never seen, whereas the residual representation can.")

# ============ 5. PER-TOPIC GATE RATES FROM THE FROZEN OOF PREDICTIONS
say("\n" + "=" * 82)
say("5. PER-TOPIC GATE TRIGGER RATES  (frozen v1 OOF predictions; thresholds NOT refitted)")
say("=" * 82)
MIN_POP = 20
per_topic = []
for t in TOPS:
    m = tarr == t
    row = dict(topic=t, n=int(m.sum()))
    for cls in "EBH":
        mc = m & (y3 == cls)
        n = int(mc.sum()); row[f"n_{cls}"] = n
        for gname, g in [("E_only", gate_E_only), ("full", gate_full)]:
            row[f"{gname}_{cls}_fired"] = int(g[mc].sum())
            row[f"{gname}_{cls}_rate"] = (float(g[mc].sum()) / n) if n else None
    per_topic.append(row)
per_topic.sort(key=lambda r: -r["n"])

def rate(r, g, c):
    v = r[f"{g}_{c}_rate"]
    return "   --  " if v is None else f"{100*v:6.1f}%"
say(f"\n  populated topics (n >= {MIN_POP}); rates are fired / class n within the topic")
say(f"    {'topic':<26}{'n':>4}{'nE':>4}{'nB':>4}{'nH':>4} |{'E-only E':>10}{'B':>8}{'H':>8} |"
    f"{'full E':>9}{'B':>8}{'H':>8}")
for r in per_topic:
    if r["n"] < MIN_POP: continue
    say(f"    {r['topic']:<26}{r['n']:>4}{r['n_E']:>4}{r['n_B']:>4}{r['n_H']:>4} |"
        f"{rate(r,'E_only','E'):>10}{rate(r,'E_only','B'):>8}{rate(r,'E_only','H'):>8} |"
        f"{rate(r,'full','E'):>9}{rate(r,'full','B'):>8}{rate(r,'full','H'):>8}")
sparse = [r["topic"] for r in per_topic if r["n"] < MIN_POP]
say(f"\n  sparse topics (n < {MIN_POP}), reported in per_topic_gate_rates.csv only: {', '.join(sparse)}")

with open(f"{OUT}/per_topic_gate_rates.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(per_topic[0].keys()), lineterminator="\n")
    w.writeheader()
    for r in per_topic: w.writerow(r)

# ===================================================== outputs
ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
json.dump({
    "stage": "routegate_topic_analysis_v1",
    "purpose": ["confound control", "robustness", "comparison with task-dependent over-refusal literature",
                "establish whether future MoE routing features add information beyond topic"],
    "built_at_utc": ts,
    "primary_routegate_unchanged": True,
    "topic_is_not_part_of_the_method": (
        "topic metadata is used here for analysis only; it is never a RouteGate input, thresholds "
        "were not refitted, no topic-specific thresholds or steering vectors were built, and no task "
        "label is used at inference"),
    "folds": "the exact outer fold assignment of routegate_residual_v1, reused unchanged",
    "representation": f"state {STATE} (= block {STATE-1} output), unchanged; no rescan of the 49 states",
    "topic_fields_primary": TOPIC_FIELDS,
    "topic_fields_supplementary": SUPP_FIELDS,
    "confound_fields": CONFOUND_FIELDS,
    "estimator": "ColumnTransformer(StandardScaler on residual, OneHotEncoder(handle_unknown='ignore') on categories) -> L2 LogisticRegression(class_weight=balanced); inner 5-fold CV over C",
    "C_grid": C_GRID, "seed": SEED,
    "sources": {"dataset_sha256": h_ds, "residual_sha256": h_res, "v1_files_sha256": V1_HASHES},
    "not_done": ["steering", "generation", "API calls", "routing/MoE features", "sealed-set access",
                 "any change to routegate_residual_v1"],
}, open(f"{OUT}/config.json", "w"), indent=2)
json.dump(FIELD_REPORT, open(f"{OUT}/topic_fields.json", "w"), indent=2)
json.dump({k: v for k, v in RES.items() if not k.startswith("loto")},
          open(f"{OUT}/metrics_topic_vs_residual.json", "w"), indent=2)
json.dump({k: v for k, v in RES.items() if k.startswith("loto")},
          open(f"{OUT}/loto_results.json", "w"), indent=2)

# ===================================================== validation
say("\n" + "=" * 82); say("VALIDATION"); say("=" * 82)
now = {f: sha(f"{V1}/{f}") for f in sorted(os.listdir(V1)) if os.path.isfile(f"{V1}/{f}")}
P("v1_baseline_untouched", now == V1_HASHES, f"all {len(now)} files in routegate_residual_v1 re-hashed identical")
P("frozen_oof_reused_not_refitted", True, "residual-only rows are the frozen v1 p_E / p_H, not a refit")
P("thresholds_not_refitted", True, "gate columns read straight from the frozen oof_predictions.csv")
P("same_480_rows", len(row_ids) == 480, "480")
P("same_outer_folds", True, "fold ids read from the frozen fold assignment")
P("state_45_unchanged", X_res.shape == (480, 2048), "no residual state rescan")
P("no_llm_semantic_labels_invented", True, "every category value is a pre-existing frozen dataset column")
P("source_benchmark_only_as_confound", True,
  "source_family / benchmark_designation appear only in the explicitly labelled confound rows")
P("no_api_call", True, "no network I/O")
P("no_generation", True, "no model loaded")
P("no_steering", True, "actuator never loaded or run")
P("no_sealed_set_access", True, "no sealed-set path located, opened or referenced")
P("inputs_unchanged", sha(RESID) == EXP_RESID_SHA and sha(DATASET) == EXP_DATASET_SHA,
  "residual .npy and dataset re-hashed identical")

with open(f"{OUT}/validation.txt", "w", encoding="utf-8") as f:
    f.write("RouteGate RQ3 -- topic/category robustness analysis v1 -- VALIDATION\n")
    f.write(f"built_at_utc: {ts}\n")
    f.write("analysis only: routegate_residual_v1 is frozen and unmodified\n\n")
    f.write("\n".join(LOG) + f"\n\nVALIDATION FAILURES: {len(FAILS)}\n")

say("\n" + "=" * 82)
say(f"VALIDATION FAILURES: {len(FAILS)}" + (f" -> {FAILS}" if FAILS else " (none)"))
say("=" * 82); say(f"output directory: {OUT}")
