#!/usr/bin/env python3
"""Out-of-fold training of the single-direction coefficient controller.

Five controllers, one per outer fold. Each trains on the four other folds,
early-stops on an inner split of those four, and predicts ``g_R`` only for the
fold it never saw. No final all-development refit is performed here.

    g_R(x)     = sigmoid(f_theta(h_state32))
    delta_R,l  = 0.15 * g_R(x) * r_l(x) * d_R,l     l = historical 31, 32

Cached activations only: **zero language-model forward passes**. The edit lands
on state 32 and nothing upstream of it is edited, so the steered state-32
activation is exact from the frozen capture. Nothing is generated and no judge
is called.

PROVISIONAL. Historical blocks 31-32 have not completed behavioural
adjudication, so every coefficient here is conditional on an actuator whose
causal effect is unmeasured. No operating point or deployment claim follows.

Usage
-----
    python experiments/single_direction_controller/train_gate_controller.py \\
        --direction-bank .../validated_direction_bank \\
        --output-dir outputs/single_direction_controller/gate_oof_v1 --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate.data import load_input_manifest, load_local_paths     # noqa: E402
from routegate.dualsteer import load_validated_bank                  # noqa: E402
from routegate.ebh import load_ebh_frame                             # noqa: E402
from routegate.frozenprobe import (                                  # noqa: E402
    PROBE_ARTIFACTS,
    PROBE_PREDICTIONS,
    PROBE_RUN_DIRECTORY,
    REPRODUCTION_TOLERANCE,
    TorchProbe,
    assert_repair_direction_reduces_the_repair_logit,
    read_saved_predictions,
    reconstruct_and_verify,
)
from routegate.gatecontroller import (                               # noqa: E402
    ALPHA_R,
    CONTROLLER_PARAMETERS,
    FORBIDDEN_FEATURES,
    MAX_STEPS,
    N_SPLITS,
    PATIENCE,
    PROVISIONAL_REASON,
    PROVISIONAL_STATUS,
    READ_STATE,
    SCHEMA,
    TRAINING_SPEC_NAME,
    GateControllerError,
    STRATIFICATION_FIELDS,
    apply_counterfactual_clamp,
    apply_safety_veto,
    assert_covariates_are_not_features,
    assert_optimiser_is_controller_only,
    build_controller,
    build_training_specification,
    controller_optimiser,
    count_parameters,
    gate,
    inner_fold_for,
    protected_mask,
    repair_loss,
    safety_loss,
    seed_for,
    size_loss,
    steer,
    stratified_distribution,
    total_loss,
    veto_threshold,
)
from routegate.probeheads import (                                   # noqa: E402
    REPAIR_HEAD,
    SAFETY_HEAD,
    load_bank_fold_assignment,
    state_matrix,
)
from routegate.runrecord import (                                    # noqa: E402
    environment_record,
    utc_now,
    write_json,
)
from routegate.twosite import CANDIDATE_BLOCKS                       # noqa: E402
from routegate.onepassbudget import block_input_state                # noqa: E402
from routegate.validation import (                                   # noqa: E402
    Checklist,
    ValidationError,
    assert_config_is_sealed_free,
    assert_not_sealed,
    looks_sealed,
    sha256_file,
)

PACKAGE_VERSION = "0.1.0"
EXPECTED_ROWS = 480
FOLD_DIR = ("outputs/causal_repair/orthogonalized_population_scan/"
            "direction_bank")
CUTOFFS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90)


def claim_output_dir(path: Path | str, create: bool) -> Path:
    path = Path(path)
    assert_not_sealed(path, "controller training output directory")
    if path.exists():
        if not path.is_dir():
            raise GateControllerError(f"{path} exists and is not a directory")
        if sorted(p.name for p in path.iterdir()):
            raise GateControllerError(
                f"{path} already holds results. A training run is written "
                f"once, into an empty directory; a previous one is never "
                f"partially overwritten.")
    elif create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paths", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="validate every input, hash, probe and gate; "
                             "train nothing and write nothing")
    return parser


def _fold_history(net, optimiser, tensors, masks, inner_mask, train_mask,
                  tau_hi, torch) -> Dict[str, Any]:
    """Train one controller, early-stopping on the inner split only."""
    best = {"loss": float("inf"), "step": 0, "state": None}
    history: List[Dict[str, float]] = []
    for step in range(1, MAX_STEPS + 1):
        net.train()
        optimiser.zero_grad(set_to_none=True)
        g = gate(net, tensors["x"][train_mask])
        steered = steer(tensors["h"][train_mask], g,
                        tensors["budget"][train_mask], tensors["direction"])
        z_r = tensors["repair"].logit(steered)
        z_h_steered = tensors["safety"].logit(steered)
        loss_r = repair_loss(z_r, masks["is_e"][train_mask])
        loss_s = safety_loss(tensors["z_h"][train_mask], z_h_steered,
                             masks["protected"][train_mask])
        loss_g = size_loss(g)
        loss = total_loss(loss_r, loss_s, loss_g)
        loss.backward()
        optimiser.step()

        net.eval()
        with torch.no_grad():
            gv = gate(net, tensors["x"][inner_mask])
            sv = steer(tensors["h"][inner_mask], gv,
                       tensors["budget"][inner_mask], tensors["direction"])
            vr = repair_loss(tensors["repair"].logit(sv),
                             masks["is_e"][inner_mask])
            vs = safety_loss(tensors["z_h"][inner_mask],
                             tensors["safety"].logit(sv),
                             masks["protected"][inner_mask])
            vg = size_loss(gv)
            validation = float(total_loss(vr, vs, vg))
        history.append({"step": step, "train_total": float(loss),
                        "train_repair": float(loss_r),
                        "train_safety": float(loss_s),
                        "train_size": float(loss_g),
                        "inner_total": validation,
                        "inner_repair": float(vr),
                        "inner_safety": float(vs), "inner_size": float(vg)})
        if validation < best["loss"] - 1e-9:
            best = {"loss": validation, "step": step,
                    "state": {k: v.detach().clone()
                              for k, v in net.state_dict().items()}}
        elif step - best["step"] >= PATIENCE:
            break
    if best["state"] is not None:
        net.load_state_dict(best["state"])
    return {"history": history, "selected_step": best["step"],
            "inner_best_loss": best["loss"], "steps_run": len(history)}


def run(args: argparse.Namespace) -> int:
    checks = Checklist("single_direction_controller_oof")
    spec = build_training_specification()
    checks.record("status", True, f"{spec['status']}: {PROVISIONAL_REASON[:60]}…")
    checks.require("no_final_refit",
                   spec["oof_protocol"]["final_refit_performed"] is False,
                   "the all-development refit is not performed at this stage")
    checks.require("h_minus_e_is_never_applied",
                   spec["intervention"]["h_minus_e_is_never_applied"] is True
                   and spec["intervention"]["alpha_S"] == 0.0,
                   "d_S = H - E is retired; alpha_S is 0 permanently")

    local_paths = (load_local_paths(args.paths) if args.paths
                   else load_local_paths())
    assert_config_is_sealed_free(local_paths, "local path file")
    claim_output_dir(args.output_dir, create=False)

    # ---- the frozen probe run, by digest ---------------------------------
    probe_dir = REPO_ROOT / PROBE_RUN_DIRECTORY
    for name, expected in sorted(PROBE_ARTIFACTS.items()):
        actual = sha256_file(probe_dir / name)
        checks.require(f"probe_artifact_{name}", actual == expected,
                       f"{name} hashes to {actual}, pinned {expected}")

    # ---- corpus, folds, activations --------------------------------------
    frame = load_ebh_frame()
    checks.require("corpus_rows", frame.n == EXPECTED_ROWS,
                   f"{frame.n} row(s), expected {EXPECTED_ROWS}")
    sealed = [r for r in frame.row_ids if looks_sealed(r)]
    checks.require("no_sealed_row", not sealed,
                   f"{len(sealed)} row id(s) read as held-out material")
    assignment, source = load_bank_fold_assignment(REPO_ROOT / FOLD_DIR)
    checks.require("n_splits", assignment.n_splits == N_SPLITS,
                   f"{assignment.n_splits} folds, expected {N_SPLITS}")
    checks.record("fold_assignment_sha256", True,
                  source.fold_assignment_sha256)

    # ---- reconstruct the probes and prove they reproduce ------------------
    saved = read_saved_predictions(probe_dir / PROBE_PREDICTIONS, READ_STATE)
    probes = reconstruct_and_verify(frame, assignment, saved, READ_STATE,
                                    REPRODUCTION_TOLERANCE, checks)

    # ---- the direction, by digest ----------------------------------------
    from routegate.twosite import build_candidate_record
    expected_bank = build_candidate_record()["direction"]["artifact_sha256"]
    bank = load_validated_bank(assert_not_sealed(Path(args.direction_bank)),
                               expected_bank, checks)
    d_repair, _ = bank.directions_for_state(READ_STATE)
    d_repair = np.asarray(d_repair, dtype=np.float64)
    kappa = assert_repair_direction_reduces_the_repair_logit(
        probes, d_repair, checks)

    design = state_matrix(frame, READ_STATE)
    budget_state = block_input_state(min(CANDIDATE_BLOCKS))
    budgets = np.linalg.norm(
        np.asarray(frame.activations[:, budget_state, :], dtype=np.float64),
        axis=-1)
    checks.record("budget_reference_state", True,
                  f"state {budget_state} (input of historical block "
                  f"{min(CANDIDATE_BLOCKS)}), from the unsteered capture")

    print(f"[gate] {spec['status']}")
    print(f"[gate] read state {READ_STATE} (historical block "
          f"{spec['controller']['read_historical_block']}), final prompt token")
    print(f"[gate] blocks {list(CANDIDATE_BLOCKS)}  alpha_R={ALPHA_R} "
          f"alpha_S=0  parameters={CONTROLLER_PARAMETERS}")
    print(f"[gate] probes reproduced to {REPRODUCTION_TOLERANCE:g}")
    print(f"[gate] kappa_R {[round(v, 4) for v in kappa[REPAIR_HEAD].values()]}")
    print(f"[gate] kappa_H {[round(v, 4) for v in kappa[SAFETY_HEAD].values()]}"
          f"  (diagnostic, not a gate)")
    print(f"[gate] forward passes: 0 (cached activations only)")

    if args.dry_run:
        print(f"[gate] dry run: nothing trained, nothing written")
        print(f"[gate] {checks.as_dict()['checks_run']} check(s), all passed: "
              f"{checks.all_passed}")
        return 0 if checks.all_passed else 2

    import torch
    torch.use_deterministic_algorithms(True)

    output_dir = claim_output_dir(args.output_dir, create=True)
    labels = np.array(frame.labels)
    fold_of = np.array([assignment.fold_of(r) for r in frame.row_ids])
    coefficients = np.full(frame.n, np.nan)
    vetoed = np.zeros(frame.n, dtype=bool)
    clamped = np.full(frame.n, np.nan)
    z_r_before = np.full(frame.n, np.nan)
    z_r_after = np.full(frame.n, np.nan)
    z_h_before = np.full(frame.n, np.nan)
    z_h_after = np.full(frame.n, np.nan)
    per_fold: List[Dict[str, Any]] = []

    for k in range(N_SPLITS):
        torch.manual_seed(seed_for(k))
        inner = inner_fold_for(k)
        held = fold_of == k
        train = ~held
        inner_mask_full = fold_of == inner
        inner_train = train & ~inner_mask_full

        repair_probe = TorchProbe(probes[REPAIR_HEAD][k])
        safety_probe = TorchProbe(probes[SAFETY_HEAD][k])
        mu = design[inner_train].mean(axis=0)
        sd = np.maximum(design[inner_train].std(axis=0), 1e-8)

        x = torch.as_tensor((design - mu) / sd, dtype=torch.float32)
        if k == 0:
            assert_covariates_are_not_features(x, checks=checks)
        h = torch.as_tensor(design, dtype=torch.float32)
        b = torch.as_tensor(budgets, dtype=torch.float32)
        d = torch.as_tensor(d_repair, dtype=torch.float32)
        for tensor in (x, h, b, d):
            tensor.requires_grad_(False)
        with torch.no_grad():
            z_h_unsteered = safety_probe.logit(h)
            z_r_unsteered = repair_probe.logit(h)

        is_h = torch.as_tensor(labels == "H")
        is_e = torch.as_tensor(labels == "E")
        tau_hi = veto_threshold(
            z_h_unsteered[torch.as_tensor(train) & is_h].tolist())
        protected = protected_mask(z_h_unsteered, is_h, tau_hi)

        tensors = {"x": x, "h": h, "budget": b, "direction": d,
                   "repair": repair_probe, "safety": safety_probe,
                   "z_h": z_h_unsteered}
        masks = {"is_e": is_e, "protected": protected}

        net = build_controller(seed_for(k))
        optimiser = controller_optimiser(net)
        assert_optimiser_is_controller_only(
            optimiser, net, [x, h, b, d, repair_probe.weight,
                             safety_probe.weight], checks)

        inner_outcome = _fold_history(
            net, optimiser, tensors, masks,
            torch.as_tensor(inner_mask_full), torch.as_tensor(inner_train),
            tau_hi, torch)

        # retrain on ALL outer-training rows for the selected step count
        net = build_controller(seed_for(k))
        optimiser = controller_optimiser(net)
        train_t = torch.as_tensor(train)
        for _ in range(max(1, inner_outcome["selected_step"])):
            net.train()
            optimiser.zero_grad(set_to_none=True)
            g = gate(net, x[train_t])
            steered = steer(h[train_t], g, b[train_t], d)
            loss = total_loss(
                repair_loss(repair_probe.logit(steered), is_e[train_t]),
                safety_loss(z_h_unsteered[train_t],
                            safety_probe.logit(steered), protected[train_t]),
                size_loss(g))
            loss.backward()
            optimiser.step()

        net.eval()
        held_t = torch.as_tensor(held)
        with torch.no_grad():
            g_held = gate(net, x[held_t])
            steered = steer(h[held_t], g_held, b[held_t], d)
            zr_after = repair_probe.logit(steered)
            zh_after = safety_probe.logit(steered)
            veto = apply_safety_veto(g_held, z_h_unsteered[held_t], tau_hi)
            clamp = apply_counterfactual_clamp(
                g_held, z_h_unsteered[held_t], b[held_t],
                float(kappa[SAFETY_HEAD][k]), tau_hi)
            # in-sample optimism diagnostic
            g_train = gate(net, x[train_t])

        idx = np.flatnonzero(held)
        coefficients[idx] = g_held.numpy()
        vetoed[idx] = (veto.numpy() == 0.0) & (g_held.numpy() > 0.0)
        clamped[idx] = clamp.numpy()
        z_r_before[idx] = z_r_unsteered[held_t].numpy()
        z_r_after[idx] = zr_after.numpy()
        z_h_before[idx] = z_h_unsteered[held_t].numpy()
        z_h_after[idx] = zh_after.numpy()

        per_fold.append({
            "fold": k, "inner_fold": inner,
            "n_train": int(train.sum()), "n_inner_train": int(inner_train.sum()),
            "n_inner_validation": int(inner_mask_full.sum()),
            "n_held_out": int(held.sum()),
            "selected_step": inner_outcome["selected_step"],
            "steps_run": inner_outcome["steps_run"],
            "inner_best_loss": inner_outcome["inner_best_loss"],
            "tau_hi": float(tau_hi),
            "kappa_R": float(kappa[REPAIR_HEAD][k]),
            "kappa_H": float(kappa[SAFETY_HEAD][k]),
            "seed": seed_for(k),
            "held_out_mean_g": float(g_held.mean()),
            "in_sample_mean_g": float(g_train.mean()),
            "optimism_mean_g": float(g_train.mean() - g_held.mean()),
            "held_out_vetoed": int(vetoed[idx].sum()),
            "history": inner_outcome["history"],
        })
        print(f"[gate] fold {k}: inner {inner}, step "
              f"{inner_outcome['selected_step']}/{inner_outcome['steps_run']}, "
              f"mean g_R {float(g_held.mean()):.4f}, "
              f"vetoed {int(vetoed[idx].sum())}/{int(held.sum())}", flush=True)

    return _report(output_dir, frame, labels, fold_of, coefficients, vetoed,
                   clamped, z_r_before, z_r_after, z_h_before, z_h_after,
                   per_fold, spec, kappa, source, checks, bank, expected_bank,
                   list(frame.topic), list(frame.task_frame))


def _distribution(values: np.ndarray) -> Dict[str, float]:
    clean = values[~np.isnan(values)]
    return {"n": int(clean.size), "mean": float(clean.mean()),
            "std": float(clean.std()), "min": float(clean.min()),
            "p25": float(np.percentile(clean, 25)),
            "median": float(np.median(clean)),
            "p75": float(np.percentile(clean, 75)),
            "max": float(clean.max())}


def _report(output_dir, frame, labels, fold_of, coefficients, vetoed, clamped,
            z_r_before, z_r_after, z_h_before, z_h_after, per_fold, spec,
            kappa, source, checks, bank, expected_bank, topic,
            task_frame) -> int:
    checks.require("every_row_predicted_once",
                   not np.isnan(coefficients).any(),
                   f"{int(np.isnan(coefficients).sum())} row(s) have no "
                   f"out-of-fold coefficient")
    checks.require("coefficients_in_range",
                   bool((coefficients >= 0.0).all()
                        and (coefficients <= 1.0).all()),
                   "every g_R lies in [0, 1]")
    checks.require("alpha_never_exceeds_the_frozen_strength",
                   float((ALPHA_R * coefficients).max()) <= ALPHA_R + 1e-12,
                   f"max applied alpha_R "
                   f"{float((ALPHA_R * coefficients).max()):.6f}")
    checks.require("the_clamp_never_raises_a_gate",
                   bool((clamped <= coefficients + 1e-9).all()),
                   "the counterfactual clamp raised a gate")

    rows = []
    for i, rid in enumerate(frame.row_ids):
        rows.append({
            "row_id": rid, "final_ebh": labels[i], "fold": int(fold_of[i]),
            # diagnostic strata: frozen labels, never controller features
            "topic_standardized": topic[i], "task_frame": task_frame[i],
            # enough for a later hard-layer-router run to report arm+coefficient
            "arm_id": "repair_015_blocks31_32",
            "historical_blocks": "31,32", "package_states": "32,33",
            "g_R": float(coefficients[i]),
            "alpha_R_applied": float(ALPHA_R * coefficients[i]),
            "g_R_after_veto": 0.0 if vetoed[i] else float(coefficients[i]),
            "g_R_after_clamp": float(clamped[i]),
            "vetoed": bool(vetoed[i]),
            "z_repair_before": float(z_r_before[i]),
            "z_repair_after": float(z_r_after[i]),
            "z_repair_delta": float(z_r_after[i] - z_r_before[i]),
            "z_safety_before": float(z_h_before[i]),
            "z_safety_after": float(z_h_after[i]),
            "z_safety_delta": float(z_h_after[i] - z_h_before[i]),
        })
    fields = list(rows[0])
    with open(Path(output_dir) / "oof_coefficients.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_class = {c: _distribution(coefficients[labels == c])
                for c in ("E", "B", "H")}
    intervention_rates = {
        f"{c:.2f}": {klass: float((coefficients[labels == klass] >= c).mean())
                     for klass in ("E", "B", "H")}
        for c in CUTOFFS}
    by_topic = stratified_distribution(coefficients, topic, CUTOFFS)
    by_task_frame = stratified_distribution(coefficients, task_frame, CUTOFFS)
    probe_shift = {
        klass: {"repair_logit_delta": _distribution(
                    (z_r_after - z_r_before)[labels == klass]),
                "safety_logit_delta": _distribution(
                    (z_h_after - z_h_before)[labels == klass])}
        for klass in ("E", "B", "H")}

    summary = {
        "schema": SCHEMA, "name": TRAINING_SPEC_NAME,
        "status": PROVISIONAL_STATUS, "status_reason": PROVISIONAL_REASON,
        "created_utc": utc_now(), "rows": int(frame.n),
        "forward_passes": 0, "generated_responses": 0, "judge_calls": 0,
        "final_refit_performed": False,
        "coefficient_distribution_by_class": by_class,
        "coefficient_distribution_overall": _distribution(coefficients),
        "intervention_rate_at_cutoff": intervention_rates,
        "coefficient_distribution_by_topic": by_topic,
        "coefficient_distribution_by_task_frame": by_task_frame,
        "stratification": spec["stratification"],
        "probe_logit_shift_by_class": probe_shift,
        "veto": {"vetoed_rows": int(vetoed.sum()),
                 "by_class": {c: int(vetoed[labels == c].sum())
                              for c in ("E", "B", "H")},
                 "rule": spec["brake"]["veto_rule"]},
        "clamp": {"rows_reduced": int((clamped < coefficients - 1e-9).sum()),
                  "never_raises_a_gate": True},
        "optimism": {
            "per_fold": [{"fold": f["fold"],
                          "in_sample_mean_g": f["in_sample_mean_g"],
                          "held_out_mean_g": f["held_out_mean_g"],
                          "optimism": f["optimism_mean_g"]}
                         for f in per_fold],
            "mean_optimism": float(np.mean([f["optimism_mean_g"]
                                            for f in per_fold])),
            "note": "in-sample mean gate minus held-out mean gate, per fold. "
                    "The probe is in-sample on the controller's training rows, "
                    "so this is reported rather than assumed away.",
        },
        "kappa": {"repair": {str(k): v for k, v in kappa[REPAIR_HEAD].items()},
                  "safety": {str(k): v for k, v in kappa[SAFETY_HEAD].items()}},
        "per_fold": [{k: v for k, v in f.items() if k != "history"}
                     for f in per_fold],
    }
    write_json(Path(output_dir) / "summary.json", summary)
    write_json(Path(output_dir) / "training_histories.json",
               {"per_fold": [{"fold": f["fold"], "history": f["history"]}
                             for f in per_fold]})
    write_json(Path(output_dir) / "training_specification.json", spec)
    write_json(Path(output_dir) / "fold_assignments.json",
               {"n_splits": N_SPLITS,
                "fold_assignment_sha256": source.fold_assignment_sha256,
                "fold_of_row": {rid: int(fold_of[i])
                                for i, rid in enumerate(frame.row_ids)}})
    write_json(Path(output_dir) / "validation.json", checks.as_dict())

    manifest = {
        "schema": SCHEMA, "status": PROVISIONAL_STATUS,
        "status_reason": PROVISIONAL_REASON,
        "created_utc": utc_now(), "package_version": PACKAGE_VERSION,
        "specification": TRAINING_SPEC_NAME,
        "controller_parameters": CONTROLLER_PARAMETERS,
        "read_state": READ_STATE, "alpha_R": ALPHA_R, "alpha_S": 0.0,
        "historical_blocks": list(CANDIDATE_BLOCKS),
        "direction_bank_sha256": expected_bank,
        "probe_artifacts": dict(PROBE_ARTIFACTS),
        "fold_assignment_sha256": source.fold_assignment_sha256,
        "forward_passes": 0, "loads_a_model": False,
        "generates_responses": False, "calls_a_judge": False,
        "reads_the_sealed_set": False, "final_refit_performed": False,
        "environment": environment_record(PACKAGE_VERSION),
        "checklist": checks.as_dict(),
    }
    write_json(Path(output_dir) / "run_manifest.json", manifest)
    digests = {name: sha256_file(Path(output_dir) / name)
               for name in sorted(p.name for p in Path(output_dir).iterdir())}
    write_json(Path(output_dir) / "output_hashes.json", digests)

    print(f"\n[gate] {'class':<6} {'n':>4} {'mean g':>8} {'median':>8} "
          f"{'max':>8}")
    for c in ("E", "B", "H"):
        d = by_class[c]
        print(f"[gate] {c:<6} {d['n']:>4} {d['mean']:>8.4f} "
              f"{d['median']:>8.4f} {d['max']:>8.4f}")
    print(f"[gate] vetoed {int(vetoed.sum())} row(s); clamp reduced "
          f"{summary['clamp']['rows_reduced']}")
    print(f"[gate] wrote {output_dir}")
    for name, digest in sorted(digests.items()):
        print(f"[gate]   {name:<30} {digest}")
    print(f"[gate] {PROVISIONAL_STATUS}")
    return 0 if checks.all_passed else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
