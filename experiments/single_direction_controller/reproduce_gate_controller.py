#!/usr/bin/env python3
"""Reproduce gate_oof_v1 deterministically and persist a deployable controller.

    python experiments/single_direction_controller/\
        reproduce_gate_controller.py --direction-bank <bank> \
        --output-dir outputs/single_direction_controller/gate_oof_v1_repro \
        --dry-run

WHY
---
``gate_oof_v1`` produced 480 out-of-fold coefficients and **no weights**. Its
manifest says ``final_refit_performed: false`` and its ``output_hashes.json``
lists seven files, none of them a checkpoint. A column of per-row coefficients
is not a controller: applying it at generation time would be a row-id lookup,
which cannot score a prompt the table has never seen, and which would leak the
whole development set into every arm that used it.

This stage re-runs the *same* training and keeps what the first run threw away.

WHAT IT DOES NOT DO
-------------------
It does not tune anything. Architecture, loss, regularisation, optimiser,
seeds, folds, thresholds and the early-stopping rule are imported from
``routegate.gatecontroller`` and the committed
``training_specification.json``; this file chooses none of them.

It does not mutate ``gate_oof_v1``. It reads that run's outputs to compare
against and writes somewhere new; a run directory that already holds files is
refused.

The one thing v1 did not fix is the **refit step count**, because v1 performed
no refit. That is declared here rather than tuned: the median of the five
fold-selected steps. A median over five already-selected integers introduces no
search and cannot be moved by looking at an outcome.

DETERMINISM
-----------
BLAS threading changes reduction order, and the probe reconstruction is gated
at 1e-12. On a 36-core node the unpinned reduction drifts to ~1.6e-12 and the
gate fails -- correctly. This stage pins the thread count before importing
numpy or torch and records what it pinned, so the gate means the same thing on
every machine.

EQUIVALENCE
-----------
"The same computation" is decided by ``routegate.reproequivalence``, a
versioned, dtype-aware rule: a float32 budget on the coefficients, exact
agreement on folds, labels, early-stopping steps, veto thresholds and probe
digests, no row moving across any reporting, veto, clamp or intervention
boundary, no rank inversion the tolerance can see, and an applied-edit
difference far inside one bfloat16 rounding step of the state it is written
into.

It replaces this file's own first attempt, ``COEFFICIENT_TOLERANCE = 1e-9``,
which refused a reproduction that had matched every fold and every stopping
step because it landed 2.384186e-07 -- two float32 epsilons -- away. That
constant came from this runner and from no frozen protocol; the superseded
rule, and what it did, are recorded in ``reproequivalence.SUPERSEDED_RULES``
and in ``docs/decision_log.md``. ``--tolerance`` may tighten the budget and is
refused if it would loosen it.

FAIL-CLOSED, AND INSPECTABLE
----------------------------
The comparison is written to ``reproduction_diagnostic.json`` **before** the
exit status is decided, so a refused run leaves the evidence it was refused
on. Checkpoints and the deployment refit are produced only after the rule
passes: a run that is not equivalent never yields a controller.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

#: Pinned BEFORE numpy or torch is imported -- these are read at load time.
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS")
PINNED_THREADS = "1"
for _name in THREAD_VARIABLES:
    os.environ.setdefault(_name, PINNED_THREADS)

import numpy as np                                                # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import importlib.util                                             # noqa: E402

from routegate.gatecontroller import (                            # noqa: E402
    ALPHA_R,
    N_SPLITS,
    apply_counterfactual_clamp,
    build_controller,
    controller_optimiser,
    gate,
    inner_fold_for,
    protected_mask,
    repair_loss,
    safety_loss,
    seed_for,
    size_loss,
    steer,
    total_loss,
    veto_threshold,
)
from routegate.livecontroller import (                            # noqa: E402
    KIND_FOLD,
    KIND_REFIT,
    build_checkpoint,
    gate_from_activation,
)
from routegate.reproequivalence import (                          # noqa: E402
    DIAGNOSTIC_FILENAME,
    GATE_ABS_TOLERANCE,
    RULE_VERSION,
    SUPERSEDED_RULES,
    assert_rule_is_admissible,
    assert_tolerance_is_not_loosened,
    decision_margins,
    describe_tolerance,
    evaluate as evaluate_equivalence,
    write_diagnostic,
)
from routegate.runrecord import environment_record, utc_now       # noqa: E402
from routegate.validation import (                                # noqa: E402
    Checklist,
    ValidationError,
    sha256_file,
)
from routegate import __version__ as PACKAGE_VERSION              # noqa: E402

TRAINER = (REPO_ROOT / "experiments" / "single_direction_controller"
           / "train_gate_controller.py")

#: The run this stage reproduces, and the digests it must match.
SOURCE_RUN = "gate_oof_v1"
SOURCE_ARTIFACTS = REPO_ROOT / "artifacts" / SOURCE_RUN

#: Declared by ``routegate.reproequivalence``, in float32 epsilons, and
#: admissibility-checked against this corpus's own decision margins. It is not
#: read off any observed maximum. See that module for the derivation and for
#: the superseded 1e-9 attempt.
COEFFICIENT_TOLERANCE = GATE_ABS_TOLERANCE

REPRODUCTION_SCHEMA = "routegate/gate_controller_reproduction/2"


class ReproductionError(ValidationError):
    """The reproduction does not agree with the run it reproduces."""


def load_trainer():
    """Import the frozen trainer. Its code is reused, not copied."""
    spec = importlib.util.spec_from_file_location("_gate_trainer", TRAINER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)                              # type: ignore
    return module


def read_saved_coefficients(path: Path) -> Dict[str, Dict[str, Any]]:
    """``row_id -> {fold, g_R, vetoed}`` from the source CSV."""
    import csv

    if not path.is_file():
        raise ReproductionError(
            f"{path} is absent. The reproduction compares against "
            f"{SOURCE_RUN}'s own coefficients; without them there is nothing "
            f"to agree with.")
    out: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            out[str(record["row_id"])] = {
                "fold": int(record["fold"]),
                "g_R": float(record["g_R"]),
                "g_R_after_veto": float(record["g_R_after_veto"]),
                "g_R_after_clamp": float(record["g_R_after_clamp"]),
                "vetoed": str(record["vetoed"]).lower() in ("true", "1"),
                "final_ebh": str(record["final_ebh"]),
                "z_safety_before": float(record["z_safety_before"]),
            }
    return out


def refit_step_count(selected: Sequence[int]) -> int:
    """The declared refit rule: the median of the five fold-selected steps.

    Not tuned and not tunable. It is a function of five integers that early
    stopping already chose on inner splits, so no outcome can move it.
    """
    return int(np.median(np.asarray(list(selected), dtype=np.int64)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, default=None,
                        help="gate_oof_v1's output directory")
    parser.add_argument("--paths", type=Path, default=None)
    parser.add_argument(
        "--tolerance", type=float, default=COEFFICIENT_TOLERANCE,
        help="tighten the float32 coefficient budget. A value looser than "
             "the declared rule is refused: the gate is corrected in "
             "reproequivalence, never argued away on a command line.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    trainer = load_trainer()
    checks = Checklist("gate_controller_reproduction")

    for name in THREAD_VARIABLES:
        checks.record(f"pinned_{name}", os.environ.get(name) == PINNED_THREADS,
                      f"{name}={os.environ.get(name)}")
    checks.require(
        "threads_are_pinned",
        all(os.environ.get(n) == PINNED_THREADS for n in THREAD_VARIABLES),
        "BLAS threading changes reduction order, and the probe gate is 1e-12; "
        "unpinned, this drifts and the gate fails")

    spec = trainer.build_training_specification()
    checks.require("source_spec_unchanged",
                   spec["oof_protocol"]["final_refit_performed"] is False,
                   "v1 performed no refit; this stage adds one without "
                   "editing v1's specification")
    checks.require("h_minus_e_is_never_applied",
                   spec["intervention"]["h_minus_e_is_never_applied"] is True
                   and spec["intervention"]["alpha_S"] == 0.0,
                   "d_S = H - E is retired; alpha_S is 0 permanently")

    # ---- the source run's own outputs, by digest -------------------------
    source_dir = args.source_run
    if source_dir is None:
        raise ReproductionError(
            "--source-run must name gate_oof_v1's output directory; the "
            "reproduction is defined by what it reproduces")
    import json
    pinned = json.loads(
        (SOURCE_ARTIFACTS / "output_hashes.json").read_text(encoding="utf-8"))
    for name, want in sorted(pinned.items()):
        path = Path(source_dir) / name
        got = sha256_file(path) if path.is_file() else None
        checks.require(f"source_{name}", got == want,
                       f"{name} hashes {str(got)[:12]}, pinned {want[:12]}")

    saved = read_saved_coefficients(Path(source_dir) / "oof_coefficients.csv")
    checks.require("saved_rows", len(saved) == trainer.EXPECTED_ROWS,
                   f"{len(saved)} saved coefficient(s), expected "
                   f"{trainer.EXPECTED_ROWS}")

    source_summary = json.loads(
        (Path(source_dir) / "summary.json").read_text(encoding="utf-8"))
    saved_steps = {int(f["fold"]): int(f["selected_step"])
                   for f in source_summary["per_fold"]}
    saved_tau = {int(f["fold"]): float(f["tau_hi"])
                 for f in source_summary["per_fold"]}
    assert_tolerance_is_not_loosened(args.tolerance, checks)
    checks.record(
        "superseded_equivalence_rules", True,
        "; ".join(f"{r['rule']} -> {r['outcome']}" for r in SUPERSEDED_RULES))

    # ---- everything the trainer needs, loaded exactly as it loads it -----
    prepared = _prepare(trainer, args, checks)

    print(f"[repro] reproducing {SOURCE_RUN}")
    print(f"[repro] threads pinned to {PINNED_THREADS} "
          f"({', '.join(THREAD_VARIABLES)})")
    print(f"[repro] {len(saved)} saved coefficient(s); equivalence rule "
          f"v{RULE_VERSION}, gate budget {args.tolerance:.6e} "
          f"({args.tolerance / describe_tolerance()['float32_eps']:.0f} "
          f"float32 epsilons)")
    print(f"[repro] forward passes: 0 (cached activations only)")

    # ---- is the rule admissible on this corpus at all? -------------------
    # Measured on the frozen record, before anything is trained, so a
    # tolerance this corpus cannot carry is refused by the dry run rather
    # than discovered after five folds.
    margins = _saved_decision_margins(prepared, saved, saved_tau)
    assert_rule_is_admissible(margins, checks)
    print(f"[repro] tightest gate-space margin "
          f"{margins['gate_space_margin']:.6e} at the "
          f"{margins['gate_space_binding_boundary']}, "
          f"{margins['gate_space_margin_in_tolerances']:.1f}x the budget; "
          f"min |z_H - tau_hi| {margins['safety_veto_logit_margin']:.6e}")

    # Refused now rather than after five folds: a reproduction writes
    # somewhere new so a previous one is never half-overwritten.
    out_dir = Path(args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ReproductionError(
            f"{out_dir} already holds files. A reproduction writes somewhere "
            f"new so a previous one is never half-overwritten, and "
            f"{SOURCE_RUN} is never written into.")

    if args.dry_run:
        print("[repro] dry run: nothing trained, nothing written")
        print(f"[repro] {checks.as_dict()['checks_run']} check(s), all "
              f"passed: {checks.all_passed}")
        return 0 if checks.all_passed else 2

    outcome = _reproduce(trainer, prepared, saved, args, checks)
    return _decide(outcome, prepared, saved, saved_steps, saved_tau, args,
                   checks, spec)


def _saved_decision_margins(prepared, saved, saved_tau) -> Dict[str, Any]:
    """How close ``gate_oof_v1``'s own coefficients sit to every boundary.

    Read entirely off the frozen record and the frozen brakes, so it needs no
    training and says the same thing on every machine. The clamp ceiling is
    recovered by running the frozen ``apply_counterfactual_clamp`` at
    ``g = 1``: what comes back is the ceiling itself, which is what the
    boundary is.
    """
    import torch

    frame = prepared["frame"]
    budgets = prepared["budgets"]
    kappa_h = prepared["kappa"][prepared["safety_head"]]

    gate, ceiling, z_safety, tau = [], [], [], []
    for position, row_id in enumerate(frame.row_ids):
        record = saved.get(row_id)
        if record is None:
            raise ReproductionError(
                f"{row_id} is in the corpus and absent from {SOURCE_RUN}'s "
                f"coefficients; the margins are measured against that record")
        k = int(record["fold"])
        z = torch.tensor([float(record["z_safety_before"])],
                         dtype=torch.float32)
        r = torch.tensor([float(budgets[position])], dtype=torch.float32)
        top = apply_counterfactual_clamp(
            torch.ones_like(z), z, r, float(kappa_h[k]),
            float(saved_tau[k]))
        gate.append(float(record["g_R"]))
        ceiling.append(float(top[0]))
        z_safety.append(float(record["z_safety_before"]))
        tau.append(float(saved_tau[k]))
    return decision_margins(gate, ceiling, z_safety, tau)


def _decide(outcome, prepared, saved, saved_steps, saved_tau, args, checks,
            spec) -> int:
    """Evaluate the rule, write the diagnostic, and only then decide.

    The order is the whole point. ``budget_equivalence_v1`` raised inside its
    own report and discarded 960 completed forward passes; this writes the
    comparison first and lets the exit status follow it.
    """
    report = evaluate_equivalence(
        row_ids=outcome["row_ids"], saved=saved,
        reproduced_gate=outcome["gate"],
        reproduced_fold=outcome["fold"],
        reproduced_label=outcome["label"],
        reproduced_vetoed=outcome["vetoed"],
        reproduced_clamp_ceiling=outcome["clamp_ceiling"],
        reproduced_safety_logit=outcome["z_safety"],
        saved_steps=saved_steps, reproduced_steps=outcome["selected_steps"],
        saved_tau=saved_tau, reproduced_tau=outcome["tau_hi"],
        pinned_probe_digests=dict(prepared["pinned_probe_digests"]),
        actual_probe_digests=dict(outcome["probe_digests"]),
        budgets=outcome["budgets"],
        written_state_norms=outcome["written_state_norms"],
        alpha=ALPHA_R, checkpoint_gate=outcome["checkpoint_gate"],
        tolerance=float(args.tolerance))

    out_dir = Path(args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ReproductionError(
            f"{out_dir} already holds files. A reproduction writes somewhere "
            f"new so a previous one is never half-overwritten, and "
            f"{SOURCE_RUN} is never written into.")
    diagnostic = write_diagnostic(out_dir, report, {
        "reproduces": SOURCE_RUN,
        "source_run": str(args.source_run),
        "created_utc": utc_now(),
        "package_version": PACKAGE_VERSION,
        "pinned_threads": {n: os.environ.get(n) for n in THREAD_VARIABLES},
        "refit_performed": False,
        "checkpoints_written": False,
        "forward_passes": 0,
        "contains_prompt_or_response_text": False,
        "contains_per_row_coefficients": False,
        "reads_the_sealed_set": False,
        "environment": environment_record(PACKAGE_VERSION),
        "checklist": checks.as_dict(),
    })
    print(f"[repro] wrote {diagnostic}")

    error = report.coefficient_error
    print(f"[repro] max |reproduced - saved| = {error['max']:.6e} "
          f"({error['max_in_float32_epsilons']:.3f} float32 epsilons) over "
          f"{error['rows_compared']} row(s)")
    print(f"[repro] {error['rows_with_any_difference']} row(s) differ at all; "
          f"median {error['median']:.3e}, p99 {error['quantiles']['p99']:.3e}")
    print(f"[repro] decision moves: "
          f"{report.decision_margins['reporting_bucket_moves']} bucket, "
          f"{report.decision_margins['veto_decision_moves']} veto, "
          f"{report.decision_margins['clamp_decision_moves']} clamp")
    print(f"[repro] applied edit differs by at most "
          f"{report.edit_difference['max_edit_difference_in_bf16_ulps']:.3e} "
          f"bfloat16 ULP of the state it is written into")
    for clause in report.clauses:
        checks.record(clause.name, clause.passed, clause.detail)

    if not report.passed:
        for clause in report.failures:
            print(f"[FAIL] {clause.name}: {clause.detail}", file=sys.stderr)
        print(f"[repro] NOT EQUIVALENT. The comparison is in "
              f"{diagnostic.name}; nothing else was written and no controller "
              f"was produced.", file=sys.stderr)
        return 2

    # Only an equivalent run earns a deployable controller.
    return _write(outcome, prepared, saved, args, checks, spec, report,
                  diagnostic)


def _prepare(trainer, args, checks) -> Dict[str, Any]:
    """Corpus, folds, probes, direction, budgets: the trainer's own path."""
    from routegate.data import load_local_paths
    from routegate.dualsteer import load_validated_bank
    from routegate.frozenprobe import (
        assert_repair_direction_reduces_the_repair_logit,
        read_saved_predictions,
        reconstruct_and_verify,
    )
    from routegate.probeheads import REPAIR_HEAD, SAFETY_HEAD
    from routegate.validation import assert_not_sealed

    local_paths = (load_local_paths(args.paths) if args.paths
                   else load_local_paths())
    probe_dir = REPO_ROOT / trainer.PROBE_RUN_DIRECTORY
    pinned_probe_digests: Dict[str, str] = {}
    actual_probe_digests: Dict[str, str] = {}
    for name, expected in sorted(trainer.PROBE_ARTIFACTS.items()):
        actual = sha256_file(probe_dir / name)
        pinned_probe_digests[name] = expected
        actual_probe_digests[name] = actual
        checks.require(f"probe_artifact_{name}", actual == expected,
                       f"{name} hashes {actual[:12]}, pinned {expected[:12]}")

    frame = trainer.load_ebh_frame()
    checks.require("corpus_rows", frame.n == trainer.EXPECTED_ROWS,
                   f"{frame.n} row(s), expected {trainer.EXPECTED_ROWS}")
    assignment, source = trainer.load_bank_fold_assignment(
        REPO_ROOT / trainer.FOLD_DIR)
    checks.require("n_splits", assignment.n_splits == N_SPLITS,
                   f"{assignment.n_splits} folds, expected {N_SPLITS}")

    saved_predictions = read_saved_predictions(
        probe_dir / trainer.PROBE_PREDICTIONS, trainer.READ_STATE)
    probes = reconstruct_and_verify(
        frame, assignment, saved_predictions, trainer.READ_STATE,
        trainer.REPRODUCTION_TOLERANCE, checks)

    from routegate.twosite import build_candidate_record
    expected_bank = build_candidate_record()["direction"]["artifact_sha256"]
    bank = load_validated_bank(assert_not_sealed(Path(args.direction_bank)),
                               expected_bank, checks)
    d_repair, _ = bank.directions_for_state(trainer.READ_STATE)
    d_repair = np.asarray(d_repair, dtype=np.float64)
    kappa = assert_repair_direction_reduces_the_repair_logit(
        probes, d_repair, checks)

    design = trainer.state_matrix(frame, trainer.READ_STATE)
    budget_state = trainer.block_input_state(min(trainer.CANDIDATE_BLOCKS))
    budgets = np.linalg.norm(
        np.asarray(frame.activations[:, budget_state, :], dtype=np.float64),
        axis=-1)

    # The equivalence rule's edit clause needs, for every site the actuator
    # writes at, the budget it scales by and the norm of the state it lands
    # on -- the latter sets the bfloat16 rounding step the difference is
    # measured against. Both are read from the unsteered capture.
    def _state_norm(state: int) -> np.ndarray:
        return np.linalg.norm(
            np.asarray(frame.activations[:, state, :], dtype=np.float64),
            axis=-1)

    site_budgets = {int(b): _state_norm(trainer.block_input_state(b))
                    for b in trainer.CANDIDATE_BLOCKS}
    written_state_norms = {int(b): _state_norm(int(b) + 1)
                           for b in trainer.CANDIDATE_BLOCKS}
    return {"frame": frame, "assignment": assignment, "source": source,
            "probes": probes, "kappa": kappa, "design": design,
            "budgets": budgets, "d_repair": d_repair, "bank": bank,
            "expected_bank": expected_bank,
            "site_budgets": site_budgets,
            "written_state_norms": written_state_norms,
            "pinned_probe_digests": pinned_probe_digests,
            "actual_probe_digests": actual_probe_digests,
            "repair_head": REPAIR_HEAD, "safety_head": SAFETY_HEAD,
            "local_paths": local_paths}


def _train_one(trainer, prepared, k, torch):
    """One fold, trained exactly as the frozen trainer trains it."""
    from routegate.frozenprobe import TorchProbe

    frame = prepared["frame"]
    design, budgets = prepared["design"], prepared["budgets"]
    labels = np.array(frame.labels)
    fold_of = np.array([prepared["assignment"].fold_of(r)
                        for r in frame.row_ids])

    torch.manual_seed(seed_for(k))
    inner = inner_fold_for(k)
    held = fold_of == k
    train = ~held
    inner_mask_full = fold_of == inner
    inner_train = train & ~inner_mask_full

    repair_probe = TorchProbe(prepared["probes"][prepared["repair_head"]][k])
    safety_probe = TorchProbe(prepared["probes"][prepared["safety_head"]][k])
    mu = design[inner_train].mean(axis=0)
    sd = np.maximum(design[inner_train].std(axis=0), 1e-8)

    x = torch.as_tensor((design - mu) / sd, dtype=torch.float32)
    h = torch.as_tensor(design, dtype=torch.float32)
    b = torch.as_tensor(budgets, dtype=torch.float32)
    d = torch.as_tensor(prepared["d_repair"], dtype=torch.float32)
    for tensor in (x, h, b, d):
        tensor.requires_grad_(False)
    with torch.no_grad():
        z_h_unsteered = safety_probe.logit(h)

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
    inner_outcome = trainer._fold_history(
        net, optimiser, tensors, masks, torch.as_tensor(inner_mask_full),
        torch.as_tensor(inner_train), tau_hi, torch)

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
    with torch.no_grad():
        g_held = gate(net, x[torch.as_tensor(held)])
    return {"net": net, "mu": mu, "sd": sd, "tau_hi": float(tau_hi),
            "selected_step": inner_outcome["selected_step"],
            "held": held, "train": train,
            "g_held": g_held.numpy().astype(np.float64),
            "n_train": int(train.sum()),
            "x": x, "h": h, "b": b, "d": d, "is_e": is_e,
            "z_h": z_h_unsteered, "protected": protected,
            "repair": repair_probe, "safety": safety_probe}


def _reproduce(trainer, prepared, saved, args, checks) -> Dict[str, Any]:
    """Retrain every fold and collect what the equivalence rule needs.

    Nothing here decides. Disagreement is *collected*, not raised, so the
    comparison can be written before an exit status is chosen -- including the
    disagreements a raise would previously have hidden.
    """
    import torch
    torch.use_deterministic_algorithms(True)
    from routegate.gatecontroller import apply_safety_veto

    frame = prepared["frame"]
    row_ids = list(frame.row_ids)
    labels = list(frame.labels)
    budgets = prepared["budgets"]
    kappa_h = prepared["kappa"][prepared["safety_head"]]

    folds: Dict[int, Dict[str, Any]] = {}
    per_fold: List[Dict[str, Any]] = []
    gate_of: Dict[str, float] = {}
    fold_of: Dict[str, int] = {}
    label_of: Dict[str, str] = {}
    vetoed_of: Dict[str, bool] = {}
    ceiling_of: Dict[str, float] = {}
    z_safety_of: Dict[str, float] = {}
    checkpoint_gate: Dict[str, float] = {}
    selected_steps: Dict[int, int] = {}
    tau_by_fold: Dict[int, float] = {}
    worst_so_far = 0.0

    for k in range(N_SPLITS):
        outcome = _train_one(trainer, prepared, k, torch)
        folds[k] = outcome
        selected_steps[k] = int(outcome["selected_step"])
        tau_by_fold[k] = float(outcome["tau_hi"])

        index = np.flatnonzero(outcome["held"])
        held_ids = [row_ids[i] for i in index]
        g_held = np.asarray(outcome["g_held"], dtype=np.float64)
        z_held = outcome["z_h"][torch.as_tensor(outcome["held"])]
        r_held = torch.as_tensor(budgets[index], dtype=torch.float32)

        # Both brakes come from the frozen implementations, evaluated at
        # g = 1 so what is recovered is the decision surface itself: the veto
        # keep/drop flag, and the counterfactual ceiling the clamp would take
        # a minimum against. Neither reads the reproduced gate.
        ones = torch.ones_like(z_held)
        keep = apply_safety_veto(ones, z_held, outcome["tau_hi"])
        ceiling = apply_counterfactual_clamp(
            ones, z_held, r_held, float(kappa_h[k]), outcome["tau_hi"])

        # The deployable checkpoint, scored through the live path, before any
        # file is written: a controller that only reproduces in memory is not
        # a controller.
        payload = build_checkpoint(
            KIND_FOLD, k, outcome["net"].state_dict(), outcome["mu"],
            outcome["sd"], outcome["tau_hi"], outcome["selected_step"],
            outcome["n_train"])
        g_checkpoint = gate_from_activation(
            prepared["design"][index], payload)

        errors = []
        for position, row_id in enumerate(held_ids):
            gate_of[row_id] = float(g_held[position])
            fold_of[row_id] = k
            label_of[row_id] = str(labels[index[position]])
            vetoed_of[row_id] = bool(float(keep[position]) == 0.0)
            ceiling_of[row_id] = float(ceiling[position])
            z_safety_of[row_id] = float(z_held[position])
            checkpoint_gate[row_id] = float(g_checkpoint[position])
            record = saved.get(row_id)
            if record is not None:
                errors.append(abs(float(g_held[position]) - record["g_R"]))
        worst_so_far = max([worst_so_far] + errors)

        per_fold.append({"fold": k, "selected_step": outcome["selected_step"],
                         "tau_hi": outcome["tau_hi"],
                         "n_train": outcome["n_train"],
                         "n_held_out": int(outcome["held"].sum()),
                         "held_out_mean_g": float(g_held.mean())})
        print(f"[repro] fold {k}: step {outcome['selected_step']}, "
              f"mean g_R {g_held.mean():.6f}, "
              f"max |delta| so far {worst_so_far:.3e}", flush=True)

    return {"folds": folds, "per_fold": per_fold, "row_ids": row_ids,
            "gate": gate_of, "fold": fold_of, "label": label_of,
            "vetoed": vetoed_of, "clamp_ceiling": ceiling_of,
            "z_safety": z_safety_of, "checkpoint_gate": checkpoint_gate,
            "selected_steps": selected_steps, "tau_hi": tau_by_fold,
            "budgets": prepared["site_budgets"],
            "written_state_norms": prepared["written_state_norms"],
            "probe_digests": prepared["actual_probe_digests"],
            "max_error": worst_so_far}


def _refit(prepared, folds, steps, torch):
    """The all-development refit. Deployment only; scores no held-out row."""
    from routegate.frozenprobe import TorchProbe

    design, budgets = prepared["design"], prepared["budgets"]
    frame = prepared["frame"]
    labels = np.array(frame.labels)

    # Fold 0's probes are used for the refit's loss; every fold's probe is a
    # reconstruction of the same frozen head, and the refit is not scored
    # out-of-fold, so no held-out claim rests on this choice.
    repair_probe = TorchProbe(prepared["probes"][prepared["repair_head"]][0])
    safety_probe = TorchProbe(prepared["probes"][prepared["safety_head"]][0])

    mu = design.mean(axis=0)
    sd = np.maximum(design.std(axis=0), 1e-8)
    torch.manual_seed(seed_for(0))
    x = torch.as_tensor((design - mu) / sd, dtype=torch.float32)
    h = torch.as_tensor(design, dtype=torch.float32)
    b = torch.as_tensor(budgets, dtype=torch.float32)
    d = torch.as_tensor(prepared["d_repair"], dtype=torch.float32)
    with torch.no_grad():
        z_h = safety_probe.logit(h)
    is_h = torch.as_tensor(labels == "H")
    is_e = torch.as_tensor(labels == "E")
    tau_hi = veto_threshold(z_h[is_h].tolist())
    protected = protected_mask(z_h, is_h, tau_hi)

    net = build_controller(seed_for(0))
    optimiser = controller_optimiser(net)
    for _ in range(max(1, steps)):
        net.train()
        optimiser.zero_grad(set_to_none=True)
        g = gate(net, x)
        steered = steer(h, g, b, d)
        loss = total_loss(repair_loss(repair_probe.logit(steered), is_e),
                          safety_loss(z_h, safety_probe.logit(steered),
                                      protected),
                          size_loss(g))
        loss.backward()
        optimiser.step()
    net.eval()
    return {"net": net, "mu": mu, "sd": sd, "tau_hi": float(tau_hi),
            "selected_step": steps, "n_train": int(design.shape[0])}


def _write(outcome, prepared, saved, args, checks, spec, report,
           diagnostic) -> int:
    """The deployable artifacts. Reached only by a run the rule passed."""
    import json
    import torch

    out_dir = Path(args.output_dir)

    # The refit happens here, after equivalence, and not before: an
    # all-development controller built on a run that did not reproduce would
    # be a deployment artifact with nothing behind it.
    steps = refit_step_count([f["selected_step"] for f in outcome["per_fold"]])
    refit = _refit(prepared, outcome["folds"], steps, torch)
    checks.record("refit_steps", True,
                  f"{steps} step(s), the median of "
                  f"{[f['selected_step'] for f in outcome['per_fold']]}")

    written: Dict[str, str] = {}
    for k, fold in sorted(outcome["folds"].items()):
        payload = build_checkpoint(
            KIND_FOLD, k, fold["net"].state_dict(), fold["mu"], fold["sd"],
            fold["tau_hi"], fold["selected_step"], fold["n_train"])
        path = out_dir / f"controller_fold_{k}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True),
                        encoding="utf-8")
        written[path.name] = payload["weights_sha256"]

    payload = build_checkpoint(
        KIND_REFIT, None, refit["net"].state_dict(), refit["mu"], refit["sd"],
        refit["tau_hi"], refit["selected_step"], refit["n_train"])
    path = out_dir / "controller_all_development_refit.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True),
                    encoding="utf-8")
    written[path.name] = payload["weights_sha256"]

    summary = {
        "schema": REPRODUCTION_SCHEMA,
        "reproduces": SOURCE_RUN,
        "created_utc": utc_now(),
        "package_version": PACKAGE_VERSION,
        "pinned_threads": {name: os.environ.get(name)
                           for name in THREAD_VARIABLES},
        "equivalence": {
            "rule_version": RULE_VERSION,
            "tolerance": describe_tolerance(),
            "applied_tolerance": float(args.tolerance),
            "diagnostic": diagnostic.name,
            "passed": report.passed,
            "clauses": [c.as_dict() for c in report.clauses],
            "superseded_rules": [dict(r) for r in SUPERSEDED_RULES],
        },
        "max_abs_coefficient_error": report.coefficient_error["max"],
        "max_abs_coefficient_error_in_float32_epsilons":
            report.coefficient_error["max_in_float32_epsilons"],
        "rows_compared": report.coefficient_error["rows_compared"],
        "per_fold": outcome["per_fold"],
        "refit_steps": steps,
        "refit_rule": "median of the five fold-selected steps",
        "final_refit_performed": True,
        "checkpoints": written,
        "forward_passes": 0,
        "loads_a_model": False,
        "generates_responses": False,
        "calls_a_judge": False,
        "reads_the_sealed_set": False,
        "contains_prompt_or_response_text": False,
        "contains_per_row_coefficients": False,
        "environment": environment_record(PACKAGE_VERSION),
        "checklist": checks.as_dict(),
    }
    (out_dir / "reproduction_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    # The diagnostic was written before the decision; restate the outcome in
    # it now that there is one, without disturbing what it recorded.
    write_diagnostic(out_dir, report, {
        "reproduces": SOURCE_RUN,
        "source_run": str(args.source_run),
        "created_utc": utc_now(),
        "package_version": PACKAGE_VERSION,
        "pinned_threads": {n: os.environ.get(n) for n in THREAD_VARIABLES},
        "refit_performed": True,
        "refit_steps": steps,
        "checkpoints_written": True,
        "checkpoints": written,
        "forward_passes": 0,
        "contains_prompt_or_response_text": False,
        "contains_per_row_coefficients": False,
        "reads_the_sealed_set": False,
        "environment": environment_record(PACKAGE_VERSION),
        "checklist": checks.as_dict(),
    })

    print(f"[repro] wrote {out_dir}")
    print(f"[repro] EQUIVALENT under rule v{RULE_VERSION}: "
          f"{len(report.clauses)} clause(s), all passed")
    print(f"[repro] {len(written)} checkpoint(s): 5 fold + 1 refit")
    return 0 if checks.all_passed else 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
