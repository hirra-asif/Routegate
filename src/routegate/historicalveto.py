"""Read the frozen residual/routing H vetoes for a historical diagnostic.

These two recorded vetoes support dynamic_raw_h_transition_v1. They are not the
final RouteGate decision rule. The loader was extracted from the retired guarded
pilot without changing its input hashes, thresholds or 1e-9 boundary convention.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .thesisresults import EXPECTED_INPUT_DIGESTS
from .validation import Checklist, ValidationError, assert_hash

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "historical_h_veto_v1"


class HistoricalVetoError(ValidationError):
    """The recorded veto inputs do not define a valid diagnostic."""


class HistoricalVetoBlocked(HistoricalVetoError):
    """A required frozen input is absent or incomplete."""


@dataclass(frozen=True)
class VetoSource:
    """One frozen H veto: where its probabilities and its threshold live."""

    family: str
    predictions: str
    predictions_digest_key: str
    thresholds: str
    thresholds_digest_key: str
    probability_column: str
    threshold_path: Tuple[str, ...]
    rule: str

    def prediction_path(self, root: Optional[Path] = None) -> Path:
        return (root or REPO_ROOT) / self.predictions

    def threshold_file(self, root: Optional[Path] = None) -> Path:
        return (root or REPO_ROOT) / self.thresholds


#: The safety-first RouteGate residual veto: the H half of the frozen v1 gate
#: ``gate_full = (p_E >= threshold_E) AND (p_H < threshold_H)``.
RESIDUAL_VETO = VetoSource(
    family="residual",
    predictions="outputs/routegate_residual_v1/oof_predictions.csv",
    predictions_digest_key="residual_oof_predictions",
    thresholds="outputs/routegate_residual_v1/thresholds.json",
    thresholds_digest_key="residual_thresholds",
    probability_column="p_H",
    threshold_path=("threshold_H",),
    rule="H flagged iff p_H >= threshold_H; recall >= 0.99, then maximise "
         "safe specificity; largest satisfying t")

#: The MoE-routing veto, from the routing sensor family calibrated under the
#: identical rule and tie-break, copied verbatim from the residual run.
ROUTING_VETO = VetoSource(
    family="routing",
    predictions="outputs/routegate_routing_analysis_v1/oof_predictions.csv",
    predictions_digest_key="routing_oof_predictions",
    thresholds="outputs/routegate_routing_analysis_v1/thresholds.json",
    thresholds_digest_key="routing_thresholds",
    probability_column="routing_p_H",
    threshold_path=("families", "routing", "threshold_H", "threshold"),
    rule="H flagged iff p_H >= threshold_H; recall >= 0.99, then maximise "
         "safe specificity; largest satisfying t")

VETO_SOURCES: Tuple[VetoSource, ...] = (RESIDUAL_VETO, ROUTING_VETO)

#: A row exactly ON a threshold is VETOED. The threshold rule is ``flag iff
#: p_H >= t`` and ``t`` is itself one of the out-of-fold probabilities, so a
#: row sits exactly on it; and the residual run record stores ``p_H`` to ten
#: decimal places while the routing record stores the same number in full, so
#: a bare ``p_H < t`` reads that row two different ways depending on which file
#: it came from. Backing the comparison off by this much decides it the safe
#: way from either file, and the disagreement is recorded rather than papered
#: over.
VETO_TIE_EPSILON = 1e-9
VETO_RULE = ("permit = (residual p_H < tau_H_residual - eps) AND "
             "(routing p_H < tau_H_routing - eps)")
VETO_TIE_NOTE = (
    "eps = 1e-9. The frozen rule flags H iff p_H >= tau, and tau is one of the "
    "recorded out-of-fold probabilities, so at least one row lies exactly on "
    "it. That row is vetoed, from either source file, at either precision.")

#: How far the two run records may disagree about the SAME residual
#: probability. They store it at different precisions; anything beyond the
#: rounding that explains means they are not the same run.
VETO_CROSS_CHECK_ATOL = 1e-9


@dataclass(frozen=True)
class VetoReading:
    """Both hard-veto signals for one row, and what they decided."""

    row_id: str
    residual_p_h: float
    residual_threshold: float
    residual_permits: bool
    routing_p_h: float
    routing_threshold: float
    routing_permits: bool

    @property
    def permits(self) -> bool:
        return self.residual_permits and self.routing_permits

    def as_dict(self) -> Dict[str, Any]:
        return {
            "residual_p_H": self.residual_p_h,
            "residual_threshold_H": self.residual_threshold,
            "residual_veto_permits": self.residual_permits,
            "routing_p_H": self.routing_p_h,
            "routing_threshold_H": self.routing_threshold,
            "routing_veto_permits": self.routing_permits,
            "both_vetoes_permit": self.permits,
            "tie_epsilon": VETO_TIE_EPSILON,
            "rule": VETO_RULE,
        }


def _dig(payload: Mapping[str, Any], path: Sequence[str], where: str) -> Any:
    node: Any = payload
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            raise HistoricalVetoBlocked(
                f"{where} carries no {'.'.join(path)}; the frozen threshold "
                f"this veto needs is not in the record that is supposed to "
                f"hold it")
        node = node[key]
    return node


def load_hard_vetoes(root: Optional[Path] = None,
                     checks: Optional[Checklist] = None
                     ) -> Tuple[Dict[str, VetoReading], Dict[str, Any]]:
    """Both frozen H vetoes, hash-gated, for every row they cover.

    Neither threshold is recomputed. Both files are checked against the digests
    the completed thesis-figure derivation already pinned them at, so a run
    against an edited sensor record refuses before it reads a probability.
    """
    checks = checks or Checklist(PROTOCOL)
    readings: Dict[str, Dict[str, Any]] = {}
    provenance: Dict[str, Any] = {}

    for source in VETO_SOURCES:
        pred = source.prediction_path(root)
        thr = source.threshold_file(root)
        for path in (pred, thr):
            if not path.is_file():
                raise HistoricalVetoBlocked(
                    f"{path} is absent. The {source.family} H veto is a frozen "
                    f"run record; it is read, never rebuilt, and this "
                    f"experiment stops rather than proceeding without one of "
                    f"its two hard brakes.")
        pred_sha = assert_hash(
            pred, EXPECTED_INPUT_DIGESTS[source.predictions_digest_key],
            f"{source.family} out-of-fold predictions")
        thr_sha = assert_hash(
            thr, EXPECTED_INPUT_DIGESTS[source.thresholds_digest_key],
            f"{source.family} thresholds")
        blob = json.loads(thr.read_text(encoding="utf-8"))
        threshold = float(_dig(blob, source.threshold_path, str(thr)))
        with open(pred, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if source.probability_column not in (rows[0] if rows else {}):
            raise HistoricalVetoBlocked(
                f"{pred} has no {source.probability_column} column")
        for row in rows:
            rid = row["row_id"]
            entry = readings.setdefault(rid, {})
            if source.family in entry:
                raise HistoricalVetoError(f"{pred}: {rid} appears twice")
            entry[source.family] = (float(row[source.probability_column]),
                                    threshold)
        provenance[source.family] = {
            "predictions": source.predictions,
            "predictions_sha256": pred_sha,
            "thresholds": source.thresholds,
            "thresholds_sha256": thr_sha,
            "probability_column": source.probability_column,
            "threshold_H": threshold,
            "threshold_rule": source.rule,
            "threshold_is_frozen": True,
            "rows": len(rows),
        }
        checks.record(f"{source.family}_veto_source", True,
                      f"{pred.name} {pred_sha[:12]}, threshold_H {threshold!r}")

    incomplete = sorted(r for r, e in readings.items() if len(e) != len(VETO_SOURCES))
    checks.require(
        "every_row_carries_both_veto_signals", not incomplete,
        f"{len(incomplete)} row(s) are scored by only one sensor family"
        if incomplete else f"{len(readings)} row(s), two signals each")

    out: Dict[str, VetoReading] = {}
    for rid, entry in readings.items():
        rp, rt = entry["residual"]
        gp, gt = entry["routing"]
        out[rid] = VetoReading(
            row_id=rid,
            residual_p_h=rp, residual_threshold=rt,
            residual_permits=bool(rp < rt - VETO_TIE_EPSILON),
            routing_p_h=gp, routing_threshold=gt,
            routing_permits=bool(gp < gt - VETO_TIE_EPSILON))

    provenance["rule"] = VETO_RULE
    provenance["tie_epsilon"] = VETO_TIE_EPSILON
    provenance["tie_note"] = VETO_TIE_NOTE
    provenance["conservatism"] = (
        "requiring BOTH families to permit is strictly more conservative than "
        "either alone: a row either family flags is not steered")
    return out, provenance

