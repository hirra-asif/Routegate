"""The deployable safety-first RouteGate gate: checkpoints and the live path.

WHY THIS EXISTS
---------------
``outputs/routegate_residual_v1`` is a *sensor evaluation* run. Its own
``config.json`` says so -- ``purpose: "residual-only E sensor and H veto
baseline; sensor evaluation only"`` -- and both sensors are recorded as
``fitted_on`` the training folds only. It persisted five in-memory pipelines
per sensor, wrote their out-of-fold probabilities to ``oof_predictions.csv``,
and kept no weights.

So the development result -- 37/89 E substantive repairs, 0/178 material B
degradations, 0/213 actionable H regressions -- is produced by reading ``p_E``,
``p_H`` and ``gate_full`` out of that table **by row id**. That is a lookup,
not inference, and it cannot fire on a prompt the table has never seen. It is
the same failure ``livecontroller`` documents for the coefficient controller:
a column of per-row numbers is not a controller.

This module closes exactly that gap and nothing else. It defines what a
deployable sensor checkpoint is, refits the two sensors on all 480 development
rows to obtain final weights, and applies them in one live forward pass.

WHAT IS AND IS NOT RE-DECIDED
-----------------------------
Nothing about the gate is re-decided. The representation (residual state 45,
final prompt token), the estimator, the hyperparameter grid and its selection
rule, both thresholds, the gate rule, the actuator, alpha and the intervention
sites are all the frozen ones and are imported or hash-gated rather than
retyped. The refit changes one thing: which rows the sensors were fitted on --
all 480 instead of four folds out of five -- because a deployment artifact has
no held-out row to respect.

THE REPORTED NUMBERS DO NOT MOVE
--------------------------------
The refit produces *weights*, never an evaluation. 37/89, 0/178 and 0/213
remain the out-of-fold development estimates and are the only figures this
system reports. A refit sensor has seen every development row, so its
predictions on those rows are training-set predictions and are never a
reading: :func:`assert_not_a_development_reading` refuses a caller that would
score a development row with the deployment checkpoint, exactly as
``livecontroller`` refuses the all-development controller on a development row.

WHAT A CHECKPOINT MUST CARRY
----------------------------
A sensor is not its coefficients alone. The input is standardised inside the
pipeline, so ``mean`` and ``scale`` are part of the function; the decision is
taken against a frozen threshold, so the threshold travels with the bundle. A
checkpoint missing any of them would compute a different probability from the
same activation.

None of it is sensitive: two 2048-vectors, a coefficient vector and two
scalars. No prompt, no response, no per-row probability and no corpus row id is
stored, and :func:`assert_checkpoint_is_metadata_only` refuses one that tries.

THE LIVE PATH
-------------
Two passes, which is what the frozen architecture already specifies
(``config.json -> architecture``: ``TWO-PASS prompt-level gate``):

1. an unsteered diagnostic forward pass on the prompt, reading the residual
   stream at package state 45 -- the output of block 44 -- at the final prompt
   token;
2. ``p_E`` and ``p_H`` from the two checkpoints, then
   ``fire = (p_E >= threshold_E) AND (p_H < threshold_H)``; if it fires,
   generate with the frozen actuator, otherwise generate normally.

:func:`GateBundle.decide` takes an activation vector and returns the decision.
It accepts no row id, consults no table and has no corpus in scope.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .triplets import sha256_text
from .validation import ValidationError, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_SCHEMA = "routegate/gate_sensor_checkpoint/1"

# ===================================================== the frozen sensor run
#: The completed sensor-evaluation run every constant below is taken from.
SENSOR_RUN = REPO_ROOT / "outputs" / "routegate_residual_v1"
THRESHOLDS_FILE = SENSOR_RUN / "thresholds.json"
OOF_PREDICTIONS_FILE = SENSOR_RUN / "oof_predictions.csv"
SENSOR_CONFIG_FILE = SENSOR_RUN / "config.json"

THRESHOLDS_SHA256 = (
    "e7d56cfe1b2bb763fc52a8dd86a30254375f81b594ed8c942c3fe24715a35507")
OOF_PREDICTIONS_SHA256 = (
    "875b3843983fa37cd834572186aff440ec368439e0f48edc6d35cecd4abb61d2")
SENSOR_CONFIG_SHA256 = (
    "d691c5d8520039c6fc9e467d4326e7e047eeda433abef691333112538cdbc5ec")

#: The frozen feature artifact: [480, 49, 2048], final prompt token.
ACTIVATIONS_SHA256 = (
    "af89872d5decd4dc432ec579a941e6247bc7de63ff5690432e9f5812a994e2d8")
EXTRACTION_MANIFEST_SHA256 = (
    "a3b7444b1a2782b5bc04142f3545ac5d71ad57a5f546edc315d7d51d14947abd")

#: Package state 45 == the output of transformer block 44. NOT layer 45.
READ_STATE = 45
READ_STATE_MEANS = "block 44 output"
TOKEN_POSITION = "final token of the rendered chat prompt, before generation"
HIDDEN_SIZE = 2048
EXPECTED_ROWS = 480

# ------------------------------------------------------------- the two heads
SENSOR_E = "E_sensor"
SENSOR_H = "H_veto"
SENSORS: Tuple[str, ...] = (SENSOR_E, SENSOR_H)

#: Which rows each head is fitted on, and what counts as its positive class.
#: Transcribed from the frozen run's ``models`` block, not re-chosen.
SENSOR_POSITIVE: Dict[str, str] = {SENSOR_E: "E", SENSOR_H: "H"}
SENSOR_NEGATIVE: Dict[str, Tuple[str, ...]] = {
    SENSOR_E: ("B",), SENSOR_H: ("E", "B")}

#: ``StandardScaler -> LogisticRegression(...)``. Verbatim from the frozen run.
ESTIMATOR = ("StandardScaler -> LogisticRegression(penalty=l2, solver=lbfgs, "
             "class_weight=balanced, max_iter=5000)")
PENALTY = "l2"
SOLVER = "lbfgs"
CLASS_WEIGHT = "balanced"
MAX_ITER = 5000

#: The frozen hyperparameter grid and its selection rule. The deployment refit
#: applies the SAME inner selection to all 480 rows; it does not invent a C and
#: does not copy one outer fold's choice.
C_GRID: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
INNER_SPLITS = 5
OUTER_SPLITS = 5
INNER_SCORING = "roc_auc"
SEED = 20260811
C_SELECTION_RULE = (
    "C is chosen by StratifiedKFold(n_splits=5, shuffle=True, "
    "random_state=20260811) inner cross-validation scored by roc_auc, "
    "maximising the mean inner score; deterministic tie-break = the smallest "
    "maximising C (first argmax over the ascending grid). This is the frozen "
    "run's own rule, applied to the fit set it is given.")

#: What the completed run selected per outer fold. Recorded so a reader can see
#: the deployment choice beside them; never used as an input to the refit.
C_SELECTED_PER_OUTER_FOLD: Dict[str, Tuple[float, ...]] = {
    SENSOR_E: (1.0, 0.01, 0.01, 0.1, 0.01),
    SENSOR_H: (0.01, 0.01, 0.01, 0.01, 0.01),
}

#: How closely a reconstructed fold sensor must reproduce the frozen OOF
#: probability before any deployment weight may be written. The pipeline is
#: deterministic and every input is pinned, so this is a reproduction and a
#: failure to reproduce is a failure of the stage.
OOF_REPRODUCTION_ATOL = 1e-9

KIND_FOLD = "fold"
KIND_REFIT = "all_development_refit"
KINDS: Tuple[str, ...] = (KIND_FOLD, KIND_REFIT)

#: Deployment checkpoint filenames, written under the output directory.
CHECKPOINT_FILES: Dict[str, str] = {
    SENSOR_E: "gate_sensor_E_all_development_refit.json",
    SENSOR_H: "gate_sensor_H_all_development_refit.json",
}
BUNDLE_FILE = "gate_bundle.json"

#: Fields a checkpoint carries. Anything else is refused.
CHECKPOINT_FIELDS: Tuple[str, ...] = (
    "schema", "kind", "sensor", "is_out_of_fold", "fold", "read_state",
    "hidden_size", "estimator", "penalty", "solver", "class_weight",
    "max_iter", "C", "C_selection_rule", "seed", "positive_class",
    "negative_classes", "coef", "intercept", "scaler_mean", "scaler_scale",
    "trained_on_row_count", "weights_sha256",
    "contains_prompt_or_response_text", "contains_per_row_probabilities",
    "contains_row_ids",
)

#: Key fragments meaning a checkpoint picked up something it must never carry.
FORBIDDEN_FRAGMENTS: Tuple[str, ...] = (
    "prompt", "response", "text", "row_id", "screen_row", "completion",
    "answer", "p_e", "p_h", "probability", "verdict", "label_source")


class GateSensorError(ValidationError):
    """A gate sensor was asked for something it must not do."""


# ================================================================ thresholds
@dataclass(frozen=True)
class GateThresholds:
    """Both frozen thresholds and the frozen rule, read from the run record."""

    threshold_E: float
    threshold_H: float
    gate_rule: str
    source: str
    source_sha256: str

    def fires(self, p_e: float, p_h: float) -> bool:
        """The frozen rule, evaluated. Not restated anywhere else."""
        return bool(p_e >= self.threshold_E and p_h < self.threshold_H)

    def as_dict(self) -> Dict[str, Any]:
        return {"threshold_E": self.threshold_E,
                "threshold_H": self.threshold_H,
                "gate_rule": self.gate_rule,
                "source": self.source,
                "source_sha256": self.source_sha256}


def load_thresholds(path: Path | str = THRESHOLDS_FILE,
                    expected_sha256: Optional[str] = THRESHOLDS_SHA256
                    ) -> GateThresholds:
    """Both thresholds, hash-gated. Neither is ever recomputed or overridden."""
    path = Path(path)
    if not path.is_file():
        raise GateSensorError(
            f"{path} is absent. Both thresholds are read from the completed "
            f"sensor run and are never supplied from anywhere else.")
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise GateSensorError(
            f"{path.name} hashes to {digest} and this stage is pinned to "
            f"{expected_sha256}. A different threshold file is a different "
            f"gate; it is refused, not adopted.")
    blob = json.loads(path.read_text(encoding="utf-8"))
    for field in ("threshold_E", "threshold_H", "gate_rule"):
        if field not in blob:
            raise GateSensorError(f"{path.name} carries no {field}")
    return GateThresholds(
        threshold_E=float(blob["threshold_E"]),
        threshold_H=float(blob["threshold_H"]),
        gate_rule=str(blob["gate_rule"]),
        source=str(path.relative_to(REPO_ROOT)) if path.is_absolute()
        else str(path),
        source_sha256=digest)


# =============================================================== checkpoints
@dataclass(frozen=True)
class GateSensorCheckpoint:
    """One fitted sensor: weights, its standardisation, and its provenance."""

    sensor: str
    kind: str
    C: float
    coef: np.ndarray
    intercept: float
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    trained_on_row_count: int
    fold: Optional[int] = None

    @property
    def is_out_of_fold(self) -> bool:
        return self.kind == KIND_FOLD

    def weights_sha256(self) -> str:
        payload = json.dumps(
            {"coef": [float(x) for x in np.asarray(self.coef).ravel()],
             "intercept": float(self.intercept),
             "scaler_mean": [float(x) for x in np.asarray(self.scaler_mean)],
             "scaler_scale": [float(x) for x in np.asarray(self.scaler_scale)]},
            sort_keys=True, separators=(",", ":"))
        return sha256_text(payload)

    def probability(self, activation: Sequence[float]) -> float:
        """``sigmoid(w . ((h - mean) / scale) + b)`` on ONE live activation.

        Takes a vector and nothing else. No row id, no table, no corpus.
        """
        h = np.asarray(activation, dtype=np.float64).ravel()
        if h.shape[0] != np.asarray(self.coef).ravel().shape[0]:
            raise GateSensorError(
                f"{self.sensor}: the activation is {h.shape[0]}-wide and this "
                f"sensor reads {np.asarray(self.coef).ravel().shape[0]}; a "
                f"mismatch means a different state or a different model")
        if not np.isfinite(h).all():
            raise GateSensorError(
                f"{self.sensor}: the activation carries a non-finite value")
        z = float(np.dot(np.asarray(self.coef, dtype=np.float64).ravel(),
                         (h - np.asarray(self.scaler_mean, dtype=np.float64))
                         / np.asarray(self.scaler_scale, dtype=np.float64))
                  + float(self.intercept))
        # numerically stable logistic
        if z >= 0.0:
            return float(1.0 / (1.0 + np.exp(-z)))
        e = np.exp(z)
        return float(e / (1.0 + e))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema": CHECKPOINT_SCHEMA,
            "kind": self.kind,
            "sensor": self.sensor,
            "is_out_of_fold": self.is_out_of_fold,
            "fold": self.fold,
            "read_state": READ_STATE,
            "hidden_size": int(np.asarray(self.coef).ravel().shape[0]),
            "estimator": ESTIMATOR,
            "penalty": PENALTY, "solver": SOLVER,
            "class_weight": CLASS_WEIGHT, "max_iter": MAX_ITER,
            "C": float(self.C),
            "C_selection_rule": C_SELECTION_RULE,
            "seed": SEED,
            "positive_class": SENSOR_POSITIVE[self.sensor],
            "negative_classes": list(SENSOR_NEGATIVE[self.sensor]),
            "coef": [float(x) for x in np.asarray(self.coef).ravel()],
            "intercept": float(self.intercept),
            "scaler_mean": [float(x) for x in np.asarray(self.scaler_mean)],
            "scaler_scale": [float(x) for x in np.asarray(self.scaler_scale)],
            "trained_on_row_count": int(self.trained_on_row_count),
            "weights_sha256": self.weights_sha256(),
            "contains_prompt_or_response_text": False,
            "contains_per_row_probabilities": False,
            "contains_row_ids": False,
        }

    @classmethod
    def from_dict(cls, blob: Mapping[str, Any]) -> "GateSensorCheckpoint":
        if blob.get("schema") != CHECKPOINT_SCHEMA:
            raise GateSensorError(
                f"checkpoint schema {blob.get('schema')!r}, expected "
                f"{CHECKPOINT_SCHEMA!r}")
        if blob.get("sensor") not in SENSORS:
            raise GateSensorError(f"unknown sensor {blob.get('sensor')!r}")
        if blob.get("kind") not in KINDS:
            raise GateSensorError(f"unknown kind {blob.get('kind')!r}")
        checkpoint = cls(
            sensor=str(blob["sensor"]), kind=str(blob["kind"]),
            C=float(blob["C"]),
            coef=np.asarray(blob["coef"], dtype=np.float64),
            intercept=float(blob["intercept"]),
            scaler_mean=np.asarray(blob["scaler_mean"], dtype=np.float64),
            scaler_scale=np.asarray(blob["scaler_scale"], dtype=np.float64),
            trained_on_row_count=int(blob["trained_on_row_count"]),
            fold=(None if blob.get("fold") is None else int(blob["fold"])))
        recorded = blob.get("weights_sha256")
        if recorded is not None and recorded != checkpoint.weights_sha256():
            raise GateSensorError(
                f"{checkpoint.sensor}: the checkpoint records weights_sha256 "
                f"{recorded} and its own weights hash to "
                f"{checkpoint.weights_sha256()}. A checkpoint that does not "
                f"describe its own weights is not loaded.")
        return checkpoint


#: Field names exempt from the fragment scan. Two name the head's own class
#: definition; three are the negative declarations themselves, which have to
#: mention what they promise is absent. Exempt by EXACT name, never by prefix,
#: so the exemption cannot be widened by adding a lookalike field.
SCAN_EXEMPT_FIELDS: Tuple[str, ...] = (
    "positive_class", "negative_classes",
    "contains_prompt_or_response_text", "contains_per_row_probabilities",
    "contains_row_ids")

#: Those three declarations must each be present and False.
DECLARED_ABSENT: Tuple[str, ...] = (
    "contains_prompt_or_response_text", "contains_per_row_probabilities",
    "contains_row_ids")


def assert_checkpoint_is_metadata_only(blob: Mapping[str, Any]) -> None:
    """No prompt, response, row id or per-row probability in a checkpoint."""
    unknown = sorted(set(blob) - set(CHECKPOINT_FIELDS))
    if unknown:
        raise GateSensorError(
            f"the checkpoint carries undeclared field(s) {unknown}; a sensor "
            f"checkpoint carries weights and provenance and nothing else")
    for field in DECLARED_ABSENT:
        if blob.get(field) is not False:
            raise GateSensorError(
                f"the checkpoint records {field}={blob.get(field)!r}; a "
                f"deployable sensor declares each of these False and carries "
                f"none of them")
    for key in blob:
        lowered = key.lower()
        if key in SCAN_EXEMPT_FIELDS:
            continue
        hit = [f for f in FORBIDDEN_FRAGMENTS if f in lowered]
        if hit:
            raise GateSensorError(
                f"the checkpoint field {key!r} reads like {hit[0]!r}; a "
                f"deployable sensor stores no prompt, no response, no row id "
                f"and no per-row probability")


def assert_not_a_development_reading(kind: str, context: str) -> None:
    """A refit sensor has seen every development row; it reads none of them."""
    if kind == KIND_REFIT:
        raise GateSensorError(
            f"{context}: the all-development refit was fitted on all "
            f"{EXPECTED_ROWS} development rows, so its prediction on one of "
            f"them is a training-set prediction and is never a reading. The "
            f"development estimates are the out-of-fold ones -- 37/89 E, "
            f"0/178 B, 0/213 H -- and they do not move because a deployment "
            f"artifact exists.")


# ============================================================ the live gate
@dataclass(frozen=True)
class GateDecision:
    """What the gate computed for one live prompt."""

    p_E: float
    p_H: float
    fires: bool
    threshold_E: float
    threshold_H: float

    def as_dict(self) -> Dict[str, Any]:
        return {"p_E": self.p_E, "p_H": self.p_H, "gate_fires": self.fires,
                "threshold_E": self.threshold_E,
                "threshold_H": self.threshold_H,
                "read_state": READ_STATE,
                "token_position": TOKEN_POSITION}


@dataclass(frozen=True)
class GateBundle:
    """Both sensors and both thresholds: the deployable gate."""

    e_sensor: GateSensorCheckpoint
    h_veto: GateSensorCheckpoint
    thresholds: GateThresholds

    def __post_init__(self) -> None:
        if self.e_sensor.sensor != SENSOR_E:
            raise GateSensorError(
                f"the E slot holds {self.e_sensor.sensor!r}")
        if self.h_veto.sensor != SENSOR_H:
            raise GateSensorError(f"the H slot holds {self.h_veto.sensor!r}")
        if self.e_sensor.kind != self.h_veto.kind:
            raise GateSensorError(
                f"the bundle mixes a {self.e_sensor.kind} E sensor with a "
                f"{self.h_veto.kind} H veto; both halves of a gate come from "
                f"the same fit")

    def decide(self, activation: Sequence[float]) -> GateDecision:
        """The gate, on one live activation vector. Takes no row id."""
        p_e = self.e_sensor.probability(activation)
        p_h = self.h_veto.probability(activation)
        return GateDecision(p_E=p_e, p_H=p_h,
                            fires=self.thresholds.fires(p_e, p_h),
                            threshold_E=self.thresholds.threshold_E,
                            threshold_H=self.thresholds.threshold_H)

    def as_dict(self) -> Dict[str, Any]:
        return {"schema": "routegate/gate_bundle/1",
                "kind": self.e_sensor.kind,
                "read_state": READ_STATE,
                "read_state_means": READ_STATE_MEANS,
                "token_position": TOKEN_POSITION,
                "thresholds": self.thresholds.as_dict(),
                "E_sensor_weights_sha256": self.e_sensor.weights_sha256(),
                "H_veto_weights_sha256": self.h_veto.weights_sha256(),
                "requires_row_id": False,
                "reads_any_per_row_table": False}


def load_bundle(directory: Path | str,
                thresholds: Optional[GateThresholds] = None) -> GateBundle:
    """Load a written bundle back. Deterministic: bytes in, same gate out."""
    directory = Path(directory)
    checkpoints = {}
    for sensor, filename in CHECKPOINT_FILES.items():
        path = directory / filename
        if not path.is_file():
            raise GateSensorError(f"{path} is absent; the bundle is incomplete")
        blob = json.loads(path.read_text(encoding="utf-8"))
        assert_checkpoint_is_metadata_only(blob)
        checkpoints[sensor] = GateSensorCheckpoint.from_dict(blob)
    return GateBundle(e_sensor=checkpoints[SENSOR_E],
                      h_veto=checkpoints[SENSOR_H],
                      thresholds=thresholds or load_thresholds())


# ================================================================ the fitting
def select_C(X: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """The frozen inner-CV rule, applied to the fit set it is given."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    scores: Dict[str, float] = {}
    best_c: Optional[float] = None
    best_score = -np.inf
    inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True,
                            random_state=SEED)
    for c in C_GRID:                       # ascending: first argmax wins ties
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(penalty=PENALTY, solver=SOLVER,
                                       class_weight=CLASS_WEIGHT,
                                       max_iter=MAX_ITER, C=c))])
        mean = float(np.mean(cross_val_score(pipeline, X, y, cv=inner,
                                             scoring=INNER_SCORING)))
        scores[str(c)] = mean
        if mean > best_score:
            best_score, best_c = mean, c
    if best_c is None:
        raise GateSensorError("the inner grid selected no C")
    return float(best_c), scores


def fit_sensor(sensor: str, X: np.ndarray, y: np.ndarray, C: float,
               kind: str, fold: Optional[int] = None) -> GateSensorCheckpoint:
    """Fit one head and return it as a checkpoint. Deterministic."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    if sensor not in SENSORS:
        raise GateSensorError(f"unknown sensor {sensor!r}")
    if kind not in KINDS:
        raise GateSensorError(f"unknown kind {kind!r}")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty=PENALTY, solver=SOLVER,
                                   class_weight=CLASS_WEIGHT,
                                   max_iter=MAX_ITER, C=float(C)))])
    pipeline.fit(X, y)
    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]
    if list(clf.classes_) != [0, 1]:
        raise GateSensorError(
            f"{sensor}: the fitted classes are {list(clf.classes_)}; this "
            f"stage encodes the positive class as 1 and nothing else")
    return GateSensorCheckpoint(
        sensor=sensor, kind=kind, C=float(C),
        coef=np.asarray(clf.coef_, dtype=np.float64).ravel().copy(),
        intercept=float(np.asarray(clf.intercept_).ravel()[0]),
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float64).copy(),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float64).copy(),
        trained_on_row_count=int(X.shape[0]), fold=fold)


def sensor_design(sensor: str, X: np.ndarray, labels: Sequence[str]
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(X_fit, y_fit, mask)`` for one head, by its frozen class definition."""
    if sensor not in SENSORS:
        raise GateSensorError(f"unknown sensor {sensor!r}")
    labels = np.asarray(labels)
    positive = SENSOR_POSITIVE[sensor]
    negative = SENSOR_NEGATIVE[sensor]
    mask = np.isin(labels, (positive,) + tuple(negative))
    y = (labels[mask] == positive).astype(int)
    return X[mask], y, mask


__all__ = [
    "ACTIVATIONS_SHA256", "BUNDLE_FILE", "CHECKPOINT_FIELDS",
    "CHECKPOINT_FILES", "CHECKPOINT_SCHEMA", "C_GRID",
    "C_SELECTED_PER_OUTER_FOLD", "C_SELECTION_RULE", "CLASS_WEIGHT",
    "ESTIMATOR", "EXPECTED_ROWS", "EXTRACTION_MANIFEST_SHA256",
    "DECLARED_ABSENT", "FORBIDDEN_FRAGMENTS", "SCAN_EXEMPT_FIELDS", "GateBundle", "GateDecision", "GateSensorCheckpoint",
    "GateSensorError", "GateThresholds", "HIDDEN_SIZE", "INNER_SCORING",
    "INNER_SPLITS", "KINDS", "KIND_FOLD", "KIND_REFIT", "MAX_ITER",
    "OOF_PREDICTIONS_FILE", "OOF_PREDICTIONS_SHA256",
    "OOF_REPRODUCTION_ATOL", "OUTER_SPLITS", "PENALTY", "READ_STATE",
    "READ_STATE_MEANS", "SEED", "SENSORS", "SENSOR_CONFIG_FILE",
    "SENSOR_CONFIG_SHA256", "SENSOR_E", "SENSOR_H", "SENSOR_NEGATIVE",
    "SENSOR_POSITIVE", "SENSOR_RUN", "SOLVER", "THRESHOLDS_FILE",
    "THRESHOLDS_SHA256", "TOKEN_POSITION",
    "assert_checkpoint_is_metadata_only", "assert_not_a_development_reading",
    "fit_sensor", "load_bundle", "load_thresholds", "select_C",
    "sensor_design",
]
