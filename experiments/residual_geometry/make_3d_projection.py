#!/usr/bin/env python3
"""A three-dimensional held-out view of the corrected residual-geometry run.

This is a **descriptive figure**, not evidence. The corrected analysis already
decided what is true, by nested selection against a permutation null over twenty
repeated stratified splits; nothing here re-decides it. What this script does is
show one held-out fold of one repeat in three dimensions, so that the separation
the numbers report can be looked at rather than only read.

Two properties make it honest enough to publish alongside those numbers.

**Nothing is refitted on the rows being shown.** The scaler, the PLS projection
and the logistic decision plane are fitted on that fold's *training* rows only.
Every point drawn was held out of the basis that placed it, so a point's
position is a prediction about it, not a summary of it.

**The split is the frozen one.** The fold assignment is read back from
``outputs/residual_geometry/corrected/cv_assignments.csv`` and re-bound by
row_id, so this figure shows the same division of the corpus that produced the
reported ROC-AUCs. No split is drawn here, and a file that does not describe
exactly the analysed rows is refused rather than adapted to.

One caveat is stated on the figure itself and repeated here, because it is the
thing a reader could most easily be misled by: **PLS is supervised.** It is
given the training labels and finds the directions that separate them, so a
projection that separates the training classes is partly a description of the
method. The held-out points are what carry information, which is why they are
the only points drawn.

The ellipsoids are covariance ellipsoids at a fixed number of standard
deviations of the training scores. They are not confidence regions, they are not
credible regions, and they carry no coverage guarantee -- they are a drawn
summary of where each training class sat.

Usage:
    python experiments/residual_geometry/make_3d_projection.py
    python experiments/residual_geometry/make_3d_projection.py \\
        --state 45 --repeat 0 --fold 0 --n-components 3
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate.data import (                                        # noqa: E402
    describe_state,
    load_development_inputs,
    load_input_manifest,
)
from routegate.runrecord import (                                   # noqa: E402
    environment_record,
    utc_now,
    write_json,
)
from routegate.splits import load_repeated_fold_assignments         # noqa: E402
from routegate.validation import ValidationError, sha256_file       # noqa: E402
from routegate import __version__ as PACKAGE_VERSION                # noqa: E402

DEFAULT_RUN_DIR = REPO_ROOT / "outputs" / "residual_geometry" / "corrected"
ASSIGNMENTS_FILENAME = "cv_assignments.csv"
FIGURES_DIRNAME = "figures"

METADATA_SCHEMA = "routegate/pls3d_projection/1"
FIGURE_TITLE = "Held-out 3D residual landscape of over-refusal"

# The sentence the figure must carry, verbatim, wherever it is rendered.
LEAKAGE_STATEMENT = ("Descriptive visualisation; preprocessing and projection "
                     "fitted on training rows only")

# What the ellipsoids are, said in the one place that decides the wording. They
# are a drawn summary of the training scatter and nothing more; calling them a
# confidence region would claim a coverage property that no part of this
# construction establishes.
ELLIPSOID_CAPTION = ("shaded volumes are {n_sd:g}-standard-deviation covariance "
                     "ellipsoids of the training scores, not confidence regions")

DEFAULT_STATE = 45
DEFAULT_REPEAT = 0
DEFAULT_FOLD = 0
DEFAULT_COMPONENTS = 3
DEFAULT_N_SD = 1.5

# The decision plane's classifier. Separate constants from the frozen probe in
# routegate.probes on purpose: that probe is fitted in 2048 standardised
# dimensions and its settings are part of a reported result, while this one is
# fitted in three projected dimensions purely to draw a surface. Borrowing the
# reported probe's name for a differently-shaped model would invite the two to
# be read as the same object.
PLANE_PENALTY = "l2"
PLANE_C = 1.0
PLANE_CLASS_WEIGHT = "balanced"
PLANE_SOLVER = "lbfgs"
PLANE_MAX_ITER = 5000
PLANE_TOL = 1e-6
PLANE_RANDOM_STATE = 20260805

# Below this the largest plane coefficient is numerically zero and the surface
# is not a surface; the figure says so rather than drawing a fabricated one.
PLANE_EPS = 1e-12

AXIS_LABELS = ("PLS component 1", "PLS component 2", "PLS component 3")

# The figure has three axes, so three components are drawn whatever is fitted.
# Asking for more is allowed and changes nothing about them: PLS extracts
# components sequentially, so components 1-3 of a four-component fit are the
# components 1-3 of a three-component fit.
PLOT_DIMENSIONS = 3

MARKERS = {"B": "o", "E": "^"}
CENTROID_MARKER = "D"

# The angles an ellipsoid's sphere grid must contain if the surface is to reach
# the ends of its principal axes: the four quarter-turns of azimuth, and the
# equator. They are made segment boundaries rather than hoped for, because a
# plain linspace hits them only at resolutions where the arithmetic happens to
# divide evenly.
AZIMUTH_BREAKPOINTS = np.pi * np.array([0.0, 0.5, 1.0, 1.5, 2.0])
POLAR_BREAKPOINTS = np.pi * np.array([0.0, 0.5, 1.0])

# The smallest grid that can carry those angles: one sample per breakpoint.
MIN_ELLIPSOID_RESOLUTION = len(AZIMUTH_BREAKPOINTS)

# A negative eigenvalue smaller than this fraction of the covariance's scale is
# floating-point error and is clipped; anything larger is a defect and raises.
EIGENVALUE_TOLERANCE = 1e-8


# --------------------------------------------------------------- the split
def read_frozen_assignments(path: Path | str, row_ids: Sequence[str]):
    """Recover the corrected run's split, bound to the analysed row order.

    This is the reuse, and it is a load rather than a rebuild on purpose.
    ``load_repeated_fold_assignments`` refuses a file that does not name exactly
    the analysed rows, so a figure cannot silently be drawn against a different
    corpus, a different row order, or a freshly generated split that happens to
    have the same shape.
    """
    path = Path(path)
    if not path.exists():
        raise ValidationError(
            f"{path} does not exist. This figure reuses the corrected run's "
            f"frozen split; it does not create one. Run the corrected analysis "
            f"first, or point --assignments at its cv_assignments.csv.")
    return load_repeated_fold_assignments(path, row_ids)


@dataclass
class Split:
    """One (repeat, fold) division, as positions into the analysed row order."""

    repeat: int
    fold: int
    train_index: np.ndarray
    held_out_index: np.ndarray
    train_row_ids: List[str]
    held_out_row_ids: List[str]
    source: str

    def audit(self) -> None:
        """The leakage check, stated over row identity rather than positions."""
        overlap = sorted(set(self.train_row_ids) & set(self.held_out_row_ids))
        if overlap:
            raise ValidationError(
                f"{len(overlap)} row(s) are in both the training and the "
                f"held-out set of repeat {self.repeat} fold {self.fold} "
                f"(first: {overlap[0]}). The projection would then be fitted on "
                f"rows it is used to place.")
        if self.train_index.size == 0 or self.held_out_index.size == 0:
            raise ValidationError(
                f"repeat {self.repeat} fold {self.fold} has "
                f"{self.train_index.size} training and "
                f"{self.held_out_index.size} held-out rows; both must be "
                f"non-empty")


def select_split(repeated: Any, repeat: int, fold: int) -> Split:
    """The training and held-out positions for one repeat and one fold."""
    if not 0 <= repeat < repeated.n_repeats:
        raise ValidationError(
            f"repeat {repeat} is outside 0..{repeated.n_repeats - 1} in the "
            f"frozen assignment")
    assignment = repeated.assignments[repeat]
    if not 0 <= fold < assignment.n_splits:
        raise ValidationError(
            f"fold {fold} is outside 0..{assignment.n_splits - 1} in the frozen "
            f"assignment")
    train_index, held_out_index = assignment.splits()[fold]
    row_ids = list(assignment.row_ids)
    split = Split(
        repeat=repeat,
        fold=fold,
        train_index=np.asarray(train_index, dtype=int),
        held_out_index=np.asarray(held_out_index, dtype=int),
        train_row_ids=[row_ids[i] for i in train_index],
        held_out_row_ids=[row_ids[i] for i in held_out_index],
        source=assignment.source,
    )
    split.audit()
    return split


# ---------------------------------------------------------- the projection
@dataclass
class Projection:
    """A scaler, a PLS basis and a decision plane, all fitted on training rows."""

    scaler: StandardScaler
    pls: PLSRegression
    classifier: LogisticRegression
    n_components: int
    train_scores: np.ndarray
    n_train: int

    @property
    def train_scores_3d(self) -> np.ndarray:
        """The three components the figure draws."""
        return self.train_scores[:, :PLOT_DIMENSIONS]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project rows into the training basis. Fits nothing.

        The only path by which held-out rows reach the figure. Both objects were
        fitted before this method could be called, and neither is updated by it.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValidationError(
                f"expected a [rows, features] matrix, got shape {X.shape}")
        if X.shape[1] != int(self.scaler.n_features_in_):
            raise ValidationError(
                f"these rows have {X.shape[1]} features but the projection was "
                f"fitted on {int(self.scaler.n_features_in_)}")
        scores = self.pls.transform(self.scaler.transform(X))
        return np.asarray(scores, dtype=np.float64)


def fit_projection(X_train: np.ndarray, y_train: np.ndarray,
                   n_components: int = DEFAULT_COMPONENTS) -> Projection:
    """Standardise, project, then fit a plane -- each on the training rows only.

    ``scale=False`` on the PLS is not a stylistic choice: ``StandardScaler`` has
    already put every column on unit variance using the training rows' own
    statistics, and letting PLS scale again would apply a second, separately
    estimated normalisation to data that is already standardised.

    The classifier is fitted on the three drawn training *scores*, so the plane
    it defines is a surface in the space actually being looked at rather than a
    2048-dimensional hyperplane projected down into it.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)
    if X_train.ndim != 2:
        raise ValidationError(
            f"expected a [rows, features] training matrix, got {X_train.shape}")
    if X_train.shape[0] != y_train.shape[0]:
        raise ValidationError(
            f"{X_train.shape[0]} training rows but {y_train.shape[0]} labels")
    if n_components < PLOT_DIMENSIONS:
        raise ValidationError(
            f"the figure draws {PLOT_DIMENSIONS} axes, so at least "
            f"{PLOT_DIMENSIONS} components are needed; got {n_components}")
    if n_components > X_train.shape[1]:
        raise ValidationError(
            f"cannot take {n_components} components from {X_train.shape[1]} "
            f"features")
    if n_components >= X_train.shape[0]:
        raise ValidationError(
            f"cannot take {n_components} components from {X_train.shape[0]} "
            f"training rows")
    classes = sorted({int(v) for v in y_train})
    if classes != [0, 1]:
        raise ValidationError(
            f"the training rows carry classes {classes}; both E (1) and B (0) "
            f"must be present for a projection or a plane to mean anything")

    scaler = StandardScaler().fit(X_train)
    standardised = scaler.transform(X_train)

    pls = PLSRegression(n_components=int(n_components), scale=False)
    pls.fit(standardised, np.asarray(y_train, dtype=np.float64))
    train_scores = np.asarray(pls.transform(standardised), dtype=np.float64)

    classifier = LogisticRegression(
        penalty=PLANE_PENALTY, C=PLANE_C, class_weight=PLANE_CLASS_WEIGHT,
        solver=PLANE_SOLVER, max_iter=PLANE_MAX_ITER, tol=PLANE_TOL,
        random_state=PLANE_RANDOM_STATE)
    classifier.fit(train_scores[:, :PLOT_DIMENSIONS],
                   np.asarray(y_train, dtype=int))

    return Projection(scaler=scaler, pls=pls, classifier=classifier,
                      n_components=int(n_components), train_scores=train_scores,
                      n_train=int(X_train.shape[0]))


# ------------------------------------------------------------- the geometry
def class_geometry(scores: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """Per-class centroid and covariance of the training scores.

    Sample covariance, ``ddof=1``. Described everywhere as a covariance, never
    as an interval: it summarises where the training rows of a class fell, and
    makes no statement about where a future row would fall.
    """
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    if scores.shape[0] != y.shape[0]:
        raise ValidationError(
            f"{scores.shape[0]} score rows but {y.shape[0]} labels")
    geometry: Dict[str, Dict[str, Any]] = {}
    for label, value in (("B", 0), ("E", 1)):
        member = scores[y == value]
        if member.shape[0] <= scores.shape[1]:
            raise ValidationError(
                f"class {label} has {member.shape[0]} training row(s) in "
                f"{scores.shape[1]} dimensions; a covariance needs more rows "
                f"than dimensions to be anything but degenerate")
        geometry[label] = {
            "n": int(member.shape[0]),
            "centroid": member.mean(axis=0),
            "covariance": np.cov(member, rowvar=False, ddof=1),
        }
    return geometry


def _angles_through(breakpoints: np.ndarray, count: int) -> np.ndarray:
    """``count`` ascending samples spanning ``breakpoints``, hitting each exactly.

    A single ``linspace`` over the whole span lands on the quarter and half
    angles only when ``count - 1`` happens to divide by the number of segments.
    At ``count = 24`` it does not: ``linspace(0, pi, 24)`` would need index 11.5
    to reach ``pi/2``, so ``sin`` peaks at 0.9977 and the sampled surface stops
    short of the equator -- and therefore short of four of the six
    principal-axis endpoints. Sampling each segment separately makes every
    breakpoint exact by construction, at any count.

    The intervals are shared out evenly, earliest segments first, so the grid
    stays near-uniform and the result is a pure function of ``count``.
    """
    segments = len(breakpoints) - 1
    intervals = int(count) - 1
    if intervals < segments:
        raise ValidationError(
            f"a resolution of {count} cannot sample {segments + 1} required "
            f"angles; at least {segments + 1} points are needed")
    share = [intervals // segments] * segments
    for index in range(intervals % segments):
        share[index] += 1
    pieces = [np.linspace(breakpoints[i], breakpoints[i + 1], share[i] + 1)
              for i in range(segments)]
    # linspace reproduces its endpoints exactly, so every breakpoint survives;
    # the shared endpoints are dropped once so the total is exactly ``count``.
    return np.concatenate([pieces[0]] + [piece[1:] for piece in pieces[1:]])


def ellipsoid_surface(centroid: np.ndarray, covariance: np.ndarray,
                      n_sd: float = DEFAULT_N_SD, resolution: int = 40
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A closed surface at ``n_sd`` standard deviations of a 3D covariance.

    The unit sphere stretched along the eigenvectors of the covariance by
    ``n_sd * sqrt(eigenvalue)``. ``eigh`` is used rather than ``eig`` because a
    covariance is symmetric, which guarantees real eigenvalues and an orthogonal
    basis.

    The angular grid is built through fixed breakpoints rather than as a plain
    ``linspace``, so the six points where the surface touches the ends of its
    principal axes are sampled exactly at every resolution. Without that, the
    drawn ellipsoid is uniformly a little too small, by an amount that depends
    on the resolution -- which is the sort of error that never looks like one.

    The surface is closed: the azimuth runs the full turn with both ends
    present, and the polar angle runs pole to pole.
    """
    centroid = np.asarray(centroid, dtype=np.float64).ravel()
    covariance = np.asarray(covariance, dtype=np.float64)
    if centroid.shape != (3,):
        raise ValidationError(
            f"an ellipsoid needs a 3-vector centroid, got {centroid.shape}")
    if covariance.shape != (3, 3):
        raise ValidationError(
            f"an ellipsoid needs a 3x3 covariance, got {covariance.shape}")
    if n_sd <= 0:
        raise ValidationError(f"n_sd must be positive, got {n_sd}")
    if not np.allclose(covariance, covariance.T):
        raise ValidationError(
            "the covariance is not symmetric; numpy.linalg.eigh reads one "
            "triangle only, so an asymmetric matrix would be silently replaced "
            "by a symmetric one instead of rejected")

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    # A covariance is positive semi-definite, so a negative eigenvalue is
    # rounding error -- unless it is too large to be, in which case the matrix
    # is not a covariance and no ellipsoid describes it. Clipping that away
    # would draw a confident shape over a real defect.
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    tolerance = EIGENVALUE_TOLERANCE * scale
    smallest = float(eigenvalues.min())
    if smallest < -tolerance:
        raise ValidationError(
            f"the covariance has eigenvalue {smallest:.6g}, past the "
            f"{-tolerance:.3g} that floating-point error explains; it is not "
            f"positive semi-definite and does not describe an ellipsoid")
    radii = float(n_sd) * np.sqrt(np.clip(eigenvalues, 0.0, None))

    resolution = int(resolution)
    azimuth = _angles_through(AZIMUTH_BREAKPOINTS, resolution)
    polar = _angles_through(POLAR_BREAKPOINTS, resolution)
    unit = np.stack([
        np.outer(np.cos(azimuth), np.sin(polar)),
        np.outer(np.sin(azimuth), np.sin(polar)),
        np.outer(np.ones_like(azimuth), np.cos(polar)),
    ])
    stretched = unit * radii[:, None, None]
    rotated = eigenvectors @ stretched.reshape(3, -1)
    points = rotated.reshape(3, resolution, resolution)
    points = points + centroid[:, None, None]
    return points[0], points[1], points[2]


def decision_plane(coef: np.ndarray, intercept: float,
                   limits: Sequence[Tuple[float, float]], resolution: int = 14
                   ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The surface ``w . z + b = 0``, as three meshes in component order.

    Solved for whichever axis carries the largest weight, so a plane that is
    near-vertical in one axis is still drawn correctly instead of exploding.
    Returns ``None`` when every weight is numerically zero -- there is no
    surface then, and drawing one would invent a boundary the model does not
    have.

    ``limits`` sets the drawn extent only. It comes from the points already on
    the figure; the plane's orientation comes from the training fit alone.
    """
    weights = np.asarray(coef, dtype=np.float64).ravel()
    bias = float(np.asarray(intercept, dtype=np.float64).ravel()[0])
    if weights.shape != (3,):
        raise ValidationError(
            f"a plane in three dimensions needs three weights, got "
            f"{weights.shape}")
    if len(limits) != 3:
        raise ValidationError(f"expected three axis limits, got {len(limits)}")

    solved = int(np.argmax(np.abs(weights)))
    if abs(weights[solved]) < PLANE_EPS:
        return None
    free = [axis for axis in range(3) if axis != solved]

    first = np.linspace(*limits[free[0]], int(resolution))
    second = np.linspace(*limits[free[1]], int(resolution))
    mesh_first, mesh_second = np.meshgrid(first, second)

    surface: List[Optional[np.ndarray]] = [None, None, None]
    surface[free[0]] = mesh_first
    surface[free[1]] = mesh_second
    surface[solved] = -(weights[free[0]] * mesh_first
                        + weights[free[1]] * mesh_second + bias) / weights[solved]
    return surface[0], surface[1], surface[2]


def axis_limits(points: Sequence[np.ndarray], pad: float = 0.12
                ) -> List[Tuple[float, float]]:
    """A padded bounding box over every set of points that will be drawn."""
    stacked = np.vstack([np.asarray(p, dtype=np.float64).reshape(-1, 3)
                         for p in points if np.asarray(p).size])
    low = stacked.min(axis=0)
    high = stacked.max(axis=0)
    span = np.where(high - low > 0, high - low, 1.0)
    return [(float(low[i] - pad * span[i]), float(high[i] + pad * span[i]))
            for i in range(3)]


# ---------------------------------------------------------------- the names
def output_names(state: int, repeat: int, fold: int) -> Dict[str, str]:
    """The three filenames, which carry the settings that produced them.

    A figure whose name does not say which state, repeat and fold it shows
    becomes unattributable the moment a second one exists.
    """
    stem = f"heldout_pls3d_state{int(state)}_repeat{int(repeat)}_fold{int(fold)}"
    return {"stem": stem, "png": f"{stem}.png", "svg": f"{stem}.svg",
            "metadata": f"{stem}_metadata.json"}


# ------------------------------------------------------------- the metadata
def scaler_settings(scaler: StandardScaler) -> Dict[str, Any]:
    return {
        "class": "sklearn.preprocessing.StandardScaler",
        "with_mean": bool(scaler.with_mean),
        "with_std": bool(scaler.with_std),
        "n_features_in": int(scaler.n_features_in_),
        # int when no column had a NaN, per-feature array when one did; the
        # ravel copes with both rather than assuming the common case.
        "n_samples_seen": int(np.asarray(scaler.n_samples_seen_).ravel()[0]),
        "fitted_on": "training rows only",
    }


def pls_settings(pls: PLSRegression) -> Dict[str, Any]:
    return {
        "class": "sklearn.cross_decomposition.PLSRegression",
        "n_components": int(pls.n_components),
        "components_plotted": PLOT_DIMENSIONS,
        "scale": bool(pls.scale),
        "scale_note": "False because StandardScaler has already standardised "
                      "the columns on the training rows",
        "max_iter": int(pls.max_iter),
        "tol": float(pls.tol),
        "supervised": True,
        "supervised_note": "PLS uses the training labels, so training-class "
                           "separation is partly a property of the method; the "
                           "held-out points are what carry information",
        "fitted_on": "training rows only",
    }


def classifier_settings(classifier: LogisticRegression) -> Dict[str, Any]:
    return {
        "class": "sklearn.linear_model.LogisticRegression",
        "penalty": PLANE_PENALTY,
        "C": PLANE_C,
        "class_weight": PLANE_CLASS_WEIGHT,
        "solver": PLANE_SOLVER,
        "max_iter": PLANE_MAX_ITER,
        "tol": PLANE_TOL,
        "random_state": PLANE_RANDOM_STATE,
        "fitted_in": "the three-dimensional PLS training scores",
        "fitted_on": "training rows only",
        "coef": [float(v) for v in np.asarray(classifier.coef_).ravel()],
        "intercept": float(np.asarray(classifier.intercept_).ravel()[0]),
        "role": "retained as a diagnostic fitted in the three-dimensional training "
                "projection; the minimalist publication figure does not draw "
                "the decision plane and no accuracy is claimed from it",
    }


def build_metadata(state: int, repeat: int, fold: int, split: Split,
                   projection: Projection, y_train: np.ndarray,
                   y_held_out: np.ndarray, n_sd: float,
                   input_hashes: Dict[str, Any], split_record: Dict[str, Any],
                   names: Dict[str, str]) -> Dict[str, Any]:
    """Everything needed to argue with the figure, or to rebuild it."""
    y_train = np.asarray(y_train, dtype=int)
    y_held_out = np.asarray(y_held_out, dtype=int)
    return {
        "schema": METADATA_SCHEMA,
        "created_utc": utc_now(),
        "package_version": PACKAGE_VERSION,
        "figure_title": FIGURE_TITLE,

        "descriptive_only": True,
        "descriptive_only_note": (
            "This figure is descriptive. It re-displays one held-out fold of "
            "one repeat of the corrected run's frozen split and establishes "
            "nothing on its own; the reported evidence is the nested-selection "
            "estimate against the permutation null over all repeats."),
        "leakage_statement": LEAKAGE_STATEMENT,
        "ellipsoid_meaning": (
            f"{n_sd:g}-standard-deviation covariance ellipsoids of the training "
            f"scores; not confidence regions, not credible regions, and "
            f"carrying no coverage guarantee"),

        "state": int(state),
        "state_meaning": describe_state(int(state)),
        "repeat": int(repeat),
        "held_out_fold": int(fold),
        "n_components": int(projection.n_components),

        "counts": {
            "train": int(y_train.shape[0]),
            "held_out": int(y_held_out.shape[0]),
            "train_E": int((y_train == 1).sum()),
            "train_B": int((y_train == 0).sum()),
            "held_out_E": int((y_held_out == 1).sum()),
            "held_out_B": int((y_held_out == 0).sum()),
        },
        "row_ids": {
            "train": list(split.train_row_ids),
            "held_out": list(split.held_out_row_ids),
        },
        "row_id_overlap": sorted(set(split.train_row_ids)
                                 & set(split.held_out_row_ids)),

        "input_hashes": input_hashes,
        "split": split_record,

        "scaler": scaler_settings(projection.scaler),
        "pls": pls_settings(projection.pls),
        "classifier": classifier_settings(projection.classifier),

        "outputs": {"png": names["png"], "svg": names["svg"],
                    "metadata": names["metadata"]},
        "environment": environment_record(PACKAGE_VERSION),
    }


# ---------------------------------------------------------------- the figure
def _style_panes(ax, plotting, *, transparent: bool = False) -> None:
    """Recede the 3D box without removing spatial reference cues."""
    pane_colour = (1.0, 1.0, 1.0, 0.0) if transparent else (0.985, 0.989, 0.996, 1.0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        setter = getattr(axis, "set_pane_color", None)
        if callable(setter):
            setter(pane_colour)
        else:                                   # older spelling
            pane = getattr(axis, "pane", None)
            if pane is not None:
                pane.set_facecolor(pane_colour)
                pane.set_alpha(pane_colour[-1])
        line = getattr(axis, "line", None)
        if line is not None:
            line.set_color("#AEB8C7")
            line.set_linewidth(0.8)

    # mplot3d keeps its grid styling in private axis dictionaries on the
    # Matplotlib versions used by the project. Guard every access so the figure
    # still renders on versions that later expose a public API.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        info = getattr(axis, "_axinfo", None)
        if isinstance(info, dict) and "grid" in info:
            info["grid"].update({"color": (0.72, 0.76, 0.82, 0.34),
                                 "linewidth": 0.65,
                                 "linestyle": "-"})


def _draw_shadow_projection(ax, held_out_scores: np.ndarray,
                            y_held_out: np.ndarray,
                            limits: Sequence[Tuple[float, float]],
                            colours: Dict[str, str]) -> None:
    """Put faint held-out shadows on the floor to strengthen depth perception."""
    z_floor = float(limits[2][0])
    for label, value in (("B", 0), ("E", 1)):
        member = held_out_scores[y_held_out == value]
        if member.size == 0:
            continue
        ax.scatter(member[:, 0], member[:, 1],
                   np.full(member.shape[0], z_floor),
                   marker=MARKERS[label], s=15, color=colours[label],
                   edgecolors="none", alpha=0.10, depthshade=False, zorder=0)


def _draw_scene(ax, held_out_scores: np.ndarray, y_held_out: np.ndarray,
                geometry: Dict[str, Dict[str, Any]], projection: Projection,
                ellipsoids: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                limits: Sequence[Tuple[float, float]], colours: Dict[str, str],
                *, elev: float, azim: float, compact: bool,
                show_plane: bool, show_shadows: bool) -> None:
    """Draw one view of the same fitted geometry; fit nothing."""
    neutral = "#657084"
    ink = "#172033"

    if show_shadows:
        _draw_shadow_projection(ax, held_out_scores, y_held_out, limits, colours)

    plane = (decision_plane(projection.classifier.coef_,
                            projection.classifier.intercept_, limits)
             if show_plane else None)
    if plane is not None:
        ax.plot_surface(*plane, color=neutral, alpha=0.075,
                        linewidth=0, shade=False, antialiased=True, zorder=1)

    # Smooth translucent volumes plus a very light structural mesh. The mesh
    # gives the eye a 3D cue without implying that the surface is observed data.
    for label in ("B", "E"):
        surface = ellipsoids[label]
        ax.plot_surface(*surface, color=colours[label], alpha=0.115,
                        linewidth=0, shade=True, antialiased=True, zorder=2)
        if not compact:
            ax.plot_wireframe(*surface, rstride=5, cstride=5,
                              color=colours[label], alpha=0.10,
                              linewidth=0.45, zorder=2)

    for label, value in (("B", 0), ("E", 1)):
        member = held_out_scores[y_held_out == value]
        ax.scatter(member[:, 0], member[:, 1], member[:, 2],
                   marker=MARKERS[label], s=26 if compact else 48,
                   color=colours[label], edgecolors="#FFFFFF",
                   linewidths=0.55 if compact else 0.8, alpha=0.96,
                   depthshade=False, zorder=4)

    centroids = {label: np.asarray(geometry[label]["centroid"], dtype=np.float64)
                 for label in ("B", "E")}
    # A single connector makes the class displacement legible without turning
    # the plot into a distance claim or adding another fitted statistic.
    ax.plot([centroids["B"][0], centroids["E"][0]],
            [centroids["B"][1], centroids["E"][1]],
            [centroids["B"][2], centroids["E"][2]],
            color=ink, alpha=0.46, linewidth=1.25,
            linestyle=(0, (3, 3)), zorder=4)
    for label in ("B", "E"):
        centroid = centroids[label]
        ax.scatter([centroid[0]], [centroid[1]], [centroid[2]],
                   marker=CENTROID_MARKER, s=120 if compact else 245,
                   color=colours[label], edgecolors=ink,
                   linewidths=1.0 if compact else 1.45,
                   depthshade=False, zorder=5)
        if not compact:
            ax.text(centroid[0], centroid[1], centroid[2], f"  {label}",
                    color=ink, fontsize=9.2, fontweight="bold", zorder=6)

    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_box_aspect((1.14, 1.0, 0.92))
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_proj_type("ortho" if compact else "persp")
    except (AttributeError, TypeError):
        pass
    ax.grid(True)

    if compact:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.tick_params(length=0, pad=0)
    else:
        ax.set_xlabel(AXIS_LABELS[0], labelpad=12)
        ax.set_ylabel(AXIS_LABELS[1], labelpad=12)
        ax.set_zlabel(AXIS_LABELS[2], labelpad=10)
        ax.tick_params(axis="both", which="major", labelsize=8.5,
                       colors="#566174", pad=1.5)


def render_figure(figures_dir: Path, names: Dict[str, str],
                  held_out_scores: np.ndarray, y_held_out: np.ndarray,
                  geometry: Dict[str, Dict[str, Any]],
                  projection: Projection, state: int, repeat: int, fold: int,
                  n_sd: float = DEFAULT_N_SD, elev: float = 18.0,
                  azim: float = -58.0) -> List[Path]:
    """Draw a restrained, publication-style 3D figure.

    The composition deliberately echoes the visual grammar of strong
    representation-geometry figures -- small points, translucent class volumes,
    light 3D scaffolding and a compact legend -- without copying another
    figure's code or layout. Fits nothing; every input is already fitted.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from routegate import plotting

    plt = plotting.configure()                 # Agg: no display is ever needed
    held_out_scores = np.asarray(held_out_scores, dtype=np.float64)
    y_held_out = np.asarray(y_held_out, dtype=int)

    # A distinct but restrained green/coral pair. These are local to this
    # descriptive figure and do not alter the package-wide result colours.
    colours = {"B": "#2AA876", "E": "#E95D4F"}

    fig = plt.figure(figsize=(10.8, 8.25))
    ax = fig.add_subplot(111, projection="3d")

    ellipsoids = {
        label: ellipsoid_surface(
            geometry[label]["centroid"],
            geometry[label]["covariance"],
            n_sd=n_sd,
            resolution=18,
        )
        for label in ("B", "E")
    }
    drawn = [held_out_scores]
    drawn += [
        np.stack([g[0].ravel(), g[1].ravel(), g[2].ravel()], axis=1)
        for g in ellipsoids.values()
    ]
    limits = axis_limits(drawn, pad=0.09)

    # Training-derived covariance envelopes. Visible facet edges make the
    # volumes read as light geometric envelopes rather than glossy blobs.
    for label in ("B", "E"):
        ax.plot_surface(
            *ellipsoids[label],
            color=colours[label],
            alpha=0.17,
            linewidth=0.28,
            edgecolor=colours[label],
            antialiased=True,
            shade=False,
            rcount=18,
            ccount=18,
            zorder=1,
        )

    # Held-out rows only: small, crisp points like a paper figure rather than
    # oversized infographic markers.
    for label, value in (("B", 0), ("E", 1)):
        member = held_out_scores[y_held_out == value]
        ax.scatter(
            member[:, 0],
            member[:, 1],
            member[:, 2],
            marker=MARKERS[label],
            s=24,
            color=colours[label],
            edgecolors="white",
            linewidths=0.35,
            alpha=0.90,
            depthshade=False,
            zorder=3,
        )

    # Restrained centroid markers: small black diamonds, not large crosses.
    for label in ("B", "E"):
        centroid = np.asarray(geometry[label]["centroid"], dtype=np.float64)
        ax.scatter(
            [centroid[0]],
            [centroid[1]],
            [centroid[2]],
            marker=CENTROID_MARKER,
            s=38,
            color="#171717",
            edgecolors="white",
            linewidths=0.65,
            depthshade=False,
            zorder=4,
        )

    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_xlabel(AXIS_LABELS[0], labelpad=8)
    ax.set_ylabel(AXIS_LABELS[1], labelpad=8)
    ax.set_zlabel(AXIS_LABELS[2], labelpad=8)
    ax.set_box_aspect((1.08, 1.0, 0.88))
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True)
    _style_panes(ax, plotting)

    # Lighten the scaffolding without removing the useful 3D reference grid.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis._axinfo["grid"]["color"] = (0.78, 0.78, 0.78, 0.55)
            axis._axinfo["grid"]["linewidth"] = 0.65
        except (AttributeError, KeyError, TypeError):
            pass

    n_e = int((y_held_out == 1).sum())
    n_b = int((y_held_out == 0).sum())
    handles = [
        Line2D(
            [], [], linestyle="none", marker=MARKERS["B"],
            color=colours["B"], markersize=6.5, markeredgecolor="white",
            label=f"B  benign compliance  (n={n_b})",
        ),
        Line2D(
            [], [], linestyle="none", marker=MARKERS["E"],
            color=colours["E"], markersize=6.5, markeredgecolor="white",
            label=f"E  over-refusal  (n={n_e})",
        ),
        Patch(
            facecolor=colours["B"], edgecolor=colours["B"], alpha=0.20,
            label=f"training covariance envelopes  ({n_sd:g} sd)",
        ),
        Line2D(
            [], [], linestyle="none", marker=CENTROID_MARKER,
            color="#171717", markersize=5.3, markeredgecolor="white",
            label="training centroids",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.075, 0.835),
        frameon=True,
        framealpha=0.92,
        edgecolor="#D8D8D8",
        fontsize=8.8,
        handletextpad=0.7,
        borderpad=0.65,
    )

    # Compact paper-style title block: no large paragraph above the plot.
    fig.suptitle(
        FIGURE_TITLE,
        x=0.5,
        y=0.965,
        ha="center",
        color=plotting.INK,
        fontsize=17.0,
        fontweight="semibold",
    )
    descriptor = describe_state(int(state))
    fig.text(
        0.5,
        0.918,
        f"Qwen3-30B-A3B · {descriptor} · repeat {repeat}, held-out fold {fold}",
        ha="center",
        va="top",
        color=plotting.INK_SECONDARY,
        fontsize=10.0,
    )
    fig.text(
        0.5,
        0.878,
        f"Only held-out rows are plotted ({n_e} E, {n_b} B); "
        f"the scaler and supervised PLS basis were fitted on training rows only.",
        ha="center",
        va="top",
        color=plotting.INK_MUTED,
        fontsize=8.8,
    )
    fig.text(
        0.5,
        0.025,
        f"Descriptive visualisation only. Shaded volumes are "
        f"{n_sd:g}-standard-deviation training covariance envelopes, "
        f"not confidence regions; the reported evidence is the nested-selection "
        f"estimate and permutation null.",
        ha="center",
        va="bottom",
        color=plotting.INK_MUTED,
        fontsize=7.9,
        wrap=True,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.08)

    figures_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for key in ("png", "svg"):
        path = figures_dir / names[key]
        fig.savefig(path, dpi=240, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


# -------------------------------------------------------------------- driver
def run(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    assignments_path = Path(args.assignments or run_dir / ASSIGNMENTS_FILENAME)
    figures_dir = Path(args.output_dir or run_dir / FIGURES_DIRNAME)

    input_manifest = load_input_manifest()
    inputs = (load_development_inputs(paths_path=args.paths) if args.paths
              else load_development_inputs())
    print(f"[inputs] {inputs.n} analysed rows "
          f"(E {inputs.n_positive} / B {inputs.n_negative}), "
          f"{inputs.n_states} states")

    repeated = read_frozen_assignments(assignments_path, inputs.row_ids)
    print(f"[split] reusing {assignments_path.name}: {repeated.n_repeats} "
          f"repeat(s) x {repeated.n_splits} folds")
    split = select_split(repeated, int(args.repeat), int(args.fold))

    X = inputs.state_matrix(int(args.state))
    y = np.asarray(inputs.y, dtype=int)
    X_train, y_train = X[split.train_index], y[split.train_index]
    X_held_out, y_held_out = X[split.held_out_index], y[split.held_out_index]
    print(f"[split] repeat {args.repeat} fold {args.fold}: "
          f"{X_train.shape[0]} training, {X_held_out.shape[0]} held out "
          f"(E {int((y_held_out == 1).sum())} / "
          f"B {int((y_held_out == 0).sum())})")

    projection = fit_projection(X_train, y_train, int(args.n_components))
    held_out_scores = projection.transform(X_held_out)[:, :PLOT_DIMENSIONS]
    geometry = class_geometry(projection.train_scores_3d, y_train)
    print(f"[fit] scaler, PLS({args.n_components}) and plane fitted on "
          f"{X_train.shape[0]} training rows; held-out rows transformed only")

    names = output_names(int(args.state), int(args.repeat), int(args.fold))
    assignments_sha256 = sha256_file(assignments_path)
    metadata = build_metadata(
        state=int(args.state), repeat=int(args.repeat), fold=int(args.fold),
        split=split, projection=projection, y_train=y_train,
        y_held_out=y_held_out, n_sd=float(args.n_sd),
        input_hashes={
            "dataset": {"path": str(inputs.dataset_path),
                        "sha256": inputs.dataset_sha256,
                        "expected": input_manifest["dataset"]["sha256"]},
            "extraction_manifest": {
                "path": str(inputs.extraction_manifest_path),
                "sha256": inputs.extraction_manifest_sha256,
                "expected": input_manifest["extraction_manifest"]["sha256"]},
            "activation": {"path": str(inputs.activation_path),
                           "sha256": inputs.activation_sha256,
                           "expected": input_manifest["activation"]["sha256"],
                           "shape": list(inputs.activations.shape),
                           "dtype": inputs.activations.dtype.name},
            "cv_assignments": {"path": str(assignments_path),
                               "sha256": assignments_sha256},
        },
        split_record={
            "path": str(assignments_path),
            "sha256": assignments_sha256,
            "source": split.source,
            "n_repeats": int(repeated.n_repeats),
            "n_splits": int(repeated.n_splits),
            "base_seed": int(repeated.base_seed),
            "reused_not_regenerated": True,
            "note": "read back from the corrected run and re-bound by row_id; "
                    "no split is generated by this script",
        },
        names=names)

    figures_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = figures_dir / names["metadata"]
    write_json(metadata_path, metadata)

    written = render_figure(
        figures_dir, names, held_out_scores, y_held_out, geometry, projection,
        state=int(args.state), repeat=int(args.repeat), fold=int(args.fold),
        n_sd=float(args.n_sd), elev=float(args.elev), azim=float(args.azim))

    for path in written + [metadata_path]:
        print(f"[wrote] {path}")
    print("[done] descriptive figure only; no result was computed or changed")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", type=int, default=DEFAULT_STATE,
                        help=f"state to project (default {DEFAULT_STATE})")
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT,
                        help=f"repeat of the frozen split (default "
                             f"{DEFAULT_REPEAT})")
    parser.add_argument("--fold", type=int, default=DEFAULT_FOLD,
                        help=f"held-out fold (default {DEFAULT_FOLD})")
    parser.add_argument("--n-components", type=int, default=DEFAULT_COMPONENTS,
                        help=f"PLS components to fit (default "
                             f"{DEFAULT_COMPONENTS}; the figure always draws "
                             f"the first {PLOT_DIMENSIONS})")
    parser.add_argument("--n-sd", type=float, default=DEFAULT_N_SD,
                        help="standard deviations for the covariance "
                             "ellipsoids (not a confidence level)")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--assignments", type=Path, default=None,
                        help="cv_assignments.csv to reuse (default: the one in "
                             "--run-dir)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: <run-dir>/figures")
    parser.add_argument("--paths", type=Path, default=None)
    parser.add_argument("--elev", type=float, default=18.0)
    parser.add_argument("--azim", type=float, default=-58.0)
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        raise SystemExit(2)