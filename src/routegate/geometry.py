"""Descriptive geometry of the residual stream.

Everything here describes the analysed rows; none of it is a held-out estimate,
and none of it feeds a reported performance number. It exists so that the
question "what does the separation look like?" can be answered alongside "how
well can it be read out?".

Degenerate states are handled explicitly rather than allowed to produce a NaN
that then propagates into a summary. On this corpus state 0 -- the embedding
output at the final prompt token -- is one identical vector for every prompt, so
its centroid separation is exactly zero and its standardised separation is
undefined; that is reported as undefined, and excluded from maxima.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .validation import ValidationError


@dataclass
class CentroidSeparation:
    """How far apart the two class means sit at one state."""

    state: int
    centroid_distance: float
    pooled_within_class_sd: float
    standardised_distance: Optional[float]   # None when undefined
    mean_norm_E: float
    mean_norm_B: float
    centroid_cosine: Optional[float]
    degenerate: bool

    def as_dict(self) -> Dict[str, object]:
        record = asdict(self)
        for key in ("standardised_distance", "centroid_cosine"):
            if record[key] is None:
                record[key] = ""      # undefined, written as empty, never NaN
        return record


def centroid_separation(state_matrix: np.ndarray, y: np.ndarray,
                        state: int = -1) -> CentroidSeparation:
    """Class-mean separation at one state, in raw activation space."""
    X = np.asarray(state_matrix, dtype=np.float64)
    y = np.asarray(y).astype(int)
    if X.shape[0] != y.shape[0]:
        raise ValidationError("design matrix and labels disagree on row count")
    n = X.shape[0]
    if int((y == 1).sum()) == 0 or int((y == 0).sum()) == 0:
        raise ValidationError("both classes are required for a centroid contrast")

    centroid_e = X[y == 1].mean(axis=0)
    centroid_b = X[y == 0].mean(axis=0)
    distance = float(np.linalg.norm(centroid_e - centroid_b))

    within_e = X[y == 1] - centroid_e
    within_b = X[y == 0] - centroid_b
    pooled_variance = (float((within_e ** 2).sum()) +
                       float((within_b ** 2).sum())) / max(n - 2, 1)
    pooled_sd = float(np.sqrt(pooled_variance))

    degenerate = bool(float(X.std(axis=0).max()) == 0.0)
    standardised = None if pooled_sd == 0.0 else float(distance / pooled_sd)

    norm_e = float(np.linalg.norm(centroid_e))
    norm_b = float(np.linalg.norm(centroid_b))
    cosine = (None if norm_e == 0.0 or norm_b == 0.0
              else float(np.dot(centroid_e, centroid_b) / (norm_e * norm_b)))

    return CentroidSeparation(
        state=int(state),
        centroid_distance=distance,
        pooled_within_class_sd=pooled_sd,
        standardised_distance=standardised,
        mean_norm_E=float(np.linalg.norm(X[y == 1], axis=1).mean()),
        mean_norm_B=float(np.linalg.norm(X[y == 0], axis=1).mean()),
        centroid_cosine=cosine,
        degenerate=degenerate,
    )


def cosine_similarity_matrix(directions: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between row vectors.

    A zero row (a degenerate state's direction) yields zeros rather than NaN.
    """
    matrix = np.asarray(directions, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    unit = matrix / norms
    return np.clip(unit @ unit.T, -1.0, 1.0)


def fit_projection(train_matrix: np.ndarray, n_components: int = 2,
                   random_state: int = 20260805) -> Tuple[StandardScaler, PCA]:
    """Fit standardisation and PCA on training rows only.

    Returned so the caller must explicitly apply them to held-out rows; there is
    no path that fits on the rows it then displays as held out.
    """
    scaler = StandardScaler().fit(np.asarray(train_matrix, dtype=np.float64))
    projector = PCA(n_components=n_components, svd_solver="full",
                    random_state=random_state)
    projector.fit(scaler.transform(np.asarray(train_matrix, dtype=np.float64)))
    return scaler, projector


def project(matrix: np.ndarray, scaler: StandardScaler, projector: PCA) -> np.ndarray:
    """Project rows into a basis fitted elsewhere."""
    return projector.transform(scaler.transform(np.asarray(matrix, dtype=np.float64)))


def separation_table(records: Sequence[CentroidSeparation]) -> List[Dict[str, object]]:
    return [r.as_dict() for r in records]
