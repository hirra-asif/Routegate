"""Control feature sets, for asking what a headline number is actually made of.

Nothing here is executed by the pilot reproduction. The reproduction's only job
is to recover the frozen numbers; controls belong to the corrected analysis that
follows it.

Each builder returns a design matrix over the same analysed rows, in the same
order, so a control can be run through the same probe and the same folds as the
residual-stream probe and the results are directly comparable.

Four controls are provided, in increasing order of how much they threaten a
naive reading of the headline:

``prompt_length_features``
    Character and whitespace-token counts. A surface-form control: if prompt
    length alone separated the classes, the probe would be reading formatting.

``source_family_features``
    One-hot benchmark provenance. The classes are not drawn from the source
    benchmarks in equal proportion, so this measures what provenance alone buys.

``activation_norm_features``
    The L2 norm of the residual vector, one scalar, no direction. This is the
    sharpest control: separability it recovers is magnitude, not geometry.

``unit_l2_normalised_states``
    The residual vectors projected onto the unit sphere, which removes the
    magnitude the previous control isolates. What survives is directional.

Two of these are also *activation variants*: whole feature sets that the probe
is run over at every state, beside the raw one (``ACTIVATION_VARIANTS``). Where
a transformation needs to be fitted -- norm-residualisation does, unit-L2
normalisation does not -- it is fitted on the training rows of each fold and
applied unchanged to the held-out rows, and the whole-sample forms kept here are
marked as descriptive.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .validation import ValidationError


PROMPT_LENGTH_FEATURE_NAMES = ("prompt_characters", "prompt_whitespace_tokens")

PROMPT_LENGTH_NOTE = (
    "surface text statistics only: character count and whitespace-token count. "
    "The extraction manifest saved no token count, and this package neither "
    "loads a tokeniser nor retokenises a prompt, so no model-token-length "
    "control exists and none is reported as one.")


def prompt_length_features(prompt_text: Sequence[str],
                           include_words: bool = True) -> np.ndarray:
    """[n, 1] or [n, 2]: character count, optionally whitespace-token count.

    Not a token count. Whitespace tokens are a surface statistic of the prompt
    string; the model's tokeniser is never loaded and no prompt is retokenised.
    Where a saved token count exists it should be used instead -- on this
    extraction none does, which is recorded by the input gate
    ``saved_prompt_token_count_columns`` and repeated wherever this is reported.
    """
    chars = np.array([[len(t)] for t in prompt_text], dtype=np.float64)
    if not include_words:
        return chars
    words = np.array([[len(t.split())] for t in prompt_text], dtype=np.float64)
    return np.hstack([chars, words])


def categorical_column(values: Sequence[str]) -> np.ndarray:
    """[n, 1] of raw category strings, for a fold-fitted one-hot encoder.

    The encoding itself belongs inside the pipeline (see
    ``probes.make_categorical_probe``), so the categories are learned from the
    training rows of each fold and a category seen only in the held-out rows is
    ignored rather than silently given a column.
    """
    return np.asarray(list(values), dtype=object).reshape(-1, 1)


def source_family_features(source_family: Sequence[str]
                           ) -> Tuple[np.ndarray, List[str]]:
    """[n, k] one-hot over the distinct source families, plus their names.

    Descriptive use only -- the categories come from all the rows. Inside
    cross-validation use ``categorical_column`` with a fold-fitted encoder.
    """
    families = sorted(set(source_family))
    if len(families) < 2:
        raise ValidationError(
            f"one-hot source control needs at least two families, got {families}")
    matrix = np.zeros((len(source_family), len(families)), dtype=np.float64)
    for i, family in enumerate(source_family):
        matrix[i, families.index(family)] = 1.0
    return matrix, families


def activation_norm_features(state_matrix: np.ndarray) -> np.ndarray:
    """[n, 1]: the L2 norm of each row. Magnitude with the direction discarded."""
    return np.linalg.norm(np.asarray(state_matrix, dtype=np.float64),
                          axis=1).reshape(-1, 1)


def unit_l2_normalised_states(state_matrix: np.ndarray) -> np.ndarray:
    """[n, d]: each row scaled to unit L2 norm. Direction with magnitude removed.

    A zero row -- which is what a fully degenerate state produces after
    centring -- is left as zero rather than turned into a division by zero.
    """
    matrix = np.asarray(state_matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def fit_norm_residualiser(train_matrix: np.ndarray) -> np.ndarray:
    """[2, d] coefficients of each column on ``[1, ||x||]``, from training rows.

    Norm-residualisation is *not* sample-local: it needs a regression, and a
    regression fitted on all the rows would carry held-out information into
    every fold. The fit therefore happens on training rows only, and the
    coefficients are applied unchanged to the held-out rows.
    """
    matrix = np.asarray(train_matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1)
    design = np.column_stack([np.ones_like(norms), norms])
    coefficients, *_ = np.linalg.lstsq(design, matrix, rcond=None)
    return coefficients


def apply_norm_residualiser(matrix: np.ndarray,
                            coefficients: np.ndarray) -> np.ndarray:
    """[n, d]: rows with the training-fitted dependence on ``||x||`` removed."""
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1)
    design = np.column_stack([np.ones_like(norms), norms])
    return matrix - design @ np.asarray(coefficients, dtype=np.float64)


def norm_residualised_states(state_matrix: np.ndarray) -> np.ndarray:
    """[n, d]: each column with its linear dependence on ``||x||`` removed.

    A stricter magnitude control than unit normalisation: it strips the
    component of every feature that is predictable from the row norm, leaving
    the part of the geometry that magnitude cannot explain.

    Descriptive use only -- it fits on the rows it transforms. Inside
    cross-validation use ``norm_residualiser_factory``, which fits per fold.
    """
    return apply_norm_residualiser(state_matrix,
                                   fit_norm_residualiser(state_matrix))


# ------------------------------------------------------- activation variants
# Four feature sets over the same state, run through the same probe and the
# same folds, so the magnitude component and the directional component of a
# headline number can be told apart.
ACTIVATION_VARIANTS = ("raw", "unit_l2", "norm_residualised", "activation_norm")

VARIANT_DESCRIPTIONS: Dict[str, str] = {
    "raw": "the residual vector as extracted; direction and magnitude together",
    "unit_l2": "each row scaled to unit L2 norm before standardisation; a "
               "sample-local transformation, so it uses no information from any "
               "other row and cannot leak across a fold boundary",
    "norm_residualised": "each feature with its training-fold linear dependence "
                         "on ||x|| removed; the regression is fitted on training "
                         "rows only",
    "activation_norm": "||x||_2 alone, one scalar per example, no direction at "
                       "all; its scaler and classifier are fitted on training "
                       "rows only",
}

# Variants that are a function of one row and nothing else. These can be
# computed once for the whole state matrix: doing so is arithmetically
# identical to doing it inside each fold, and it cannot leak.
SAMPLE_LOCAL_VARIANTS = ("raw", "unit_l2", "activation_norm")


def variant_matrix(name: str, state_matrix: np.ndarray) -> np.ndarray:
    """The design matrix for a sample-local variant, computed once per state."""
    if name not in SAMPLE_LOCAL_VARIANTS:
        raise ValidationError(
            f"{name!r} is not sample-local; use variant_transform_factory")
    matrix = np.asarray(state_matrix, dtype=np.float64)
    if name == "raw":
        return matrix
    if name == "unit_l2":
        return unit_l2_normalised_states(matrix)
    return activation_norm_features(matrix)


def variant_transform_factory(name: str):
    """A ``(X, train_index) -> transform`` factory, or ``None`` if sample-local.

    The returned transform is fitted on the training rows of one fold and then
    applied to both the training and the held-out rows of that fold.
    """
    if name in SAMPLE_LOCAL_VARIANTS:
        return None
    if name != "norm_residualised":
        raise ValidationError(f"unknown activation variant {name!r}")

    def factory(X: np.ndarray, train_index: np.ndarray):
        coefficients = fit_norm_residualiser(np.asarray(X)[train_index])
        return lambda rows: apply_norm_residualiser(rows, coefficients)

    return factory


def variant_description(name: str) -> str:
    if name not in VARIANT_DESCRIPTIONS:
        raise ValidationError(f"unknown activation variant {name!r}")
    return VARIANT_DESCRIPTIONS[name]


def permuted_labels(y: np.ndarray, seed: int) -> np.ndarray:
    """A deterministic label permutation, for building a null distribution.

    A permutation null for a layer-selected result must repeat the whole state
    search per permutation and take its maximum; permuting at an
    already-selected state answers a narrower question.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    return y[rng.permutation(y.shape[0])]


def control_catalogue() -> Dict[str, str]:
    """What each control is for, for the run record."""
    return {
        "prompt_length": PROMPT_LENGTH_NOTE,
        "source_family": "one-hot benchmark provenance; the known composition confound",
        "topic": "one-hot standardised topic, fitted per training fold",
        "task_frame": "one-hot task frame, fitted per training fold",
        "activation_norm": "||x||_2 at one state, a single scalar; magnitude without direction",
        "unit_l2_normalised": "residual vectors on the unit sphere; direction without magnitude",
        "norm_residualised": "features with their linear dependence on ||x|| removed",
        "label_permutation": "deterministic label shuffle for an empirical null",
    }
