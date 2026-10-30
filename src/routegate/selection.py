"""Choosing a state, and pricing the choice.

The pilot reported the best of 49 states, scored on the same rows the search ran
over. Two separate things are wrong with that, and this module holds the two
corrections.

``nested_state_selection``
    The state is chosen inside each outer training fold, by an inner
    cross-validation that never sees the outer held-out rows, and the chosen
    state is then scored on those held-out rows. What comes out is an estimate
    of "fit a probe at the state this procedure would pick", which is the
    quantity a reader assumes the headline is.

``permutation_null``
    Under a shuffled label vector the whole 49-state search is repeated and its
    maximum recorded, so the null distribution is a distribution of *maxima* and
    prices the selection. A null computed at an already-selected state answers a
    narrower question and is optimistic about the same result.

Both reuse the fixed feature matrices and the fixed split assignment; only the
labels move. Both exclude degenerate states from every maximum -- a constant
design matrix cannot win on merit, and ``np.argmax`` would happily return one.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from .controls import permuted_labels
from .probes import (
    OutOfFoldResult,
    cross_val_out_of_fold,
    fit_probe_on_fold,
    make_probe,
)
from .splits import FoldAssignment, RepeatedFoldAssignment, build_stratified_folds
from .validation import ValidationError, safe_argmax

# A state's feature matrix, by state index. The caller owns the cache, so the
# same matrices are reused across repeats, permutations and selection.
FeatureLookup = Callable[[int], np.ndarray]


def fold_roc_aucs(y: np.ndarray, result: OutOfFoldResult) -> List[Optional[float]]:
    """ROC-AUC within each held-out fold; ``None`` where a fold has one class."""
    y = np.asarray(y).astype(int)
    values: List[Optional[float]] = []
    for outcome in result.folds:
        held = y[outcome.test_index]
        if int((held == 1).sum()) == 0 or int((held == 0).sum()) == 0:
            values.append(None)
            continue
        values.append(float(roc_auc_score(held, outcome.scores)))
    return values


def mean_cross_validated_roc_auc(y: np.ndarray, result: OutOfFoldResult) -> float:
    """Mean of the per-fold ROC-AUCs -- the statistic the null is built from.

    Folds whose held-out rows happen to carry one class contribute nothing
    rather than a NaN. If no fold is scoreable the caller is told, not handed a
    number that means nothing.
    """
    values = [v for v in fold_roc_aucs(y, result) if v is not None]
    if not values:
        raise ValidationError(
            "no fold held out both classes, so no cross-validated ROC-AUC exists")
    return float(np.mean(values))


def _state_search(feature_for_state: FeatureLookup, y: np.ndarray,
                  folds: FoldAssignment, states: Sequence[int],
                  probe_factory, transform_factory) -> List[float]:
    """Mean cross-validated ROC-AUC at each candidate state."""
    scores = []
    for state in states:
        result = cross_val_out_of_fold(
            feature_for_state(state), y, folds, probe_factory=probe_factory,
            keep_fold_detail=True, transform_factory=transform_factory)
        scores.append(mean_cross_validated_roc_auc(y, result))
    return scores


# ------------------------------------------------------------ nested selection
@dataclass
class NestedFoldRecord:
    repeat: int
    outer_fold: int
    selected_state: int
    inner_mean_roc_auc: float
    inner_splits: int
    n_train: int
    n_held_out: int
    held_out_E: int
    held_out_B: int
    outer_fold_roc_auc: Optional[float]

    def as_dict(self) -> Dict[str, object]:
        record = asdict(self)
        if record["outer_fold_roc_auc"] is None:
            record["outer_fold_roc_auc"] = ""
        return record


@dataclass
class NestedSelectionResult:
    """Out-of-fold scores from probes whose state was chosen without them."""

    scores: np.ndarray
    fold_of_row: np.ndarray
    selected_state_of_row: np.ndarray
    records: List[NestedFoldRecord] = field(default_factory=list)

    @property
    def selected_states(self) -> List[int]:
        return [r.selected_state for r in self.records]


def nested_state_selection(feature_for_state: FeatureLookup, y: np.ndarray,
                           outer_folds: FoldAssignment,
                           states: Sequence[int],
                           inner_splits: int = 5,
                           seed: int = 20260805,
                           repeat: int = 0,
                           probe_factory=make_probe,
                           transform_factory=None) -> NestedSelectionResult:
    """Choose the state inside each outer training fold; score it outside.

    ``states`` is the candidate set, and it is the caller's job to have removed
    degenerate states from it -- they are still scored elsewhere, but a state
    that is constant by construction is not a candidate for selection.
    """
    y = np.asarray(y).astype(int)
    states = list(states)
    if not states:
        raise ValidationError("nested selection needs at least one candidate state")

    scores = np.full(y.shape[0], np.nan, dtype=np.float64)
    fold_of_row = np.full(y.shape[0], -1, dtype=int)
    selected_of_row = np.full(y.shape[0], -1, dtype=int)
    records: List[NestedFoldRecord] = []

    for k, (train_index, test_index) in enumerate(outer_folds.splits()):
        y_train = y[train_index]
        smallest = int(min((y_train == 1).sum(), (y_train == 0).sum()))
        if smallest < 2:
            raise ValidationError(
                f"outer fold {k} leaves {smallest} example(s) of a class, which "
                f"cannot support an inner cross-validation")
        used_inner = int(min(inner_splits, smallest))
        inner = build_stratified_folds(
            y_train, [str(int(i)) for i in train_index], n_splits=used_inner,
            seed=seed + 1000 * repeat + k)

        inner_scores = _state_search(
            lambda state: feature_for_state(state)[train_index], y_train, inner,
            states, probe_factory, transform_factory)
        best = safe_argmax(inner_scores, f"inner search, outer fold {k}")
        chosen = int(states[best])

        outcome = fit_probe_on_fold(
            feature_for_state(chosen), y, train_index, test_index, fold=k,
            probe_factory=probe_factory, keep_fold_detail=False,
            transform_factory=transform_factory)
        scores[test_index] = outcome.scores
        fold_of_row[test_index] = k
        selected_of_row[test_index] = chosen

        held = y[test_index]
        both_classes = int((held == 1).sum()) > 0 and int((held == 0).sum()) > 0
        records.append(NestedFoldRecord(
            repeat=int(repeat),
            outer_fold=k,
            selected_state=chosen,
            inner_mean_roc_auc=float(inner_scores[best]),
            inner_splits=used_inner,
            n_train=int(train_index.size),
            n_held_out=int(test_index.size),
            held_out_E=int((held == 1).sum()),
            held_out_B=int((held == 0).sum()),
            outer_fold_roc_auc=(float(roc_auc_score(held, outcome.scores))
                                if both_classes else None),
        ))

    if np.isnan(scores).any():
        raise ValidationError("a row was never held out by the outer folds")
    return NestedSelectionResult(scores=scores, fold_of_row=fold_of_row,
                                 selected_state_of_row=selected_of_row,
                                 records=records)


# ------------------------------------------------------------ permutation null
@dataclass
class PermutationRecord:
    permutation: int
    seed: Optional[int]
    is_observed: bool
    max_mean_roc_auc: float
    argmax_state: int
    n_states_searched: int

    def as_dict(self) -> Dict[str, object]:
        record = asdict(self)
        if record["seed"] is None:
            record["seed"] = ""
        return record


def search_maximum(feature_for_state: FeatureLookup, y: np.ndarray,
                   repeated: RepeatedFoldAssignment, states: Sequence[int],
                   probe_factory=make_probe, transform_factory=None
                   ) -> Dict[str, object]:
    """The whole state search, reduced to its maximum.

    Every candidate state is scored by mean cross-validated ROC-AUC, averaged
    over the repeats in ``repeated``, and the maximum is returned with the state
    that produced it. This is the statistic the permutation null is a
    distribution of, and it is computed by this one function for both the real
    labels and every permutation, so the two cannot drift apart.
    """
    states = list(states)
    per_state = []
    for state in states:
        X = feature_for_state(state)
        values = []
        for _, folds in repeated:
            result = cross_val_out_of_fold(
                X, y, folds, probe_factory=probe_factory, keep_fold_detail=True,
                transform_factory=transform_factory)
            values.append(mean_cross_validated_roc_auc(y, result))
        per_state.append(float(np.mean(values)))
    best = safe_argmax(per_state, "state search")
    return {"max_mean_roc_auc": float(per_state[best]),
            "argmax_state": int(states[best]),
            "by_state": {int(s): float(v) for s, v in zip(states, per_state)},
            "n_states_searched": len(states)}


def assert_permutation_is_usable(y_permuted: np.ndarray,
                                 repeated: RepeatedFoldAssignment,
                                 seed: int) -> None:
    """Refuse a shuffle that leaves a training fold with one class."""
    for repeat, folds in repeated:
        for train_index, _ in folds.splits():
            labels = y_permuted[train_index]
            if int((labels == 1).sum()) == 0 or int((labels == 0).sum()) == 0:
                raise ValidationError(
                    f"permutation seed {seed} leaves repeat {repeat} with a "
                    f"single-class training fold")


def permutation_search(feature_for_state: FeatureLookup, y: np.ndarray,
                       repeated: RepeatedFoldAssignment, states: Sequence[int],
                       seed: int, probe_factory=make_probe,
                       transform_factory=None) -> Dict[str, object]:
    """One permutation: shuffle the labels, repeat the whole state search.

    The features and the split assignment are exactly the ones the observed
    statistic used. Only the labels move, and they move deterministically with
    ``seed``, so a permutation can be re-run on its own and checked.
    """
    y_permuted = permuted_labels(y, seed=seed)
    assert_permutation_is_usable(y_permuted, repeated, seed)
    return search_maximum(feature_for_state, y_permuted, repeated, states,
                          probe_factory, transform_factory)


def permutation_null(feature_for_state: FeatureLookup, y: np.ndarray,
                     repeated: RepeatedFoldAssignment, states: Sequence[int],
                     n_permutations: int = 200,
                     base_seed: int = 20260805,
                     probe_factory=make_probe,
                     transform_factory=None,
                     runner: Optional[Callable[[Sequence[int]],
                                               List[Dict[str, object]]]] = None,
                     progress=None) -> Dict[str, object]:
    """A null distribution of *maxima*, for a state-selected result.

    The feature matrices and the split assignment are fixed; only the labels are
    shuffled, deterministically, by ``base_seed + permutation``. Each
    permutation repeats the entire state search and contributes its maximum, so
    the null carries the selection the headline number also carries.

    The p-value is the standard ``(1 + #{null >= observed}) / (1 + n)``, whose
    floor is ``1 / (1 + n)``: with 20 permutations that floor is 0.048, which is
    why the pilot's p = 0.0476 said less than it appeared to.

    ``runner`` schedules the permutations -- it is handed the seeds and must
    return one ``search_maximum`` outcome per seed, in order. The default runs
    them one after another; a caller with cores to spare can pass a parallel
    one. The science is the same either way: each permutation is an independent
    deterministic search over the same fixed matrices.
    """
    y = np.asarray(y).astype(int)
    states = list(states)
    if n_permutations < 1:
        raise ValidationError("a permutation null needs at least one permutation")

    observed = search_maximum(feature_for_state, y, repeated, states,
                              probe_factory, transform_factory)
    records = [PermutationRecord(
        permutation=-1, seed=None, is_observed=True,
        max_mean_roc_auc=observed["max_mean_roc_auc"],
        argmax_state=observed["argmax_state"],
        n_states_searched=observed["n_states_searched"])]

    seeds = [base_seed + index for index in range(n_permutations)]

    def serially(all_seeds: Sequence[int]) -> List[Dict[str, object]]:
        outcomes = []
        for index, seed in enumerate(all_seeds):
            outcome = permutation_search(feature_for_state, y, repeated, states,
                                         seed, probe_factory, transform_factory)
            outcomes.append(outcome)
            if progress is not None:
                progress(index, outcome["max_mean_roc_auc"])
        return outcomes

    outcomes = (runner or serially)(seeds)
    if len(outcomes) != n_permutations:
        raise ValidationError(
            f"the permutation runner returned {len(outcomes)} outcomes for "
            f"{n_permutations} permutations")

    null_values: List[float] = []
    for index, (seed, outcome) in enumerate(zip(seeds, outcomes)):
        null_values.append(float(outcome["max_mean_roc_auc"]))
        records.append(PermutationRecord(
            permutation=index, seed=seed, is_observed=False,
            max_mean_roc_auc=float(outcome["max_mean_roc_auc"]),
            argmax_state=int(outcome["argmax_state"]),
            n_states_searched=int(outcome["n_states_searched"])))

    null = np.asarray(null_values, dtype=float)
    at_least_as_extreme = int((null >= observed["max_mean_roc_auc"]).sum())
    return {
        "observed": observed,
        "records": records,
        "null_values": null_values,
        "n_permutations": n_permutations,
        "n_at_least_as_extreme": at_least_as_extreme,
        "p_value": float((1 + at_least_as_extreme) / (1 + n_permutations)),
        "p_value_floor": float(1.0 / (1 + n_permutations)),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)) if null.size > 1 else 0.0,
        "null_max": float(null.max()),
        "null_quantile_95": float(np.quantile(null, 0.95)),
        "statistic": "maximum over searched states of the mean cross-validated "
                     "ROC-AUC, averaged over repeats",
    }
