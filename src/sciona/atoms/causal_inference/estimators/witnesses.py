"""Ghost witnesses for causal inference estimator patterns."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_symmetrized_prediction_fusion(
    predictions: AbstractArray,
) -> AbstractArray:
    """Ghost witness for symmetrized prediction fusion.

    Takes a 1D array of paired predictions, returns a 1D array of the
    same shape with antisymmetry enforced.
    """
    return AbstractArray(shape=predictions.shape, dtype="float64")


def witness_two_stage_independence_direction(
    independence_scores: AbstractArray,
    direction_scores: AbstractArray,
    symmetrize: bool = True,
) -> AbstractArray:
    """Ghost witness for two-stage independence-direction combination.

    Takes two equal-length 1D arrays, returns a 1D array of the same shape.
    """
    return AbstractArray(shape=independence_scores.shape, dtype="float64")


def witness_left_right_decomposed_prediction(
    left_scores: AbstractArray,
    right_scores: AbstractArray,
    symmetrize: bool = True,
) -> AbstractArray:
    """Ghost witness for left-right decomposed prediction.

    Takes two equal-length 1D arrays, returns a 1D array of the same shape.
    """
    return AbstractArray(shape=left_scores.shape, dtype="float64")


def witness_weighted_ensemble_combination(
    prediction_arrays: list[AbstractArray],
    weights: AbstractArray | None = None,
) -> AbstractArray:
    """Ghost witness for weighted ensemble combination.

    Takes a list of 1D arrays and optional weights. Returns a 1D array
    (if weighted) or a 2D stacked array (if unweighted).
    """
    if prediction_arrays:
        return AbstractArray(shape=prediction_arrays[0].shape, dtype="float64")
    return AbstractArray(shape=(0,), dtype="float64")
