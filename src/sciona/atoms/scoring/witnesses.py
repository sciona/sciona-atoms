"""Ghost witnesses for scoring atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_elo_rating_update(
    rating_a: float,
    rating_b: float,
    score_a: float,
    k: float = 32.0,
) -> tuple[float, float]:
    """Ghost witness for Elo rating update."""
    return (rating_a, rating_b)


def witness_nab_anomaly_score(
    detections: AbstractArray,
    anomaly_windows: AbstractArray,
    A_tp: float = 1.0,
    A_fp: float = -0.11,
    A_fn: float = -1.0,
) -> float:
    """Ghost witness for NAB anomaly benchmark score."""
    return 0.0


def witness_expected_f1_threshold(
    probabilities: AbstractArray,
    n_true: int,
) -> tuple[AbstractArray, float]:
    """Ghost witness for expected F1 maximization."""
    return (probabilities, 0.5)


def witness_hierarchical_topdown_reconcile(
    bottom_forecasts: AbstractArray,
    proportions: AbstractArray,
    top_forecast: float,
) -> AbstractArray:
    """Ghost witness for top-down hierarchical reconciliation."""
    return bottom_forecasts


def witness_probability_weighted_adjustment(
    predictions: AbstractArray,
    probabilities: AbstractArray,
    shift: float = 0.0,
) -> AbstractArray:
    """Ghost witness for probability-weighted prediction adjustment."""
    if predictions.shape != probabilities.shape:
        raise ValueError("predictions and probabilities must have equal shape")
    return AbstractArray(shape=predictions.shape, dtype="float64")
