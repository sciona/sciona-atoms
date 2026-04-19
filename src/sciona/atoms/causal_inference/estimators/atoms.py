"""Prediction-stage patterns for pairwise causal direction estimation.

These atoms implement classifier-agnostic prediction patterns used in
causal direction scoring. They operate on raw prediction arrays from
any underlying classifier — the patterns themselves are what encode
domain-specific structural knowledge about the causal inference problem.

Derived from the 2nd-place solution to the Kaggle Cause-Effect Pairs
challenge (Fonollosa, 2013).

Source: https://github.com/jarfo/cause-effect (Apache 2.0)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_left_right_decomposed_prediction,
    witness_symmetrized_prediction_fusion,
    witness_two_stage_independence_direction,
    witness_weighted_ensemble_combination,
)


@register_atom(witness_symmetrized_prediction_fusion)
@icontract.require(lambda predictions: len(predictions) >= 2, "need at least one pair")
@icontract.require(lambda predictions: len(predictions) % 2 == 0, "predictions must contain paired rows (A,B),(B,A)")
@icontract.ensure(lambda result: len(result) > 0, "output must be non-empty")
def symmetrized_prediction_fusion(
    predictions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Enforce antisymmetry on paired causal direction predictions.

    For each adjacent pair of predictions (A→B, B→A), sets:
        score[A→B] = (pred[A→B] - pred[B→A]) / 2
        score[B→A] = -score[A→B]

    This enforces the physical constraint that score(A→B) = -score(B→A),
    doubling the effective data and removing any classifier bias toward
    consistently predicting positive or negative. Only the asymmetric
    signal survives.

    Expects predictions to be ordered as alternating pairs:
    [pred(A1→B1), pred(B1→A1), pred(A2→B2), pred(B2→A2), ...].

    Args:
        predictions: Raw classifier predictions, shape (2n,).

    Returns:
        Antisymmetrized predictions, shape (2n,).
    """
    result = predictions.copy()
    result[0::2] = (predictions[0::2] - predictions[1::2]) / 2.0
    result[1::2] = -result[0::2]
    return result


@register_atom(witness_two_stage_independence_direction)
@icontract.require(lambda independence_scores, direction_scores: len(independence_scores) == len(direction_scores), "scores must have equal length")
@icontract.require(lambda independence_scores: len(independence_scores) >= 2, "need at least one pair")
@icontract.ensure(lambda result: len(result) > 0, "output must be non-empty")
def two_stage_independence_direction(
    independence_scores: NDArray[np.float64],
    direction_scores: NDArray[np.float64],
    symmetrize: bool = True,
) -> NDArray[np.float64]:
    """Combine independence and direction scores via multiplicative gating.

    Decomposes causal inference into two stages:
    1. Independence: P(dependent) — are the variables related at all?
    2. Direction: which way does the causal arrow point?

    The final score is P(dependent) * direction_score. This means
    independent pairs get scores near 0 regardless of the direction
    classifier output — architecturally principled because you should
    not assign a causal direction to independent variables.

    When symmetrize=True, independence scores are averaged across each
    pair and direction scores are antisymmetrized.

    Args:
        independence_scores: P(dependent) from the independence classifier,
            shape (2n,). Values should be in [0, 1].
        direction_scores: Raw direction predictions, shape (2n,).
        symmetrize: Whether to enforce paired symmetry (default True).

    Returns:
        Combined causal scores, shape (2n,).
    """
    ind = independence_scores.copy()
    dir_ = direction_scores.copy()
    if symmetrize:
        ind[0::2] = (independence_scores[0::2] + independence_scores[1::2]) / 2.0
        ind[1::2] = ind[0::2]
        dir_[0::2] = (direction_scores[0::2] - direction_scores[1::2]) / 2.0
        dir_[1::2] = -dir_[0::2]
    return ind * dir_


@register_atom(witness_left_right_decomposed_prediction)
@icontract.require(lambda left_scores, right_scores: len(left_scores) == len(right_scores), "scores must have equal length")
@icontract.require(lambda left_scores: len(left_scores) >= 2, "need at least one pair")
@icontract.ensure(lambda result: len(result) > 0, "output must be non-empty")
def left_right_decomposed_prediction(
    left_scores: NDArray[np.float64],
    right_scores: NDArray[np.float64],
    symmetrize: bool = True,
) -> NDArray[np.float64]:
    """Combine left (X→Y) and right (Y→X) classifier scores by differencing.

    Trains two separate binary classifiers — one for "is X→Y?" and one for
    "is Y→X?" — then takes their difference as the causal score. This avoids
    forcing a single classifier to learn both the independence boundary and
    the direction boundary simultaneously.

    Args:
        left_scores: P(X→Y) from the left classifier, shape (2n,).
        right_scores: P(Y→X) from the right classifier, shape (2n,).
        symmetrize: Whether to enforce paired antisymmetry (default True).

    Returns:
        Causal direction scores, shape (2n,). Positive = X→Y.
    """
    result = left_scores - right_scores
    if symmetrize:
        result[0::2] = (result[0::2] - result[1::2]) / 2.0
        result[1::2] = -result[0::2]
    return result


@register_atom(witness_weighted_ensemble_combination)
@icontract.require(lambda prediction_arrays: len(prediction_arrays) > 0, "need at least one prediction array")
@icontract.require(
    lambda prediction_arrays, weights: weights is None or len(weights) == len(prediction_arrays),
    "weights must match prediction count",
)
@icontract.ensure(lambda result: len(result) > 0, "output must be non-empty")
def weighted_ensemble_combination(
    prediction_arrays: list[NDArray[np.float64]],
    weights: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Combine predictions from multiple estimators via weighted average.

    When weights are provided, computes the weighted dot product.
    When weights are None, returns the unweighted stack for downstream
    processing.

    The three estimator architectures (ID, Symmetric, OneStep) have
    different inductive biases, and their combination provides robustness
    against misspecification of any single model.

    Args:
        prediction_arrays: List of prediction arrays, each shape (n,).
        weights: Optional weight vector, shape (k,). If None, returns
            the stacked array.

    Returns:
        Combined predictions, shape (n,) if weighted, or (k, n) if unweighted.
    """
    stacked = np.array(prediction_arrays)
    if weights is not None:
        return np.dot(weights, stacked)
    return stacked
