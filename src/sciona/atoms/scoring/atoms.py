"""Scoring and evaluation primitives in pure numpy.

Implements reusable scoring, thresholding, and reconciliation utilities
for competition pipelines: Elo rating systems, anomaly benchmark scoring,
F1-optimal thresholding, and hierarchical forecast reconciliation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_elo_rating_update,
    witness_expected_f1_threshold,
    witness_hierarchical_topdown_reconcile,
    witness_nab_anomaly_score,
)


@register_atom(witness_elo_rating_update)
@icontract.require(lambda k: k > 0.0, "k must be positive")
@icontract.require(lambda score_a: 0.0 <= score_a <= 1.0, "score_a must be in [0, 1]")
@icontract.ensure(
    lambda rating_a, rating_b, result: abs((result[0] + result[1]) - (rating_a + rating_b)) < 1e-9,
    "total rating must be conserved",
)
def elo_rating_update(
    rating_a: float,
    rating_b: float,
    score_a: float,
    k: float = 32.0,
) -> tuple[float, float]:
    """Update Elo ratings after a match. score_a=1 for win, 0.5 draw, 0 loss."""
    expected_a = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    expected_b = 1.0 - expected_a
    new_a = rating_a + k * (score_a - expected_a)
    new_b = rating_b + k * ((1.0 - score_a) - expected_b)
    return new_a, new_b


@register_atom(witness_nab_anomaly_score)
@icontract.require(lambda detections: detections.ndim == 1, "detections must be 1-D")
@icontract.require(
    lambda anomaly_windows: anomaly_windows.ndim == 2 and anomaly_windows.shape[1] == 2,
    "anomaly_windows must have shape (n_windows, 2) with [start, end] pairs",
)
@icontract.ensure(lambda result: np.isfinite(result), "score must be finite")
def nab_anomaly_score(
    detections: NDArray[np.float64],
    anomaly_windows: NDArray[np.float64],
    A_tp: float = 1.0,
    A_fp: float = -0.11,
    A_fn: float = -1.0,
) -> float:
    """NAB anomaly benchmark score with early detection reward.

    Scores detections against labeled anomaly windows using a sigmoid-weighted
    reward for true positives (earlier detection scores higher), a fixed
    penalty for false positives, and a penalty for missed windows.
    """
    n_windows = anomaly_windows.shape[0]
    score = 0.0

    window_detected = np.zeros(n_windows, dtype=bool)

    for det in detections:
        det_val = float(det)
        matched = False
        for w_idx in range(n_windows):
            w_start = float(anomaly_windows[w_idx, 0])
            w_end = float(anomaly_windows[w_idx, 1])
            if w_start <= det_val <= w_end:
                if not window_detected[w_idx]:
                    window_detected[w_idx] = True
                    # Sigmoid-weighted: earlier in window scores higher
                    window_len = max(w_end - w_start, 1e-12)
                    relative_pos = (det_val - w_start) / window_len
                    # Sigmoid: 2 / (1 + exp(5 * relative_pos)) gives higher
                    # reward for earlier detections
                    sigmoid_weight = 2.0 / (1.0 + np.exp(5.0 * relative_pos))
                    score += A_tp * sigmoid_weight
                matched = True
                break
        if not matched:
            score += A_fp

    # Penalty for undetected windows
    n_missed = int(np.sum(~window_detected))
    score += A_fn * n_missed

    return float(score)


@register_atom(witness_expected_f1_threshold)
@icontract.require(lambda probabilities: probabilities.ndim == 1, "probabilities must be 1-D")
@icontract.require(
    lambda probabilities: np.all((probabilities >= 0.0) & (probabilities <= 1.0)),
    "probabilities must be in [0, 1]",
)
@icontract.require(lambda n_true: n_true >= 0, "n_true must be non-negative")
@icontract.ensure(lambda result: 0.0 <= result[1] <= 1.0, "threshold must be in [0, 1]")
def expected_f1_threshold(
    probabilities: NDArray[np.float64],
    n_true: int,
) -> tuple[NDArray[np.float64], float]:
    """Faron's expected F1 maximization via probability thresholding.

    Sort probabilities descending and find the number of predictions k
    that maximizes the expected F1 = 2k / (k + n_true). Returns binary
    predictions and the optimal threshold.
    """
    sorted_probs = np.sort(probabilities)[::-1]
    n = len(sorted_probs)

    best_f1 = 0.0
    best_k = 0
    cumsum = 0.0
    for k in range(1, n + 1):
        cumsum += sorted_probs[k - 1]
        expected_f1 = 2.0 * cumsum / (k + n_true) if (k + n_true) > 0 else 0.0
        if expected_f1 > best_f1:
            best_f1 = expected_f1
            best_k = k

    if best_k == 0:
        threshold = 1.0
    else:
        threshold = float(sorted_probs[best_k - 1])

    predictions = (probabilities >= threshold).astype(np.float64)
    return predictions, threshold


@register_atom(witness_hierarchical_topdown_reconcile)
@icontract.require(
    lambda bottom_forecasts: bottom_forecasts.ndim == 1,
    "bottom_forecasts must be 1-D",
)
@icontract.require(
    lambda bottom_forecasts, proportions: proportions.shape == bottom_forecasts.shape,
    "proportions must match bottom_forecasts shape",
)
@icontract.ensure(
    lambda result, bottom_forecasts: result.shape == bottom_forecasts.shape,
    "result must preserve shape",
)
def hierarchical_topdown_reconcile(
    bottom_forecasts: NDArray[np.float64],
    proportions: NDArray[np.float64],
    top_forecast: float,
) -> NDArray[np.float64]:
    """Top-down proportional hierarchical forecast reconciliation.

    Distributes a top-level aggregate forecast to bottom-level series
    using historical proportions, ensuring the reconciled forecasts
    sum to the top forecast.
    """
    return proportions * top_forecast
