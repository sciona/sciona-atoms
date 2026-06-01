from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_silhouette_samples,
    witness_aggregate_mean_score,
)

@register_atom(witness_compute_silhouette_samples, name="compute_silhouette_samples")
@icontract.require(lambda X, labels: X.ndim == 2, "Precondition failed: X.ndim == 2")
@icontract.require(lambda X, labels: labels.ndim == 1, "Precondition failed: labels.ndim == 1")
@icontract.require(lambda X, labels: len(np.unique(labels)) >= 2, "Precondition failed: len(np.unique(labels)) >= 2")
@icontract.ensure(lambda result, X, labels: sample_scores.shape[0] == X.shape[0], "Postcondition failed: sample_scores.shape[0] == X.shape[0]")
@icontract.ensure(lambda result, X, labels: np.all(sample_scores >= -1.0), "Postcondition failed: np.all(sample_scores >= -1.0)")
@icontract.ensure(lambda result, X, labels: np.all(sample_scores <= 1.0), "Postcondition failed: np.all(sample_scores <= 1.0)")
def compute_silhouette_samples(X: NDArray[np.float64], labels: NDArray[np.int32]) -> NDArray[np.float64]:
    """Calculate Rousseeuw silhouette coefficients for every individual point.

    Args:
        X: NDArray[np.float64]
        labels: NDArray[np.int32]

    Returns:
        sample_scores: NDArray[np.float64]
    """
    import sklearn.metrics
    return sklearn.metrics.silhouette_samples(X=X, labels=labels) # type: ignore

@register_atom(witness_aggregate_mean_score, name="aggregate_mean_score")
@icontract.require(lambda sample_scores: sample_scores.ndim == 1, "Precondition failed: sample_scores.ndim == 1")
@icontract.ensure(lambda result, sample_scores: mean_score >= -1.0, "Postcondition failed: mean_score >= -1.0")
@icontract.ensure(lambda result, sample_scores: mean_score <= 1.0, "Postcondition failed: mean_score <= 1.0")
def aggregate_mean_score(sample_scores: NDArray[np.float64]) -> float:
    """Compute the average silhouette score across all samples.

    Args:
        sample_scores: NDArray[np.float64]

    Returns:
        mean_score: float
    """
    import sklearn.metrics
    return sklearn.metrics.silhouette_score(sample_scores=sample_scores) # type: ignore

