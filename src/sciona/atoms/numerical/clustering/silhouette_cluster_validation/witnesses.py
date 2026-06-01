from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_silhouette_samples(X: AbstractArray, labels: AbstractArray) -> AbstractArray:
    """Ghost witness for compute_silhouette_samples."""
    _ = (X, labels)
    return AbstractArray(shape=X.shape, dtype=X.dtype)

def witness_aggregate_mean_score(sample_scores: AbstractArray) -> AbstractScalar:
    """Ghost witness for aggregate_mean_score."""
    _ = (sample_scores)
    return AbstractScalar(dtype="float64")

