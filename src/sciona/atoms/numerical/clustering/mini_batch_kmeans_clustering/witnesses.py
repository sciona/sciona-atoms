from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_sample_batch(X: AbstractArray, batch_size: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for sample_batch."""
    _ = (X, batch_size)
    return AbstractArray(shape=X.shape, dtype=X.dtype)

def witness_update_batch_centroids(batch: AbstractArray, centroids: AbstractArray, counts: AbstractArray) -> AbstractArray:
    """Ghost witness for update_batch_centroids."""
    _ = (batch, centroids, counts)
    return AbstractArray(shape=batch.shape, dtype=batch.dtype)

