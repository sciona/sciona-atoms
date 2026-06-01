from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_sample_batch,
    witness_update_batch_centroids,
)

@register_atom(witness_sample_batch, name="sample_batch")
@icontract.require(lambda X, batch_size: batch_size <= X.shape[0], "Precondition failed: batch_size <= X.shape[0]")
@icontract.ensure(lambda result, X, batch_size: batch.shape[0] == batch_size, "Postcondition failed: batch.shape[0] == batch_size")
def sample_batch(X: NDArray[np.float64], batch_size: int) -> NDArray[np.float64]:
    """Draw a random row slice of specified batch size from the dataset.

    Args:
        X: NDArray[np.float64]
        batch_size: int

    Returns:
        batch: NDArray[np.float64]
    """
    import sklearn.cluster
    return sklearn.cluster.MiniBatchKMeans(X=X, batch_size=batch_size) # type: ignore

@register_atom(witness_update_batch_centroids, name="update_batch_centroids")
@icontract.require(lambda batch, centroids, counts: batch.shape[1] == centroids.shape[1], "Precondition failed: batch.shape[1] == centroids.shape[1]")
@icontract.ensure(lambda result, batch, centroids, counts: updated_centroids.shape == centroids.shape, "Postcondition failed: updated_centroids.shape == centroids.shape")
def update_batch_centroids(batch: NDArray[np.float64], centroids: NDArray[np.float64], counts: NDArray[np.int64]) -> NDArray[np.float64]:
    """Update centroid coordinate averages incrementally based on batch assignments.

    Args:
        batch: NDArray[np.float64]
        centroids: NDArray[np.float64]
        counts: NDArray[np.int64]

    Returns:
        updated_centroids: NDArray[np.float64]
    """
    import sklearn.cluster
    return sklearn.cluster.MiniBatchKMeans(batch=batch, centroids=centroids, counts=counts) # type: ignore

