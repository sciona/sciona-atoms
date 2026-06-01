from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_find_core_points,
    witness_propagate_labels,
)

@register_atom(witness_find_core_points, name="find_core_points")
@icontract.require(lambda neighborhood_counts, min_samples: min_samples >= 1, "Precondition failed: min_samples >= 1")
@icontract.ensure(lambda result, neighborhood_counts, min_samples: core_mask.shape == neighborhood_counts.shape, "Postcondition failed: core_mask.shape == neighborhood_counts.shape")
def find_core_points(neighborhood_counts: NDArray[np.int64], min_samples: int) -> NDArray[np.bool_]:
    """Verify local neighbor density to classify points as core.

    Args:
        neighborhood_counts: NDArray[np.int64]
        min_samples: int

    Returns:
        core_mask: NDArray[np.bool_]
    """
    import sklearn.cluster
    return sklearn.cluster.DBSCAN(neighborhood_counts=neighborhood_counts, min_samples=min_samples) # type: ignore

@register_atom(witness_propagate_labels, name="propagate_labels")
@icontract.require(lambda neighbors, core_mask: neighbors.shape[0] == core_mask.shape[0], "Precondition failed: neighbors.shape[0] == core_mask.shape[0]")
@icontract.ensure(lambda result, neighbors, core_mask: labels.shape[0] == core_mask.shape[0], "Postcondition failed: labels.shape[0] == core_mask.shape[0]")
def propagate_labels(neighbors: Any, core_mask: NDArray[np.bool_]) -> NDArray[np.int32]:
    """Perform graph search connectivity traversal to group core components and attach border points.

    Args:
        neighbors: scipy.sparse.csr_matrix
        core_mask: NDArray[np.bool_]

    Returns:
        labels: NDArray[np.int32]
    """
    import sklearn.cluster
    return sklearn.cluster.DBSCAN(neighbors=neighbors, core_mask=core_mask) # type: ignore

