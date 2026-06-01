from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_solve_optics_ordering,
    witness_extract_optics_dbscan,
)

@register_atom(witness_solve_optics_ordering, name="solve_optics_ordering")
@icontract.require(lambda X, min_samples, max_eps: min_samples >= 2, "Precondition failed: min_samples >= 2")
@icontract.ensure(lambda result, X, min_samples, max_eps: ordering.shape[0] == X.shape[0], "Postcondition failed: ordering.shape[0] == X.shape[0]")
def solve_optics_ordering(X: NDArray[np.float64], min_samples: int, max_eps: float = None) -> NDArray[np.int64]:
    """Traverse spatial points to establish hierarchical reachability and order listings.

    Args:
        X: NDArray[np.float64]
        min_samples: int
        max_eps: float

    Returns:
        ordering: NDArray[np.int64]
    """
    import sklearn.cluster
    return sklearn.cluster.OPTICS(X=X, min_samples=min_samples, max_eps=max_eps) # type: ignore

@register_atom(witness_extract_optics_dbscan, name="extract_optics_dbscan")
@icontract.require(lambda ordering, reachability, eps: eps > 0.0, "Precondition failed: eps > 0.0")
@icontract.ensure(lambda result, ordering, reachability, eps: labels.shape[0] == ordering.shape[0], "Postcondition failed: labels.shape[0] == ordering.shape[0]")
def extract_optics_dbscan(ordering: NDArray[np.int64], reachability: NDArray[np.float64], eps: float) -> NDArray[np.int32]:
    """Perform threshold cuts on the reachability order to retrieve static DBSCAN-like labels.

    Args:
        ordering: NDArray[np.int64]
        reachability: NDArray[np.float64]
        eps: float

    Returns:
        labels: NDArray[np.int32]
    """
    import sklearn.cluster
    return sklearn.cluster.cluster_optics_dbscan(ordering=ordering, reachability=reachability, eps=eps) # type: ignore

