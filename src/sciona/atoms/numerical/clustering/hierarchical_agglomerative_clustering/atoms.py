from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_linkage,
    witness_cut_tree,
)

@register_atom(witness_compute_linkage, name="compute_linkage")
@icontract.require(lambda X, method, metric: X.ndim == 2, "Precondition failed: X.ndim == 2")
@icontract.require(lambda X, method, metric: X.shape[0] >= 2, "Precondition failed: X.shape[0] >= 2")
@icontract.ensure(lambda result, X, method, metric: linkage_matrix.shape == (X.shape[0] - 1, 4), "Postcondition failed: linkage_matrix.shape == (X.shape[0] - 1, 4)")
def compute_linkage(X: NDArray[np.float64], method: str, metric: str = None) -> NDArray[np.float64]:
    """Calculate bottom-up merges to construct a hierarchical clustering dendrogram.

    Args:
        X: NDArray[np.float64]
        method: str
        metric: str

    Returns:
        linkage_matrix: NDArray[np.float64]
    """
    import scipy.cluster.hierarchy
    return scipy.cluster.hierarchy.linkage(X=X, method=method, metric=metric) # type: ignore

@register_atom(witness_cut_tree, name="cut_tree")
@icontract.require(lambda linkage_matrix, n_clusters: n_clusters > 0, "Precondition failed: n_clusters > 0")
@icontract.ensure(lambda result, linkage_matrix, n_clusters: labels.shape[0] == linkage_matrix.shape[0] + 1, "Postcondition failed: labels.shape[0] == linkage_matrix.shape[0] + 1")
def cut_tree(linkage_matrix: NDArray[np.float64], n_clusters: int) -> NDArray[np.int32]:
    """Cut the hierarchical tree at a specified depth to extract static labels.

    Args:
        linkage_matrix: NDArray[np.float64]
        n_clusters: int

    Returns:
        labels: NDArray[np.int32]
    """
    import scipy.cluster.hierarchy
    return scipy.cluster.hierarchy.fcluster(linkage_matrix=linkage_matrix, n_clusters=n_clusters) # type: ignore

