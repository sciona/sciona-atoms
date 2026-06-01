from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_build_ball_tree,
    witness_query_ball_tree,
)

@register_atom(witness_build_ball_tree, name="build_ball_tree")
@icontract.require(lambda data, metric: data.ndim == 2, "Precondition failed: data.ndim == 2")
@icontract.require(lambda data, metric: data.shape[0] > 0, "Precondition failed: data.shape[0] > 0")
@icontract.ensure(lambda result, data, metric: tree is not None, "Postcondition failed: tree is not None")
def build_ball_tree(data: NDArray[np.float64], metric: str = None) -> Any:
    """Construct a spatial BallTree metric index.

    Args:
        data: NDArray[np.float64]
        metric: Must be a valid distance metric

    Returns:
        tree: sklearn.neighbors.BallTree
    """
    import sklearn.neighbors
    return sklearn.neighbors.BallTree(data=data, metric=metric) # type: ignore

@register_atom(witness_query_ball_tree, name="query_ball_tree")
@icontract.require(lambda tree, query_points, k: k >= 1, "Precondition failed: k >= 1")
@icontract.require(lambda tree, query_points, k: query_points.shape[-1] == tree.data.shape[1], "Precondition failed: query_points.shape[-1] == tree.data.shape[1]")
@icontract.ensure(lambda result, tree, query_points, k: distances.shape == indices.shape, "Postcondition failed: distances.shape == indices.shape")
def query_ball_tree(tree: Any, query_points: NDArray[np.float64], k: int = None) -> NDArray[np.float64]:
    """Query a pre-built BallTree index for k-nearest neighbors.

    Args:
        tree: sklearn.neighbors.BallTree
        query_points: NDArray[np.float64]
        k: int

    Returns:
        distances: NDArray[np.float64]
    """
    import sklearn.neighbors.BallTree
    return sklearn.neighbors.BallTree.query(tree=tree, query_points=query_points, k=k) # type: ignore

