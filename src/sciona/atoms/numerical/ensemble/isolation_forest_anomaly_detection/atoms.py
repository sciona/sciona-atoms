from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_build_isolation_trees,
    witness_compute_path_lengths,
)

@register_atom(witness_build_isolation_trees, name="build_isolation_trees")
@icontract.require(lambda X, n_estimators, max_samples: X.ndim == 2, "Precondition failed: X.ndim == 2")
@icontract.require(lambda X, n_estimators, max_samples: n_estimators > 0, "Precondition failed: n_estimators > 0")
@icontract.ensure(lambda result, X, n_estimators, max_samples: forest is not None, "Postcondition failed: forest is not None")
def build_isolation_trees(X: NDArray[np.float64], n_estimators: int, max_samples: int = None) -> Any:
    """Construct an ensemble of random isolation partition trees.

    Args:
        X: NDArray[np.float64]
        n_estimators: int
        max_samples: int

    Returns:
        forest: Any
    """
    import sklearn.ensemble
    return sklearn.ensemble.IsolationForest(X=X, n_estimators=n_estimators, max_samples=max_samples) # type: ignore

@register_atom(witness_compute_path_lengths, name="compute_path_lengths")
@icontract.require(lambda forest, query_points: query_points.ndim == 2, "Precondition failed: query_points.ndim == 2")
@icontract.ensure(lambda result, forest, query_points: path_lengths.shape[0] == query_points.shape[0], "Postcondition failed: path_lengths.shape[0] == query_points.shape[0]")
def compute_path_lengths(forest: Any, query_points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Traverse query coordinates down isolation trees to gather path-length statistics.

    Args:
        forest: Any
        query_points: NDArray[np.float64]

    Returns:
        path_lengths: NDArray[np.float64]
    """
    import sklearn.ensemble
    return sklearn.ensemble.IsolationForest(forest=forest, query_points=query_points) # type: ignore

