from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_solve_lle_weights,
    witness_solve_lle_embedding,
)

@register_atom(witness_solve_lle_weights, name="solve_lle_weights")
@icontract.require(lambda X, indices, reg: X.ndim == 2, "Precondition failed: X.ndim == 2")
@icontract.require(lambda X, indices, reg: indices.ndim == 2, "Precondition failed: indices.ndim == 2")
@icontract.ensure(lambda result, X, indices, reg: np.allclose(weights.sum(axis=1), 1.0), "Postcondition failed: np.allclose(weights.sum(axis=1), 1.0)")
def solve_lle_weights(X: NDArray[np.float64], indices: NDArray[np.int64], reg: float = None) -> NDArray[np.float64]:
    """Solve constrained quadratic regressions to find local coordinate reconstruction weights.

    Args:
        X: NDArray[np.float64]
        indices: NDArray[np.int64]
        reg: float

    Returns:
        weights: NDArray[np.float64]
    """
    import sklearn.manifold
    return sklearn.manifold.LocallyLinearEmbedding(X=X, indices=indices, reg=reg) # type: ignore

@register_atom(witness_solve_lle_embedding, name="solve_lle_embedding")
@icontract.require(lambda indices, weights, n_components: n_components > 0, "Precondition failed: n_components > 0")
@icontract.ensure(lambda result, indices, weights, n_components: embedding.shape == (indices.shape[0], n_components), "Postcondition failed: embedding.shape == (indices.shape[0], n_components)")
def solve_lle_embedding(indices: NDArray[np.int64], weights: NDArray[np.float64], n_components: int) -> NDArray[np.float64]:
    """Solve the sparse eigenvalue problem to extract weight preserving target embedding coordinates.

    Args:
        indices: NDArray[np.int64]
        weights: NDArray[np.float64]
        n_components: int

    Returns:
        embedding: NDArray[np.float64]
    """
    import sklearn.manifold
    return sklearn.manifold.LocallyLinearEmbedding(indices=indices, weights=weights, n_components=n_components) # type: ignore

