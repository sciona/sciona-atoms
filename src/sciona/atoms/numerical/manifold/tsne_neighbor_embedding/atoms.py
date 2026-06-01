from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_solve_perplexity_probabilities,
    witness_optimize_layout_kl,
)

@register_atom(witness_solve_perplexity_probabilities, name="solve_perplexity_probabilities")
@icontract.require(lambda distances, perplexity: perplexity > 0.0, "Precondition failed: perplexity > 0.0")
@icontract.ensure(lambda result, distances, perplexity: np.allclose(probabilities.sum(), 1.0), "Postcondition failed: np.allclose(probabilities.sum(), 1.0)")
def solve_perplexity_probabilities(distances: NDArray[np.float64], perplexity: float) -> NDArray[np.float64]:
    """Use binary search to solve for Gaussian variances matching perplexity targets and yield symmetric joint probabilities.

    Args:
        distances: NDArray[np.float64]
        perplexity: float

    Returns:
        probabilities: NDArray[np.float64]
    """
    import sklearn.manifold
    return sklearn.manifold.TSNE(distances=distances, perplexity=perplexity) # type: ignore

@register_atom(witness_optimize_layout_kl, name="optimize_layout_kl")
@icontract.require(lambda probabilities, initial_layout, n_iter: n_iter > 0, "Precondition failed: n_iter > 0")
@icontract.ensure(lambda result, probabilities, initial_layout, n_iter: embedding.shape[1] == initial_layout.shape[1], "Postcondition failed: embedding.shape[1] == initial_layout.shape[1]")
def optimize_layout_kl(probabilities: NDArray[np.float64], initial_layout: NDArray[np.float64], n_iter: int) -> NDArray[np.float64]:
    """Run gradient descent optimization to position points minimizing KL divergence under a student-t distribution.

    Args:
        probabilities: NDArray[np.float64]
        initial_layout: NDArray[np.float64]
        n_iter: int

    Returns:
        embedding: NDArray[np.float64]
    """
    import sklearn.manifold
    return sklearn.manifold.TSNE(probabilities=probabilities, initial_layout=initial_layout, n_iter=n_iter) # type: ignore

