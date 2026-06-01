from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_pairwise_distances,
    witness_assemble_rbf_system,
    witness_solve_rbf_weights,
    witness_evaluate_rbf_predictions,
)

@register_atom(witness_compute_pairwise_distances, name="compute_pairwise_distances")
@icontract.require(lambda XA, XB: XA.shape[1] == XB.shape[1], "Precondition failed: XA.shape[1] == XB.shape[1]")
@icontract.ensure(lambda result, XA, XB: distances.shape == (XA.shape[0], XB.shape[0]), "Postcondition failed: distances.shape == (XA.shape[0], XB.shape[0])")
def compute_pairwise_distances(XA: NDArray[np.float64], XB: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute Euclidean pairwise distance matrix between two sets of coordinates.

    Args:
        XA: Shape N x D
        XB: Shape M x D

    Returns:
        distances: NDArray[np.float64]
    """
    import scipy.spatial.distance
    return scipy.spatial.distance.cdist(XA=XA, XB=XB) # type: ignore

@register_atom(witness_assemble_rbf_system, name="assemble_rbf_system")
@icontract.require(lambda distances, kernel_name, epsilon, x_train, degree: distances is not None, "Precondition failed: distances is not None")
@icontract.ensure(lambda result, distances, kernel_name, epsilon, x_train, degree: result is not None, "Postcondition failed: result is not None")
def assemble_rbf_system(distances: NDArray[np.float64], kernel_name: str, epsilon: float, x_train: NDArray[np.float64], degree: int) -> NDArray[np.float64]:
    """Apply RBF kernel function to distances and augment with polynomial terms to form symmetric linear system.

    Args:
        distances: NDArray[np.float64]
        kernel_name: str
        epsilon: float
        x_train: NDArray[np.float64]
        degree: int

    Returns:
        A: NDArray[np.float64]
    """
    import scipy.interpolate
    return scipy.interpolate.RBFInterpolator(distances=distances, kernel_name=kernel_name, epsilon=epsilon, x_train=x_train, degree=degree) # type: ignore

@register_atom(witness_solve_rbf_weights, name="solve_rbf_weights")
@icontract.require(lambda A, rhs: A.shape[0] == A.shape[1], "Precondition failed: A.shape[0] == A.shape[1]")
@icontract.ensure(lambda result, A, rhs: len(weights) == len(rhs), "Postcondition failed: len(weights) == len(rhs)")
def solve_rbf_weights(A: NDArray[np.float64], rhs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solve the symmetric linear system for RBF weights using Cholesky or SVD.

    Args:
        A: NDArray[np.float64]
        rhs: NDArray[np.float64]

    Returns:
        weights: NDArray[np.float64]
    """
    import scipy.linalg
    return scipy.linalg.solve(A=A, rhs=rhs) # type: ignore

@register_atom(witness_evaluate_rbf_predictions, name="evaluate_rbf_predictions")
@icontract.require(lambda weights, distances_eval, x_eval, x_train, kernel_name, epsilon, degree: weights is not None, "Precondition failed: weights is not None")
@icontract.ensure(lambda result, weights, distances_eval, x_eval, x_train, kernel_name, epsilon, degree: result is not None, "Postcondition failed: result is not None")
def evaluate_rbf_predictions(weights: NDArray[np.float64], distances_eval: NDArray[np.float64], x_eval: NDArray[np.float64], x_train: NDArray[np.float64], kernel_name: str, epsilon: float, degree: int) -> NDArray[np.float64]:
    """Evaluate the RBF model on target coordinates using computed weights.

    Args:
        weights: NDArray[np.float64]
        distances_eval: NDArray[np.float64]
        x_eval: NDArray[np.float64]
        x_train: NDArray[np.float64]
        kernel_name: str
        epsilon: float
        degree: int

    Returns:
        result: NDArray[np.float64]
    """
    import scipy.interpolate
    return scipy.interpolate.RBFInterpolator(weights=weights, distances_eval=distances_eval, x_eval=x_eval, x_train=x_train, kernel_name=kernel_name, epsilon=epsilon, degree=degree) # type: ignore

