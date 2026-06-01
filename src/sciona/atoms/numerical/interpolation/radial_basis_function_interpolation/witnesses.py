from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_pairwise_distances(XA: AbstractArray, XB: AbstractArray) -> AbstractArray:
    """Ghost witness for compute_pairwise_distances."""
    _ = (XA, XB)
    return AbstractArray(shape=XA.shape, dtype=XA.dtype)

def witness_assemble_rbf_system(distances: AbstractArray, kernel_name: AbstractScalar | str, epsilon: AbstractScalar | float, x_train: AbstractArray, degree: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for assemble_rbf_system."""
    _ = (distances, kernel_name, epsilon, x_train, degree)
    return AbstractArray(shape=distances.shape, dtype=distances.dtype)

def witness_solve_rbf_weights(A: AbstractArray, rhs: AbstractArray) -> AbstractArray:
    """Ghost witness for solve_rbf_weights."""
    _ = (A, rhs)
    return AbstractArray(shape=A.shape, dtype=A.dtype)

def witness_evaluate_rbf_predictions(weights: AbstractArray, distances_eval: AbstractArray, x_eval: AbstractArray, x_train: AbstractArray, kernel_name: AbstractScalar | str, epsilon: AbstractScalar | float, degree: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for evaluate_rbf_predictions."""
    _ = (weights, distances_eval, x_eval, x_train, kernel_name, epsilon, degree)
    return AbstractArray(shape=weights.shape, dtype=weights.dtype)

