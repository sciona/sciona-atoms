from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_gmm_e_step(X: AbstractArray, means: AbstractArray, covariances: AbstractArray, weights: AbstractArray) -> AbstractArray:
    """Ghost witness for gmm_e_step."""
    _ = (X, means, covariances, weights)
    return AbstractArray(shape=X.shape, dtype=X.dtype)

def witness_gmm_m_step(X: AbstractArray, responsibilities: AbstractArray, covariance_type: AbstractScalar | str) -> AbstractArray:
    """Ghost witness for gmm_m_step."""
    _ = (X, responsibilities, covariance_type)
    return AbstractArray(shape=X.shape, dtype=X.dtype)

