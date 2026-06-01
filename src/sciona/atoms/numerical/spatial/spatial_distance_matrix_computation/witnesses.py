from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_invert_covariance(cov: AbstractArray, ridge: AbstractScalar | float) -> AbstractArray:
    """Ghost witness for invert_covariance."""
    _ = (cov, ridge)
    return AbstractArray(shape=cov.shape, dtype=cov.dtype)

def witness_compute_mahalanobis_cdist(XA: AbstractArray, XB: AbstractArray, inv_cov: AbstractArray) -> AbstractArray:
    """Ghost witness for compute_mahalanobis_cdist."""
    _ = (XA, XB, inv_cov)
    return AbstractArray(shape=XA.shape, dtype=XA.dtype)

