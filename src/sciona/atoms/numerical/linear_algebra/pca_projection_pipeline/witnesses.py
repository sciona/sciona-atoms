from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_center_features(X: AbstractArray) -> AbstractArray:
    """Ghost witness for center_features."""
    _ = (X)
    return AbstractArray(shape=X.shape, dtype=X.dtype)

def witness_factorize_svd(centered: AbstractArray, n_components: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for factorize_svd."""
    _ = (centered, n_components)
    return AbstractArray(shape=centered.shape, dtype=centered.dtype)

def witness_project_coordinates(centered: AbstractArray, components: AbstractArray) -> AbstractArray:
    """Ghost witness for project_coordinates."""
    _ = (centered, components)
    return AbstractArray(shape=centered.shape, dtype=centered.dtype)

