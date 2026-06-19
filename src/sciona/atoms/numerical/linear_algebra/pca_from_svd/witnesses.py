from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_center_data(X: AbstractArray) -> Tuple[AbstractArray, AbstractArray]:
    """Ghost witness for center_data."""
    _ = (X)
    mean_shape = (X.shape[1],) if len(X.shape) > 1 else ()
    return AbstractArray(shape=X.shape, dtype=X.dtype), AbstractArray(shape=mean_shape, dtype=X.dtype)

def witness_pca_svd_decompose(X_centered: AbstractArray, n_components: AbstractScalar | int) -> Tuple[AbstractArray, AbstractArray]:
    """Ghost witness for pca_svd_decompose."""
    _ = (X_centered, n_components)
    # We can symbolically represent n_components
    c_val = int(n_components) if isinstance(n_components, int) else 1
    return AbstractArray(shape=(c_val, X_centered.shape[1]), dtype=X_centered.dtype), AbstractArray(shape=(c_val,), dtype=X_centered.dtype)

def witness_calculate_pca_variance(singular_values: AbstractArray, n_samples: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for calculate_pca_variance."""
    _ = (singular_values, n_samples)
    return AbstractArray(shape=singular_values.shape, dtype=singular_values.dtype)


