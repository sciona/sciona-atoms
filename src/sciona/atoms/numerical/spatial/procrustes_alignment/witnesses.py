from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_center_and_normalize(data: AbstractArray) -> AbstractArray:
    """Ghost witness for center_and_normalize."""
    _ = (data)
    return AbstractArray(shape=data.shape, dtype=data.dtype)

def witness_compute_optimal_rotation(standardized_ref: AbstractArray, standardized_src: AbstractArray) -> AbstractArray:
    """Ghost witness for compute_optimal_rotation."""
    _ = (standardized_ref, standardized_src)
    return AbstractArray(shape=standardized_ref.shape, dtype=standardized_ref.dtype)

def witness_apply_transform_and_measure(standardized_ref: AbstractArray, standardized_src: AbstractArray, rotation: AbstractArray) -> AbstractArray:
    """Ghost witness for apply_transform_and_measure."""
    _ = (standardized_ref, standardized_src, rotation)
    return AbstractArray(shape=standardized_ref.shape, dtype=standardized_ref.dtype)

