from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_validate_and_sort_descriptive_inputs(data: AbstractArray) -> AbstractArray:
    """Ghost witness for validate_and_sort_descriptive_inputs."""
    _ = (data)
    return AbstractArray(shape=data.shape, dtype=data.dtype)

def witness_compute_robust_location(sorted_data: AbstractArray, trim_ratio: AbstractScalar | float) -> AbstractScalar:
    """Ghost witness for compute_robust_location."""
    _ = (sorted_data, trim_ratio)
    return AbstractScalar(dtype="float64")

def witness_compute_robust_scale(sorted_data: AbstractArray) -> AbstractScalar:
    """Ghost witness for compute_robust_scale."""
    _ = (sorted_data)
    return AbstractScalar(dtype="float64")

