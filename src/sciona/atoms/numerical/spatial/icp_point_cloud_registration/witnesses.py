from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_find_correspondences(source: AbstractArray, target: AbstractArray) -> AbstractArray:
    """Ghost witness for find_correspondences."""
    _ = (source, target)
    return AbstractArray(shape=source.shape, dtype=source.dtype)

def witness_estimate_rigid_transform(source: AbstractArray, matched_target: AbstractArray) -> AbstractArray:
    """Ghost witness for estimate_rigid_transform."""
    _ = (source, matched_target)
    return AbstractArray(shape=source.shape, dtype=source.dtype)

