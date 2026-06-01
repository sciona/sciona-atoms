from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_find_core_points(neighborhood_counts: AbstractArray, min_samples: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for find_core_points."""
    _ = (neighborhood_counts, min_samples)
    return AbstractArray(shape=neighborhood_counts.shape, dtype=neighborhood_counts.dtype)

def witness_propagate_labels(neighbors: AbstractScalar | Any, core_mask: AbstractArray) -> AbstractArray:
    """Ghost witness for propagate_labels."""
    _ = (neighbors, core_mask)
    return AbstractArray(shape=core_mask.shape, dtype=core_mask.dtype)

