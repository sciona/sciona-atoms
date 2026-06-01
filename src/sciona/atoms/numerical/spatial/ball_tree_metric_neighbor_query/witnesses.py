from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_build_ball_tree(data: AbstractArray, metric: AbstractScalar | str) -> AbstractScalar:
    """Ghost witness for build_ball_tree."""
    _ = (data, metric)
    return AbstractScalar(dtype="float64")

def witness_query_ball_tree(tree: AbstractScalar | Any, query_points: AbstractArray, k: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for query_ball_tree."""
    _ = (tree, query_points, k)
    return AbstractArray(shape=query_points.shape, dtype=query_points.dtype)

