from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_solve_shortest_paths(knn_graph: AbstractScalar | Any) -> AbstractArray:
    """Ghost witness for solve_shortest_paths."""
    _ = (knn_graph)
    return AbstractArray(shape=(), dtype="float64")

def witness_classical_mds(geodesics: AbstractArray, n_components: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for classical_mds."""
    _ = (geodesics, n_components)
    return AbstractArray(shape=geodesics.shape, dtype=geodesics.dtype)

