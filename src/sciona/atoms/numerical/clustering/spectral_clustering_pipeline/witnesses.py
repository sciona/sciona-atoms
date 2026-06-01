from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_build_graph_laplacian(affinity: AbstractScalar | Any) -> AbstractScalar:
    """Ghost witness for build_graph_laplacian."""
    _ = (affinity)
    return AbstractScalar(dtype="float64")

def witness_solve_laplacian_eigenvectors(laplacian: AbstractScalar | Any, n_components: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for solve_laplacian_eigenvectors."""
    _ = (laplacian, n_components)
    return AbstractArray(shape=(), dtype="float64")

