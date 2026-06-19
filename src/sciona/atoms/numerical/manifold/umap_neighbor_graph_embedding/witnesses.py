from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_build_fuzzy_simplicial_set(X: AbstractArray, n_neighbors: AbstractScalar | int, metric: AbstractScalar | str = None) -> AbstractArray:
    """Ghost witness for build_fuzzy_simplicial_set."""
    _ = (X, n_neighbors, metric)
    return AbstractArray(shape=(X.shape[0], X.shape[0]), dtype=X.dtype)

def witness_optimize_umap_layout(fuzzy_graph: AbstractArray, n_epochs: AbstractScalar | int, min_dist: AbstractScalar | float) -> AbstractArray:
    """Ghost witness for optimize_umap_layout."""
    _ = (fuzzy_graph, n_epochs, min_dist)
    return AbstractArray(shape=(fuzzy_graph.shape[0], 2), dtype="float64")


