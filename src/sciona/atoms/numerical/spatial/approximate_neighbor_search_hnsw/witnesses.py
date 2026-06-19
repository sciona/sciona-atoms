from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_build_hnsw_index(data: AbstractArray, M: AbstractScalar | int = None, ef_construction: AbstractScalar | int = None) -> AbstractScalar:
    """Ghost witness for build_hnsw_index."""
    _ = (data, M, ef_construction)
    return AbstractScalar(dtype="object")

def witness_query_hnsw_index(index: AbstractScalar | Any, query_points: AbstractArray, k: AbstractScalar | int = None, ef_search: AbstractScalar | int = None) -> Tuple[AbstractArray, AbstractArray]:
    """Ghost witness for query_hnsw_index."""
    _ = (index, query_points, k, ef_search)
    k_val = int(k) if isinstance(k, (int, float)) else 1
    return (
        AbstractArray(shape=(query_points.shape[0], k_val), dtype="float64"),
        AbstractArray(shape=(query_points.shape[0], k_val), dtype="int64")
    )


