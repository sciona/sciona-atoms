from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_extract_random_subspace_basis(A: AbstractArray, k: AbstractScalar | int, p: AbstractScalar | int, n_iter: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for extract_random_subspace_basis."""
    _ = (A, k, p, n_iter)
    k_val = int(k) if isinstance(k, (int, float)) else 1
    p_val = int(p) if isinstance(p, (int, float)) else 0
    return AbstractArray(shape=(A.shape[0], k_val + p_val), dtype=A.dtype)

def witness_factorize_subspace_projection(A: AbstractArray, Q: AbstractArray, k: AbstractScalar | int) -> Tuple[AbstractArray, AbstractArray, AbstractArray]:
    """Ghost witness for factorize_subspace_projection."""
    _ = (A, Q, k)
    k_val = int(k) if isinstance(k, (int, float)) else 1
    return (
        AbstractArray(shape=(A.shape[0], k_val), dtype=A.dtype),
        AbstractArray(shape=(k_val,), dtype=A.dtype),
        AbstractArray(shape=(k_val, A.shape[1]), dtype=A.dtype)
    )


