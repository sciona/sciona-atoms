from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_dense_svd(A: AbstractArray, full_matrices: AbstractScalar | bool) -> AbstractArray:
    """Ghost witness for dense_svd."""
    _ = (A, full_matrices)
    return AbstractArray(shape=A.shape, dtype=A.dtype)

