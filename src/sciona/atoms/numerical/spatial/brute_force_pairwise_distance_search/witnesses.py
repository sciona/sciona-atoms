from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_pairwise_cdist(XA: AbstractArray, XB: AbstractArray, metric: AbstractScalar | str) -> AbstractArray:
    """Ghost witness for compute_pairwise_cdist."""
    _ = (XA, XB, metric)
    return AbstractArray(shape=XA.shape, dtype=XA.dtype)

