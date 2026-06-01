from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_AP_message_update(S: AbstractArray, R: AbstractArray, A: AbstractArray, damping: AbstractScalar | float) -> AbstractArray:
    """Ghost witness for AP_message_update."""
    _ = (S, R, A, damping)
    return AbstractArray(shape=S.shape, dtype=S.dtype)

def witness_extract_exemplars(R: AbstractArray, A: AbstractArray) -> AbstractArray:
    """Ghost witness for extract_exemplars."""
    _ = (R, A)
    return AbstractArray(shape=R.shape, dtype=R.dtype)

