from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_ransac_draw_sample(X: AbstractArray, min_samples: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for ransac_draw_sample."""
    _ = (X, min_samples)
    return AbstractArray(shape=X.shape, dtype=X.dtype)

def witness_evaluate_consensus(residuals: AbstractArray, threshold: AbstractScalar | float) -> AbstractArray:
    """Ghost witness for evaluate_consensus."""
    _ = (residuals, threshold)
    return AbstractArray(shape=residuals.shape, dtype=residuals.dtype)

