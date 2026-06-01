from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_solve_perplexity_probabilities(distances: AbstractArray, perplexity: AbstractScalar | float) -> AbstractArray:
    """Ghost witness for solve_perplexity_probabilities."""
    _ = (distances, perplexity)
    return AbstractArray(shape=distances.shape, dtype=distances.dtype)

def witness_optimize_layout_kl(probabilities: AbstractArray, initial_layout: AbstractArray, n_iter: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for optimize_layout_kl."""
    _ = (probabilities, initial_layout, n_iter)
    return AbstractArray(shape=probabilities.shape, dtype=probabilities.dtype)

