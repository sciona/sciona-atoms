from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_observed_statistic(x: AbstractArray, y: AbstractArray, metric: AbstractScalar | str) -> AbstractScalar:
    """Ghost witness for compute_observed_statistic."""
    _ = (x, y, metric)
    return AbstractScalar(dtype="float64")

def witness_generate_permutation_null(x: AbstractArray, y: AbstractArray, metric: AbstractScalar | str, n_permutations: AbstractScalar | int, seed: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for generate_permutation_null."""
    _ = (x, y, metric, n_permutations, seed)
    return AbstractArray(shape=x.shape, dtype=x.dtype)

def witness_calculate_permutation_p_value(null_distribution: AbstractArray, observed_stat: AbstractScalar | float, alternative: AbstractScalar | str) -> AbstractScalar:
    """Ghost witness for calculate_permutation_p_value."""
    _ = (null_distribution, observed_stat, alternative)
    return AbstractScalar(dtype="float64")

