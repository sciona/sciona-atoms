from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_generate_bootstrap_resamples(data: AbstractArray, n_resamples: AbstractScalar | int, seed: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for generate_bootstrap_resamples."""
    _ = (data, n_resamples, seed)
    return AbstractArray(shape=data.shape, dtype=data.dtype)

def witness_compute_bootstrap_statistics(resamples: AbstractArray, statistic_fn: AbstractArray) -> AbstractArray:
    """Ghost witness for compute_bootstrap_statistics."""
    _ = (resamples, statistic_fn)
    return AbstractArray(shape=resamples.shape, dtype=resamples.dtype)

def witness_calculate_bootstrap_intervals(bootstrap_distribution: AbstractArray, observed_statistic: AbstractScalar | float, confidence_level: AbstractScalar | float, method: AbstractScalar | str) -> AbstractScalar:
    """Ghost witness for calculate_bootstrap_intervals."""
    _ = (bootstrap_distribution, observed_statistic, confidence_level, method)
    return AbstractScalar(dtype="float64")

