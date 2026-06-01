from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_test_variance_homogeneity(x1: AbstractArray, x2: AbstractArray) -> AbstractScalar:
    """Ghost witness for test_variance_homogeneity."""
    _ = (x1, x2)
    return AbstractScalar(dtype="float64")

def witness_compute_t_moments(x: AbstractArray) -> AbstractScalar:
    """Ghost witness for compute_t_moments."""
    _ = (x)
    return AbstractScalar(dtype="float64")

def witness_compute_welch_t_statistic(mean1: AbstractScalar | float, variance1: AbstractScalar | float, n1: AbstractScalar | int, mean2: AbstractScalar | float, variance2: AbstractScalar | float, n2: AbstractScalar | int) -> AbstractScalar:
    """Ghost witness for compute_welch_t_statistic."""
    _ = (mean1, variance1, n1, mean2, variance2, n2)
    return AbstractScalar(dtype="float64")

def witness_evaluate_t_significance(t_stat: AbstractScalar | float, df: AbstractScalar | float, alternative: AbstractScalar | str) -> AbstractScalar:
    """Ghost witness for evaluate_t_significance."""
    _ = (t_stat, df, alternative)
    return AbstractScalar(dtype="float64")

