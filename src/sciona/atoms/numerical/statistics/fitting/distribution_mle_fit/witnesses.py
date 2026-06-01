from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_initialize_distribution_parameters(data: AbstractArray, dist_name: AbstractScalar | str) -> AbstractScalar:
    """Ghost witness for initialize_distribution_parameters."""
    _ = (data, dist_name)
    return AbstractScalar(dtype="float64")

def witness_optimize_log_likelihood(data: AbstractArray, dist_name: AbstractScalar | str, init_params: AbstractScalar | float) -> AbstractScalar:
    """Ghost witness for optimize_log_likelihood."""
    _ = (data, dist_name, init_params)
    return AbstractScalar(dtype="float64")

def witness_compute_mle_fit_diagnostics(data: AbstractArray, fitted_params: AbstractScalar | float, neg_log_like: AbstractScalar | float) -> AbstractScalar:
    """Ghost witness for compute_mle_fit_diagnostics."""
    _ = (data, fitted_params, neg_log_like)
    return AbstractScalar(dtype="float64")

