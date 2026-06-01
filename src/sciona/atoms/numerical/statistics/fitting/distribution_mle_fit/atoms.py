from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_initialize_distribution_parameters,
    witness_optimize_log_likelihood,
    witness_compute_mle_fit_diagnostics,
)

@register_atom(witness_initialize_distribution_parameters, name="initialize_distribution_parameters")
@icontract.require(lambda data, dist_name: data.ndim == 1, "Precondition failed: data.ndim == 1")
@icontract.require(lambda data, dist_name: len(data) >= 5, "Precondition failed: len(data) >= 5")
@icontract.ensure(lambda result, data, dist_name: len(result) >= 2, "Postcondition failed: len(result) >= 2")
def initialize_distribution_parameters(data: NDArray[np.float64], dist_name: str) -> float:
    """Calculate starting values for MLE optimization using method-of-moments or heuristics.

    Args:
        data: NDArray[np.float64]
        dist_name: str

    Returns:
        init_params: tuple[float, ...]
    """
    import scipy.stats.rv_continuous
    return scipy.stats.rv_continuous._fitstart(data=data, dist_name=dist_name) # type: ignore

@register_atom(witness_optimize_log_likelihood, name="optimize_log_likelihood")
@icontract.require(lambda data, dist_name, init_params: len(init_params) >= 2, "Precondition failed: len(init_params) >= 2")
@icontract.ensure(lambda result, data, dist_name, init_params: fitted_params[len(fitted_params)-1] > 0.0, "Postcondition failed: fitted_params[len(fitted_params)-1] > 0.0")
def optimize_log_likelihood(data: NDArray[np.float64], dist_name: str, init_params: float) -> float:
    """Formulate negative log-likelihood and optimize parameters using numerical solvers.

    Args:
        data: NDArray[np.float64]
        dist_name: str
        init_params: tuple[float, ...]

    Returns:
        fitted_params: tuple[float, ...]
    """
    import scipy.optimize
    return scipy.optimize.minimize(data=data, dist_name=dist_name, init_params=init_params) # type: ignore

@register_atom(witness_compute_mle_fit_diagnostics, name="compute_mle_fit_diagnostics")
@icontract.require(lambda data, fitted_params, neg_log_like: data is not None, "Precondition failed: data is not None")
@icontract.ensure(lambda result, data, fitted_params, neg_log_like: 'aic' in result, "Postcondition failed: 'aic' in result")
@icontract.ensure(lambda result, data, fitted_params, neg_log_like: 'bic' in result, "Postcondition failed: 'bic' in result")
def compute_mle_fit_diagnostics(data: NDArray[np.float64], fitted_params: float, neg_log_like: float) -> float:
    """Calculate information criteria (AIC, BIC) and evaluate standard errors.

    Args:
        data: NDArray[np.float64]
        fitted_params: tuple[float, ...]
        neg_log_like: float

    Returns:
        diagnostics: dict[str, float]
    """
    import scipy
    return scipy.stats(data=data, fitted_params=fitted_params, neg_log_like=neg_log_like) # type: ignore

