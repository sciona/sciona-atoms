from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_initialize_mcmc_chain,
    witness_execute_metropolis_hastings_loop,
    witness_compute_mcmc_chain_diagnostics,
)

@register_atom(witness_initialize_mcmc_chain, name="initialize_mcmc_chain")
@icontract.require(lambda initial_state, log_posterior_fn: initial_state.ndim == 1, "Precondition failed: initial_state.ndim == 1")
@icontract.ensure(lambda result, initial_state, log_posterior_fn: np.isfinite(start_log_prob), "Postcondition failed: np.isfinite(start_log_prob)")
def initialize_mcmc_chain(initial_state: NDArray[np.float64], log_posterior_fn: Callable[[NDArray[np.float64]], float]) -> NDArray[np.float64]:
    """Verify parameter bounds, log-posterior validity at start coordinates, and setup random generators.

    Args:
        initial_state: NDArray[np.float64]
        log_posterior_fn: Callable[[NDArray[np.float64]], float]

    Returns:
        validated_start: NDArray[np.float64]
    """
    import numpy.random
    return numpy.random.default_rng(initial_state=initial_state, log_posterior_fn=log_posterior_fn) # type: ignore

@register_atom(witness_execute_metropolis_hastings_loop, name="execute_metropolis_hastings_loop")
@icontract.require(lambda validated_start, log_posterior_fn, proposal_std, n_draws, seed: n_draws > 0, "Precondition failed: n_draws > 0")
@icontract.require(lambda validated_start, log_posterior_fn, proposal_std, n_draws, seed: proposal_std > 0.0, "Precondition failed: proposal_std > 0.0")
@icontract.ensure(lambda result, validated_start, log_posterior_fn, proposal_std, n_draws, seed: raw_chain.shape == (n_draws, len(validated_start)), "Postcondition failed: raw_chain.shape == (n_draws, len(validated_start))")
def execute_metropolis_hastings_loop(validated_start: NDArray[np.float64], log_posterior_fn: Callable[[NDArray[np.float64]], float], proposal_std: float, n_draws: int, seed: int = None) -> NDArray[np.float64]:
    """Run the core state proposal, acceptance ratio calculation, and transition loop.

    Args:
        validated_start: NDArray[np.float64]
        log_posterior_fn: Callable[[NDArray[np.float64]], float]
        proposal_std: float
        n_draws: int
        seed: int | None

    Returns:
        raw_chain: NDArray[np.float64]
    """
    import numpy.random.Generator
    return numpy.random.Generator.normal(validated_start=validated_start, log_posterior_fn=log_posterior_fn, proposal_std=proposal_std, n_draws=n_draws, seed=seed) # type: ignore

@register_atom(witness_compute_mcmc_chain_diagnostics, name="compute_mcmc_chain_diagnostics")
@icontract.require(lambda raw_chain, burn_in_ratio: 0.0 <= burn_in_ratio < 1.0, "Precondition failed: 0.0 <= burn_in_ratio < 1.0")
@icontract.require(lambda raw_chain, burn_in_ratio: raw_chain.ndim == 2, "Precondition failed: raw_chain.ndim == 2")
@icontract.ensure(lambda result, raw_chain, burn_in_ratio: posterior_samples.shape[0] < raw_chain.shape[0], "Postcondition failed: posterior_samples.shape[0] < raw_chain.shape[0]")
@icontract.ensure(lambda result, raw_chain, burn_in_ratio: r_hat >= 1.0, "Postcondition failed: r_hat >= 1.0")
def compute_mcmc_chain_diagnostics(raw_chain: NDArray[np.float64], burn_in_ratio: float = None) -> NDArray[np.float64]:
    """Calculate Gelman-Rubin R-hat, effective sample size (ESS), and discard burn-in.

    Args:
        raw_chain: NDArray[np.float64]
        burn_in_ratio: float

    Returns:
        posterior_samples: NDArray[np.float64]
    """
    import statsmodels
    return statsmodels.stats(raw_chain=raw_chain, burn_in_ratio=burn_in_ratio) # type: ignore

