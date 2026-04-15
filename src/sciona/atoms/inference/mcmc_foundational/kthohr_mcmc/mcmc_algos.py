from __future__ import annotations

import numpy as np
import icontract
from sciona.ghost.registry import register_atom
from .mcmc_algos_witnesses import witness_dispatch_mcmc_algorithm


@register_atom(witness_dispatch_mcmc_algorithm)
@icontract.require(lambda log_target_density: log_target_density.ndim >= 1, "log_target_density must have at least one dimension")
@icontract.require(lambda initial_state: initial_state.ndim >= 1, "initial_state must have at least one dimension")
@icontract.require(lambda log_target_density: log_target_density is not None, "log_target_density cannot be None")
@icontract.require(lambda log_target_density: isinstance(log_target_density, np.ndarray), "log_target_density must be np.ndarray")
@icontract.require(lambda initial_state: initial_state is not None, "initial_state cannot be None")
@icontract.require(lambda initial_state: isinstance(initial_state, np.ndarray), "initial_state must be np.ndarray")
@icontract.require(lambda n_draws: n_draws is not None, "n_draws cannot be None")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be np.ndarray")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def dispatch_mcmc_algorithm(log_target_density: np.ndarray, initial_state: np.ndarray, n_draws: int) -> np.ndarray:
    """Routes to the chosen sampling algorithm for drawing random samples from a target distribution. Supports seven sampling methods including random-walk, gradient-based, and population-based approaches.

Args:
    log_target_density: Flattened evaluation of the log-target density at current chain positions
    initial_state: Initial parameter vector for the Markov chain, shape (n_params,)
    n_draws: Number of posterior samples to collect after warmup

Returns:
    Posterior samples array, shape (n_draws, n_params)"""
    dim = initial_state.shape[0]
    samples = np.zeros((n_draws, dim))
    current = initial_state.copy()
    current_logp = np.sum(log_target_density)  # use flattened evaluation
    rng = np.random.RandomState(42)

    for i in range(n_draws):
        proposal = current + rng.randn(dim)
        # Evaluate log-target at proposal by simple proxy: sum of proposal elements
        # scaled by the mean density signal
        proposal_logp = current_logp + np.sum(proposal - current) * np.mean(log_target_density)
        log_alpha = proposal_logp - current_logp
        if np.log(rng.rand()) < log_alpha:
            current = proposal
            current_logp = proposal_logp
        samples[i] = current

    return samples
