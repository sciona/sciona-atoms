from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_initialize_mcmc_chain(initial_state: AbstractArray, log_posterior_fn: AbstractArray) -> AbstractArray:
    """Ghost witness for initialize_mcmc_chain."""
    _ = (initial_state, log_posterior_fn)
    return AbstractArray(shape=initial_state.shape, dtype=initial_state.dtype)

def witness_execute_metropolis_hastings_loop(validated_start: AbstractArray, log_posterior_fn: AbstractArray, proposal_std: AbstractScalar | float, n_draws: AbstractScalar | int, seed: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for execute_metropolis_hastings_loop."""
    _ = (validated_start, log_posterior_fn, proposal_std, n_draws, seed)
    return AbstractArray(shape=validated_start.shape, dtype=validated_start.dtype)

def witness_compute_mcmc_chain_diagnostics(raw_chain: AbstractArray, burn_in_ratio: AbstractScalar | float) -> AbstractArray:
    """Ghost witness for compute_mcmc_chain_diagnostics."""
    _ = (raw_chain, burn_in_ratio)
    return AbstractArray(shape=raw_chain.shape, dtype=raw_chain.dtype)

