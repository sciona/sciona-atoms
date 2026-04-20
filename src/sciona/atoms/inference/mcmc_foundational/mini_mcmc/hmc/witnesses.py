from __future__ import annotations
from sciona.ghost.abstract import AbstractArray, AbstractDistribution, AbstractMCMCTrace, AbstractScalar, AbstractSignal

def witness_initializehmcstate(target: AbstractArray, initial_positions: AbstractArray, step_size: AbstractScalar, n_leapfrog: AbstractScalar, seed: AbstractScalar) -> tuple[AbstractMCMCTrace, AbstractArray]:
    """Shape-and-type check for initialize hmc state. Returns output metadata without running the real computation."""
    hmc_state = AbstractMCMCTrace(n_samples=0, n_chains=1, param_dims=(1,), warmup_steps=0)
    kernel_static = AbstractArray(shape=("K",), dtype="float64")
    return (hmc_state, kernel_static)

def witness_leapfrogproposalkernel(
    proposal_state_in: AbstractMCMCTrace,
    kernel_static: AbstractArray,
    log_prob_oracle: AbstractArray,
) -> AbstractMCMCTrace:
    """Shape-and-type check for leapfrog proposal kernel. Returns output metadata without running the real computation."""
    return AbstractMCMCTrace(
        n_samples=proposal_state_in.n_samples,
        n_chains=proposal_state_in.n_chains,
        param_dims=proposal_state_in.param_dims,
        warmup_steps=proposal_state_in.warmup_steps,)

def witness_metropolishmctransition(
    chain_state_in: AbstractMCMCTrace,
    kernel_static: AbstractArray,
    log_prob_oracle: AbstractArray,
) -> tuple[AbstractMCMCTrace, AbstractArray]:
    """Shape-and-type check for metropolis hmc transition. Returns output metadata without running the real computation."""
    _ = log_prob_oracle
    chain_state_out = AbstractMCMCTrace(
        n_samples=chain_state_in.n_samples,
        n_chains=chain_state_in.n_chains,
        param_dims=chain_state_in.param_dims,
        warmup_steps=chain_state_in.warmup_steps,
    )
    stats = AbstractArray(shape=("3",), dtype="float64")
    return (chain_state_out, stats)

def witness_runsamplingloop(
    hmc_state_in: AbstractMCMCTrace,
    kernel_static: AbstractArray,
    n_collect: AbstractScalar,
    n_discard: AbstractScalar,
    log_prob_oracle: AbstractArray,
) -> tuple[AbstractArray, AbstractMCMCTrace, AbstractMCMCTrace]:
    """Shape-and-type check for run sampling loop. Returns output metadata without running the real computation."""
    _ = kernel_static, n_collect, n_discard, log_prob_oracle
    samples = AbstractArray(shape=("N", "D"), dtype="float64")
    trace = AbstractMCMCTrace(
        n_samples=0,
        n_chains=hmc_state_in.n_chains,
        param_dims=hmc_state_in.param_dims,
        warmup_steps=0,
    )
    final_state = AbstractMCMCTrace(
        n_samples=hmc_state_in.n_samples,
        n_chains=hmc_state_in.n_chains,
        param_dims=hmc_state_in.param_dims,
        warmup_steps=hmc_state_in.warmup_steps,
    )
    return (samples, trace, final_state)
