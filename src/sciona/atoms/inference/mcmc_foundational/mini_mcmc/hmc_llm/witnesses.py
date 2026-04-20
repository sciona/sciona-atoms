from __future__ import annotations
from sciona.ghost.abstract import AbstractArray, AbstractDistribution, AbstractMCMCTrace, AbstractRNGState, AbstractScalar, AbstractSignal

def witness_initializehmckernelstate(
    initial_positions: AbstractArray,
    step_size: AbstractScalar,
) -> tuple[AbstractMCMCTrace, AbstractArray]:
    """Shape-and-type check for initialize hmc kernel state. Returns output metadata without running the real computation."""
    _ = initial_positions, step_size
    kernel_spec = AbstractArray(shape=("K",), dtype="float64")
    chain_state = AbstractMCMCTrace(n_samples=0, n_chains=1, param_dims=(1,), warmup_steps=0)
    return (chain_state, kernel_spec)


def witness_initializesamplerrng(seed: AbstractScalar) -> AbstractRNGState:
    """Shape-and-type check for initialize sampler rng. Returns output metadata without running the real computation."""
    _ = seed
    return AbstractRNGState(seed=0, consumed=0, is_split=False)


def witness_hamiltoniantransitionkernel(
    state_in: AbstractArray,
    kernel_spec: AbstractArray,
    prng_key_in: AbstractRNGState,
    logp_oracle: AbstractArray,
) -> tuple[AbstractMCMCTrace, AbstractRNGState, AbstractArray]:
    """Shape-and-type check for hamiltonian transition kernel. Returns output metadata without running the real computation."""
    _ = state_in, kernel_spec, logp_oracle
    state_out = AbstractMCMCTrace(
        n_samples=0,
        n_chains=1,
        param_dims=(1,),
        warmup_steps=0,
    )
    rng_out = prng_key_in.advance(n_draws=1)
    stats = AbstractArray(shape=("3",), dtype="float64")
    return (state_out, rng_out, stats)


def witness_collectposteriorchain(
    n_collect: AbstractScalar,
    n_discard: AbstractScalar,
    chain_state_0: AbstractMCMCTrace,
    kernel_spec: AbstractArray,
    prng_key_state: AbstractRNGState,
    logp_oracle: AbstractArray,
) -> tuple[AbstractArray, AbstractMCMCTrace, AbstractRNGState, AbstractMCMCTrace]:
    """Shape-and-type check for collect posterior chain. Returns output metadata without running the real computation."""
    _ = n_collect, n_discard, chain_state_0, kernel_spec, logp_oracle
    samples = AbstractArray(shape=("N", "D"), dtype="float64")
    final_state = AbstractMCMCTrace(
        n_samples=0,
        n_chains=1,
        param_dims=(1,),
        warmup_steps=0,
    )
    final_rng = prng_key_state
    trace = AbstractMCMCTrace(
        n_samples=0, n_chains=1, param_dims=(1,), warmup_steps=0,
    )
    return (samples, final_state, final_rng, trace)
