from __future__ import annotations

from typing import Callable, Literal

import icontract
import numpy as np
from sciona.ghost.registry import register_atom

from .de import build_de_transition_kernel
from .hmc import buildhmckernelfromlogdensityoracle
from .mcmc_algos_witnesses import witness_dispatch_mcmc_algorithm
from .rwmh import constructrandomwalkmetropoliskernel


AlgorithmName = Literal["rwmh", "hmc", "de"]


@register_atom(witness_dispatch_mcmc_algorithm)
@icontract.require(lambda algorithm: algorithm in {"rwmh", "hmc", "de"}, "algorithm must be rwmh, hmc, or de")
@icontract.require(lambda target_log_kernel: callable(target_log_kernel), "target_log_kernel must be callable")
@icontract.require(lambda initial_state: isinstance(initial_state, np.ndarray), "initial_state must be a numpy array")
@icontract.require(lambda n_draws: int(n_draws) >= 1, "n_draws must be at least one")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be a numpy array")
def dispatch_mcmc_algorithm(
    algorithm: AlgorithmName,
    target_log_kernel: Callable[[np.ndarray], float],
    initial_state: np.ndarray,
    n_draws: int,
    rng_key: np.ndarray | None = None,
) -> np.ndarray:
    """Run a small explicit MCMC dispatcher over repaired local kernels.

    The dispatcher accepts a real target log-density oracle, chooses one of the
    repaired NumPy transition kernels, threads the RNG key through every draw,
    and returns the sampled states. It deliberately does not mimic the upstream
    C++ overload table or settings object.
    """
    if rng_key is None:
        key = np.array([42], dtype=np.int64)
    else:
        key = np.asarray(rng_key, dtype=np.int64)

    if algorithm == "rwmh":
        kernel = constructrandomwalkmetropoliskernel(target_log_kernel)
    elif algorithm == "hmc":
        kernel = buildhmckernelfromlogdensityoracle(target_log_kernel)
    else:
        kernel = build_de_transition_kernel(target_log_kernel)

    state = np.asarray(initial_state, dtype=np.float64)
    samples = np.zeros((int(n_draws),) + state.shape, dtype=np.float64)
    for draw_index in range(int(n_draws)):
        state, key = kernel(state, key)
        samples[draw_index] = state
    return samples
