from __future__ import annotations

from typing import Callable

import icontract
import numpy as np
from sciona.ghost.registry import register_atom

from .de_witnesses import witness_build_de_transition_kernel


def _seed_from_key(rng_key: np.ndarray) -> int:
    key = np.asarray(rng_key, dtype=np.int64).ravel()
    if key.size == 0:
        raise ValueError("rng must contain at least one integer seed value")
    return int(np.abs(key).sum()) % (2**31 - 1)


@register_atom(witness_build_de_transition_kernel)
@icontract.require(lambda target_log_kernel: callable(target_log_kernel), "target_log_kernel must be callable")
@icontract.require(lambda gamma_scale: float(gamma_scale) > 0.0, "gamma_scale must be positive")
@icontract.ensure(lambda result: callable(result), "result must be a transition kernel")
def build_de_transition_kernel(
    target_log_kernel: Callable[[np.ndarray], float],
    gamma_scale: float = 2.38,
) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Build a minimal differential-evolution Metropolis transition kernel.

    The state is a population matrix of shape ``(n_chains, n_dimensions)``.
    Each row proposes from two other population members and accepts using the
    supplied target log-density. This is a source-shaped NumPy subset, not a
    binding to the full KTHOHR settings object or burn-in loop.
    """

    def _de_kernel(state: np.ndarray, rng: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        population = np.asarray(state, dtype=np.float64)
        if population.ndim != 2:
            raise ValueError("DE state must be a two-dimensional population matrix")
        n_members, dim = population.shape
        if n_members < 3:
            raise ValueError("DE transition requires at least three population members")

        local_rng = np.random.RandomState(_seed_from_key(rng))
        next_population = population.copy()
        gamma = float(gamma_scale) / np.sqrt(2.0 * float(dim))

        for chain_index in range(n_members):
            choices = [idx for idx in range(n_members) if idx != chain_index]
            first, second = local_rng.choice(choices, size=2, replace=False)
            proposal = population[chain_index] + gamma * (population[first] - population[second])
            current_logp = float(target_log_kernel(population[chain_index]))
            proposal_logp = float(target_log_kernel(proposal))
            if not np.isfinite(proposal_logp):
                proposal_logp = -np.inf
            log_accept = min(0.0, proposal_logp - current_logp)
            if np.log(local_rng.uniform()) < log_accept:
                next_population[chain_index] = proposal

        rng_out = np.asarray(local_rng.randint(0, 2**31 - 1, size=np.asarray(rng).shape), dtype=np.int64)
        return next_population, rng_out

    return _de_kernel
