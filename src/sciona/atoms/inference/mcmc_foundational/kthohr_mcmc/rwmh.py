from __future__ import annotations

from typing import Callable

import icontract
import numpy as np
from sciona.ghost.registry import register_atom

from .rwmh_witnesses import witness_constructrandomwalkmetropoliskernel


def _seed_from_key(rng_key: np.ndarray) -> int:
    key = np.asarray(rng_key, dtype=np.int64).ravel()
    if key.size == 0:
        raise ValueError("rng must contain at least one integer seed value")
    return int(np.abs(key).sum()) % (2**31 - 1)


@register_atom(witness_constructrandomwalkmetropoliskernel)
@icontract.require(lambda target_log_kernel: callable(target_log_kernel), "target_log_kernel must be callable")
@icontract.require(lambda proposal_scale: float(proposal_scale) > 0.0, "proposal_scale must be positive")
@icontract.ensure(lambda result: callable(result), "result must be a transition kernel")
def constructrandomwalkmetropoliskernel(
    target_log_kernel: Callable[[np.ndarray], float],
    proposal_scale: float = 1.0,
) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Build a minimal Random-Walk Metropolis transition kernel.

    The returned kernel performs one Gaussian random-walk proposal from the
    supplied current state, evaluates the caller-provided log density at the
    current and proposed states, applies the Metropolis accept/reject rule, and
    returns the next state plus a new explicit RNG key.
    """

    def _rwmh_kernel(state: np.ndarray, rng: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        current = np.asarray(state, dtype=np.float64)
        local_rng = np.random.RandomState(_seed_from_key(rng))
        proposal = current + float(proposal_scale) * local_rng.normal(size=current.shape)

        current_logp = float(target_log_kernel(current))
        proposal_logp = float(target_log_kernel(proposal))
        if not np.isfinite(proposal_logp):
            proposal_logp = -np.inf
        if not np.isfinite(current_logp):
            current_logp = -np.inf

        log_accept = min(0.0, proposal_logp - current_logp)
        if np.log(local_rng.uniform()) < log_accept:
            state_out = proposal
        else:
            state_out = current.copy()
        rng_out = np.asarray(local_rng.randint(0, 2**31 - 1, size=np.asarray(rng).shape), dtype=np.int64)
        return state_out, rng_out

    return _rwmh_kernel
