from __future__ import annotations

from typing import Callable

import icontract
import numpy as np
from sciona.ghost.registry import register_atom

from .witnesses import witness_metropolishastingstransitionkernel, witness_targetlogkerneloracle


def _seed_from_key(rng_key: np.ndarray) -> int:
    key = np.asarray(rng_key, dtype=np.int64).ravel()
    if key.size == 0:
        raise ValueError("rng_key_in must contain at least one integer seed value")
    return int(np.abs(key).sum()) % (2**31 - 1)


def _next_key(local_rng: np.random.RandomState, shape: tuple[int, ...]) -> np.ndarray:
    return np.asarray(local_rng.randint(0, 2**31 - 1, size=shape), dtype=np.int64)


def _proposal_matrix(prop_scaling_mat: np.ndarray | None, dim: int) -> np.ndarray:
    if prop_scaling_mat is None:
        return np.eye(dim, dtype=np.float64)
    matrix = np.asarray(prop_scaling_mat, dtype=np.float64)
    if matrix.shape != (dim, dim):
        raise ValueError("prop_scaling_mat must have shape (state_dim, state_dim)")
    return matrix


@register_atom(witness_metropolishastingstransitionkernel)
@icontract.require(lambda state_in: isinstance(state_in, np.ndarray), "state_in must be a numpy array")
@icontract.require(lambda temper_val: float(temper_val) > 0.0, "temper_val must be positive")
@icontract.require(lambda target_log_kernel: callable(target_log_kernel), "target_log_kernel must be callable")
@icontract.require(lambda rng_key_in: isinstance(rng_key_in, np.ndarray), "rng_key_in must be a numpy array")
@icontract.ensure(lambda result: isinstance(result[0], np.ndarray) and isinstance(result[1], np.ndarray))
def metropolishastingstransitionkernel(
    state_in: np.ndarray,
    temper_val: float,
    target_log_kernel: Callable[[np.ndarray], float],
    rng_key_in: np.ndarray,
    prop_scaling_mat: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one source-shaped AEES local Metropolis step.

    The step follows the KTHOHR ``single_step_mh`` update: propose a Gaussian
    random walk from the current state, divide the log-kernel difference by the
    temperature, clip the log acceptance value at ``0.01``, and return either
    the proposal or the original state with an advanced explicit RNG key.
    """
    state = np.asarray(state_in, dtype=np.float64)
    flat_state = state.ravel()
    local_rng = np.random.RandomState(_seed_from_key(rng_key_in))
    scaling = _proposal_matrix(prop_scaling_mat, flat_state.size)

    noise = local_rng.normal(size=flat_state.size)
    proposal_flat = flat_state + np.sqrt(float(temper_val)) * scaling @ noise
    proposal = proposal_flat.reshape(state.shape)

    val_new = float(target_log_kernel(proposal))
    val_prev = float(target_log_kernel(state))
    if not np.isfinite(val_new):
        val_new = -np.inf
    if not np.isfinite(val_prev):
        val_prev = -np.inf

    log_accept = min(0.01, (val_new - val_prev) / float(temper_val))
    if local_rng.uniform() < float(np.exp(log_accept)):
        state_out = proposal
    else:
        state_out = state.copy()
    return state_out, _next_key(local_rng, np.asarray(rng_key_in).shape)


@register_atom(witness_targetlogkerneloracle)
@icontract.require(lambda state_candidate: isinstance(state_candidate, np.ndarray), "state_candidate must be a numpy array")
@icontract.require(lambda weights: isinstance(weights, np.ndarray), "weights must be a numpy array")
@icontract.require(lambda means: isinstance(means, np.ndarray), "means must be a numpy array")
@icontract.require(lambda variances: isinstance(variances, np.ndarray), "variances must be a numpy array")
@icontract.require(lambda temper_val: float(temper_val) > 0.0, "temper_val must be positive")
@icontract.ensure(lambda result: np.isfinite(result), "target log-kernel must be finite")
def targetlogkerneloracle(
    state_candidate: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    temper_val: float = 1.0,
) -> float:
    """Evaluate the AEES example Gaussian-mixture log kernel.

    This is the explicit target oracle used by the upstream AEES mixture
    example, with component means arranged as ``(n_components, n_dimensions)``.
    The returned value is optionally scaled for callers that need a tempered
    pointwise score; the Metropolis transition itself expects an unscaled
    target oracle when reproducing the upstream temperature rule.
    """
    x = np.asarray(state_candidate, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    mu = np.asarray(means, dtype=np.float64)
    sig_sq = np.asarray(variances, dtype=np.float64).ravel()

    if mu.ndim != 2 or mu.shape[1] != x.size:
        raise ValueError("means must have shape (n_components, n_dimensions)")
    if w.size != mu.shape[0] or sig_sq.size != mu.shape[0]:
        raise ValueError("weights and variances must match the number of means")
    if np.any(w < 0.0) or not np.isclose(np.sum(w), 1.0):
        raise ValueError("weights must be non-negative and sum to one")
    if np.any(sig_sq <= 0.0):
        raise ValueError("variances must be positive")

    log_terms = []
    dim = float(x.size)
    for weight, mean, variance in zip(w, mu, sig_sq, strict=True):
        dist_sq = float(np.sum((x - mean) ** 2))
        log_norm = -0.5 * dim * np.log(2.0 * np.pi * variance)
        log_terms.append(np.log(weight) + log_norm - 0.5 * dist_sq / variance)

    max_log = max(log_terms)
    log_density = max_log + float(np.log(np.sum(np.exp(np.asarray(log_terms) - max_log))))
    return float(temper_val) * log_density
