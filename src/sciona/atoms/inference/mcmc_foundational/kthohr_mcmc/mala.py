from __future__ import annotations

from typing import Callable

import icontract
import numpy as np
from sciona.ghost.registry import register_atom

from .mala_witnesses import witness_mala_proposal_adjustment


def _log_mvn_density(value: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> float:
    diff = np.asarray(value, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    sign, log_det = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("proposal covariance must be positive definite")
    solve = np.linalg.solve(cov, diff)
    dim = float(diff.size)
    return float(-0.5 * (dim * np.log(2.0 * np.pi) + log_det + diff @ solve))


@register_atom(witness_mala_proposal_adjustment)
@icontract.require(lambda prop_vals: isinstance(prop_vals, np.ndarray), "prop_vals must be a numpy array")
@icontract.require(lambda prev_vals: isinstance(prev_vals, np.ndarray), "prev_vals must be a numpy array")
@icontract.require(lambda step_size: float(step_size) > 0.0, "step_size must be positive")
@icontract.require(lambda precond_mat: isinstance(precond_mat, np.ndarray), "precond_mat must be a numpy array")
@icontract.require(lambda mala_mean_fn: callable(mala_mean_fn), "mala_mean_fn must be callable")
@icontract.ensure(lambda result: np.isfinite(result), "adjustment must be finite")
def mala_proposal_adjustment(
    prop_vals: np.ndarray,
    prev_vals: np.ndarray,
    step_size: float,
    precond_mat: np.ndarray,
    mala_mean_fn: Callable[[np.ndarray, float], np.ndarray],
) -> float:
    """Compute the unbounded-source MALA proposal log-ratio adjustment.

    This implements the upstream unbounded branch:
    ``log q(prev | prop) - log q(prop | prev)`` using the caller's MALA mean
    function and ``step_size**2 * precond_mat`` as the Gaussian proposal
    covariance.
    """
    proposal = np.asarray(prop_vals, dtype=np.float64).ravel()
    previous = np.asarray(prev_vals, dtype=np.float64).ravel()
    if proposal.shape != previous.shape:
        raise ValueError("prop_vals and prev_vals must have the same shape")
    precond = np.asarray(precond_mat, dtype=np.float64)
    if precond.shape != (proposal.size, proposal.size):
        raise ValueError("precond_mat must have shape (n_dimensions, n_dimensions)")

    covariance = float(step_size) ** 2 * precond
    prop_mean = np.asarray(mala_mean_fn(proposal, float(step_size)), dtype=np.float64).ravel()
    prev_mean = np.asarray(mala_mean_fn(previous, float(step_size)), dtype=np.float64).ravel()
    if prop_mean.shape != proposal.shape or prev_mean.shape != previous.shape:
        raise ValueError("mala_mean_fn must return vectors matching the input shape")

    return _log_mvn_density(previous, prop_mean, covariance) - _log_mvn_density(proposal, prev_mean, covariance)
