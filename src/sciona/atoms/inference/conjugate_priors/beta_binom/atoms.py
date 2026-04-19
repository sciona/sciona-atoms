from __future__ import annotations
"""Auto-generated atom wrappers following the sciona pattern."""


import numpy as np

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import witness_posterior_randmodel, witness_posterior_randmodel_weighted

# juliacall unavailable; reimplemented in pure numpy


def _validate_prior_and_data(pri: np.ndarray, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prior = np.asarray(pri, dtype=np.float64)
    observations = np.asarray(data, dtype=np.float64)
    if prior.shape != (2,) or np.any(~np.isfinite(prior)) or np.any(prior <= 0.0):
        raise ValueError("pri must be a finite positive two-element [alpha, beta] vector")
    if np.any(~np.isfinite(observations)) or np.any((observations != 0.0) & (observations != 1.0)):
        raise ValueError("data must contain binary observations in {0, 1}")
    return prior, observations

@register_atom(witness_posterior_randmodel)
@icontract.require(lambda pri: pri is not None, "pri cannot be None")
@icontract.require(lambda G: G is not None, "G cannot be None")
@icontract.require(lambda data: data is not None, "data cannot be None")
@icontract.ensure(lambda result: result is not None, "Posterior Randmodel output must not be None")
def posterior_randmodel(pri: np.ndarray, G: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Posterior randmodel.

    Args:
        pri: Prior parameter array.
        G: Graph adjacency or weight array.
        data: Observed data array.

    Returns:
        Posterior Beta parameters [alpha', beta'].
    """
    del G
    prior, observations = _validate_prior_and_data(pri, data)
    alpha = prior[0] + np.sum(observations)
    beta = prior[1] + observations.size - np.sum(observations)
    return np.array([alpha, beta])


@register_atom(witness_posterior_randmodel_weighted)
@icontract.require(lambda pri: pri is not None, "pri cannot be None")
@icontract.require(lambda G: G is not None, "G cannot be None")
@icontract.require(lambda data: data is not None, "data cannot be None")
@icontract.require(lambda w: w is not None, "w cannot be None")
@icontract.ensure(lambda result: result is not None, "Posterior Randmodel output must not be None")
def posterior_randmodel_weighted(pri: np.ndarray, G: np.ndarray, data: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Posterior randmodel.

    Args:
        pri: Prior parameter array.
        G: Graph adjacency or weight array.
        data: Observed data array.
        w: Per-observation weight array.

    Returns:
        Posterior Beta parameters [alpha', beta'] with weighted observations.
    """
    del G
    prior, observations = _validate_prior_and_data(pri, data)
    weights = np.asarray(w, dtype=np.float64)
    if weights.shape != observations.shape:
        raise ValueError("w must have one weight per observation")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("w must contain finite non-negative weights")
    alpha = prior[0] + np.sum(weights * observations)
    beta = prior[1] + np.sum(weights * (1.0 - observations))
    return np.array([alpha, beta])
