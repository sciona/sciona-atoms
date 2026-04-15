from __future__ import annotations
"""Auto-generated atom wrappers following the sciona pattern."""


import numpy as np

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import witness_posterior_randmodel

# juliacall unavailable; reimplemented in pure numpy


# Witness functions should be imported from the generated witnesses module

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
    # Beta-Binomial conjugate update: pri = [alpha, beta]
    alpha = pri[0] + np.sum(data)
    beta = pri[1] + len(data) - np.sum(data)
    return np.array([alpha, beta])


@register_atom(witness_posterior_randmodel)
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
    # Weighted Beta-Binomial conjugate update
    alpha = pri[0] + np.sum(w * data)
    beta = pri[1] + np.sum(w * (1.0 - data))
    return np.array([alpha, beta])

