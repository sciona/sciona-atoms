from __future__ import annotations

import numpy as np
import icontract
from sciona.ghost.registry import register_atom

from .bernoulli_witnesses import witness_bernoulli_probabilistic_oracle


@register_atom(witness_bernoulli_probabilistic_oracle)
@icontract.require(lambda p: isinstance(p, (float, int, np.number)), "p must be numeric")
@icontract.require(lambda p: 0.0 <= float(p) <= 1.0, "p must satisfy 0 <= p <= 1")
@icontract.require(lambda x: x is not None, "x cannot be None")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def bernoulli_probabilistic_oracle(p: float, x: np.ndarray) -> np.ndarray:
    """Evaluate a Bernoulli distribution as a pure probabilistic oracle.

    Args:
        p: Probability parameter, 0 <= p <= 1.
        x: Observed outcome(s), each value in {0,1}.

    Returns:
        Elementwise Bernoulli log-likelihood values.
    """
    observations = np.asarray(x, dtype=np.float64)
    if np.any(~np.isfinite(observations)) or np.any((observations != 0.0) & (observations != 1.0)):
        raise ValueError("x must contain Bernoulli observations in {0, 1}")
    p_clamped = np.clip(float(p), 1e-15, 1.0 - 1e-15)
    return observations * np.log(p_clamped) + (1.0 - observations) * np.log(1.0 - p_clamped)
