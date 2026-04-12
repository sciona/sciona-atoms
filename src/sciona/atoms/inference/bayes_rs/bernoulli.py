from __future__ import annotations

import numpy as np
import icontract
from ageoa.ghost.registry import register_atom

from .bernoulli_witnesses import witness_bernoulli_probabilistic_oracle


@register_atom(witness_bernoulli_probabilistic_oracle)
@icontract.require(lambda p: isinstance(p, (float, int, np.number)), "p must be numeric")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def bernoulli_probabilistic_oracle(p: float, x: np.ndarray) -> np.ndarray:
    """Evaluate a Bernoulli distribution as a pure probabilistic oracle.

    Args:
        p: Probability parameter, 0 <= p <= 1.
        x: Observed outcome(s), each value in {0,1}.

    Returns:
        Elementwise Bernoulli log-likelihood values.
    """
    # Bernoulli log-likelihood: x*log(p) + (1-x)*log(1-p)
    p_clamped = np.clip(float(p), 1e-15, 1.0 - 1e-15)
    return x * np.log(p_clamped) + (1.0 - x) * np.log(1.0 - p_clamped)
