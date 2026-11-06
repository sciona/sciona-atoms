"""Causal rolling location/scale thresholds independent of detector or domain."""
import math
import numpy as np
from sciona.ghost.registry import register_atom


def witness_prior_location_scale_threshold(*args, **kwargs):
    raise ValueError('Concrete finite observations and explicit controls required')


@register_atom(witness_prior_location_scale_threshold)
def prior_location_scale_threshold(values: list, history: int, minimum: int,
                                   ddof: int, multiplier: float) -> list:
    """Return prior-only rolling mean plus multiplier times standard deviation.

    At index i use values[max(0, i-history):i], excluding the current value.
    Return None until at least minimum prior observations exist. Empty input
    returns an empty list. Require 1 <= minimum <= history and 0 <= ddof <
    minimum; all counts are Python integers. Values and multiplier are finite
    Python real numbers, excluding booleans. Multiplier may be zero or negative.
    NumPy float64 reductions define arithmetic; reject conversion or reduction
    overflow. Preserve input. Missing-value handling, comparisons, labels and
    application-specific multiplier choices belong to the caller.
    """
    if (not isinstance(values, list)
            or any(type(v) not in (int, float) for v in values)):
        raise ValueError('Finite Python real observation list required')
    if (any(type(v) is not int for v in (history, minimum, ddof))
            or not 1 <= minimum <= history or not 0 <= ddof < minimum):
        raise ValueError('Valid integer history, minimum and ddof required')
    if type(multiplier) not in (int, float):
        raise ValueError('Finite real multiplier required')
    try:
        with np.errstate(over='ignore', invalid='ignore'):
            x = np.asarray(values, dtype=np.float64)
        scale = float(multiplier)
    except (OverflowError, ValueError):
        raise ValueError('Float64 conversion overflow') from None
    if not np.isfinite(x).all() or not math.isfinite(scale):
        raise ValueError('Finite float64 values and multiplier required')
    result = [None] * len(x)
    for i in range(minimum, len(x)):
        prior = x[max(0, i-history):i]
        with np.errstate(over='ignore', invalid='ignore'):
            threshold = float(prior.mean() + scale * prior.std(ddof=ddof))
        if not math.isfinite(threshold):
            raise ValueError('Threshold reduction overflow')
        result[i] = threshold
    return result
