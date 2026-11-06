"""Explicit uncentered row scaling, without domain or optimization policy."""
import numpy as np
from sciona.ghost.registry import register_atom


def witness_normalize_rows_by_scale(*args, **kwargs):
    raise ValueError('Concrete finite rows and explicit scale statistic required')


@register_atom(witness_normalize_rows_by_scale)
def normalize_rows_by_scale(values: np.ndarray, statistic: str) -> np.ndarray:
    """Divide each row by its mean absolute value or population standard deviation.

    Accept nonempty finite 2D float32/float64 arrays. Statistic must be exactly
    'mean_absolute' or 'population_std'. Keep NumPy input-dtype reductions and
    output dtype. Population standard deviation uses ddof=0; the numerator is
    NOT centered. Reject zero or nonfinite scales and nonfinite results, without
    an epsilon stabilizer. Return a fresh array; input is unchanged. Meaning of
    each row and column, units and downstream use are entirely caller policy.
    """
    if (not isinstance(values, np.ndarray) or values.ndim!=2 or not values.size
            or values.dtype not in (np.dtype('float32'), np.dtype('float64'))
            or not np.isfinite(values).all()):
        raise ValueError('Nonempty finite 2D float32/float64 array required')
    if type(statistic) is not str or statistic not in ('mean_absolute', 'population_std'):
        raise ValueError('Explicit mean_absolute or population_std statistic required')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        scale = (np.mean(np.abs(values), axis=1, keepdims=True) if statistic=='mean_absolute'
                 else np.std(values, axis=1, ddof=0, keepdims=True))
        if np.any(scale==0) or not np.isfinite(scale).all():
            raise ValueError('Row scale is zero or nonfinite')
        result = values/scale
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite row normalization')
    return result
