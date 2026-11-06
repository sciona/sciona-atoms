"""Domain-independent columnwise mean of a central rank interval."""
import numpy as np
from sciona.ghost.registry import register_atom


def witness_central_rank_mean(*args, **kwargs):
    raise ValueError('Concrete finite numeric matrix and retained count required')


@register_atom(witness_central_rank_mean)
def central_rank_mean(values: np.ndarray, retained: int) -> np.ndarray:
    """Sort each column and average a caller-sized central rank interval.

    Rows are observations and columns independent quantities. Require at least
    one row, finite real numeric values and a Python integer retained count in
    [1, rows]. Zero columns return an empty float64 vector. Convert to a fresh
    column-contiguous float64 matrix, sort ascending, then apply NumPy mean to
    ranks [floor((rows-retained)/2), floor((rows-retained)/2)+retained).
    An odd number of discarded ranks removes one more from the upper tail.
    Reject conversion or reduction overflow, including intermediate sum overflow.
    Do not change inputs. Selection, units, identities and interpretation belong
    to the caller; no model names, categories or fixed ensemble size are assumed.
    """
    if (not isinstance(values, np.ndarray) or values.ndim != 2 or values.shape[0] == 0
            or values.dtype.kind not in 'iuf' or not np.isfinite(values).all()):
        raise ValueError('Finite real matrix with at least one row required')
    if type(retained) is not int or not 1 <= retained <= values.shape[0]:
        raise ValueError('Retained count must be an integer within row count')
    with np.errstate(over='ignore', invalid='ignore'):
        ordered = np.array(values, dtype=np.float64, order='F', copy=True)
        if not np.isfinite(ordered).all():
            raise ValueError('Float64 conversion overflow')
        ordered.sort(axis=0)
        start = (ordered.shape[0] - retained) // 2
        result = ordered[start:start + retained].mean(axis=0)
    if not np.isfinite(result).all():
        raise ValueError('Central mean reduction overflow')
    return result
