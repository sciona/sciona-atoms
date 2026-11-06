"""Domain-independent indexing of finite scores between ordered cut points."""
import numpy as np
from sciona.ghost.registry import register_atom


def witness_ordered_cut_indices(*args, **kwargs):
    raise ValueError('Concrete finite floating vectors and an explicit tie policy required')


@register_atom(witness_ordered_cut_indices)
def ordered_cut_indices(values: np.ndarray, cut_points: np.ndarray, tie_policy: str) -> np.ndarray:
    """Return the index of each value among strictly increasing cut points.

    Inputs are finite one-dimensional float32 or float64 arrays. Values must
    be nonempty; an empty cut vector assigns index zero to every value.
    'upper' places equality above a cut; 'lower' places equality below it.
    Use NumPy searchsorted without rounding, clipping or casting either
    input. Return a fresh integer vector with indices from zero through the
    number of cuts. Inputs remain unchanged. Boundaries, units, class names
    and interpretation are caller policy; no calibration or learning occurs.
    """
    for array in (values, cut_points):
        if (not isinstance(array, np.ndarray) or array.ndim != 1
                or array.dtype not in (np.dtype('float32'), np.dtype('float64'))
                or not np.isfinite(array).all()):
            raise ValueError('Finite one-dimensional float32 or float64 arrays required')
    if not values.size:
        raise ValueError('Nonempty values required')
    if np.any(cut_points[1:] <= cut_points[:-1]):
        raise ValueError('Strictly increasing cut points required')
    if not isinstance(tie_policy, str) or tie_policy not in ('upper', 'lower'):
        raise ValueError('Explicit upper or lower tie policy required')
    return np.searchsorted(cut_points, values, side='right' if tie_policy == 'upper' else 'left')
