"""Ordered finite circular filtering with explicit caller-selected taps."""
import numbers
import numpy as np
from sciona.ghost.registry import register_atom


def witness_ordered_circular_filter(*args, **kwargs):
    raise ValueError('Concrete finite values and filter controls required')


@register_atom(witness_ordered_circular_filter)
def ordered_circular_filter(values: np.ndarray, shifts: object, weights: object, axis: int) -> np.ndarray:
    """Sum weighted circular shifts in the supplied order, preserving float dtype.

    Nonempty finite float32/float64 arrays only. Axis is explicit and nonnegative;
    positive shifts move values toward increasing indices with wraparound.
    Shifts are integers and weights are finite real scalars representable in the
    input dtype. The first weighted shift initializes the result; later terms
    are added left to right without normalization or regrouping. Negative weights
    and repeated shifts are allowed. Reject nonfinite intermediate arithmetic.
    Returns a new array without changing caller inputs. Axis meanings, sampling
    intervals and filter interpretation belong to the caller; no domain policy.
    """
    values=np.asarray(values)
    if values.dtype not in (np.dtype('float32'),np.dtype('float64')) or values.ndim<1 or not values.size or not np.isfinite(values).all():
        raise ValueError('Nonempty finite float32/float64 array required')
    if isinstance(axis,(bool,np.bool_)) or not isinstance(axis,numbers.Integral) or not 0<=axis<values.ndim:
        raise ValueError('Explicit nonnegative axis required')
    shifts=np.asarray(shifts);weights=np.asarray(weights)
    if (shifts.ndim!=1 or shifts.dtype.kind not in 'iu' or not shifts.size
            or weights.ndim!=1 or weights.dtype.kind not in 'fiu' or weights.shape!=shifts.shape
            or not np.isfinite(weights).all()):
        raise ValueError('Equal nonempty integer shifts and finite real weights required')
    result=None
    with np.errstate(over='ignore',invalid='ignore'):
        for shift,weight in zip(shifts,weights,strict=True):
            coefficient=float(weight)
            if not np.isfinite(values.dtype.type(coefficient)):
                raise ValueError('Weight outside input dtype range')
            term=np.roll(values,int(shift),axis=int(axis))*coefficient
            result=term if result is None else result+term
            if not np.isfinite(result).all():
                raise ValueError('Nonfinite circular filter arithmetic')
    return result
