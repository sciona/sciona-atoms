"""Explicit graph ports for the three-system causal score ensemble."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom
from .atoms import weighted_ensemble_combination


def witness_three_system_causal_ensemble(
    independence_direction: AbstractArray, left_right: AbstractArray,
    one_step: AbstractArray, weights: AbstractArray,
) -> AbstractArray:
    if len(independence_direction.shape) != 1 or left_right.shape != independence_direction.shape or one_step.shape != independence_direction.shape or weights.shape != (3,):
        raise ValueError('three matching score vectors and three weights required')
    return AbstractArray(shape=one_step.shape, dtype='float64')


@register_atom(witness_three_system_causal_ensemble)
def three_system_causal_ensemble(
    independence_direction: NDArray[np.float64], left_right: NDArray[np.float64],
    one_step: NDArray[np.float64], weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Combine three paired causal score vectors with explicit graph ports.

    All scores must be finite, nonempty, equally shaped 1D arrays in [-1,1]
    with adjacent-pair antisymmetry (absolute tolerance 1e-8). Weights follow
    independence/direction, left/right, one-step order and must be three finite
    nonnegative values summing to one within 1e-8. No normalization is performed.
    Delegates the weighted sum to the existing ensemble implementation.
    This adapter combines predictions; it does not fit or evaluate classifiers.
    """
    arrays=[]
    for value in (independence_direction,left_right,one_step):
        a=np.asarray(value)
        if a.dtype.kind not in 'iuf' or a.ndim != 1 or a.size == 0 or a.size % 2:
            raise ValueError('nonempty even-length real score vectors required')
        a=a.astype(np.float64)
        if not np.all(np.isfinite(a)) or np.any(np.abs(a)>1):
            raise ValueError('finite scores in [-1,1] required')
        if not np.allclose(a[0::2],-a[1::2],rtol=0,atol=1e-8):
            raise ValueError('adjacent-pair antisymmetry required')
        arrays.append(a)
    if len({a.shape for a in arrays}) != 1:
        raise ValueError('matching score shapes required')
    w=np.asarray(weights)
    if w.dtype.kind not in 'iuf' or w.shape != (3,) or not np.all(np.isfinite(w)) or np.any(w<0) or not np.isclose(w.sum(),1.,rtol=0,atol=1e-8):
        raise ValueError('three nonnegative normalized weights required')
    with np.errstate(all='raise'):
        return weighted_ensemble_combination(arrays,w.astype(np.float64))
