"""Convert a positive cycle duration into ordinary frequency."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.dimensions import DimensionalSignature
from sciona.ghost.registry import register_atom


def witness_frequency_from_period(period_seconds: AbstractArray) -> AbstractArray:
    return AbstractArray(shape=period_seconds.shape,dtype='float64')


@register_atom(witness_frequency_from_period,dim_map={
    'period_seconds':DimensionalSignature(T=1),'return':DimensionalSignature(T=-1),
})
def frequency_from_period(period_seconds: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ordinary frequency in hertz from a positive period in seconds.

    Accept finite real scalars or arrays, cast to float64 and preserve shape.
    Reject zero/negative periods, booleans, complex/nonnumeric/nonfinite values,
    and floating-point overflow or underflow. The result is finite and strictly
    positive, preserving both nonzero conditions of the source derivation.
    Inputs are not mutated. This converts a supplied cycle duration; it does not
    estimate a period from a signal or compute angular frequency.

    Reference: OpenStax University Physics Volume 1, section 15.1,
    https://openstax.org/books/university-physics-volume-1/pages/15-1-simple-harmonic-motion
    """
    value=np.asarray(period_seconds)
    if value.dtype.kind not in 'iuf':raise ValueError('real numeric period required')
    with np.errstate(all='raise'):
        value=value.astype(np.float64)
        if not np.all(np.isfinite(value)) or np.any(value<=0):raise ValueError('finite positive period required')
        frequency=np.asarray(np.reciprocal(value),dtype=np.float64)
    if not np.all(np.isfinite(frequency)) or np.any(frequency<=0):raise ValueError('finite positive frequency required')
    return frequency
