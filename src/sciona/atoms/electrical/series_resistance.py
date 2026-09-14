"""Ohmic series resistance under an explicit nonzero-current derivation domain."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.dimensions import DimensionalSignature
from sciona.ghost.registry import register_atom

RESISTANCE_DIMENSION = DimensionalSignature(M=1,L=2,T=-3,I=-2)


def witness_series_resistance(resistance_a: AbstractArray, resistance_b: AbstractArray,
                              current: AbstractArray) -> AbstractArray:
    return AbstractArray(shape=np.broadcast_shapes(resistance_a.shape,resistance_b.shape,current.shape),dtype='float64')


@register_atom(witness_series_resistance, dim_map={
    'resistance_a':RESISTANCE_DIMENSION, 'resistance_b':RESISTANCE_DIMENSION,
    'current':DimensionalSignature(I=1), 'return':RESISTANCE_DIMENSION,
})
def series_resistance(resistance_a: NDArray[np.float64], resistance_b: NDArray[np.float64],
                      current: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return equivalent resistance in ohms for two ohmic series components.

    Resistance inputs are nonnegative ohms; current is nonzero amperes. Inputs
    must be finite real numeric values with NumPy-compatible broadcast shapes.
    The current guard preserves the domain of the supporting division-based
    derivation even though the final sum is independent of current. Returns a
    float64 array and rejects overflow. Inputs are never mutated.

    Model assumptions: one series branch, the same current through both ohmic
    components, and additive voltage drops. No claim for nonlinear devices or
    other circuit topologies. Reference: OpenStax University Physics Volume 2,
    section 10.2, https://openstax.org/books/university-physics-volume-2/pages/10-2-resistors-in-series-and-parallel
    """
    values=[]
    for argument in (resistance_a,resistance_b,current):
        value=np.asarray(argument)
        if value.dtype.kind not in 'iuf':raise ValueError('finite real numeric inputs required')
        value=value.astype(np.float64)
        if not np.all(np.isfinite(value)):raise ValueError('finite real numeric inputs required')
        values.append(value)
    a,b,i=np.broadcast_arrays(*values)
    if np.any(a<0) or np.any(b<0):raise ValueError('resistance must be nonnegative')
    if np.any(i==0):raise ValueError('current must be nonzero for this derivation')
    with np.errstate(over='raise',invalid='raise'):
        result=np.add(a,b,dtype=np.float64)
    if not np.all(np.isfinite(result)):raise ValueError('finite output required')
    return result
