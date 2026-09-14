"""Circular Newtonian two-body period with explicit SI inputs."""
from decimal import Decimal, localcontext
import math

import numpy as np
from numpy.typing import NDArray

from sciona.ghost.abstract import AbstractArray
from sciona.ghost.dimensions import DimensionalSignature
from sciona.ghost.registry import register_atom


def witness_circular_orbital_period(separation_metres: AbstractArray, mass1_kg: AbstractArray,
                                  mass2_kg: AbstractArray, gravitational_constant: float) -> AbstractArray:
    if separation_metres.shape != mass1_kg.shape or separation_metres.shape != mass2_kg.shape:
        raise ValueError('Separation and masses must have identical shapes')
    return AbstractArray(shape=separation_metres.shape, dtype='float64')


@register_atom(witness_circular_orbital_period, dim_map={
    'separation_metres': DimensionalSignature(L=1),
    'mass1_kg': DimensionalSignature(M=1), 'mass2_kg': DimensionalSignature(M=1),
    'gravitational_constant': DimensionalSignature(L=3, M=-1, T=-2),
    'return': DimensionalSignature(T=1),
})
def circular_orbital_period(separation_metres: NDArray[np.float64], mass1_kg: NDArray[np.float64],
                           mass2_kg: NDArray[np.float64], gravitational_constant: float) -> NDArray[np.float64]:
    """Return 2*pi*sqrt(r**3/(G*(m1+m2))) seconds for a circular orbit.

    Positive finite SI separation and masses must have identical nonempty
    shapes; G is an explicit positive finite scalar in m**3/(kg*s**2).
    No broadcasting, unit inference or choice of physical constant is made.
    Inputs are converted to float64; boolean, complex and nonnumeric inputs
    are rejected. Output shape is preserved and inputs are not mutated.

    A 100-digit Decimal evaluation per element avoids intermediate binary64
    overflow and underflow, including overflow of the mass sum or r**3. Pi
    uses Python's binary64 math.pi approximation. Reject outputs rounding to
    zero or infinity; positive subnormals are supported. This implementation
    prioritizes numerical range over throughput.

    The physical model is two isolated Newtonian point masses in a circular
    orbit. r is their separation, not either body's barycentric radius. This
    does not determine whether input objects actually satisfy that model.
    No relativistic, external-perturbation or extended-body correction.
    """
    def positive(value, scalar=False):
        raw = np.asarray(value)
        if raw.dtype.kind not in 'iuf' or not raw.size or (scalar and raw.ndim != 0):
            raise ValueError('Positive real arrays and scalar G required')
        with np.errstate(over='raise', invalid='raise'):
            result = raw.astype(np.float64)
        if np.any(~np.isfinite(result)) or np.any(result <= 0):
            raise ValueError('Finite positive inputs required')
        return result

    radius, mass1, mass2 = map(positive, (separation_metres, mass1_kg, mass2_kg))
    gravity = positive(gravitational_constant, scalar=True)
    if radius.shape != mass1.shape or radius.shape != mass2.shape:
        raise ValueError('Separation and masses must have identical shapes')
    result = np.empty(radius.shape, dtype=np.float64)
    with localcontext() as context:
        context.prec = 100
        g = Decimal.from_float(float(gravity))
        twice_pi = 2*Decimal.from_float(math.pi)
        for index in np.ndindex(radius.shape):
            r, a, b = (Decimal.from_float(float(v[index])) for v in (radius, mass1, mass2))
            value = float(twice_pi*(r**3/(g*(a+b))).sqrt())
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Period outside positive finite float64 range')
            result[index] = value
    return result
