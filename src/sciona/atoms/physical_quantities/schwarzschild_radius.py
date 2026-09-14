"""Schwarzschild length scale from positive mass and explicit SI constants."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.dimensions import DimensionalSignature
from sciona.ghost.registry import register_atom


def witness_schwarzschild_radius(mass_kg: AbstractArray, gravitational_constant: float,
                                 speed_of_light: float) -> AbstractArray:
    return AbstractArray(shape=mass_kg.shape,dtype='float64')


@register_atom(witness_schwarzschild_radius,dim_map={
    'mass_kg':DimensionalSignature(M=1),
    'gravitational_constant':DimensionalSignature(L=3,M=-1,T=-2),
    'speed_of_light':DimensionalSignature(L=1,T=-1),
    'return':DimensionalSignature(L=1),
})
def schwarzschild_radius(mass_kg: NDArray[np.float64], gravitational_constant: float,
                         speed_of_light: float) -> NDArray[np.float64]:
    """Compute 2*G*M/c**2 in metres, preserving the mass input shape.

    Supply positive finite mass in kilograms, scalar G in m**3/(kg*s**2), and
    scalar vacuum light speed c in m/s. Constants are explicit for provenance;
    this function neither chooses their values nor infers input units.
    Reject boolean, complex, nonnumeric, empty or nonpositive inputs and
    results outside the positive finite float64 range. Positive subnormal
    outputs are permitted with the corresponding loss of relative precision.
    Binary scaling avoids intermediate overflow/underflow. Inputs are not
    mutated and constants cannot broadcast or change output shape.

    This returns a length scale. Its event-horizon interpretation assumes the
    Schwarzschild regime: spherical, nonrotating, uncharged, asymptotically
    flat vacuum exterior. Mass alone does not establish that an object is a
    black hole. The Newtonian escape-speed argument reproduces the formula
    but is not a derivation of general relativity or a model of light motion.
    Reference: OpenStax University Physics Volume 1, section 13.7,
    https://openstax.org/books/university-physics-volume-1/pages/13-7-einsteins-theory-of-gravity
    """
    def positive(value, scalar=False):
        raw=np.asarray(value)
        if raw.dtype.kind not in 'iuf' or not raw.size or (scalar and raw.ndim):
            raise ValueError('Positive real mass and scalar constants required')
        with np.errstate(over='raise',invalid='raise'):
            result=raw.astype(np.float64)
        if np.any(~np.isfinite(result)) or np.any(result<=0):
            raise ValueError('Finite positive inputs required')
        return result
    mass=positive(mass_kg)
    g=positive(gravitational_constant,True)
    c=positive(speed_of_light,True)
    mf,me=np.frexp(mass)
    gf,ge=np.frexp(g)
    cf,ce=np.frexp(c)
    with np.errstate(over='raise',invalid='raise',divide='raise',under='ignore'):
        result=np.asarray(np.ldexp((2*gf*mf)/(cf*cf),me+ge-2*ce),dtype=np.float64)
    if np.any(~np.isfinite(result)) or np.any(result<=0):
        raise ValueError('Radius outside positive finite float64 range')
    return result
