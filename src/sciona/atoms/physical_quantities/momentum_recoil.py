"""Real three-dimensional momentum recoil and squared Euclidean norm."""
from fractions import Fraction
import math
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.dimensions import DimensionalSignature
from sciona.ghost.registry import register_atom


def witness_momentum_recoil(incoming_momentum: AbstractArray, outgoing_momentum: AbstractArray) -> tuple[AbstractArray, AbstractArray]:
    shape = incoming_momentum.shape
    if shape != outgoing_momentum.shape or not shape or shape[-1] != 3:
        raise ValueError('Identical shapes with final axis length three required')
    return AbstractArray(shape=shape, dtype='float64'), AbstractArray(shape=shape[:-1], dtype='float64')


@register_atom(witness_momentum_recoil, dim_map={
    'incoming_momentum': DimensionalSignature(M=1, L=1, T=-1),
    'outgoing_momentum': DimensionalSignature(M=1, L=1, T=-1),
})
def momentum_recoil(incoming_momentum: NDArray[np.float64], outgoing_momentum: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return p1-p2 and its squared Euclidean norm in SI momentum units.

    Inputs are finite real float64-converted arrays of identical nonempty shape
    (...,3), in the same orthonormal frame, measured in kg*m/s. Zero and signed
    components are valid. No broadcasting, complex inputs or unit inference.
    Recoil shape matches the input; squared-norm shape omits the final axis,
    with units kg**2*m**2/s**2. Inputs are not mutated.

    Exact rational arithmetic on converted inputs computes the difference and
    sum of squared differences before independently rounding each output to
    float64. The norm is of the exact difference, not of its rounded output
    vector. This avoids cancellation in the expanded dot-product identity.
    Reject nonzero results rounding to zero, or results outside finite float64;
    exact zero and positive subnormal squared norms are supported. Rational
    per-vector evaluation prioritizes accuracy over throughput.

    Interpreting recoil as a third body's momentum assumes total initial
    momentum p1, final momentum p2+recoil and momentum conservation in that
    frame. This does not check energy conservation or solve a scattering event.
    """
    values = []
    for item in (incoming_momentum, outgoing_momentum):
        raw = np.asarray(item)
        if raw.dtype.kind not in 'iuf' or not raw.size or raw.ndim == 0 or raw.shape[-1] != 3:
            raise ValueError('Nonempty real arrays with final axis length three required')
        with np.errstate(over='raise', invalid='raise'):
            value = raw.astype(np.float64)
        if not np.all(np.isfinite(value)):
            raise ValueError('Finite momentum components required')
        values.append(value)
    incoming, outgoing = values
    if incoming.shape != outgoing.shape:
        raise ValueError('Identical momentum shapes required')
    recoil = np.empty_like(incoming)
    norm = np.empty(incoming.shape[:-1], dtype=np.float64)

    def rounded(exact):
        try:
            value = float(exact)
        except OverflowError as error:
            raise ValueError('Result outside finite float64 range') from error
        if not math.isfinite(value) or (value == 0 and exact != 0):
            raise ValueError('Result outside nonzero finite float64 range')
        return value

    for index in np.ndindex(norm.shape):
        differences = [Fraction.from_float(float(a))-Fraction.from_float(float(b))
                       for a, b in zip(incoming[index], outgoing[index])]
        recoil[index] = [rounded(d) for d in differences]
        norm[index] = rounded(sum(d*d for d in differences))
    return recoil, norm
