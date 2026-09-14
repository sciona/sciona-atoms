"""Ordered real quadratic roots with exact float64 discriminant classification."""
from decimal import Decimal,localcontext
from fractions import Fraction
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_real_quadratic_roots(a: AbstractArray,b: AbstractArray,c: AbstractArray) -> tuple[AbstractArray,AbstractArray]:
    if a.shape!=b.shape or a.shape!=c.shape:raise ValueError('Coefficient shapes must match')
    return (AbstractArray(shape=a.shape,dtype='float64'),AbstractArray(shape=a.shape,dtype='float64'))


@register_atom(witness_real_quadratic_roots)
def real_quadratic_roots(a: NDArray[np.float64],b: NDArray[np.float64],c: NDArray[np.float64]) -> tuple[NDArray[np.float64],NDArray[np.float64]]:
    """Return lower and upper real roots of a*x**2+b*x+c=0.

    Coefficients must be finite real float64-compatible scalars or nonempty
    arrays of identical shape, with a nonzero. No broadcasting or implicit
    linear fallback. Boolean, complex and nonnumeric inputs fail. Units must
    be consistent across polynomial terms; this routine does not infer them.
    Discriminants are classified exactly for the float64-converted inputs.
    Negative discriminants fail rather than being clamped to repeated roots.
    Repeated roots appear twice, exact zero roots are supported, and sign of a
    does not change the lower/upper output convention. Inputs are not mutated.

    Exact rational discriminants avoid overflow and cancellation in b²-4ac.
    A 100-digit Decimal square root and stable q=-0.5*(b+sign(b)*sqrt(D))
    followed by q/a and c/q avoid cancellation of the smaller root. This is
    intentionally a scalar-per-element reference implementation, not a fast
    vectorized solver. Positive or negative subnormal outputs are supported;
    nonzero roots rounding to zero and roots exceeding finite float64 fail.
    Outputs describe numerical coefficients, not uncertainty in measurements.
    This is an explicitly corrected implementation, not parity with the
    defective pinned PDG quadratic proof.
    """
    values=[]
    for item in [a,b,c]:
        raw=np.asarray(item)
        if raw.dtype.kind not in 'iuf' or not raw.size:raise ValueError('Nonempty real coefficients required')
        with np.errstate(over='raise',invalid='raise'):value=raw.astype(np.float64)
        if not np.all(np.isfinite(value)):raise ValueError('Finite coefficients required')
        values.append(value)
    aa,bb,cc=values
    if aa.shape!=bb.shape or aa.shape!=cc.shape:raise ValueError('Coefficient shapes must match')
    if np.any(aa==0):raise ValueError('Nonzero quadratic coefficient required')
    lower=np.empty(aa.shape,dtype=np.float64);upper=np.empty_like(lower)
    with localcontext() as ctx:
        ctx.prec=100
        for i,(av,bv,cv) in enumerate(zip(aa.flat,bb.flat,cc.flat)):
            ar,br,cr=map(lambda x:Fraction.from_float(float(x)),(av,bv,cv))
            discriminant=br*br-4*ar*cr
            if discriminant<0:raise ValueError('Negative discriminant has no real roots')
            def dec(r):return Decimal(r.numerator)/Decimal(r.denominator)
            A,B,C=map(dec,(ar,br,cr))
            if discriminant==0:
                roots=[-B/(2*A)]*2
            else:
                radical=dec(discriminant).sqrt()
                q=-(B+(radical if br>=0 else -radical))/2
                roots=[q/A,C/q]
            rounded=[]
            for root in roots:
                value=float(root)
                if not np.isfinite(value) or (value==0 and root!=0):raise ValueError('Root outside nonzero finite float64 range')
                rounded.append(value)
            lower.flat[i],upper.flat[i]=sorted(rounded)
    return lower,upper
