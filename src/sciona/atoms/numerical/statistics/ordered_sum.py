"""Domain-independent ordered linear combination without weight normalization."""
import numpy as np
from sciona.ghost.registry import register_atom


def witness_ordered_weighted_sum(*args, **kwargs):
    raise ValueError('Concrete finite numeric rows and aligned coefficients required')


@register_atom(witness_ordered_weighted_sum)
def ordered_weighted_sum(values: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Multiply rows by coefficients, then add in row order using float64.

    Require a nonempty two-dimensional real numeric array and one finite real
    coefficient per row. Convert both inputs to float64 before arithmetic.
    Start with the first product, then add each remaining product left to right;
    do not use a dot product, pairwise reduction, or normalization. Negative,
    zero and non-unit-sum coefficients are valid. Reject overflow in any product
    or partial sum. Return a fresh vector without changing either input. Rows,
    columns, coefficients, units and interpretation are caller policy.
    """
    if (not isinstance(values, np.ndarray) or values.ndim!=2 or not values.size
            or values.dtype.kind not in 'iuf' or not np.isfinite(values).all()):
        raise ValueError('Nonempty finite real numeric matrix required')
    if (not isinstance(coefficients, np.ndarray) or coefficients.shape!=(values.shape[0],)
            or coefficients.dtype.kind not in 'iuf' or not np.isfinite(coefficients).all()):
        raise ValueError('One finite real coefficient per row required')
    with np.errstate(over='ignore',invalid='ignore'):
        x=values.astype(np.float64);weights=coefficients.astype(np.float64)
        if not np.isfinite(x).all() or not np.isfinite(weights).all():
            raise ValueError('Float64 conversion overflow')
        result=x[0]*weights[0]
        if not np.isfinite(result).all():raise ValueError('Weighted product overflow')
        for row,weight in zip(x[1:],weights[1:]):
            term=row*weight
            if not np.isfinite(term).all():raise ValueError('Weighted product overflow')
            result=result+term
            if not np.isfinite(result).all():raise ValueError('Ordered partial sum overflow')
    return result
