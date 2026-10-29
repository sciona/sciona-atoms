"""Small numerical feature operations with no task-specific column ordering."""
from collections import Counter
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_empirical_entropy(samples: AbstractArray, correction: float = 0.) -> float:
    if len(samples.shape) not in (1,2) or min(samples.shape)<1:
        raise ValueError('Nonempty vector or row matrix required')
    return 0.


@register_atom(witness_empirical_entropy)
def empirical_entropy(samples: NDArray[np.float64], correction: float = 0.) -> float:
    """Natural-log empirical entropy with optional finite-count correction.

    Each scalar (1D) or entire row (2D) is a discrete symbol; finite real values
    are compared exactly, without binning. Returns -sum(p log p) plus
    correction*(observed_symbols-1)/(2*sample_count). Correction must be finite
    and nonnegative; zero is the ordinary plug-in estimator. Keeps first-seen
    symbol order for reproducible summation. No task or distribution inference.
    """
    values=np.asarray(samples)
    if values.dtype.kind not in 'iuf' or values.ndim not in (1,2) or min(values.shape)<1 or not np.isfinite(values).all():
        raise ValueError('Nonempty finite real vector or row matrix required')
    if (isinstance(correction,(bool,np.bool_)) or not isinstance(correction,(int,float,np.integer,np.floating))
            or not np.isfinite(correction) or correction<0):
        raise ValueError('Finite nonnegative correction required')
    symbols=values if values.ndim==1 else map(tuple,values)
    counts=np.asarray(list(Counter(symbols).values()),dtype=float)
    p=counts/counts.sum()
    with np.errstate(over='raise',invalid='raise'):
        result=float(-np.dot(p,np.log(p))+correction*(len(p)-1)/(2*counts.sum()))
    if not np.isfinite(result):raise ValueError('Nonfinite entropy')
    return result


def witness_mixed_power_moment(x: AbstractArray,y: AbstractArray,x_power: int,y_power: int) -> float:
    if len(x.shape)!=1 or x.shape!=y.shape or x.shape[0]<1:
        raise ValueError('Aligned nonempty vectors required')
    return 0.


@register_atom(witness_mixed_power_moment)
def mixed_power_moment(x: NDArray[np.float64],y: NDArray[np.float64],x_power: int,y_power: int) -> float:
    """Mean of x**x_power * y**y_power for aligned finite real vectors.

    Powers are nonnegative integers. Inputs are converted to float64 before
    exponentiation, preventing integer overflow. No centering or scaling is
    performed: callers supply normalized samples when that is their intended
    statistic. Rejects nonfinite arithmetic and leaves inputs unchanged.
    """
    arrays=[np.asarray(v) for v in (x,y)]
    if any(v.dtype.kind not in 'iuf' or v.ndim!=1 or v.size==0 or not np.isfinite(v).all() for v in arrays) or arrays[0].shape!=arrays[1].shape:
        raise ValueError('Aligned nonempty finite real vectors required')
    for power in (x_power,y_power):
        if isinstance(power,(bool,np.bool_)) or not isinstance(power,(int,np.integer)) or power<0:
            raise ValueError('Nonnegative integer powers required')
    a,b=[v.astype(np.float64,copy=False) for v in arrays]
    with np.errstate(over='raise',invalid='raise'):
        result=float(np.mean(a**x_power*b**y_power))
    if not np.isfinite(result):raise ValueError('Nonfinite moment')
    return result


def witness_stack_feature_columns(columns: list) -> AbstractArray:
    if not columns or any(len(c.shape)!=1 or c.shape!=columns[0].shape or c.shape[0]<1 for c in columns):
        raise ValueError('Aligned nonempty vector columns required')
    return AbstractArray(shape=(columns[0].shape[0],len(columns)),dtype='float64')


@register_atom(witness_stack_feature_columns)
def stack_feature_columns(columns: list) -> NDArray[np.float64]:
    """Stack finite real vectors in caller-specified column order, without fitting.

    Every column must have the same positive row count. No feature selection,
    name inference, normalization, imputation or broadcasting occurs. Output is
    a new float64 matrix. Column semantics and row alignment belong to callers.
    """
    if not isinstance(columns,list) or not columns:raise ValueError('Nonempty column list required')
    arrays=[np.asarray(column) for column in columns]
    if any(a.dtype.kind not in 'iuf' or a.ndim!=1 or a.size==0 or not np.isfinite(a).all() or a.shape!=arrays[0].shape for a in arrays):
        raise ValueError('Aligned nonempty finite real columns required')
    result=np.column_stack(arrays).astype(np.float64)
    if not np.isfinite(result).all():raise ValueError('Nonfinite converted columns')
    return result
