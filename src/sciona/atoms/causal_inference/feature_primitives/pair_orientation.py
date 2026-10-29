"""Explicit observation orientation and signed target augmentation for the recipe."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_orient_typed_pairs(pairs:list,variable_types:list) -> tuple:
    if not pairs or len(pairs)!=len(variable_types):raise ValueError('Aligned nonempty pair metadata required')
    xs=[];ys=[];xt=[];yt=[]
    for pair,types in zip(pairs,variable_types):
        if len(pair)!=2 or len(types)!=2 or any(t not in {'Numerical','Categorical','Binary'} for t in types):
            raise ValueError('Recognized pair types required')
        shapes=[v.shape for v in pair]
        if len(shapes[0])!=1 or shapes[0]!=shapes[1] or shapes[0][0]<3:
            raise ValueError('Aligned vectors with at least three samples required')
        xs.extend([AbstractArray(shape=shapes[0],dtype='float64') for _ in range(2)])
        ys.extend([AbstractArray(shape=shapes[0],dtype='float64') for _ in range(2)])
        xt.extend(types);yt.extend(reversed(types))
    return xs,xt,ys,yt


@register_atom(witness_orient_typed_pairs)
def orient_typed_pairs(pairs:list,variable_types:list) -> tuple:
    """Validate then expose adjacent forward/reverse samples and explicit types.

    Returns four aligned lists: x samples, x types, y samples, y types. Each
    input pair contributes (x,y) then (y,x). Vectors may differ in length across
    pairs; within a pair lengths match and are at least three. Samples are
    finite real and nonconstant; Binary requires exactly two observed values.
    Copies every output array into float64. Computes no statistics or features.
    """
    if not pairs or len(pairs)!=len(variable_types):raise ValueError('Aligned nonempty pairs required')
    xs=[];ys=[];xt=[];yt=[]
    for pair,types in zip(pairs,variable_types):
        if len(pair)!=2 or len(types)!=2 or any(t not in {'Numerical','Categorical','Binary'} for t in types):
            raise ValueError('Recognized pair types required')
        arrays=[]
        for value,kind in zip(pair,types):
            a=np.asarray(value)
            if a.dtype.kind not in 'iuf' or a.ndim!=1 or len(a)<3 or not np.isfinite(a).all() or len(set(a))<2:
                raise ValueError('Nonconstant finite real vectors required')
            if kind=='Binary' and len(set(a))!=2:raise ValueError('Binary variables require exactly two values')
            arrays.append(a.astype(float,copy=True))
        if arrays[0].shape!=arrays[1].shape:raise ValueError('Paired sample lengths must match')
        x,y=arrays
        xs.extend([x.copy(),y.copy()]);ys.extend([y.copy(),x.copy()])
        xt.extend(types);yt.extend(reversed(types))
    return xs,xt,ys,yt


def witness_augment_pair_labels(pair_labels:AbstractArray,pairs:list) -> AbstractArray:
    if pair_labels.shape!=(len(pairs),) or not pairs:raise ValueError('One label per pair required')
    return AbstractArray(shape=(2*len(pairs),),dtype='int64')


@register_atom(witness_augment_pair_labels)
def augment_pair_labels(pair_labels:NDArray[np.int64],pairs:list) -> NDArray[np.int64]:
    """Interleave each {-1,0,+1} integer label with its signed negation.

    Requires one label per nonempty input pair list; observations are not read.
    Converts to int64 before negation, including unsigned inputs, so reverse
    labels cannot wrap to large positive values. Does not modify caller labels.
    """
    labels=np.asarray(pair_labels)
    if not pairs or labels.dtype.kind not in 'iu' or labels.shape!=(len(pairs),) or not set(labels.tolist())<={-1,0,1}:
        raise ValueError('One integer causal label per pair required')
    labels=labels.astype(np.int64)
    return np.column_stack((labels,-labels)).reshape(-1)
