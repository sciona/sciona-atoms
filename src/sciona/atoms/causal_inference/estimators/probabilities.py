"""Explicit class-labelled probability conversion for causal direction models."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_causal_direction_probability_score(
    probabilities: AbstractArray, class_labels: AbstractArray,
) -> AbstractArray:
    if len(probabilities.shape) != 2 or probabilities.shape[1] != 3:
        raise ValueError('three probability columns required')
    if class_labels.shape != (3,):
        raise ValueError('three class labels required')
    return AbstractArray(shape=(probabilities.shape[0],), dtype='float64')


@register_atom(witness_causal_direction_probability_score)
def causal_direction_probability_score(
    probabilities: NDArray[np.float64], class_labels: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Return P(label=1) minus P(label=-1) for each classifier prediction.

    Rows are predictions; the three columns correspond to the supplied labels
    -1, 0 and 1 in any order. Label 0 denotes independence. Inputs must be finite
    real numeric arrays, with probabilities in [0,1] and each row summing to one
    within absolute tolerance 1e-8. Invalid distributions are rejected without
    normalization. Returns a float64 vector in [-1,1], without input mutation.

    This implements the three-class score conversion used by the upstream
    cause-effect estimator. It does not train a classifier, establish causality,
    or enforce symmetry between reversed observation pairs.

    Source: https://github.com/jarfo/cause-effect/blob/master/estimator.py
    (Pipeline.predict, three-class probability conversion).
    """
    p = np.asarray(probabilities)
    labels = np.asarray(class_labels)
    if p.dtype.kind not in 'iuf' or labels.dtype.kind not in 'iu':
        raise ValueError('real probabilities and integer class labels required')
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] != 3:
        raise ValueError('nonempty matrix with three probability columns required')
    if labels.shape != (3,) or set(labels.tolist()) != {-1, 0, 1}:
        raise ValueError('class labels must contain -1, 0, 1 exactly once')
    with np.errstate(all='raise'):
        p = p.astype(np.float64)
        if not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
            raise ValueError('finite probabilities in [0,1] required')
        if not np.allclose(p.sum(axis=1), 1., rtol=0., atol=1e-8):
            raise ValueError('probability rows must sum to one')
        positive = int(np.flatnonzero(labels == 1)[0])
        negative = int(np.flatnonzero(labels == -1)[0])
        return p[:, positive] - p[:, negative]


def witness_binary_positive_class_probability(probabilities: AbstractArray, class_labels: AbstractArray) -> AbstractArray:
    if len(probabilities.shape)!=2 or probabilities.shape[1]!=2 or class_labels.shape!=(2,):
        raise ValueError('two probability columns and labels required')
    return AbstractArray(shape=(probabilities.shape[0],),dtype='float64')


@register_atom(witness_binary_positive_class_probability)
def binary_positive_class_probability(probabilities: NDArray[np.float64], class_labels: NDArray[np.int64]) -> NDArray[np.float64]:
    """Select P(label=1) from a binary {0,1} or {-1,1} classifier.

    Explicit labels determine column order. Reject nonfinite, negative,
    nonnormalized or otherwise malformed distributions. Does not rescale
    a binary probability into a signed score or infer classifier semantics.
    """
    p=np.asarray(probabilities)
    labels=np.asarray(class_labels)
    if p.dtype.kind not in 'iuf' or p.ndim!=2 or p.shape[0]==0 or p.shape[1]!=2:
        raise ValueError('nonempty real two-column probabilities required')
    if labels.dtype.kind not in 'iu' or labels.shape!=(2,) or set(labels.tolist()) not in ({0,1},{-1,1}):
        raise ValueError('binary integer labels {0,1} or {-1,1} required')
    if not np.all(np.isfinite(p)) or np.any(p<0) or np.any(p>1) or not np.allclose(p.sum(axis=1),1.,rtol=0,atol=1e-8):
        raise ValueError('finite normalized probabilities required')
    return p[:,int(np.flatnonzero(labels==1)[0])].astype(np.float64,copy=True)
