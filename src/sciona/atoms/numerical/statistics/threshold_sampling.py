# Sampling arithmetic adapted from lfz/DSB2017, commit
# 0ac3eb9f383bf0127c587e0502c59ac84d9ba6a2, via sciona.dsb_components.
# MIT License
#
# Copyright (c) 2023 lfz
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Explicit row repetition and sequential score sampling without domain units."""
import numbers
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(name + ' must be a nonnegative integer')
    return int(value)


def witness_repeat_rows_by_threshold(rows: AbstractArray, column: int,
                                     thresholds: object, multiplicities: object) -> AbstractArray:
    raise ValueError('Threshold repetition requires concrete values to determine output cardinality')


@register_atom(witness_repeat_rows_by_threshold)
def repeat_rows_by_threshold(rows: NDArray[np.float64], column: int,
                             thresholds: object, multiplicities: object) -> NDArray[np.float64]:
    """Repeat each row by the sum of multiplicities for strictly exceeded thresholds.

    Thresholds are finite, strictly increasing caller-unit values, including
    negative values when appropriate. Multiplicities are nonnegative integers.
    Rows below or equal to the first threshold disappear; there is no implicit
    base copy. Retains input row order and dtype, returning a fresh matrix.
    Empty row populations are valid. No labels, units or sampling policy inferred.
    """
    values = np.asarray(rows)
    limits = np.asarray(thresholds)
    column = _integer(column, 'column')
    if (values.ndim != 2 or values.shape[1] < 1 or column >= values.shape[1]
            or values.dtype.kind not in 'iuf' or not np.isfinite(values).all()):
        raise ValueError('Finite real row matrix and valid column required')
    if (limits.ndim != 1 or not limits.size or limits.dtype.kind not in 'iuf'
            or not np.isfinite(limits).all() or np.any(limits[1:] <= limits[:-1])):
        raise ValueError('Nonempty finite strictly increasing thresholds required')
    if not isinstance(multiplicities, (list, tuple, np.ndarray)):
        raise ValueError('Multiplicity sequence required')
    counts_per_threshold = [_integer(v, 'multiplicity') for v in multiplicities]
    if len(counts_per_threshold) != len(limits):
        raise ValueError('One multiplicity per threshold required')
    if sum(counts_per_threshold) > np.iinfo(np.intp).max:
        raise ValueError('Multiplicity sum exceeds index capacity')
    counts = (values[:, column, None] > limits) @ np.asarray(counts_per_threshold, dtype=np.intp)
    if sum(map(int, counts)) > np.iinfo(np.intp).max // values.shape[1]:
        raise ValueError('Output size exceeds index capacity')
    return np.repeat(values, counts, axis=0)


def _sampling_controls(k, temperature, probability_floor, random_state, rng):
    k = _integer(k, 'k')
    for value, name in [(temperature, 'temperature'), (probability_floor, 'probability_floor')]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real) or not np.isfinite(value) or value <= 0:
            raise ValueError(name + ' must be finite and positive')
    if probability_floor > 1:
        raise ValueError('Probability floor must not exceed one')
    if rng is not None and random_state is not None:
        raise ValueError('Supply an RNG or a seed, not both')
    if random_state is not None and _integer(random_state, 'random_state') >= 2**32:
        raise ValueError('Seed must fit uint32')
    if rng is not None and not isinstance(rng, np.random.RandomState):
        raise ValueError('A NumPy RandomState is required for explicit RNG continuity')
    return k


def witness_sequential_softmax_indices(scores: AbstractArray, k: int,
                                      temperature: float = 1., probability_floor: float = 1e-5,
                                      random_state: object = None, rng: object = None) -> AbstractArray:
    k = _sampling_controls(k, temperature, probability_floor, random_state, rng)
    if len(scores.shape) != 1 or scores.shape[0] < 0:
        raise ValueError('Known vector length required')
    return AbstractArray(shape=(min(k, scores.shape[0]),), dtype='int64')


@register_atom(witness_sequential_softmax_indices)
def sequential_softmax_indices(scores: NDArray[np.float64], k: int,
                               temperature: float = 1., probability_floor: float = 1e-5,
                               random_state: object = None, rng: object = None) -> NDArray[np.int64]:
    """Select indices without replacement, recomputing floored softmax each draw.

    Each draw divides remaining scores by temperature, normalizes exponentials,
    floors probabilities, then normalizes again. The floor applies before the
    second normalization, not as a guaranteed final lower bound. If k covers
    all candidates, return original order without consuming RNG state. Empty
    vectors and k=0 are valid. Uses a local RandomState from a seed, or advances
    the supplied RandomState; never uses NumPy's global RNG. No model fitting.
    """
    k = _sampling_controls(k, temperature, probability_floor, random_state, rng)
    values = np.asarray(scores)
    if values.ndim != 1 or values.dtype.kind not in 'iuf' or not np.isfinite(values).all():
        raise ValueError('Finite real score vector required')
    values = values.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Scores must be representable as float64')
    if k >= len(values):
        return np.arange(len(values), dtype=np.int64)
    generator = np.random.RandomState(random_state) if rng is None else rng
    remaining = np.arange(len(values))
    selected = []
    for _ in range(k):
        with np.errstate(over='ignore', invalid='ignore'):
            scaled = values[remaining] / temperature
            shifted = scaled - np.max(scaled)
        if not np.isfinite(shifted).all():
            with np.errstate(over='ignore'):
                shifted = (values[remaining] - np.max(values[remaining])) / temperature
        weights = np.exp(shifted)
        probabilities = np.maximum(weights / weights.sum(), probability_floor)
        probabilities = np.asarray(probabilities / probabilities.sum(), dtype=float)
        index = int(generator.choice(len(remaining), size=1, replace=False, p=probabilities)[0])
        selected.append(remaining[index])
        remaining = np.delete(remaining, index)
    return np.asarray(selected, dtype=np.int64)
