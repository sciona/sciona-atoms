# Adapted from lfz/DSB2017 spatial resampling arithmetic.
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

"""Spacing-driven regular-grid resampling with explicit effective spacing."""
import numbers
import numpy as np
from scipy.ndimage import zoom
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def _geometry(shape, spacing, requested_spacing, order):
    spacing = np.asarray(spacing, dtype=float)
    requested = np.asarray(requested_spacing, dtype=float)
    if (len(shape) != 3 or min(shape) < 1 or spacing.shape != (3,) or requested.shape != (3,)
            or not np.all(np.isfinite(spacing)) or not np.all(np.isfinite(requested))
            or np.any(spacing <= 0) or np.any(requested <= 0)
            or isinstance(order, (bool, np.bool_)) or not isinstance(order, numbers.Integral)
            or not 0 <= order <= 5):
        raise ValueError('Positive 3D geometry and integer interpolation order 0 through 5 required')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        target = np.round(np.asarray(shape) * spacing / requested)
    if not np.all(np.isfinite(target)) or np.any(target < 1) or np.any(target >= np.iinfo(np.intp).max):
        raise ValueError('Requested spacing produces an invalid output shape')
    return target, spacing * shape / target


def witness_resample_regular_volume(volume: AbstractArray, spacing: object,
                                    requested_spacing: object, order: int = 1) -> tuple:
    target, effective = _geometry(volume.shape, spacing, requested_spacing, order)
    if np.dtype(volume.dtype).kind not in 'fiu':
        raise ValueError('Real numeric volume required')
    return AbstractArray(shape=tuple(int(n) for n in target), dtype=volume.dtype), effective


@register_atom(witness_resample_regular_volume)
def resample_regular_volume(volume: object, spacing: object,
                            requested_spacing: object, order: int = 1) -> tuple:
    """Resample a finite 3D array; return array and effective axis spacing.

    Shape is rounded from shape*spacing/requested_spacing using NumPy rounding.
    Interpolation uses scipy.ndimage.zoom with nearest boundary extension,
    default prefiltering and grid_mode=False. Output retains the input dtype;
    integer interpolation therefore retains SciPy's integer rounding behavior.
    Effective spacing is spacing*input_shape/output_shape and can differ from
    requested spacing. Axis order and physical units are caller-defined; no
    orientation, intensity normalization, segmentation or coordinate mapping.
    """
    volume = np.asarray(volume)
    if volume.dtype.kind not in 'fiu' or not np.all(np.isfinite(volume)):
        raise ValueError('Finite real numeric volume required')
    target, effective = _geometry(volume.shape, spacing, requested_spacing, order)
    result = zoom(volume, target / volume.shape, mode='nearest', order=int(order))
    return result, effective
