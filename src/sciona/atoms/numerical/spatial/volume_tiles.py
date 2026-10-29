# Adapted from lfz/DSB2017 split/combine arithmetic.
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

"""Domain-independent, ordered cubic tiles with explicit grid geometry."""
from itertools import product
import math
import numbers
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def _integer(value, name, minimum=1):
    if isinstance(value, (bool,np.bool_)) or not isinstance(value,numbers.Integral) or value<minimum:
        raise ValueError(name+' must be an integer >= '+str(minimum))
    return int(value)


def _geometry(side_len,stride,margin):
    side_len=_integer(side_len,'side_len');stride=_integer(stride,'stride');margin=_integer(margin,'margin',0)
    if side_len<=margin or side_len%stride or margin%stride:
        raise ValueError('Side must exceed margin and both must align to stride')
    return side_len,stride,margin


def witness_split_volume_tiles(volume: AbstractArray,side_len: int,max_stride: int,margin: int) -> tuple:
    side_len,_,margin=_geometry(side_len,max_stride,margin)
    if len(volume.shape)!=4 or min(volume.shape)<1:
        raise ValueError('Known positive C,Z,Y,X shape required')
    grid=tuple((n+side_len-1)//side_len for n in volume.shape[1:])
    return AbstractArray(shape=(math.prod(grid),volume.shape[0],*((side_len+2*margin,)*3)),dtype=volume.dtype),grid


@register_atom(witness_split_volume_tiles)
def split_volume_tiles(volume: NDArray[np.float64],side_len: int,max_stride: int,margin: int) -> tuple:
    """Split finite C,Z,Y,X arrays into ordered edge-padded cubic tiles and grid.

    Returns T,C,Z,Y,X tiles and a three-integer grid, with X varying fastest.
    Each tile has side_len+2*margin samples per spatial axis. Boundary values
    are extended by edge padding. Retains input dtype; no intensity transform,
    coordinate generation or model inference. Caller supplies spatial units.
    """
    side_len,_,margin=_geometry(side_len,max_stride,margin)
    volume=np.asarray(volume)
    if volume.ndim!=4 or min(volume.shape)<1 or volume.dtype.kind not in 'fiu' or not np.isfinite(volume).all():
        raise ValueError('Nonempty finite real C,Z,Y,X array required')
    grid=tuple((n+side_len-1)//side_len for n in volume.shape[1:])
    padding=[(0,0)]+[(margin,g*side_len-n+margin) for n,g in zip(volume.shape[1:],grid)]
    padded=np.pad(volume,padding,mode='edge')
    tiles=[]
    for index in product(*(range(g) for g in grid)):
        slices=tuple(slice(i*side_len,(i+1)*side_len+2*margin) for i in index)
        tiles.append(padded[(slice(None),)+slices])
    return np.stack(tiles),grid


def _combine_shape(shape,grid,side_len,stride,margin):
    side_len,stride,margin=_geometry(side_len,stride,margin)
    grid=tuple(_integer(g,'grid entry') for g in grid)
    side,border=side_len//stride,margin//stride
    extent=side+2*border
    if (len(grid)!=3 or len(shape)<4 or shape[0]!=math.prod(grid)
            or shape[1:4]!=(extent,)*3 or any(n<1 for n in shape[4:])):
        raise ValueError('Complete ordered tile grid and aligned margins required')
    return grid,side,border,tuple(g*side for g in grid)+tuple(shape[4:])


def witness_combine_volume_tiles(tiles: AbstractArray,grid: object,side_len: int,stride: int,margin: int) -> AbstractArray:
    _,_,_,shape=_combine_shape(tiles.shape,grid,side_len,stride,margin)
    return AbstractArray(shape=shape,dtype='float32')


@register_atom(witness_combine_volume_tiles)
def combine_volume_tiles(tiles: NDArray[np.float64],grid: object,side_len: int,stride: int,margin: int) -> NDArray[np.float32]:
    """Remove margins and reconstruct finite T,Z,Y,X,... tiles as Z,Y,X,... .

    Trailing payload axes are optional and opaque. Output is float32, retaining
    the padded grid extent rather than trimming to the original volume shape.
    Requires every tile in X-fastest order; no overlap averaging or missing-tile
    imputation. Side and margin are input-grid units, divided by output stride.
    Rejects nonfinite input or float32 overflow; leaves caller arrays unchanged.
    """
    tiles=np.asarray(tiles)
    grid,side,border,shape=_combine_shape(tiles.shape,grid,side_len,stride,margin)
    if tiles.dtype.kind not in 'fiu' or not np.isfinite(tiles).all():
        raise ValueError('Finite real tiles required')
    result=np.empty(shape,dtype=np.float32)
    with np.errstate(over='raise',invalid='raise'):
        for tile,index in zip(tiles,product(*(range(g) for g in grid))):
            slices=tuple(slice(i*side,(i+1)*side) for i in index)
            result[slices]=tile[border:border+side,border:border+side,border:border+side]
    if not np.isfinite(result).all():raise ValueError('Nonfinite float32 reconstruction')
    return result
