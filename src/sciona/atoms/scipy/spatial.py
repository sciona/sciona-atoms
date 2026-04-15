"""SciPy spatial atom wrappers for the general-provider scratch repo."""

from __future__ import annotations

import icontract
import numpy as np
import scipy.spatial
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_voronoi_tessellation(
    points: AbstractArray,
    furthest_site: AbstractArray,
    incremental: AbstractArray,
    qhull_options: AbstractArray,
) -> AbstractArray:
    """Return witness metadata for Voronoi tessellation without executing SciPy."""
    return AbstractArray(shape=points.shape, dtype="float64")


def witness_delaunay_triangulation(
    points: AbstractArray,
    furthest_site: AbstractArray,
    incremental: AbstractArray,
    qhull_options: AbstractArray,
) -> AbstractArray:
    """Return witness metadata for Delaunay triangulation without executing SciPy."""
    return AbstractArray(shape=points.shape, dtype="float64")


@register_atom(witness_voronoi_tessellation)  # type: ignore[untyped-decorator]
@icontract.require(lambda points: points is not None, "points cannot be None")
@icontract.ensure(lambda result: result is not None, "Voronoi tessellation output must not be None")
def voronoi_tessellation(
    points: np.ndarray,
    furthest_site: bool = False,
    incremental: bool = False,
    qhull_options: str | None = None,
) -> scipy.spatial.Voronoi:
    """Compute the Voronoi diagram for a set of input points."""
    return scipy.spatial.Voronoi(
        points,
        furthest_site=furthest_site,
        incremental=incremental,
        qhull_options=qhull_options,
    )


@register_atom(witness_delaunay_triangulation)  # type: ignore[untyped-decorator]
@icontract.require(lambda points: points is not None, "points cannot be None")
@icontract.ensure(lambda result: result is not None, "Delaunay triangulation output must not be None")
def delaunay_triangulation(
    points: np.ndarray,
    furthest_site: bool = False,
    incremental: bool = False,
    qhull_options: str | None = None,
) -> scipy.spatial.Delaunay:
    """Compute the Delaunay triangulation for a set of input points."""
    return scipy.spatial.Delaunay(
        points,
        furthest_site=furthest_site,
        incremental=incremental,
        qhull_options=qhull_options,
    )


# Compatibility aliases matching the original source family naming.
voronoitessellation = voronoi_tessellation
delaunaytriangulation = delaunay_triangulation


__all__ = [
    "voronoi_tessellation",
    "delaunay_triangulation",
    "voronoitessellation",
    "delaunaytriangulation",
    "witness_voronoi_tessellation",
    "witness_delaunay_triangulation",
]

