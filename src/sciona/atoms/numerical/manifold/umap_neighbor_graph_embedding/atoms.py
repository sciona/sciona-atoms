from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_build_fuzzy_simplicial_set,
    witness_optimize_umap_layout,
)

@register_atom(witness_build_fuzzy_simplicial_set, name="build_fuzzy_simplicial_set")
@icontract.require(lambda X, n_neighbors, metric=None: X.ndim == 2)
@icontract.require(lambda X, n_neighbors, metric=None: n_neighbors > 1)
@icontract.ensure(lambda result, X: result.shape == (X.shape[0], X.shape[0]))
def build_fuzzy_simplicial_set(
    X: NDArray[np.float64],
    n_neighbors: int,
    metric: str = "euclidean",
) -> scipy.sparse.coo_matrix:
    """Construct high-dimensional symmetric fuzzy neighborhood matrices matching metric geometry.

    Parameters
    ----------
    X : NDArray[np.float64]
        Input coordinates matrix of shape (n_samples, n_features).
    n_neighbors : int
        Number of nearest neighbors to consider.
    metric : str, default "euclidean"
        Distance metric to use.

    Returns
    -------
    fuzzy_graph : scipy.sparse.coo_matrix
        Sparse neighborhood graph matrix of shape (n_samples, n_samples).
    """
    import umap.umap_
    if metric is None:
        metric = "euclidean"
    fuzzy_graph, _, _ = umap.umap_.fuzzy_simplicial_set(
        X,
        n_neighbors=n_neighbors,
        random_state=np.random.RandomState(42),
        metric=metric,
    )
    return fuzzy_graph.tocoo()

@register_atom(witness_optimize_umap_layout, name="optimize_umap_layout")
@icontract.require(lambda fuzzy_graph, n_epochs, min_dist: min_dist >= 0.0)
@icontract.require(lambda fuzzy_graph, n_epochs, min_dist: n_epochs >= 0)
@icontract.ensure(lambda result, fuzzy_graph: result.shape == (fuzzy_graph.shape[0], 2))
@icontract.ensure(lambda result: np.all(np.isfinite(result)))
def optimize_umap_layout(
    fuzzy_graph: scipy.sparse.coo_matrix,
    n_epochs: int,
    min_dist: float,
) -> NDArray[np.float64]:
    """Optimize coordinates via stochastic gradient descent minimizing fuzzy set cross-entropy.

    Parameters
    ----------
    fuzzy_graph : scipy.sparse.coo_matrix
        Topological fuzzy graph of shape (n_samples, n_samples).
    n_epochs : int
        Number of optimization epochs (SGD iterations).
    min_dist : float
        Minimum distance parameter for layout.

    Returns
    -------
    embedding : NDArray[np.float64]
        Low-dimensional coordinate matrix of shape (n_samples, 2).
    """
    import umap.umap_
    a, b = umap.umap_.find_ab_params(1.0, min_dist)
    n_samples = fuzzy_graph.shape[0]
    data_dummy = np.zeros((n_samples, 1))
    embedding, _ = umap.umap_.simplicial_set_embedding(
        data=data_dummy,
        graph=fuzzy_graph,
        n_components=2,
        initial_alpha=1.0,
        a=a,
        b=b,
        gamma=1.0,
        negative_sample_rate=5,
        n_epochs=n_epochs,
        init="spectral",
        random_state=np.random.RandomState(42),
        metric="euclidean",
        metric_kwds={},
        densmap=False,
        densmap_kwds={},
        output_dens=False,
    )
    return embedding


