from __future__ import annotations
import os

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.sparse.csgraph
import icontract

from sciona.ghost.registry import register_atom
from sciona.ghost.witnesses import (
    witness_graph_laplacian, witness_graph_fourier_transform,
    witness_inverse_graph_fourier_transform,
    witness_heat_kernel_diffusion,
)
from sciona.atoms.scipy.witnesses import (
    witness_allpairsshortestpath,
    witness_minimumspanningtree,
    witness_singlesourceshortestpath,
)

_SLOW_CHECKS = os.environ.get("SCIONA_SLOW_CHECKS", "0") == "1"

SparseGraphInput = np.ndarray | scipy.sparse.spmatrix
SparseGraphOutput = np.ndarray | scipy.sparse.spmatrix


def _is_symmetric(m: SparseGraphInput, atol: float = 1e-10) -> bool:
    """Check that a sparse matrix is symmetric within tolerance."""
    if m.shape[0] != m.shape[1]:
        return False
    if scipy.sparse.issparse(m):
        diff = m - m.T
        if diff.nnz == 0:
            return True
        return bool(np.all(np.abs(diff.data) < atol))
    return bool(np.allclose(np.asarray(m), np.asarray(m).T, atol=atol))


def _is_square_graph(m: SparseGraphInput) -> bool:
    """Check that a graph adjacency object is a square matrix."""
    return len(m.shape) == 2 and m.shape[0] == m.shape[1]


def _eigenvalues_nonneg(L: SparseGraphOutput, k: int = 1) -> bool:
    """Check that the smallest eigenvalue of L is >= -epsilon (PSD check)."""
    n = L.shape[0]
    if n < 3:
        L_dense = L.toarray() if scipy.sparse.issparse(L) else np.asarray(L)
        eigvals = np.linalg.eigvalsh(L_dense)
        return bool(np.all(eigvals >= -1e-8))
    k_check = min(k, n - 2)
    eigvals = scipy.sparse.linalg.eigsh(
        scipy.sparse.csr_matrix(L),
        k=k_check,
        which="SM",
        return_eigenvectors=False,
    )
    return bool(np.all(eigvals >= -1e-8))


def _total_variation(L: scipy.sparse.spmatrix, x: np.ndarray) -> float:
    """Compute the total variation x^T L x of signal x on graph with Laplacian L."""
    Lx = L.dot(x)
    return float(x.dot(Lx))


def _dense_symmetric_laplacian(L: SparseGraphInput) -> np.ndarray:
    """Return a dense symmetric Laplacian matrix for spectral graph-signal helpers."""
    L_dense = L.toarray() if scipy.sparse.issparse(L) else np.asarray(L, dtype=float)
    return 0.5 * (L_dense + L_dense.T)


def _laplacian_eigendecomposition(L: SparseGraphInput, k: int | None) -> tuple[np.ndarray, np.ndarray]:
    n = L.shape[0]
    if k is not None and (k <= 0 or k > n):
        raise ValueError("k must be between 1 and the graph size")
    if k is None or k >= n - 1:
        eigenvalues, eigenvectors = np.linalg.eigh(_dense_symmetric_laplacian(L))
    else:
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
            scipy.sparse.csr_matrix(L),
            k=k,
            which="SM",
        )
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors


@register_atom(witness_graph_laplacian)
@icontract.require(lambda W: _is_symmetric(W), "Weight matrix W must be symmetric")
@icontract.require(lambda W: _is_square_graph(W), "Weight matrix W must be square")
@icontract.ensure(lambda result, W: result.shape == W.shape, "Laplacian shape must match input shape")
@icontract.ensure(
    lambda result: _eigenvalues_nonneg(result, k=1),
    "Graph Laplacian must be positive semi-definite",
    enabled=_SLOW_CHECKS,
)
def graph_laplacian(
    W: SparseGraphInput,
    normed: bool = False,
    use_out_degree: bool = False,
    symmetrized: bool = False,
) -> SparseGraphOutput:
    """Return SciPy's graph Laplacian for a square adjacency matrix.

    This is a narrow wrapper around ``scipy.sparse.csgraph.laplacian``.
    It intentionally returns only the Laplacian matrix, not SciPy's
    optional diagonal side output.

    Args:
        W: Symmetric weighted adjacency matrix of shape (n, n).
        normed: If True, compute the normalized Laplacian.
        use_out_degree: If True on a directed graph, use out-degree
            rather than in-degree when forming the degree matrix.
        symmetrized: If True, use SciPy's symmetrized graph option.

    Returns:
        The graph Laplacian with the same shape as ``W``.

    """
    return scipy.sparse.csgraph.laplacian(
        W,
        normed=normed,
        return_diag=False,
        use_out_degree=use_out_degree,
        symmetrized=symmetrized,
    )


@register_atom(witness_graph_fourier_transform)
@icontract.require(lambda L: _is_square_graph(L), "Laplacian L must be square")
@icontract.require(lambda L, x: x.shape[0] == L.shape[0], "Signal length must equal graph size")
@icontract.ensure(
    lambda result: isinstance(result, tuple) and len(result) == 3,
    "Must return (x_hat, eigenvalues, eigenvectors)",
)
@icontract.ensure(
    lambda result, L: result[0].shape[0] <= L.shape[0],
    "Coefficient count must be consistent with graph size",
)
def graph_fourier_transform(
    L: SparseGraphInput,
    x: np.ndarray,
    k: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute graph Fourier coefficients from a Laplacian eigenbasis.

    Projects the signal x onto the eigenvectors of the graph Laplacian L.
    This is a graph signal processing helper implemented with NumPy/SciPy
    sparse eigensolvers, not a direct scipy.sparse.csgraph API wrapper.

    Args:
        L: Symmetric graph Laplacian of shape (n, n).
        x: Graph signal of length n.
        k: Number of lowest-frequency eigenvectors to use. If None, uses all n.

    Returns:
        Tuple of (x_hat, eigenvalues, eigenvectors) where x_hat are the
        GFT coefficients, eigenvalues are the graph frequencies, and
        eigenvectors are the GFT basis vectors.

    """
    eigenvalues, eigenvectors = _laplacian_eigendecomposition(L, k)
    x_hat = eigenvectors.T @ x
    return x_hat, eigenvalues, eigenvectors


@register_atom(witness_inverse_graph_fourier_transform)
@icontract.require(lambda x_hat, eigenvectors: x_hat.shape[0] == eigenvectors.shape[1], "Coefficient count must match number of eigenvectors")
@icontract.ensure(lambda result, eigenvectors: result.shape[0] == eigenvectors.shape[0], "Output length must equal graph size")
def inverse_graph_fourier_transform(
    x_hat: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """Reconstruct a graph signal from graph Fourier coefficients.

    Reconstructs the graph signal from GFT coefficients and eigenvectors.
    Full reconstruction requires a complete eigenbasis; truncated bases return
    the least-squares projection onto the retained low-frequency subspace.

    Args:
        x_hat: GFT coefficients of length k.
        eigenvectors: GFT basis vectors of shape (n, k).

    Returns:
        Reconstructed graph signal of length n.

    """
    return eigenvectors @ x_hat


@register_atom(witness_heat_kernel_diffusion)
@icontract.require(lambda L: _is_square_graph(L), "Laplacian L must be square")
@icontract.require(lambda t: t >= 0, "Diffusion time t must be non-negative")
@icontract.require(lambda L, x: x.shape[0] == L.shape[0], "Signal length must equal graph size")
@icontract.ensure(lambda result, x: result.shape == x.shape, "Output shape must be preserved")
@icontract.ensure(
    lambda result, L, x: _total_variation(L, result) <= _total_variation(L, x) + 1e-8,
    "Heat diffusion must reduce total variation (smoothing)",
    enabled=_SLOW_CHECKS,
)
def heat_kernel_diffusion(
    L: SparseGraphInput,
    x: np.ndarray,
    t: float,
    k: int | None = None,
) -> np.ndarray:
    """Apply spectral heat-kernel diffusion to a graph signal.

    Computes exp(-t*L) @ x, which smooths the signal x over the graph
    topology. This helper uses the graph Fourier basis from the Laplacian and
    is not a direct scipy.sparse.csgraph routine.

    Args:
        L: Symmetric graph Laplacian of shape (n, n).
        x: Graph signal of length n.
        t: Diffusion time parameter. Must be >= 0. Larger values
            produce smoother outputs.
        k: Number of eigenvectors to use for the approximation.
            If None, uses all n eigenvectors.

    Returns:
        The diffused graph signal of length n.

    """
    eigenvalues, eigenvectors = _laplacian_eigendecomposition(L, k)
    x_hat = eigenvectors.T @ x
    heat_filter = np.exp(-t * eigenvalues)
    x_hat_filtered = heat_filter * x_hat
    return eigenvectors @ x_hat_filtered

@register_atom(witness_singlesourceshortestpath)
@icontract.require(lambda csgraph: _is_square_graph(csgraph), "csgraph must be a square graph matrix")
@icontract.require(lambda indices: indices is not None, "indices must select at least one source node")
@icontract.ensure(lambda result: result is not None, "Single-source shortest-path output must not be None")
def single_source_shortest_path(
    csgraph: SparseGraphInput,
    indices: np.ndarray | int = 0,
    method: str = "auto",
    directed: bool = True,
    return_predecessors: bool = False,
    unweighted: bool = False,
    overwrite: bool = False,
) -> tuple[np.ndarray, ...] | np.ndarray:
    """Compute SciPy shortest-path distances from selected source nodes."""
    return scipy.sparse.csgraph.shortest_path(
        csgraph,
        method=method,
        directed=directed,
        return_predecessors=return_predecessors,
        unweighted=unweighted,
        overwrite=overwrite,
        indices=indices,
    )

@register_atom(witness_allpairsshortestpath)
@icontract.require(lambda csgraph: _is_square_graph(csgraph), "csgraph must be a square graph matrix")
@icontract.ensure(lambda result: result is not None, "All-pairs shortest-path output must not be None")
def all_pairs_shortest_path(
    csgraph: SparseGraphInput,
    method: str = "auto",
    directed: bool = True,
    return_predecessors: bool = False,
    unweighted: bool = False,
    overwrite: bool = False,
) -> tuple[np.ndarray, ...] | np.ndarray:
    """Compute all-pairs shortest paths with SciPy's shortest-path API."""
    return scipy.sparse.csgraph.shortest_path(
        csgraph,
        method=method,
        directed=directed,
        return_predecessors=return_predecessors,
        unweighted=unweighted,
        overwrite=overwrite,
        indices=None,
    )

@register_atom(witness_minimumspanningtree)
@icontract.require(lambda csgraph: _is_square_graph(csgraph), "csgraph must be a square graph matrix")
@icontract.ensure(lambda result: result is not None, "Minimum spanning tree output must not be None")
def minimum_spanning_tree(
    csgraph: SparseGraphInput,
    overwrite: bool = False,
) -> scipy.sparse.csr_matrix:
    """Extract the minimum spanning tree of a sparse weighted graph."""
    return scipy.sparse.csgraph.minimum_spanning_tree(csgraph, overwrite=overwrite)

singlesourceshortestpath = single_source_shortest_path
allpairsshortestpath = all_pairs_shortest_path
minimumspanningtree = minimum_spanning_tree
