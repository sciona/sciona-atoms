from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse
import pytest
from icontract.errors import ViolationError

from sciona.atoms.numerical.linear_algebra.normal_equations_ridge_solve import (
    compute_gram_matrix,
    apply_tikhonov_shift_and_solve,
)
from sciona.atoms.numerical.linear_algebra.pca_from_svd import (
    center_data,
    pca_svd_decompose,
    calculate_pca_variance,
)
from sciona.atoms.numerical.linear_algebra.graph_laplacian_spectral_embedding import (
    compute_laplacian_matrix,
    solve_smallest_eigen,
)
from sciona.atoms.numerical.linear_algebra.randomized_svd_factorization import (
    extract_random_subspace_basis,
    factorize_subspace_projection,
)
from sciona.atoms.numerical.linear_algebra.eigenvector_centrality_arnoldi import (
    solve_dominant_eigenvector,
    perron_frobenius_correct,
)
from sciona.atoms.numerical.manifold.umap_neighbor_graph_embedding import (
    build_fuzzy_simplicial_set,
    optimize_umap_layout,
)
from sciona.atoms.numerical.spatial.approximate_neighbor_search_hnsw import (
    build_hnsw_index,
    query_hnsw_index,
)

def test_normal_equations_ridge_solve_behavior() -> None:
    A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    b = np.array([1.0, 2.0, 3.0])
    
    # Test Gram matrix computation
    Gram, Ab = compute_gram_matrix(A, b)
    assert Gram.shape == (2, 2)
    assert Ab.shape == (2,)
    np.testing.assert_allclose(Gram, A.T @ A)
    np.testing.assert_allclose(Ab, A.T @ b)

    # Test solving with shift
    x = apply_tikhonov_shift_and_solve(Gram, Ab, alpha=0.1)
    assert x.shape == (2,)
    assert np.all(np.isfinite(x))

    # Test contract checks
    with pytest.raises(ViolationError):
        apply_tikhonov_shift_and_solve(Gram, Ab, alpha=-0.1)

def test_pca_from_svd_behavior() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    
    # Center data
    X_centered, mean = center_data(X)
    assert X_centered.shape == X.shape
    assert mean.shape == (2,)
    np.testing.assert_allclose(mean, np.mean(X, axis=0))
    np.testing.assert_allclose(X_centered, X - mean)

    # Decompose
    components, s_vals = pca_svd_decompose(X_centered, n_components=1)
    assert components.shape == (1, 2)
    assert s_vals.shape == (1,)

    # Variance
    var = calculate_pca_variance(s_vals, n_samples=3)
    assert var.shape == (1,)
    np.testing.assert_allclose(var, (s_vals ** 2) / 2.0)

    # Contract violation
    with pytest.raises(ViolationError):
        pca_svd_decompose(X_centered, n_components=3)

def test_graph_laplacian_spectral_embedding_behavior() -> None:
    # 3-node path graph: 0-1-2
    adj = scipy.sparse.csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    
    # Compute laplacian
    L = compute_laplacian_matrix(adj, normalized=False)
    assert L.shape == (3, 3)
    np.testing.assert_allclose(L.toarray(), [[1, -1, 0], [-1, 2, -1], [0, -1, 1]])

    # Solve smallest eigenvalues
    e_vals, e_vecs = solve_smallest_eigen(L, k=2)
    assert e_vals.shape == (2,)
    assert e_vecs.shape == (3, 2)
    assert np.all(np.isfinite(e_vals))

def test_randomized_svd_factorization_behavior() -> None:
    A = np.random.rand(10, 5)
    
    # Extract basis
    Q = extract_random_subspace_basis(A, k=2, p=1, n_iter=2)
    assert Q.shape == (10, 3)

    # Factorize
    U_k, s_k, Vh_k = factorize_subspace_projection(A, Q, k=2)
    assert U_k.shape == (10, 2)
    assert s_k.shape == (2,)
    assert Vh_k.shape == (2, 5)

def test_eigenvector_centrality_arnoldi_behavior() -> None:
    adj = scipy.sparse.csr_matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    
    # Solve dominant eigenvector
    val, vec = solve_dominant_eigenvector(adj, max_iter=100)
    assert isinstance(val, (complex, np.complex128))
    assert vec.shape == (3,)

    # Correct via Perron-Frobenius
    scores = perron_frobenius_correct(vec)
    assert scores.shape == (3,)
    assert np.all(scores >= 0)
    np.testing.assert_allclose(np.linalg.norm(scores), 1.0)

def test_umap_neighbor_graph_embedding_behavior() -> None:
    pytest.importorskip("umap")
    X = np.random.rand(15, 4)
    
    # Build fuzzy simplicial set
    g = build_fuzzy_simplicial_set(X, n_neighbors=5)
    assert g.shape == (15, 15)
    assert scipy.sparse.isspmatrix(g)

    # Optimize layout
    emb = optimize_umap_layout(g, n_epochs=10, min_dist=0.1)
    assert emb.shape == (15, 2)
    assert np.all(np.isfinite(emb))

def test_approximate_neighbor_search_hnsw_behavior() -> None:
    pytest.importorskip("hnswlib")
    data = np.random.rand(20, 5)
    
    # Build index
    idx = build_hnsw_index(data, M=12, ef_construction=100)
    assert idx is not None

    # Query index
    queries = np.random.rand(3, 5)
    dists, indices = query_hnsw_index(idx, queries, k=4, ef_search=20)
    assert dists.shape == (3, 4)
    assert indices.shape == (3, 4)
    assert dists.dtype == np.float64
    assert indices.dtype == np.int64


def test_cdf_regression_head_behavior() -> None:
    from sciona.atoms.numerical.statistics.regression import cdf_regression_head

    x = np.array([[0.0, 0.0], [1.0, np.log(2.0)]])
    y = np.array([0.0, 1.0])
    
    # Test normal CDF
    probs_normal = cdf_regression_head(x, y, distribution="normal")
    assert probs_normal.shape == (2,)
    # normal: mean 0, scale 1 -> at 0.0 is 0.5
    # normal: mean 1, scale 2 -> at 1.0 is 0.5
    np.testing.assert_allclose(probs_normal, [0.5, 0.5], atol=1e-6)

    # Test logistic CDF
    probs_logistic = cdf_regression_head(x, y, distribution="logistic")
    assert probs_logistic.shape == (2,)
    np.testing.assert_allclose(probs_logistic, [0.5, 0.5], atol=1e-6)

    # Contract violation
    with pytest.raises(ViolationError):
        cdf_regression_head(x, y, distribution="invalid_dist")


def test_mlp_regression_head_behavior() -> None:
    from sciona.atoms.numerical.statistics.regression import mlp_regression_head

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    w1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    b1 = np.array([-0.5, -0.5])
    w2 = np.array([[2.0], [2.0]])
    b2 = np.array([1.0])

    # forward:
    # layer 1: x @ w1 + b1 = [[0.5, 1.5], [2.5, 3.5]]
    # relu: no change (all positive)
    # layer 2: [[0.5, 1.5], [2.5, 3.5]] @ [[2], [2]] + 1 = [[5.0], [13.0]]
    result = mlp_regression_head(x, [w1, w2], [b1, b2], activation="relu")
    assert result.shape == (2, 1)
    np.testing.assert_allclose(result, [[5.0], [13.0]])

    # Contract violation (dimension mismatch)
    w_bad = np.array([[1.0, 0.0, 0.0]])
    with pytest.raises(ViolationError):
        mlp_regression_head(x, [w_bad], [b1], activation="relu")


def test_jaccard_similarity_threshold_behavior() -> None:
    from sciona.atoms.numerical.spatial.jaccard_similarity_threshold import jaccard_similarity_threshold

    sets = [
        {1, 2, 3},
        {2, 3, 4},
        {5, 6}
      ]
    # J(0, 1) = |{2,3}| / |{1,2,3,4}| = 2/4 = 0.5
    # J(0, 2) = 0.0
    # J(1, 2) = 0.0
    results = jaccard_similarity_threshold(sets, threshold=0.4)
    assert results == [(0, 1, 0.5)]

    results_low = jaccard_similarity_threshold(sets, threshold=0.0)
    assert len(results_low) == 3

    # Contract violation
    with pytest.raises(ViolationError):
        jaccard_similarity_threshold(sets, threshold=1.5)
