from __future__ import annotations

import numpy as np
import pytest


def test_all_eight_functions_import() -> None:
    from sciona.atoms.causal_inference.feature_primitives.atoms import (
        discretize_and_bin,
        hsic_independence_test,
        igci_asymmetry_score,
        knn_entropy_estimator,
        normalized_error_probability,
        polyfit_nonlinearity_asymmetry,
        polyfit_residual_error,
        uniform_divergence,
    )

    assert callable(igci_asymmetry_score)
    assert callable(hsic_independence_test)
    assert callable(knn_entropy_estimator)
    assert callable(uniform_divergence)
    assert callable(normalized_error_probability)
    assert callable(discretize_and_bin)
    assert callable(polyfit_nonlinearity_asymmetry)
    assert callable(polyfit_residual_error)


@pytest.fixture()
def numeric_pair():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100).astype(np.float64)
    y = (2.0 * x + rng.standard_normal(100) * 0.3).astype(np.float64)
    return x, y


def test_igci_asymmetry_score_returns_finite_scalar(numeric_pair) -> None:
    from sciona.atoms.causal_inference.feature_primitives.atoms import igci_asymmetry_score

    x, y = numeric_pair
    result = igci_asymmetry_score(x, "Numerical", y, "Numerical")
    assert isinstance(result, float)
    assert np.isfinite(result)


def test_hsic_independence_test_returns_finite_scalar(numeric_pair) -> None:
    from sciona.atoms.causal_inference.feature_primitives.atoms import hsic_independence_test

    x, y = numeric_pair
    result = hsic_independence_test(x, y)
    assert isinstance(result, float)
    assert np.isfinite(result)


def test_knn_entropy_estimator_returns_finite_scalar(numeric_pair) -> None:
    from sciona.atoms.causal_inference.feature_primitives.atoms import knn_entropy_estimator

    x, _y = numeric_pair
    result = knn_entropy_estimator(x, "Numerical")
    assert isinstance(result, float)
    assert np.isfinite(result)


def test_uniform_divergence_returns_finite_scalar(numeric_pair) -> None:
    from sciona.atoms.causal_inference.feature_primitives.atoms import uniform_divergence

    x, _y = numeric_pair
    result = uniform_divergence(x, "Numerical")
    assert isinstance(result, float)
    assert np.isfinite(result)


def test_normalized_error_probability_returns_value_in_0_1(numeric_pair) -> None:
    from sciona.atoms.causal_inference.feature_primitives.atoms import normalized_error_probability

    x, y = numeric_pair
    result = normalized_error_probability(x, "Numerical", y, "Numerical")
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_discretize_and_bin_returns_same_shape(numeric_pair) -> None:
    from sciona.atoms.causal_inference.feature_primitives.atoms import discretize_and_bin

    x, _y = numeric_pair
    result = discretize_and_bin(x, "Numerical")
    assert isinstance(result, np.ndarray)
    assert result.shape == x.shape


def test_polyfit_nonlinearity_asymmetry_returns_finite_scalar(numeric_pair) -> None:
    from sciona.atoms.causal_inference.feature_primitives.atoms import polyfit_nonlinearity_asymmetry

    x, y = numeric_pair
    result = polyfit_nonlinearity_asymmetry(x, "Numerical", y, "Numerical")
    assert isinstance(result, float)
    assert np.isfinite(result)


def test_polyfit_residual_error_returns_finite_scalar(numeric_pair) -> None:
    from sciona.atoms.causal_inference.feature_primitives.atoms import polyfit_residual_error

    x, y = numeric_pair
    result = polyfit_residual_error(x, "Numerical", y, "Numerical")
    assert isinstance(result, float)
    assert np.isfinite(result)


def test_constant_inputs_do_not_crash() -> None:
    from sciona.atoms.causal_inference.feature_primitives.atoms import (
        discretize_and_bin,
        knn_entropy_estimator,
        uniform_divergence,
    )

    const = np.ones(50, dtype=np.float64)
    # These should not raise
    knn_entropy_estimator(const, "Numerical")
    uniform_divergence(const, "Numerical")
    discretize_and_bin(const, "Numerical")


def test_polyfit_functions_with_categorical_type(numeric_pair) -> None:
    from sciona.atoms.causal_inference.feature_primitives.atoms import (
        polyfit_nonlinearity_asymmetry,
        polyfit_residual_error,
    )

    x, y = numeric_pair
    r1 = polyfit_nonlinearity_asymmetry(x, "Categorical", y, "Categorical")
    assert isinstance(r1, float)
    assert np.isfinite(r1)

    r2 = polyfit_residual_error(x, "Categorical", y, "Categorical")
    assert isinstance(r2, float)
    assert np.isfinite(r2)
