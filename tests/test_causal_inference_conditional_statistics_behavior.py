from __future__ import annotations

import numpy as np


def test_all_four_atoms_import() -> None:
    from sciona.atoms.causal_inference.conditional_statistics.atoms import (
        conditional_distribution_similarity,
        conditional_noise_entropy_variance,
        conditional_noise_kurtosis_variance,
        conditional_noise_skewness_variance,
    )
    assert callable(conditional_noise_entropy_variance)
    assert callable(conditional_noise_skewness_variance)
    assert callable(conditional_noise_kurtosis_variance)
    assert callable(conditional_distribution_similarity)


def test_finite_scalar_outputs() -> None:
    from sciona.atoms.causal_inference.conditional_statistics.atoms import (
        conditional_distribution_similarity,
        conditional_noise_entropy_variance,
        conditional_noise_kurtosis_variance,
        conditional_noise_skewness_variance,
    )
    from sciona.atoms.causal_inference.feature_primitives.atoms import NUMERICAL

    np.random.seed(42)
    x = np.random.randn(300)
    y = x ** 2 + 0.1 * np.random.randn(300)

    for fn in [
        conditional_noise_entropy_variance,
        conditional_noise_skewness_variance,
        conditional_noise_kurtosis_variance,
        conditional_distribution_similarity,
    ]:
        result = fn(x, NUMERICAL, y, NUMERICAL)
        assert isinstance(result, float), f"{fn.__name__} returned {type(result)}"
        assert np.isfinite(result), f"{fn.__name__} returned non-finite: {result}"
        assert result >= 0.0, f"{fn.__name__} returned negative: {result}"


def test_constant_input_returns_zero() -> None:
    from sciona.atoms.causal_inference.conditional_statistics.atoms import (
        conditional_noise_entropy_variance,
        conditional_noise_skewness_variance,
    )
    from sciona.atoms.causal_inference.feature_primitives.atoms import NUMERICAL

    const = np.ones(100)
    y = np.random.randn(100)
    assert conditional_noise_entropy_variance(const, NUMERICAL, y, NUMERICAL) == 0.0
    assert conditional_noise_skewness_variance(const, NUMERICAL, y, NUMERICAL) == 0.0


def test_low_unique_count_does_not_crash() -> None:
    from sciona.atoms.causal_inference.conditional_statistics.atoms import (
        conditional_noise_kurtosis_variance,
    )
    from sciona.atoms.causal_inference.feature_primitives.atoms import NUMERICAL

    x = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    y = np.array([3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0])
    result = conditional_noise_kurtosis_variance(x, NUMERICAL, y, NUMERICAL)
    assert np.isfinite(result)
