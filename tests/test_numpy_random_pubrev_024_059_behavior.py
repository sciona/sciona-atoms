from __future__ import annotations

import numpy as np
import pytest

from sciona.atoms.numpy import random as atoms


def test_rand_matches_legacy_global_rng_for_numpy_dimension_args() -> None:
    np.random.seed(123)
    expected = np.random.rand(2, 3)

    np.random.seed(123)
    actual = atoms.rand(2, 3)

    np.testing.assert_allclose(actual, expected)


def test_rand_seed_and_size_extension_use_generator_random() -> None:
    expected = np.random.default_rng(123).random((2, 3))

    np.testing.assert_allclose(atoms.rand(2, 3, seed=123), expected)
    np.testing.assert_allclose(atoms.rand(size=(2, 3), seed=123), expected)
    assert isinstance(atoms.rand(seed=123), float)

    with pytest.raises(ValueError):
        atoms.rand(2, size=(2,))


def test_uniform_forwards_scalar_array_and_seeded_generator_cases() -> None:
    low = np.array([0.0, 10.0])
    high = np.array([1.0, 11.0])
    expected = np.random.default_rng(321).uniform(low, high, size=(4, 2))

    actual = atoms.uniform(low, high, size=(4, 2), seed=321)

    np.testing.assert_allclose(actual, expected)
    assert actual.shape == (4, 2)

    np.random.seed(99)
    expected_scalar = np.random.uniform(-2.0, 3.0)
    np.random.seed(99)
    assert atoms.uniform(-2.0, 3.0) == expected_scalar


def test_default_rng_delegates_to_numpy_constructor() -> None:
    actual = atoms.default_rng(123)
    expected = np.random.default_rng(123)

    assert isinstance(actual, np.random.Generator)
    np.testing.assert_allclose(actual.random(5), expected.random(5))


def test_continuous_multivariate_sampler_forwards_numpy_draws() -> None:
    mean = np.array([0.0, 1.0])
    cov = np.array([[1.0, 0.2], [0.2, 2.0]])
    alpha = np.array([1.0, 2.0, 3.0])

    np.random.seed(77)
    expected_mvn = np.random.multivariate_normal(mean, cov, size=3, check_valid="warn", tol=1e-8)
    expected_dirichlet = np.random.dirichlet(alpha, size=3)

    np.random.seed(77)
    actual_mvn, actual_dirichlet = atoms.continuous_multivariate_sampler(mean, cov, alpha, size=3)

    np.testing.assert_allclose(actual_mvn, expected_mvn)
    np.testing.assert_allclose(actual_dirichlet, expected_dirichlet)
    assert actual_mvn.shape == (3, 2)
    assert actual_dirichlet.shape == (3, 3)


def test_discrete_event_sampler_forwards_multinomial() -> None:
    pvals = np.array([0.2, 0.3, 0.5])

    np.random.seed(88)
    expected = np.random.multinomial(10, pvals, size=4)

    np.random.seed(88)
    actual = atoms.discrete_event_sampler(10, pvals, size=4)

    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (4, 3)
    np.testing.assert_array_equal(actual.sum(axis=1), np.full(4, 10))


def test_combinatorics_sampler_forwards_permutation_and_choice() -> None:
    x = np.array([10, 20, 30, 40])
    a = np.array([1, 2, 3, 4])
    p = np.array([0.1, 0.2, 0.3, 0.4])

    np.random.seed(101)
    expected_perm = np.random.permutation(x)
    expected_choice = np.random.choice(a, size=3, replace=False, p=p)

    np.random.seed(101)
    actual_perm, actual_choice = atoms.combinatorics_sampler(x, a, size=3, replace=False, p=p)

    np.testing.assert_array_equal(actual_perm, expected_perm)
    np.testing.assert_array_equal(actual_choice, expected_choice)
