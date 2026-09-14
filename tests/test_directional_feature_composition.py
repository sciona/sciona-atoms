"""Independent synthetic arithmetic and contract tests for directional features."""

import numpy as np
import pytest

from sciona.atoms.causal_inference.feature_primitives.directional import (
    asymmetric_feature_difference, witness_asymmetric_feature_difference,
)
from sciona.ghost.abstract import AbstractArray


def test_difference_preserves_shape_inputs_and_antisymmetry():
    forward = np.array([[1, 5], [9, 2]], dtype=np.uint8)
    reverse = np.array([[3, 1], [4, 7]], dtype=np.uint8)
    before = forward.copy(), reverse.copy()
    result = asymmetric_feature_difference(forward, reverse)
    np.testing.assert_array_equal(result, [[-2.0, 4.0], [5.0, -5.0]])
    np.testing.assert_array_equal(asymmetric_feature_difference(reverse, forward), -result)
    np.testing.assert_array_equal(asymmetric_feature_difference(forward, forward), np.zeros((2, 2)))
    np.testing.assert_array_equal(forward, before[0])
    np.testing.assert_array_equal(reverse, before[1])
    assert result.dtype == np.float64
    assert not np.shares_memory(result, forward)
    assert not np.shares_memory(result, reverse)


@pytest.mark.parametrize("forward,reverse", [
    (np.ones((2, 1)), np.ones((2, 3))),
    (np.array([]), np.array([])),
    (np.array(1.0), np.array(2.0)),
    (np.array([np.nan]), np.array([0.0])),
    (np.array([np.inf]), np.array([0.0])),
    (np.array([1j]), np.array([0j])),
    (np.array(["1"]), np.array(["2"])),
])
def test_invalid_features_fail_instead_of_broadcasting_or_coercing(forward, reverse):
    with pytest.raises(ValueError):
        asymmetric_feature_difference(forward, reverse)


def test_overflow_is_not_returned_as_a_valid_feature():
    with pytest.raises(FloatingPointError):
        asymmetric_feature_difference(np.array([1.7e308]), np.array([-1.7e308]))


def test_witness_preserves_matrix_shape_and_rejects_broadcasting():
    a = AbstractArray(shape=(3, 4), dtype="float64")
    assert witness_asymmetric_feature_difference(a, a).shape == (3, 4)
    with pytest.raises(ValueError):
        witness_asymmetric_feature_difference(a, AbstractArray(shape=(3, 1), dtype="float64"))
