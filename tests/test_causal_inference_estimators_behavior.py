from __future__ import annotations

import numpy as np


def test_all_four_atoms_import() -> None:
    from sciona.atoms.causal_inference.estimators.atoms import (
        left_right_decomposed_prediction,
        symmetrized_prediction_fusion,
        two_stage_independence_direction,
        weighted_ensemble_combination,
    )
    assert callable(symmetrized_prediction_fusion)
    assert callable(two_stage_independence_direction)
    assert callable(left_right_decomposed_prediction)
    assert callable(weighted_ensemble_combination)


def test_symmetrized_enforces_antisymmetry() -> None:
    from sciona.atoms.causal_inference.estimators.atoms import symmetrized_prediction_fusion

    raw = np.array([0.8, -0.3, 0.6, -0.5])
    result = symmetrized_prediction_fusion(raw)
    assert result.shape == raw.shape
    assert np.allclose(result[0::2], -result[1::2])


def test_two_stage_zeros_independent_pairs() -> None:
    from sciona.atoms.causal_inference.estimators.atoms import two_stage_independence_direction

    ind = np.array([0.0, 0.0, 1.0, 1.0])
    dir_ = np.array([0.5, -0.5, 0.5, -0.5])
    result = two_stage_independence_direction(ind, dir_)
    assert np.allclose(result[:2], 0.0), "independent pair should be zeroed"
    assert not np.allclose(result[2:], 0.0), "dependent pair should be non-zero"


def test_left_right_enforces_antisymmetry() -> None:
    from sciona.atoms.causal_inference.estimators.atoms import left_right_decomposed_prediction

    left = np.array([0.7, 0.2, 0.6, 0.3])
    right = np.array([0.2, 0.7, 0.3, 0.6])
    result = left_right_decomposed_prediction(left, right)
    assert np.allclose(result[0::2], -result[1::2])


def test_weighted_ensemble_with_weights() -> None:
    from sciona.atoms.causal_inference.estimators.atoms import weighted_ensemble_combination

    preds = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    w = np.array([0.6, 0.4])
    result = weighted_ensemble_combination(preds, w)
    assert np.allclose(result, [1.8, 2.8])


def test_weighted_ensemble_without_weights() -> None:
    from sciona.atoms.causal_inference.estimators.atoms import weighted_ensemble_combination

    preds = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    result = weighted_ensemble_combination(preds)
    assert result.shape == (2, 2)
