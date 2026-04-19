from __future__ import annotations

import importlib
import inspect

import numpy as np
import pytest
from icontract.errors import ViolationError


atoms = importlib.import_module("sciona.atoms.numpy.linalg")

EXPECTED_ATOMS = {
    "det",
    "inv",
    "norm",
    "solve",
}


@pytest.mark.parametrize("name", sorted(EXPECTED_ATOMS))
def test_numpy_linalg_wrapper_signature_covers_current_numpy_parameters(name: str) -> None:
    wrapper_params = tuple(inspect.signature(getattr(atoms, name)).parameters)
    upstream_params = tuple(inspect.signature(getattr(np.linalg, name)).parameters)

    assert wrapper_params == upstream_params


def test_probe_records_resolve_to_live_linalg_symbols() -> None:
    probes = importlib.import_module("sciona.probes.numpy.linalg")

    records = probes.probe_records()
    assert {record["wrapper_symbol"] for record in records} == EXPECTED_ATOMS
    for record in records:
        module = importlib.import_module(str(record["module_import_path"]))
        assert getattr(module, str(record["wrapper_symbol"])).__name__ == str(record["wrapper_symbol"])
        assert str(record["atom_fqdn"]) == f"sciona.atoms.numpy.linalg.{record['wrapper_symbol']}"


def test_solve_matches_numpy_for_supported_2d_systems_and_documents_stacked_limit() -> None:
    a = np.array([[3.0, 1.0], [1.0, 2.0]])
    b_vector = np.array([9.0, 8.0])
    b_matrix = np.array([[9.0, 1.0], [8.0, 4.0]])

    np.testing.assert_allclose(atoms.solve(a, b_vector), np.linalg.solve(a, b_vector))
    np.testing.assert_allclose(atoms.solve(a, b_matrix), np.linalg.solve(a, b_matrix))

    stacked_a = np.stack([a, a + np.eye(2)])
    stacked_b = np.stack([b_vector, b_vector])[:, :, np.newaxis]
    upstream_stacked = np.linalg.solve(stacked_a, stacked_b)
    assert upstream_stacked.shape == stacked_b.shape
    with pytest.raises(ViolationError, match="a must be"):
        atoms.solve(stacked_a, stacked_b)


def test_inv_matches_numpy_for_supported_2d_matrix_and_documents_stacked_limit() -> None:
    a = np.array([[4.0, 7.0], [2.0, 6.0]])

    np.testing.assert_allclose(atoms.inv(a), np.linalg.inv(a))

    stacked = np.stack([a, a + np.eye(2)])
    upstream_stacked = np.linalg.inv(stacked)
    assert upstream_stacked.shape == stacked.shape
    with pytest.raises(ViolationError, match="a must be a square 2D matrix"):
        atoms.inv(stacked)


def test_det_matches_numpy_for_single_and_stacked_square_matrices() -> None:
    a = np.array([[4.0, 7.0], [2.0, 6.0]])
    stacked = np.stack([a, a + np.eye(2)])

    np.testing.assert_allclose(atoms.det(a), np.linalg.det(a))
    np.testing.assert_allclose(atoms.det(stacked), np.linalg.det(stacked))


def test_norm_matches_numpy_vector_matrix_axis_and_keepdims_semantics() -> None:
    x = np.arange(1.0, 13.0).reshape(3, 4)

    np.testing.assert_allclose(atoms.norm(x), np.linalg.norm(x))
    np.testing.assert_allclose(atoms.norm(x[:2, :2], ord="nuc"), np.linalg.norm(x[:2, :2], ord="nuc"))

    axis_result = atoms.norm(x, axis=1, keepdims=True)
    upstream_axis_result = np.linalg.norm(x, axis=1, keepdims=True)
    np.testing.assert_allclose(axis_result, upstream_axis_result)
    assert axis_result.shape == upstream_axis_result.shape
