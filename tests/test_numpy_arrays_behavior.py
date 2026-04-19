from __future__ import annotations

import importlib
import inspect

import numpy as np
import pytest
from icontract.errors import ViolationError


atoms = importlib.import_module("sciona.atoms.numpy.arrays")

EXPECTED_ATOMS = {
    "array",
    "dot",
    "reshape",
    "vstack",
    "zeros",
}


def _parameter_contract(function: object) -> tuple[tuple[str, object, object], ...]:
    return tuple(
        (name, parameter.kind, parameter.default)
        for name, parameter in inspect.signature(function).parameters.items()
    )


@pytest.mark.parametrize("name", sorted(EXPECTED_ATOMS))
def test_numpy_arrays_wrapper_signature_covers_current_numpy_parameters(name: str) -> None:
    wrapper_contract = _parameter_contract(getattr(atoms, name))
    upstream_contract = _parameter_contract(getattr(np, name))

    assert wrapper_contract == upstream_contract


def test_probe_records_resolve_to_live_array_symbols() -> None:
    probes = importlib.import_module("sciona.probes.numpy.arrays")

    records = probes.probe_records()
    assert {record["wrapper_symbol"] for record in records} == EXPECTED_ATOMS
    for record in records:
        module = importlib.import_module(str(record["module_import_path"]))
        assert getattr(module, str(record["wrapper_symbol"])).__name__ == str(record["wrapper_symbol"])
        assert str(record["atom_fqdn"]) == f"sciona.atoms.numpy.arrays.{record['wrapper_symbol']}"


def test_array_matches_numpy_dtype_copy_ndmin_ndmax_like_and_documents_none_limit() -> None:
    values = [1, 2, 3]

    result = atoms.array(values, dtype=np.int32, copy=None, ndmin=2, ndmax=2, like=np.array([]))
    expected = np.array(values, dtype=np.int32, copy=None, ndmin=2, ndmax=2, like=np.array([]))
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == expected.dtype
    assert result.shape == expected.shape

    object_array = atoms.array([[1], [2, 3]], dtype=object, ndmax=1)
    expected_object_array = np.array([[1], [2, 3]], dtype=object, ndmax=1)
    np.testing.assert_array_equal(object_array, expected_object_array)

    assert np.array(None).shape == ()
    with pytest.raises(ViolationError, match="Object must not be None"):
        atoms.array(None)


def test_zeros_matches_numpy_dtype_order_device_like_and_integer_shape_semantics() -> None:
    result = atoms.zeros((2, 3), dtype=np.int16, order="F", device="cpu", like=np.array([]))
    expected = np.zeros((2, 3), dtype=np.int16, order="F", device="cpu", like=np.array([]))

    np.testing.assert_array_equal(result, expected)
    assert result.dtype == expected.dtype
    assert result.flags.f_contiguous == expected.flags.f_contiguous

    np.testing.assert_array_equal(atoms.zeros(np.int64(3)), np.zeros(np.int64(3)))


def test_dot_matches_numpy_scalar_vector_matrix_nd_and_out_semantics() -> None:
    np.testing.assert_array_equal(atoms.dot(2, np.arange(3)), np.dot(2, np.arange(3)))
    np.testing.assert_array_equal(atoms.dot(np.arange(3), np.arange(3)), np.dot(np.arange(3), np.arange(3)))

    matrix = np.arange(6).reshape(2, 3)
    vector = np.arange(3)
    np.testing.assert_array_equal(atoms.dot(matrix, vector), np.dot(matrix, vector))

    left = np.arange(24).reshape(2, 3, 4)
    right = np.arange(20).reshape(4, 5)
    np.testing.assert_array_equal(atoms.dot(left, right), np.dot(left, right))

    out = np.empty((2, 5), dtype=int)
    assert atoms.dot(matrix, right[:3], out=out) is out
    np.testing.assert_array_equal(out, np.dot(matrix, right[:3]))


def test_vstack_matches_numpy_dtype_casting_and_contract_limits() -> None:
    values = [np.array([1, 2], dtype=np.int16), np.array([3, 4], dtype=np.int16)]

    result = atoms.vstack(values, dtype=np.int64, casting="safe")
    expected = np.vstack(values, dtype=np.int64, casting="safe")
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == expected.dtype

    with pytest.raises(ViolationError, match="must not be empty"):
        atoms.vstack(())


def test_reshape_matches_numpy_shape_order_copy_and_rejects_removed_newshape_keyword() -> None:
    values = np.arange(6)

    result = atoms.reshape(values, (2, 3), order="C", copy=False)
    expected = np.reshape(values, (2, 3), order="C", copy=False)
    np.testing.assert_array_equal(result, expected)
    assert np.shares_memory(result, values) == np.shares_memory(expected, values)

    copied = atoms.reshape(values, (3, 2), order="F", copy=True)
    expected_copied = np.reshape(values, (3, 2), order="F", copy=True)
    np.testing.assert_array_equal(copied, expected_copied)
    assert not np.shares_memory(copied, values)

    with pytest.raises(TypeError, match="newshape"):
        atoms.reshape(values, newshape=(2, 3))  # type: ignore[call-arg]
