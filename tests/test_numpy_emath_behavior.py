from __future__ import annotations

import importlib
import inspect

import numpy as np
import pytest
from icontract.errors import ViolationError
from sciona.ghost.abstract import AbstractArray, AbstractScalar


atoms = importlib.import_module("sciona.atoms.numpy.emath")
witnesses = importlib.import_module("sciona.atoms.numpy.witnesses")

EXPECTED_ATOMS = {
    "log",
    "log10",
    "logn",
    "power",
    "sqrt",
}


def _parameter_contract(function: object) -> tuple[tuple[str, object, object], ...]:
    return tuple(
        (name, parameter.kind, parameter.default)
        for name, parameter in inspect.signature(function).parameters.items()
    )


def _assert_same_emath_result(actual: object, expected: object) -> None:
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert np.shape(actual) == np.shape(expected)
    actual_dtype = np.asarray(actual).dtype
    expected_dtype = np.asarray(expected).dtype
    assert actual_dtype == expected_dtype


@pytest.mark.parametrize("name", sorted(EXPECTED_ATOMS))
def test_numpy_emath_wrapper_signature_matches_current_numpy_emath(name: str) -> None:
    wrapper_contract = _parameter_contract(getattr(atoms, name))
    upstream_contract = _parameter_contract(getattr(np.emath, name))

    assert wrapper_contract == upstream_contract


def test_probe_records_resolve_to_live_emath_symbols() -> None:
    probes = importlib.import_module("sciona.probes.numpy.emath")

    records = probes.probe_records()
    assert {record["wrapper_symbol"] for record in records} == EXPECTED_ATOMS
    for record in records:
        module = importlib.import_module(str(record["module_import_path"]))
        symbol = str(record["wrapper_symbol"])
        assert getattr(module, symbol).__name__ == symbol
        assert str(record["atom_fqdn"]) == f"sciona.atoms.numpy.emath.{symbol}"


def test_installed_numpy_scimath_source_uses_documented_domain_promoters() -> None:
    source_expectations = {
        "sqrt": "_fix_real_lt_zero(x)",
        "log": "_fix_real_lt_zero(x)",
        "log10": "_fix_real_lt_zero(x)",
        "logn": "_fix_real_lt_zero(n)",
        "power": "_fix_int_lt_zero(p)",
    }

    assert np.__version__ == "2.4.2"
    for name, expected_source in source_expectations.items():
        source = inspect.getsource(getattr(np.emath, name))
        assert expected_source in source


def test_sqrt_matches_numpy_emath_for_real_complex_and_negative_domains() -> None:
    _assert_same_emath_result(atoms.sqrt(4), np.emath.sqrt(4))
    _assert_same_emath_result(atoms.sqrt([-1, 4]), np.emath.sqrt([-1, 4]))
    _assert_same_emath_result(
        atoms.sqrt(complex(-4.0, -0.0)),
        np.emath.sqrt(complex(-4.0, -0.0)),
    )

    with pytest.raises(ViolationError, match="Input must not be None"):
        atoms.sqrt(None)  # type: ignore[arg-type]


def test_log_and_log10_match_numpy_emath_for_zero_and_negative_domains() -> None:
    with np.errstate(divide="ignore", invalid="ignore"):
        _assert_same_emath_result(atoms.log(0), np.emath.log(0))
        _assert_same_emath_result(atoms.log([-np.e, 0.0, np.e]), np.emath.log([-np.e, 0.0, np.e]))
        _assert_same_emath_result(atoms.log10(0), np.emath.log10(0))
        _assert_same_emath_result(atoms.log10([-10.0, 0.0, 100.0]), np.emath.log10([-10.0, 0.0, 100.0]))


def test_logn_matches_numpy_emath_for_documented_and_boundary_domains() -> None:
    with np.errstate(divide="ignore", invalid="ignore"):
        _assert_same_emath_result(atoms.logn(2, [4, 8]), np.emath.logn(2, [4, 8]))
        _assert_same_emath_result(atoms.logn(2, [-4, -8, 8]), np.emath.logn(2, [-4, -8, 8]))
        _assert_same_emath_result(atoms.logn(-2, [4, 8]), np.emath.logn(-2, [4, 8]))
        _assert_same_emath_result(atoms.logn(1, [4, 8]), np.emath.logn(1, [4, 8]))
        _assert_same_emath_result(atoms.logn(0, [4, 8]), np.emath.logn(0, [4, 8]))


def test_power_matches_numpy_emath_for_complex_and_float_promotion_domains() -> None:
    _assert_same_emath_result(atoms.power([2, 4], 2), np.emath.power([2, 4], 2))
    _assert_same_emath_result(atoms.power([2, 4], -2), np.emath.power([2, 4], -2))
    _assert_same_emath_result(atoms.power([-2, 4], 2), np.emath.power([-2, 4], 2))
    _assert_same_emath_result(atoms.power([-2, 4], 0.5), np.emath.power([-2, 4], 0.5))
    _assert_same_emath_result(atoms.power([2, 4], [2, 4]), np.emath.power([2, 4], [2, 4]))


def test_emath_witnesses_capture_shape_and_domain_dtype_promotion() -> None:
    vector_positive_int = AbstractArray(shape=(2,), dtype="int64", min_val=1.0, max_val=4.0)
    vector_maybe_negative = AbstractArray(shape=(2,), dtype="float64", min_val=-1.0, max_val=4.0)
    positive_base = AbstractScalar(dtype="int64", min_val=2.0, max_val=2.0)
    negative_base = AbstractScalar(dtype="int64", min_val=-2.0, max_val=-2.0)
    negative_power = AbstractScalar(dtype="int64", min_val=-2.0, max_val=-2.0)

    sqrt_positive = witnesses.witness_np_emath_sqrt(vector_positive_int)
    assert sqrt_positive.shape == (2,)
    assert sqrt_positive.dtype == "float64"
    assert sqrt_positive.min_val == 0.0

    log_zero_allowed = witnesses.witness_np_emath_log(
        AbstractArray(shape=(3,), dtype="float64", min_val=0.0, max_val=10.0)
    )
    assert log_zero_allowed.dtype == "float64"

    log_negative = witnesses.witness_np_emath_log(vector_maybe_negative)
    assert log_negative.dtype == "complex128"

    logn_negative_base = witnesses.witness_np_emath_logn(negative_base, vector_positive_int)
    assert logn_negative_base.dtype == "complex128"

    logn_positive = witnesses.witness_np_emath_logn(positive_base, vector_positive_int)
    assert logn_positive.dtype == "float64"

    power_negative_base = witnesses.witness_np_emath_power(vector_maybe_negative, positive_base)
    assert power_negative_base.dtype == "complex128"

    power_negative_exponent = witnesses.witness_np_emath_power(vector_positive_int, negative_power)
    assert power_negative_exponent.dtype == "float64"
