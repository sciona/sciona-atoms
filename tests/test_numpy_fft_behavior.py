from __future__ import annotations

import importlib
import inspect

import numpy as np
import pytest


atoms = importlib.import_module("sciona.atoms.numpy.fft")

EXPECTED_ATOMS = {
    "fft",
    "fftfreq",
    "fftn",
    "fftshift",
    "hfft",
    "ifft",
    "ifftn",
    "irfft",
    "rfft",
}


@pytest.mark.parametrize("name", sorted(EXPECTED_ATOMS))
def test_numpy_fft_wrapper_signature_covers_current_numpy_parameters(name: str) -> None:
    wrapper_params = tuple(inspect.signature(getattr(atoms, name)).parameters)
    upstream_params = tuple(inspect.signature(getattr(np.fft, name)).parameters)

    assert wrapper_params == upstream_params


def test_one_dimensional_complex_fft_wrappers_match_numpy_axis_and_out_semantics() -> None:
    x = np.arange(12, dtype=float).reshape(3, 4)

    fft_out = np.empty((5, 4), dtype=complex)
    fft_result = atoms.fft(x, n=5, axis=0, out=fft_out)
    assert fft_result is fft_out
    np.testing.assert_allclose(fft_result, np.fft.fft(x, n=5, axis=0))

    ifft_out = np.empty((2, 4), dtype=complex)
    ifft_result = atoms.ifft(x, n=2, axis=0, norm="ortho", out=ifft_out)
    assert ifft_result is ifft_out
    np.testing.assert_allclose(ifft_result, np.fft.ifft(x, n=2, axis=0, norm="ortho"))


def test_real_fft_wrappers_match_numpy_axis_shape_and_out_semantics() -> None:
    x = np.arange(12, dtype=float).reshape(3, 4)

    rfft_out = np.empty((3, 4), dtype=complex)
    rfft_result = atoms.rfft(x, n=4, axis=0, out=rfft_out)
    assert rfft_result is rfft_out
    np.testing.assert_allclose(rfft_result, np.fft.rfft(x, n=4, axis=0))

    spectrum = np.fft.rfft(x, n=4, axis=0)
    irfft_out = np.empty((4, 4), dtype=float)
    irfft_result = atoms.irfft(spectrum, n=4, axis=0, out=irfft_out)
    assert irfft_result is irfft_out
    np.testing.assert_allclose(irfft_result, np.fft.irfft(spectrum, n=4, axis=0))


def test_multidimensional_fft_wrappers_match_numpy_s_axes_and_out_semantics() -> None:
    x = np.arange(12, dtype=float).reshape(3, 4)

    fftn_result = atoms.fftn(x, s=(2, 5), axes=(0, 1))
    np.testing.assert_allclose(fftn_result, np.fft.fftn(x, s=(2, 5), axes=(0, 1)))

    ifftn_result = atoms.ifftn(fftn_result, s=(2, 5), axes=(0, 1))
    np.testing.assert_allclose(ifftn_result, np.fft.ifftn(fftn_result, s=(2, 5), axes=(0, 1)))

    fftn_out = np.empty((2, 4), dtype=complex)
    assert atoms.fftn(x, s=(2,), axes=(0,), out=fftn_out) is fftn_out
    np.testing.assert_allclose(fftn_out, np.fft.fftn(x, s=(2,), axes=(0,)))


def test_frequency_helpers_match_numpy_edge_parameters() -> None:
    np.testing.assert_allclose(atoms.fftfreq(4, d=-1.0), np.fft.fftfreq(4, d=-1.0))
    np.testing.assert_allclose(atoms.fftfreq(4, device="cpu"), np.fft.fftfreq(4, device="cpu"))

    x = np.arange(9, dtype=float).reshape(3, 3)
    np.testing.assert_array_equal(atoms.fftshift(x, axes=(1,)), np.fft.fftshift(x, axes=(1,)))


def test_hfft_matches_numpy_default_n_axis_and_out_semantics() -> None:
    x = np.arange(12, dtype=float).reshape(3, 4)

    default_result = atoms.hfft(x, axis=0)
    np.testing.assert_allclose(default_result, np.fft.hfft(x, axis=0))
    assert default_result.shape == (4, 4)

    out = np.empty((6, 4), dtype=float)
    result = atoms.hfft(x, n=6, axis=0, norm="forward", out=out)
    np.testing.assert_allclose(result, np.fft.hfft(x, n=6, axis=0, norm="forward"))
