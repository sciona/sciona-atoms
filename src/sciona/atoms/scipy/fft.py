from __future__ import annotations
from typing import Union
import os

import numpy as np
import scipy.fft
import icontract

from sciona.ghost.registry import register_atom
from sciona.ghost.witnesses import witness_dct, witness_idct

ArrayLike = Union[np.ndarray, list, tuple]

_SLOW_CHECKS = os.environ.get("SCIONA_SLOW_CHECKS", "0") == "1"


def _roundtrip_close(original: np.ndarray, reconstructed: np.ndarray, atol: float = 1e-10) -> bool:
    """Check that a round-trip reconstruction is close to the original."""
    return bool(np.allclose(original, reconstructed, atol=atol))


@register_atom(witness_dct)
@icontract.require(lambda x: x is not None, "Input array must not be None")
@icontract.require(lambda x: np.asarray(x).size > 0, "Input array must not be empty")
@icontract.ensure(
    lambda result, x, n: result.shape == (np.asarray(x).shape if n is None else
        tuple(n if i == (np.asarray(x).ndim - 1) else s for i, s in enumerate(np.asarray(x).shape))),
    "Output shape must be preserved (or match n along axis)",
)
@icontract.ensure(lambda result: np.isrealobj(result), "DCT output must be real-valued")
@icontract.ensure(
    lambda result, x, type, n, axis, norm: _roundtrip_close(
        np.asarray(x),
        scipy.fft.idct(result, type=type, n=n, axis=axis, norm=norm),
    ),
    "Round-trip IDCT(DCT(x)) must approximate x",
    enabled=_SLOW_CHECKS,
)
def dct(
    x: ArrayLike,
    type: int = 2,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
    overwrite_x: bool = False,
) -> np.ndarray:
    """Compute the Discrete Cosine Transform.

    Computes the DCT of the input array along the specified axis.
    The DCT is a real-valued transform related to the Discrete Fourier Transform (DFT).

    Args:
        x: Input array, must be real-valued.
        type: Type of DCT (1, 2, 3, or 4). Default is 2.
        n: Length of the transform. If n is smaller than the input,
            the input is cropped. If larger, padded with zeros.
        axis: Axis over which to compute the DCT. Default is -1.
        norm: Normalization mode. None or "ortho".
        overwrite_x: If True, the contents of x may be destroyed.

    Returns:
        The DCT of the input array, real-valued, with shape preserved.

    """
    return scipy.fft.dct(x, type=type, n=n, axis=axis, norm=norm, overwrite_x=overwrite_x)


@register_atom(witness_idct)
@icontract.require(lambda x: x is not None, "Input array must not be None")
@icontract.require(lambda x: np.asarray(x).size > 0, "Input array must not be empty")
@icontract.ensure(
    lambda result, x, n: result.shape == (np.asarray(x).shape if n is None else
        tuple(n if i == (np.asarray(x).ndim - 1) else s for i, s in enumerate(np.asarray(x).shape))),
    "Output shape must be preserved (or match n along axis)",
)
@icontract.ensure(lambda result: np.isrealobj(result), "IDCT output must be real-valued")
@icontract.ensure(
    lambda result, x, type, n, axis, norm: _roundtrip_close(
        np.asarray(x),
        scipy.fft.dct(result, type=type, n=n, axis=axis, norm=norm),
    ),
    "Round-trip DCT(IDCT(x)) must approximate x",
    enabled=_SLOW_CHECKS,
)
def idct(
    x: ArrayLike,
    type: int = 2,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
    overwrite_x: bool = False,
) -> np.ndarray:
    """Compute the Inverse Discrete Cosine Transform.

    Computes the IDCT of the input array along the specified axis.

    Args:
        x: Input array, must be real-valued.
        type: Type of DCT (1, 2, 3, or 4). Default is 2.
        n: Length of the transform. If n is smaller than the input,
            the input is cropped. If larger, padded with zeros.
        axis: Axis over which to compute the IDCT. Default is -1.
        norm: Normalization mode. None or "ortho".
        overwrite_x: If True, the contents of x may be destroyed.

    Returns:
        The IDCT of the input array, real-valued, with shape preserved.

    """
    return scipy.fft.idct(x, type=type, n=n, axis=axis, norm=norm, overwrite_x=overwrite_x)