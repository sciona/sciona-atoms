from __future__ import annotations

from typing import Callable
"""Auto-generated atom wrappers following the sciona pattern."""


import numpy as np

import icontract
from sciona.ghost.registry import register_atom
from .mala_witnesses import witness_mala_proposal_adjustment

import ctypes
import ctypes.util
from pathlib import Path


# Witness functions should be imported from the generated witnesses module

@register_atom(witness_mala_proposal_adjustment)
@icontract.require(lambda step_size: isinstance(step_size, (float, int, np.number)), "step_size must be numeric")
@icontract.ensure(lambda result: result is not None, "mala_proposal_adjustment output must not be None")
def mala_proposal_adjustment(step_size: float, vals_bound: np.ndarray, mala_mean_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Calculates the adjustment term for a Metropolis-Adjusted Langevin Algorithm (MALA) proposal. This typically involves the gradient of the log-posterior (via mala_mean_fn), which guides the proposal distribution.

    Args:
        step_size: Controls the magnitude of the Langevin dynamics step.
        vals_bound: Boundary conditions or constraints on the proposal values.
        mala_mean_fn: A function (oracle) that computes the mean of the proposal distribution, typically based on the gradient of the target log-probability.

    Returns:
        The calculated adjustment to be used in the MALA proposal.
    """
    return vals_bound + (step_size**2 / 2.0) * mala_mean_fn(vals_bound)


"""Auto-generated FFI bindings for cpp implementations."""


import ctypes
import ctypes.util
from pathlib import Path


def _mala_proposal_adjustment_ffi(step_size, vals_bound, mala_mean_fn):
    """Wrapper that calls the C++ version of mala proposal adjustment. Passes arguments through and returns the result."""
    _lib = ctypes.CDLL("./mala_proposal_adjustment.so")
    _func_name = 'mala_proposal_adjustment'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(step_size, vals_bound, mala_mean_fn)
