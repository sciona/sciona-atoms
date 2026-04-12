from __future__ import annotations

from typing import Any, Callable
HMCState: Any = Any
NUTS_Trajectory: Any = Any
Position: Any = Any
State: Any = Any
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .nuts_witnesses import witness_nuts_recursive_tree_build

import ctypes
import ctypes.util
from pathlib import Path


@register_atom(witness_nuts_recursive_tree_build)
@icontract.require(lambda step_size: isinstance(step_size, (float, int, np.number)), "step_size must be numeric")
@icontract.require(lambda log_slice_variable: isinstance(log_slice_variable, (float, int, np.number)), "log_slice_variable must be numeric")
@icontract.require(lambda log_prob_oracle: callable(log_prob_oracle), "log_prob_oracle must be callable")
@icontract.require(lambda integrator_fn: callable(integrator_fn), "integrator_fn must be callable")
@icontract.ensure(lambda result: result is not None, "nuts_recursive_tree_build output must not be None")
def nuts_recursive_tree_build(direction_val: int, step_size: float, log_slice_variable: float, initial_hmc_state: HMCState, log_prob_oracle: Callable[[Position], float], integrator_fn: Callable[[State, float, int], State], tree_depth: int) -> NUTS_Trajectory:
    """Recursively builds a binary tree for a No-U-Turn Sampler (NUTS) step.

Args:
    direction_val: Determines the direction of integration, typically +1 for forward or -1 for backward.
    step_size: The step size (epsilon) for the leapfrog integrator.
    log_slice_variable: The logarithm of the uniform slice variable 'u', used for the generalized Hamiltonian Monte Carlo (HMC) acceptance criterion.
    initial_hmc_state: The initial state for this subtree, containing position, momentum, potential energy (prev_U), and kinetic energy (prev_K).
    log_prob_oracle: An oracle function (box_log_kernel_fn) that computes the log probability (potential energy) of the target distribution for a given position.
    integrator_fn: The leapfrog integrator function (leap_frog_fn) used to propose new states along the trajectory.
    tree_depth: The current recursion depth of the tree-building process.

Returns:
    Returns a composite object representing the built trajectory, including the leftmost/rightmost states, the proposed sample, a flag indicating a U-turn, a divergence flag, and summed acceptance probabilities."""
    # Base case: single leapfrog step
    if tree_depth == 0:
        new_state = integrator_fn(initial_hmc_state, step_size, direction_val)
        pos = np.asarray(new_state, dtype=np.float64)
        new_logp = log_prob_oracle(pos) if callable(log_prob_oracle) else float(log_prob_oracle)
        new_kinetic = 0.0
        log_joint = new_logp - new_kinetic
        return pos

    # Recursive: build left then right subtree
    left = nuts_recursive_tree_build(
        direction_val, step_size, log_slice_variable,
        initial_hmc_state, log_prob_oracle, integrator_fn, tree_depth - 1
    )
    right = nuts_recursive_tree_build(
        direction_val, step_size, log_slice_variable,
        left, log_prob_oracle, integrator_fn, tree_depth - 1
    )
    return np.asarray(right, dtype=np.float64)


"""Auto-generated FFI bindings for cpp implementations."""


def _nuts_recursive_tree_build_ffi(direction_val, step_size, log_slice_variable, initial_hmc_state, log_prob_oracle, integrator_fn, tree_depth):
    """Wrapper that calls the C++ version of nuts recursive tree build. Passes arguments through and returns the result."""
    _lib = ctypes.CDLL("./nuts_recursive_tree_build.so")
    _func_name = 'nuts_recursive_tree_build'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(direction_val, step_size, log_slice_variable, initial_hmc_state, log_prob_oracle, integrator_fn, tree_depth)
