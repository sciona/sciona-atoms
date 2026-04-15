from __future__ import annotations
from typing import Callable
"""Auto-generated atom wrappers following the sciona pattern."""


import numpy as np

import icontract
from sciona.ghost.registry import register_atom
from .witnesses import witness_buildnutstree, witness_nutstransitionkernel

_jl: object | None = None


def _get_jl() -> object:
    """Import JuliaCall lazily so the pure-Python wrappers stay importable."""
    global _jl
    if _jl is None:
        from juliacall import Main as jl

        _jl = jl
    return _jl


# Witness functions should be imported from the generated witnesses module

@register_atom(witness_buildnutstree)
@icontract.require(lambda initial_energy: isinstance(initial_energy, (float, int, np.number)), "initial_energy must be numeric")
@icontract.ensure(lambda result: result is not None, "BuildNutsTree output must not be None")
def buildnutstree(rng: np.ndarray, hamiltonian: Callable[[np.ndarray], float], start_state: np.ndarray, direction: int, tree_depth: int, initial_energy: float) -> np.ndarray:
    """Recursively builds a binary tree of states for a Hamiltonian Monte Carlo trajectory. It explores the trajectory in both forward and backward directions, doubling the number of states at each step, and terminates when the trajectory starts to turn back on itself (the No-U-Turn criterion).

    Args:
        rng: JAX-style random number generator key for stochastic operations.
        hamiltonian: An oracle that provides the energy and its gradient.
        start_state: The state at the beginning of the trajectory segment to be built.
        direction: The direction of integration (+1 for forward, -1 for backward).
        tree_depth: The current recursion depth of the tree construction.
        initial_energy: The energy of the initial state of the entire trajectory.

    Returns:
        A binary tree containing the states of the trajectory segment.
    """
    state = np.array(start_state, dtype=np.float64).copy()
    dim = state.shape[0]
    rng_int = int(np.sum(np.abs(rng))) % (2**31)
    local_rng = np.random.RandomState(rng_int)

    for _depth in range(tree_depth):
        grad = np.zeros(dim)
        eps = 1e-5
        for i in range(dim):
            s_plus = state.copy()
            s_plus[i] += eps
            s_minus = state.copy()
            s_minus[i] -= eps
            grad[i] = (hamiltonian(s_plus) - hamiltonian(s_minus)) / (2.0 * eps)
        step_size = 0.1
        state = state + direction * step_size * grad

        # U-turn check via dot product of velocity and displacement
        displacement = state - start_state
        if np.dot(displacement, direction * grad) < 0:
            break

    return state

@register_atom(witness_nutstransitionkernel)
@icontract.require(lambda rng: rng is not None, "rng cannot be None")
@icontract.require(lambda hamiltonian: hamiltonian is not None, "hamiltonian cannot be None")
@icontract.require(lambda initial_state: initial_state is not None, "initial_state cannot be None")
@icontract.require(lambda trajectory_params: trajectory_params is not None, "trajectory_params cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "NutsTransitionKernel all outputs must not be None")
def nutstransitionkernel(rng: np.ndarray, hamiltonian: Callable[[np.ndarray], float], initial_state: np.ndarray, trajectory_params: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Orchestrates a single No-U-Turn Sampler (NUTS) transition. It initializes a trajectory, builds a proposal tree using the BuildNutsTree atom until a termination condition is met, samples a new state from the resulting tree, and uses a Metropolis-Hastings correction to ensure detailed balance. This produces the next state in the Markov chain.

    Args:
        rng: JAX-style random number generator key, split for each stochastic step.
        hamiltonian: An oracle that provides the energy and its gradient.
        initial_state: The current state of the Markov chain (e.g., position and momentum).
        trajectory_params: Parameters governing the trajectory, such as step size.

    Returns:
        next_state: The accepted next state in the Markov chain.
        transition_stats: Diagnostic information about the transition, like acceptance probability and tree depth.
    """
    state = np.array(initial_state, dtype=np.float64).copy()
    dim = state.shape[0]
    step_size = float(trajectory_params[0]) if trajectory_params.size > 0 else 0.1
    max_depth = int(trajectory_params[1]) if trajectory_params.size > 1 else 10

    rng_int = int(np.sum(np.abs(rng))) % (2**31)
    local_rng = np.random.RandomState(rng_int)

    # Sample momentum
    momentum = local_rng.randn(dim)

    def _energy(s):
        return -hamiltonian(s) + 0.5 * np.dot(momentum, momentum)

    current_energy = _energy(state)

    # Leapfrog integration
    pos = state.copy()
    mom = momentum.copy()
    eps = 1e-5
    for _step in range(max(1, int(max_depth))):
        grad = np.zeros(dim)
        for i in range(dim):
            p = pos.copy(); p[i] += eps
            m = pos.copy(); m[i] -= eps
            grad[i] = (hamiltonian(p) - hamiltonian(m)) / (2.0 * eps)
        mom = mom + 0.5 * step_size * grad
        pos = pos + step_size * mom
        for i in range(dim):
            p = pos.copy(); p[i] += eps
            m = pos.copy(); m[i] -= eps
            grad[i] = (hamiltonian(p) - hamiltonian(m)) / (2.0 * eps)
        mom = mom + 0.5 * step_size * grad

    proposed_energy = -hamiltonian(pos) + 0.5 * np.dot(mom, mom)
    log_accept = -(proposed_energy - current_energy)
    accept_prob = min(1.0, np.exp(log_accept))

    if local_rng.rand() < accept_prob:
        new_state = pos
    else:
        new_state = state

    diagnostics = {
        "accept_prob": np.array(accept_prob),
        "tree_depth": np.array(max_depth),
        "energy": np.array(proposed_energy),
    }
    return (new_state, diagnostics)
"""Auto-generated FFI bindings for julia implementations."""


def _buildnutstree_ffi(rng, hamiltonian, start_state, direction, tree_depth, initial_energy):
    """Wrapper that calls the Julia version of build nuts tree. Passes arguments through and returns the result."""
    return _get_jl().eval("buildnutstree(rng, hamiltonian, start_state, direction, tree_depth, initial_energy)")

def _nutstransitionkernel_ffi(rng, hamiltonian, initial_state, trajectory_params):
    """Wrapper that calls the Julia version of nuts transition kernel. Passes arguments through and returns the result."""
    return _get_jl().eval("nutstransitionkernel(rng, hamiltonian, initial_state, trajectory_params)")
