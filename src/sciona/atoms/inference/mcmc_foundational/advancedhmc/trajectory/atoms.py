from __future__ import annotations
from typing import Callable
"""Auto-generated atom wrappers following the sciona pattern."""


import numpy as np

import icontract
from sciona.ghost.registry import register_atom
from .witnesses import witness_buildnutstree, witness_nutstransitionkernel

_jl: object | None = None
_ENERGY_DIVERGENCE_LIMIT = 1000.0


def _get_jl() -> object:
    """Import JuliaCall lazily so the pure-Python wrappers stay importable."""
    global _jl
    if _jl is None:
        from juliacall import Main as jl

        _jl = jl
    return _jl


def _as_position(position: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.atleast_1d(np.asarray(position, dtype=np.float64)).copy()
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite numeric values")
    return vector


def _split_state(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vector = _as_position(state, name="state")
    if vector.size % 2 == 0:
        dim = vector.size // 2
        return vector[:dim].copy(), vector[dim:].copy()
    return vector.copy(), np.zeros_like(vector, dtype=np.float64)


def _potential(hamiltonian: Callable[[np.ndarray], float], theta: np.ndarray) -> float:
    value = float(hamiltonian(theta))
    if not np.isfinite(value):
        raise ValueError("hamiltonian returned a non-finite value")
    return value


def _grad_potential(hamiltonian: Callable[[np.ndarray], float], theta: np.ndarray) -> np.ndarray:
    eps = 1e-5
    grad = np.zeros_like(theta, dtype=np.float64)
    for idx in range(theta.size):
        plus = theta.copy()
        minus = theta.copy()
        plus[idx] += eps
        minus[idx] -= eps
        grad[idx] = (_potential(hamiltonian, plus) - _potential(hamiltonian, minus)) / (2.0 * eps)
    if not np.all(np.isfinite(grad)):
        raise ValueError("finite-difference gradient produced non-finite values")
    return grad


def _energy(hamiltonian: Callable[[np.ndarray], float], theta: np.ndarray, momentum: np.ndarray) -> float:
    return _potential(hamiltonian, theta) + 0.5 * float(np.dot(momentum, momentum))


def _leapfrog(
    hamiltonian: Callable[[np.ndarray], float],
    theta: np.ndarray,
    momentum: np.ndarray,
    step_size: float,
    direction: int,
) -> tuple[np.ndarray, np.ndarray]:
    eps = float(direction) * step_size
    momentum_next = momentum - 0.5 * eps * _grad_potential(hamiltonian, theta)
    theta_next = theta + eps * momentum_next
    momentum_next = momentum_next - 0.5 * eps * _grad_potential(hamiltonian, theta_next)
    return theta_next, momentum_next


def _pack_state(theta: np.ndarray, momentum: np.ndarray) -> np.ndarray:
    return np.concatenate([theta, momentum])


def _is_turning(theta_left: np.ndarray, theta_right: np.ndarray, momentum_left: np.ndarray, momentum_right: np.ndarray) -> bool:
    delta = theta_right - theta_left
    return bool(np.dot(delta, momentum_left) < 0.0 or np.dot(delta, momentum_right) < 0.0)


# Witness functions should be imported from the generated witnesses module

@register_atom(witness_buildnutstree)
@icontract.require(lambda initial_energy: isinstance(initial_energy, (float, int, np.number)), "initial_energy must be numeric")
@icontract.ensure(lambda result: result is not None, "BuildNutsTree output must not be None")
def buildnutstree(rng: np.ndarray, hamiltonian: Callable[[np.ndarray], float], start_state: np.ndarray, direction: int, tree_depth: int, initial_energy: float) -> np.ndarray:
    """Build a compact NUTS trajectory subtree from repeated leapfrog leaves.

    Args:
        rng: Explicit random key; accepted for API parity, but this deterministic subtree builder does not mutate it.
        hamiltonian: Potential-energy oracle on position vectors.
        start_state: Phase point ``[theta, momentum]`` at the beginning of the trajectory segment.
        direction: The direction of integration (+1 for forward, -1 for backward).
        tree_depth: NUTS tree depth; at most ``2 ** tree_depth`` leaves are generated.
        initial_energy: The energy of the initial state of the entire trajectory.

    Returns:
        Stacked phase-point leaves with shape ``(n_leaves, 2 * dim)``.
    """
    _as_position(rng, name="rng")
    if direction not in {-1, 1}:
        raise ValueError("direction must be -1 or 1")
    if int(tree_depth) < 0:
        raise ValueError("tree_depth must be non-negative")
    theta0, momentum0 = _split_state(start_state)
    step_size = 0.1
    max_leaves = 2 ** int(tree_depth)
    leaves: list[np.ndarray] = []
    theta = theta0.copy()
    momentum = momentum0.copy()

    for _ in range(max(1, max_leaves)):
        theta, momentum = _leapfrog(hamiltonian, theta, momentum, step_size, direction)
        leaf = _pack_state(theta, momentum)
        leaves.append(leaf)
        energy_delta = _energy(hamiltonian, theta, momentum) - float(initial_energy)
        if abs(energy_delta) > _ENERGY_DIVERGENCE_LIMIT:
            break
        if _is_turning(theta0, theta, momentum0, momentum):
            break

    return np.vstack(leaves)

@register_atom(witness_nutstransitionkernel)
@icontract.require(lambda rng: rng is not None, "rng cannot be None")
@icontract.require(lambda hamiltonian: hamiltonian is not None, "hamiltonian cannot be None")
@icontract.require(lambda initial_state: initial_state is not None, "initial_state cannot be None")
@icontract.require(lambda trajectory_params: trajectory_params is not None, "trajectory_params cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "NutsTransitionKernel all outputs must not be None")
def nutstransitionkernel(rng: np.ndarray, hamiltonian: Callable[[np.ndarray], float], initial_state: np.ndarray, trajectory_params: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run a compact identity-mass No-U-Turn Sampler transition.

    Args:
        rng: Explicit random key used to sample momentum, directions, and the final candidate.
        hamiltonian: Potential-energy oracle on position vectors.
        initial_state: Current position vector.
        trajectory_params: ``[step_size, max_depth, max_energy_delta]``; missing values default to ``0.1, 10, 1000``.

    Returns:
        next_state: The accepted next state in the Markov chain.
        transition_stats: Diagnostic information for acceptance probability, tree depth, energy, and numerical errors.
    """
    position0 = _as_position(initial_state, name="initial_state")
    params = np.atleast_1d(np.asarray(trajectory_params, dtype=np.float64))
    step_size = float(params[0]) if params.size > 0 else 0.1
    max_depth = int(params[1]) if params.size > 1 else 10
    max_energy_delta = float(params[2]) if params.size > 2 else _ENERGY_DIVERGENCE_LIMIT
    if not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("step_size must be finite and positive")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    if not np.isfinite(max_energy_delta) or max_energy_delta <= 0.0:
        raise ValueError("max_energy_delta must be finite and positive")

    rng_int = int(np.sum(np.abs(_as_position(rng, name="rng")))) % (2**31)
    local_rng = np.random.RandomState(rng_int)
    momentum0 = local_rng.randn(position0.size)
    initial_energy = _energy(hamiltonian, position0, momentum0)

    theta_left = position0.copy()
    theta_right = position0.copy()
    momentum_left = momentum0.copy()
    momentum_right = momentum0.copy()
    candidates: list[tuple[np.ndarray, float]] = [(position0.copy(), initial_energy)]
    numerical_error = False
    realised_depth = 0

    for depth in range(max_depth):
        direction = -1 if local_rng.rand() < 0.5 else 1
        if direction == -1:
            theta = theta_left.copy()
            momentum = momentum_left.copy()
        else:
            theta = theta_right.copy()
            momentum = momentum_right.copy()

        subtree_leaves: list[tuple[np.ndarray, np.ndarray, float]] = []
        for _ in range(2 ** depth):
            theta, momentum = _leapfrog(hamiltonian, theta, momentum, step_size, direction)
            leaf_energy = _energy(hamiltonian, theta, momentum)
            if not np.isfinite(leaf_energy) or leaf_energy - initial_energy > max_energy_delta:
                numerical_error = True
                break
            subtree_leaves.append((theta.copy(), momentum.copy(), leaf_energy))

        if direction == -1 and subtree_leaves:
            theta_left, momentum_left = subtree_leaves[-1][0].copy(), subtree_leaves[-1][1].copy()
        elif direction == 1 and subtree_leaves:
            theta_right, momentum_right = subtree_leaves[-1][0].copy(), subtree_leaves[-1][1].copy()

        candidates.extend((theta_leaf.copy(), energy_leaf) for theta_leaf, _, energy_leaf in subtree_leaves)
        realised_depth = depth + 1
        if numerical_error or _is_turning(theta_left, theta_right, momentum_left, momentum_right):
            break

    energies = np.array([energy for _, energy in candidates], dtype=np.float64)
    weights = np.exp(-(energies - np.min(energies)))
    weights = weights / np.sum(weights)
    selected = int(local_rng.choice(len(candidates), p=weights))
    new_state = candidates[selected][0].copy()
    accept_probs = np.minimum(1.0, np.exp(initial_energy - energies))

    diagnostics = {
        "accept_prob": np.array(float(np.mean(accept_probs))),
        "tree_depth": np.array(realised_depth),
        "energy": np.array(float(candidates[selected][1])),
        "n_steps": np.array(len(candidates) - 1),
        "numerical_error": np.array(float(numerical_error)),
    }
    return (new_state, diagnostics)
"""Auto-generated FFI bindings for julia implementations."""


def _buildnutstree_ffi(rng, hamiltonian, start_state, direction, tree_depth, initial_energy):
    """Wrapper that calls the Julia version of build nuts tree. Passes arguments through and returns the result."""
    return _get_jl().eval("buildnutstree(rng, hamiltonian, start_state, direction, tree_depth, initial_energy)")

def _nutstransitionkernel_ffi(rng, hamiltonian, initial_state, trajectory_params):
    """Wrapper that calls the Julia version of nuts transition kernel. Passes arguments through and returns the result."""
    return _get_jl().eval("nutstransitionkernel(rng, hamiltonian, initial_state, trajectory_params)")
