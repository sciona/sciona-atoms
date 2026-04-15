"""Particle filter reference wrappers for the sequential-filter namespace slice."""

from __future__ import annotations

from collections.abc import Mapping

import icontract
import numpy as np
from sciona.ghost.abstract import AbstractArray, AbstractScalar
from sciona.ghost.registry import register_atom

ParticleState = Mapping[str, np.ndarray | int]
ModelSpec = Mapping[str, object]
ControlValue = np.ndarray | float | int
ObservationValue = np.ndarray | float | int


def witness_filter_step_preparation_and_dispatch(
    up: AbstractArray,
    b: AbstractArray,
    a: AbstractScalar,
    o: AbstractArray,
) -> tuple[AbstractArray, AbstractArray, AbstractScalar, AbstractArray, AbstractArray]:
    """Describe a prepared particle-filter step plus deterministic RNG key."""
    rng_key = AbstractArray(shape=(1,), dtype='int64')
    return up, b, a, o, rng_key


def witness_hypothesis_propagation_kernel(
    prior_state: AbstractArray,
    model_spec: AbstractArray,
    control_t: AbstractScalar,
    rng_key: AbstractArray,
) -> tuple[AbstractArray, AbstractArray, AbstractArray]:
    """Describe propagated particle hypotheses, carried weights, and next RNG key."""
    n_particles = prior_state.shape[0] if prior_state.shape else 0
    proposed = AbstractArray(shape=prior_state.shape, dtype='float64')
    carry_weights = AbstractArray(shape=(n_particles,), dtype='float64')
    rng_key_next = AbstractArray(shape=rng_key.shape, dtype='int64')
    return proposed, carry_weights, rng_key_next


def witness_likelihood_reweight_kernel(
    proposed_state_hypotheses: AbstractArray,
    carry_weights: AbstractArray,
    observation_t: AbstractArray,
    model_spec: AbstractArray,
) -> tuple[AbstractArray, AbstractScalar]:
    """Describe normalized particle weights and scalar log likelihood."""
    normalized = AbstractArray(shape=carry_weights.shape, dtype='float64')
    log_likelihood = AbstractScalar(dtype='float64')
    return normalized, log_likelihood


def witness_resample_and_hypothesis_distribution_projection(
    proposed_state_hypotheses: AbstractArray,
    normalized_weights: AbstractArray,
    rng_key_next: AbstractArray,
    log_likelihood: AbstractScalar,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe a resampled posterior state and scalar trace summary."""
    posterior = AbstractArray(shape=proposed_state_hypotheses.shape, dtype='float64')
    trace = AbstractArray(shape=(2,), dtype='float64')
    return posterior, trace


def _rng_key_from_state(up: ParticleState) -> np.ndarray:
    rng_seed = up["rng_seed"]
    if isinstance(rng_seed, np.integer):
        rng_seed = int(rng_seed)
    if not isinstance(rng_seed, int):
        raise TypeError("prior state rng_seed must be an int")
    return np.array([rng_seed], dtype=np.int64)


@register_atom(witness_filter_step_preparation_and_dispatch)
@icontract.require(lambda up: up is not None, "prior state up cannot be None")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def filter_step_preparation_and_dispatch(
    up: ParticleState,
    b: ModelSpec,
    a: ControlValue,
    o: ObservationValue,
) -> tuple[ParticleState, ModelSpec, ControlValue, ObservationValue, np.ndarray]:
    """Prepare one explicit particle-filter step context."""
    rng_key = _rng_key_from_state(up)
    return (up, b, a, o, rng_key)


@register_atom(witness_hypothesis_propagation_kernel)
@icontract.require(lambda prior_state: prior_state is not None, "prior_state cannot be None")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def hypothesis_propagation_kernel(
    prior_state: ParticleState,
    model_spec: ModelSpec,
    control_t: ControlValue,
    rng_key: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate particle hypotheses under a noisy transition."""
    particles = (
        prior_state.get("particles", prior_state)
        if isinstance(prior_state, dict)
        else prior_state
    )
    rng = np.random.RandomState(int(rng_key[0]) if len(rng_key) > 0 else 0)
    n_particles = len(particles) if hasattr(particles, "__len__") else 1
    noise = (
        rng.randn(n_particles)
        if isinstance(particles, np.ndarray)
        else rng.randn(1)
    )
    proposed = np.asarray(particles) + noise
    carry_weights = (
        prior_state.get("weights", np.ones(n_particles) / n_particles)
        if isinstance(prior_state, dict)
        else np.ones(n_particles) / n_particles
    )
    rng_key_next = np.array(
        [int(rng_key[0]) + 1 if len(rng_key) > 0 else 1],
        dtype=np.int64,
    )
    return (proposed, carry_weights, rng_key_next)


@register_atom(witness_likelihood_reweight_kernel)
@icontract.require(lambda proposed_state_hypotheses: proposed_state_hypotheses is not None, "proposed_state_hypotheses cannot be None")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def likelihood_reweight_kernel(
    proposed_state_hypotheses: np.ndarray,
    carry_weights: np.ndarray,
    observation_t: np.ndarray,
    model_spec: ModelSpec,
) -> tuple[np.ndarray, float]:
    """Reweight particle hypotheses against the current observation."""
    obs = np.asarray(observation_t)
    particles = np.asarray(proposed_state_hypotheses)
    log_lik = (
        -0.5 * np.sum((particles - obs) ** 2, axis=-1)
        if particles.ndim > 1
        else -0.5 * (particles - obs.ravel()[0]) ** 2
    )
    log_weights = np.log(carry_weights + 1e-300) + log_lik
    max_lw = np.max(log_weights)
    weights_exp = np.exp(log_weights - max_lw)
    total = weights_exp.sum()
    normalized = weights_exp / total
    log_likelihood = float(max_lw + np.log(total) - np.log(len(particles)))
    return (normalized, log_likelihood)


@register_atom(witness_resample_and_hypothesis_distribution_projection)
@icontract.require(lambda log_likelihood: isinstance(log_likelihood, (float, int, np.number)), "log_likelihood must be numeric")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def resample_and_hypothesis_distribution_projection(
    proposed_state_hypotheses: np.ndarray,
    normalized_weights: np.ndarray,
    rng_key_next: np.ndarray,
    log_likelihood: float,
) -> tuple[object, object]:
    """Resample weighted hypotheses into a posterior particle state and trace."""
    n = len(normalized_weights)
    rng = np.random.RandomState(int(rng_key_next[0]) if len(rng_key_next) > 0 else 0)
    positions = (rng.uniform() + np.arange(n)) / n
    cumsum = np.cumsum(normalized_weights)
    indices = np.searchsorted(cumsum, positions)
    indices = np.clip(indices, 0, n - 1)
    resampled = proposed_state_hypotheses[indices]
    uniform_weights = np.ones(n) / n
    posterior = {
        'particles': resampled,
        'weights': uniform_weights,
        'rng_seed': int(rng_key_next[0]) + 1 if len(rng_key_next) > 0 else 1,
    }
    trace = {
        'log_likelihood': log_likelihood,
        'ess': 1.0 / np.sum(normalized_weights ** 2),
    }
    return (posterior, trace)
