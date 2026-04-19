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

_MIN_SCALE = 1e-12


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


def _particles_from_state(prior_state: ParticleState | np.ndarray) -> np.ndarray:
    particles = (
        prior_state.get("particles", prior_state)
        if isinstance(prior_state, Mapping)
        else prior_state
    )
    particles_array = np.asarray(particles, dtype=np.float64)
    if particles_array.size == 0:
        raise ValueError("particle array must not be empty")
    if particles_array.ndim == 0:
        particles_array = particles_array.reshape(1)
    return particles_array


def _n_particles(particles: np.ndarray) -> int:
    return int(particles.shape[0])


def _normalized_prior_weights(
    prior_state: ParticleState | np.ndarray,
    n_particles: int,
) -> np.ndarray:
    if isinstance(prior_state, Mapping) and "weights" in prior_state:
        weights = np.asarray(prior_state["weights"], dtype=np.float64)
    else:
        weights = np.ones(n_particles, dtype=np.float64)
    if weights.shape != (n_particles,):
        raise ValueError("weights must have one entry per particle")
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("weights must have positive finite total mass")
    normalized = weights / total
    if not np.all(np.isfinite(normalized)) or np.any(normalized < 0.0):
        raise ValueError("weights must be finite and non-negative")
    return normalized


def _positive_model_scale(
    model_spec: ModelSpec,
    key: str,
    default: float,
) -> float:
    value = model_spec.get(key, default) if isinstance(model_spec, Mapping) else default
    scale = float(value)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{key} must be a positive finite scale")
    return max(scale, _MIN_SCALE)


def _broadcast_control(control_t: ControlValue, particles: np.ndarray) -> np.ndarray:
    control = np.asarray(control_t, dtype=np.float64)
    if control.ndim == 0:
        return np.full_like(particles, float(control))
    return np.broadcast_to(control, particles.shape).astype(np.float64, copy=False)


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
    particles = _particles_from_state(prior_state)
    rng = np.random.RandomState(int(rng_key[0]) if len(rng_key) > 0 else 0)
    noise = rng.normal(
        loc=0.0,
        scale=_positive_model_scale(model_spec, "process_scale", 1.0),
        size=particles.shape,
    )
    proposed = particles + _broadcast_control(control_t, particles) + noise
    carry_weights = _normalized_prior_weights(prior_state, _n_particles(particles))
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
    particles = np.asarray(proposed_state_hypotheses, dtype=np.float64)
    if particles.size == 0:
        raise ValueError("proposed_state_hypotheses must not be empty")
    if particles.ndim == 0:
        particles = particles.reshape(1)
    n_particles = _n_particles(particles)
    weights = np.asarray(carry_weights, dtype=np.float64)
    if weights.shape != (n_particles,):
        raise ValueError("carry_weights must have one entry per particle")
    weight_total = float(np.sum(weights))
    if not np.isfinite(weight_total) or weight_total <= 0.0:
        raise ValueError("carry_weights must have positive finite total mass")
    weights = weights / weight_total
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("carry_weights must be finite and non-negative")

    obs = np.asarray(observation_t, dtype=np.float64)
    obs = obs.ravel()[0] if particles.ndim == 1 and obs.size == 1 else obs
    residual = particles - obs
    if particles.ndim > 1:
        residual = residual.reshape(n_particles, -1)
        dimensions = residual.shape[1]
        squared_error = np.sum(residual ** 2, axis=1)
    else:
        dimensions = 1
        squared_error = residual ** 2
    observation_scale = _positive_model_scale(model_spec, "observation_scale", 1.0)
    variance = observation_scale ** 2
    log_lik = -0.5 * (
        squared_error / variance
        + dimensions * np.log(2.0 * np.pi * variance)
    )
    log_weights = np.log(weights + 1e-300) + log_lik
    max_lw = np.max(log_weights)
    weights_exp = np.exp(log_weights - max_lw)
    total = float(weights_exp.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("likelihood weights underflowed or are not finite")
    normalized = weights_exp / total
    log_likelihood = float(max_lw + np.log(total))
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
    proposed = np.asarray(proposed_state_hypotheses, dtype=np.float64)
    weights = np.asarray(normalized_weights, dtype=np.float64)
    n = len(weights)
    if n == 0:
        raise ValueError("normalized_weights must not be empty")
    if proposed.shape[0] != n:
        raise ValueError("proposed_state_hypotheses and normalized_weights disagree")
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("normalized_weights must have positive finite total mass")
    weights = weights / total
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("normalized_weights must be finite and non-negative")
    rng = np.random.RandomState(int(rng_key_next[0]) if len(rng_key_next) > 0 else 0)
    positions = (rng.uniform() + np.arange(n)) / n
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0
    indices = np.searchsorted(cumsum, positions)
    indices = np.clip(indices, 0, n - 1)
    resampled = proposed[indices]
    uniform_weights = np.ones(n) / n
    posterior = {
        'particles': resampled,
        'weights': uniform_weights,
        'rng_seed': int(rng_key_next[0]) + 1 if len(rng_key_next) > 0 else 1,
    }
    trace = {
        'log_likelihood': log_likelihood,
        'ess': 1.0 / np.sum(weights ** 2),
    }
    return (posterior, trace)
