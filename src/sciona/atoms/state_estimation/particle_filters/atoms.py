"""Contract-level particle tracking atoms."""

from __future__ import annotations

import icontract
import numpy as np
from sciona.ghost.abstract import AbstractArray, AbstractScalar
from sciona.ghost.registry import register_atom

from .basic import (
    filter_step_preparation_and_dispatch,
    hypothesis_propagation_kernel,
    likelihood_reweight_kernel,
    resample_and_hypothesis_distribution_projection,
)


def witness_track_particle_hidden_state(
    observations: AbstractArray,
    rng_seed: AbstractScalar,
) -> tuple[AbstractArray, AbstractArray, AbstractArray]:
    """Describe posterior means, ESS fractions, and log likelihoods for particle tracking."""
    length = observations.shape[0] if observations.shape else 0
    posterior_means = AbstractArray(shape=(length,), dtype="float64")
    ess_fractions = AbstractArray(shape=(length,), dtype="float64")
    log_likelihoods = AbstractArray(shape=(length,), dtype="float64")
    return posterior_means, ess_fractions, log_likelihoods


@register_atom(witness_track_particle_hidden_state)
@icontract.require(lambda observations: isinstance(observations, np.ndarray), "observations must be an ndarray")
@icontract.require(
    lambda rng_seed: isinstance(rng_seed, (int, np.integer)),
    "rng_seed must be an integer",
)
@icontract.ensure(lambda result: result is not None, "track_particle_hidden_state output must not be None")
def track_particle_hidden_state(
    observations: np.ndarray,
    *,
    rng_seed: int = 7,
    n_particles: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Track a scalar latent state with a deterministic bootstrap particle filter."""
    obs = np.asarray(observations, dtype=float).reshape(-1)
    if obs.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty

    state = {
        "particles": np.linspace(-0.5, 0.5, int(n_particles), dtype=np.float64),
        "weights": np.ones(int(n_particles), dtype=np.float64) / float(n_particles),
        "rng_seed": int(rng_seed),
    }
    model_spec = {"process_scale": 0.12, "observation_scale": 0.25}
    posterior_means: list[float] = []
    ess_fractions: list[float] = []
    log_likelihoods: list[float] = []

    for measurement in obs:
        _, _, _, _, rng_key = filter_step_preparation_and_dispatch(
            state,
            model_spec,
            0.0,
            np.array([measurement], dtype=float),
        )
        proposed, carry_weights, rng_key_next = hypothesis_propagation_kernel(
            state,
            model_spec,
            0.0,
            rng_key,
        )
        normalized_weights, log_likelihood = likelihood_reweight_kernel(
            proposed,
            carry_weights,
            np.array([measurement], dtype=float),
            model_spec,
        )
        state, trace = resample_and_hypothesis_distribution_projection(
            proposed,
            normalized_weights,
            rng_key_next,
            log_likelihood,
        )
        posterior_means.append(float(np.mean(np.asarray(state["particles"], dtype=float))))
        ess_fractions.append(float(trace["ess"]) / float(n_particles))
        log_likelihoods.append(float(trace["log_likelihood"]))

    return (
        np.asarray(posterior_means, dtype=np.float64),
        np.asarray(ess_fractions, dtype=np.float64),
        np.asarray(log_likelihoods, dtype=np.float64),
    )
