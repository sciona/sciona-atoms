from __future__ import annotations

import numpy as np

from sciona.atoms.state_estimation.particle_filters.basic import (
    filter_step_preparation_and_dispatch,
    hypothesis_propagation_kernel,
    likelihood_reweight_kernel,
    resample_and_hypothesis_distribution_projection,
)


def test_filter_step_preparation_threads_rng_key() -> None:
    prior = {"particles": np.array([0.0, 1.0]), "weights": np.array([0.5, 0.5]), "rng_seed": 7}
    result = filter_step_preparation_and_dispatch(prior, {}, 0.0, np.array([0.0]))
    assert result[-1].tolist() == [7]


def test_particle_filter_step_returns_finite_outputs() -> None:
    prior = {"particles": np.array([0.0, 1.0, 2.0]), "weights": np.array([0.2, 0.3, 0.5]), "rng_seed": 1}
    _prior, model, control, observation, rng_key = filter_step_preparation_and_dispatch(
        prior,
        {},
        0.0,
        np.array([0.5]),
    )
    proposed, weights, rng_next = hypothesis_propagation_kernel(prior, model, control, rng_key)
    normalized, log_likelihood = likelihood_reweight_kernel(proposed, weights, observation, model)
    posterior, trace = resample_and_hypothesis_distribution_projection(
        proposed,
        normalized,
        rng_next,
        log_likelihood,
    )
    assert np.isfinite(proposed).all()
    assert np.isfinite(normalized).all()
    assert np.isfinite(log_likelihood)
    assert np.isfinite(trace["ess"])
    assert len(posterior["particles"]) == 3


def test_particle_filter_step_is_reproducible_for_fixed_seed() -> None:
    prior = {
        "particles": np.array([0.0, 1.0, 2.0, 3.0]),
        "weights": np.array([0.1, 0.2, 0.3, 0.4]),
        "rng_seed": 11,
    }
    _prior, model, control, observation, rng_key = filter_step_preparation_and_dispatch(
        prior,
        {},
        0.0,
        np.array([1.25]),
    )
    proposed_a, weights_a, rng_next_a = hypothesis_propagation_kernel(prior, model, control, rng_key)
    proposed_b, weights_b, rng_next_b = hypothesis_propagation_kernel(prior, model, control, rng_key)
    normalized_a, log_likelihood_a = likelihood_reweight_kernel(
        proposed_a,
        weights_a,
        observation,
        model,
    )
    normalized_b, log_likelihood_b = likelihood_reweight_kernel(
        proposed_b,
        weights_b,
        observation,
        model,
    )
    posterior_a, trace_a = resample_and_hypothesis_distribution_projection(
        proposed_a,
        normalized_a,
        rng_next_a,
        log_likelihood_a,
    )
    posterior_b, trace_b = resample_and_hypothesis_distribution_projection(
        proposed_b,
        normalized_b,
        rng_next_b,
        log_likelihood_b,
    )
    assert np.allclose(proposed_a, proposed_b)
    assert np.allclose(normalized_a, normalized_b)
    assert log_likelihood_a == log_likelihood_b
    assert np.array_equal(posterior_a["particles"], posterior_b["particles"])
    assert posterior_a["rng_seed"] == posterior_b["rng_seed"]
    assert trace_a == trace_b


def test_resample_projection_reinitializes_uniform_weights() -> None:
    proposed = np.array([10.0, 20.0, 30.0, 40.0])
    normalized = np.array([0.05, 0.15, 0.3, 0.5])
    posterior, trace = resample_and_hypothesis_distribution_projection(
        proposed,
        normalized,
        np.array([9], dtype=np.int64),
        -3.25,
    )
    assert np.array_equal(posterior["weights"], np.ones(4) / 4)
    assert posterior["particles"].shape == proposed.shape
    assert posterior["rng_seed"] == 10
    assert trace["log_likelihood"] == -3.25
    assert trace["ess"] > 0.0
