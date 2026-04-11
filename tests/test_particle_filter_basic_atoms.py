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
