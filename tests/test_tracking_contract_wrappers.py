from __future__ import annotations

import numpy as np

from sciona.atoms.state_estimation.kalman_filters.atoms import track_linear_gaussian_state
from sciona.atoms.state_estimation.particle_filters.atoms import track_particle_hidden_state


def test_track_linear_gaussian_state_returns_posterior_series() -> None:
    observations = np.asarray([0.0, 0.2, 0.15, 0.35], dtype=float)

    means, covariances = track_linear_gaussian_state(observations)

    assert means.shape == observations.shape
    assert covariances.shape == observations.shape
    assert np.isfinite(means).all()
    assert np.isfinite(covariances).all()


def test_track_particle_hidden_state_returns_trace_series() -> None:
    observations = np.asarray([0.0, 0.25, 0.1, -0.05], dtype=float)

    means, ess_fractions, log_likelihoods = track_particle_hidden_state(observations)

    assert means.shape == observations.shape
    assert ess_fractions.shape == observations.shape
    assert log_likelihoods.shape == observations.shape
    assert np.isfinite(means).all()
    assert np.isfinite(ess_fractions).all()
    assert np.isfinite(log_likelihoods).all()
