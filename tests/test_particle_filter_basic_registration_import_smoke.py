from __future__ import annotations

import importlib

from sciona.ghost.registry import list_registered


def test_particle_filter_basic_registration_import_smoke() -> None:
    atoms = importlib.import_module('sciona.atoms.state_estimation.particle_filters.basic')
    assert hasattr(atoms, 'filter_step_preparation_and_dispatch')
    registered = set(list_registered())
    assert 'filter_step_preparation_and_dispatch' in registered
    assert 'hypothesis_propagation_kernel' in registered
    assert 'likelihood_reweight_kernel' in registered
    assert 'resample_and_hypothesis_distribution_projection' in registered
