from __future__ import annotations

import importlib


def test_numpy_random_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.numpy.random")
    probes = importlib.import_module("sciona.probes.numpy.random")
    assert hasattr(atoms, "rand")
    assert hasattr(atoms, "uniform")
    assert hasattr(atoms, "default_rng")
    assert hasattr(atoms, "continuous_multivariate_sampler")
    assert hasattr(atoms, "discrete_event_sampler")
    assert hasattr(atoms, "combinatorics_sampler")
    assert hasattr(probes, "RANDOM_PROBE_TARGETS")
