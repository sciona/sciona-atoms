from __future__ import annotations

from importlib import import_module


def test_namespace_modules_import_cleanly() -> None:
    atoms = import_module("sciona.atoms.state_estimation.particle_filters.basic")
    probes = import_module("sciona.probes.state_estimation.particle_filters_basic")
    assert hasattr(atoms, "hypothesis_propagation_kernel")
    assert hasattr(probes, "PARTICLE_FILTER_BASIC_PROBE_TARGETS")
