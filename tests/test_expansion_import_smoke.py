from __future__ import annotations

from importlib import import_module


def test_expansion_provider_modules_import_cleanly() -> None:
    families = (
        "divide_and_conquer",
        "sequential_filter",
        "belief_propagation",
        "kalman_filter",
        "particle_filter",
    )
    for family in families:
        atoms = import_module(f"sciona.atoms.expansion.{family}")
        probes = import_module(f"sciona.probes.expansion.{family}")
        assert hasattr(atoms, f"{family.upper()}_DECLARATIONS")
        assert hasattr(probes, f"{family.upper()}_PROBE_TARGETS")
