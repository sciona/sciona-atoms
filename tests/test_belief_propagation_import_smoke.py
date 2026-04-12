import importlib


def test_belief_propagation_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.inference.belief_propagation.loopy_bp") is not None
    assert importlib.import_module("sciona.probes.inference.belief_propagation_loopy_bp") is not None
