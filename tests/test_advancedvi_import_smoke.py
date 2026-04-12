import importlib

def test_advancedvi_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.inference.advancedvi") is not None
    assert importlib.import_module("sciona.probes.inference.advancedvi") is not None
