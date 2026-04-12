import importlib


def test_bayes_rs_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.inference.bayes_rs") is not None
    assert importlib.import_module("sciona.probes.inference.bayes_rs_bernoulli") is not None
