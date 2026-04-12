import importlib


def test_mcmc_foundational_mini_mcmc_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.inference.mcmc_foundational.mini_mcmc") is not None
    assert importlib.import_module("sciona.probes.inference.mcmc_foundational_mini_mcmc") is not None
