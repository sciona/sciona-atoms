import importlib


def test_mcmc_foundational_advancedhmc_integrator_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.inference.mcmc_foundational.advancedhmc.integrator") is not None
    assert importlib.import_module("sciona.probes.inference.mcmc_foundational_advancedhmc_integrator") is not None
