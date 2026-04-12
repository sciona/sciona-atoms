import importlib


def test_mcmc_foundational_kthohr_mcmc_nuts_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.nuts") is not None
    assert importlib.import_module("sciona.probes.inference.mcmc_foundational_kthohr_mcmc_nuts") is not None
