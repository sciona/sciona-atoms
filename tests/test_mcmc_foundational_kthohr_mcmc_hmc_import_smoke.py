import importlib


def test_mcmc_foundational_kthohr_mcmc_hmc_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.hmc") is not None
    assert importlib.import_module("sciona.probes.inference.mcmc_foundational_kthohr_mcmc_hmc") is not None
