from __future__ import annotations

import importlib


def test_conjugate_priors_beta_binom_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.inference.conjugate_priors.beta_binom") is not None
    assert importlib.import_module("sciona.probes.inference.conjugate_priors_beta_binom") is not None
