from __future__ import annotations

import importlib


def test_jax_advi_optimize_advi_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.inference.jax_advi.optimize_advi") is not None
    assert importlib.import_module("sciona.probes.inference.jax_advi_optimize_advi") is not None
