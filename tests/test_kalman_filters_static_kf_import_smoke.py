from __future__ import annotations

import importlib


def test_kalman_filters_static_kf_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.state_estimation.kalman_filters.static_kf") is not None
    assert importlib.import_module("sciona.probes.state_estimation.kalman_filters_static_kf") is not None
