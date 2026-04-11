from __future__ import annotations

from importlib import import_module
from pathlib import Path
import json


def test_sequential_filter_namespace_modules_import_cleanly() -> None:
    atoms = import_module("sciona.atoms.state_estimation.kalman_filters.filter_rs")
    probes = import_module("sciona.probes.state_estimation.kalman_filter_rs")
    assert hasattr(atoms, "evaluatemeasurementoracle")
    assert hasattr(probes, "KALMAN_FILTER_RS_PROBE_TARGETS")


def test_sequential_filter_family_asset_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    asset = root / "data" / "heuristics" / "families" / "sequential_filter.json"
    payload = json.loads(asset.read_text(encoding="utf-8"))
    assert payload["family"] == "sequential_filter"
    assert "kalman_filter" in payload["family_aliases"]
