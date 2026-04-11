from __future__ import annotations

from importlib import import_module
from pathlib import Path
import json


def test_divide_and_conquer_namespace_modules_import_cleanly() -> None:
    atoms = import_module('sciona.atoms.algorithmic.divide_and_conquer.sorting')
    probes = import_module('sciona.probes.algorithmic.divide_and_conquer_sorting')
    assert hasattr(atoms, 'merge_sort')
    assert hasattr(probes, 'DIVIDE_AND_CONQUER_SORTING_PROBE_TARGETS')


def test_divide_and_conquer_family_asset_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    asset = root / 'data' / 'heuristics' / 'families' / 'divide_and_conquer.json'
    payload = json.loads(asset.read_text(encoding='utf-8'))
    assert payload['family'] == 'divide_and_conquer'
