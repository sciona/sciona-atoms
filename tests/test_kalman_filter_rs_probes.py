from __future__ import annotations

from importlib import import_module

from sciona.probes.state_estimation.kalman_filter_rs import probe_records


def test_probe_records_resolve_to_live_symbols() -> None:
    for record in probe_records():
        module = import_module(str(record["module_import_path"]))
        assert hasattr(module, str(record["wrapper_symbol"]))
