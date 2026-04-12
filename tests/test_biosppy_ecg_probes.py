from __future__ import annotations

from importlib import import_module

from sciona.probes.signal_processing.biosppy_ecg import probe_records


def test_probe_records_resolve_to_live_symbols() -> None:
    for record in probe_records():
        module = import_module(str(record["module_import_path"]))
        assert hasattr(module, str(record["wrapper_symbol"]))
        fqdn_parts = str(record["atom_fqdn"]).split(".")
        imported = import_module(".".join(fqdn_parts[:-1]))
        assert getattr(imported, fqdn_parts[-1]) is getattr(
            module,
            str(record["wrapper_symbol"]),
        )


def test_probe_records_publish_expected_signal_processing_symbols() -> None:
    wrapper_symbols = {str(record["wrapper_symbol"]) for record in probe_records()}
    assert wrapper_symbols == {
        "bandpass_filter",
        "r_peak_detection",
        "peak_correction",
        "reject_outlier_intervals",
        "template_extraction",
        "heart_rate_computation",
        "heart_rate_computation_median_smoothed",
        "ssf_segmenter",
        "christov_segmenter",
    }
