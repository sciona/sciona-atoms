"""Probe-side catalog for the signal-processing BioSPPy ECG pilot."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.signal_processing.biosppy.ecg"

ECG_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.bandpass_filter", _MODULE, "bandpass_filter"),
    ProbeTarget(f"{_MODULE}.r_peak_detection", _MODULE, "r_peak_detection"),
    ProbeTarget(f"{_MODULE}.peak_correction", _MODULE, "peak_correction"),
    ProbeTarget(f"{_MODULE}.reject_outlier_intervals", _MODULE, "reject_outlier_intervals"),
    ProbeTarget(f"{_MODULE}.template_extraction", _MODULE, "template_extraction"),
    ProbeTarget(f"{_MODULE}.heart_rate_computation", _MODULE, "heart_rate_computation"),
    ProbeTarget(
        f"{_MODULE}.heart_rate_computation_median_smoothed",
        _MODULE,
        "heart_rate_computation_median_smoothed",
    ),
    ProbeTarget(f"{_MODULE}.ssf_segmenter", _MODULE, "ssf_segmenter"),
    ProbeTarget(f"{_MODULE}.christov_segmenter", _MODULE, "christov_segmenter"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in ECG_PROBE_TARGETS
    ]
