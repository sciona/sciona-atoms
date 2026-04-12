"""Probe-side catalog for the NumPy FFT atom family."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.numpy.fft"

FFT_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.fft", _MODULE, "fft"),
    ProbeTarget(f"{_MODULE}.ifft", _MODULE, "ifft"),
    ProbeTarget(f"{_MODULE}.rfft", _MODULE, "rfft"),
    ProbeTarget(f"{_MODULE}.irfft", _MODULE, "irfft"),
    ProbeTarget(f"{_MODULE}.fftfreq", _MODULE, "fftfreq"),
    ProbeTarget(f"{_MODULE}.fftn", _MODULE, "fftn"),
    ProbeTarget(f"{_MODULE}.ifftn", _MODULE, "ifftn"),
    ProbeTarget(f"{_MODULE}.hfft", _MODULE, "hfft"),
    ProbeTarget(f"{_MODULE}.fftshift", _MODULE, "fftshift"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in FFT_PROBE_TARGETS
    ]
