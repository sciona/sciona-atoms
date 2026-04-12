"""Probe-side catalog for the SciPy signal atom family."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.scipy.signal"

SIGNAL_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.butter", _MODULE, "butter"),
    ProbeTarget(f"{_MODULE}.cheby1", _MODULE, "cheby1"),
    ProbeTarget(f"{_MODULE}.cheby2", _MODULE, "cheby2"),
    ProbeTarget(f"{_MODULE}.firwin", _MODULE, "firwin"),
    ProbeTarget(f"{_MODULE}.sosfilt", _MODULE, "sosfilt"),
    ProbeTarget(f"{_MODULE}.lfilter", _MODULE, "lfilter"),
    ProbeTarget(f"{_MODULE}.freqz", _MODULE, "freqz"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in SIGNAL_PROBE_TARGETS
    ]
