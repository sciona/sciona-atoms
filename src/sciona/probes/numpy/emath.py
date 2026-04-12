"""Probe-side catalog for the NumPy emath atom family."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.numpy.emath"

EMATH_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.sqrt", _MODULE, "sqrt"),
    ProbeTarget(f"{_MODULE}.log", _MODULE, "log"),
    ProbeTarget(f"{_MODULE}.log10", _MODULE, "log10"),
    ProbeTarget(f"{_MODULE}.logn", _MODULE, "logn"),
    ProbeTarget(f"{_MODULE}.power", _MODULE, "power"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in EMATH_PROBE_TARGETS
    ]
