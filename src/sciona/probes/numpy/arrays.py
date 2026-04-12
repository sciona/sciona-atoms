"""Probe-side catalog for the NumPy arrays atom family."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.numpy.arrays"

ARRAYS_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.array", _MODULE, "array"),
    ProbeTarget(f"{_MODULE}.zeros", _MODULE, "zeros"),
    ProbeTarget(f"{_MODULE}.dot", _MODULE, "dot"),
    ProbeTarget(f"{_MODULE}.vstack", _MODULE, "vstack"),
    ProbeTarget(f"{_MODULE}.reshape", _MODULE, "reshape"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in ARRAYS_PROBE_TARGETS
    ]
