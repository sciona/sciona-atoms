"""Probe-side catalog for the SciPy linalg atom family."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.scipy.linalg"

LINALG_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.solve", _MODULE, "solve"),
    ProbeTarget(f"{_MODULE}.inv", _MODULE, "inv"),
    ProbeTarget(f"{_MODULE}.det", _MODULE, "det"),
    ProbeTarget(f"{_MODULE}.lu_factor", _MODULE, "lu_factor"),
    ProbeTarget(f"{_MODULE}.lu_solve", _MODULE, "lu_solve"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in LINALG_PROBE_TARGETS
    ]
