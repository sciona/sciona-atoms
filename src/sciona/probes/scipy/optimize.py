"""Probe-side catalog for the SciPy optimize atom family."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.scipy.optimize"

OPTIMIZE_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.minimize", _MODULE, "minimize"),
    ProbeTarget(f"{_MODULE}.root", _MODULE, "root"),
    ProbeTarget(f"{_MODULE}.linprog", _MODULE, "linprog"),
    ProbeTarget(f"{_MODULE}.curve_fit", _MODULE, "curve_fit"),
    ProbeTarget(f"{_MODULE}.shgo", _MODULE, "shgo"),
    ProbeTarget(
        f"{_MODULE}.differential_evolution",
        _MODULE,
        "differential_evolution",
    ),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in OPTIMIZE_PROBE_TARGETS
    ]
