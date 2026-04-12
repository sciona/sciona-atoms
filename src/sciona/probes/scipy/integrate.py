"""Probe-side catalog for the SciPy integrate atom family."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.scipy.integrate"

INTEGRATE_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.quad", _MODULE, "quad"),
    ProbeTarget(f"{_MODULE}.solve_ivp", _MODULE, "solve_ivp"),
    ProbeTarget(f"{_MODULE}.simpson", _MODULE, "simpson"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in INTEGRATE_PROBE_TARGETS
    ]
