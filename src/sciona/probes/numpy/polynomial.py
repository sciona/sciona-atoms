"""Probe-side catalog for the NumPy polynomial atom family."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.numpy.polynomial"

POLYNOMIAL_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.polyval", _MODULE, "polyval"),
    ProbeTarget(f"{_MODULE}.polyfit", _MODULE, "polyfit"),
    ProbeTarget(f"{_MODULE}.polyder", _MODULE, "polyder"),
    ProbeTarget(f"{_MODULE}.polyint", _MODULE, "polyint"),
    ProbeTarget(f"{_MODULE}.polyadd", _MODULE, "polyadd"),
    ProbeTarget(f"{_MODULE}.polymul", _MODULE, "polymul"),
    ProbeTarget(f"{_MODULE}.polyroots", _MODULE, "polyroots"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in POLYNOMIAL_PROBE_TARGETS
    ]
