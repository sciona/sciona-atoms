"""Probe-side catalog for the SciPy interpolate atom family."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.scipy.interpolate"

INTERPOLATE_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.cubic_spline_fit", _MODULE, "cubic_spline_fit"),
    ProbeTarget(f"{_MODULE}.rbf_interpolator_fit", _MODULE, "rbf_interpolator_fit"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in INTERPOLATE_PROBE_TARGETS
    ]
