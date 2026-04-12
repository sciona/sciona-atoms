"""Probe-side catalog for the SciPy spatial atom family."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.scipy.spatial"

SPATIAL_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.voronoi_tessellation", _MODULE, "voronoi_tessellation"),
    ProbeTarget(f"{_MODULE}.delaunay_triangulation", _MODULE, "delaunay_triangulation"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in SPATIAL_PROBE_TARGETS
    ]

