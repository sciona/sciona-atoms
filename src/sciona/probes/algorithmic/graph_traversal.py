"""Probe-side catalog for graph traversal wrappers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.algorithmic.graph.traversal"

GRAPH_TRAVERSAL_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.bfs", _MODULE, "bfs"),
    ProbeTarget(f"{_MODULE}.dfs", _MODULE, "dfs"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in GRAPH_TRAVERSAL_PROBE_TARGETS
    ]
