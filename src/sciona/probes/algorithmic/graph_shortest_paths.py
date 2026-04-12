"""Probe-side catalog for graph shortest-path wrappers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.algorithmic.graph.shortest_paths"

GRAPH_SHORTEST_PATHS_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.dijkstra", _MODULE, "dijkstra"),
    ProbeTarget(f"{_MODULE}.bellman_ford", _MODULE, "bellman_ford"),
    ProbeTarget(f"{_MODULE}.floyd_warshall", _MODULE, "floyd_warshall"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in GRAPH_SHORTEST_PATHS_PROBE_TARGETS
    ]
