"""Probe-side catalog for the SciPy sparse-graph atom family."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.scipy.sparse_graph"

SPARSE_GRAPH_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.graph_laplacian", _MODULE, "graph_laplacian"),
    ProbeTarget(f"{_MODULE}.graph_fourier_transform", _MODULE, "graph_fourier_transform"),
    ProbeTarget(
        f"{_MODULE}.inverse_graph_fourier_transform",
        _MODULE,
        "inverse_graph_fourier_transform",
    ),
    ProbeTarget(f"{_MODULE}.heat_kernel_diffusion", _MODULE, "heat_kernel_diffusion"),
    ProbeTarget(
        f"{_MODULE}.single_source_shortest_path",
        _MODULE,
        "single_source_shortest_path",
    ),
    ProbeTarget(
        f"{_MODULE}.all_pairs_shortest_path",
        _MODULE,
        "all_pairs_shortest_path",
    ),
    ProbeTarget(f"{_MODULE}.minimum_spanning_tree", _MODULE, "minimum_spanning_tree"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in SPARSE_GRAPH_PROBE_TARGETS
    ]
