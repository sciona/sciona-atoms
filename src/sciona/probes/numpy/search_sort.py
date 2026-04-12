"""Probe-side catalog for the NumPy search-and-sort atom family."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.numpy.search_sort"

SEARCH_SORT_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.binary_search_insertion", _MODULE, "binary_search_insertion"),
    ProbeTarget(f"{_MODULE}.lexicographic_indirect_sort", _MODULE, "lexicographic_indirect_sort"),
    ProbeTarget(f"{_MODULE}.partial_sort_partition", _MODULE, "partial_sort_partition"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in SEARCH_SORT_PROBE_TARGETS
    ]
