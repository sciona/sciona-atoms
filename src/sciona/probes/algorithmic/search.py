"""Probe-side catalog for the search wrappers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.algorithmic.search"

SEARCH_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.binary_search", _MODULE, "binary_search"),
    ProbeTarget(f"{_MODULE}.linear_search", _MODULE, "linear_search"),
    ProbeTarget(f"{_MODULE}.hash_lookup", _MODULE, "hash_lookup"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in SEARCH_PROBE_TARGETS
    ]
