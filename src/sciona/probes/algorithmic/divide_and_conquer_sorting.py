"""Probe-side catalog for divide-and-conquer sorting wrappers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.algorithmic.divide_and_conquer.sorting"

DIVIDE_AND_CONQUER_SORTING_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.merge_sort", _MODULE, "merge_sort"),
    ProbeTarget(f"{_MODULE}.quicksort", _MODULE, "quicksort"),
    ProbeTarget(f"{_MODULE}.heapsort", _MODULE, "heapsort"),
    ProbeTarget(f"{_MODULE}.counting_sort", _MODULE, "counting_sort"),
    ProbeTarget(f"{_MODULE}.radix_sort", _MODULE, "radix_sort"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in DIVIDE_AND_CONQUER_SORTING_PROBE_TARGETS
    ]
