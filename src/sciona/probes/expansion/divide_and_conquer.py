"""Probe-side catalog for provider-owned divide-and-conquer expansion atoms."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.expansion.divide_and_conquer"

DIVIDE_AND_CONQUER_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.measure_split_balance", _MODULE, "measure_split_balance"),
    ProbeTarget(f"{_MODULE}.check_recursion_depth", _MODULE, "check_recursion_depth"),
    ProbeTarget(f"{_MODULE}.profile_merge_cost", _MODULE, "profile_merge_cost"),
    ProbeTarget(f"{_MODULE}.detect_subproblem_overlap", _MODULE, "detect_subproblem_overlap"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in DIVIDE_AND_CONQUER_PROBE_TARGETS
    ]
