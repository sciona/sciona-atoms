"""Probe-side catalog for provider-owned sequential-filter expansion atoms."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.expansion.sequential_filter"

SEQUENTIAL_FILTER_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.check_observability", _MODULE, "check_observability"),
    ProbeTarget(
        f"{_MODULE}.validate_innovation_whiteness",
        _MODULE,
        "validate_innovation_whiteness",
    ),
    ProbeTarget(f"{_MODULE}.detect_filter_divergence", _MODULE, "detect_filter_divergence"),
    ProbeTarget(f"{_MODULE}.adapt_process_noise", _MODULE, "adapt_process_noise"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in SEQUENTIAL_FILTER_PROBE_TARGETS
    ]
