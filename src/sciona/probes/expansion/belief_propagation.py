"""Probe-side catalog for provider-owned belief-propagation expansion atoms."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.expansion.belief_propagation"

BELIEF_PROPAGATION_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.monitor_message_convergence", _MODULE, "monitor_message_convergence"),
    ProbeTarget(
        f"{_MODULE}.validate_belief_normalization",
        _MODULE,
        "validate_belief_normalization",
    ),
    ProbeTarget(f"{_MODULE}.analyze_message_damping", _MODULE, "analyze_message_damping"),
    ProbeTarget(f"{_MODULE}.detect_graph_cycles", _MODULE, "detect_graph_cycles"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in BELIEF_PROPAGATION_PROBE_TARGETS
    ]
