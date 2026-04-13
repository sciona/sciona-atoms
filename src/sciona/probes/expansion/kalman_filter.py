"""Probe-side catalog for provider-owned Kalman-filter expansion atoms."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.expansion.kalman_filter"

KALMAN_FILTER_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.check_innovation_consistency", _MODULE, "check_innovation_consistency"),
    ProbeTarget(f"{_MODULE}.validate_covariance_pd", _MODULE, "validate_covariance_pd"),
    ProbeTarget(
        f"{_MODULE}.analyze_kalman_gain_magnitude",
        _MODULE,
        "analyze_kalman_gain_magnitude",
    ),
    ProbeTarget(f"{_MODULE}.check_state_smoothness", _MODULE, "check_state_smoothness"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in KALMAN_FILTER_PROBE_TARGETS
    ]
