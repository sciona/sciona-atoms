"""Probe-side catalog for provider-owned particle-filter expansion atoms."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.expansion.particle_filter"

PARTICLE_FILTER_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.monitor_effective_sample_size", _MODULE, "monitor_effective_sample_size"),
    ProbeTarget(f"{_MODULE}.analyze_particle_diversity", _MODULE, "analyze_particle_diversity"),
    ProbeTarget(f"{_MODULE}.track_weight_variance", _MODULE, "track_weight_variance"),
    ProbeTarget(f"{_MODULE}.check_resampling_quality", _MODULE, "check_resampling_quality"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in PARTICLE_FILTER_PROBE_TARGETS
    ]
