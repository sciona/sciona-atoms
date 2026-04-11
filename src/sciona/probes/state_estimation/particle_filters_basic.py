"""Probe-side catalog for the sequential-filter particle-filter slice."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.state_estimation.particle_filters.basic"

PARTICLE_FILTER_BASIC_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.filter_step_preparation_and_dispatch", _MODULE, "filter_step_preparation_and_dispatch"),
    ProbeTarget(f"{_MODULE}.hypothesis_propagation_kernel", _MODULE, "hypothesis_propagation_kernel"),
    ProbeTarget(f"{_MODULE}.likelihood_reweight_kernel", _MODULE, "likelihood_reweight_kernel"),
    ProbeTarget(f"{_MODULE}.resample_and_hypothesis_distribution_projection", _MODULE, "resample_and_hypothesis_distribution_projection"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in PARTICLE_FILTER_BASIC_PROBE_TARGETS
    ]
