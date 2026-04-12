"""Probe-side catalog for the sequential-filter Kalman reference slice."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.state_estimation.kalman_filters.filter_rs"

KALMAN_FILTER_RS_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.initialize_kalman_state_model", _MODULE, "initialize_kalman_state_model"),
    ProbeTarget(f"{_MODULE}.predict_latent_state_and_covariance", _MODULE, "predict_latent_state_and_covariance"),
    ProbeTarget(f"{_MODULE}.predict_latent_state_steady_state", _MODULE, "predict_latent_state_steady_state"),
    ProbeTarget(f"{_MODULE}.evaluate_measurement_oracle", _MODULE, "evaluate_measurement_oracle"),
    ProbeTarget(f"{_MODULE}.update_posterior_state_and_covariance", _MODULE, "update_posterior_state_and_covariance"),
    ProbeTarget(f"{_MODULE}.update_posterior_state_steady_state", _MODULE, "update_posterior_state_steady_state"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in KALMAN_FILTER_RS_PROBE_TARGETS
    ]
