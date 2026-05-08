"""Probe-side catalog for the static Kalman-filter slice."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.state_estimation.kalman_filters.static_kf"

KALMAN_FILTERS_STATIC_KF_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(
        f"{_MODULE}.initialize_linear_gaussian_state_model",
        _MODULE,
        "initialize_linear_gaussian_state_model",
    ),
    ProbeTarget(f"{_MODULE}.predict_latent_state", _MODULE, "predict_latent_state"),
    ProbeTarget(f"{_MODULE}.update_with_measurement", _MODULE, "update_with_measurement"),
    ProbeTarget(f"{_MODULE}.expose_latent_mean", _MODULE, "expose_latent_mean"),
    ProbeTarget(f"{_MODULE}.expose_covariance", _MODULE, "expose_covariance"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in KALMAN_FILTERS_STATIC_KF_PROBE_TARGETS
    ]
