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
    ProbeTarget(f"{_MODULE}.initializekalmanstatemodel", _MODULE, "initializekalmanstatemodel"),
    ProbeTarget(f"{_MODULE}.predictlatentstateandcovariance", _MODULE, "predictlatentstateandcovariance"),
    ProbeTarget(f"{_MODULE}.predictlatentstatesteadystate", _MODULE, "predictlatentstatesteadystate"),
    ProbeTarget(f"{_MODULE}.evaluatemeasurementoracle", _MODULE, "evaluatemeasurementoracle"),
    ProbeTarget(f"{_MODULE}.updateposteriorstateandcovariance", _MODULE, "updateposteriorstateandcovariance"),
    ProbeTarget(f"{_MODULE}.updateposteriorstatesteadystate", _MODULE, "updateposteriorstatesteadystate"),
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
