"""Probe-side catalog for the NumPy random atom family."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.numpy.random"

RANDOM_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.rand", _MODULE, "rand"),
    ProbeTarget(f"{_MODULE}.uniform", _MODULE, "uniform"),
    ProbeTarget(f"{_MODULE}.default_rng", _MODULE, "default_rng"),
    ProbeTarget(
        f"{_MODULE}.continuous_multivariate_sampler",
        _MODULE,
        "continuous_multivariate_sampler",
    ),
    ProbeTarget(f"{_MODULE}.discrete_event_sampler", _MODULE, "discrete_event_sampler"),
    ProbeTarget(f"{_MODULE}.combinatorics_sampler", _MODULE, "combinatorics_sampler"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in RANDOM_PROBE_TARGETS
    ]
