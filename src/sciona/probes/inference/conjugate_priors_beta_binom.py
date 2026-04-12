"""Probe-side catalog for the conjugate-priors beta-binomial slice."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.inference.conjugate_priors.beta_binom"

CONJUGATE_PRIORS_BETA_BINOM_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.posterior_randmodel", _MODULE, "posterior_randmodel"),
    ProbeTarget(
        f"{_MODULE}.posterior_randmodel_weighted",
        _MODULE,
        "posterior_randmodel_weighted",
    ),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in CONJUGATE_PRIORS_BETA_BINOM_PROBE_TARGETS
    ]
