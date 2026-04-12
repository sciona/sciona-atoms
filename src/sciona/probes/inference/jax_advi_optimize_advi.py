"""Probe-side catalog for the JAX ADVI optimize-advi slice."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.inference.jax_advi.optimize_advi"

JAX_ADVI_OPTIMIZE_ADVI_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.meanfieldvariationalfit", _MODULE, "meanfieldvariationalfit"),
    ProbeTarget(f"{_MODULE}.posteriordrawsampling", _MODULE, "posteriordrawsampling"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in JAX_ADVI_OPTIMIZE_ADVI_PROBE_TARGETS
    ]
