"""Probe-side catalog for the SciPy stats atom family."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.scipy.stats"

STATS_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.describe", _MODULE, "describe"),
    ProbeTarget(f"{_MODULE}.ttest_ind", _MODULE, "ttest_ind"),
    ProbeTarget(f"{_MODULE}.pearsonr", _MODULE, "pearsonr"),
    ProbeTarget(f"{_MODULE}.spearmanr", _MODULE, "spearmanr"),
    ProbeTarget(f"{_MODULE}.norm", _MODULE, "norm"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in STATS_PROBE_TARGETS
    ]
