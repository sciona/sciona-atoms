from __future__ import annotations

import importlib

from sciona.ghost.registry import list_registered


_MODULES_AND_SYMBOLS = {
    "sciona.atoms.expansion.belief_propagation": (
        "monitor_message_convergence",
        "validate_belief_normalization",
        "analyze_message_damping",
        "detect_graph_cycles",
    ),
    "sciona.atoms.expansion.divide_and_conquer": (
        "measure_split_balance",
        "check_recursion_depth",
        "profile_merge_cost",
        "detect_subproblem_overlap",
    ),
    "sciona.atoms.expansion.kalman_filter": (
        "check_innovation_consistency",
        "validate_covariance_pd",
        "analyze_kalman_gain_magnitude",
        "check_state_smoothness",
    ),
    "sciona.atoms.expansion.particle_filter": (
        "monitor_effective_sample_size",
        "analyze_particle_diversity",
        "track_weight_variance",
        "check_resampling_quality",
    ),
    "sciona.atoms.expansion.sequential_filter": (
        "check_observability",
        "validate_innovation_whiteness",
        "detect_filter_divergence",
        "adapt_process_noise",
    ),
}


def test_core_expansion_registration_import_smoke() -> None:
    for module_name, symbols in _MODULES_AND_SYMBOLS.items():
        module = importlib.import_module(module_name)
        for symbol in symbols:
            assert hasattr(module, symbol)

    registered = set(list_registered())
    expected = {symbol for symbols in _MODULES_AND_SYMBOLS.values() for symbol in symbols}
    assert expected <= registered
