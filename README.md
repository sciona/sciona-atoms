# sciona-atoms

Namespace-first pilot repo for `sciona.atoms.*` and `sciona.probes.*` provider packages.

This initial cut hosts two namespace slices:
- signal-processing atom package: `sciona.atoms.signal_processing.biosppy.ecg`
- signal-processing probe package: `sciona.probes.signal_processing.biosppy_ecg`
- signal-processing family heuristic asset: `data/heuristics/families/signal_event_rate.json`
- state-estimation atom package: `sciona.atoms.state_estimation.kalman_filters.filter_rs`
- state-estimation probe package: `sciona.probes.state_estimation.kalman_filter_rs`
- state-estimation family heuristic asset: `data/heuristics/families/sequential_filter.json`

The repo is intentionally PEP 420-style at the shared namespace roots:
- no `__init__.py` at `sciona/`
- no `__init__.py` at `sciona/atoms/`
- no `__init__.py` at `sciona/probes/`
