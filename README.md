# sciona-atoms

Namespace-first pilot repo for `sciona.atoms.*` and `sciona.probes.*` provider packages.

This initial cut hosts one signal-processing family slice:
- atom package: `sciona.atoms.signal_processing.biosppy.ecg`
- probe package: `sciona.probes.signal_processing.biosppy_ecg`
- family heuristic asset: `data/heuristics/families/signal_event_rate.json`

The repo is intentionally PEP 420-style at the shared namespace roots:
- no `__init__.py` at `sciona/`
- no `__init__.py` at `sciona/atoms/`
- no `__init__.py` at `sciona/probes/`
