from __future__ import annotations

from ageoa.ghost.abstract import AbstractArray, AbstractSignal


def witness_build_de_transition_kernel(target_log_kernel: AbstractSignal) -> AbstractArray:
    """Shape witness for a synthesized differential-evolution kernel."""
    return AbstractArray(shape=(1,), dtype="float64")
