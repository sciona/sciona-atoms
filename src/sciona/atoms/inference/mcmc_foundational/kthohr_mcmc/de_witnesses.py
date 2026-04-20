from __future__ import annotations

from sciona.ghost.abstract import AbstractScalar, AbstractSignal


def witness_build_de_transition_kernel(
    target_log_kernel: AbstractSignal,
    gamma_scale: AbstractScalar,
) -> AbstractSignal:
    """Return callable metadata for a differential-evolution transition kernel."""
    _ = target_log_kernel, gamma_scale
    return AbstractSignal(shape=(1,), dtype="callable", sampling_rate=1.0, domain="index")
