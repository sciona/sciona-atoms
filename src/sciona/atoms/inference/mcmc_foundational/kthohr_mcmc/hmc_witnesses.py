from __future__ import annotations

from sciona.ghost.abstract import AbstractScalar, AbstractSignal


def witness_buildhmckernelfromlogdensityoracle(
    target_log_kernel: AbstractSignal,
    step_size: AbstractScalar,
    n_leapfrog: AbstractScalar,
) -> AbstractSignal:
    """Return callable metadata for a minimal HMC transition kernel."""
    _ = target_log_kernel, step_size, n_leapfrog
    return AbstractSignal(shape=(1,), dtype="callable", sampling_rate=1.0, domain="index")
