from __future__ import annotations

from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal


def witness_constructrandomwalkmetropoliskernel(
    target_log_kernel: AbstractSignal,
    proposal_scale: AbstractScalar,
) -> AbstractSignal:
    """Return callable metadata for a random-walk Metropolis transition kernel."""
    _ = target_log_kernel, proposal_scale
    return AbstractSignal(shape=(1,), dtype="callable", sampling_rate=1.0, domain="index")
