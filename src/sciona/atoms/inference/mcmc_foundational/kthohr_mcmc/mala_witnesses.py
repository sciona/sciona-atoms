from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_mala_proposal_adjustment(
    state: AbstractArray,
    gradient: AbstractArray,
    step_size: AbstractArray,
) -> AbstractArray:
    """Return proposal-state metadata for one MALA adjustment step."""
    _ = gradient, step_size
    return AbstractArray(shape=(1,), dtype="float64")
