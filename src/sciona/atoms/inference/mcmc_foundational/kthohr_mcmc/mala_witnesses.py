from __future__ import annotations

from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal


def witness_mala_proposal_adjustment(
    prop_vals: AbstractArray,
    prev_vals: AbstractArray,
    step_size: AbstractScalar,
    precond_mat: AbstractArray,
    mala_mean_fn: AbstractSignal,
) -> AbstractScalar:
    """Return scalar metadata for the MALA proposal log-ratio adjustment."""
    _ = prop_vals, prev_vals, step_size, precond_mat, mala_mean_fn
    return AbstractScalar(dtype="float64")
