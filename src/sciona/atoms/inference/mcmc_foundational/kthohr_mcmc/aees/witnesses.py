from __future__ import annotations

from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal


def witness_metropolishastingstransitionkernel(
    state_in: AbstractArray,
    temper_val: AbstractScalar,
    target_log_kernel: AbstractSignal,
    rng_key_in: AbstractArray,
    prop_scaling_mat: AbstractArray,
) -> tuple[AbstractArray, AbstractArray]:
    """Return metadata for one AEES local Metropolis transition."""
    _ = temper_val, target_log_kernel, prop_scaling_mat
    return (
        AbstractArray(shape=state_in.shape, dtype="float64"),
        AbstractArray(shape=rng_key_in.shape, dtype="int64"),
    )


def witness_targetlogkerneloracle(
    state_candidate: AbstractArray,
    weights: AbstractArray,
    means: AbstractArray,
    variances: AbstractArray,
    temper_val: AbstractScalar,
) -> AbstractScalar:
    """Return scalar metadata for the AEES Gaussian-mixture log kernel."""
    _ = state_candidate, weights, means, variances, temper_val
    return AbstractScalar(dtype="float64")
