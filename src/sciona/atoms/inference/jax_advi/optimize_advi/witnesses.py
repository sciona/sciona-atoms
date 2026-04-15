from __future__ import annotations
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_meanfieldvariationalfit(theta_shape_dict: AbstractArray, log_prior_fun: AbstractArray, log_lik_fun: AbstractArray, M: AbstractScalar, constrain_fun_dict: AbstractArray, verbose: AbstractScalar, seed: AbstractScalar, n_draws: AbstractScalar, var_param_inits: AbstractArray, opt_method: AbstractScalar) -> tuple[AbstractArray, AbstractArray, AbstractArray, AbstractScalar]:
    """Shape-and-type check for mean field variational fit. Returns output metadata without running the real computation."""
    means = AbstractArray(
        shape=theta_shape_dict.shape,
        dtype="float64",
    )
    sds = AbstractArray(
        shape=theta_shape_dict.shape,
        dtype="float64",
        min_val=0.0,
    )
    objective = AbstractArray(
        shape=(),
        dtype="callable",
    )
    rng_state_out = AbstractScalar(dtype=seed.dtype)
    return means, sds, objective, rng_state_out


def witness_posteriordrawsampling(free_means: AbstractArray, free_sds: AbstractArray, constrain_fun_dict: AbstractArray, n_draws: AbstractScalar, fun_to_apply: AbstractArray, rng_state_in: AbstractScalar) -> tuple[AbstractArray, AbstractScalar]:
    """Shape-and-type check for posterior draw sampling. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=free_means.shape,
        dtype="float64",
    )
    return result

