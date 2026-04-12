from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


from typing import Any, Callable

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_meanfieldvariationalfit, witness_posteriordrawsampling

ThetaShapeDict = dict[str, tuple[int, ...]]
ParameterDict = dict[str, Any]
ConstraintFn = Callable[[Any], tuple[Any, float]]
ConstraintMap = dict[str, ConstraintFn]
LogDensityFn = Callable[[ParameterDict], Any]
VarParamInits = dict[str, tuple[float, float]]
ObjectiveFn = Callable[[Any], Any]
PosteriorTransform = Callable[[ParameterDict], Any]

DEFAULT_CONSTRAINTS: ConstraintMap = {}
DEFAULT_VAR_PARAM_INITS: VarParamInits = {
    "mean": (0.0, 0.0),
    "log_sd": (0.0, 0.0),
}


@register_atom(witness_meanfieldvariationalfit)
@icontract.require(lambda theta_shape_dict: theta_shape_dict is not None, "theta_shape_dict cannot be None")
@icontract.require(lambda log_prior_fun: log_prior_fun is not None, "log_prior_fun cannot be None")
@icontract.require(lambda log_lik_fun: log_lik_fun is not None, "log_lik_fun cannot be None")
@icontract.require(lambda M: M > 0, "M must be positive")
@icontract.require(lambda n_draws: n_draws is None or n_draws >= 0, "n_draws must be non-negative when provided")
@icontract.ensure(lambda result: all(r is not None for r in result), "MeanFieldVariationalFit all outputs must not be None")
def meanfieldvariationalfit(
    theta_shape_dict: ThetaShapeDict,
    log_prior_fun: LogDensityFn,
    log_lik_fun: LogDensityFn,
    M: int = 100,
    constrain_fun_dict: ConstraintMap = DEFAULT_CONSTRAINTS,
    verbose: bool = False,
    seed: int = 2,
    n_draws: int | None = 1000,
    var_param_inits: VarParamInits = DEFAULT_VAR_PARAM_INITS,
    opt_method: str = "trust-ncg",
) -> tuple[ParameterDict, ParameterDict, ObjectiveFn, int]:
    """Builds a stochastic Evidence Lower Bound (ELBO) objective from prior/likelihood oracles and optimizes mean-field variational parameters as immutable variational state (latent mean and latent scale). Private objective construction helper is grouped with the optimizer entrypoint.

Args:
    theta_shape_dict: Defines latent parameter block shapes.
    log_prior_fun: Pure log-probability oracle; no persistent state writes.
    log_lik_fun: Pure likelihood/log-likelihood oracle; no persistent state writes.
    M: Monte Carlo sample count, M > 0.
    constrain_fun_dict: Maps unconstrained variational coordinates to constrained parameter space.
    verbose: Logging flag only; does not alter statistical semantics.
    seed: Explicit stochastic input; treated as immutable random number generator (RNG) state.
    n_draws: Number of posterior draws requested from the fitted approximation.
    var_param_inits: Optional initial latent mean/scale values.
    opt_method: Optimization algorithm selection.

Returns:
    free_means: Optimized latent mean parameters.
    free_sds: Optimized latent standard deviations; strictly positive.
    objective_fun: Pure ELBO-like objective closure.
    rng_state_out: Advanced RNG state returned as new immutable value."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "third_party", "jax_advi"))
    from jax_advi.advi import optimize_advi_mean_field

    seed_int = int(seed) if not isinstance(seed, int) else seed
    result = optimize_advi_mean_field(
        theta_shape_dict=theta_shape_dict,
        log_prior_fun=log_prior_fun,
        log_lik_fun=log_lik_fun,
        M=M,
        constrain_fun_dict=constrain_fun_dict,
        verbose=verbose,
        seed=seed_int,
        n_draws=n_draws,
        var_param_inits=var_param_inits,
        opt_method=opt_method,
    )
    rng_state_out = seed_int + 1
    return (result["free_means"], result["free_sds"], result["objective_fun"], rng_state_out)


@register_atom(witness_posteriordrawsampling)  # type: ignore[untyped-decorator]
@icontract.require(lambda free_means: free_means is not None, "free_means cannot be None")
@icontract.require(lambda free_sds: free_sds is not None, "free_sds cannot be None")
@icontract.require(lambda constrain_fun_dict: constrain_fun_dict is not None, "constrain_fun_dict cannot be None")
@icontract.require(lambda n_draws: n_draws >= 0, "n_draws must be non-negative")
@icontract.require(lambda rng_state_in: rng_state_in is not None, "rng_state_in cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "PosteriorDrawSampling all outputs must not be None")
def posteriordrawsampling(
    free_means: ParameterDict,
    free_sds: ParameterDict,
    constrain_fun_dict: ConstraintMap,
    n_draws: int,
    fun_to_apply: PosteriorTransform | None,
    rng_state_in: int,
) -> tuple[Any, int]:
    """Samples from the fitted mean-field posterior using latent mean/scale state, applies constraint transforms, and optionally applies a post-processing function.

Args:
    free_means: Latent mean state from variational fit.
    free_sds: Latent scale state from variational fit; positive.
    constrain_fun_dict: Coordinate-wise transforms to constrained space.
    n_draws: Number of posterior draws; n_draws >= 0.
    fun_to_apply: Optional pure transformation over sampled draws.
    rng_state_in: Explicit stochastic input for reproducible sampling.

Returns:
    posterior_draws: Samples in constrained parameter space (or transformed output).
    rng_state_out: Advanced random number generator (RNG) state returned as new immutable value."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "third_party", "jax_advi"))
    from jax_advi.advi import get_posterior_draws

    rng_state_int = int(rng_state_in) if not isinstance(rng_state_in, int) else rng_state_in
    draws = get_posterior_draws(
        free_means=free_means,
        free_sds=free_sds,
        constrain_fun_dict=constrain_fun_dict,
        n_draws=n_draws,
        fun_to_apply=fun_to_apply if fun_to_apply is not None else (lambda x: x),
    )
    return (draws, rng_state_int + 1)

