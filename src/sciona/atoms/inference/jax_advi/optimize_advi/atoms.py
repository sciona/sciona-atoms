from __future__ import annotations
"""Auto-generated atom wrappers following the sciona pattern."""


from collections.abc import Mapping
from typing import Any, Callable

import numpy as np

import icontract
from sciona.ghost.registry import register_atom
from .witnesses import witness_meanfieldvariationalfit, witness_posteriordrawsampling

ThetaShapeDict = dict[str, tuple[int, ...]]
ParameterDict = dict[str, Any]
ConstraintFn = Callable[[object], tuple[object, float]]
ConstraintMap = dict[str, ConstraintFn]
LogDensityFn = Callable[[ParameterDict], object]
VarParamInits = dict[str, object]
ObjectiveFn = Callable[[object], object]
PosteriorTransform = Callable[[ParameterDict], object]

DEFAULT_CONSTRAINTS: ConstraintMap = {}
DEFAULT_VAR_PARAM_INITS: VarParamInits = {
    "mean": 0.0,
    "log_sd": 0.0,
}


def _shape_tuple(shape: object) -> tuple[int, ...]:
    if isinstance(shape, int):
        return (shape,)
    return tuple(int(dim) for dim in shape)  # type: ignore[arg-type]


def _coerce_init(value: object, shape: tuple[int, ...]) -> np.ndarray:
    if isinstance(value, tuple) and len(value) == 2 and not hasattr(value[0], "shape"):
        value = value[0]
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == ():
        return np.full(shape, float(arr), dtype=np.float64)
    return np.broadcast_to(arr, shape).astype(np.float64).copy()


def _initial_arrays(
    theta_shape_dict: ThetaShapeDict,
    var_param_inits: Mapping[str, object],
) -> tuple[ParameterDict, ParameterDict]:
    means: ParameterDict = {}
    log_sds: ParameterDict = {}
    default_mean = var_param_inits.get("mean", 0.0)
    default_log_sd = var_param_inits.get("log_sd", 0.0)
    for name, raw_shape in theta_shape_dict.items():
        shape = _shape_tuple(raw_shape)
        per_param = var_param_inits.get(name)
        if isinstance(per_param, tuple) and len(per_param) == 2:
            mean_init, log_sd_init = per_param
        else:
            mean_init, log_sd_init = default_mean, default_log_sd
        means[name] = _coerce_init(mean_init, shape)
        log_sds[name] = _coerce_init(log_sd_init, shape)
    return means, log_sds


def _flatten(params: Mapping[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray:
    return np.concatenate([np.asarray(params[name], dtype=np.float64).ravel() for name in names])


def _unpack(vector: np.ndarray, theta_shape_dict: ThetaShapeDict, names: tuple[str, ...]) -> ParameterDict:
    out: ParameterDict = {}
    offset = 0
    for name in names:
        shape = _shape_tuple(theta_shape_dict[name])
        size = int(np.prod(shape, dtype=np.int64))
        out[name] = np.asarray(vector[offset : offset + size], dtype=np.float64).reshape(shape)
        offset += size
    return out


def _apply_constraints(
    free_sample: Mapping[str, np.ndarray],
    constrain_fun_dict: Mapping[str, ConstraintFn],
) -> tuple[ParameterDict, float]:
    constrained: ParameterDict = {}
    log_jacobian = 0.0
    for name, value in free_sample.items():
        transform = constrain_fun_dict.get(name)
        if transform is None:
            constrained[name] = np.asarray(value, dtype=np.float64).copy()
            continue
        transformed = transform(value)
        if isinstance(transformed, tuple) and len(transformed) == 2:
            constrained_value, log_det = transformed
            log_jacobian += float(np.sum(log_det))
        else:
            constrained_value = transformed
        constrained[name] = np.asarray(constrained_value, dtype=np.float64)
    return constrained, log_jacobian


def _elbo_estimate(
    packed_state: np.ndarray,
    *,
    theta_shape_dict: ThetaShapeDict,
    names: tuple[str, ...],
    eps_bank: Mapping[str, np.ndarray],
    log_prior_fun: LogDensityFn,
    log_lik_fun: LogDensityFn,
    constrain_fun_dict: Mapping[str, ConstraintFn],
) -> float:
    split = len(packed_state) // 2
    means = _unpack(packed_state[:split], theta_shape_dict, names)
    log_sds = _unpack(np.clip(packed_state[split:], -20.0, 5.0), theta_shape_dict, names)
    sds = {name: np.exp(log_sds[name]) for name in names}

    sample_terms: list[float] = []
    for draw_idx in range(next(iter(eps_bank.values())).shape[0]):
        free_sample = {
            name: means[name] + sds[name] * eps_bank[name][draw_idx]
            for name in names
        }
        constrained, log_jacobian = _apply_constraints(free_sample, constrain_fun_dict)
        log_joint = float(log_prior_fun(constrained)) + float(log_lik_fun(constrained)) + log_jacobian
        log_q = 0.0
        for name in names:
            standardized = (free_sample[name] - means[name]) / sds[name]
            log_q += float(
                np.sum(
                    -0.5 * np.log(2.0 * np.pi)
                    - np.log(sds[name])
                    - 0.5 * standardized**2
                )
            )
        sample_terms.append(log_joint - log_q)
    return float(np.mean(sample_terms))


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
    seed_int = int(seed) if not isinstance(seed, int) else seed
    names = tuple(theta_shape_dict)
    initial_means, initial_log_sds = _initial_arrays(theta_shape_dict, var_param_inits)
    mean_vec = _flatten(initial_means, names)
    log_sd_vec = _flatten(initial_log_sds, names)
    packed_initial = np.concatenate([mean_vec, log_sd_vec])

    rng = np.random.default_rng(seed_int)
    eps_bank = {
        name: rng.normal(size=(int(M), *_shape_tuple(theta_shape_dict[name])))
        for name in names
    }

    def _objective_for_state(state: np.ndarray) -> float:
        return _elbo_estimate(
            state,
            theta_shape_dict=theta_shape_dict,
            names=names,
            eps_bank=eps_bank,
            log_prior_fun=log_prior_fun,
            log_lik_fun=log_lik_fun,
            constrain_fun_dict=constrain_fun_dict,
        )

    from scipy.optimize import minimize as scipy_minimize

    method = opt_method if opt_method in {"L-BFGS-B", "BFGS", "CG", "Nelder-Mead", "Powell"} else "L-BFGS-B"
    result = scipy_minimize(lambda state: -_objective_for_state(state), packed_initial, method=method)
    packed_opt = np.asarray(result.x if result.success else packed_initial, dtype=np.float64)
    split = len(packed_opt) // 2
    free_means = _unpack(packed_opt[:split], theta_shape_dict, names)
    free_sds = {
        name: np.exp(np.clip(value, -20.0, 5.0))
        for name, value in _unpack(packed_opt[split:], theta_shape_dict, names).items()
    }

    def objective_fun(state: object | None = None) -> float:
        if state is None:
            packed_state = packed_opt
        elif isinstance(state, np.ndarray):
            packed_state = np.asarray(state, dtype=np.float64)
        elif isinstance(state, tuple) and len(state) == 2:
            packed_state = np.concatenate([
                _flatten(state[0], names),  # type: ignore[arg-type]
                np.log(_flatten(state[1], names)),  # type: ignore[arg-type]
            ])
        elif isinstance(state, Mapping):
            packed_state = np.concatenate(
                [
                    _flatten(state["free_means"], names),  # type: ignore[arg-type]
                    np.log(_flatten(state["free_sds"], names)),  # type: ignore[arg-type]
                ]
            )
        else:
            raise TypeError("state must be None, a packed ndarray, (free_means, free_sds), or a mapping")
        return _objective_for_state(packed_state)

    if verbose:
        print(f"meanfieldvariationalfit optimized ELBO={objective_fun():.6g} success={result.success}")

    rng_state_out = seed_int + 1
    return (free_means, free_sds, objective_fun, rng_state_out)


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
) -> tuple[object, int]:
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
    rng_state_int = int(rng_state_in) if not isinstance(rng_state_in, int) else rng_state_in
    rng = np.random.default_rng(rng_state_int)
    names = tuple(free_means)
    free_draws: dict[str, np.ndarray] = {}
    for name in names:
        mean = np.asarray(free_means[name], dtype=np.float64)
        sd = np.asarray(free_sds[name], dtype=np.float64)
        if np.any(~np.isfinite(sd)) or np.any(sd <= 0.0):
            raise ValueError("free_sds must be finite and strictly positive")
        free_draws[name] = rng.normal(loc=mean, scale=sd, size=(int(n_draws), *mean.shape))

    if fun_to_apply is None:
        constrained_draws: dict[str, list[np.ndarray]] = {name: [] for name in names}
        for draw_idx in range(int(n_draws)):
            sample = {name: free_draws[name][draw_idx] for name in names}
            constrained, _ = _apply_constraints(sample, constrain_fun_dict)
            for name in names:
                constrained_draws[name].append(np.asarray(constrained[name], dtype=np.float64))
        draws = {name: np.stack(values, axis=0) for name, values in constrained_draws.items()}
    else:
        transformed_draws = []
        for draw_idx in range(int(n_draws)):
            sample = {name: free_draws[name][draw_idx] for name in names}
            constrained, _ = _apply_constraints(sample, constrain_fun_dict)
            transformed_draws.append(fun_to_apply(constrained))
        try:
            draws = np.asarray(transformed_draws)
        except ValueError:
            draws = transformed_draws
    return (draws, rng_state_int + 1)
