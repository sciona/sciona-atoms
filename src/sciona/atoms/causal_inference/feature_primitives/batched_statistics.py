"""Serial evaluation of one typed empirical statistic per observation pair.

This adapter does not select a feature set, order columns, fit classifiers or
infer types. Each invocation computes exactly one explicitly named statistic.
"""
import numpy as np
from scipy.stats import kurtosis,skew,pearsonr
from sklearn.metrics import adjusted_mutual_info_score
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom
from sciona.atoms.numerical.statistics.empirical_features import empirical_entropy,mixed_power_moment
from . import atoms as f
from ..conditional_statistics import atoms as c
from ..conditional_statistics.source_entropy import discretized_conditional_entropy_variance

PAIR_FUNCTIONS={
    'conditional_entropy':discretized_conditional_entropy_variance,
    'conditional_kurtosis':c.conditional_noise_kurtosis_variance,
    'conditional_similarity':c.conditional_distribution_similarity,
    'conditional_skewness':c.conditional_noise_skewness_variance,
    'igci':f.igci_asymmetry_score,'error_probability':f.normalized_error_probability,
    'polyfit_error':f.polyfit_residual_error,'polyfit':f.polyfit_nonlinearity_asymmetry,
}
STATISTICS=frozenset(PAIR_FUNCTIONS)|{
    'adjusted_mi','discrete_conditional_entropy','discrete_entropy','discrete_mi',
    'hsic','kurtosis','log_sample_count','log_unique','moment21','moment31',
    'normalized_discrete_entropy','mi_over_joint_entropy','mi_over_min_entropy',
    'normalized_entropy','correlation','skewness','uniform_divergence',
}


def _scalar(x,tx,y,ty,statistic):
    if statistic in PAIR_FUNCTIONS:return PAIR_FUNCTIONS[statistic](x,tx,y,ty)
    if statistic=='log_sample_count':return np.log(len(x))
    if statistic=='log_unique':return np.log(len(set(x)))
    if statistic=='normalized_entropy':return f.knn_entropy_estimator(x,tx)
    if statistic=='uniform_divergence':return f.uniform_divergence(x,tx)
    if statistic in {'kurtosis','skewness','moment21','moment31'}:
        nx=f._normalize(x,tx)
        if statistic=='kurtosis':return kurtosis(nx)
        if statistic=='skewness':return skew(nx)
        return mixed_power_moment(nx,f._normalize(y,ty),2 if statistic=='moment21' else 3,1)
    if statistic in {'correlation','hsic'}:
        if tx==f.CATEGORICAL and ty==f.CATEGORICAL:
            return 1-min(f.normalized_error_probability(x,tx,y,ty),f.normalized_error_probability(y,ty,x,tx))
        a=f._to_numerical(x,y) if tx==f.CATEGORICAL and ty==f.NUMERICAL else x
        b=f._to_numerical(y,x) if ty==f.CATEGORICAL and tx==f.NUMERICAL else y
        a=(a-a.mean())/a.std();b=(b-b.mean())/b.std()
        return pearsonr(a,b).statistic if statistic=='correlation' else f.hsic_independence_test(a,b)
    xd=f.discretize_and_bin(x,tx)
    if statistic=='discrete_entropy':return empirical_entropy(xd,.7)
    if statistic=='normalized_discrete_entropy':
        return empirical_entropy(xd,.7)/np.log(len(f._discretized_values(x,tx,3,3)))
    yd=f.discretize_and_bin(y,ty)
    if statistic=='adjusted_mi':return adjusted_mutual_info_score(xd,yd,average_method='arithmetic')
    ex=empirical_entropy(xd,.7);ey=empirical_entropy(yd,.7)
    joint=empirical_entropy(np.column_stack((xd,yd)),.7)
    if statistic=='discrete_conditional_entropy':return joint-ey
    mutual=max(ex+ey-joint,0.)
    if statistic=='discrete_mi':return mutual
    if statistic=='mi_over_joint_entropy':return mutual/joint
    if statistic=='mi_over_min_entropy':return mutual/min(ex,ey)
    raise ValueError('Unknown statistic')


def witness_typed_pair_statistic(xs:list,x_types:list,ys:list,y_types:list,statistic:str) -> AbstractArray:
    if statistic not in STATISTICS:raise ValueError('Unknown statistic')
    if not xs or len({len(xs),len(x_types),len(ys),len(y_types)})!=1:
        raise ValueError('Aligned nonempty observation/type lists required')
    return AbstractArray(shape=(len(xs),),dtype='float64')


@register_atom(witness_typed_pair_statistic)
def typed_pair_statistic(xs:list,x_types:list,ys:list,y_types:list,statistic:str) -> np.ndarray:
    """Compute one named statistic serially for each aligned typed pair.

    Every variable is a finite, nonconstant real vector with at least three
    samples; Binary has exactly two values. Pair lengths match, but may vary
    across rows. Explicit Numerical/Categorical/Binary types retain the reviewed
    scalar algorithms' conventions, including frequency-rank normalization,
    categorical dependence dispatch and 0.7-corrected binned entropy. These
    conventions are part of this version's contract, not inferred from data.
    No labels or population-wide statistics are used. Undefined/nonfinite
    results are rejected. A returned statistic is not proof of causality.
    """
    witness_typed_pair_statistic(xs,x_types,ys,y_types,statistic)
    result=[]
    for x,tx,y,ty in zip(xs,x_types,ys,y_types):
        arrays=[]
        for values,kind in ((x,tx),(y,ty)):
            a=np.asarray(values)
            if kind not in {'Numerical','Categorical','Binary'}:raise ValueError('Recognized variable type required')
            if a.dtype.kind not in 'iuf' or a.ndim!=1 or len(a)<3 or not np.isfinite(a).all() or len(set(a))<2:
                raise ValueError('Nonconstant finite real vectors required')
            if kind=='Binary' and len(set(a))!=2:raise ValueError('Binary requires exactly two values')
            arrays.append(a.astype(float,copy=True))
        if arrays[0].shape!=arrays[1].shape:raise ValueError('Paired lengths must match')
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            value=float(_scalar(arrays[0],tx,arrays[1],ty,statistic))
        if not np.isfinite(value):raise ValueError('Undefined statistic for this pair')
        result.append(value)
    return np.asarray(result,dtype=np.float64)
