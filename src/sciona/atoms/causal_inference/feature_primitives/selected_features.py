"""Ordered reconstruction of the source ensemble's 43 selected features.

Source: https://github.com/jarfo/cause-effect/blob/master/estimator.py
Feature definitions: https://github.com/jarfo/cause-effect/blob/master/features.py
Uses the existing ingested numerical atoms; source parity remains a review gate.
"""
from collections import Counter
import numpy as np
from scipy.stats import skew,kurtosis,pearsonr
from sklearn.metrics import adjusted_mutual_info_score
from sciona.ghost.registry import register_atom
from sciona.ghost.abstract import AbstractArray
from . import atoms as f
from ..conditional_statistics import atoms as c
from ..conditional_statistics.source_entropy import discretized_conditional_entropy_variance

FEATURE_ORDER=(
    'adjusted_mi','conditional_entropy_ab','conditional_entropy_ba',
    'conditional_kurtosis_ab','conditional_kurtosis_ba','conditional_similarity_ab','conditional_similarity_ba',
    'conditional_skewness_ab','conditional_skewness_ba','discrete_conditional_entropy_ab','discrete_conditional_entropy_ba',
    'discrete_entropy_a','discrete_entropy_b','discrete_mi','hsic','igci_ab','igci_ba','kurtosis_a','kurtosis_b',
    'log_sample_count','log_unique_a','log_unique_b','moment21_ab','moment21_ba','moment31_ab','moment31_ba',
    'normalized_discrete_entropy_a','normalized_discrete_entropy_b','mi_over_joint_entropy','mi_over_min_entropy',
    'normalized_entropy_a','normalized_entropy_b','error_probability_ab','error_probability_ba','correlation',
    'polyfit_error_ab','polyfit_error_ba','polyfit_ab','polyfit_ba','skewness_a','skewness_b','uniform_divergence_a','uniform_divergence_b',
)


def _count_entropy(values):
    counts=np.asarray(list(Counter(values).values()),dtype=float)
    p=counts/counts.sum()
    return float(-np.dot(p,np.log(p))+.7*(len(p)-1)/(2*counts.sum()))


def _ordered_features(x,tx,y,ty):
    xd=f.discretize_and_bin(x,tx);yd=f.discretize_and_bin(y,ty)
    ex=_count_entropy(xd);ey=_count_entropy(yd);joint=_count_entropy(list(zip(xd,yd)))
    mutual=max(ex+ey-joint,0.)
    nx=f._normalize(x,tx);ny=f._normalize(y,ty)
    values={'adjusted_mi':float(adjusted_mutual_info_score(xd,yd,average_method='arithmetic')),
        'discrete_conditional_entropy_ab':joint-ey,'discrete_conditional_entropy_ba':joint-ex,
        'discrete_entropy_a':ex,'discrete_entropy_b':ey,'discrete_mi':mutual,
        'log_sample_count':np.log(len(x)),'log_unique_a':np.log(len(set(x))),'log_unique_b':np.log(len(set(y))),
        'moment21_ab':np.mean(nx**2*ny),'moment21_ba':np.mean(ny**2*nx),
        'moment31_ab':np.mean(nx**3*ny),'moment31_ba':np.mean(ny**3*nx),
        'mi_over_joint_entropy':mutual/joint,'mi_over_min_entropy':mutual/min(ex,ey),
        'normalized_discrete_entropy_a':ex/np.log(len(f._discretized_values(x,tx,3,3))),
        'normalized_discrete_entropy_b':ey/np.log(len(f._discretized_values(y,ty,3,3))),
        'kurtosis_a':kurtosis(nx),'kurtosis_b':kurtosis(ny),'skewness_a':skew(nx),'skewness_b':skew(ny),
        'normalized_entropy_a':f.knn_entropy_estimator(x,tx),'normalized_entropy_b':f.knn_entropy_estimator(y,ty),
        'uniform_divergence_a':f.uniform_divergence(x,tx),'uniform_divergence_b':f.uniform_divergence(y,ty)}
    for name,fn in [('conditional_entropy',discretized_conditional_entropy_variance),('conditional_kurtosis',c.conditional_noise_kurtosis_variance),('conditional_similarity',c.conditional_distribution_similarity),('conditional_skewness',c.conditional_noise_skewness_variance),('igci',f.igci_asymmetry_score),('error_probability',f.normalized_error_probability),('polyfit_error',f.polyfit_residual_error),('polyfit',f.polyfit_nonlinearity_asymmetry)]:
        values[name+'_ab']=fn(x,tx,y,ty)
        values[name+'_ba']=fn(y,ty,x,tx)
    if tx==f.CATEGORICAL and ty==f.CATEGORICAL:
        correlation=1-min(values['error_probability_ab'],values['error_probability_ba'])
        hsic=correlation
    else:
        a=f._to_numerical(x,y) if tx==f.CATEGORICAL and ty==f.NUMERICAL else x
        b=f._to_numerical(y,x) if ty==f.CATEGORICAL and tx==f.NUMERICAL else y
        a=(a-a.mean())/a.std();b=(b-b.mean())/b.std()
        correlation=pearsonr(a,b).statistic;hsic=f.hsic_independence_test(a,b)
    values.update(correlation=correlation,hsic=hsic)
    result=np.asarray([values[name] for name in FEATURE_ORDER],dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError('source feature definition is nonfinite for this pair')
    return result


def witness_extract_causal_selected_features(pairs: list,variable_types: list) -> AbstractArray:
    if len(pairs)!=len(variable_types):raise ValueError('aligned pair type metadata required')
    return AbstractArray(shape=(2*len(pairs),len(FEATURE_ORDER)),dtype='float64')


@register_atom(witness_extract_causal_selected_features)
def extract_causal_selected_features(pairs: list,variable_types: list) -> np.ndarray:
    """Extract 43 ordered features in adjacent forward/reverse rows per pair.

    Pairs contain two equally sized finite real 1D arrays, at least three samples,
    and at least two distinct values per variable. Type pairs explicitly use
    Numerical, Categorical or Binary. Binary variables must have two values.
    Constant or undefined feature regimes are rejected rather than imputed.
    Variable-length observation arrays across pairs are supported. Does not
    train a model, infer variable types, or mutate observations. This version
    uses modern arithmetic-normalized adjusted MI; historical parity needs review.
    """
    if not pairs or len(pairs)!=len(variable_types):raise ValueError('nonempty aligned pairs and variable types required')
    rows=[]
    for pair,types in zip(pairs,variable_types):
        if len(pair)!=2 or len(types)!=2 or any(t not in {f.NUMERICAL,f.CATEGORICAL,f.BINARY} for t in types):
            raise ValueError('two variables with explicit recognized types required')
        arrays=[]
        for value,tp in zip(pair,types):
            a=np.asarray(value)
            if a.dtype.kind not in 'iuf' or a.ndim!=1 or len(a)<3 or not np.all(np.isfinite(a)) or len(set(a))<2:
                raise ValueError('nonconstant finite real sample vectors required')
            if tp==f.BINARY and len(set(a))!=2:raise ValueError('binary variables require exactly two values')
            arrays.append(a.astype(float,copy=True))
        x,y=arrays;tx,ty=types
        if x.shape!=y.shape:raise ValueError('paired sample lengths must match')
        rows.extend([_ordered_features(x,tx,y,ty),_ordered_features(y,ty,x,tx)])
    return np.stack(rows)


def witness_prepare_causal_feature_matrices(training_pairs: list,prediction_pairs: list,training_types: list,prediction_types: list,pair_labels: np.ndarray) -> tuple:
    return (AbstractArray(shape=(2*len(training_pairs),43),dtype='float64'),AbstractArray(shape=(2*len(training_pairs),),dtype='int64'),AbstractArray(shape=(2*len(prediction_pairs),43),dtype='float64'))


@register_atom(witness_prepare_causal_feature_matrices)
def prepare_causal_feature_matrices(training_pairs: list,prediction_pairs: list,training_types: list,prediction_types: list,pair_labels: np.ndarray) -> tuple:
    """Extract aligned training/test features and augment training orientations.

    Each integer label describes its forward training pair. Reverse rows receive
    its negation; prediction pairs receive no labels. Feature extraction uses no
    training-label information or prediction-set statistics.
    """
    labels=np.asarray(pair_labels)
    if labels.dtype.kind not in 'iu' or labels.shape!=(len(training_pairs),) or not set(labels.tolist())<={-1,0,1}:
        raise ValueError('one integer causal label per training pair required')
    augmented=np.column_stack((labels,-labels)).reshape(-1)
    return (extract_causal_selected_features(training_pairs,training_types),augmented,
            extract_causal_selected_features(prediction_pairs,prediction_types))
