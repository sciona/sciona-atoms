"""Source-compatible conditional entropy using the discretized response."""
from collections import Counter
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.registry import register_atom
from .witnesses import witness_conditional_noise_entropy_variance
from ..feature_primitives.atoms import discretize_and_bin,_discrete_entropy,_discretized_values,CATEGORICAL


@register_atom(witness_conditional_noise_entropy_variance)
def discretized_conditional_entropy_variance(x: NDArray[np.float64],tx: str,
    y: NDArray[np.float64],ty: str,ffactor: int=3,maxdev: int=3,minc: int=10) -> float:
    """Evaluate the source fit_noise_entropy definition on discretized X and Y.

    Include X bins with strictly more than minc samples. Compute bias-corrected
    discrete entropy of discretized Y in each bin and divide their standard
    deviation by log of the source-defined discretized Y alphabet size.
    Returns zero when no X bin qualifies or the Y alphabet has one value.
    Distinct from the older atom that measures entropy of raw Y values.
    Source: jarfo/cause-effect features.py, fit_noise_entropy.
    """
    x=np.asarray(x);y=np.asarray(y)
    if x.ndim!=1 or y.shape!=x.shape or len(x)<2 or x.dtype.kind not in 'iuf' or y.dtype.kind not in 'iuf' or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError('aligned finite real sample vectors required')
    if tx not in {'Numerical','Categorical','Binary'} or ty not in {'Numerical','Categorical','Binary'}:
        raise ValueError('recognized variable types required')
    for value in [ffactor,maxdev,minc]:
        if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<1:raise ValueError('positive integer bin parameters required')
    xd=discretize_and_bin(x.astype(float,copy=True),tx,ffactor,maxdev)
    yd=discretize_and_bin(y.astype(float,copy=True),ty,ffactor,maxdev)
    entropies=[_discrete_entropy(yd[xd==value],CATEGORICAL) for value,count in Counter(xd).items() if count>minc]
    alphabet=len(_discretized_values(yd,ty,ffactor,maxdev))
    if not entropies or alphabet<=1:return 0.
    result=float(np.std(entropies)/np.log(alphabet))
    if not np.isfinite(result):raise ValueError('undefined conditional entropy')
    return result
