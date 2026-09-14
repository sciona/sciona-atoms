import itertools
import numpy as np
import pytest
from sciona.atoms.causal_inference.estimators.probabilities import causal_direction_probability_score


def test_label_permutation_invariance_and_independence_score():
    labels=np.array([-1,0,1])
    p=np.array([[.1,.2,.7],[0.,1.,0.],[1.,0.,0.]])
    before=p.copy()
    for order in itertools.permutations(range(3)):
        order=list(order)
        np.testing.assert_allclose(causal_direction_probability_score(p[:,order],labels[order]),[.6,0.,-1.])
    np.testing.assert_array_equal(p,before)
    np.testing.assert_allclose(causal_direction_probability_score(p, -labels),[-.6,0.,1.])


@pytest.mark.parametrize('p,labels',[
    ([[.2,.2,.2]],[-1,0,1]),
    ([[1.1,0.,-.1]],[-1,0,1]),
    ([[np.nan,0.,1.]],[-1,0,1]),
    ([[np.inf,0.,0.]],[-1,0,1]),
    ([[0.,0.,1.]],[-1,1,1]),
    ([[0.,0.,1.]],[-1.,0.,1.]),
    ([[True,False,False]],[-1,0,1]),
    ([[0.,1.]],[-1,1]),
    (np.empty((0,3)),[-1,0,1]),
])
def test_invalid_distribution_or_labels_rejected(p,labels):
    with pytest.raises(ValueError):causal_direction_probability_score(p,labels)


def test_binary_probability_uses_label_one_without_signed_rescaling():
    from sciona.atoms.causal_inference.estimators.probabilities import binary_positive_class_probability
    p=np.array([[.8,.2],[.1,.9]])
    np.testing.assert_array_equal(binary_positive_class_probability(p,np.array([1,-1])),[.8,.1])
    np.testing.assert_array_equal(binary_positive_class_probability(p[:,::-1],np.array([0,1])),[.8,.1])
    with pytest.raises(ValueError):binary_positive_class_probability(p,np.array([0,2]))
    with pytest.raises(ValueError):binary_positive_class_probability(p*2,np.array([0,1]))
