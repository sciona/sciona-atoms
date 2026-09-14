import numpy as np
import pytest
from sciona.atoms.causal_inference.estimators.ensemble_ports import three_system_causal_ensemble


def test_ensemble_preserves_inputs_and_branch_order():
    branches=[np.array([.2,-.2]),np.array([.6,-.6]),np.array([-1.,1.])]
    before=[a.copy() for a in branches]
    for index in range(3):
        weights=np.eye(3)[index]
        np.testing.assert_array_equal(three_system_causal_ensemble(*branches,weights),branches[index])
    for a,b in zip(branches,before):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('branch,weights',[
    ([.2,.2],[1.,0.,0.]),
    ([.2,-.2,.1],[1.,0.,0.]),
    ([np.nan,np.nan],[1.,0.,0.]),
    ([2.,-2.],[1.,0.,0.]),
    ([.2,-.2],[.2,.2,.2]),
    ([.2,-.2],[1.1,-.1,0.]),
    ([.2,-.2],[1.,0.]),
])
def test_invalid_pairing_and_weights_rejected(branch,weights):
    with pytest.raises(ValueError):
        three_system_causal_ensemble(np.array(branch),np.array([.3,-.3]),np.array([.4,-.4]),np.array(weights))
