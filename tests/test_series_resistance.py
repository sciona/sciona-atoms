"""Synthetic circuit parameters only."""
import numpy as np
import pytest
from sciona.atoms.electrical.series_resistance import series_resistance,witness_series_resistance
from sciona.ghost.abstract import AbstractArray


def test_broadcasting_current_independence_and_no_mutation():
    a=np.array([[2.],[5.]])
    b=np.array([3.,7.])
    original=a.copy()
    np.testing.assert_array_equal(series_resistance(a,b,1),[[5,9],[8,12]])
    np.testing.assert_array_equal(series_resistance(a,b,-4),series_resistance(a,b,1))
    np.testing.assert_array_equal(a,original)
    witness=witness_series_resistance(AbstractArray(shape=(2,1),dtype='float64'),AbstractArray(shape=(2,),dtype='float64'),AbstractArray(shape=(),dtype='float64'))
    assert witness.shape==(2,2)


@pytest.mark.parametrize('arguments',[(1,2,0),(-1,2,1),(1,np.nan,1),(1,np.inf,1),(True,2,1),(1+1j,2,1),('1',2,1)])
def test_invalid_parameters_rejected(arguments):
    with pytest.raises(ValueError):series_resistance(*arguments)


def test_overflow_and_incompatible_shapes():
    with pytest.raises(FloatingPointError):series_resistance(1.7e308,1.7e308,1)
    with pytest.raises(ValueError):series_resistance(np.ones(2),np.ones(3),1)


def test_community_review_bundle_does_not_claim_human_certification():
    from pathlib import Path
    from sciona.atoms.audit_review_bundles import load_review_bundle_entries,merge_audit_manifest_entries
    entries=load_review_bundle_entries(Path(__file__).resolve().parents[1]/'data/audit_reviews/series_resistance_pending.json')
    merged,skipped=merge_audit_manifest_entries([],entries)
    assert not skipped and len(merged)==1
    assert merged[0]['review_status']=='approved'
    assert merged[0]['semantic_status']=='pass'
    assert merged[0]['publication_tier']==3
    assert merged[0]['review_source']=='automated'
    assert not merged[0]['review_required_actions']
