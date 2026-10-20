"""Saved-node reconstruction, covariance, and independent torus-block checks."""
import math
import numpy as np
from c1_paths import data_root
from assemble_genus2_tail_completion import assemble
from torus_primary_low_order import torus_primary_h
from virasoro_blocks import TorusOnePointVirasoroBlock


def run_checks():
    summary,a=assemble()
    with np.load(data_root('two_loop')/'tail_completion/tail_audit_arrays.npz') as expected:
        assert set(a)==set(expected.files)
        for key in a:
            np.testing.assert_allclose(a[key],expected[key],rtol=3e-12,atol=1e-18,err_msg=key)
    groups=[v for k,v in a.items() if k.startswith('group_')]
    r=a['radii'];mean=np.zeros(len(r));cov=np.zeros((len(r),len(r)))
    for g in groups:
        m=np.array([math.fsum(g[:,j])/len(g) for j in range(len(r))])
        mean+=m
        for curve in g:cov+=np.outer(curve-m,curve-m)/(len(g)*(len(g)-1))
    np.testing.assert_allclose(mean,a['resolved_mean'],rtol=2e-14)
    np.testing.assert_allclose(cov,a['resolved_covariance'],rtol=2e-13,atol=1e-20)
    assert len(groups)==9
    assert not summary['full_free_energy_available'] and not summary['certified']
    assert summary['remaining_residual_region']=='0 < |Omega12| < 0.00001'
    assert summary['deep_finite_band_status']['successful']==16
    assert summary['finite_band_005_01']['all_eight_nodes_B10_N14']
    errors=[]
    for h in [1.,1.001,1.4,3.,12.]:
        for e in [1.,1.001,1.4,5.,20.]:
            reference=TorusOnePointVirasoroBlock(25.,h,e)
            for q in [.00433,.003j,-.002+.001j]:
                for level in range(5):
                    expected=reference.elliptic_block(q,level)
                    errors.append(abs(torus_primary_h(q,h,e,level)-expected)/max(1,abs(expected)))
    assert len(errors)==375 and max(errors)<1e-12
    print('Genus-two tail completion checks passed; 375 independent block cases, nine covariance groups, all 24 new/refined nodes.')


if __name__=='__main__':run_checks()
