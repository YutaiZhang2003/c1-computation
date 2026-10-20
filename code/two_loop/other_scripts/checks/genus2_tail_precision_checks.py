"""Regression checks for the two tail failures found in the 2026-08-31 audit."""
import math
import numpy as np
from genus2_plumbing_atlas import format_chart_coordinate
from genus2_hybrid_period_map import HybridPeriodMapConfig, refine_schottky_inverse
from liouville_genus2 import format_complex


def run_checks():
    # This logarithm occurs at the originally failed finite-band node.
    log_q = complex(-22864904.1761933714, .207908366244225429)
    assert complex(format_chart_coordinate(log_q)) == log_q
    old_period_error = abs(complex(format_complex(log_q))-log_q)/(2*math.pi)
    assert old_period_error > 1e-9, 'fixture no longer exercises precision loss'

    # A valid, small separating period used to terminate at a 3e-10
    # residual despite the required 1e-10 tolerance. Exercise the actual
    # inverse and its independent forward certificate, not a mock solver.
    rho = 1e-5
    epsilon = rho * np.exp(.7247126202867059j)
    target = np.array([[.19620664976537228+3.5252517069894465j, epsilon],
                       [epsilon, -.2579593788832426+2.5037790781201457j]])
    logs = (2j*math.pi*target[0,0], 2j*math.pi*target[1,1],
            complex(math.log(2*math.pi*rho), .7247126202867059-math.pi/2))
    q = tuple(np.exp(z) for z in logs)
    result = refine_schottky_inverse('glasses',target,q,initial_log_q=logs,
        config=HybridPeriodMapConfig(tolerance=1e-10,agreement_tolerance=1e-10),
        fit_word_length=7)
    assert result.success and result.residual <= 1e-10
    assert result.stability <= 1e-10
    # The fitting objective must use the same multiprecision regime as
    # its forward certificate. Binary64 collocation can also overflow at
    # this seam; the independently gated Schottky branch remains usable.
    rho_deep=1e-7
    deep_target=target.copy()
    deep_target[0,1]=deep_target[1,0]=rho_deep*np.exp(.7247126202867059j)
    deep_logs=logs[:2]+(complex(math.log(2*math.pi*rho_deep),logs[2].imag),)
    deep=refine_schottky_inverse('glasses',deep_target,tuple(np.exp(z) for z in deep_logs),
        initial_log_q=deep_logs,
        config=HybridPeriodMapConfig(tolerance=1e-12,agreement_tolerance=1e-12),fit_word_length=7)
    assert deep.success and deep.residual<=1e-12 and deep.stability<=1e-12
    print('Tail precision regressions passed:',
          {'old_serialization_period_error':old_period_error,
           'deep_period_residual':result.residual,'deep_period_stability':result.stability,
           'deeper_period_residual':deep.residual})


if __name__=='__main__':
    run_checks()
