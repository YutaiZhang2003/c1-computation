#!/usr/bin/env python3
"""Fit/covariance regressions and frozen-data normalization round trip."""
import unittest

import numpy as np

from genus2_integrand_normalization import integration_kernel_scale_to_current
from plot_genus2_absolute_bry import (
    DEFAULT_SOURCE, V6, design_matrix, fit_radius_ansatz, load_frozen_assembly,
)


class AbsoluteBryPlotChecks(unittest.TestCase):
    def setUp(self):
        self.r = np.geomspace(.5, 15, 71)
        self.x = design_matrix(self.r)
        self.beta = np.array([.048, .069])
        self.beta_cov = np.array([[2e-8, -3e-8], [-3e-8, 9e-7]])
        self.y = self.x @ self.beta
        self.c = self.x @ self.beta_cov @ self.x.T

    def test_exact_constrained_ansatz(self):
        fit = fit_radius_ansatz(self.r, self.y, self.c)
        np.testing.assert_allclose(fit['coefficients'], self.beta, rtol=1e-13)
        np.testing.assert_allclose(fit['prediction'], self.y, rtol=1e-13)

    def test_full_correlations_with_singular_covariance(self):
        fit = fit_radius_ansatz(self.r, self.y, self.c)
        self.assertEqual(fit['empirical_covariance_rank'], 2)
        np.testing.assert_allclose(fit['covariance'], self.beta_cov, rtol=1e-12)
        naive = fit_radius_ansatz(self.r, self.y, np.diag(np.diag(self.c)))
        self.assertGreater(fit['standard_error'][0], 2*naive['standard_error'][0])

    def test_half_scales_fit_and_errors_not_shape(self):
        before = fit_radius_ansatz(self.r, self.y, self.c)
        after = fit_radius_ansatz(self.r, self.y/2, self.c/4)
        np.testing.assert_allclose(after['coefficients'], before['coefficients']/2)
        np.testing.assert_allclose(after['covariance'], before['covariance']/4)
        np.testing.assert_allclose(after['standard_error'], before['standard_error']/2)

    def test_invalid_input_rejected(self):
        for bad in ([0.,1.,2.], [1.,np.nan,2.], [-1.,1.,2.]):
            with self.assertRaises(ValueError):
                design_matrix(bad)
        with self.assertRaises(ValueError):
            fit_radius_ansatz([1.,1.,1.], [1.,1.,1.], np.eye(3))
        bad = self.c.copy(); bad[0,1] += 1
        with self.assertRaises(ValueError):
            fit_radius_ansatz(self.r,self.y,bad)

    def test_frozen_source_mean_covariance_and_unknowns_preserved(self):
        data = load_frozen_assembly(DEFAULT_SOURCE)
        self.assertEqual(integration_kernel_scale_to_current(V6), .5)
        r,y = data['radii'],data['values']
        i = int(np.argmin(abs(r-1)))
        self.assertAlmostEqual(y[i], .1643989288677627, places=14)
        self.assertFalse(data['source_summary']['full_free_energy_available'])
        incomplete = [b['band'] for b in data['source_summary']['band_status'] if not b['complete']]
        self.assertEqual(incomplete, [2])
        self.assertNotIn('residual_band_2',data['groups'])
        self.assertNotIn('old_deep_residual',data['groups'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
