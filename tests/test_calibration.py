import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmark'))
from calibration import weights, objective, fit_repaired_calibrators


class CalibrationTests(unittest.TestCase):
    def test_censoring_weights(self):
        y, w = weights(np.array([100., 200., 1500.]), np.array([0, 1, 0]))
        np.testing.assert_array_equal(y, [0, 1, 0])
        np.testing.assert_allclose(w, [0, 1.5, 1.5])

    def test_objective_gradient(self):
        theta = np.array([.1, .8])
        x, y, w = np.array([-2., -1., .2]), np.array([0., 1., 1.]), np.array([1., 1.4, 1.1])
        _, analytic = objective(theta, x, y, w)
        step = np.eye(2) * 1e-6
        numeric = [(objective(theta + d, x, y, w)[0] - objective(theta - d, x, y, w)[0]) / 2e-6 for d in step]
        np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-8)

    def test_zero_event_update(self):
        fits = fit_repaired_calibrators(np.array([.1, .2, .3]), np.array([1500., 1600., 1700.]), np.zeros(3))
        self.assertTrue(np.isneginf(fits['intercept_ML'][0]))
        self.assertTrue(np.isfinite(fits['shrink_intercept'][0]))
        self.assertTrue(fits['intercept_slope_ML'][2])


if __name__ == '__main__':
    unittest.main()
