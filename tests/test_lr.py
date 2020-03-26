import numpy as np
import unittest

from sklearn.linear_model import LogisticRegression

from lir.calibration import FractionCalibrator, ScalingCalibrator
from lir.lr import calculate_cllr
from lir.lr import scorebased_cllr
from lir.util import Xy_to_Xn, Xn_to_Xy


class TestLR(unittest.TestCase):
    def test_fraction_calibrator(self):
        points_h0 = np.array([ 1, 2, 4, 8 ])
        points_h1 = np.array([ 2, 6, 8, 9 ])
        p0 = np.array([1., 1., .75, .5, .5 , .25, .25, .25, .25, 0., 0.])
        p1 = np.array([0., 0., .25, .25, .25, .25, .5, .5, .75, 1, 1])

        cal = FractionCalibrator(value_range=[0,10])
        cal.fit(*Xn_to_Xy(points_h0, points_h1))

        lr = cal.transform(np.arange(11))
        np.testing.assert_almost_equal(cal.p0, p0)
        np.testing.assert_almost_equal(cal.p1, p1)
        with np.errstate(divide='ignore'):
            np.testing.assert_almost_equal(lr, p1/p0)

    def test_calculate_cllr(self):
        self.assertAlmostEqual(1, calculate_cllr([1, 1], [1, 1]).cllr)
        self.assertAlmostEqual(2, calculate_cllr([3.]*2, [1/3.]*2).cllr)
        self.assertAlmostEqual(2, calculate_cllr([3.]*20, [1/3.]*20).cllr)
        self.assertAlmostEqual(0.4150374992788437, calculate_cllr([1/3.]*2, [3.]*2).cllr)
        self.assertAlmostEqual(0.7075187496394219, calculate_cllr([1/3.]*2, [1]).cllr)
        self.assertAlmostEqual(0.507177646488535, calculate_cllr([1/100.]*100, [1]).cllr)
        self.assertAlmostEqual(0.5400680236656377, calculate_cllr([1/100.]*100 + [100], [1]).cllr)
        self.assertAlmostEqual(0.5723134914863265, calculate_cllr([1/100.]*100 + [100]*2, [1]).cllr)
        self.assertAlmostEqual(0.6952113122368764, calculate_cllr([1/100.]*100 + [100]*6, [1]).cllr)
        self.assertAlmostEqual(1.0000000000000000, calculate_cllr([1], [1]).cllr)
        self.assertAlmostEqual(1.0849625007211563, calculate_cllr([2], [2]*2).cllr)
        self.assertAlmostEqual(1.6699250014423126, calculate_cllr([8], [8]*8).cllr)

    def test_classifier_cllr(self):
        np.random.seed(0)
        clf = LogisticRegression(solver='lbfgs')
        cal = ScalingCalibrator(FractionCalibrator())

        prev_cllr = 1
        for i in range(1, 10):
            X0 = np.random.normal(loc=[-1]*3, scale=.1, size=(i, 3))
            X1 = np.random.normal(loc=[1]*3, scale=.1, size=(i, 3))
            cllr = scorebased_cllr(clf, cal, X0, X1, X0, X1).cllr
            self.assertLess(cllr, prev_cllr)
            prev_cllr = cllr

        cal = FractionCalibrator()

        X0 = np.random.normal(loc=[-1]*3, size=(100, 3))
        X1 = np.random.normal(loc=[1]*3, size=(100, 3))
        self.assertAlmostEqual(.13257776120905165, scorebased_cllr(clf, cal, X0, X1, X0, X1).cllr)

        X0 = np.random.normal(loc=[-.5]*3, size=(100, 3))
        X1 = np.random.normal(loc=[.5]*3, size=(100, 3))
        self.assertAlmostEqual(.6514624971651655, scorebased_cllr(clf, cal, X0, X1, X0, X1).cllr)

        X0 = np.random.normal(loc=[0]*3, size=(100, 3))
        X1 = np.random.normal(loc=[0]*3, size=(100, 3))
        self.assertAlmostEqual(1.3502413785060203, scorebased_cllr(clf, cal, X0, X1, X0, X1).cllr)

        X = np.random.normal(loc=[0]*3, size=(400, 3))
        self.assertAlmostEqual(1.3742926488365286, scorebased_cllr(clf, cal, X[:100], X[100:200], X[200:300], X[300:400]).cllr)


if __name__ == '__main__':
    unittest.main()
