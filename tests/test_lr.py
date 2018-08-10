import numpy as np
import unittest

from sklearn.linear_model import LogisticRegression

from liar.calibration import FractionCalibrator
from liar.lr import calculate_cllr
from liar.lr import scorebased_cllr
from liar.util import Xy_to_Xn, Xn_to_Xy


class TestLR(unittest.TestCase):
    def test_fraction_calibrator(self):
        points_h0 = np.array([ 1, 2, 4, 8 ])
        points_h1 = np.array([ 2, 6, 8, 9 ])
        p0 = np.array([ 1., 1., .8, .6, .6, .4, .4, .4, .4, .2, .2 ])
        p1 = np.array([ .2, .2, .4, .4, .4, .4, .6, .6, .8, 1, 1 ])

        cal = FractionCalibrator(value_range=[0,10])
        cal.fit(*Xn_to_Xy(points_h0, points_h1))

        lr = cal.transform(np.arange(11))
        np.testing.assert_almost_equal(cal.p0, p0)
        np.testing.assert_almost_equal(cal.p1, p1)
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
        clf = LogisticRegression()
        cal = FractionCalibrator()

        prev_cllr = 1
        for i in range(1, 10):
            X0 = np.random.normal(loc=[-1]*3, scale=.1, size=(i, 3))
            X1 = np.random.normal(loc=[1]*3, scale=.1, size=(i, 3))
            cllr = scorebased_cllr(clf, cal, X0, X1, X0, X1).cllr
            self.assertLess(cllr, prev_cllr)
            prev_cllr = cllr

        X0 = np.random.normal(loc=[-1]*3, size=(100, 3))
        X1 = np.random.normal(loc=[1]*3, size=(100, 3))
        self.assertAlmostEqual(0.16621035922640143, scorebased_cllr(clf, cal, X0, X1, X0, X1).cllr)

        X0 = np.random.normal(loc=[-.5]*3, size=(100, 3))
        X1 = np.random.normal(loc=[.5]*3, size=(100, 3))
        self.assertAlmostEqual(0.5777326125584675, scorebased_cllr(clf, cal, X0, X1, X0, X1).cllr)

        X0 = np.random.normal(loc=[0]*3, size=(100, 3))
        X1 = np.random.normal(loc=[0]*3, size=(100, 3))
        self.assertAlmostEqual(1.2408757158136576, scorebased_cllr(clf, cal, X0, X1, X0, X1).cllr)

        X = np.random.normal(loc=[0]*3, size=(400, 3))
        self.assertAlmostEqual(1.3233586487690963, scorebased_cllr(clf, cal, X[:100], X[100:200], X[200:300], X[300:400]).cllr)


if __name__ == '__main__':
    unittest.main()
