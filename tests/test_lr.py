import numpy
import unittest

from sklearn.linear_model import LogisticRegression

from liar import *


class TestLR(unittest.TestCase):
    def test_probability_fraction(self):
        points_h0 = [ 1, 2, 4, 8 ]
        points_h1 = [ 2, 6, 8, 9 ]

        for point, expected_p in zip(range(11), [ 1., 1., .8, .6, .6, .4, .4, .4, .4, .2, .2 ]):
            self.assertAlmostEqual(expected_p, probability_fraction(point, points_h0, class_value=0, value_range=[0,10]))

        for point, expected_p in zip(range(11), [ .2, .2, .4, .4, .4, .4, .6, .6, .8, 1, 1 ]):
            self.assertAlmostEqual(expected_p, probability_fraction(point, points_h1, class_value=10, value_range=[0,10]))

    def test_calibrate_lr(self):
        points_h0 = [ 1, 2, 4, 8 ]
        points_h1 = [ 2, 6, 8, 9 ]
        for point, expected_lr in zip(range(11), [ .2, .2, .5, 2/3., 2/3., 1, 1.5, 1.5, 2, 5, 5 ]):
            self.assertAlmostEqual(expected_lr, calibrate_lr(point, points_h0, points_h1, probability_fraction, value_range=[0,10]))

    def test_calculate_cllr(self):
        self.assertAlmostEqual(1, calculate_cllr([1, 1], [1, 1]))
        self.assertAlmostEqual(2, calculate_cllr([3.]*2, [1/3.]*2))
        self.assertAlmostEqual(2, calculate_cllr([3.]*20, [1/3.]*20))
        self.assertAlmostEqual(0.4150374992788437, calculate_cllr([1/3.]*2, [3.]*2))
        self.assertAlmostEqual(0.7075187496394219, calculate_cllr([1/3.]*2, [1]))
        self.assertAlmostEqual(0.507177646488535, calculate_cllr([1/100.]*100, [1]))
        self.assertAlmostEqual(0.5400680236656377, calculate_cllr([1/100.]*100 + [100], [1]))
        self.assertAlmostEqual(0.5723134914863265, calculate_cllr([1/100.]*100 + [100]*2, [1]))
        self.assertAlmostEqual(0.6952113122368764, calculate_cllr([1/100.]*100 + [100]*6, [1]))
        self.assertAlmostEqual(1.0000000000000000, calculate_cllr([1], [1]))
        self.assertAlmostEqual(1.0849625007211563, calculate_cllr([2], [2]*2))
        self.assertAlmostEqual(1.6699250014423126, calculate_cllr([8], [8]*8))

    def test_classifier_cllr(self):
        numpy.random.seed(0)
        clf = LogisticRegression()

        prev_cllr = 1
        for i in range(1, 10):
            X0 = numpy.random.normal(loc=[-1]*3, scale=.1, size=(i, 3))
            X1 = numpy.random.normal(loc=[1]*3, scale=.1, size=(i, 3))
            cllr = classifier_cllr(clf, X0, X1, X0, X1)
            self.assertLess(cllr, prev_cllr)
            prev_cllr = cllr

        X0 = numpy.random.normal(loc=[-1]*3, size=(100, 3))
        X1 = numpy.random.normal(loc=[1]*3, size=(100, 3))
        self.assertAlmostEqual(0.1901544891867276, classifier_cllr(clf, X0, X1, X0, X1))

        X0 = numpy.random.normal(loc=[-.5]*3, size=(100, 3))
        X1 = numpy.random.normal(loc=[.5]*3, size=(100, 3))
        self.assertAlmostEqual(0.6153060581423102, classifier_cllr(clf, X0, X1, X0, X1))

        X0 = numpy.random.normal(loc=[0]*3, size=(100, 3))
        X1 = numpy.random.normal(loc=[0]*3, size=(100, 3))
        self.assertAlmostEqual(1.285423922204846, classifier_cllr(clf, X0, X1, X0, X1))

        X = numpy.random.normal(loc=[0]*3, size=(400, 3))
        self.assertAlmostEqual(1.3683601658310476, classifier_cllr(clf, X[:100], X[100:200], X[200:300], X[300:400]))


if __name__ == '__main__':
    unittest.main()
