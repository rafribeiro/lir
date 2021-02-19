import numpy as np
import unittest

from context import lir

from lir.metrics import devpav

class TestDevPAV(unittest.TestCase):
    def test_devpav_error(self):
        lrs = np.ones(10)
        y = np.concatenate([np.ones(10)])
        with self.assertRaises(ValueError):
            devpav(lrs, y, 10)

    def test_devpav(self):
        # naive system
        lrs = np.ones(10)
        y = np.concatenate([np.ones(5), np.zeros(5)])
        self.assertEqual(devpav(lrs, y, 10), 0)

        # badly calibrated naive system
        lrs = 2*np.ones(10)
        y = np.concatenate([np.ones(5), np.zeros(5)])
        self.assertAlmostEqual(devpav(lrs, y, 10), 0)  # TODO: what should be the outcome?

        # infinitely bad calibration
        lrs = np.array([5, 5, 5, .2, .2, .2, np.inf])
        y = np.concatenate([np.ones(3), np.zeros(4)])
        self.assertAlmostEqual(devpav(lrs, y, 10), np.inf)

        # binary system
        lrs = np.array([5, 5, 5, .2, 5, .2, .2, .2])
        y = np.concatenate([np.ones(4), np.zeros(4)])
        self.assertAlmostEqual(devpav(lrs, y, 1000), 0.1390735326086103)  # TODO: check this value externally

        # somewhat normal
        lrs = np.array([6, 5, 5, .2, 5, .2, .2, .1])
        y = np.concatenate([np.ones(4), np.zeros(4)])
        self.assertAlmostEqual(devpav(lrs, y, 1000), 0.262396610197457)  # TODO: check this value externally
