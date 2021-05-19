import unittest

import numpy as np


class TestExtremeValuesPlotting(unittest.TestCase):
    def setUp(self):
        self.extreme_x = np.array([1., 0., 0.3, 0.4, 0., 0.999999999999])
        self.y = np.array([1, 0, 0, 0, 0, 1])

    def plot_pav(self):
