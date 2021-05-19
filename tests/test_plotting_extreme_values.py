import unittest

import numpy as np

from lir import plot_pav


class TestExtremeValuesPlotting(unittest.TestCase):
    def setUp(self):
        self.extreme_scores = np.array([1., 0., 0.3, 0.4, 0., 0.999999999999])
        self.y = np.array([1, 0, 0, 0, 0, 1])

    def test_plot_pav(self):
        with self.assertWarns(UserWarning,
                              msg="Some pre-calibrated lrs were inf or -inf and were not used for the PAV transformation and "
                                  "subsequent plotting"):
            lrs = self.extreme_scores/(1-self.extreme_scores)
            plot_pav(lrs, self.y, add_misleading=2)


if __name__ == "__main__":
    unittest.main()
