import unittest.mock

import numpy as np

import lir.plotting as plotting


class TestPlotPav(unittest.TestCase):
    @unittest.mock.patch("%s.plotting.plt" % __name__)
    def test_remove_inf(self, mock_plt):
        x = np.array([1, 1, 0, 0.4, 0.8])
        y = np.array([1, 1, 0, 0, 1])
        with np.errstate(divide='ignore'):
            infinite_lrs = x / (1 - x)
        plotting.plot_pav(infinite_lrs, y, add_misleading=1)
        mock_plt.text.assert_called_once_with(-0.6760912590556811, 1.1020599913279625,
                                              '2 pre-calibrated lr(s) were inf and were not used for '
                                              'the PAV transformation!',
                                              fontsize=14, ha='left', style='oblique', wrap=True)


if __name__ == "__main__":
    unittest.main()
