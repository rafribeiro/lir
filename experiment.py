import numpy as np

from lir.calibration import DummyCalibrator
from lir.plotting import plot_score_distribution_and_calibrator_fit

cal = DummyCalibrator()

x = np.array([1, 1, 0, 0.4, 0.8])
y = np.array([1, 1, 0, 0, 1])

plot_score_distribution_and_calibrator_fit(cal, x, y, plot_log_odds=True)

