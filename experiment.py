import numpy as np

from lir.calibration import DummyCalibrator
from lir.plotting import plot_score_distribution_and_calibrator_fit

cal = DummyCalibrator()

x = np.array([1, 1, 0, 0.4, 0.8])
y = np.array([1, 1, 0, 0, 1])

with np.errstate(divide='ignore'):
    log_odds = np.log10(x/(1-x))

plot_score_distribution_and_calibrator_fit(cal, log_odds, y)

