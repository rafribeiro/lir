import numpy as np

from lir.calibration import DummyCalibrator, IsotonicCalibrator
from lir.plotting import plot_score_distribution_and_calibrator_fit

cal = IsotonicCalibrator(add_misleading=1)

x = np.array([0.1,0.2,0.4,0.8, 0.7, 0.4, 0.85, 0.8, 1, 0, 0.9])
y = np.array([0,0,0,1, 1, 0, 1, 1, 1, 0, 1])

with np.errstate(divide='ignore'):
    log_odds = np.log10(x/(1-x))

cal.fit(log_odds, y)

plot_score_distribution_and_calibrator_fit(cal, log_odds, y)

