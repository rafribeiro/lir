import numpy as np

from lir import plot_pav, plot_lr_histogram

x = np.array([1., 0., 0.3, 0.4, 0., 0.999999999999])
y = np.array([1, 0, 0, 0, 0, 1])

# log_odds
lrs = np.log(x/(1-x))

# plot_pav is dus een probleem met inf
#plot_pav(lrs, y, add_misleading=10)

# plot_lr_histogram
plot_lr_histogram(lrs, y)