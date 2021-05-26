import numpy as np
from lir import plotting
x = np.array([1, 0.4, 0.8, 0.6, 0.7, 0.75, 0.3,0.01, 0.5, 0.5, 0.5, 0.5])
y = np.array([1,0, 1, 1, 0, 1, 1, 0,1,1,1,1])
# x= np.array([0.4, 0.5, 0.6])
# y = np.array([0,1,1])

with np.errstate(divide='ignore'):
    infinite_lrs = x / (1 - x)

plotting.plot_pav(infinite_lrs, y, add_misleading=False,  show=True)