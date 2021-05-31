import csv

import numpy as np

from lir import IsotonicCalibrator, plot_pav

with open('lir/data/LRsdifferentnormalLLRdistribmu_s=11N_ss=300.csv', 'r') as diff_lrs:
    reader = csv.DictReader(diff_lrs)
    x_diff = [float(row['x']) for row in reader]

with open('lir/data/LRssamenormalLLRdistribmu_s=17N_ss=300.csv', 'r') as same_lrs:
    reader = csv.DictReader(same_lrs)
    x_same = [float(row['x']) for row in reader]

x = np.array(x_diff + x_same)
y = np.array([0] * len(x_diff) + [1] * len(x_same))

plot_pav(x, y)

