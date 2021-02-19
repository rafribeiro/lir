import collections
import warnings

import numpy as np

from .calibration import IsotonicCalibrator
from .util import Xn_to_Xy, Xy_to_Xn, to_probability, LR


LrStats = collections.namedtuple('LrStats', ['avg_llr', 'avg_llr_class0', 'avg_llr_class1', 'avg_p0_class0', 'avg_p1_class0', 'avg_p0_class1', 'avg_p1_class1', 'cllr_class0', 'cllr_class1', 'cllr', 'lr_class0', 'lr_class1', 'cllr_min', 'cllr_cal'])


def cllr(lrs, y, weights=(1, 1)):
    """
    Calculates a log likelihood ratio cost (C_llr) for a series of likelihood
    ratios.

    Nico Brümmer and Johan du Preez, Application-independent evaluation of speaker detection, In: Computer Speech and
    Language 20(2-3), 2006.

    Parameters
    ----------
    lrs : a numpy array of LRs
    y : a numpy array of labels (0 or 1)

    Returns
    -------
    cllr
        the log likelihood ratio cost
    """

    # ignore errors:
    #   divide -> ignore divide by zero
    #   over -> ignore scalar overflow
    with np.errstate(divide='ignore', over='ignore'):
        lrs0, lrs1 = Xy_to_Xn(lrs, y)
        cllr0 = weights[0] * np.mean(np.log2(1 + lrs0)) if weights[0] > 0 else 0
        cllr1 = weights[1] * np.mean(np.log2(1 + 1/lrs1)) if weights[1] > 0 else 0
        return (cllr0 + cllr1) / sum(weights)


def cllr_min(lrs, y, weights=(1, 1)):
    """
    Estimates the discriminative power from a collection of likelihood ratios.

    Parameters
    ----------
    lrs : a numpy array of LRs
    y : a numpy array of labels (0 or 1)

    Returns
    -------
    cllr_min
        the log likelihood ratio cost
    """
    cal = IsotonicCalibrator()
    lrmin = cal.fit_transform(to_probability(lrs), y)
    return cllr(lrmin, y, weights)


def devpav(lrs, y, resolution=1000):
    """
    Estimate devPAV, a metric for calibration.

    devPAV is the cumulative deviation of the PAV transformation from
    the identity line. It is calculated in the LR range where misleading LRs
    occur.

    See also: P. Vergeer, Measuring calibration of likelihood ratio systems: a
    comparison of four systems, including a new metric devPAV, to appear

    This implementation estimates devPAV by calculating the average deviation
    for a large number of LRs.

    Parameters
    ----------
    lrs : a numpy array of LRs
    y : a numpy array of labels (0 or 1)
    resolution : the number of measurements in the range of misleading evidence; a higher value yields a more accurate estimation

    Returns
    -------
    devPAV
        an estimation of devPAV
    """
    lrs0, lrs1 = Xy_to_Xn(lrs, y)
    if len(lrs0) == 0 or len(lrs1) == 0:
        raise ValueError('devpav: illegal input: at least one value is required for each class')

    # find misleading LR extremes
    first_misleading = np.min(lrs1)
    last_misleading = np.max(lrs0)
    if first_misleading > last_misleading:  # test for perfect discrimination
        return 0

    if np.isinf(first_misleading) or np.isinf(last_misleading):  # test for infinitely misleading LRs
        return np.inf

    # calibrate on the input LRs
    cal = IsotonicCalibrator()
    cal.fit_transform(to_probability(lrs), y)

    # take `resolution` points evenly divided along the range of misleading LRs
    xlr = np.exp(np.linspace(np.log(first_misleading), np.log(last_misleading), resolution))
    pavlr = cal.transform(to_probability(xlr))

    #print('range', first_misleading, last_misleading)
    #for pair in zip(xlr, pavlr):
    #    print('X', pair)

    devlr = np.absolute(np.log10(xlr) - np.log10(pavlr))
    return (np.sum(devlr) / resolution) * (np.log10(last_misleading) - np.log10(first_misleading))


def calculate_lr_statistics(lr_class0, lr_class1):
    """
    Calculates various statistics for a collection of likelihood ratios.

    Parameters
    ----------
    lr_class0 : list of float
        Likelihood ratios which are calculated for measurements which are
        sampled from class 0.
    lr_class1 : list of float
        Likelihood ratios which are calculated for measurements which are
        sampled from class 1.

    Returns
    -------
    LrStats
        Likelihood ratio statistics.
    """
    assert len(lr_class0) > 0
    assert len(lr_class1) > 0

    def avg(*args):
        return sum(args) / len(args)

    if type(lr_class0[0]) == LR:
        avg_p0_class0 = avg(*[lr.p0 for lr in lr_class0])
        avg_p1_class0 = avg(*[lr.p1 for lr in lr_class0])
        avg_p0_class1 = avg(*[lr.p0 for lr in lr_class1])
        avg_p1_class1 = avg(*[lr.p1 for lr in lr_class1])
        lr_class0 = np.array([ lr.lr for lr in lr_class0 ])
        lr_class1 = np.array([ lr.lr for lr in lr_class1 ])
    else:
        if type(lr_class0) == list:
            lr_class0 = np.array(lr_class0)
            lr_class1 = np.array(lr_class1)

        avg_p0_class0 = None
        avg_p1_class0 = None
        avg_p0_class1 = None
        avg_p1_class1 = None

    with warnings.catch_warnings():
        try:
            avg_llr_class0 = np.mean(np.log2(1/lr_class0))
            avg_llr_class1 = np.mean(np.log2(lr_class1))
            avg_llr = avg(avg_llr_class0, avg_llr_class1)
        except RuntimeWarning:
            # possibly illegal LRs such as 0 or inf
            avg_llr_class0 = np.nan
            avg_llr_class1 = np.nan
            avg_llr = np.nan

    lrs, y = Xn_to_Xy(lr_class0, lr_class1)
    cllr_class0 = cllr(lrs, y, weights=(1, 0))
    cllr_class1 = cllr(lrs, y, weights=(0, 1))
    cllr_ = .5 * (cllr_class0 + cllr_class1)

    cllrmin_class0 = cllr_min(lrs, y, weights=(1, 0))
    cllrmin_class1 = cllr_min(lrs, y, weights=(0, 1))
    cllrmin = .5 * (cllrmin_class0 + cllrmin_class1)

    return LrStats(avg_llr, avg_llr_class0, avg_llr_class1, avg_p0_class0, avg_p1_class0, avg_p0_class1, avg_p1_class1, cllr_class0, cllr_class1, cllr_, lr_class0, lr_class1, cllrmin, cllr_ - cllrmin)
