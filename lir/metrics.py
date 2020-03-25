import collections

import numpy as np

from .calibration import IsotonicCalibrator
from .util import Xn_to_Xy, Xy_to_Xn, to_probability, LR


LrStats = collections.namedtuple('LrStats', ['avg_llr', 'avg_llr_class0', 'avg_llr_class1', 'avg_p0_class0', 'avg_p1_class0', 'avg_p0_class1', 'avg_p1_class1', 'cllr_class0', 'cllr_class1', 'cllr', 'lr_class0', 'lr_class1', 'cllr_min', 'cllr_cal'])


def cllr(lrs, y):
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
    with np.errstate(divide='ignore'):
        lrs0, lrs1 = Xy_to_Xn(lrs, y)
        cllr0 = np.mean(np.log2(1 + lrs0))
        cllr1 = np.mean(np.log2(1 + 1/lrs1))
        return .5 * (cllr0 + cllr1)


def cllr_min(lrs, y):
    """
    Calculates the discriminative power for a series of likelihood ratios.

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
    return cllr(lrmin, y)


def calculate_cllr(lr_class0, lr_class1):
    """
    Calculates a log likelihood ratio cost (C_llr) for a series of likelihood
    ratios.

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

    avg_llr_class0 = np.mean(np.log2(1/lr_class0))
    avg_llr_class1 = np.mean(np.log2(lr_class1))
    avg_llr = avg(avg_llr_class0, avg_llr_class1)

    cllr, cllr_class0, cllr_class1 = cllr(*Xn_to_Xy(lr_class0, lr_class1))

    irc = IsotonicCalibrator()
    lr, y = Xn_to_Xy(lr_class0 / (lr_class0 + 1), lr_class1 / (lr_class1 + 1))  # translate LRs to range 0..1
    lrmin = irc.fit_transform(lr, y)
    cllrmin, cllrmin_class0, cllrmin_class1 = cllr(lrmin, y)

    return LrStats(avg_llr, avg_llr_class0, avg_llr_class1, avg_p0_class0, avg_p1_class0, avg_p0_class1, avg_p1_class1, cllr_class0, cllr_class1, cllr, lr_class0, lr_class1, cllrmin, cllr - cllrmin)
