import collections
import inspect
import warnings

import numpy as np

LR = collections.namedtuple('LR', ['lr', 'p0', 'p1'])

#for Gaussian and KDE-calibrator fitting: remove negInf, Inf and compensate
def compensate_and_remove_negInf_Inf(log_odds_X, y):
        el_H1 = np.all(np.array([log_odds_X != np.Inf, y == 1]), axis=0)
        el_H2 = np.all(np.array([log_odds_X != -1 * np.Inf, y == 0]), axis=0)
        n_H1 = np.sum(y)
        numerator = np.sum(el_H1)/n_H1
        denominator = np.sum(el_H2)/(len(y)-n_H1)
        y = y[np.any(np.array([el_H1, el_H2]), axis=0)]
        log_odds_X = log_odds_X[np.any(np.array([el_H1, el_H2]), axis=0)]
        return log_odds_X, y, numerator, denominator


#for calirbation training on log_odds domain. Check whether negInf under H1 and Inf under H2 occurs and give error if so
def check_misleading_Inf_negInf(log_odds_X, y):
    # sanity checks
    # give error message if H1's contain zeros and H2's contain ones
    if np.any(np.isneginf(log_odds_X[y == 1])) and np.any(np.isposinf(log_odds_X[y == 0])):
        raise ValueError('Your data is possibly problematic for this calibrator. You have negInf under H1 and Inf under H2 after logodds transform. If you really want to proceed, adjust probs in order to get finite values on the logodds domain')
    # give error message if H1's contain zeros
    if np.any(np.isneginf(log_odds_X[y == 1])):
        raise ValueError('Your data is possibly problematic for this calibrator. You have negInf under H1 after logodds transform. If you really want to proceed, adjust probs in order to get finite values on the logodds domain')
    # give error message if H2's contain ones
    if np.any(np.isposinf(log_odds_X[y == 0])):
        raise ValueError('Your data is possibly problematic for this calibrator. You have Inf under H2 after logodds transform. If you really want to proceed, adjust probs in order to get finite values on the logodds domain')


def get_classes_from_Xy(X, y, classes=None):
    assert len(X.shape) >= 1, f'expected: X has at least 1 dimensions; found: {len(X.shape)} dimensions'
    assert len(y.shape) == 1, f'expected: y is a 1-dimensional array; found: {len(y.shape)} dimensions'
    assert X.shape[0] == y.size, f'dimensions of X and y do not match; found: {X.shape[0]} != {y.size}'

    return np.unique(y) if classes is None else np.asarray(classes)


def Xn_to_Xy(*Xn):
    """
    Convert Xn to Xy format.

    Xn is a format where samples are divided into separate variables based on class.
    Xy is a format where all samples are concatenated, with an equal length variable y indicating class."""
    Xn = [np.asarray(X) for X in Xn]
    X = np.concatenate(Xn)
    y = np.concatenate([np.ones((X.shape[0],), dtype=np.int8) * i for i, X in enumerate(Xn)])
    return X, y


def Xy_to_Xn(X, y, classes=[0, 1]):
    """
    Convert Xy to Xn format.

    Xn is a format where samples are divided into separate variables based on class.
    Xy is a format where all samples are concatenated, with an equal length variable y indicating class."""

    classes = get_classes_from_Xy(X, y, classes)
    return [X[y == yvalue] for yvalue in classes]


def to_probability(odds):
    """
    Converts odds to a probability

    Returns
    -------
       1                , for odds values of inf
       odds / (1 + odds), otherwise
    """
    inf_values = odds == np.inf
    with np.errstate(invalid='ignore'):
        p = np.divide(odds, (1 + odds))
    p[inf_values] = 1
    return p


def to_odds(p):
    """
    Converts a probability to odds
    """
    with np.errstate(divide='ignore'):
        return p / (1 - p)


def to_log_odds(p):
    np.seterr(divide='ignore')
    complement = np.add(1, np.multiply(-1, p))
    log_odds = np.add(np.log10(p), np.multiply(-1, np.log10(complement)))
    np.seterr(divide='warn')
    return (log_odds)

def ln_to_log(ln_data):
    log_data = np.multiply(np.log10(np.exp(1)), ln_data)
    return(log_data)


def warn_deprecated():
    warnings.warn(f'the function `{inspect.stack()[1].function}` is no longer maintained; please check documentation for alternatives')
