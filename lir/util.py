import collections
import inspect
import warnings

import numpy as np

LR = collections.namedtuple('LR', ['lr', 'p0', 'p1'])


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

def from_log_odds_to_p(log_odds):
    p = to_probability(np.float_power(10, log_odds))
    return(p)

def ln_to_log(ln_data):
    log_data = np.multiply(np.log10(np.exp(1)), ln_data)
    return(log_data)


def warn_deprecated():
    warnings.warn(f'the function `{inspect.stack()[1].function}` is no longer maintained; please check documentation for alternatives')
