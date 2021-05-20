import collections

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
    odds = to_odds(p)
    return np.nan_to_num(np.log10(odds))


def inf_in_array(x):
    """
    Checks if there are inf or -inf in array
    :param x: np.array
    :return: bool
    """
    return any([value in (np.Inf, -np.Inf) for value in x])

def count_inf_in_array(x):
    """
    Gets as input np.array and counts the number of inf and -inf values in array
    :param x: np.array
    :return: integer count
    """
    return np.sum([value in (np.Inf, -np.Inf) for value in x])

def remove_inf_x_y(x, y):
    """
    Removes inf and -inf from scores and removes corresponding labels.
    :param x: np.array with scores or lrs
    :param y: np.array with labels
    :return: Tuple[np.array, np.array] where infinity lrs have been removed
    """
    inf_filter = [value not in (np.Inf, -np.Inf) for value in x]
    return x[inf_filter], y[inf_filter]
