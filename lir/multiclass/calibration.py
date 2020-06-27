"""
A multiclass calibrator implements at least the following two methods:

fit(self, X, y)

    X is a two dimensional array of scores; rows are samples; columns are scores 0..1 per class
    y is a one dimensional array of classes 0..n
    returns self

transform(self, X)

    X is a two dimensional array of scores, as in fit()
    returns a two dimensional array of lrs; same dimensions as X
"""
import logging

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.linear_model import LogisticRegression

from ..calibration import DummyCalibrator
from .util import to_odds, get_classes_from_scores_Xy

LOG = logging.getLogger(__name__)


class LogitCalibrator(BaseEstimator, TransformerMixin):
    """
    Calculates a likelihood ratio of a score value, provided it is from one of
    several distributions. Uses logistic regression for interpolation.
    """

    def fit(self, X, y):
        self._classes = get_classes_from_scores_Xy(X, y, np.unique(y))
        self._logit = []
        for cls in self._classes:
            logit = LogisticRegression(class_weight='balanced')
            y_cls = np.zeros(y.shape)
            y_cls[y==cls] = 1
            logit.fit(X, y_cls)
            self._logit.append(logit)

        return self

    def transform(self, X):
        assert X.shape[1] == self._classes.size

        p = []
        for cls in range(self._classes.size):
            p.append(self._logit[cls].predict_proba(X)[:, 1])  # probability of class 1

        lrs = to_odds(np.stack(p, axis=1))
        return lrs
