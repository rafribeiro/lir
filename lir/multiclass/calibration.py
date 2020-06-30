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
import math
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KernelDensity

from ..calibration import DummyCalibrator
from .util import to_odds, get_classes_from_scores_Xy, to_probability

LOG = logging.getLogger(__name__)


class LogitCalibrator(BaseEstimator, TransformerMixin):
    """
    Calculates a likelihood ratio of a score value, provided it is from one of
    several distributions. Uses logistic regression for interpolation.
    """

    def fit(self, X, y):
        self._logit = LogisticRegression(class_weight='balanced')
        self._logit.fit(X, y)

        return self

    def transform(self, X):
        self.p = self._logit.predict_proba(X)
        lrs = to_odds(self.p)
        return lrs


class BalancedPriorCalibrator(BaseEstimator, TransformerMixin):
    def __init__(self, backend):
        self.backend = backend

    def fit(self, X, y):
        self.backend.fit(X, y)
        return self

    def transform(self, X):
        X = to_probability(self.backend.transform(X))
        self.priors = np.ones(X.shape[1]) / X.shape[1]

        priors_sum = np.sum(self.priors)
        prior_odds = self.priors / (priors_sum - self.priors)
        lrs = to_odds(X) / prior_odds
        self.p = to_probability(lrs)
        return lrs
