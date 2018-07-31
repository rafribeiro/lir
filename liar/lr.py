import collections
import logging
import math

import numpy as np
import sklearn
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.linear_model import LogisticRegression
import sklearn.mixture


LOG = logging.getLogger(__name__)


def Xn_to_Xy(*Xn):
    X = np.concatenate(Xn)
    y = np.concatenate([ np.ones((X.shape[0],), dtype=np.int8)*i for i, X in enumerate(Xn) ])
    return X, y


class NormalizedCalibrator(BaseEstimator, TransformerMixin):
    """
    Normalizer for any calibration function.

    Scales the probability density function of a calibrator so that the
    probability mass is 1.
    """

    def __init__(self, calibrator, add_one=False, sample_size=100, value_range=(0,1)):
        self.calibrator = calibrator
        self.add_one = add_one
        self.value_range = value_range
        self.step_size = (value_range[1] - value_range[0]) / sample_size

    def fit(self, X0, X1):
        self.X0n = X0.shape[0]
        self.X1n = X1.shape[0]
        self.calibrator.fit(X0, X1)
        self.calibrator.transform(np.arange(self.value_range[0], self.value_range[1], self.step_size))
        self.p0mass = np.sum(self.calibrator.p0) / 100
        self.p1mass = np.sum(self.calibrator.p1) / 100

    def transform(self, X):
        self.calibrator.transform(X)
        self.p0 = self.calibrator.p0 / self.p0mass
        self.p1 = self.calibrator.p1 / self.p1mass
        if self.add_one:
            self.p0 = self.X0n / (self.X0n + 1) * self.p0 + 1 / self.X0n
            self.p1 = self.X1n / (self.X1n + 1) * self.p1 + 1 / self.X1n
        return self.p1 / self.p0

    def __getattr__(self, name):
        return getattr(self.calibrator, name)


class FractionCalibrator(BaseEstimator, TransformerMixin):
    """
    Calculates a likelihood ratio of the distance of a score value to the
    extremes of its value range.
    """

    def __init__(self, value_range=(0,1)):
        self.value_range = value_range

    def fit(self, X0, X1):
        self._abs_points0 = np.abs(self.value_range[0] - X0)
        self._abs_points1 = np.abs(self.value_range[1] - X1)

    def density(self, X, class_value, points):
        X = np.abs(self.value_range[class_value] - X)
        return (1 + np.array([ points[points >= x].shape[0] for x in X ])) / (1 + len(points))

    def transform(self, X):
        self.p0 = self.density(X, 0, self._abs_points0)
        self.p1 = self.density(X, 1, self._abs_points1)
        return self.p1 / self.p0


class KDECalibrator(BaseEstimator, TransformerMixin):
    """
    Calculates a likelihood ratio of a score value, provided it is from one of
    two distributions. Uses kernel density estimation (KDE) for interpolation.
    """

    def bandwidth_silverman(self, X):
        """
        Estimates the optimal bandwidth parameter using Silverman's rule of
        thumb.
        """
        assert len(X) > 0
        v = math.pow(np.std(X), 5) / len(X) * 4./3
        return math.pow(v, .2)

    def bandwidth_scott(self, X):
        """
        Not implemented.
        """
        raise

    def fit(self, X0, X1):
        bandwidth0 = self.bandwidth_silverman(X0)
        bandwidth1 = self.bandwidth_silverman(X1)

        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        self._kde0 = sklearn.neighbors.KernelDensity(kernel='gaussian', bandwidth=bandwidth0).fit(X0)
        self._kde1 = sklearn.neighbors.KernelDensity(kernel='gaussian', bandwidth=bandwidth1).fit(X1)
        self._base_value0 = 1./X0.shape[0]
        self._base_value1 = 1./X1.shape[0]

    def transform(self, X):
        X = X.reshape(-1, 1)
        self.p0 = self._base_value0 + np.exp(self._kde0.score_samples(X))
        self.p1 = self._base_value1 + np.exp(self._kde1.score_samples(X))
        return self.p1 / self.p0


class LogitCalibrator(BaseEstimator, TransformerMixin):
    """
    Calculates a likelihood ratio of a score value, provided it is from one of
    two distributions. Uses logistic regression for interpolation.
    """

    def fit(self, X0, X1):
        self._logit = LogisticRegression(class_weight='balanced')
        self._logit.fit(*Xn_to_Xy(X0, X1))

    def transform(self, X):
        X = self._logit.predict_proba(X)[:,1] # probability of class 1
        self.p0 = (1 - X)
        self.p1 = X
        return self.p1 / self.p0


class GaussianCalibrator(BaseEstimator, TransformerMixin):
    """
    Calculates a likelihood ratio of a score value, provided it is from one of
    two distributions. Uses a gaussian mixture model for interpolation.
    """

    def fit(self, X0, X1):
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        self._model0 = sklearn.mixture.GaussianMixture().fit(X0)
        self._model1 = sklearn.mixture.GaussianMixture().fit(X1)
        self._base_value0 = 1. / X0.shape[0]
        self._base_value1 = 1. / X1.shape[0]

    def transform(self, X):
        X = X.reshape(-1, 1)
        self.p0 = self._base_value0 + np.exp(self._model0.score_samples(X))
        self.p1 = self._base_value1 + np.exp(self._model1.score_samples(X))
        return self.p1 / self.p0


_PRINT_ISOTONIC_WARNING = True
_PRINT_SKLEARN_WARNING = True
class IsotonicCalibrator(BaseEstimator, TransformerMixin):
    """
    Calculates a likelihood ratio of a score value, provided it is from one of
    two distributions. Uses isotonic regression for interpolation.
    """

    def __init__(self, add_one=False, use_sklearn=True):
        global _PRINT_ISOTONIC_WARNING, _PRINT_SKLEARN_WARNING

        self.add_one = add_one

        self.use_sklearn = use_sklearn
        if self.use_sklearn:
            self._ir = sklearn.isotonic.IsotonicRegression()

        if use_sklearn and _PRINT_SKLEARN_WARNING:
            LOG.warning('Rolf zegt: [sklearn implementation of isotonic regression] appears incorrect when weights and multiple identical values are present')
            _PRINT_SKLEARN_WARNING = False
        if not use_sklearn and _PRINT_ISOTONIC_WARNING:
            LOG.warning('broken implementation of isotonic regression (see unit test)')
            _PRINT_ISOTONIC_WARNING = False

    def _isotonic_regression(y,
                             weight):
        """
        implementation of isotonic regression

        :param y: ordered input values
        :param weight: associated weights
        :return: function values such that the function is non-decreasing and minimises the weighted SSE: ∑ w_i (y_i - f_i)^2
        """

        n = y.shape[0]
        # The algorithm proceeds by iteratively updating the solution
        # array.

        solution = y.copy()

        if n <= 1:
            return solution

        n -= 1
        pooled = 1
        while pooled > 0:
            # repeat until there are no more adjacent violators.
            i = 0
            pooled = 0
            while i < n:
                k = i
                while k < n and solution[k] >= solution[k + 1]:
                    k += 1
                if solution[i] != solution[k]:
                    # solution[i:k + 1] is a decreasing subsequence, so
                    # replace each point in the subsequence with the
                    # weighted average of the subsequence.
                    numerator = 0.0
                    denominator = 0.0
                    for j in range(i, k + 1):
                        numerator += solution[j] * weight[j]
                        denominator += weight[j]
                    for j in range(i, k + 1):
                        solution[j] = numerator / denominator
                    pooled = 1
                i = k + 1
        return solution

    def fit(self, X0, X1, add_one=None):
        if not self.use_sklearn:
            raise ValueError('not implemented')

        # prevent extreme LRs
        if add_one or (add_one is None and self.add_one):
            X0 = np.append(X0, 1)
            X1 = np.append(X1, 0)

        X0n = X0.shape[0]
        X1n = X1.shape[0]
        X, y = Xn_to_Xy(X0, X1)
        weight = np.concatenate([ [X1n] * X0n, [X0n] * X1n ])
        self._ir.fit(X, y, sample_weight=weight)

    def transform(self, X):
        if not self.use_sklearn:
            raise ValueError('not implemented')

        posterior = self._ir.transform(X)

        self.p0 = (1 - posterior)
        self.p1 = posterior
        with np.errstate(divide='ignore'):
            return self.p1 / self.p0

    def fit_transform(self, X0, X1, add_one=None):
        # prevent extreme LRs
        if add_one or (add_one is None and self.add_one):
            X0 = np.append(X0, 1)
            X1 = np.append(X1, 0)

        X0n = X0.shape[0]
        X1n = X1.shape[0]
        X, y = Xn_to_Xy(X0, X1)
        weight = np.concatenate([ [X1n] * X0n, [X0n] * X1n ])

        if self.use_sklearn:
            posterior = self._ir.fit_transform(X, y, sample_weight=weight)
        else:
            y = y * 1.0 # be sure to have floats
            sor = np.argsort(X)
            dhat = IsotonicCalibrator._isotonic_regression(y[sor], weight[sor])
            posterior = dhat[np.argsort(sor)]

        self.p0 = (1 - posterior)
        self.p1 = posterior
        with np.errstate(divide='ignore'):
            lr = self.p1 / self.p0

        return lr[:X0n], lr[X0n:]


class DummyCalibrator(BaseEstimator, TransformerMixin):
    """
    Calculates a likelihood ratio of a score value, provided it is from one of
    two distributions. No calibration is applied. Instead, the score value is
    interpreted as a posterior probability of the value being sampled from
    class 1.
    """
    def fit(self, X0, X1):
        self._base_value0 = 1. / X0.shape[0]
        self._base_value1 = 1. / X1.shape[0]

    def transform(self, X):
        self.p0 = self._base_value0 + (1 - X)
        self.p1 = self._base_value0 + X
        return self.p1 / self.p0


class CalibratedScorer:
    def __init__(self, scorer, calibrator, fit_calibrator=False):
        self.scorer = scorer
        self.calibrator = calibrator
        self.fit_calibrator = fit_calibrator

    def fit(self, X, y):
        self.scorer.fit(X, y)
        if self.fit_calibrator:
            p = self.scorer.predict_proba(X)
            self.calibrator.fit(p[y==0,1], p[y==1,1])

    def predict_lr(self, X):
        X = self.scorer.predict_proba(X)[:,1]
        return self.calibrator.transform(X)


class CalibratedScorerCV:
    def __init__(self, scorer, calibrator, n_splits):
        self.scorer = scorer
        self.calibrator = calibrator
        self.n_splits = n_splits

    def fit(self, X, y):
        kf = sklearn.model_selection.StratifiedKFold(n_splits=self.n_splits, shuffle=True)

        class0_calibrate = np.empty([0, 1])
        class1_calibrate = np.empty([0, 1])
        for train_index, cal_index in kf.split(X, y):
            self.scorer.fit(X[train_index], y[train_index])
            p = self.scorer.predict_proba(X[cal_index])
            class0_calibrate = np.append(class0_calibrate, p[y[cal_index]==0,1])
            class1_calibrate = np.append(class1_calibrate, p[y[cal_index]==1,1])
        self.calibrator.fit(class0_calibrate, class1_calibrate)

        self.scorer.fit(X, y)

    def predict_lr(self, X):
        scores = self.scorer.predict_proba(X)[:,1] # probability of class 1
        return self.calibrator.transform(scores)


LR = collections.namedtuple('LR', ['lr', 'p0', 'p1'])


def calibrate_lr(point, calibrator):
    """
    Calculates a calibrated likelihood ratio (LR).

    P(E|H1) / P(E|H0)
    
    Parameters
    ----------
    point : float
        The point of interest for which the LR is calculated. This point is
        sampled from either class 0 or class 1.
    calibrator : a calibration function
        A calibration function which is fitted for two distributions.

    Returns
    -------
    tuple<3> of float
        A likelihood ratio, with positive values in favor of class 1, and the
        probabilities P(E|H0), P(E|H1).
    """
    lr = calibrator.transform(np.array(point))[0]
    return LR(lr, calibrator.p0[0], calibrator.p1[0])


LrStats = collections.namedtuple('LrStats', ['avg_llr', 'avg_llr_class0', 'avg_llr_class1', 'avg_p0_class0', 'avg_p1_class0', 'avg_p0_class1', 'avg_p1_class1', 'cllr_class0', 'cllr_class1', 'cllr', 'lr_class0', 'lr_class1', 'cllr_min', 'cllr_cal'])


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

    def _cllr(lr0, lr1):
        with np.errstate(divide='ignore'):
            cllr0 = np.mean(np.log2(1 + lr0))
            cllr1 = np.mean(np.log2(1 + 1/lr1))
            return avg(cllr0, cllr1), cllr0, cllr1

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

    cllr, cllr_class0, cllr_class1 = _cllr(lr_class0, lr_class1)

    irc = IsotonicCalibrator()
    lrmin_class0, lrmin_class1 = irc.fit_transform(lr_class0 / (lr_class0 + 1), lr_class1 / (lr_class1 + 1))
    cllrmin, cllrmin_class0, cllrmin_class1 = _cllr(lrmin_class0, lrmin_class1)

    return LrStats(avg_llr, avg_llr_class0, avg_llr_class1, avg_p0_class0, avg_p1_class0, avg_p0_class1, avg_p1_class1, cllr_class0, cllr_class1, cllr, lr_class0, lr_class1, cllrmin, cllr - cllrmin)


def apply_scorer(scorer, X):
    return scorer.predict_proba(X)[:,1] # probability of class 1


def scorebased_lr(scorer, calibrator, X0_train, X1_train, X0_calibrate, X1_calibrate, X_disputed):
    """
    Trains a classifier, calibrates the outcome and calculates a LR for a single
    sample.

    P(E|H1) / P(E|H0)

    All numpy arrays have two dimensions, representing samples and features.

    Parameters
    ----------
    scorer : classifier
        A model to be trained. Must support probability output.
    X0_train : numpy array
        Training set of samples from class 0
    X1_train : numpy array
        Training set of samples from class 1
    X0_calibrate : numpy array
        Calibration set of samples from class 0
    X1_calibrate : numpy array
        Calibration set of samlpes from class 1
    X_disputed : numpy array
        Test samples of unknown class

    Returns
    -------
    float
        A likelihood ratio for `X_disputed`
    """
    scorer.fit(*Xn_to_Xy(X0_train, X1_train))
    calibrator.fit(scorer.predict_proba(X0_calibrate)[:,1], scorer.predict_proba(X1_calibrate)[:,1])
    scorer = CalibratedScorer(scorer, calibrator)

    return scorer.predict_lr(X_disputed)


def calibrated_cllr(calibrator, class0_calibrate, class1_calibrate, class0_test=None, class1_test=None):
    calibrator.fit(class0_calibrate, class1_calibrate)

    use_calibration_set_for_test = class0_test is None
    if use_calibration_set_for_test:
        class0_test = class0_calibrate
        class1_test = class1_calibrate

    lrs0 = calibrator.transform(class0_test)
    lrs1 = calibrator.transform(class1_test)

    lrs0 = [ LR(*stats) for stats in zip(lrs0, calibrator.p0, calibrator.p1) ]
    lrs1 = [ LR(*stats) for stats in zip(lrs1, calibrator.p0, calibrator.p1) ]

    return calculate_cllr(lrs0, lrs1)


def scorebased_cllr(scorer, calibrator, X0_train, X1_train, X0_calibrate, X1_calibrate, X0_test=None, X1_test=None):
    """
    Trains a classifier on a training set, calibrates the outcome with a
    calibration set, and calculates a LR (likelihood ratio) for all samples in
    a test set. Calculates a Cllr (log likelihood ratio cost) value for the LR
    values. If no test set is provided, the calibration set is used for both
    calibration and testing.

    All numpy arrays have two dimensions, representing samples and features.

    Parameters
    ----------
    scorer : classifier
        A model to be trained. Must support probability output.
    density_function : function
        A density function which is used to deterimine the density of
        a classifier outcome when sampled from either of both classes.
    X0_train : numpy array
        Training set for class 0
    X1_train : numpy array
        Training set for class 1
    X0_calibrate : numpy array
        Calibration set for class 0
    X1_calibrate : numpy array
        Calibration set for class 1
    X0_test : numpy array
        Test set for class 0
    X1_test : numpy array
        Test set for class 1

    Returns
    -------
    float
        A likelihood ratio for `X_test`
    """
    if X0_test is None:
        LOG.debug('scorebased_cllr: training_size: {train0}/{train1}; calibration size: {cal0}/{cal1}'.format(train0=X0_train.shape[0], train1=X1_train.shape[0], cal0=X0_calibrate.shape[0], cal1=X1_calibrate.shape[0]))
    else:
        LOG.debug('scorebased_cllr: training_size: {train0}/{train1}; calibration size: {cal0}/{cal1}; test size: {test0}/{test1}'.format(train0=X0_train.shape[0], train1=X1_train.shape[0], cal0=X0_calibrate.shape[0], cal1=X1_calibrate.shape[0], test0=X0_test.shape[0], test1=X1_test.shape[0]))

    scorer.fit(*Xn_to_Xy(X0_train, X1_train))

    class0_test = apply_scorer(scorer, X0_test) if X0_test is not None else None
    class1_test = apply_scorer(scorer, X1_test) if X0_test is not None else None

    return calibrated_cllr(calibrator, apply_scorer(scorer, X0_calibrate), apply_scorer(scorer, X1_calibrate), class0_test, class1_test)


def scorebased_lr_kfold(scorer, calibrator, n_splits, X0_train, X1_train, X_disputed):
    scorer = CalibratedScorerCV(scorer, calibrator, n_splits=n_splits)
    scorer.fit(*Xn_to_Xy(X0_train, X1_train))
    return scorer.predict_lr(X_disputed)


def scorebased_cllr_kfold(scorer, calibrator, n_splits, X0_train, X1_train, X0_test, X1_test):
    LOG.debug('scorebased_cllr_kfold: training_size: {train0}/{train1}; test size: {test0}/{test1}'.format(train0=X0_train.shape[0], train1=X1_train.shape[0], test0=X0_test.shape[0], test1=X1_test.shape[0]))

    X_disputed = np.concatenate([X0_test, X1_test])
    lrs = scorebased_lr_kfold(scorer, calibrator, n_splits, X0_train, X1_train, X_disputed)

    assert X0_test.shape[0] + X1_test.shape[0] == len(lrs)
    lrs0 = lrs[:X0_test.shape[0]]
    lrs1 = lrs[X0_test.shape[0]:]

    lrs0 = [ LR(*stats) for stats in zip(lrs0, calibrator.p0, calibrator.p1) ]
    lrs1 = [ LR(*stats) for stats in zip(lrs1, calibrator.p0, calibrator.p1) ]

    return calculate_cllr(lrs0, lrs1)
