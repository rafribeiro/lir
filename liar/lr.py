import collections
import math

import numpy as np
import sklearn
import sklearn.mixture


class NoneAttr:
    def __getattr__(self, name):
        return None


class AbstractProbabilityFunction:
    def __init__(self, points, class_value, value_range, remove_from_reference_points, parent):
        self.points = points if points is not None else parent.points
        self.class_value = class_value if class_value is not None else parent.class_value
        self.value_range = value_range if value_range is not None else parent.value_range
        self.remove_from_reference_points = remove_from_reference_points if remove_from_reference_points is not None else parent.remove_from_reference_points

    def __call__(self, **kw):
        func = self.clone(parent=self, **kw)
        if 'point' in kw:
            return func.probability(kw['point'])
        else:
            return func


class probability_fraction(AbstractProbabilityFunction):
    """
    Calculates the probability of the distance of `point` to `value_range[class_value]`,
    provided it is from the same distribution as `reference_points`.

    Parameters
    ----------
    point : float
        The point of interest for which the probability is calculated.
    reference_points : list of float
        Points from the same distribution as `point`.
    class_value : int
        The class from which `reference_points` are sampled; this should be
        either `0` or `1`.
    value_range : tuple of two floats
        All values of `reference_points` and `point` must be within this
        range.

    Returns
    -------
    float
        A probability value.
    """
    def __init__(self, points=None, class_value=None, value_range=None, remove_from_reference_points=None, parent=NoneAttr()):
        super().__init__(points, class_value, value_range, remove_from_reference_points, parent)

        if points is not None:
            self._abs_points = sorted([abs(self.value_range[class_value] - p) for p in points])
        else:
            self._abs_points = parent._abs_points

    def clone(self, **kw):
        return probability_fraction(**kw)

    def probability(self, point):
        point = abs(self.value_range[self.class_value] - point)
        add_one = float(not self.remove_from_reference_points)
        return (add_one + len([p for p in self._abs_points if p >= point])) / (add_one + len(self._abs_points))


class probability_kde(AbstractProbabilityFunction):
    """
    Calculates the probability of the value of `point`, provided it is from the
    same distribution as `reference_points`. Uses kernel density estimation
    (KDE) for interpolation.

    Parameters
    ----------
    point : float
        The point of interest for which the probability is calculated.
    reference_points : list of float
        Points from the same distribution as `point`.
    class_value : int
        ignored
    value_range : tuple of two floats
        ignored

    Returns
    -------
    float
        A probability value.
    """
    def __init__(self, points=None, class_value=None, value_range=None, remove_from_reference_points=None, bandwidth=None, base_value=None, parent=NoneAttr()):
        super().__init__(points, class_value, value_range, remove_from_reference_points, parent)

        self.bandwidth = bandwidth if bandwidth is not None else parent.bandwidth
        self.base_value_param = base_value if base_value is not None else parent.base_value_param

        if points is not None:
            bandwidth = self.bandwidth if self.bandwidth is not None else np.std(points)
            self._kde = sklearn.neighbors.KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(np.array(points).reshape(-1,1))
            self._base_value = self.base_value_param if self.base_value_param is not None else 1. / len(points)
        else:
            self._kde = parent._kde
            self._base_value = parent._base_value

    def clone(self, **kw):
        return probability_kde(**kw)

    def probability(self, point):
        p = self._base_value + np.exp(self._kde.score_samples(np.array(point).reshape(1,1)))[0]
        return p


class probability_gaussian_mixture(AbstractProbabilityFunction):
    """
    Calculates the probability of the value of `point`, provided it is from the
    same distribution as `reference_points`. Uses a gaussian mixture model for
    interpolation.

    Parameters
    ----------
    point : float
        The point of interest for which the probability is calculated.
    reference_points : list of float
        Points from the same distribution as `point`.
    class_value : int
        ignored
    value_range : tuple of two floats
        ignored

    Returns
    -------
    float
        A probability value.
    """
    def __init__(self, points=None, class_value=None, value_range=None, remove_from_reference_points=None, parent=NoneAttr()):
        super().__init__(points, class_value, value_range, remove_from_reference_points, parent)

        if points is not None:
            self._model = sklearn.mixture.GaussianMixture().fit(np.array(points).reshape(-1,1))
        else:
            self._model = parent._model

    def clone(self, **kw):
        return probability_gaussian_mixture(**kw)

    def probability(self, point):
        p = np.exp(self._model.score_samples(np.array(point).reshape(1,1)))[0]
        assert p > 0
        return p


class probability_copy(AbstractProbabilityFunction):
    """
    Calculates the probability of point being sampled from the given class. The
    point value is interpreted as a probability of the point being sampled from
    class 1. If `class_value` is 1, the value of `point` is returned; if
    `class_value` is 0, then 1 - `point` is returned.

    Parameters
    ----------
    point : float
        The point of interest for which the probability is calculated.
    reference_points : list of float
        ignored
    class_value : int
        The class from which `reference_points` are sampled; this should be
        either 0 or 1.
    value_range : tuple of two floats
        ignored and assumed to be [0,1]

    Returns
    -------
    float
        A probability value.
    """

    def __init__(self, points=None, class_value=None, value_range=None, remove_from_reference_points=None, parent=NoneAttr()):
        super().__init__(points, class_value, value_range, remove_from_reference_points, parent)

    def clone(self, **kw):
        return probability_copy(**kw)

    def probability(self, point):
        return point * self.class_value + (1 - point) * (1 - self.class_value)


def calibrate_lr(point, probability_function_class0, probability_function_class1):
    """
    Calculates a calibrated likelihood ratio (LR).
    
    Parameters
    ----------
    point : float
        The point of interest for which the LR is calculated. This point is
        sampled from either class 0 or class 1.
    points_class0 : list of float
        Points which are sampled from class 0, i.e. the class represented by
        `value_range[0]`.
    points_class1 : list of float
        Points which are sampled from class 1, i.e. the class represented by
        `value_range[1]`.
    value_range : tuple of two floats
        All values of `points_class0`, `points_class1` and `point` must be
        within this range.

    Returns
    -------
    float
        A likelihood ratio, with positive values in favor of class 1.
    """
    return probability_function_class1.probability(point) / probability_function_class0.probability(point)


LrStats = collections.namedtuple('LrStats', ['avg_llr', 'avg_llr_class0', 'avg_llr_class1', 'cllr_class0', 'cllr_class1', 'cllr'])


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

    avg_llr_class0 = avg(*[math.log(1/lr, 2) for lr in lr_class0])
    avg_llr_class1 = avg(*[math.log(lr, 2) for lr in lr_class1])
    avg_llr = avg(avg_llr_class0, avg_llr_class1)

    cllr_class0 = avg(*[math.log(1 + lr, 2) for lr in lr_class0])
    cllr_class1 = avg(*[math.log(1 + 1/lr, 2) for lr in lr_class1])
    cllr = avg(cllr_class0, cllr_class1)

    return LrStats(avg_llr, avg_llr_class0, avg_llr_class1, cllr_class0, cllr_class1, cllr)


def prepare_model(clf, X0_train, X1_train):
    assert X0_train.shape[1] == X1_train.shape[1]
    X_train = np.concatenate([X0_train, X1_train])
    y_train = np.concatenate([np.zeros(X0_train.shape[0]), np.ones(X1_train.shape[0])])

    clf.fit(X_train, y_train)
    return clf


def apply_model(clf, X):
    return list(clf.predict_proba(X)[:,1]) # probability of class 1


def classifier_lr(clf, X0_train, X1_train, X0_calibrate, X1_calibrate, X_disputed, probability_function=probability_fraction):
    """
    Trains a classifier, calibrates the outcome and calculates a LR for a single
    sample.

    All numpy arrays have two dimensions, representing samples and features.

    Parameters
    ----------
    clf : classifier
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
        Test sample of unknown class

    Returns
    -------
    float
        A likelihood ratio for `X_disputed`
    """
    clf, points0, points1 = prepare_model(clf, X0_train, X1_train, X0_calibrate, X1_calibrate)
    point = clf.predict_proba(X_disputed.reshape(1,-1))[0][1] # probability of class 1
    probability_function_class0 = probability_function(points0, 0, [0,1])
    probability_function_class1 = probability_function(points1, 1, [0,1])
    return calibrate_lr(point, probability_function_class0, probability_function_class1)


def calibrated_cllr(probability_function, class0_calibrate, class1_calibrate, class0_test=None, class1_test=None):
    pfunc0 = probability_function(points=class0_calibrate, class_value=0)
    pfunc1 = probability_function(points=class1_calibrate, class_value=1)

    use_calibration_set_for_test = class0_test is None
    if use_calibration_set_for_test:
        class0_test = class0_calibrate
        class1_test = class1_calibrate

    lrs0 = []
    for point in class0_test:
        lrs0.append(calibrate_lr(point, pfunc0(remove_from_reference_points=use_calibration_set_for_test), pfunc1))

    lrs1 = []
    for point in class1_test:
        lrs1.append(calibrate_lr(point, pfunc0, pfunc1(remove_from_reference_points=use_calibration_set_for_test)))

    return calculate_cllr(lrs0, lrs1)


def classifier_cllr(clf, probability_function, X0_train, X1_train, X0_calibrate, X1_calibrate, X0_test=None, X1_test=None):
    """
    Trains a classifier on a training set, calibrates the outcome with a
    calibration set, and calculates a LR (likelihood ratio) for all samples in
    a test set. Calculates a Cllr (log likelihood ratio cost) value for the LR
    values. If no test set is provided, the calibration set is used for both
    calibration and testing.

    All numpy arrays have two dimensions, representing samples and features.

    Parameters
    ----------
    clf : classifier
        A model to be trained. Must support probability output.
    probability_function : function
        A probability function which is used to deterimine the probability of
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
    clf = prepare_model(clf, X0_train, X1_train)
    probability_function = probability_function(value_range=[0,1])

    class0_test = apply_model(clf, X0_test) if X0_test is not None else None
    class1_test = apply_model(clf, X1_test) if X0_test is not None else None

    return calibrated_cllr(probability_function, apply_model(clf, X0_calibrate), apply_model(clf, X1_calibrate), class0_test, class1_test)


def classifier_cllr_kfold(clf, probability_function, n_splits, X0_train, X1_train, X0_test, X1_test):
    probability_function = probability_function(value_range=[0,1])

    kf = sklearn.model_selection.KFold(n_splits=n_splits, shuffle=True)
    class0_calibrate = []
    class1_calibrate = []
    for X0_tc_index, X1_tc_index in zip(kf.split(X0_train), kf.split(X1_train)):
        clf = prepare_model(clf, X0_train[X0_tc_index[0]], X1_train[X1_tc_index[0]])
        class0_calibrate.extend(apply_model(clf, X0_train[X0_tc_index[1]]))
        class1_calibrate.extend(apply_model(clf, X1_train[X1_tc_index[1]]))

    clf = prepare_model(clf, X0_train, X1_train)

    class0_test = apply_model(clf, X0_test)
    class1_test = apply_model(clf, X1_test)

    return calibrated_cllr(probability_function, class0_calibrate, class1_calibrate, class0_test, class1_test)
