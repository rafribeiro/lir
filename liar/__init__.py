import math

import numpy as np


def probability_fraction(point, reference_points, class_value, value_range=[0,1]):
    """
    Calculates the probability of the distance of `point` to `class_value`,
    provided it is from the same distribution as `reference_points`.

    Parameters
    ----------
    point : float
        The point of interest for which the probability is calculated.
    reference_points : list of float
        Points from the same distribution as `point`.
    class_value : float
        The class from which `reference_points` are sampled; this should be
        either `value_range[0]` or `value_range[1]`.
    value_range : tuple of two floats
        All values of `reference_points` and `point` must be within this
        range.

    Returns
    -------
    float
        A probability value.
    """
    point = abs(class_value - point)
    reference_points = sorted([abs(class_value - p) for p in reference_points] + [point])
    return float(len(reference_points) - reference_points.index(point)) / len(reference_points)


def calibrate_lr(point, reference_points_left, reference_points_right, probability_density_function, value_range=[0,1]):
    """
    Calculates a calibrated likelihood ratio (LR).
    
    Parameters
    ----------
    point : float
        The point of interest for which the LR is calculated. This point is
        sampled from the same distribution of either `reference_points_left`
        or `reference_points_right`.
    reference_points_left : list of float
        Points which are sampled from the class represented by
        `value_range[0]`.
    reference_points_right : list of float
        Points which are sampled from the class represented by
        `value_range[1]`.
    value_range : tuple of two floats
        All values of `reference_points_left`, `reference_points_right` and
        `point` must be within this range.

    Returns
    -------
    float
        A likelihood ratio.
    """
    return probability_density_function(point, reference_points_right, value_range[1], value_range) / probability_density_function(point, reference_points_left, value_range[0], value_range)


def calculate_cllr(lr_defense, lr_prosecutor):
    """
    Calculates a log likelihood ratio cost (C_llr) for a series of likelihood
    ratios.

    Parameters
    ----------
    lr_defense : list of float
        Likelihood ratios which are calculated for measurements which are
        sampled from the defence's hypothesis class.
    lr_prosecutor : list of float
        Likelihood ratios which are calculated for measurements which are
        sampled from the prosecutor's hypothesis class.

    Returns
    -------
    float
        A log likelihood ratio cost value.
    """
    def avg(lst):
        return sum(lst) / len(lst)

    cllr_defense = avg([math.log(1 + lr, 2) for lr in lr_defense])
    cllr_prosecutor = avg([math.log(1 + 1/lr, 2) for lr in lr_prosecutor])
    return .5 * cllr_defense + .5 * cllr_prosecutor


def prepare_model(clf, X0_train, X1_train, X0_calibrate, X1_calibrate):
    assert X0_train.shape[1] == X1_train.shape[1]
    X_train = np.concatenate([X0_train, X1_train])
    y_train = np.concatenate([np.zeros(X0_train.shape[0]), np.ones(X1_train.shape[0])])

    clf.fit(X_train, y_train)
    points0 = clf.predict_proba(X0_calibrate)[:,1] # probability of class 1
    points1 = clf.predict_proba(X1_calibrate)[:,1] # probability of class 1
    return clf, list(points0), list(points1)


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
        Training set for hypothesis 0 (defense's hypothesis)
    X1_train : numpy array
        Training set for hypothesis 1 (prosecutor's hypothesis)
    X0_calibrate : numpy array
        Calibration set for hypothesis 0 (defense's hypothesis)
    X1_calibrate : numpy array
        Calibration set for hypothesis 1 (prosecutor's hypothesis)
    X_disputed : numpy array
        Test sample of unknown class

    Returns
    -------
    float
        A likelihood ratio for `X_disputed`
    """
    clf, points0, points1 = prepare_model(clf, X0_train, X1_train, X0_calibrate, X1_calibrate)
    point = clf.predict_proba(X_disputed.reshape(1,-1))[0][1] # probability of class 1
    return calibrate_lr(point, points0, points1, probability_function)


def classifier_cllr(clf, X0_train, X1_train, X0_calibrate, X1_calibrate, probability_function=probability_fraction):
    """
    Trains a classifier, calibrates the outcome and calculates a LR for all
    samples.

    All numpy arrays have two dimensions, representing samples and features.

    Parameters
    ----------
    clf : classifier
        A model to be trained. Must support probability output.
    X0_train : numpy array
        Training set for hypothesis 0 (defense's hypothesis)
    X1_train : numpy array
        Training set for hypothesis 1 (prosecutor's hypothesis)
    X0_calibrate : numpy array
        Calibration set for hypothesis 0 (defense's hypothesis)
    X1_calibrate : numpy array
        Calibration set for hypothesis 1 (prosecutor's hypothesis)
    X_test : numpy array
        Test sample of unknown class

    Returns
    -------
    float
        A likelihood ratio for `X_test`
    """
    clf, points0, points1 = prepare_model(clf, X0_train, X1_train, X0_calibrate, X1_calibrate)
    lr0 = [ calibrate_lr(points0[i], points0[:i] + points0[i+1:], points1, probability_function) for i in range(len(points0)) ]
    lr1 = [ calibrate_lr(points1[i], points0, points1[:i] + points1[i+1:], probability_function) for i in range(len(points1)) ]
    return calculate_cllr(lr0, lr1)
