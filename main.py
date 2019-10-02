import time
from collections import OrderedDict

import pandas as pd

from components.methods import MLP, SVM, RandomForest, Random, Truth, KDE, GaussianMCMC, NaiveBayes, LogisticRegression
from components.metrics import CLLR, CLLR_min, LRMeanDiff, LRMeanAbsDiff
from components.truth_distributions import Gaussian, Cauchy, BimodalGaussian
from plotting import plot_log_lrs, plot_performance


def evaluate(truth_h1_dist, train_samples_h1, calibration_samples_h1, evaluation_samples_h1,
             truth_h2_dist, train_samples_h2, calibration_samples_h2, evaluation_samples_h2,
             methods=(
                     KDE(), GaussianMCMC(), NaiveBayes(weighting=True), LogisticRegression(weighting=True),
                     RandomForest(weighting=True),
                     SVM(weighting=True),
                     MLP()),
             evaluations=(CLLR(), LRMeanDiff(), LRMeanAbsDiff(), CLLR_min())
             ):
    eval_scores = OrderedDict()
    oracle = Truth(truth_h1_dist, truth_h2_dist)
    methods = list(methods)
    methods.append(oracle)
    methods.append(Random())
    true_log_lrs_h1 = oracle.get_log_lrs(evaluation_samples_h1)
    true_log_lrs_h2 = oracle.get_log_lrs(evaluation_samples_h2)

    # save all lr predictions
    uncalibrated_log_lrs = OrderedDict()
    calibrated_log_lrs = OrderedDict()
    train_log_lrs = OrderedDict()

    for method in methods:
        start = time.time()
        method.fit(train_samples_h1, train_samples_h2)
        mid = time.time()
        log_lrs_h1 = method.get_log_lrs(evaluation_samples_h1)
        log_lrs_h2 = method.get_log_lrs(evaluation_samples_h2)

        train_log_lrs[method.name()] = [method.get_log_lrs(train_samples_h1),
                                        method.get_log_lrs(train_samples_h2)]

        method.calibrate(calibration_samples_h1, calibration_samples_h2)

        log_lrs_h1_calibrated = method.get_log_lrs(evaluation_samples_h1, calibrated=True)
        log_lrs_h2_calibrated = method.get_log_lrs(evaluation_samples_h2, calibrated=True)

        end = time.time()
        eval_scores[method.name()] = {}
        for evaluation in evaluations:
            eval_scores[method.name()][evaluation.name()] = evaluation.evaluate(
                log_lrs_h1, log_lrs_h2, true_log_lrs_h1,
                true_log_lrs_h2)
            eval_scores[method.name()][evaluation.name() + '_calibrated'] = evaluation.evaluate(
                log_lrs_h1_calibrated, log_lrs_h2_calibrated, true_log_lrs_h1,
                true_log_lrs_h2)
        eval_scores[method.name()]['training time'] = mid - start
        eval_scores[method.name()]['evaluation time'] = end - mid
        uncalibrated_log_lrs[method.name()] = [log_lrs_h1, log_lrs_h2]
        calibrated_log_lrs[method.name()] = [log_lrs_h1_calibrated, log_lrs_h2_calibrated]
    return eval_scores, uncalibrated_log_lrs, calibrated_log_lrs, train_log_lrs


if __name__ == '__main__':
    n_dimensions = (2, 4, 10, 20, 50, 100)
    # allowed to give one int if sample sizes for h1 and h2 are equal
    n_samples_h1_h2 = (10, 100, 1000)
    n_calibration = 100
    n_evaluation = 1000
    scores = dict()
    for n_dimension in n_dimensions:
        for n_sample in n_samples_h1_h2:
            if type(n_sample) == int:
                # same sample sizes
                n_sample = (n_sample, n_sample)
            # define distributions
            truth_h1_dist = Gaussian(n_dimensions=n_dimension, mean=0, sigma=1, cov=0)
            truth_h2_dist = Gaussian(n_dimensions=n_dimension, mean=2, sigma=1, cov=0)
            # truth_h1_dist = BimodalGaussian(n_dimensions=n_dimension, means=[0, 4], sigma=1, cov=0)
            # truth_h2_dist = BimodalGaussian(n_dimensions=n_dimension, means=[-2, 2], sigma=1, cov=0)
            # truth_h2_dist = Cauchy(n_dimension, 2, 1/100, 0)

            # generate data
            samples_h1 = truth_h1_dist.sample(n_sample[0])
            samples_h2 = truth_h2_dist.sample(n_sample[1])
            evaluation_samples_h1 = truth_h1_dist.sample(n_evaluation)
            evaluation_samples_h2 = truth_h2_dist.sample(n_evaluation)
            calibration_samples_h1 = truth_h1_dist.sample(n_calibration)
            calibration_samples_h2 = truth_h2_dist.sample(n_calibration)

            print('evaluating dim {}, n {}'.format(n_dimension, n_sample))
            tuple_value = (n_dimension, n_sample)
            scores[tuple_value], uncalibrated_log_lrs, calibrated_log_lrs, train_log_lrs = \
                evaluate(truth_h1_dist, samples_h1, calibration_samples_h1, evaluation_samples_h1,
                         truth_h2_dist, samples_h2, calibration_samples_h2, evaluation_samples_h2)

            print(pd.DataFrame(scores[tuple_value]))

            # TODO could do better than first two dimensions
            plot_log_lrs(uncalibrated_log_lrs, calibrated_log_lrs, train_log_lrs, tuple_value,
                         [evaluation_samples_h1[:, :2], evaluation_samples_h2[:, :2]],
                         [samples_h1[:, :2], samples_h2[:, :2]],
                         fig_prefix=truth_h1_dist.name() + '___' + truth_h2_dist.name())

    plot_performance(scores, truth_h1_dist.name() + '___' + truth_h2_dist.name())
