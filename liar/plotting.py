import math

import numpy as np
import matplotlib.pyplot as plt

from . import *


class AbstractCllrEvaluator:
    def __init__(self, name):
        self.name = name

    def __call__(self, x,
                 class0_train=None, class1_train=None, class0_calibrate=None, class1_calibrate=None, class0_test=None, class1_test=None, distribution_mean_delta=None,
                 train_folds=None, train_reuse=False, repeat=1):
        if distribution_mean_delta is not None:
            self._distribution_mean_delta = distribution_mean_delta

        resolve = lambda value: value if value is None or not callable(value) else value(x)

        cllr = []
        for run in range(repeat):
            if class0_train is not None and train_folds is not None:
                cllr.append(self.cllr_kfold(train_folds, resolve(class0_train), resolve(class1_train), resolve(class0_test), resolve(class1_test)))
            elif class0_calibrate is not None:
                cllr.append(self.cllr(resolve(class0_train), resolve(class1_train), resolve(class0_calibrate), resolve(class1_calibrate), resolve(class0_test), resolve(class1_test)))
            elif class0_train is not None and train_reuse:
                class0_train_instance = resolve(class0_train)
                class1_train_instance = resolve(class1_train)
                cllr.append(self.cllr(class0_train_instance, class1_train_instance, class0_train_instance, class1_train_instance, resolve(class0_test), resolve(class1_test)))

        return cllr


class NormalCllrEvaluator(AbstractCllrEvaluator):
    def __init__(self, name, loc0, scale0, loc1, scale1):
        super().__init__(name)

        self._loc0 = loc0
        self._scale0 = scale0
        self._loc1 = loc1
        self._scale1 = scale1
        self._distribution_mean_delta = None

    def _get_probability(X, mu, sigma):
        return np.exp(-np.power(X - mu, 2) / (2*sigma*sigma)) / math.sqrt(2*math.pi*sigma*sigma)

    def _get_lr(self, X):
        # calculate P(E|H0)
        X_p0 = NormalCllrEvaluator._get_probability(X, self._loc0, self._scale0)
        # calculate P(E|H1)
        X_p1 = NormalCllrEvaluator._get_probability(X, self._loc1, self._scale1)
        # calculate LR
        return X_p1 / X_p0

    def cllr_kfold(self, n_splits, X0_train, X1_train, X0_test, X1_test):
        return self.calculate_cllr(X0_test, X1_test)

    def cllr(self, class0_train, class1_train, class0_calibrate, class1_calibrate, class0_test, class1_test):
        return self.calculate_cllr(class0_test, class1_test)

    def calculate_cllr(self, class0_test, class1_test):
        assert class0_test.shape[1] == 1

        # adjust loc1
        if self._distribution_mean_delta is not None:
            self._loc1 = self._loc0 + self._distribution_mean_delta

        # sample from H0
        X0_lr = self._get_lr(class0_test)
        # sample from H1
        X1_lr = self._get_lr(class1_test)

        cllr = calculate_cllr(X0_lr, X1_lr)
        return cllr


class ClassifierCllrEvaluator(AbstractCllrEvaluator):
    def __init__(self, name, clf, probability_function):
        super().__init__(name)

        self._clf = clf
        self._pfunc = probability_function

    def cllr_kfold(self, n_splits, X0_train, X1_train, X0_test, X1_test):
        cllr = classifier_cllr_kfold(self._clf, self._pfunc, n_splits, X0_train, X1_train, X0_test, X1_test)
        return cllr

    def cllr(self, class0_train, class1_train, class0_calibrate, class1_calibrate, class0_test, class1_test):
        cllr = classifier_cllr(self._clf, self._pfunc, class0_train, class1_train, class0_calibrate, class1_calibrate, class0_test, class1_test)
        return cllr


def cllr_average(cllr_lst):
    return sum([d.cllr for d in cllr_lst]) / len(cllr_lst)


def cllr_stdev(cllr_lst):
    return np.std([d.cllr for d in cllr_lst])


def llr_average(cllr_lst):
    return sum([d.avg_llr_class0 for d in cllr_lst]) / len(cllr_lst)


def llr_stdev(cllr_lst):
    return np.std([d.avg_llr_class0 for d in cllr_lst])


def makeplot(xlabel, generators, experiments):
    ax_cllr = None

    xvalues, _ = zip(*experiments)

    for g in generators:
        stats = [ g(x=x, **genargs) for x, genargs in experiments ]

        if ax_cllr is None:
            if len(stats[0]) == 1:
                nrows = 2
                ax_cstd = None
            else:
                nrows = 4
                ax_cstd = plt.subplot(nrows, 1, 2)
                plt.ylabel('std(C_llr)')

                ax_lrstd = plt.subplot(nrows, 1, 4)
                plt.ylabel('std(2log(lr_h0))')

                plt.xlabel(xlabel)

            ax_cllr = plt.subplot(nrows, 1, 1)
            plt.ylabel('C_llr')

            ax_llr = plt.subplot(nrows, 1, 2+int(ax_cstd is not None))
            plt.ylabel('2log(lr_h0)')

            if nrows == 2:
                plt.xlabel(xlabel)

        cllrplot = ax_cllr.plot(xvalues, [ cllr_average(d) for d in stats ], 'o--', label=g.name)[0]
        ax_cllr.plot(xvalues, [ (cllr_average(d)-cllr_stdev(d), cllr_average(d)+cllr_stdev(d)) for d in stats ], '_', color=cllrplot.get_color())

        llrplot = ax_llr.plot(xvalues, [ llr_average(d) for d in stats ], 'o--', label=g.name)[0]
        ax_llr.plot(xvalues, [ (llr_average(d)-llr_stdev(d), llr_average(d)+llr_stdev(d)) for d in stats ], '_', color=llrplot.get_color())

        if ax_cstd is not None:
            ax_cstd.plot(xvalues, [ cllr_stdev(d) for d in stats ], 'o--', label=g.name)
            ax_lrstd.plot(xvalues, [ llr_stdev(d) for d in stats ], 'o--', label=g.name)

    handles, labels = ax_cllr.get_legend_handles_labels()
    ax_cllr.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, 1))
    plt.show()
