#!/usr/bin/env python3

import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

import liar


class AbstractCllrEvaluator:
    def __init__(self, name):
        self.name = name

    def __call__(self, class0_train_generator=None, class1_train_generator=None, class0_test_generator=None, class1_test_generator=None, class0_train=None, class1_train=None, class0_test=None, class1_test=None, distribution_mean_delta=None, repeat=1):
        if distribution_mean_delta is not None:
            self._distribution_mean_delta = distribution_mean_delta

        cllr = []
        for run in range(repeat):
            if class0_train_generator is not None:
                class0_train = class0_train_generator()
            if class1_train_generator is not None:
                class1_train = class1_train_generator()
            if class0_test_generator is not None:
                class0_test = class0_test_generator()
            if class1_test_generator is not None:
                class1_test = class1_test_generator()
    
            if class0_test is not None:
                cllr.append(self.cllr(class0_train, class1_train, class0_test, class1_test))

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

    def cllr(self, class0_train, class1_train, class0_test, class1_test):
        # adjust loc1
        if self._distribution_mean_delta is not None:
            self._loc1 = self._loc0 + self._distribution_mean_delta

        # sample from H0
        X0_lr = self._get_lr(class0_test)
        # sample from H1
        X1_lr = self._get_lr(class1_test)

        cllr = liar.calculate_cllr(X0_lr, X1_lr)
        return cllr


class ClassifierCllrEvaluator(AbstractCllrEvaluator):
    def __init__(self, name, clf, probability_function):
        super().__init__(name)

        self._clf = clf
        self._pfunc = probability_function

    def cllr(self, class0_train, class1_train, class0_test, class1_test):
        cllr = liar.classifier_cllr(self._clf, class0_train, class1_train, class0_test, class1_test, probability_function=self._pfunc)
        return cllr


def cllr_average(cllr_lst):
    return sum([d.cllr for d in cllr_lst]) / len(cllr_lst)


def llr_average(cllr_lst):
    return sum([d.avg_llr for d in cllr_lst]) / len(cllr_lst)


def cllr_stdev(cllr_lst):
    return np.std([d.cllr for d in cllr_lst])


def makeplot(xlabel, xvalues, generators, generator_args):
    ax_cllr = plt.subplot(1,3,1)
    plt.xlabel(xlabel)
    plt.ylabel('Cllr')

    ax_llr = plt.subplot(1,3,2)
    plt.xlabel(xlabel)
    plt.ylabel('llr')

    ax_stdev = plt.subplot(1,3,3)
    plt.xlabel(xlabel)
    plt.ylabel('cllr/stdev')

    for g in generators:
        stats = [ g(**genargs) for genargs in generator_args ]
        ax_cllr.plot(xvalues, [ cllr_average(d) for d in stats ], label=g.name)
        ax_llr.plot(xvalues, [ llr_average(d) for d in stats ], label=g.name)
        ax_stdev.plot(xvalues, [ cllr_stdev(d) for d in stats ], label=g.name)

    handles, labels = ax_cllr.get_legend_handles_labels()
    ax_cllr.legend(handles, labels)
    ax_llr.legend(handles, labels)
    ax_stdev.legend(handles, labels)
    plt.show()


def plot_scheidbaarheid():
    xvalues = np.arange(0, 6, 1)
    generator_args = [ {
        'class0_train': np.random.normal(loc=0, scale=1, size=(100, 1)),
        'class1_train': np.random.normal(loc=d, scale=1, size=(100, 1)),
        'class0_test': np.random.normal(loc=0, scale=1, size=(100, 1)),
        'class1_test': lambda: np.random.normal(loc=d, scale=1, size=(100, 1)),
        'distribution_mean_delta': d,
        } for d in xvalues ]

    generators = [
        NormalCllrEvaluator('real lr', 0, 1, 0, 1),
        ClassifierCllrEvaluator('logit', LogisticRegression(), liar.probability_fraction),
        ClassifierCllrEvaluator('logit/cor', LogisticRegression(), liar.probability_copy),
    ]

    makeplot('dx', xvalues, generators, generator_args)


def plot_datasize():
    xvalues = range(2, 12)
    generator_args = []
    for x in xvalues:
        datasize = int(math.pow(2, x))
        generator_args.append({
            'class0_train_generator': lambda: np.random.normal(loc=0, size=(datasize, 1)),
            'class1_train_generator': lambda: np.random.normal(loc=1.5, size=(datasize, 1)),
            'class0_test_generator': lambda: np.random.normal(loc=0, size=(datasize, 1)),
            'class1_test_generator': lambda: np.random.normal(loc=1.5, size=(datasize, 1)),
            'repeat': 10,
        })

    generators = [
        NormalCllrEvaluator('real lr', 0, 1, 1.5, 1),
        ClassifierCllrEvaluator('logit', LogisticRegression(), liar.probability_fraction),
        ClassifierCllrEvaluator('logit/cor', LogisticRegression(), liar.probability_copy),
    ]

    makeplot('dx', xvalues, generators, generator_args)


if __name__ == '__main__':
    #plot_scheidbaarheid()
    plot_datasize()
