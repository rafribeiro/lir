import logging
import numpy as np
from scipy import stats
from sklearn import datasets


class Gaussian:
    def __init__(self, n_dimensions, mean, sigma, cov=0):
        self.n_dimensions = n_dimensions
        self.mean = mean
        self.sigma = sigma
        self.cov = cov
        if self.cov == 0:
            # independent
            self.dist = stats.multivariate_normal([self.mean] * self.n_dimensions, [self.sigma] * self.n_dimensions)
        else:
            cov_matrix = np.identity(self.n_dimensions) * (self.sigma - self.cov) + self.cov * np.ones(
                (self.n_dimensions, self.n_dimensions))
            self.dist = stats.multivariate_normal([self.mean] * self.n_dimensions, cov_matrix)

    def pdf(self, x):
        return self.dist.pdf(x)

    def sample(self, n):
        return self.dist.rvs(n)

    def name(self):
        return "Gaussian_dim_{}_mu_{}_sigma_{}_cov_{}".format(self.n_dimensions, self.mean, self.sigma, self.cov)


class Cauchy:
    def __init__(self, n_dimensions, mean, scale, cov=0):
        # "mean"
        self.n_dimensions = n_dimensions
        self.mean = mean
        self.scale = scale
        self.cov = cov
        if self.cov == 0:
            # independent
            self.dist = stats.cauchy([self.mean] * self.n_dimensions, [self.scale] * self.n_dimensions)
        else:
            raise NotImplementedError

    def pdf(self, x):
        return np.sum(self.dist.pdf(x), axis=1)

    def sample(self, n):
        return self.dist.rvs((n, self.n_dimensions))

    def name(self):
        return "Cauchy_dim_{}_mu_{}_scale_{}_cov_{}".format(self.n_dimensions, self.mean, self.scale, self.cov)


class BimodalGaussian:
    def __init__(self, n_dimensions, means, sigma, cov=0):
        self.n_dimensions = n_dimensions
        self.means = means
        self.sigma = sigma
        self.cov = cov

        models = []
        for mean in self.means:
            models.append(stats.multivariate_normal([mean] * self.n_dimensions, [self.sigma] * self.n_dimensions))
        self.model = MixtureModel(models, [1 / len(self.means)] * len(self.means))

    def sample(self, size):
        return np.array(self.model.sample(size))

    def pdf(self, x):
        return self.model.pdf(x)

    def name(self):
        return "bimodalGaussian_dim_{}_mus_{}_sigma_{}_cov_{}".format(self.n_dimensions, self.means, self.sigma, self.cov)


class MixtureModel():
    def __init__(self, submodels, weights):
        self.submodels = submodels
        self.weights = weights
        if len(submodels) != len(weights):
            raise ValueError('should be as many models as weights!')
        if np.sum(weights) != 1:
            logging.info('normalising weights {}'.format(weights))
            self.weights = [w / np.sum(weights) for w in weights]

    def pdf(self, x):
        pdf = self.submodels[0].pdf(x) * self.weights[0]
        for i, submodel in enumerate(self.submodels[1:]):
            pdf += submodel.pdf(x) * self.weights[i + 1]
        return pdf

    def sample(self, size):
        models_chosen = np.random.choice(self.submodels, p=self.weights, size=size)
        return [m.rvs(1) for m in models_chosen]


def multi_variate_gaussian_random_dependent_same_params(n_dimensions, mean, sigma):
    # TODO more sensible sigma usage? now normalising trace
    cov = datasets.make_spd_matrix(n_dimensions)
    cov = cov / np.trace(cov) * sigma * len(cov)
    return stats.multivariate_normal([mean] * n_dimensions, cov)