import numpy as np
import pymc3 as pm
from scipy import stats
from sklearn import neighbors, linear_model, naive_bayes, neural_network, svm, ensemble
from metrics import scores_to_calibrated_lrs


class BaseMethod:
    def fit(self, samples_h1, samples_h2):
        pass

    def get_log_lrs(self, sample, calibrated=False):
        if calibrated:
            uncalibrated_log_lrs = self.get_uncalibrated_log_lrs(sample)
            return np.array([self.calibrate_log_lr(ll) for ll in uncalibrated_log_lrs])
        else:
            return self.get_uncalibrated_log_lrs(sample)

    def calibrate(self, samples_h1, samples_h2):
        scores_h1 = self.get_log_lrs(samples_h1)
        scores_h2 = self.get_log_lrs(samples_h2)

        # save the sorted scores and accompanying LRs, as a score->LR mapping
        self.calibration_scores = np.hstack((scores_h1, scores_h2))
        sor = np.argsort(self.calibration_scores)
        self.calibrated_log_lrs = np.log(scores_to_calibrated_lrs(scores_h1, scores_h2)[sor])
        self.calibration_scores = self.calibration_scores[sor]

    def calibrate_log_lr(self, log_lr):
        if log_lr < min(self.calibration_scores):
            return self.calibrated_log_lrs[0]
        if log_lr >= max(self.calibration_scores):
            return self.calibrated_log_lrs[-1]
        for i, cal_score in enumerate(self.calibration_scores):
            if log_lr < cal_score:
                # simple linear interpolation
                alpha = (log_lr - self.calibration_scores[i - 1]) / (cal_score - self.calibration_scores[i - 1])
                if not (0 <= alpha < 1) and not np.isnan(alpha):
                    raise ValueError('issue with calibration scores, alpha of {}'.format(alpha))
                return self.calibrated_log_lrs[i] * alpha + self.calibrated_log_lrs[i - 1] * (1 - alpha)

    def name(self):
        raise NotImplementedError


class KDE(BaseMethod):
    def __init__(self):
        super(KDE, self).__init__()
        self.h1 = neighbors.KernelDensity(leaf_size=1000)
        self.h2 = neighbors.KernelDensity(leaf_size=1000)

    def fit(self, samples_h1, samples_h2):
        self.h1.fit(samples_h1)
        self.h2.fit(samples_h2)

    def get_uncalibrated_log_lrs(self, sample):
        return self.h1.score_samples(sample) - self.h2.score_samples(sample)

    def name(self):
        return 'KDE'


class GaussianMCMC(BaseMethod):
    def __init__(self):
        super(GaussianMCMC, self).__init__()
        self.h1_sigmas = 1
        self.h2_sigmas = 1

    def fit(self, samples_h1, samples_h2):
        n_h1, d = samples_h1.shape
        n_h2, _ = samples_h2.shape

        # define a model. now assuming independence and known sd
        with pm.Model() as sb_model:
            # sd_dist = pm.HalfCauchy.dist(beta=2.5, shape=3)
            # chol_packed = pm.LKJCholeskyCov('chol_packed',
            #     n=3, eta=2, sd_dist=sd_dist)
            # chol = pm.expand_packed_triangular(3, chol_packed)
            # vals = pm.MvNormal('vals', mu=mu, chol=chol, observed=data)
            # mu_h1 = pm.Uniform('mu_h1', lower=np.min(samples_h1), upper=np.max(samples_h1), shape=d)
            # mu_h2 = pm.Uniform('mu_h2', lower=np.min(samples_h2), upper=np.max(samples_h2), shape=d)
            mu_h1 = pm.Normal('mu_h1', mu=0, sd=1, shape=d)
            mu_h2 = pm.Normal('mu_h2', mu=0, sd=1, shape=d)
            obs_h1 = pm.Normal('obs', mu=mu_h1, sd=self.h1_sigmas, observed=samples_h1)
            obs_h2 = pm.Normal('obs_h2', mu=mu_h2, sd=self.h2_sigmas, observed=samples_h2)

        # sample using MCMC
        with sb_model:
            # Draw wamples
            trace = pm.sample(500, njobs=3)

        # point estimates
        self.h1_mus = np.median(trace['mu_h1'], axis=0)
        self.h2_mus = np.median(trace['mu_h2'], axis=0)

    def get_uncalibrated_log_lrs(self, sample):
        d = sample.shape[1]
        return stats.multivariate_normal(self.h1_mus, np.identity(d) * self.h1_sigmas).logpdf(sample) - \
               stats.multivariate_normal(self.h2_mus, np.identity(d) * self.h2_sigmas).logpdf(sample)

    def name(self):
        return 'MCMC'


class LogisticRegression(BaseMethod):
    def __init__(self, weighting):
        super(LogisticRegression, self).__init__()
        self.model = linear_model.LogisticRegression()
        self.weighting = weighting

    def fit(self, samples_h1, samples_h2):
        if self.weighting:
            self.model.fit(np.concatenate((samples_h1, samples_h2)), [0] * len(samples_h1) + [1] * len(samples_h2),
                           sample_weight=[len(samples_h2)] * len(samples_h1) + [len(samples_h1)] * len(samples_h2))
        else:
            self.model.fit(np.concatenate((samples_h1, samples_h2)), [0] * len(samples_h1) + [1] * len(samples_h2))

    def get_uncalibrated_log_lrs(self, sample):
        probas = self.model.predict_proba(sample)
        return np.log(probas[:, 0]) - np.log(probas[:, 1])

    def name(self):
        return 'log reg ' + ('w' if self.weighting else 'nw')


class NaiveBayes(BaseMethod):
    def __init__(self, weighting):
        super(NaiveBayes, self).__init__()
        self.model = naive_bayes.GaussianNB()
        self.weighting = weighting

    def fit(self, samples_h1, samples_h2):
        if self.weighting:
            self.model.fit(np.concatenate((samples_h1, samples_h2)), [0] * len(samples_h1) + [1] * len(samples_h2),
                           sample_weight=[len(samples_h2)] * len(samples_h1) + [len(samples_h1)] * len(samples_h2))
        else:
            self.model.fit(np.concatenate((samples_h1, samples_h2)), [0] * len(samples_h1) + [1] * len(samples_h2))

    def get_uncalibrated_log_lrs(self, sample):
        probas = self.model.predict_proba(sample)
        return np.log(probas[:, 0]) - np.log(probas[:, 1])

    def name(self):
        return 'NB ' + ('w' if self.weighting else 'nw')


class MLP(BaseMethod):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = neural_network.MLPClassifier()

    def fit(self, samples_h1, samples_h2):
        self.model.fit(np.concatenate((samples_h1, samples_h2)), [0] * len(samples_h1) + [1] * len(samples_h2))

    def get_uncalibrated_log_lrs(self, sample):
        probas = self.model.predict_proba(sample)
        return np.log(probas[:, 0]) - np.log(probas[:, 1])

    def name(self):
        return 'MLP'


class SVM(BaseMethod):
    def __init__(self, weighting):
        super(SVM, self).__init__()
        self.model = svm.SVC(probability=True)
        self.weighting = weighting

    def fit(self, samples_h1, samples_h2):
        if self.weighting:
            self.model.fit(np.concatenate((samples_h1, samples_h2)), [0] * len(samples_h1) + [1] * len(samples_h2),
                           sample_weight=[len(samples_h2)] * len(samples_h1) + [len(samples_h1)] * len(samples_h2))
        else:
            self.model.fit(np.concatenate((samples_h1, samples_h2)), [0] * len(samples_h1) + [1] * len(samples_h2))

    def get_uncalibrated_log_lrs(self, sample):
        probas = self.model.predict_proba(sample)
        return np.log(probas[:, 0]) - np.log(probas[:, 1])

    def name(self):
        return 'SVM ' + ('w' if self.weighting else 'nw')


class RandomForest(BaseMethod):
    def __init__(self, weighting):
        super(RandomForest, self).__init__()
        self.model = ensemble.RandomForestClassifier()
        self.weighting = weighting

    def fit(self, samples_h1, samples_h2):
        if self.weighting:
            self.model.fit(np.concatenate((samples_h1, samples_h2)), [0] * len(samples_h1) + [1] * len(samples_h2),
                           sample_weight=[len(samples_h2)] * len(samples_h1) + [len(samples_h1)] * len(samples_h2))
        else:
            self.model.fit(np.concatenate((samples_h1, samples_h2)), [0] * len(samples_h1) + [1] * len(samples_h2))

    def get_uncalibrated_log_lrs(self, sample):
        probas = self.model.predict_proba(sample)
        # TODO less rough machine precision solution
        probas = (probas - .5) * .99999999 + .5

        return np.log(probas[:, 0]) - np.log(probas[:, 1])

    def name(self):
        return 'RF ' + ('w' if self.weighting else 'nw')


class Random(BaseMethod):
    def get_log_lrs(self, sample, calibrated=False):
        return [0] * len(sample)
        # return stats.norm.rvs(0, 1, len(sample))

    def name(self):
        return 'random'


class Truth(BaseMethod):
    def __init__(self, truth_h1_dist, truth_h2_dist):
        super(Truth, self).__init__()
        self.h1 = truth_h1_dist
        self.h2 = truth_h2_dist

    def calibrate(self, samples_h1, samples_h2):
        pass

    def get_log_lrs(self, sample, calibrated=False):
        return np.log(self.h1.pdf(sample)) - np.log(self.h2.pdf(sample))

    def name(self):
        return 'Oracle'
