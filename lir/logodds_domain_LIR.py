import numpy as np
import pandas as pd
#from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from lir import metrics
import lir.plotting
from lir import ece
import lir
import lir.util
import math as math
import warnings as warnings
from typing import Optional, Tuple, Union
from lir.util import Xy_to_Xn
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture

# specify paths and filenames
pathin = 'P:\FSM\Advies_R&D\MCMC_in_Python\data\input\\'
# pathout = 'Y:\WISK_STAT\Advies_R&D\MCMC_in_Python\\resultaten\\'
mu_s = 11#1, 6, 11, 17
filename_log_LRs_H1 = 'LRssamenormalLLRdistribmu_s=' + str(mu_s) + 'N_ss=300.csv'
filename_log_LRs_H2 = 'LRsdifferentnormalLLRdistribmu_s=' + str(mu_s) + 'N_ss=300.csv'
pathfile_data_H1 = pathin + filename_log_LRs_H1
pathfile_data_H2 = pathin + filename_log_LRs_H2


## train part
# reading data file
df_H1 = pd.read_csv(pathfile_data_H1, header = [0])
df_H2 = pd.read_csv(pathfile_data_H2, header = [0])
# convert to np.array
np_H1 = np.array(df_H1.iloc[1:10, 1])
np_H2 = np.array(df_H2.iloc[1:10, 1])


# convert to LRs probability domain
np_H1_prob = lir.util.to_probability(np_H1)
np_H2_prob = lir.util.to_probability(np_H2)


# test: making pathological data, adding 0 under H1 and 1 under H2
# np_H1_prob = np.append(np_H1_prob, np.zeros(1))
# np_H2_prob = np.append(np_H2_prob, np.ones(1))

# test: making difficult data, adding 0 and nearly 0 under H2, 1 and nearly 1 under H1
# np_H1_prob = np.append(np_H1_prob, np.ones(1))
# np_H1_prob = np.append(np_H1_prob, 1 - np.float_power(10, -16))
# np_H1_prob = np.append(np_H1_prob, 1 - np.float_power(10, -324))
# np_H2_prob = np.append(np_H2_prob, np.zeros(1))
# np_H2_prob = np.append(np_H2_prob, np.float_power(10, -323)) # dit gaat nog net
# np_H2_prob = np.append(np_H2_prob, np.float_power(10, -324)) # dit gaat net niet meer

# concatenate the scores
scores = np.append(np_H2_prob,np_H1_prob)
print("scores")
print(scores)
exit(0)


# generate GTs
Y_train = np.concatenate(( np.zeros(len(np_H2_prob))), np.ones(len(np_H1_prob)))

#for Gaussian and KDE-calibrator fitting: remove negInf, Inf and compensate
def compensate_and_remove_negInf_Inf(log_odds_X, y):
        el_H1 = np.all(np.array([log_odds_X != np.Inf, y == 1]), axis=0)
        el_H2 = np.all(np.array([log_odds_X != -1 * np.Inf, y == 0]), axis=0)
        n_H1 = np.sum(y)
        numerator = np.sum(el_H1)/n_H1
        denominator = np.sum(el_H2)/(len(y)-n_H1)
        y = y[np.any(np.array([el_H1, el_H2]), axis=0)]
        log_odds_X = log_odds_X[np.any(np.array([el_H1, el_H2]), axis=0)]
        return log_odds_X, y, numerator, denominator


def to_log_odds(X):
    np.seterr(divide='ignore')
    complement = np.add(1, np.multiply(-1, X))
    log_odds = np.add(np.log10(X), np.multiply(-1, np.log10(complement)))
    np.seterr(divide='warn')
    return(log_odds)

#for calirbation training on log_odds domain. Check whether negInf under H1 and Inf under H2 occurs and give error if so
def check_misleading_Inf_negInf(log_odds_X, y):
    # sanity checks
    # give error message if H1's contain zeros and H2's contain ones
    if np.any(np.isneginf(log_odds_X[y == 1])) and np.any(np.isposinf(log_odds_X[y == 0])):
        raise ValueError('Your data is possibly problematic for this calibration type. You have negInf under H1 and Inf under H2 after logodds transform. If you really want to proceed, transform probs in order to get finite values on the logodds domain')
    # give error message if H1's contain zeros
    if np.any(np.isneginf(log_odds_X[y == 1])):
        raise ValueError('Your data is possibly problematic for this calibration type. You have negInf under H1 after logodds transform. If you really want to proceed, transform probs in order to get finite values on the logodds domain')
    # give error message if H2's contain ones
    if np.any(np.isposinf(log_odds_X[y == 0])):
        raise ValueError('Your data is possibly problematic for this calibration type. You have Inf under H2 after logodds transform. If you really want to proceed, transform probs in order to get finite values on the logodds domain')


def ln_to_log(ln_data):
    log_data = np.multiply(np.log10(np.exp(1)), ln_data)
    return(log_data)



class KDECalibrator(BaseEstimator, TransformerMixin):
    """
    Calculates a likelihood ratio of a score value, provided it is from one of
    two distributions. Uses kernel density estimation (KDE) for interpolation.
    """

    def __init__(self, bandwidth: Optional[Union[float, Tuple[Optional[float], Optional[float]]]] = None, to_log_odds=True):
        """

        :param bandwidth:
            * If None is provided the Silverman's rule of thumb is
            used to calculate the bandwidth for both distributions (independently)
            * If a single float is provided this is used as the bandwith for both
            distributions
            * If a tuple is provided, the first entry is used for the bandwidth
            of the first distribution (kde0) and the second entry for the second
            distribution (if value is None: Silverman's rule of thumb is used)
        """
        self.bandwidth: Tuple[Optional[float], Optional[float]] = \
            self._parse_bandwidth(bandwidth)
        self._kde0: Optional[KernelDensity] = None
        self._kde1: Optional[KernelDensity] = None
        self.to_log_odds = to_log_odds
    @staticmethod
    def bandwidth_silverman(X):
        """
        Estimates the optimal bandwidth parameter using Silverman's rule of
        thumb.
        """
        assert len(X) > 0

        std = np.std(X)
        if std == 0:
            # can happen eg if std(X) = 0
            warnings.warn('silverman bandwidth cannot be calculated if standard deviation is 0', RuntimeWarning)
            LOG.info('found a silverman bandwidth of 0 (using dummy value)')
            std = 1

        v = math.pow(std, 5) / len(X) * 4. / 3
        return math.pow(v, .2)

    @staticmethod
    def bandwidth_scott(X):
        """
        Not implemented.
        """
        raise

    def fit(self, X, y):
        if self.to_log_odds:

            #transform to logodds
            self.X = to_log_odds(X)

            # check if data is sane
            check_misleading_Inf_negInf(self.X,y)

            # KDE needs finite scale. Inf and negInf are treated as point masses at the extremes.
            # Remove them from data for KDE and calculate fraction data that is left.
            # LRs in finite range will be corrected for fractions in transform function
            X, y, self.numerator, self.denominator = compensate_and_remove_negInf_Inf(self.X,y)
        X0, X1 = Xy_to_Xn(X, y)
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)

        bandwidth0 = self.bandwidth[0] or self.bandwidth_silverman(X0)
        bandwidth1 = self.bandwidth[1] or self.bandwidth_silverman(X1)

        self._kde0 = KernelDensity(kernel='gaussian', bandwidth=bandwidth0).fit(X0)
        self._kde1 = KernelDensity(kernel='gaussian', bandwidth=bandwidth1).fit(X1)
        return self

    def transform(self, X):
        assert self._kde0 is not None, "KDECalibrator.transform() called before fit"
        self.p0 = np.empty(np.shape(X))
        self.p1 = np.empty(np.shape(X))
        if self.to_log_odds:
            # initiate LRs_output
            LLRs_output = np.empty(np.shape(X))

            # transform probs to log_odds
            X = to_log_odds(X)

            # get inf and neginf
            wh_inf = np.isposinf(X)
            wh_neginf = np.isneginf(X)

            # assign hard values for extremes
            LLRs_output[wh_inf] = np.inf
            LLRs_output[wh_neginf] = -1 * np.inf
            self.p0[wh_inf] = 0
            self.p1[wh_inf] = 1
            self.p0[wh_neginf] = 1
            self.p1[wh_neginf] = 0

            # get elements that are not inf or neginf
            el = np.all(np.array([np.isposinf(X) == False, np.isneginf(X) == False]), axis=0)
            X = X[el]

        # perform KDE as usual
        X = X.reshape(-1, 1)
        ln_H1 = self._kde1.score_samples(X)
        ln_H2 = self._kde0.score_samples(X)
        ln_dif = np.add(ln_H1, np.multiply(-1, ln_H2))
        log10_dif = ln_to_log(ln_dif)

        #calculate p0 and p1's (redundant?)
        if self.to_log_odds:
            self.p0[el] = np.multiply(self.denominator, np.exp(ln_H2))
            self.p1[el] = np.multiply(self.numerator, np.exp(ln_H1))

            # apply correction for fraction of negInf and Inf data
            log10_compensator = np.add(np.log10(self.numerator), np.multiply(-1, np.log(self.denominator)))
            LLRs_output[el] = np.add(log10_compensator, log10_dif)
            return np.float_power(10, LLRs_output)
        else:
            self.p0 = np.exp(ln_H2)
            self.p1 = np.exp(ln_H1)
            return np.float_power(10, log10_dif)

    @staticmethod
    def _parse_bandwidth(bandwidth: Optional[Union[float, Tuple[float, float]]]) \
            -> Tuple[Optional[float], Optional[float]]:
        """
        Returns bandwidth as a tuple of two (optional) floats.
        Extrapolates a single bandwidth
        :param bandwidth: provided bandwidth
        :return: bandwidth used for kde0, bandwidth used for kde1
        """
        if bandwidth is None:
            return None, None
        elif isinstance(bandwidth, float):
            return bandwidth, bandwidth
        elif len(bandwidth) == 2:
            return bandwidth
        else:
            raise ValueError('Invalid input for bandwidth')


class GaussianCalibrator(BaseEstimator, TransformerMixin):
    """
    Calculates a likelihood ratio of a score value, provided it is from one of
    two distributions. Uses a gaussian mixture model for interpolation.
    """
    def __init__(self, to_log_odds=True, n_components_H0=1, n_components_H1=1):
        self.to_log_odds = to_log_odds
        self.n_components_H1 = n_components_H1
        self.n_components_H0 = n_components_H0

    def fit(self, X, y):
        if self.to_log_odds:

            #transform probs to logodds
            self.X = to_log_odds(X)

            #check whether training data is sane
            check_misleading_Inf_negInf(self.X,y)

            # Gaussian mixture needs finite scale. Inf and negInf are treated as point masses at the extremes.
            # Remove them from data for Gaussian mixture and calculate fraction data that is left.
            # LRs in finite range will be corrected for fractions in transform function
            X, y, self.numerator, self.denominator = compensate_and_remove_negInf_Inf(self.X,y)

        # perform Gaussian mixture as usual
        X0, X1 = Xy_to_Xn(X, y)
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        self._model0 = GaussianMixture(n_components=self.n_components_H0).fit(X0)
        self._model1 = GaussianMixture(n_components=self.n_components_H1).fit(X1)
        return self

    def transform(self, X):
        self.p0 = np.empty(np.shape(X))
        self.p1 = np.empty(np.shape(X))
        if self.to_log_odds:
            # initiate LLRs_output
            LLRs_output = np.empty(np.shape(X))

            # transform probs to log_odds
            X = to_log_odds(X)

            # get inf and neginf
            wh_inf = np.isposinf(X)
            wh_neginf = np.isneginf(X)

            # assign hard values for extremes
            LLRs_output[wh_inf] = np.inf
            LLRs_output[wh_neginf] = -1 * np.inf
            self.p0[wh_inf] = 0
            self.p1[wh_inf] = 1
            self.p0[wh_neginf] = 1
            self.p1[wh_neginf] = 0

            # get elements that are not inf or neginf
            el = np.all(np.array([np.isposinf(X) == False, np.isneginf(X) == False]), axis=0)
            X = X[el]

        #perform density calculations for X as usual
        X = X.reshape(-1, 1)
        ln_H1 = self._model1.score_samples(X)
        ln_H2 = self._model0.score_samples(X)
        ln_dif = np.add(ln_H1, np.multiply(-1, ln_H2))
        log10_dif = ln_to_log(ln_dif)

        # calculation of p0 and p1's redundant?
        if self.to_log_odds:
            self.p0[el] = np.multiply(self.denominator, np.exp(ln_H2))
            self.p1[el] = np.multiply(self.numerator, np.exp(ln_H1))

            #apply correction for fraction of Infs and negInfs
            log10_compensator = np.add(np.log10(self.numerator), np.multiply(-1, np.log(self.denominator)))
            LLRs_output[el] = np.add(log10_compensator, log10_dif)
            return np.float_power(10, LLRs_output)
        else:
            self.p0 = np.exp(ln_H2)
            self.p1 = np.exp(ln_H1)
            return np.float_power(10, log10_dif)




class LogitCalibrator(BaseEstimator, TransformerMixin):
    """
    Calculates a likelihood ratio of a score value, provided it is from one of
    two distributions. Uses logistic regression for interpolation.
    """

    def __init__(self, to_log_odds=True):
        self.to_log_odds = to_log_odds

    def fit(self, X, y):
        if self.to_log_odds:
            # sanity check
            X = to_log_odds(X)
            check_misleading_Inf_negInf(X, y)

            # if data is sane, remove Inf under H1 and minInf under H2 from the data if present (if present, these prevent logistic regression to train while the loss is zero, so they can be safely removed)
            el_H1 = np.all(np.array([np.isposinf(X) == False, y == 1]), axis=0)
            el_H2 = np.all(np.array([np.isneginf(X) == False, y == 0]), axis=0)
            y = y[np.any(np.array([el_H1, el_H2]), axis=0)]
            X = X[np.any(np.array([el_H1, el_H2]), axis=0)]

        # train logistic regression
        X = X.reshape(-1, 1)
        self._logit = LogisticRegression(class_weight='balanced')
        self._logit.fit(X, y)
        return self

    def transform(self, X):
        if self.to_log_odds:

            # initiate LLRs_output
            LLRs_output = np.empty(np.shape(X))

            # transform probs to log_odds
            X = to_log_odds(X)

            # get boundary log_odds values
            zero_elements = np.where(X == -1 * np.inf)
            ones_elements = np.where(X == np.inf)

            # assign desired output for these boundary values to LLRs_output
            LLRs_output[zero_elements] = np.multiply(-1, np.inf)
            LLRs_output[ones_elements] = np.inf

            # get elements with values between negInf and Inf (the boundary values)
            between_elements = np.all(np.array([X != np.inf, X != -1 * np.inf]), axis=0)

            # get LLRs for X[between_elements]
            LnLRs = np.add(self._logit.intercept_, np.multiply(self._logit.coef_, X[between_elements]))
            LLRs = ln_to_log(LnLRs)
            LLRs = np.reshape(LLRs, np.sum(between_elements))
            LLRs_output[between_elements] = LLRs

            # calculation of self.p1 and self.p0 is redundant?
            LRs = np.float_power(10, LLRs_output)
            self.p1 = np.divide(LRs, np.add(1, LRs))
            self.p0 = np.divide(1, np.add(1, LRs))
            return np.float_power(10, LLRs_output)
        else:
            # calculation of self.p1 and self.p0 is redundant?
            self.p1 = self._logit.predict_proba(X.reshape(-1, 1))[:, 1]  # probability of class 1
            self.p0 = (1 - self.p1)

            # get LLRs for X
            LnLRs = np.add(self._logit.intercept_, np.multiply(self._logit.coef_, X))
            LLRs = ln_to_log(LnLRs)
            LLRs = LLRs.reshape(len(X))
            return np.float_power(10, LLRs)






# train part KDE
# initialize a calibrator
calibrator = KDECalibrator()  # use KDE for calibration
# calibrator = KDECalibrator(to_log_odds = False)  # use KDE for calibration

# train part Gaussion Mixture model
#calibrator = GaussianCalibrator(n_components_H0=4, n_components_H1=2)
#calibrator = GaussianCalibrator()
#calibrator = GaussianCalibrator(to_log_odds = False)


# # train part logreg
# # initialize a calibrator
#calibrator = LogitCalibrator()  # use logreg for calibration
#calibrator = LogitCalibrator(to_log_odds=False)


# transform scores?
#scores = to_log_odds(scores)
# calibrator fitten
calibrator.fit(scores, Y_train)



# test part (on train data)
lrs_cal_train = calibrator.transform(scores)
print(lrs_cal_train)

# plot to check
f = plt.figure()
plt.plot(to_log_odds(scores), np.log10(lrs_cal_train), 'o', color='black')
#plt.plot(scores, np.log10(lrs_cal_train), 'o', color='black')
x = np.linspace(-10,10,100)
plt.ylim((-10, 10))
plt.plot(x, x)
plt.title("ziet er goed uit")
plt.xlabel("LLRstest")
plt.ylabel("LLRstest cal")
# file =
# pathfile = pathout + file
plt.show()
