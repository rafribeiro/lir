from functools import partial
from scipy.optimize import minimize
import numpy as np
from lir import to_log_odds


class Bind(partial):
    """
    An improved version of partial which accepts Ellipsis (...) as a placeholder.
    Can be used to fix parameters not at the end of the list of parameters (which is a limitation of partial).
    """

    def __call__(self, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        iargs = iter(args)
        args = (next(iargs) if arg is ... else arg for arg in self.args)
        return self.func(*args, *iargs, **keywords)


def four_parameter_logistic_model(s, a, b, c, d):
    """
    inputs:
            s: n * 1 np.array of scores
            a,b,c,d,: floats defining 4PL model.
                a and b are the familiar logistic parameters.
                c and d respectively floor and ceil the posterior probability
    output:
            p: n * 1 np.array. Posterior probabilities of succes given each s (and a,b,c,d)
    """
    p = c + ((1 - c) / (1 + d)) * 1 / (1 + np.exp(-a * s - b))
    return p


def negative_log_likelihood_balanced(X, y, model, params):
    """
    calculates neg_llh of probabiliistic binary classifier.
    The llh is balanced in the sense that the total weight of '1'-labels is equal to the total weight of '0'-labels.

    inputs:
        X: n * 1 np.array of scores
        y: n * 1 np.array of labels (Booleans). H1 --> 1, H2 --> 0.
        model: model that links score to posterior probabilities
        params: parameters of the model. They can be fixed or varied to get ML-estimators in def fit in class FourParameterLogisticCalibrator

    output:
        neg_llh_balanced: float, balanced negative log likelihood (base = exp)
    """

    probs = model(X, *params)
    neg_llh_balanced = -np.sum(np.log(probs ** y * (1 - probs) ** (1 - y)) / (y * np.sum(y) + (1 - y) * np.sum(1 - y)))
    return neg_llh_balanced


class FourParameterLogisticCalibrator:
    """
    Calculates a likelihood ratio of a score value, provided it is from one of two distributions.
    Depending on the training data, a 2-, 3- or 4-parameter logistic model is used.
    """
    def __int__(self):
        self.coef_ = None

    def fit(self, X, y):
        # check for negative inf for '1'-labels or inf for '0'-labels
        X = to_log_odds(X)
        estimate_c = np.any(np.isneginf(X[y == 1]))
        estimate_d = np.any(np.isposinf(X[y == 0]))

        # define bounds for a and b
        bounds = [(-np.inf, np.inf), (-np.inf, np.inf)]

        if estimate_c and estimate_d:
            # then define 4PL-logistic model
            self.model = four_parameter_logistic_model
            bounds.extend([(10**-10, 1-10**-10), (10**-10, np.inf)])
        elif estimate_c:
            # then define 3-PL logistic model. Set 'd' to 0
            self.model = partial(four_parameter_logistic_model, d=0)
            # use very small values since limits result in -inf llh
            bounds.append((10**-10, 1-10**-10))
        elif estimate_d:
            # then define 3-PL logistic model. Set 'c' to 0
            # use bind since 'c' is intermediate variable. In that case partial does not work.
            self.model = Bind(four_parameter_logistic_model, ..., ..., ..., 0, ...)
            # use very small value since limits result in -inf llh
            bounds.append((10**-10, np.inf))
        else:
            # define ordinary logistic model (no regularization, so maximum likelihood estimates)
            self.model = partial(four_parameter_logistic_model, c=0, d=0)
        # define function to minimize
        objective_function = partial(negative_log_likelihood_balanced, X, y, self.model)

        result = minimize(objective_function, np.array([.1] * (2 + estimate_d + estimate_c)),
                          bounds=bounds)
        assert result.success
        self.coef_ = result.x

    def predict_proba(self, X):
        X = to_log_odds(X)
        proba = self.model(X, *self.coef_)
        result = np.stack([1-proba, proba], axis=1)
        return result