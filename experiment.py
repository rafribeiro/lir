from functools import partial

from scipy.optimize import minimize
import numpy as np

from lir import to_log_odds


def four_pl(s, a, b, c, d):
    return c + ((1 - c) / (1 + d)) * 1 / (1 + np.exp(-a * s - b))


def negative_log_likelihood_balanced(X, y, model, params):
    probs = model(X, *params)
    try:
        v = -np.sum(np.log(probs ** y * (1 - probs) ** (1 - y)) / (y * np.sum(y) + (1 - y) * np.sum(1 - y)))
        return v
    except Exception as e:
        print(v)
        raise e


def optimize_callback(result):
    print(result)


class FourPL:
    def __int__(self):
        self.coef_ = None

    def fit(self, X, y):
        # check for negative inf
        X = to_log_odds(X)
        estimate_c = np.any(np.isneginf(X[y == 1]))
        estimate_d = np.any(np.isposinf(X[y == 0]))
        bounds = [(-np.inf, np.inf), (-np.inf, np.inf)]

        if estimate_c and estimate_d:
            self.model = four_pl
            bounds.extend([(0, 1), (0, np.inf)])
        elif estimate_c:
            self.model = partial(four_pl, d=0)
            bounds.append((10**-10, 1-10**-10))
        elif estimate_d:
            self.model = partial(four_pl, c=0)
            bounds.append((0, np.inf))
        else:
            self.model = partial(four_pl, c=0, d=0)

        f = partial(negative_log_likelihood_balanced, X, y, self.model)

        result = minimize(f, np.array([.1] * (2 + estimate_d + estimate_c)),
                          bounds=bounds, callback=optimize_callback)
        assert result.success
        self.coef_ = result.x

    def predict_proba(self, X):
        proba = self.model(X, *self.coef_)
        result = np.stack([1-proba, proba], axis=1)
        assert result.shape[0] == proba.shape[0]
        return result


# functie uitschrijven

# optimizer aanroepen
