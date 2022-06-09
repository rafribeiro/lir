import numpy as np

def negative_log_likelihood_balanced(X, y, model, params):
    """
    calculates neg_llh of probabilistic binary classifier.
    The llh is balanced in the sense that the total weight of '1'-labels is equal to the total weight of '0'-labels.

    inputs:
        X: n * 1 np.array of scores
        y: n * 1 np.array of labels (Booleans). H1 --> 1, H2 --> 0.
        model: model that links score to posterior probabilities
        params: mapping of parameter names to values of the model.
    output:
        neg_llh_balanced: float, balanced negative log likelihood (base = exp)
    """

    probs = model(X, *params)
    neg_llh_balanced = -np.sum(np.log(probs ** y * (1 - probs) ** (1 - y)) / (y * np.sum(y) + (1 - y) * np.sum(1 - y)))
    return neg_llh_balanced