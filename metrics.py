import numpy as np
from sklearn.isotonic import IsotonicRegression


def _isotonic_regression(y,
                         weight):
    """
    implementation of isotonic regression
    
    :param y: ordered input values
    :param weight: associated weights
    :return: function values such that the function is non-decreasing and minimises the weighted SSE: ∑ w_i (y_i - f_i)^2
    """

    n = y.shape[0]
    # The algorithm proceeds by iteratively updating the solution
    # array.

    solution = y.copy()

    if n <= 1:
        return solution

    n -= 1
    pooled = 1
    while pooled > 0:
        # repeat until there are no more adjacent violators.
        i = 0
        pooled = 0
        while i < n:
            k = i
            while k < n and solution[k] >= solution[k + 1]:
                k += 1
            if solution[i] != solution[k]:
                # solution[i:k + 1] is a decreasing subsequence, so
                # replace each point in the subsequence with the
                # weighted average of the subsequence.
                numerator = 0.0
                denominator = 0.0
                for j in range(i, k + 1):
                    numerator += solution[j] * weight[j]
                    denominator += weight[j]
                for j in range(i, k + 1):
                    solution[j] = numerator / denominator
                pooled = 1
            i = k + 1
    return solution


def cllr(Hp_LRs, Hd_LRs):
    """
    log likelihood ratio cost

    This is a measure of performance for an LR-generating method. It can be interpreted in an information theoretic manner (this implementation uses log2, hence bits).
    :param Hp_LRs: LRs obtained for a set of H1 examples
    :param Hd_LRs: LRs obtained for a set of H2 examples
    :return: A single non-negative number, lower is better. 0 means perfection: inf for H1 and 0 for H2.
    """
    Np = len(Hp_LRs)
    Nd = len(Hd_LRs)
    return 1.0 / 2.0 * (
        (1.0 / Np * sum([np.log2(1 + 1.0 / LR) for LR in Hp_LRs])) + 1.0 / Nd * sum([np.log2(1 + LR) for LR in Hd_LRs]))


def cllr_min(Hp_scores, Hd_scores, hp_prior=0.5, useSKLearn=False):
    """
    Compute the lowest possible Cllr, a measure of discriminating power

    :param Hp_scores: scores for H1 examples
    :param Hd_scores: scores for H2 examples
    :param Hp_prior: prior for H1 (=1-prior for H2)
    :param useSKLearn: use the PAV implementation from SKLearn. Defaults to False as it appears incorrect when weights and multiple identical values are present. Alternative implementation is used otherwise.
    :return: Cllr after 'optimal' calibration for these scores
    """
    lrs = posterior_to_lr(posterior_from_scores_pav(Hp_scores, Hd_scores, useSKLearn), hp_prior)
    # the Cllr_min is the Cllr after computing lrs using the PAV
    return cllr(lrs[0:len(Hp_scores)], lrs[len(Hp_scores):])


def posterior_to_lr(hp_posteriors, hp_prior=0.5):
    """
    Convert posterior (of H1) to LR

    :param hp_posteriors: vector of posteriors for H1 (\in [0,1])
    :param hp_prior: Prior for H1 (\in [0,1])
    :return: vector of likelihood ratios (\in [0, inf])
    """
    return 1.0 / (1.0 / np.array(hp_posteriors) - 1) * (1 - hp_prior) / hp_prior


def posterior_from_scores_pav(Hp_scores, Hd_scores, useSKLearn=False, weigh_samples=True, normalise=True,
                              add_low_h1_high_h2=False):
    """
    Returns the 'optimal' posterior from the scores

    IsotonicRegression here is solved using Pool Adjacent Violators (PAV). In particular, PAV returns a set of posteriors (hence a function) that minimises
    np.average(weights * ((posteriors - ideal_posterior) ** 2))
    under the constraint that the function is non-decreasing, and weights and ideal_posterior are input vectors. 
    For our purposes ideal_posterior is 1 for H1 and 0 for H2. However, this function is only non-decreasing if the lowest H1 score is higher than the highest H2 score (i.e. the classifier is perfect)
    :param Hp_scores: scores for H1 examples
    :param Hd_scores: scores for H2 examples
    :param useSKLearn: use the PAV implementation from SKLearn. Defaults to False as it appears incorrect when weights and multiple identical values are present. Alternative implementation is used otherwise.
    :param weigh_samples: use weights to ignore the proportion of H1/H2 samples. If set to False, there is an implicit assumption that the proportion of H1 samples is the prior for H1
    :param normalise: scale weights to be in [0,1]
    :param add_low_h1_high_h2: add a lowest score to the H1 and a highest score to the H2 to ensure we don't get 0's and 1's
    :return:
    """
    if add_low_h1_high_h2:
        # keep track of the 'dummy' scores to delete before returning
        to_delete=[]
        # add a lowest score to the H1 and a highest score to the H2 to ensure we don't get 0's and 1's
        if min(Hd_scores) < min(Hp_scores):
            to_delete.append(len(Hp_scores))
            Hp_scores=np.append(Hp_scores, min(Hd_scores))
        if max(Hp_scores) > max(Hd_scores):
            to_delete.append(len(Hp_scores)+len(Hd_scores))
            Hd_scores=np.append(Hd_scores, max(Hp_scores))
    scores = np.hstack((Hp_scores, Hd_scores))
    if normalise:
        scores = scores - np.min(scores)
        scores = scores / np.max(scores)
    nHp = len(Hp_scores)
    nHd = len(Hd_scores)
    ideal_posterior = np.hstack((np.ones((nHp)), np.zeros((nHd))))
    if weigh_samples:
        # adjust for the difference in sample size
        weights = np.hstack(([nHd] * nHp, [nHp] * nHd))
    else:
        weights = np.hstack(([1] * nHp, [1] * nHd))
    if useSKLearn:
        ir = IsotonicRegression()
        hp_posteriors = ir.fit_transform(scores,
                                         ideal_posterior, sample_weight=weights)
    else:
        sor = np.argsort(scores)
        # this being non-parametric, the scores are really just used for sorting
        dhat = _isotonic_regression(np.array(ideal_posterior[sor]), weights[sor])
        # after getting the weights, put them back in the right order
        hp_posteriors = dhat[np.argsort(sor)]
    # get rid of 0's and 1's, which lead to -/+inf in the log LR later
    # there are other tricks for this, e.g. appending an H1 to the bottom and H2 to the top scores (Bruemmer, Preez 2006)
    # impact should however be negligible for our current application of calculating cllr
    if not add_low_h1_high_h2:
        min_val = min(min(scores), min(1 - scores)) / 2
        hp_posteriors[hp_posteriors == 0] = min_val
        hp_posteriors[hp_posteriors == 1] = 1 - min_val
    else:
        # remove dummy scores
        return np.delete(hp_posteriors, to_delete)
    return hp_posteriors


def scores_to_calibrated_lrs(scores_h1, scores_h2):
    """
    given a set of scores/lrs/log lrs calculated on a calibration set, returns the calibrated LRs using
    isotonic regression (PAV)
    """
    return posterior_to_lr(posterior_from_scores_pav(scores_h1, scores_h2, add_low_h1_high_h2=True))


def get_metrics(h1_scores, h2_scores, h1_lrs, h2_lrs, hp_prior, use_sklearn=False):
    performance = cllr(h1_lrs, h2_lrs)
    discriminating_power = cllr_min(h1_scores, h2_scores, hp_prior, use_sklearn)
    calibration = performance - discriminating_power
    return performance, discriminating_power, calibration


def test_cllr_common_sense():
    H1_LRs = [5, 8, 10, 15, 70]
    H2_LRs = [0.2, 0.5, 10]
    assert cllr(H1_LRs, H2_LRs) > 0
    assert cllr(H1_LRs, H2_LRs) < 1
    assert cllr(H2_LRs, H1_LRs) > cllr(H1_LRs, H2_LRs)
    assert cllr([1, 1], [1]) == 1


def test_cllr_min():
    for _ in range(10):
        H1_scores = np.random.rand(np.random.randint(2, 10))
        H2_scores = np.random.rand(np.random.randint(2, 10))
        # note that Cllr takes LRs, not scores. We here just take the mapping score -> LR to be the identity function.
        # Cllr_min should be lower than Cllr for any monotonically increasing mapping (thus also for identity)
        assert cllr_min(H1_scores, H2_scores) < cllr(H1_scores, H2_scores)
        # or for any of these mappings
        for f in [lambda x: x / 0.5, lambda x: x ** 2]:
            assert cllr_min(H1_scores, H2_scores) < cllr([f(x) for x in H1_scores], [f(x) for x in H2_scores])
