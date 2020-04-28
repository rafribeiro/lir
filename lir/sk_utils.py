from sklearn.pipeline import Pipeline

from lir import CalibratedScorer, CalibratedScorerCV


class LirPipeline(Pipeline):
    """
    As lir can expects a `predict_lr` in its estimator this class adds the `predict` from the `Pipeline`
    also as `predict_lr`.
    In this way the `Pipeline` from sklearn, can be easily used in an lir based pipelnes

    Parameters
    ----------
    X : iterable
        Data to predict on. Must fulfill input requirements of first step
        of the pipeline.

    Returns
    -------
    y_pred : array-like
        Likelihood-ratios for the data in X.
    """
    def predict_lr(self, X):
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict_lr(Xt)

    def _validate_steps(self):
        names, estimators = zip(*self.steps)

        # validate names
        self._validate_names(names)

        # validate estimators
        transformers = estimators[:-1]
        estimator = estimators[-1]

        for t in transformers:
            if t is None or t == 'passthrough':
                continue
            if (not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not
                    hasattr(t, "transform")):
                raise TypeError("All intermediate steps should be "
                                "transformers and implement fit and transform "
                                "or be the string 'passthrough' "
                                "'%s' (type %s) doesn't" % (t, type(t)))

        # We allow last estimator to be None as an identity transformation
        if (estimator is not None and estimator != 'passthrough'
                and not hasattr(estimator, "fit")):
            raise TypeError(
                "Last step of Pipeline should implement fit "
                "or be the string 'passthrough'. "
                "'%s' (type %s) doesn't" % (estimator, type(estimator)))

        if (type(estimator) is not CalibratedScorer):
            raise TypeError(
                "LirPipeline should only be used with CalibratedScorer."
                "Used the default Pipeline instead"
            )
