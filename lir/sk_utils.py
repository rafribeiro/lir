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
