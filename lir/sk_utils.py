from sklearn.pipeline import Pipeline

from lir import CalibratedScorer, CalibratedScorerCV


class CalibratedScorerSk(CalibratedScorer):
    """
    As sklearn expects a `predict` in its estimator this class adds the `predict_lr` from the `CalibratedScorer`
    also as `predict`.
    In this way the `CalibratedScorer`, can be easily used in sklearn based pipelnes
    """
    def predict(self, X):
        return super().predict_lr(X)


class CalibratedScorerCVSk(CalibratedScorerCV):
    """
    As sklearn expects a `predict` in its estimator this class adds the `predict_lr` from the `CalibratedScorerCV`
    also as `predict`.
    In this way the `CalibratedScorerCV`, can be easily used in sklearn based pipelnes
    """
    def predict(self, X):
        return super().predict_lr(X)


class LirPipeline(Pipeline):
    """
    As lir can expects a `predict_lr` in its estimator this class adds the `predict` from the `Pipeline`
    also as `predict_lr`.
    In this way the `Pipeline` from sklearn, can be easily used in an lir based pipelnes
    """
    def predict_lr(self, X):
        return super().predict(self, X)
