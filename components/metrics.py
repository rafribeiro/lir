import numpy as np

import metrics


class CLLR:
    def evaluate(self, log_lrs_h1, log_lrs_h2, true_log_lrs_h1, true_log_lrs_h2):
        return metrics.cllr(np.exp(log_lrs_h1), np.exp(log_lrs_h2))

    def name(self):
        return 'Cllr'


class CLLR_min:
    def evaluate(self, log_lrs_h1, log_lrs_h2, true_log_lrs_h1, true_log_lrs_h2):
        return metrics.cllr_min(np.exp(log_lrs_h1), np.exp(log_lrs_h2))

    def name(self):
        return 'Cllr_min'


class LRMeanAbsDiff:
    def evaluate(self, log_lrs_h1, log_lrs_h2, true_log_lrs_h1, true_log_lrs_h2):
        return np.mean(
            [np.mean(np.abs(log_lrs_h1 - true_log_lrs_h1)), np.mean(np.abs(log_lrs_h2 - true_log_lrs_h2))]) / (
                   np.mean(np.abs(np.concatenate((true_log_lrs_h1, true_log_lrs_h2)))))

    def name(self):
        return 'mean log LR absolute difference (norm)'


class LRMeanDiff:
    def evaluate(self, log_lrs_h1, log_lrs_h2, true_log_lrs_h1, true_log_lrs_h2):
        return np.mean([np.mean(log_lrs_h1 - true_log_lrs_h1), np.mean(log_lrs_h2 - true_log_lrs_h2)])

    def name(self):
        return 'mean log LR difference'
