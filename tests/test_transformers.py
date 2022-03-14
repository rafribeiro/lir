#!/usr/bin/env python3

import numpy as np
import unittest
import warnings

from scipy.stats import rankdata

from context import lir

from lir.transformers import InstancePairing, RankTransformer

warnings.simplefilter("error")


class TestRankTransformer(unittest.TestCase):
    def test_fit_transform(self):
        """When X itself is transformed, it should give it's own ranks"""
        X = np.array([[0.1, 0.4, 0.5],
                      [0.2, 0.5, 0.55],
                      [0.15, 0.51, 0.55],
                      [0.18, 0.45, 0.56]])
        rank_transformer = RankTransformer()
        rank_transformer.fit(X)
        ranks = rank_transformer.transform(X)
        self.assertSequenceEqual(ranks.tolist(), (rankdata(X, axis=0)/len(X)).tolist(),
                                 'Ranking X and RankTransform X should give the same results')

    def test_extrapolation(self):
        """Values smaller than the lowest value should map to 0,
        values larger than the highest value should map to 1"""
        X = np.array([[0.1, 0.2, 0.3],
                      [0.2, 0.2, 0.4],
                      [0.3, 0.2, 0.5]])
        Z = np.array([[0.0, 0.1, 0.2],
                      [1.0, 1.0, 1.0]])
        rank_transformer = RankTransformer()
        rank_transformer.fit(X)
        ranks = rank_transformer.transform(Z)
        self.assertSequenceEqual(ranks.tolist(),
                                 np.array([[0, 0, 0], [1, 1, 1]]).tolist(),
                                 'Elements smaller than lowest value should map'
                                 ' to 0, larger than highest value to 1')


class TestPairing(unittest.TestCase):
    def test_pairing(self):
        X = np.arange(30).reshape(10, 3)
        y = np.concatenate([np.arange(5), np.arange(5)])

        pairing = InstancePairing()
        X_pairs, y_pairs = pairing.transform(X, y)

        self.assertEqual(np.sum(y_pairs == 1), 5, 'number of same source pairs')
        self.assertEqual(np.sum(y_pairs == 0), 2*(8+6+4+2), 'number of different source pairs')

        pairing = InstancePairing(different_source_limit='balanced')
        X_pairs, y_pairs = pairing.transform(X, y)

        self.assertEqual(np.sum(y_pairs == 1), 5, 'number of same source pairs')
        self.assertEqual(np.sum(y_pairs == 0), 5, 'number of different source pairs')

        self.assertTrue(np.all(pairing.pairing[:,0] != pairing.pairing[:,1]), 'identity in pairs')
        

if __name__ == '__main__':
    unittest.main()
