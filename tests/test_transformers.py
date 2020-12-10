#!/usr/bin/env python3

import numpy as np
import unittest
import warnings

from context import lir

from lir.transformers import InstancePairing


warnings.simplefilter("error")


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
