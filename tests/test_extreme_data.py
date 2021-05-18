import unittest
from math import floor, ceil
from typing import Tuple
import numpy as np


def create_test_data(length: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create test data in the range [0, 1] of size length.
    Contains one 0 and one 1. Also contains one number very close to 1.
    """
    np.random.seed(42)
    X = np.concatenate([np.random.beta(7, 3, size=(floor(length/2), 1)),
                        np.random.beta(3, 7, size=(ceil(length/2), 1))])
    y = np.concatenate([np.zeros(floor(length/2)), np.ones(ceil(length/2))])
    # add a zero and one score:
    X[0] = 1
    X[length-1] = 0
    if length > 2:
        # add a score close to 1:
        X[length-2] = 1-10**-16
    return X, y


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
