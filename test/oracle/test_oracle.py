"""
Test functions of simOracle modules
"""

from __future__ import division

import random
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.datasets import load_iris

from dpyacl.oracle import SimulatedOracle, ConsoleHumanOracle


class TestOracle(unittest.TestCase):

    X, y = load_iris(return_X_y=True)
    X = X[0:100, ]
    o = y[0:50]

    simOracle = SimulatedOracle(labels=o)
    conHumOracle = ConsoleHumanOracle(labels=o)

    def test_SimulatedOracle(self):
        # Query the simOracle by the labeled indexes
        for i in range(10):
            r = random.randrange(0, 50)
            test, _ = self.simOracle.query(instances=None, indexes=r)
            assert test == self.y[r]

        # Add new knowledge to the simOracle and the query by it index
        for i in range(10):
            self.simOracle.add_knowledge(self.y[50 + i], 50 + i)
            test, _ = self.simOracle.query(instances=None, indexes=50 + i)
            assert test == self.y[50 + i]

        for i in range(5):
            knowl = self.y[(60 + 10 * i): (70 + 10 * i)]
            label = [(60 + 10 * i + j) for j in range(10)]
            assert len(knowl) == len(label)
            self.simOracle.add_knowledge(knowl, label)

        for i in range(50):
            test, _ = self.simOracle.query(instances=None, indexes=60 + i)
            assert test == self.y[60 + i]

    def test_HumanSimulatedOracle(self):

        # Query the simOracle by the labeled indexes
        expected = np.asarray([0, 1, 1, 1, 0])
        # https://dev.to/vergeev/how-to-test-input-processing-in-python-3
        with unittest.mock.patch('builtins.input', side_effect=['L', 0, 'L', 1, 'L', 1, 'L', 1, 'L', 0]):
            for i in range(5):
                r = random.randrange(51, 100)
                test, _ = self.conHumOracle.query(instances=[self.X[r]], indexes=r)
                assert test == expected[i]


if __name__ == '__main__':
    unittest.main()
