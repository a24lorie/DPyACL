import unittest

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from dpyacl.strategies.single_label.error_reduction import QueryRegressionStd
from dpyacl.core.stop_criteria import MaxIteration
from dpyacl.experiment.context import HoldOutExperiment
from dpyacl.metrics.evaluation import Mse
from dpyacl.oracle import SimulatedOracleQueryIndex
from dpyacl.scenario.scenario import PoolBasedSamplingScenario
from dpyacl.core.misc.misc import split


class TestRegression(unittest.TestCase):
    # logging.basicConfig(level=logging.DEBUG)

    # Get the data
    __X = np.random.choice(np.linspace(0, 20, 10000), size=200, replace=False).reshape(-1, 1)
    __y = np.sin(__X) + np.random.normal(scale=0.3, size=__X.shape)

    # assembling initial training set
    __train_idx, __test_idx, __label_idx, __unlabel_idx = split(
        X=__X,
        y=__y,
        test_ratio=0.3,
        initial_label_rate=0.05,
        split_count=1,
        all_class=True)

    # defining the kernel for the Gaussian process
    __ml_technique = GaussianProcessRegressor(
        kernel=RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
               + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1)))

    def test_One_iteration(self):

        experiment = HoldOutExperiment(
            self.__X,
            self.__y,
            scenario_type=PoolBasedSamplingScenario,
            train_idx=self.__train_idx,
            test_idx=self.__test_idx,
            label_idx=self.__label_idx,
            unlabel_idx=self.__unlabel_idx,
            ml_technique=self.__ml_technique,
            performance_metrics=[Mse(squared=True)],
            query_strategy=QueryRegressionStd(),
            oracle=SimulatedOracleQueryIndex(labels=self.__y),
            stopping_criteria=MaxIteration(1),
            self_partition=False
        )

        result = experiment.evaluate(verbose=False)
        regressor = result[0].ml_technique

        # plotting the initial estimation
        with plt.style.context('seaborn-white'):
            plt.figure(figsize=(14, 7))
            x = np.linspace(0, 20, 1000)
            pred, std = regressor.predict(x.reshape(-1, 1), return_std=True)
            plt.plot(x, pred)
            plt.fill_between(x, pred.reshape(-1, ) - std, pred.reshape(-1, ) + std, alpha=0.2)
            plt.scatter(self.__X, self.__y, c='k')
            plt.title('Initial estimation')
            plt.show()

    def test_fifteen_iteration(self):

        experiment = HoldOutExperiment(
            self.__X,
            self.__y,
            scenario_type=PoolBasedSamplingScenario,
            train_idx=self.__train_idx,
            test_idx=self.__test_idx,
            label_idx=self.__label_idx,
            unlabel_idx=self.__unlabel_idx,
            ml_technique=self.__ml_technique,
            performance_metrics=[Mse(squared=True)],
            query_strategy=QueryRegressionStd(),
            oracle=SimulatedOracleQueryIndex(labels=self.__y),
            stopping_criteria=MaxIteration(15),
            self_partition=False
        )

        result = experiment.evaluate(verbose=False)
        regressor = result[0].ml_technique

        # plotting the initial estimation
        with plt.style.context('seaborn-white'):
            plt.figure(figsize=(14, 7))
            x = np.linspace(0, 20, 1000)
            pred, std = regressor.predict(x.reshape(-1, 1), return_std=True)
            plt.plot(x, pred)
            plt.fill_between(x, pred.reshape(-1, ) - std, pred.reshape(-1, ) + std, alpha=0.2)
            plt.scatter(self.__X, self.__y, c='k')
            plt.title('Initial estimation')
            plt.show()

if __name__ == '__main__':
    unittest.main()
