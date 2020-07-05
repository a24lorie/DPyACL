import time
import unittest

import matplotlib.pyplot as plt
import numpy as np
from distributed import Client
from sklearn.datasets import make_classification
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import LogisticRegression

from dpyacl.core.stop_criteria import MaxIteration
from dpyacl.experiment import ExperimentAnalyserFactory
from dpyacl.experiment.context import HoldOutExperiment
from dpyacl.metrics import Accuracy, Mse, Recall, ZeroOneLoss, F1, HammingLoss, RocAuc, Precision
from dpyacl.oracle import SimulatedOracleQueryIndex
from dpyacl.scenario.scenario import PoolBasedSamplingScenario
from dpyacl.strategies.single_label import QueryInstanceRandom


class TestMetrics(unittest.TestCase):

    # Get the data
    __X_class, __y_class = make_classification(n_samples=100, n_features=10, n_informative=2, n_redundant=2,
                                   n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None,
                                   flip_y=0.01,
                                   class_sep=1.0,
                                   hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)

    # Get the data
    __X_reg = np.random.choice(np.linspace(0, 20, 1000), size=100, replace=False).reshape(-1, 1)
    __y_reg = np.sin(__X_reg) + np.random.normal(scale=0.3, size=__X_reg.shape)

    __ml_technique_class = LogisticRegression(solver='sag')
    __ml_technique_reg = GaussianProcessRegressor(
            kernel=RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
                   + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1)))

    __query_strategy = QueryInstanceRandom()

    __client = Client("tcp://192.168.2.100:8786")

    def test_accuracy(self):
        performance_metrics = [Accuracy()]

        experiment = HoldOutExperiment(
            self.__X_class,
            self.__y_class,
            scenario_type=PoolBasedSamplingScenario,
            ml_technique=self.__ml_technique_class,
            performance_metrics=performance_metrics,
            query_strategy=self.__query_strategy,
            oracle=SimulatedOracleQueryIndex(labels=self.__y_class),
            stopping_criteria=MaxIteration(value=10),
            self_partition=True,
            test_ratio=0.3,
            initial_label_rate=0.05,
            all_class=True
        )

        start_time = time.time()
        result = experiment.evaluate(client=self.__client, verbose=True)
        print()
        print("---Active Learning experiment %s seconds ---" % (time.time() - start_time))

        query_analyser = ExperimentAnalyserFactory.experiment_analyser(
            performance_metrics=[metric.metric_name for metric in performance_metrics],
            method_name=self.__query_strategy.query_function_name,
            method_results=result,
            type="queries"
        )

        # get a brief description of the experiment
        query_analyser.plot_learning_curves(title='Active Learning experiment results')

    def test_mse(self):
        performance_metrics = [Mse(squared=False)]

        experiment = HoldOutExperiment(
            self.__X_reg,
            self.__y_reg,
            scenario_type=PoolBasedSamplingScenario,
            ml_technique=self.__ml_technique_reg,
            performance_metrics=performance_metrics,
            query_strategy=self.__query_strategy,
            oracle=SimulatedOracleQueryIndex(labels=self.__y_reg),
            stopping_criteria=MaxIteration(value=20),
            self_partition=True,
            test_ratio=0.3,
            initial_label_rate=0.05,
            all_class=True
        )

        start_time = time.time()
        result = experiment.evaluate(client=self.__client, verbose=True)
        print()
        print("---Active Learning experiment %s seconds ---" % (time.time() - start_time))

        query_analyser = ExperimentAnalyserFactory.experiment_analyser(
            performance_metrics=[metric.metric_name for metric in performance_metrics],
            method_name=self.__query_strategy.query_function_name,
            method_results=result,
            type="queries"
        )

        # get a brief description of the experiment
        query_analyser.plot_learning_curves(title='Active Learning experiment results')

        result = experiment.evaluate(verbose=True)
        regressor = result[0].ml_technique

        # plotting the initial estimation
        with plt.style.context('seaborn-white'):
            plt.figure(figsize=(14, 7))
            x = np.linspace(0, 20, 1000)
            pred, std = regressor.predict(x.reshape(-1, 1), return_std=True)
            plt.plot(x, pred)
            plt.fill_between(x, pred.reshape(-1, ) - std, pred.reshape(-1, ) + std, alpha=0.2)
            plt.scatter(self.__X_reg, self.__y_reg, c='k')
            plt.title('Initial estimation')
            plt.show()

    def test_rmse(self):
        performance_metrics = [Mse(squared=True)]

        experiment = HoldOutExperiment(
            self.__X_reg,
            self.__y_reg,
            scenario_type=PoolBasedSamplingScenario,
            ml_technique=self.__ml_technique_reg,
            performance_metrics=performance_metrics,
            query_strategy=self.__query_strategy,
            oracle=SimulatedOracleQueryIndex(labels=self.__y_reg),
            stopping_criteria=MaxIteration(value=10),
            self_partition=True,
            test_ratio=0.3,
            initial_label_rate=0.05,
            all_class=True
        )

        start_time = time.time()
        result = experiment.evaluate(client=self.__client, verbose=True)
        print()
        print("---Active Learning experiment %s seconds ---" % (time.time() - start_time))

        query_analyser = ExperimentAnalyserFactory.experiment_analyser(
            performance_metrics=[metric.metric_name for metric in performance_metrics],
            method_name=self.__query_strategy.query_function_name,
            method_results=result,
            type="queries"
        )

        # get a brief description of the experiment
        query_analyser.plot_learning_curves(title='Active Learning experiment results')

        result = experiment.evaluate(verbose=True)
        regressor = result[0].ml_technique

        # plotting the initial estimation
        with plt.style.context('seaborn-white'):
            plt.figure(figsize=(14, 7))
            x = np.linspace(0, 20, 1000)
            pred, std = regressor.predict(x.reshape(-1, 1), return_std=True)
            plt.plot(x, pred)
            plt.fill_between(x, pred.reshape(-1, ) - std, pred.reshape(-1, ) + std, alpha=0.2)
            plt.scatter(self.__X_reg, self.__y_reg, c='k')
            plt.title('Initial estimation')
            plt.show()

    def test_zero_one_loss(self):
        performance_metrics = [ZeroOneLoss()]

        experiment = HoldOutExperiment(
            self.__X_class,
            self.__y_class,
            scenario_type=PoolBasedSamplingScenario,
            ml_technique=self.__ml_technique_class,
            performance_metrics=performance_metrics,
            query_strategy=self.__query_strategy,
            oracle=SimulatedOracleQueryIndex(labels=self.__y_class),
            stopping_criteria=MaxIteration(value=10),
            self_partition=True,
            test_ratio=0.3,
            initial_label_rate=0.05,
            all_class=True
        )

        start_time = time.time()
        result = experiment.evaluate(client=self.__client, verbose=True)
        print()
        print("---Active Learning experiment %s seconds ---" % (time.time() - start_time))

        query_analyser = ExperimentAnalyserFactory.experiment_analyser(
            performance_metrics=[metric.metric_name for metric in performance_metrics],
            method_name=self.__query_strategy.query_function_name,
            method_results=result,
            type="queries"
        )

        # get a brief description of the experiment
        query_analyser.plot_learning_curves(title='Active Learning experiment results')

    def test_f1(self):
        performance_metrics = [F1()]

        experiment = HoldOutExperiment(
            self.__X_class,
            self.__y_class,
            scenario_type=PoolBasedSamplingScenario,
            ml_technique=self.__ml_technique_class,
            performance_metrics=performance_metrics,
            query_strategy=self.__query_strategy,
            oracle=SimulatedOracleQueryIndex(labels=self.__y_class),
            stopping_criteria=MaxIteration(value=10),
            self_partition=True,
            test_ratio=0.3,
            initial_label_rate=0.05,
            all_class=True
        )

        start_time = time.time()
        result = experiment.evaluate(client=self.__client, verbose=True)
        print()
        print("---Active Learning experiment %s seconds ---" % (time.time() - start_time))

        query_analyser = ExperimentAnalyserFactory.experiment_analyser(
            performance_metrics=[metric.metric_name for metric in performance_metrics],
            method_name=self.__query_strategy.query_function_name,
            method_results=result,
            type="queries"
        )

        # get a brief description of the experiment
        query_analyser.plot_learning_curves(title='Active Learning experiment results')

    def test_hamming_loss(self):
        performance_metrics = [HammingLoss()]

        experiment = HoldOutExperiment(
            self.__X_class,
            self.__y_class,
            scenario_type=PoolBasedSamplingScenario,
            ml_technique=self.__ml_technique_class,
            performance_metrics=performance_metrics,
            query_strategy=self.__query_strategy,
            oracle=SimulatedOracleQueryIndex(labels=self.__y_class),
            stopping_criteria=MaxIteration(value=10),
            self_partition=True,
            test_ratio=0.3,
            initial_label_rate=0.05,
            all_class=True
        )

        start_time = time.time()
        result = experiment.evaluate(client=self.__client, verbose=True)
        print()
        print("---Active Learning experiment %s seconds ---" % (time.time() - start_time))

        query_analyser = ExperimentAnalyserFactory.experiment_analyser(
            performance_metrics=[metric.metric_name for metric in performance_metrics],
            method_name=self.__query_strategy.query_function_name,
            method_results=result,
            type="queries"
        )

        # get a brief description of the experiment
        query_analyser.plot_learning_curves(title='Active Learning experiment results')

    def test_roc_auc(self):
        performance_metrics = [RocAuc()]

        experiment = HoldOutExperiment(
            self.__X_class,
            self.__y_class,
            scenario_type=PoolBasedSamplingScenario,
            ml_technique=self.__ml_technique_class,
            performance_metrics=performance_metrics,
            query_strategy=self.__query_strategy,
            oracle=SimulatedOracleQueryIndex(labels=self.__y_class),
            stopping_criteria=MaxIteration(value=10),
            self_partition=True,
            test_ratio=0.3,
            initial_label_rate=0.05,
            all_class=True
        )

        start_time = time.time()
        result = experiment.evaluate(client=self.__client, verbose=True)
        print()
        print("---Active Learning experiment %s seconds ---" % (time.time() - start_time))

        query_analyser = ExperimentAnalyserFactory.experiment_analyser(
            performance_metrics=[metric.metric_name for metric in performance_metrics],
            method_name=self.__query_strategy.query_function_name,
            method_results=result,
            type="queries"
        )

        # get a brief description of the experiment
        query_analyser.plot_learning_curves(title='Active Learning experiment results')

    def test_precision(self):
        performance_metrics = [Precision()]

        experiment = HoldOutExperiment(
            self.__X_class,
            self.__y_class,
            scenario_type=PoolBasedSamplingScenario,
            ml_technique=self.__ml_technique_class,
            performance_metrics=performance_metrics,
            query_strategy=self.__query_strategy,
            oracle=SimulatedOracleQueryIndex(labels=self.__y_class),
            stopping_criteria=MaxIteration(value=10),
            self_partition=True,
            test_ratio=0.3,
            initial_label_rate=0.05,
            all_class=True
        )

        start_time = time.time()
        result = experiment.evaluate(client=self.__client, verbose=True)
        print()
        print("---Active Learning experiment %s seconds ---" % (time.time() - start_time))

        query_analyser = ExperimentAnalyserFactory.experiment_analyser(
            performance_metrics=[metric.metric_name for metric in performance_metrics],
            method_name=self.__query_strategy.query_function_name,
            method_results=result,
            type="queries"
        )

        # get a brief description of the experiment
        query_analyser.plot_learning_curves(title='Active Learning experiment results')

    def test_recall(self):
        performance_metrics = [Recall()]

        experiment = HoldOutExperiment(
            self.__X_class,
            self.__y_class,
            scenario_type=PoolBasedSamplingScenario,
            ml_technique=self.__ml_technique_class,
            performance_metrics=performance_metrics,
            query_strategy=self.__query_strategy,
            oracle=SimulatedOracleQueryIndex(labels=self.__y_class),
            stopping_criteria=MaxIteration(value=10),
            self_partition=True,
            test_ratio=0.3,
            initial_label_rate=0.05,
            all_class=True
        )

        start_time = time.time()
        result = experiment.evaluate(client=self.__client, verbose=True)
        print()
        print("---Active Learning experiment %s seconds ---" % (time.time() - start_time))

        query_analyser = ExperimentAnalyserFactory.experiment_analyser(
            performance_metrics=[metric.metric_name for metric in performance_metrics],
            method_name=self.__query_strategy.query_function_name,
            method_results=result,
            type="queries"
        )

        # get a brief description of the experiment
        query_analyser.plot_learning_curves(title='Active Learning experiment results')

if __name__ == '__main__':
    unittest.main()
