import time
import unittest

import matplotlib.pyplot as plt
import numpy as np
from distributed import Client
from sklearn.datasets import make_classification
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import LogisticRegression

from dpyacl.core.misc.misc import split
from dpyacl.core.stop_criteria import UnlabelSetEmpty, PercentOfUnlabel, MaxIteration
from dpyacl.experiment import ExperimentAnalyserFactory
from dpyacl.experiment.context import HoldOutExperiment
from dpyacl.metrics import Accuracy, Mse, Recall
from dpyacl.oracle import SimulatedOracleQueryIndex
from dpyacl.scenario.scenario import PoolBasedSamplingScenario
from dpyacl.strategies.single_label import QueryInstanceRandom, QueryMarginSampling, QueryDistanceToBoundarySampling, \
    QueryLeastConfidentSampling, QueryEntropySampling, QueryRegressionStd
from dpyacl.strategies.single_label.error_reduction import QueryExpectedLogLoss, QueryExpectedCeroOneLoss
from dpyacl.strategies.single_label.query_by_comitee import QueryVoteEntropy, QueryKullbackLeiblerDivergence


class TestSingleLabelQueryStrategy(unittest.TestCase):

    # Get the data
    __X, __y = make_classification(n_samples=100, n_features=10, n_informative=2, n_redundant=2,
                                   n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None,
                                   flip_y=0.01,
                                   class_sep=1.0,
                                   hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)

    __ml_technique = LogisticRegression(solver='sag')
    __performance_metrics = [Accuracy(), Recall()]

    __client = Client("tcp://192.168.2.100:8786")

    def test_random_query(self):

        query_strategy = QueryInstanceRandom()

        # init the ALExperiment
        experiment = HoldOutExperiment(
            self.__X,
            self.__y,
            scenario_type=PoolBasedSamplingScenario,
            ml_technique=self.__ml_technique,
            performance_metrics=self.__performance_metrics,
            query_strategy=query_strategy,
            oracle=SimulatedOracleQueryIndex(labels=self.__y),
            stopping_criteria=MaxIteration(20),
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
                            performance_metrics= [metric.metric_name for metric in self.__performance_metrics],
                            method_name=query_strategy.query_function_name,
                            method_results=result,
                            type="queries"
                        )

        # get a brief description of the experiment
        query_analyser.plot_learning_curves(title='Active Learning experiment results')

    def test_query_entropy_sampling(self):

        query_strategy = QueryEntropySampling()

        # init the ALExperiment
        experiment = HoldOutExperiment(
            self.__X,
            self.__y,
            scenario_type=PoolBasedSamplingScenario,
            ml_technique=self.__ml_technique,
            performance_metrics=self.__performance_metrics,
            query_strategy=query_strategy,
            oracle=SimulatedOracleQueryIndex(labels=self.__y),
            stopping_criteria=UnlabelSetEmpty(),
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
            performance_metrics=[metric.metric_name for metric in self.__performance_metrics],
            method_name=query_strategy.query_function_name,
            method_results=result,
            type="queries"
        )

        # get a brief description of the experiment
        query_analyser.plot_learning_curves(title='Active Learning experiment results')

    def test_query_least_confident_sampling(self):

        query_strategy = QueryLeastConfidentSampling()

        # init the ALExperiment
        experiment = HoldOutExperiment(
            self.__X,
            self.__y,
            scenario_type=PoolBasedSamplingScenario,
            ml_technique=self.__ml_technique,
            performance_metrics=self.__performance_metrics,
            query_strategy=query_strategy,
            oracle=SimulatedOracleQueryIndex(labels=self.__y),
            stopping_criteria=UnlabelSetEmpty(),
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
            performance_metrics=[metric.metric_name for metric in self.__performance_metrics],
            method_name=query_strategy.query_function_name,
            method_results=result,
            type="queries"
        )

        # get a brief description of the experiment
        query_analyser.plot_learning_curves(title='Active Learning experiment results')

    def test_query_margin_sampling(self):

        query_strategy = QueryMarginSampling()

        # init the ALExperiment
        experiment = HoldOutExperiment(
            self.__X,
            self.__y,
            scenario_type=PoolBasedSamplingScenario,
            ml_technique=self.__ml_technique,
            performance_metrics=self.__performance_metrics,
            query_strategy=query_strategy,
            oracle=SimulatedOracleQueryIndex(labels=self.__y),
            stopping_criteria=UnlabelSetEmpty(),
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
            performance_metrics=[metric.metric_name for metric in self.__performance_metrics],
            method_name=query_strategy.query_function_name,
            method_results=result,
            type="queries"
        )

        # get a brief description of the experiment
        query_analyser.plot_learning_curves(title='Active Learning experiment results')

    def test_query_distance_to_boundary_sampling(self):

        query_strategy = QueryDistanceToBoundarySampling()

        # init the ALExperiment
        experiment = HoldOutExperiment(
            self.__X,
            self.__y,
            scenario_type=PoolBasedSamplingScenario,
            ml_technique=self.__ml_technique,
            performance_metrics=self.__performance_metrics,
            query_strategy=query_strategy,
            oracle=SimulatedOracleQueryIndex(labels=self.__y),
            stopping_criteria=UnlabelSetEmpty(),
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
            performance_metrics=[metric.metric_name for metric in self.__performance_metrics],
            method_name=query_strategy.query_function_name,
            method_results=result,
            type="queries"
        )

        # get a brief description of the experiment
        query_analyser.plot_learning_curves(title='Active Learning experiment results')

    def test_query_regression_std(self):

        # Get the data
        X = np.random.choice(np.linspace(0, 20, 1000), size=100, replace=False).reshape(-1, 1)
        y = np.sin(X) + np.random.normal(scale=0.3, size=X.shape)

        # assembling initial training set
        train_idx, test_idx, label_idx, unlabel_idx = split(
            X=X,
            y=y,
            test_ratio=0.3,
            initial_label_rate=0.05,
            split_count=1,
            all_class=True)

        # defining the kernel for the Gaussian process
        ml_technique = GaussianProcessRegressor(
            kernel=RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
                   + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1)))

        experiment = HoldOutExperiment(
            X,
            y,
            scenario_type=PoolBasedSamplingScenario,
            train_idx=train_idx,
            test_idx=test_idx,
            label_idx=label_idx,
            unlabel_idx=unlabel_idx,
            ml_technique=ml_technique,
            performance_metrics=[Mse(squared=True)],
            query_strategy=QueryRegressionStd(),
            oracle=SimulatedOracleQueryIndex(labels=y),
            stopping_criteria=PercentOfUnlabel(value=70),
            self_partition=False
        )

        result = experiment.evaluate(verbose=True)
        regressor = result[0].ml_technique

        # plotting the initial estimation
        with plt.style.context('seaborn-white'):
            plt.figure(figsize=(14, 7))
            x = np.linspace(0, 20, 1000)
            pred, std = regressor.predict(x.reshape(-1, 1), return_std=True)
            plt.plot(x, pred)
            plt.fill_between(x, pred.reshape(-1, ) - std, pred.reshape(-1, ) + std, alpha=0.2)
            plt.scatter(X, y, c='k')
            plt.title('Initial estimation')
            plt.show()

    def test_query_cero_one_loss(self):

        query_strategy = QueryExpectedCeroOneLoss()

        # init the ALExperiment
        experiment = HoldOutExperiment(
            self.__X,
            self.__y,
            scenario_type=PoolBasedSamplingScenario,
            ml_technique=self.__ml_technique,
            performance_metrics=self.__performance_metrics,
            query_strategy=query_strategy,
            oracle=SimulatedOracleQueryIndex(labels=self.__y),
            stopping_criteria=MaxIteration(5),
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
            performance_metrics=[metric.metric_name for metric in self.__performance_metrics],
            method_name=query_strategy.query_function_name,
            method_results=result,
            type="queries"
        )

        # get a brief description of the experiment
        query_analyser.plot_learning_curves(title='Active Learning experiment results')

    def test_query_log_loss(self):

        query_strategy = QueryExpectedLogLoss()

        # init the ALExperiment
        experiment = HoldOutExperiment(
            self.__X,
            self.__y,
            scenario_type=PoolBasedSamplingScenario,
            ml_technique=self.__ml_technique,
            performance_metrics=self.__performance_metrics,
            query_strategy=query_strategy,
            oracle=SimulatedOracleQueryIndex(labels=self.__y),
            stopping_criteria=MaxIteration(5),
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
            performance_metrics=[metric.metric_name for metric in self.__performance_metrics],
            method_name=query_strategy.query_function_name,
            method_results=result,
            type="queries"
        )

        # get a brief description of the experiment
        query_analyser.plot_learning_curves(title='Active Learning experiment results')

    def test_vote_entropy(self):

        query_strategy = QueryVoteEntropy(n_jobs=5)

        # init the ALExperiment
        experiment = HoldOutExperiment(
            self.__X,
            self.__y,
            scenario_type=PoolBasedSamplingScenario,
            ml_technique=self.__ml_technique,
            performance_metrics=self.__performance_metrics,
            query_strategy=query_strategy,
            oracle=SimulatedOracleQueryIndex(labels=self.__y),
            stopping_criteria=MaxIteration(20),
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
            performance_metrics=[metric.metric_name for metric in self.__performance_metrics],
            method_name=query_strategy.query_function_name,
            method_results=result,
            type="queries"
        )

        # get a brief description of the experiment
        query_analyser.plot_learning_curves(title='Active Learning experiment results')

    def test_kullback_leibler_divergence(self):

        query_strategy = QueryKullbackLeiblerDivergence(n_jobs=5)

        # init the ALExperiment
        experiment = HoldOutExperiment(
            self.__X,
            self.__y,
            scenario_type=PoolBasedSamplingScenario,
            ml_technique=self.__ml_technique,
            performance_metrics=self.__performance_metrics,
            query_strategy=query_strategy,
            oracle=SimulatedOracleQueryIndex(labels=self.__y),
            stopping_criteria=MaxIteration(20),
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
            performance_metrics=[metric.metric_name for metric in self.__performance_metrics],
            method_name=query_strategy.query_function_name,
            method_results=result,
            type="queries"
        )

        # get a brief description of the experiment
        query_analyser.plot_learning_curves(title='Active Learning experiment results')

if __name__ == '__main__':
    unittest.main()
