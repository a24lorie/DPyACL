import time
import unittest

import dask.array as da
import numpy as np
from distributed import Client
from sklearn.datasets import make_classification, load_diabetes, load_iris
from sklearn.linear_model import LogisticRegression

from dpyacl.core.stop_criteria import UnlabelSetEmpty, MaxIteration
from dpyacl.experiment import ExperimentAnalyserFactory
from dpyacl.experiment.context import HoldOutExperiment, CrossValidationExperiment
from dpyacl.metrics import Accuracy
from dpyacl.metrics.evaluation import F1, HammingLoss
from dpyacl.oracle import SimulatedOracleQueryIndex
from dpyacl.scenario import PoolBasedSamplingScenario
from dpyacl.strategies.single_label import QueryInstanceRandom, QueryMarginSampling
from dpyacl.strategies.single_label.query_by_comitee import QueryVoteEntropy


class TestEvaluation(unittest.TestCase):
    # logging.basicConfig(level=logging.DEBUG)

    # Get the data
    __X, __y = load_iris(return_X_y=True)
    # __X, __y = load_diabetes(return_X_y=True)

    # __X, __y = make_classification(n_samples=100, n_features=10, n_informative=2, n_redundant=2,
    #                                n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None,
    #                                flip_y=0.01,
    #                                class_sep=1.0,
    #                                hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)

    __X = da.from_array(__X)
    __y = da.from_array(__y)

    __client = Client("tcp://192.168.2.100:8786")
    # __client = Client(processes=False)
    # __client = None

    def test_hold_out_randomQuery_unlabelSetEmpty(self):
        ml_technique = LogisticRegression(solver='sag')
        stopping_criteria = MaxIteration(50)
        query_strategy = QueryInstanceRandom()
        performance_metrics = [Accuracy(),  F1(average='weighted'), HammingLoss()]

        # init the ALExperiment
        experiment = HoldOutExperiment(
            self.__X,
            self.__y,
            scenario_type=PoolBasedSamplingScenario,
            ml_technique=ml_technique,
            performance_metrics=performance_metrics,
            query_strategy=query_strategy,
            oracle=SimulatedOracleQueryIndex(labels=self.__y),
            stopping_criteria=stopping_criteria,
            self_partition=True,
            test_ratio=0.3,
            initial_label_rate=0.05,
            all_class=True
        )

        start_time = time.time()
        result = experiment.evaluate(client= self.__client, verbose=True)
        print()
        print("---Active Learning experiment %s seconds ---" % (time.time() - start_time))

        query_analyser = ExperimentAnalyserFactory.experiment_analyser(
                            performance_metrics= [metric.metric_name for metric in performance_metrics],
                            method_name=query_strategy.query_function_name,
                            method_results=result,
                            type="queries"
                        )

        # get a brief description of the experiment
        query_analyser.plot_learning_curves(title='Active Learning experiment results')

    def test_hold_out_marginSamplingQuery_unlabelSetEmpty(self):

        ml_technique = LogisticRegression()
        stopping_criteria = UnlabelSetEmpty()
        query_strategy = QueryMarginSampling()
        performance_metrics = [Accuracy(),  F1(average='weighted'), HammingLoss()]
        # performance_metrics = [Mse(square=False), Mse(square=True)]

        # init the ALExperiment
        experiment = HoldOutExperiment(
            self.__X,
            self.__y,
            scenario_type=PoolBasedSamplingScenario,
            ml_technique=ml_technique,
            performance_metrics=performance_metrics,
            query_strategy=query_strategy,
            oracle=SimulatedOracleQueryIndex(labels=self.__y),
            stopping_criteria=stopping_criteria,
            self_partition=True,
            test_ratio=0.3,
            initial_label_rate=0.05,
            all_class=False
        )

        result = experiment.evaluate(client=self.__client, verbose=True)

        query_analyser = ExperimentAnalyserFactory.experiment_analyser(
                            performance_metrics=[metric.metric_name for metric in performance_metrics],
                            method_name=query_strategy.query_function_name,
                            method_results=result,
                            type="queries"
                        )
        # get a brief description of the experiment
        query_analyser.plot_learning_curves(title='Active Learning experiment results')

        np.random.seed(0)
        indices = np.random.permutation(len(self.__X))
        iris_X_test = self.__X[indices[-10:]]
        print(result[0].ml_technique.predict(iris_X_test))

    def test_cross_validation_randomQuery_unlabelSetEmpty(self):

        ml_technique = LogisticRegression(solver='sag')
        stopping_criteria = UnlabelSetEmpty()
        query_strategy = QueryInstanceRandom()
        performance_metrics = [Accuracy(), F1(average='weighted'), HammingLoss()]

        # init the ALExperiment
        experiment = CrossValidationExperiment(
            self.__X,
            self.__y,
            scenario_type=PoolBasedSamplingScenario,
            ml_technique=ml_technique,
            performance_metrics=performance_metrics,
            query_strategy=query_strategy,
            oracle=SimulatedOracleQueryIndex(labels=self.__y),
            stopping_criteria=stopping_criteria,
            self_partition=True,
            kfolds=3,
            test_ratio=0.3,
            initial_label_rate=0.05,
            all_class=True
        )

        results = experiment.evaluate(verbose=True, multithread=True)

        for result in results:
            query_analyser = ExperimentAnalyserFactory.experiment_analyser(
                performance_metrics=[metric.metric_name for metric in performance_metrics],
                method_name=query_strategy.query_function_name,
                method_results=result,
                type="queries"
            )

            # get a brief description of the experiment
            query_analyser.plot_learning_curves(title='Active Learning experiment results')

    def test_cross_validation_randomQuery_MaxIteration(self):

        ml_technique = LogisticRegression()
        # ml_technique = BernoulliNB()
        # ml_technique = svm.SVC(kernel='rbf', probability=True)
        # ml_technique = svm.NuSVC(gamma='auto', probability=True)
        # stopping_criteria = PercentOfUnlabel(70)
        stopping_criteria = MaxIteration(50)
        # stopping_criteria = TimeLimit(2)
        # query_strategy = QueryInstanceRandom()
        query_strategy = QueryInstanceRandom()

        performance_metrics = [Accuracy(), F1(average='weighted'), HammingLoss()]

        # init the ALExperiment
        experiment = CrossValidationExperiment(
            self.__X,
            self.__y,
            scenario_type=PoolBasedSamplingScenario,
            ml_technique=ml_technique,
            performance_metrics=performance_metrics,
            query_strategy=query_strategy,
            oracle=SimulatedOracleQueryIndex(labels=self.__y),
            stopping_criteria=stopping_criteria,
            self_partition=True,
            kfolds=10,
            oracle_name='SimulatedOracleQueryIndex',
            test_ratio=0.3,
            initial_label_rate=0.05,
            all_class=True
        )

        results = experiment.evaluate(verbose=True, multithread=True, max_threads=10, client=self.__client)

        for result in results:
            query_analyser = ExperimentAnalyserFactory.experiment_analyser(
                performance_metrics=[metric.metric_name for metric in performance_metrics],
                method_name=query_strategy.query_function_name,
                method_results=result,
                type="queries"
            )

            # get a brief description of the experiment
            query_analyser.plot_learning_curves(title='Active Learning experiment results')

    def test_cross_validation_vote_entropy_MaxIteration(self):

        ml_technique = LogisticRegression()
        # ml_technique = BernoulliNB()
        # ml_technique = svm.SVC(kernel='rbf', probability=True)
        # ml_technique = svm.NuSVC(gamma='auto', probability=True)
        # stopping_criteria = PercentOfUnlabel(70)
        stopping_criteria = MaxIteration(30)
        # stopping_criteria = TimeLimit(2)
        # query_strategy = QueryInstanceRandom()
        query_strategy = QueryVoteEntropy()

        performance_metrics = [Accuracy(), F1(average='weighted'), HammingLoss()]

        # init the ALExperiment
        experiment = CrossValidationExperiment(
            self.__X,
            self.__y,
            scenario_type=PoolBasedSamplingScenario,
            ml_technique=ml_technique,
            performance_metrics=performance_metrics,
            query_strategy=query_strategy,
            oracle=SimulatedOracleQueryIndex(labels=self.__y),
            stopping_criteria=stopping_criteria,
            self_partition=True,
            kfolds=10,
            oracle_name='SimulatedOracleQueryIndex',
            test_ratio=0.3,
            initial_label_rate=0.05,
            all_class=True
        )

        results = experiment.evaluate(verbose=True, multithread=True, max_threads=10, client=self.__client)

        for result in results:
            query_analyser = ExperimentAnalyserFactory.experiment_analyser(
                performance_metrics=[metric.metric_name for metric in performance_metrics],
                method_name=query_strategy.query_function_name,
                method_results=result,
                type="queries"
            )

            # get a brief description of the experiment
            query_analyser.plot_learning_curves(title='Active Learning experiment results')


if __name__ == '__main__':
    unittest.main()
