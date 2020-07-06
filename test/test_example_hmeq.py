import time
import unittest

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, hamming_loss

from dpyacl.query.uncertainty_sampling import QueryInstanceRandom, QueryMarginSampling
from dpyacl.core.stop_criteria import MaxIteration
from dpyacl.experiment import ExperimentAnalyserFactory
from dpyacl.experiment.context import HoldOutExperiment, CrossValidationExperiment
from dpyacl.metrics import Accuracy
from dpyacl.metrics.evaluation import F1, HammingLoss, Precision, Recall
from dpyacl.oracle import SimulatedOracleQueryIndex
from dpyacl.scenario import PoolBasedSamplingScenario
from dpyacl.core.misc.misc import split


class TestActiveLearningHMQE(unittest.TestCase):
    # logging.basicConfig(level=logging.DEBUG)

    @classmethod
    def setUp(self):
        df = pd.read_csv('../resources/data/hmeq.csv')  # import the dataset
        df.dropna(axis=0, how='any', inplace=True)
        df = pd.get_dummies(df, columns=['REASON', 'JOB'])

        self.__X = df.drop(['BAD'], axis=1)
        self.__y = df.filter(['BAD'], axis=1)

        self.__train_idx, self.__test_idx, self.__label_idx, self.__unlabel_idx = split(
            X=self.__X,
            y=self.__y['BAD'],
            test_ratio=0.3,
            initial_label_rate=0.05,
            split_count=1,
            all_class=True)

    def test_ActiveLearning_HoldHout(self):

        # INI the ALExperiment -----------------------------------------------------------------------------------------
        al_ml_technique = LogisticRegression(solver='liblinear')
        stopping_criteria = MaxIteration(50)
        query_strategy = QueryMarginSampling()
        performance_metrics = [
                Accuracy(),
                F1(average='macro'),
                HammingLoss(),
                Precision(average='macro'),
                Recall(average='macro')]

        experiment = HoldOutExperiment(
            self.__X,
            self.__y['BAD'],
            scenario_type=PoolBasedSamplingScenario,
            train_idx=self.__train_idx,
            test_idx=self.__test_idx,
            label_idx=self.__label_idx,
            unlabel_idx=self.__unlabel_idx,
            ml_technique=al_ml_technique,
            performance_metrics=performance_metrics,
            query_strategy=query_strategy,
            oracle=SimulatedOracleQueryIndex(labels=self.__y),
            stopping_criteria=stopping_criteria,
            self_partition=False
        )

        print("")
        start_time = time.time()
        result = experiment.evaluate(verbose=False)
        print("---Active Learning experiment %s seconds ---" % (time.time() - start_time))

        query_analyser = ExperimentAnalyserFactory.experiment_analyser(
            performance_metrics=[metric.metric_name for metric in performance_metrics],
            method_name=query_strategy.query_function_name,
            method_results=result,
            type="queries"
        )
        # get a brief description of the experiment
        query_analyser.plot_learning_curves(title='Active Learning experiment results')

        foldIndex = 0
        train_x = self.__X.iloc[self.__train_idx[foldIndex], :]
        train_y = self.__y.iloc[self.__train_idx[foldIndex], :]
        test_x = self.__X.iloc[self.__test_idx[foldIndex], :]
        test_y = self.__y.iloc[self.__test_idx[foldIndex], :]

        active_y_pred = result[0].ml_technique.predict(test_x)

        print("Active Learning Accuracy score : ", accuracy_score(test_y, active_y_pred))
        print("Active Learning F1 score: ", f1_score(test_y, active_y_pred, average='macro', zero_division=0))
        print("Active Learning Hamming Loss", hamming_loss(test_y, active_y_pred))
        print("Active Learning Precision score : ", precision_score(test_y, active_y_pred, average='macro', zero_division=0))
        print("Active Learning Recall score : ", recall_score(test_y, active_y_pred, average='macro', zero_division=0))

        # END the ALExperiment -----------------------------------------------------------------------------------------

        # INI the PLExperiment -----------------------------------------------------------------------------------------
        pl_ml_technique = LogisticRegression(solver='liblinear')

        print("")
        start_time = time.time()
        pl_ml_technique.fit(train_x, train_y)
        print("---Passive Learning experiment %s seconds ---" % (time.time() - start_time))

        passive_y_pred = pl_ml_technique.predict(test_x)

        print("Pasive Learning Accuracy score : ", accuracy_score(test_y, passive_y_pred))
        print("Pasive Learning F1 score: ", f1_score(test_y, passive_y_pred, average='macro', zero_division=0))
        print("Pasive Learning Hamming Loss", hamming_loss(test_y, passive_y_pred))
        print("Pasive Learning Precision score : ", precision_score(test_y, passive_y_pred, average='macro', zero_division=0))
        print("Pasive Learning Recall score : ", recall_score(test_y, passive_y_pred, average='macro', zero_division=0))
        # END the PLExperiment -----------------------------------------------------------------------------------------

    def test_hold_out_marginSamplingQuery_unlabelSetEmpty(self):

        ml_technique = LogisticRegression(solver='liblinear')
        stopping_criteria = MaxIteration(50)
        query_strategy = QueryMarginSampling()
        performance_metrics = [
            Accuracy(),
            F1(average='macro'),
            HammingLoss(),
            Precision(average='macro'),
            Recall(average='macro')]

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

        result = experiment.evaluate(verbose=False)

        query_analyser = ExperimentAnalyserFactory.experiment_analyser(
            performance_metrics=[metric.metric_name for metric in performance_metrics],
            method_name=query_strategy.query_function_name,
            method_results=result,
            type="queries"
        )
        # get a brief description of the experiment
        query_analyser.plot_learning_curves(title='Active Learning experiment results')

    def test_cross_validation_randomQuery_unlabelSetEmpty_singleThread(self):

        ml_technique = LogisticRegression(solver='liblinear')
        stopping_criteria = MaxIteration(50)
        query_strategy = QueryInstanceRandom()
        performance_metrics = [
            Accuracy(),
            F1(average='macro'),
            HammingLoss(),
            Precision(average='macro'),
            Recall(average='macro')]

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
            test_ratio=0.3,
            initial_label_rate=0.05,
            all_class=True
        )

        results = experiment.evaluate(verbose=False)

        for result in results:
            query_analyser = ExperimentAnalyserFactory.experiment_analyser(
                performance_metrics=[metric.metric_name for metric in performance_metrics],
                method_name=query_strategy.query_function_name,
                method_results=result,
                type="queries"
            )

            # get a brief description of the experiment
            query_analyser.plot_learning_curves(title='Active Learning experiment results')

    def test_cross_validation_randomQuery_unlabelSetEmpty_multithread(self):

        ml_technique = LogisticRegression(solver='liblinear')
        # ml_technique = BernoulliNB()
        # ml_technique = svm.SVC(kernel='rbf', probability=True)
        # ml_technique = svm.NuSVC(gamma='auto', probability=True)
        # stopping_criteria = PercentOfUnlabel(70)
        stopping_criteria = MaxIteration(50)
        query_strategy = QueryMarginSampling()
        performance_metrics = [
            Accuracy(),
            F1(average='macro'),
            HammingLoss(),
            Precision(average='macro'),
            Recall(average='macro')]

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

        results = experiment.evaluate(verbose=False, multithread=True, max_threads=10)

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
