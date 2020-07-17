from sklearn.datasets import load_iris
from pclal.algorithms import ClassicActiveLearning
from pclal.core import UnlabeledSetEmpty, MaxIteration

import unittest

from pclal.core.query_strategies import QueryInstanceRandom, QueryEntropySampling
from pclal.core.query_strategies import QueryMarginSampling, QueryLeastConfidentSampling


class TestClassicAlLinearReg(unittest.TestCase):
    # Get the data
    __X, __y = load_iris(return_X_y=True)

    def test_QueryInstanceRandom(self):
        # init the AlExperiment
        al = ClassicActiveLearning(self.__X, self.__y, stopping_criteria=UnlabeledSetEmpty())

        # create a kfold experiment
        al.kfold(n_splits=10)

        # set the query strategy
        strategy = QueryInstanceRandom()
        al.set_query_strategy(strategy=QueryInstanceRandom())

        # set the metric for experiment.
        al.set_performance_metric('accuracy_score')

        # Execute the experiment
        al.execute(verbose=False)

        # get the experiemnt result
        # stateIO = al.get_experiment_result()

        # get a brief description of the experiment
        al.plot_learning_curve(title='Alexperiment result %s' % strategy.query_function_name)

    def test_QueryEntropySampling(self):

        # init the AlExperiment
        al = ClassicActiveLearning(self.__X, self.__y, stopping_criteria=UnlabeledSetEmpty())

        # create a kfold experiment
        al.kfold(n_splits=10)

        # set the query strategy
        strategy = QueryEntropySampling()
        al.set_query_strategy(strategy=QueryEntropySampling())

        # set the metric for experiment.
        al.set_performance_metric('accuracy_score')

        # by default,run in multi-thread.
        al.execute()

        # get the experiemnt result
        stateIO = al.get_experiment_result()

        # get a brief description of the experiment
        al.plot_learning_curve(title='Alexperiment result %s' % strategy.query_function_name)

    def test_QueryMarginSampling(self):

        # init the AlExperiment
        al = ClassicActiveLearning(self.__X, self.__y, stopping_criteria=UnlabeledSetEmpty())

        # create a kfold experiment
        al.kfold(n_splits=10)

        # set the query strategy
        strategy = QueryMarginSampling()
        al.set_query_strategy(strategy=QueryMarginSampling())

        # set the metric for experiment.
        al.set_performance_metric('accuracy_score')

        # by default,run in multi-thread.
        al.execute()

        # get the experiemnt result
        stateIO = al.get_experiment_result()

        # get a brief description of the experiment
        al.plot_learning_curve(title='Alexperiment result %s' % strategy.query_function_name)

    def test_QueryLeastConfidentSampling(self):

        # init the AlExperiment
        al = ClassicActiveLearning(self.__X, self.__y, stopping_criteria=UnlabeledSetEmpty())

        # create a kfold experiment
        al.kfold(n_splits=10)

        # set the query strategy
        strategy = QueryLeastConfidentSampling()
        al.set_query_strategy(strategy=strategy)

        # set the metric for experiment.
        al.set_performance_metric('accuracy_score')

        # by default,run in multi-thread.
        al.execute()

        # get the experiemnt result
        stateIO = al.get_experiment_result()

        # get a brief description of the experiment
        al.plot_learning_curve(title='Alexperiment result %s' % strategy.query_function_name)


if __name__ == '__main__':
    unittest.main()
