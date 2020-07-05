"""
Class that represents a Classical Active Learning Process.
"""
# Authors: Alfredo Lorie

import copy
from abc import ABCMeta, abstractmethod

from distributed import Client

from ..core.stop_criteria import AbstractStopCriterion, MaxIteration
from ..core.state import StateItem, State
from ..scenario.scenario import AbstractScenario

__all__ = ['IAlgorithm', 'AbstractALAlgorithm', 'ClassicActiveLearning']


class IAlgorithm(metaclass=ABCMeta):
    """
    Active Learning Algorithm interface.
    """

    @abstractmethod
    def execute(self, **kwargs) -> None:
        """Start the active learning algorithm
        Parameters
        ----------
        kwargs: optional
        """
        pass


class AbstractALAlgorithm(IAlgorithm, metaclass=ABCMeta):

    def __init__(self,
                 scenario: AbstractScenario,
                 stopping_criteria: AbstractStopCriterion = MaxIteration,
                 **kwargs):
        self._scenario = scenario
        self._stopping_criterion = stopping_criteria
        self._experiment_result = []

    @abstractmethod
    def execute(self, verbose=False, **kwargs) -> None:
        pass

    def get_experiment_result(self):
        """
            Get the information stored in stateIO
        Returns
        ----------
        experiment_resuly: State
            return the stateIO of the experiment.
        """
        if len(self._experiment_result) == 0:
            raise Exception('There is no experiment result.Use start_query() get experiment result firstly.')
        return copy.deepcopy(self._experiment_result)


class ClassicActiveLearning(AbstractALAlgorithm, metaclass=ABCMeta):

    def __init__(self,
                 scenario: AbstractScenario,
                 stopping_criteria: AbstractStopCriterion,
                 batch_size=1,
                 **kwargs):
        """Start the active learning algorithm sequentially
        Parameters
        ----------

        stopping_criteria: {MaxIteration, UnlabeledSetEmpty}, optional (default=MaxIteration),
        model:
        performance_metric='accuracy_score',
        batch_size=1,
        kwargs: optional
        """
        super().__init__(scenario,
                         stopping_criteria,
                         **kwargs)

    def execute(self, verbose=False, client: Client = None, **kwargs) -> State:
        """
            Start the active-learning main loop.

            Parameters
            ----------
            :param verbose:
            :param client:
            :param kwargs: optional
            :return:
        """
        round = 0
        saver = self._scenario.initIteration(verbose, **kwargs)

        while not self._stopping_criterion.is_stop():
            # Train and evaluate Model over the labeled instances
            label_pred, label_perf = self._scenario.executeLabeledTraining(client=client)

            if round == 0:
                saver.set_initial_point(label_perf)

            select_ind = self._scenario.selectInstances(client=client)

            # show label values
            self._scenario.labelInstances(select_ind, verbose)

            # update label and unlabel instaces
            self._scenario.updateLabelledData(select_ind)

            # save intermediate results
            st = StateItem(select_index=select_ind,
                           performance_metrics=[metric['name'] for metric in label_perf],
                           performance=label_perf)

            saver.add_state(st)

            # update stopping_criteria
            self._stopping_criterion.update_information(saver)

            round += 1;

        saver.set_ml_technique(copy.deepcopy(self._scenario.__getattribute__("_ml_technique")))
        self._experiment_result.append(copy.deepcopy(saver))
        return saver
