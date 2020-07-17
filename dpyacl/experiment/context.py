"""
Active Learning Experiment Context

Supports
1. HoldOutExperiment
2. CrossValidationExperiment
"""
# Authors: Alfredo Lorie

import copy
import importlib
import multiprocessing as mp
import os
import time
from abc import ABCMeta, abstractmethod

import dask.array as da
import numpy as np
from distributed import as_completed, Client

from dpyacl.core.collections import IndexCollection
from dpyacl.core.misc.misc import split, check_X_y
from dpyacl.core.stop_criteria import AbstractStopCriterion
from dpyacl.learner import ClassicActiveLearning
from dpyacl.metrics import BaseMetrics
from dpyacl.oracle.oracle import Oracle
from dpyacl.scenario import *
from dpyacl.strategies.stategies import SinlgeLabelIndexQuery

__all__ = ['AbstractExperiment', 'HoldOutExperiment', 'CrossValidationExperiment']


class AbstractExperiment(metaclass=ABCMeta):

    def __init__(self,
                 client: Client,
                 X, Y,
                 ml_technique,
                 scenario_type: AbstractScenario,
                 performance_metrics: [],
                 query_strategy: SinlgeLabelIndexQuery,
                 oracle: Oracle,
                 stopping_criteria: AbstractStopCriterion,
                 self_partition: bool,
                 kfolds: int = 1,
                 **kwargs):
        """
        Parameters
        ----------
        :param client: distributed.Client
        :param X: array-like
            Data matrix with [n_samples, n_features]
        :param Y: array-like, optional
            labels of given data [n_samples, n_labels] or [n_samples]
        :param ml_technique
        :param scenario_type: Sub-Type of AbstractScenario
            Type of Active Learning scenario to use
        :param performance_metrics: array-like of BaseMetrics elements
        :param query_strategy: SinlgeLabelIndexQuery
        :param oracle: Oracle
        :param stopping_criteria: AbstractStopCriterion
        :param self_partition: bool
        :param kfolds: int, optional (default=1)
             If self_partition is True Random split data k sets according to the extra parameters
                -> test_ratio: float, optional (default=0.3)
                    Ratio of test set
                -> initial_label_rate: float, optional (default=0.05)
                    Ratio of initial label set
                    e.g. Initial_labelset*(1-test_ratio)*n_samples
                -> all_class: bool, optional (default=True)
                    Whether each split will contain at least one instance for each class.
                    If False, a totally random split will be performed.

            If self_partition is False the following the following parameter must be specified
                -> train_idx:
                -> test_idx:
                -> label_idx:
                ->  unlabel_idx:
        :param kwargs: optional
            Extra parameters
        """
        self._client = client

        if type(X) is da.core.Array:
            self._X = X.persist()
        else:
            self._X = da.from_array(X, chunks=len(X) // 50).persist()

        if isinstance(Y , da.core.Array):
            self._Y = Y.persist()
        else:
            self._Y = da.from_array(Y, chunks=len(Y) // 50).persist()

        # Persists the Dask Storage Structures
        if client is not None and kwargs.pop("rebalance", False):
            client.rebalance(self._X)
            client.rebalance(self._Y)

        check_X_y(self._X, self._Y, accept_sparse='csc', multi_output=True, distributed=False)

        self._scenario_type = scenario_type
        if self._scenario_type is None:
            raise ValueError("required param 'scenario_type' can not be empty")
        if not issubclass(self._scenario_type, AbstractScenario):
            raise ValueError("the 'scenario_type' must be a subclass of 'AbstractScenario'")

        if self_partition:
            self._kfolds = kfolds
            self._train_idx, self._test_idx, self._label_idx, self._unlabel_idx = split(
                X=self._X,
                y=self._Y,
                test_ratio=kwargs.pop("test_ratio", 0.3),
                initial_label_rate=kwargs.pop("initial_label_rate", 0.05),
                split_count=self._kfolds,
                all_class=kwargs.pop("all_class", True))
        else:
            train_idx = kwargs.pop("train_idx", None)
            test_idx = kwargs.pop("test_idx", None)
            label_idx = kwargs.pop("label_idx", None)
            unlabel_idx = kwargs.pop("unlabel_idx", None)

            if train_idx is None:
                raise ValueError("required param 'train_idx' can not be empty ")
            if test_idx is None:
                raise ValueError("required param 'test_idx' can not be empty ")
            if label_idx is None:
                raise ValueError("required param 'label_idx' can not be empty ")
            if unlabel_idx is None:
                raise ValueError("required param 'unlabel_idx' can not be empty ")

            num_inst_x, num_feat = da.shape(self._X)
            num_inst_y, num_labels = da.shape(self._Y) if len(da.shape(self._Y)) > 1 else (da.shape(self._Y)[0], 1)
            folds_train, num_inst_train = np.shape(train_idx)
            folds_test, num_inst_test = np.shape(test_idx)
            folds_labeled, num_inst_labeled = np.shape(label_idx)
            folds_unlabeled, num_inst_unlabeled = np.shape(unlabel_idx)

            if num_inst_x != num_inst_y:
                raise ValueError("Different numbers of instances for inputs (x:%s, y:%s)" % (num_inst_x, num_inst_y))

            if folds_train != folds_test or folds_test != folds_labeled or folds_labeled != folds_unlabeled:
                raise ValueError("Different numbers of folds for inputs (train_idx:%s, test_idx:%s "
                                 "label_idx:%s, unlabel_idx:%s)"
                                 % (folds_train, folds_test, folds_labeled, folds_unlabeled))
            if kfolds != folds_test:
                raise ValueError("Number of folds for inputs (train_idx:%s, test_idx:%s "
                                 "label_idx:%s, unlabel_idx:%s) must be equals to kfolds:%s param"
                                 % (folds_train, folds_test, folds_labeled, folds_unlabeled, kfolds))

            if num_inst_train + num_inst_test != num_inst_x:
                raise ValueError("The sum of the number of instances for train_idx and test_idx must be equal to the "
                                 "number of instances for x"
                                 "(num_inst_x:%s, num_inst_train:%s num_inst_test:%s)"
                                 % (num_inst_x, num_inst_train, num_inst_test))

            if num_inst_labeled + num_inst_unlabeled != num_inst_train:
                raise ValueError(
                    "The sum of the number of instances for label_idx and unlabel_idx must be equal to the "
                    "number of instances for train_idx"
                    "(num_inst_labeled:%s, num_inst_unlabeled:%s num_inst_unlabeled:%s)"
                    % (num_inst_labeled, num_inst_unlabeled, num_inst_unlabeled))

            self._kfolds = folds_train
            self._train_idx = train_idx
            self._test_idx = test_idx
            self._label_idx = label_idx
            self._unlabel_idx = unlabel_idx

        self._ml_technique = ml_technique
        if self._ml_technique is None:
            raise ValueError("required param 'ml_technique' can not be empty")

        self._performance_metrics = performance_metrics
        if self._performance_metrics is None or len(self._performance_metrics) == 0:
            raise ValueError("required param 'performance_metric' can not be empty")
        else:
            for metric in self._performance_metrics:
                if not isinstance(metric, BaseMetrics):
                    raise ValueError("the elements in 'performance_metrics' must be of type BaseMetrics")

        self._query_strategy = query_strategy
        if self._query_strategy is None:
            raise ValueError("required param 'query_strategy' can not be empty")

        self._oracle = oracle
        if self._oracle is None:
            raise ValueError("required param 'oracle' can not be empty")

        self._stopping_criteria = stopping_criteria
        if self._stopping_criteria is None:
            raise ValueError("required param 'stopping_criteria' can not be empty")

        # Dynamically create the scenario Type given the arguments
        importlib.import_module(self._scenario_type.__module__)

        self._scenario = eval(self._scenario_type.__qualname__)(
            X=self._X, y=self._Y,
            train_idx=self._train_idx[0],
            test_idx=self._test_idx[0],
            label_idx=copy.deepcopy(IndexCollection(self._label_idx[0])),
            unlabel_idx=copy.deepcopy(IndexCollection(self._unlabel_idx[0])),
            ml_technique=self._ml_technique,
            performance_metrics=self._performance_metrics,
            query_strategy=self._query_strategy,
            oracle=self._oracle
        )

    @abstractmethod
    def evaluate(self, **kwargs):
        pass


class HoldOutExperiment(AbstractExperiment, metaclass=ABCMeta):

    def __init__(self,
                 client: Client,
                 X, Y,
                 scenario_type: AbstractScenario,
                 ml_technique,
                 performance_metrics: [],
                 query_strategy: SinlgeLabelIndexQuery,
                 oracle: Oracle,
                 stopping_criteria: AbstractStopCriterion,
                 self_partition: bool,
                 **kwargs):
        """
        Parameters
        ----------
        :param client: distributed.Client
        :param X: array-like
            Data matrix with [n_samples, n_features]
        :param Y: array-like, optional
            labels of given data [n_samples, n_labels] or [n_samples]
        :param scenario_type: Sub-Type of AbstractScenario
            Type of Active Learning scenario to use
        :param ml_technique
        :param performance_metrics: array-like of BaseMetrics elements
        :param query_strategy: SinlgeLabelIndexQuery
        :param oracle: Oracle
        :param stopping_criteria: AbstractStopCriterion
        :param self_partition: bool
        :param kwargs: optional
            Extra parameters
        """
        super().__init__(client=client,
                         X=X, Y=Y,
                         scenario_type=scenario_type,
                         ml_technique=ml_technique,
                         performance_metrics=performance_metrics,
                         query_strategy=query_strategy,
                         oracle=oracle,
                         stopping_criteria=stopping_criteria,
                         self_partition=self_partition,
                         kfolds=1,
                         **kwargs)

    def evaluate(self, client: Client = None, **kwargs):
        folds = [None] * 1

        algorithm = ClassicActiveLearning(
            scenario=self._scenario,
            stopping_criteria=self._stopping_criteria
        )

        # Execute the experiment
        algorithm.execute(client=client, **kwargs)

        # return the experiment results
        folds[0] = algorithm.get_experiment_result()[0]

        return folds


class CrossValidationExperiment(AbstractExperiment, metaclass=ABCMeta):

    def __init__(self,
                 client: Client,
                 X, Y,
                 scenario_type: AbstractScenario,
                 ml_technique,
                 performance_metrics: [],
                 query_strategy: SinlgeLabelIndexQuery,
                 oracle: Oracle,
                 stopping_criteria: AbstractStopCriterion,
                 self_partition: bool,
                 kfolds=5,
                 **kwargs):
        """
        Parameters
        ----------
        :param client: distributed.Client
        :param X: array-like
            Data matrix with [n_samples, n_features]
        :param Y: array-like, optional
            labels of given data [n_samples, n_labels] or [n_samples]
        :param scenario_type: Sub-Type of AbstractScenario
            Type of Active Learning scenario to use
        :param ml_technique
        :param performance_metrics: array-like of BaseMetrics elements
        :param query_strategy: SinlgeLabelIndexQuery
        :param oracle: Oracle
        :param stopping_criteria: AbstractStopCriterion
        :param self_partition: bool
        :param kfolds:
        :param kwargs: optional
            Extra parameters
        """
        super().__init__(client=client,
                         X=X, Y=Y,
                         scenario_type=scenario_type,
                         ml_technique=ml_technique,
                         performance_metrics=performance_metrics,
                         query_strategy=query_strategy,
                         oracle=oracle,
                         stopping_criteria=stopping_criteria,
                         self_partition=self_partition,
                         kfolds=kfolds,
                         **kwargs)

    def call_script(self, ordinal):
        print('Thread %s -> ' % ordinal)
        time.sleep(1)
        print('Thread %s ->  Finished:' % ordinal)
        return ordinal

    def evaluate(self, client: Client = None, rebalance = False, **kwargs):

        import tracemalloc
        tracemalloc.start()

        multithread = kwargs.pop("multithread", True)

        # A dictionary which will contain a list the future info in the key, and the filename in the value
        folds = []

        if multithread:
            max_threads = kwargs.pop('max_threads', os.cpu_count() if os.cpu_count() is not None else 5)

            with mp.get_context("spawn").Pool(processes=max_threads) as pool:
                for k in range(self._kfolds):
                    # foldExec = pool.apply_async(self.call_script, args=(k, ))

                    foldExec = pool.apply_async(
                        ClassicActiveLearning(
                            scenario=copy.deepcopy(self._scenario),
                            stopping_criteria=copy.deepcopy(self._stopping_criteria)
                        ).execute,
                        args=(),
                        kwds=kwargs
                    )
                    folds.append(foldExec)

                output = [p.get() for p in folds]

            return map(lambda item: [item], output)
        else:
            # Loop through the files, and run the parse function for each file, sending the file-name to it, along with the kwargs of parser_variables.
            # The results of the functions can come back in any order.

            for k in range(self._kfolds):
                algorithm = ClassicActiveLearning(
                    scenario=self._scenario,
                    stopping_criteria=copy.deepcopy(self._stopping_criteria)
                )

                # Execute the experiment
                algorithm.execute(**kwargs)

                # return the experiment results
                folds.append(algorithm.get_experiment_result())

        return folds