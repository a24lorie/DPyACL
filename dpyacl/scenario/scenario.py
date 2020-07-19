import inspect
from abc import ABCMeta, abstractmethod

import dask.array as da
import joblib
import numpy as np
from dask import compute
from dask.delayed import delayed
from distributed import Client

from ..core.collections import IndexCollection
from ..core.state import State
from ..metrics.evaluation import BaseMetrics
from ..oracle import Oracle
from ..strategies.single_label import QueryInstanceRandom
from ..strategies.stategies import SingleLabelIndexQuery

__all__ = ['AbstractScenario', 'PoolBasedSamplingScenario']


class AbstractScenario(metaclass=ABCMeta):

    def __init__(self,
                 X, y,
                 train_idx,
                 test_idx,
                 label_idx: IndexCollection,
                 unlabel_idx: IndexCollection,
                 ml_technique,
                 performance_metrics: [],
                 query_strategy: SingleLabelIndexQuery = QueryInstanceRandom(),
                 oracle: Oracle = None,
                 batch_size=1,
                 **kwargs):

        self._X = X
        self._Y = y

        self._train_idx = train_idx
        self._test_idx = test_idx
        self._label_idx: IndexCollection = label_idx
        self._unlabel_idx: IndexCollection = unlabel_idx

        self._performance_metrics = performance_metrics
        if self._performance_metrics is None or len(self._performance_metrics) == 0:
            raise ValueError("required param 'performance_metric' can not be empty")
        else:
            for metric in self._performance_metrics:
                if not isinstance(metric, BaseMetrics):
                    raise ValueError("the elements in 'performance_metrics' must be of type BaseMetrics")
        self._metrics = True

        self._ml_technique = ml_technique
        if self._ml_technique is None:
            raise ValueError("required param 'ml_technique' can not be empty")

        self._query_strategy = query_strategy
        if self._query_strategy is None:
            raise ValueError("required param 'query_strategy' can not be empty")

        self._oracle = oracle
        if self._oracle is None:
            raise ValueError("required param 'simOracle' can not be empty")

        self._scenario_result = []
        self._batch_size = batch_size

    def initIteration(self, verbose, **kwargs):

        initial_point = kwargs.pop('initial_point', None)
        check_flag = kwargs.pop('check_flag', True)
        print_interval = kwargs.pop('print_interval', 1)

        return State(round=0,
                     train_idx=self._train_idx, test_idx=self._test_idx,
                     init_L=self._label_idx, init_U=self._unlabel_idx,
                     performance_metrics=[metric.metric_name for metric in self._performance_metrics],
                     initial_point=initial_point, check_flag=check_flag,
                     verbose=verbose, print_interval=print_interval)

    def remainingUnlabeledInstances(self):
        return len(self._unlabel_idx) > 0

    def executeLabeledTraining(self, client: Client = None):
        # Train Model over the labeled instances
        if client is not None:
            with joblib.parallel_backend("dask"):
                self._ml_technique.fit(X=self._X[self._label_idx.index, :], y=self._Y[self._label_idx.index])

                # predict the results over the labeled test instances
                label_pred = self._ml_technique.predict(self._X[self._test_idx, :])
        else:
            self._ml_technique.fit(X=self._X[self._label_idx.index, :], y=self._Y[self._label_idx.index])
            # predict the results over the labeled test instances
            label_pred = self._ml_technique.predict(self._X[self._test_idx, :])

        # performance calc for all metrics
        label_perf = []
        for metric in self._performance_metrics:
            value = delayed(metric.compute(y_true=self._Y[self._test_idx], y_pred=label_pred))
            label_perf.append(delayed({"name": metric.metric_name, "value": value}))

        return label_pred, compute(label_perf)[0]

    def labelInstances(self, select_ind, client: Client = None, verbose=False):
        # For each selected instance retrieve from the simOracle the labeled instances
        labels, cost = self._oracle.query(instances = self._X[select_ind], indexes=select_ind)

        labels_iterator =  zip(select_ind, labels)

        for item in labels_iterator:
            if select_ind == 0: # choose the first item
                result = da.concatenate([
                                item[1] if isinstance(item[1], (list, np.ndarray)) else [item[1]],
                                self._Y[item[0]+1:]
                ], axis=0)
            elif select_ind is len(self._Y) - 1: # choose the last item
                result = da.concatenate([
                                self._Y[: item[0]],
                                item[1] if isinstance(item[1], (list, np.ndarray)) else [item[1]]
                ], axis=0)
            else: # any other item
                result = da.concatenate([
                                self._Y[: item[0]],
                                item[1] if isinstance(item[1], (list, np.ndarray)) else [item[1]],
                                self._Y[item[0] + 1:]
                ],axis=0)

        if client is not None:
            self._Y = result.persist()
            client.rebalance(self._Y)
        else:
            self._Y = result

        if verbose:
            print("Label: %s, Cost: %s" % (labels, cost))

    def updateLabelledData(self, select_ind):
        self._label_idx.update(select_ind)
        self._unlabel_idx.difference_update(select_ind)

    @abstractmethod
    def selectInstances(self, client: Client = None):
        pass


class PoolBasedSamplingScenario(AbstractScenario, metaclass=ABCMeta):

    def __init__(self,
                 X, y,
                 train_idx,
                 test_idx,
                 label_idx: IndexCollection,
                 unlabel_idx: IndexCollection,
                 ml_technique,
                 performance_metrics: [],
                 query_strategy: SingleLabelIndexQuery,
                 oracle: Oracle = None,
                 batch_size=1,
                 **kwargs):

        super().__init__(
            X=X, y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            label_idx=label_idx,
            unlabel_idx=unlabel_idx,
            ml_technique=ml_technique,
            performance_metrics=performance_metrics,
            query_strategy=query_strategy,
            oracle=oracle,
            batch_size=batch_size,
            **kwargs)

    def selectInstances(self, client: Client = None):
        if 'model' in inspect.getfullargspec(self._query_strategy.select)[0]:
            return self._query_strategy.select(X=self._X,
                                               y=self._Y,
                                               label_index=self._label_idx,
                                               unlabel_index=self._unlabel_idx,
                                               batch_size=self._batch_size,
                                               model=self._ml_technique,
                                               client=client)
        else:
            return self._query_strategy.select(X=self._X,
                                               y=self._Y,
                                               label_index=self._label_idx,
                                               unlabel_index=self._unlabel_idx,
                                               batch_size=self._batch_size)

