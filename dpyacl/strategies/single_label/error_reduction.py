"""
Error reduction query strategies implementation

1. QueryExpectedLogLoss
2. QueryExpectedCeroOneLoss

Added distributed processing capabilities with Dask
Modified implementation with an Object Oriented design
"""
# Authors: Ying-Peng Tang, Alfredo Lorie


import collections
import functools
import operator
from abc import ABCMeta

import dask.array as da
import numpy as np
from dask import delayed
from distributed import Client

from ..stategies import SinlgeLabelIndexQuery, ExpectedErrorReductionStrategy
from ...core.misc import nlargestarg, nsmallestarg

__all__ = ['QueryExpectedLogLoss', 'QueryExpectedCeroOneLoss', 'QueryRegressionStd']


# ----------------------------------------------------------------------------------------------------------------------
# Expected Error Reduction Strategy
# ----------------------------------------------------------------------------------------------------------------------
class QueryExpectedCeroOneLoss(ExpectedErrorReductionStrategy, metaclass=ABCMeta):
    """

    """

    @property
    def query_function_name(self):
        return "ExpectedCeroOneLoss"

    def _loss(self, prob):
        """Compute expected log-loss.

        Parameters
        ----------
        prob: 2d array, shape [n_samples, n_classes]
            The probabilistic prediction matrix for the unlabeled set.

        Returns
        -------
        log_loss: float
            The sum of log_loss for the prob.
        """
        log_loss = []
        for i in range(len(prob)):
            for p in list(prob[i]):
                log_loss.append(delayed(1 - p ))
        return delayed(sum)(log_loss).compute()

    def select(self, X, y, label_index, unlabel_index, model=None, client:Client = None):
        innner_unlabel_index, scores = super().select(X, y, label_index, unlabel_index, model=model, client=client)
        return self._select_by_prediction(unlabel_index=innner_unlabel_index, predict=scores)

    def _select_by_prediction(self, unlabel_index, predict):
        """
        Perform basic validation for indexes selection for queryin

        Parameters
        ----------
        :param unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples. Should be one-to-one
            correspondence to the prediction matrix.
        :param predict: array, [n_samples, n_classes]
            The prediction matrix for the unlabeled set.
        :param kwargs: optional
        """

        return unlabel_index[tuple(nsmallestarg(predict, self._batch_size))]


class QueryExpectedLogLoss(ExpectedErrorReductionStrategy, metaclass=ABCMeta):
    """

    """

    @property
    def query_function_name(self):
        return "ExpectedLogLoss"

    def _loss(self, prob):
        """Compute expected log-loss.

        Parameters
        ----------
        prob: 2d array, shape [n_samples, n_classes]
            The probabilistic prediction matrix for the unlabeled set.

        Returns
        -------
        log_loss: float
            The sum of log_loss for the prob.
        """
        log_loss = []
        for i in range(len(prob)):
            for p in list(prob[i]):
                log_loss. append(delayed(p * da.log(p)))
        return delayed(sum)(log_loss).compute()

    def select(self, X, y, label_index, unlabel_index, model=None, client:Client = None):
        innner_unlabel_index, scores = super().select(X, y, label_index, unlabel_index, model=model, client=client)
        return self._select_by_prediction(unlabel_index=innner_unlabel_index, predict=scores)

    def _select_by_prediction(self, unlabel_index, predict):
        """
        Perform basic validation for indexes selection for queryin

        Parameters
        ----------
        :param unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples. Should be one-to-one
            correspondence to the prediction matrix.
        :param predict: array, [n_samples, n_classes]
            The prediction matrix for the unlabeled set.
        :param kwargs: optional
        """

        return unlabel_index[tuple(nsmallestarg(predict, self._batch_size))]


class QueryRegressionStd(SinlgeLabelIndexQuery, metaclass=ABCMeta):
    """
    """

    @property
    def query_function_name(self):
        return "QueryRegressionStd"

    def select(self, X, y, label_index, unlabel_index, model=None, client: Client = None):
        """Select indexes randomly.
        Parameters
        ----------
        label_index: object
            Add this parameter to ensure the consistency of api of strategies.
            Please ignore it.
        unlabel_index: collections.abc.Iterable
            The indexes of unlabeled set.
        batch_size: int, optional (default=1)
            Selection batch size.
        Returns
        -------
        selected_idx: list
            The selected indexes which is a subset of unlabel_index.
        """
        super().select(X, y, label_index, unlabel_index, model=model)

        if X is None:
            raise Exception('Data matrix is not provided')

        if model is None:
            raise Exception('Model is not provided.')

        assert (self._batch_size > 0)
        assert (isinstance(unlabel_index, collections.abc.Iterable))
        unlabel_index = np.asarray(unlabel_index)

        if len(unlabel_index) <= self._batch_size:
            return unlabel_index

        unlabel_x = X[unlabel_index, :]

        pred, std = self._get_pred(unlabel_x, model, proba=False, return_std=True)
        return self._select_by_prediction(unlabel_index=unlabel_index,
                                          predict=da.from_array(functools.reduce(operator.iconcat, pred, [])),
                                          std=da.from_array(std))

    def _select_by_prediction(self, unlabel_index, predict, **kwargs):

        predict_shape = np.shape(predict)
        assert (len(predict_shape) in [1, 2])

        std = kwargs.pop("std", None)
        if std is None:
            raise ValueError("")
        else:
            tpl = da.from_array(unlabel_index)
            return tpl[nlargestarg(std, self._batch_size)].compute()
