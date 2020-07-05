from abc import ABCMeta

import numpy as np

import dask.array as da
from ..stategies import SinlgeLabelIndexQuery
from ...core.misc import randperm

__all__ = ['QueryInstanceRandom']


# ----------------------------------------------------------------------------------------------------------------------
# Random Sampling Strategy
# ----------------------------------------------------------------------------------------------------------------------
class QueryInstanceRandom(SinlgeLabelIndexQuery, metaclass=ABCMeta):
    """
    Randomly sample a batch of indexes from the unlabeled indexes.
    """

    @property
    def query_function_name(self):
        return "InstanceRandom"

    def select(self, X, y, label_index, unlabel_index, **kwargs):
        """
        Select indexes randomly

        Parameters
        ----------
        :param X: array
            The [n_samples, n_features] training samples with n-features per instance.
        :param y: array
            The The [n_samples] label vector.
        :param label_index: object
            Add this parameter to ensure the consistency of api of strategies.
            Please ignore it.
        :param unlabel_index: collections.abc.Iterable
            The indexes of unlabeled set.

        Return
        -------
        :return selected_idx: list
            The selected indexes which is a subset of unlabel_index.
        """

        super().select(X, y, label_index, unlabel_index,  **kwargs)

        if len(unlabel_index) <= self._batch_size:
            return np.array(unlabel_index)

        tpl = da.from_array(unlabel_index.index)
        return tpl[randperm(len(unlabel_index) - 1, self._batch_size)].compute()

    def _select_by_prediction(self, unlabel_index, predict, **kwargs):
        pass