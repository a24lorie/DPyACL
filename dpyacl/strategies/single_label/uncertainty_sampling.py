"""
Uncertainty Sampling strategies implementation

1. QueryEntropySampling
2. QueryLeastConfidentSampling
3. QueryMarginSampling
4. QueryDistanceToBoundarySampling

Added distributed processing capabilities with Dask
Modified implementation with an Object Oriented design
"""
# Authors: Ying-Peng Tang, Alfredo Lorie

from abc import ABCMeta

import dask.array as da
import numpy as np
from dask import delayed
from distributed import Client

from ..stategies import InstanceUncertaintyStrategy
from ...core.misc import nsmallestarg, nlargestarg

__all__ = ['QueryEntropySampling',
           'QueryLeastConfidentSampling',
           'QueryMarginSampling',
           'QueryDistanceToBoundarySampling'
           ]


# ----------------------------------------------------------------------------------------------------------------------
# Uncertainty Sampling Strategy
# ----------------------------------------------------------------------------------------------------------------------
class QueryEntropySampling(InstanceUncertaintyStrategy, metaclass=ABCMeta):
    """
    Entropy Sampling strategies strategy.
    Measurement to calculate uncertainty:
        -- 'entropy' x* = argmax -sum(P(yi|x)logP(yi|x))

        --'distance_to_boundary' Only available in binary classification, x* = argmin |f(x)|,
            your model should have 'decision_function' method which will return a 1d array.
    """

    @property
    def query_function_name(self):
        return "EntropySampling"

    def select(self, X, y, label_index, unlabel_index, model=None, client: Client = None):
        unlabel_idx, pv = super().select(X, y, label_index, unlabel_index, model=model, client=client)
        return self._select_by_prediction(unlabel_index=unlabel_idx, predict=pv)

    def _select_by_prediction(self, unlabel_index, predict):
        super()._select_by_prediction(unlabel_index, predict)

        # calc entropy
        predict[predict <= 0] = 1e-06  # avoid zero division

        entro = []
        for vec in predict:
            entro.append(delayed(sum)(vec * da.log(vec)))

        tpl = da.from_array(unlabel_index)
        return tpl[nlargestarg(delayed(entro).compute(), self._batch_size)].compute()


class QueryLeastConfidentSampling(InstanceUncertaintyStrategy, metaclass=ABCMeta):
    """
    Entropy Sampling strategies strategy.
    Measurement to calculate uncertainty:
        --'least_confident' x* = argmax 1-P(y_hat|x) ,where y_hat = argmax P(yi|x)
    """

    @property
    def query_function_name(self):
        return "LeastConfidentSampling"

    def select(self, X, y, label_index, unlabel_index, model=None, client: Client = None):
        unlabel_idx, pv = super().select(X, y, label_index, unlabel_index, model=model, client=client)
        return self._select_by_prediction(unlabel_index=unlabel_idx, predict=pv)

    def _select_by_prediction(self, unlabel_index, predict):
        """Select indexes from the unlabel_index for querying.

        Parameters
        ----------
        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples. Should be one-to-one
            correspondence to the prediction matrix.
        predict: 2d array, shape [n_samples, n_classes]
            The probabilistic prediction matrix for the unlabeled set.
        batch_size: int, optional (default=1)
            Selection batch size.
        Returns
        -------
        selected_idx: list
            The selected indexes which is a subset of unlabel_index.
        """
        super()._select_by_prediction(unlabel_index, predict)
        predict_shape = np.shape(predict)

        # calc least_confident
        pat = predict.map_blocks(lambda vec: np.sort(vec))
        tpl = da.from_array(unlabel_index)
        return tpl[nlargestarg(1 - pat[:, predict_shape[1] - 1], self._batch_size)].compute()


class QueryMarginSampling(InstanceUncertaintyStrategy, metaclass=ABCMeta):
    """
    Entropy Sampling strategies strategy.
    Measurement to calculate uncertainty:
        --'margin' x* = argmax P(y_hat1|x) - P(y_hat2|x), where y_hat1 and y_hat2 are the first and second
                most probable class labels under the model, respectively.
    """

    @property
    def query_function_name(self):
        return "MarginSamplingQuery"

    def select(self, X, y, label_index, unlabel_index, model=None, client: Client = None):
        unlabel_idx, pv = super().select(X, y, label_index, unlabel_index, model=model, client=client)
        return self._select_by_prediction(unlabel_index=unlabel_idx, predict=pv)

    def _select_by_prediction(self, unlabel_index, predict):
        super()._select_by_prediction(unlabel_index, predict)
        predict_shape = np.shape(predict)

        # calc margin
        pat = predict.map_blocks(lambda vec: np.sort(vec))
        tpl = da.from_array(unlabel_index)
        return tpl[
            nlargestarg(pat[:, predict_shape[1] - 2] - pat[:, predict_shape[1] - 1], self._batch_size)].compute()


class QueryDistanceToBoundarySampling(InstanceUncertaintyStrategy, metaclass=ABCMeta):
    """
   Entropy Sampling strategies strategy.
   Measurement to calculate uncertainty:
       --'distance_to_boundary' Only available in binary classification, x* = argmin |f(x)|,
                your model should have 'decision_function' method which will return a 1d array.
   """

    @property
    def query_function_name(self):
        return "DistanceToBoundary"

    def select(self, X, y, label_index, unlabel_index, model=None, client: Client = None):
        if not hasattr(model, 'decision_function'):
            raise TypeError(
                'model object must implement decision_function methods in distance_to_boundary measure.')

        unlabel_idx = np.asarray(unlabel_index)
        unlabel_data = X[unlabel_idx, :]

        return self._select_by_prediction(
                            unlabel_index=unlabel_idx,
                            predict=da.from_array(model.decision_function(unlabel_data)))

    def _select_by_prediction(self, unlabel_index, predict):
        predict_shape = da.shape(predict)

        assert (len(predict_shape) in [1, 2])
        if len(predict_shape) == 2:
            if predict_shape[1] != 1:
                raise Exception('1d or 2d with 1 column array is expected, but received: \n%s' % str(predict))
            else:
                pv = da.absolute(predict.flatten())
        else:
            pv = da.absolute(predict)

        tpl = da.from_array(unlabel_index)
        return tpl[nsmallestarg(pv, self._batch_size)].compute()
