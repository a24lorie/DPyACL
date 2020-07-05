"""
Single Label active learning strategies

Implementation with an Object Oriented design
Distributed processing capabilities with Dask
"""
# Authors: Alfredo Lorie

import collections.abc
import copy
import warnings
from abc import abstractmethod, ABCMeta

import dask.array as da
import joblib
import numpy as np
from dask import delayed
from distributed import Client
from sklearn.ensemble import BaggingClassifier
from sklearn.utils import check_X_y

__all__ = [
    'BaseQueryStrategy',
    'SinlgeLabelIndexQuery',
    'InstanceUncertaintyStrategy',
    'ExpectedErrorReductionStrategy',
    'QueryByCommitteeStategy'
]

# ----------------------------------------------------------------------------------------------------------------------
# Abstract Base Query Class
# ----------------------------------------------------------------------------------------------------------------------
from dpyacl.core.misc import nlargestarg


class BaseQueryStrategy(metaclass=ABCMeta):
    """
    Base strategies class.
    """

    def __init__(self, batch_size=1):
        """

        :param batch_size: int, default=1
            Selection batch size.
        """
        assert (batch_size > 0)
        self._batch_size = batch_size

    @property
    @abstractmethod
    def query_function_name(self):
        pass

    @abstractmethod
    def select(self, *args, **kwargs):
        pass


# ----------------------------------------------------------------------------------------------------------------------
# Base Query Index Class
# ----------------------------------------------------------------------------------------------------------------------
class SinlgeLabelIndexQuery(BaseQueryStrategy, metaclass=ABCMeta):
    """
    The base class for the selection method which imposes a constraint on the parameters of select()
    """

    @property
    @abstractmethod
    def query_function_name(self):
        pass

    @abstractmethod
    def select(self, X, y, label_index, unlabel_index, **kwargs):
        """
        Select instances to strategies

        Parameters
        ----------
        :param X: array
            The [n_samples, n_features] training samples with n-features per instance.
        :param y: array
            The The [n_samples] label vector.
        :param label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.
        :param unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples.
        """

        if X is not None and y is not None:
            check_X_y(X, y, accept_sparse='csc', multi_output=True)

    def _get_pred(self, unlabel_x, model, proba=True, **kwargs):
        """
        Get the prediction results of the unlabeled set.

        Parameters
        ----------
        :param unlabel_x: array
            The [n_samples, n_features]  matrix of the unlabeled set.
        :param model: object
            Model object which has the prediction capabilities.
        :param proba: bool
            Whether to get the prediction  for the unlabeled dataset or the prediction probabilities
        :param kwargs: optional

        Returns
        -------
        pv: dask.array
            Probability predictions matrix with shape [n_samples, n_classes].
        """

        if proba:
            if not hasattr(model, 'predict_proba'):
                raise Exception('model object must implement predict_proba methods in current algorithm.')
            proba = model.predict_proba(unlabel_x)
            pv = da.asarray(proba)
            spv = da.shape(pv)
            if len(spv) != 2 or spv[1] == 1:
                raise Exception('2d array with [n_samples, n_class] is expected, but received: \n%s' % str(pv))
            return pv
        else:
            return model.predict(unlabel_x, kwargs)

    @abstractmethod
    def _select_by_prediction(self, unlabel_index, predict, **kwargs):
        """
        Perform basic validation for indexes selection for queryin

        Parameters
        ----------
        :param unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples. Should be one-to-one
            correspondence to the prediction matrix.
        :param predict: dask.array, [n_samples, n_classes]
            The prediction matrix for the unlabeled set.
        :param kwargs: optional
        """

        assert (self._batch_size > 0)
        assert (isinstance(unlabel_index, collections.abc.Iterable))
        unlabel_index = np.asarray(unlabel_index)
        if len(unlabel_index) <= self._batch_size:
            return unlabel_index

        predict_shape = da.shape(predict)

        if len(predict_shape) != 2 or predict_shape[1] == 1:
            raise Exception('2d array with the shape [n_samples, n_classes]'
                            ' is expected, but received shape: \n%s' % str(predict_shape))


# ----------------------------------------------------------------------------------------------------------------------
# Uncertainty Sampling Strategy
# ----------------------------------------------------------------------------------------------------------------------
class InstanceUncertaintyStrategy(SinlgeLabelIndexQuery, metaclass=ABCMeta):
    """
    Uncertainty strategies strategy, this type of strategy needs a probabilistic model.

    The implement of uncertainty measure includes:
    1. margin sampling
    2. least confident
    3. entropy
    4. distance to boundary
    """

    @property
    def query_function_name(self):
        return "InstanceUncertaintyStrategy"

    def select(self, X, y, label_index, unlabel_index, model=None, client: Client = None):
        """
        Select indexes from the unlabel_index for querying.

        Parameters
        ----------
        :param X: array
            The [n_samples, n_features] training samples with n-features per instance.
        :param y: array
            The The [n_samples] label vector.
        :param label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.
        :param unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples.
        :param model: object, optional (default=None)
            Current classification model, should have the 'predict_proba' method for probabilistic output.
            If not provided, LogisticRegression with default parameters implemented by sklearn will be used.
        :param client:

        Returns
        -------
        selected_idx: list
            The selected indexes which is a subset of unlabel_index.
        """
        super().select(X, y, label_index, unlabel_index, model=model)

        if X is None:
            raise Exception('Data matrix is not provided, use select_by_prediction_mat() instead.')

        if model is None:
            raise Exception('Model is not provided.')

        assert (self._batch_size > 0)
        assert (isinstance(unlabel_index, collections.abc.Iterable))
        unlabel_index = np.asarray(unlabel_index)

        # # TODO cambiar
        # if len(unlabel_index) <= self._batch_size:
        #     return unlabel_index

        return unlabel_index, self._get_pred(X[unlabel_index, :], model, proba=True)


# ----------------------------------------------------------------------------------------------------------------------
# Expected Error Reduction Strategy
# ----------------------------------------------------------------------------------------------------------------------
class ExpectedErrorReductionStrategy(SinlgeLabelIndexQuery, metaclass=ABCMeta):
    """
    The objective of this active learning strategy is to reduce the expected total number of incorrect predictions.

    Parameters
    ----------
    X: 2D array, optional (default=None)
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.

    y: array-like, optional (default=None)
        Label matrix of the whole dataset. It is a reference which will not use additional memory.

    References
    ----------
    Burr Settles. Active Learning Literature Survey. Computer Sciences Technical Report 1648,
    University of Wisconsin–Madison. 2009.
    """

    @property
    def query_function_name(self):
        return "ExpectedErrorReductionStrategy"

    @abstractmethod
    def _loss(self, prob):
        """Compute expected -loss.

        Parameters
        ----------
        prob: 2d array, shape [n_samples, n_classes]
            The probabilistic prediction matrix for the unlabeled set.

        Returns
        -------
        log_loss: float
            The sum of _loss for the prob.
        """
        pass

    def select(self, X, y, label_index, unlabel_index, model=None, client: Client = None):
        """
        Select indexes from the unlabel_index for querying.

        Parameters
        ----------

        :param X: array
            The [n_samples, n_features] training samples with n-features per instance.
        :param y: array
            The The [n_samples] label vector.
        :param label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.
        :param unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples.
        :param model: object, optional (default=None)
            Current classification model, should have the 'predict_proba' method for probabilistic output.
            If not provided, LogisticRegression with default parameters implemented by sklearn will be used.
        :param client:

        Returns
        -------
        selected_idx: list
            The selected indexes which is a subset of unlabel_index.
        """

        assert (isinstance(unlabel_index, collections.Iterable))
        assert (isinstance(label_index, collections.Iterable))

        #  TODO cambiar
        # if len(unlabel_index) <= self._batch_size:
        #     return unlabel_index

        if X is None or y is None:
            raise Exception('Data matrix is not provided.')

        if model is None:
            raise Exception('Model is not provided.')

        label_index = np.asarray(label_index)
        unlabel_index = np.asarray(unlabel_index)

        scores = da.from_array([])
        classes = da.unique(y).compute()
        pv = self._get_pred(X[unlabel_index, :], proba=True, model=model)
        predict_shape = da.shape(pv)

        # for each class
        for i in range(predict_shape[0]):
            new_train_X = delayed(
                X[da.concatenate([da.from_array(label_index), da.from_array([unlabel_index[i]])], axis=0), :])
            unlabel_ind = list(unlabel_index)
            unlabel_ind.pop(i)
            new_unlabel_X = delayed(X[unlabel_ind, :])
            score = da.from_array([])

            for yi in classes:
                new_model = delayed(copy.deepcopy(model))

                if client is not None:
                    with joblib.parallel_backend("dask"):
                        delayed(new_model.fit(new_train_X,
                                              y[da.concatenate([da.from_array(label_index), da.from_array([yi])],
                                                               axis=0)]))
                        prob = delayed(new_model.predict_proba(new_unlabel_X))
                else:
                    delayed(new_model.fit(new_train_X,
                                          y[da.concatenate([da.from_array(label_index), da.from_array([yi])], axis=0)]))
                    prob = delayed(new_model.predict_proba(new_unlabel_X))

                score = da.concatenate([score, da.from_array([pv[i, yi] * self._loss(prob.compute())])], axis=0)

            scores = da.concatenate([scores, da.from_array([da.sum(score)])], axis=0)

        return unlabel_index, scores


# ----------------------------------------------------------------------------------------------------------------------
# Query By Committee Strategy
# ----------------------------------------------------------------------------------------------------------------------
class QueryByCommitteeStategy(SinlgeLabelIndexQuery):
    """Query-By-Committee Strategy

    Parameters
    ----------
    method: str, optional (default=query_by_bagging)
        Method name. This class only implement query_by_bagging for now.

    disagreement: str
        method to calculate disagreement of committees. should be one of ['vote_entropy', 'KL_divergence']

    References
    ----------
    [1] H.S. Seung, M. Opper, and H. Sompolinsky. Query by committee.
        In Proceedings of the ACM Workshop on Computational Learning Theory,
        pages 287-294, 1992.

    [2] N. Abe and H. Mamitsuka. Query learning strategies using boosting and bagging.
        In Proceedings of the International Conference on Machine Learning (ICML),
        pages 1-9. Morgan Kaufmann, 1998.
    """

    def __init__(self, batch_size=1, n_jobs=None):
        """

        :param batch_size: int, default=1
            Selection batch size.
        """
        assert (batch_size > 0)
        self._batch_size = batch_size
        self._n_jobs = n_jobs

    @property
    def query_function_name(self):
        return "QueryByCommiteStategy"

    @abstractmethod
    def agreement(self, estimators):
        pass

    def select(self, X, y, label_index, unlabel_index, model=None, client: Client = None):
        """Select indexes from the unlabel_index for querying.

        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.

        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples.

        model: object, optional (default=None)
            Current classification model, should have the 'predict_proba' method for probabilistic output.
            If not provided, LogisticRegression with default parameters implemented by sklearn will be used.

        n_jobs: int, optional (default=None)
            How many threads will be used in training bagging.

        Returns
        -------
        selected_idx: list
            The selected indexes which is a subset of unlabel_index.
        """

        assert (self._batch_size > 0)
        assert (isinstance(unlabel_index, collections.Iterable))
        assert (isinstance(label_index, collections.Iterable))

        label_index = np.asarray(label_index)
        unlabel_index = np.asarray(unlabel_index)

        if len(unlabel_index) <= self._batch_size:
            return unlabel_index

        if X is None or y is None:
            raise Exception('Data matrix is not provided, use select_by_prediction_mat() instead.')

        if model is None:
            raise Exception('Model is not provided.')

        # bagging
        if client is not None:
            with joblib.parallel_backend("dask"):
                bagging = BaggingClassifier(model, n_jobs=self._n_jobs)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    bagging.fit(X[label_index, :], y[label_index,])
        else:
            bagging = BaggingClassifier(model, n_jobs=self._n_jobs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bagging.fit(X[label_index, :], y[label_index,])

        return unlabel_index, bagging.estimators_

    def _select_by_prediction(self, unlabel_index, predict):
        """
        Perform basic validation for indexes selection for queryin

        Parameters
        ----------
        :param unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples. Should be one-to-one
            correspondence to the prediction matrix.
        :param predict: dask.array, [n_samples, n_classes]
            The prediction matrix for the unlabeled set.
        :param kwargs: optional
        """
        return unlabel_index[tuple(nlargestarg(predict, self._batch_size))]

    @staticmethod
    def check_committee_results(estimators):
        """check the validity of given committee predictions.

        Parameters
        ----------
        predict_matrices: list
            The prediction matrix for each committee.
            Each committee predict matrix should have the shape [n_samples, n_classes] for probabilistic output
            or [n_samples] for class output.

        Returns
        -------
        input_shape: tuple
            The shape of the predict_matrix

        committee_size: int
            The number of committees.

        """
        shapes = [np.shape(X) for X in estimators if X is not None]
        uniques = np.unique(shapes, axis=0)
        if len(uniques) > 1:
            raise Exception("Found input variables with inconsistent numbers of"
                            " shapes: %r" % [int(l) for l in shapes])
        committee_size = len(estimators)
        if not committee_size > 1:
            raise ValueError("Two or more committees are expected, but received: %d" % committee_size)
        input_shape = uniques[0]
        return input_shape, committee_size
