"""
Query By Comittee strategies implementation

1. QueryVoteEntropy
2. QueryKullbackLeiblerDivergence

Added distributed processing capabilities with Dask
Modified implementation with an Object Oriented design
"""
# Authors: Ying-Peng Tang, Alfredo Lorie

import collections
from abc import ABCMeta
from multiprocessing.connection import Client

import dask.array as da
from dask import delayed, compute
import numpy as np

from ..stategies import QueryByCommitteeStategy

__all__ = ['QueryVoteEntropy', 'QueryKullbackLeiblerDivergence']


# ----------------------------------------------------------------------------------------------------------------------
# Expected Error Reduction Strategy
# ----------------------------------------------------------------------------------------------------------------------
class QueryVoteEntropy(QueryByCommitteeStategy, metaclass=ABCMeta):
    """

    """

    @property
    def query_function_name(self):
        return "VoteEntropy"

    def agreement(self, estimators):
        """
        Implementation of Query By Committee strategy, variant: Vote entropy.

        The vote entropy approach is used for measuring the level of disagreement.

        I. Dagan and S. Engelson. Committee-based sampling for training probabilistic
        classifiers. In Proceedings of the International Conference on Machine
        Learning (ICML), pages 150–157. Morgan Kaufmann, 1995.

        :param estimators:
        :return:
        """
        score = []
        input_shape, committee_size = QueryByCommitteeStategy.check_committee_results(estimators)
        if len(input_shape) == 2:
            ele_uni = da.unique(estimators).compute()
            if not (len(ele_uni) == 2 and 0 in ele_uni and 1 in ele_uni):
                raise ValueError("The predicted label matrix must only contain 0 and 1")

            # calc each instance
            for i in range(input_shape[0]):
                instance_mat = da.from_array(np.array([X[i, :] for X in estimators if X is not None])).compute()
                voting = da.sum(instance_mat, axis=0)

                tmp = []
                for vote in voting:
                    if vote != 0:
                        tmp.append(delayed(vote / len(estimators) * np.log(vote / len(estimators))))
                score.append(-delayed(sum)(tmp))
        else:
            input_mat = da.from_array(np.array([X for X in estimators if X is not None])).compute()
            # for each instance
            for i in range(input_shape[0]):
                count_dict = collections.Counter(input_mat[:, i])
                tmp = []
                for key in count_dict:
                    tmp.append(delayed(count_dict[key] / committee_size * np.log(count_dict[key] / committee_size)))
                score.append(-delayed(sum)(tmp))

        return compute(score)[0]

    def select(self, X, y, label_index, unlabel_index, model=None, client: Client = None):
        unlabel_index = np.asarray(unlabel_index)
        innner_unlabel_index, estimators = super().select(X, y, label_index, unlabel_index, model=model, client=client)
        return self._select_by_prediction(unlabel_index=innner_unlabel_index,
                                          predict=self.agreement([estimator.predict(X[unlabel_index, :]) for estimator in estimators]))

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
        """
        return super()._select_by_prediction(unlabel_index, predict)


class QueryKullbackLeiblerDivergence(QueryByCommitteeStategy, metaclass=ABCMeta):
    """

    """

    @property
    def query_function_name(self):
        return "KullbackLeiblerDivergence"

    def agreement(self, estimators):
        """
        Implementation of Query By Committee query strategy, variant:
        
        Kullback-Leibler (KL) divergence.
        
        
        A. McCallum and K. Nigam. Employing EM in pool-based active learning for text
        classification. In Proceedings of the International Conference on Machine
        Learning (ICML), pages 359–367. Morgan Kaufmann, 1998.

        :param estimators:
        :return:
        """
        score = []
        input_shape, committee_size = QueryByCommitteeStategy.check_committee_results(estimators)
        if len(input_shape) == 2:
            label_num = input_shape[1]
            # calc kl div for each instance
            for i in range(input_shape[0]):
                instance_mat = da.from_array(np.array([X[i, :] for X in estimators if X is not None]))
                tmp = []
                # calc each label
                for lab in range(label_num):
                    committee_consensus = da.sum(instance_mat[:, lab]) / committee_size
                    for committee in range(committee_size):
                        tmp.append(delayed(instance_mat[committee, lab] * np.log(instance_mat[committee, lab] / committee_consensus)))
                score.append(delayed(sum)(tmp))
        else:
            raise Exception(
                "A 2D probabilistic prediction matrix must be provided, with the shape like [n_samples, n_class]")
        return compute(score)[0]

    def select(self, X, y, label_index, unlabel_index, model=None, client: Client = None):
        unlabel_index = np.asarray(unlabel_index)
        innner_unlabel_index, estimators = super().select(X, y, label_index, unlabel_index, model=model, client=client)
        return self._select_by_prediction(unlabel_index=innner_unlabel_index,
                                          predict=self.agreement(
                                              [estimator.predict_proba(X[unlabel_index, :]) for estimator in estimators]))

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
        """
        return super()._select_by_prediction(unlabel_index, predict)
