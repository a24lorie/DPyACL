"""
Pre-defined oracle class
Implement classical situation
"""
# Authors: Ying-Peng Tang, Alfredo Lorie

import os
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.utils.validation import check_array

from dpyacl.core.collections import BaseCollection
from dpyacl.core.misc.misc import check_one_to_one_correspondence, unpack

__all__ = ['Oracle',
           'SimulatedOracleQueryIndex'
           ]


class AbstractOracle(metaclass=ABCMeta):
    """
    Basic class of virtual Oracle for experiment
    This class will build a dictionary between index-label in the __init__().
    When querying, the queried_index should be one of the key in the dictionary.
    And the label which corresponds to the key will be returned.
    """

    @abstractmethod
    def query_by_index(self, indexes):
        """Return cost and queried info.
        Parameters
        ----------
        indexes: array
            Queried indexes.
        Returns
        -------
        Labels of queried indexes AND cost
        """
        pass


class Oracle(AbstractOracle, metaclass=ABCMeta):
    """Oracle in active learning whose role is to label the given strategies.
    This class implements basic definition of oracle used in experiment.
    Oracle can provide information given both instance or index. The returned
    information is depending on specific scenario.
    Parameters
    ----------
    labels:  array-like
        label matrix. shape like [n_samples, n_classes] or [n_samples]
    examples: array-like, optional (default=None)
        array of _examples, initialize with this parameter to make
        "strategies by instance" available. Shape like [n_samples, n_features]
    indexes: array-like, optional (default=None)
        index of _examples, if not provided, it will be generated
        automatically started from 0.
    cost: array_like, optional (default=None)
        costs of each queried instance, should have the same length
        and is one-to-one correspondence of y, default is 1.
    """

    def __init__(self, labels, examples=None, indexes=None, cost=None):

        if not check_one_to_one_correspondence(labels, examples, indexes, cost):
            raise ValueError("Different length of parameters found. "
                             "All parameters should be list type with the same length")

        labels = check_array(labels, ensure_2d=False, dtype=None)

        if isinstance(labels[0], np.generic):
            # self._label_type = type(np.asscalar(labels[0])) # deprecated in numpy v1.16
            self._label_type = type(labels[0].item())
        else:
            self._label_type = type(labels[0])
        self._label_dim = labels.ndim

        # check parameters
        self._indexes = indexes if indexes is not None else [i for i in range(len(labels))]
        self._cost_flag = True if cost is not None else False
        self._instance_flag = True if examples is not None else False

        # several _indexes construct
        if self._instance_flag:
            examples = [tuple(vec) for vec in examples]
            self._exa2ind = dict(zip(examples, self._indexes))
        if self._cost_flag:
            self._ind2all = dict(zip(self._indexes, zip(labels, cost)))
        else:
            self._ind2all = dict(zip(self._indexes, labels))

    @property
    def index_keys(self):
        return self._ind2all.keys()

    @property
    def example_keys(self):
        if self._instance_flag:
            return np.asarray(self._exa2ind.keys())
        else:
            return None

    def _add_one_entry(self, label, index, example=None, cost=None):
        """Adding entry to the oracle.
        Add new entry to the oracle for future querying where index is the queried elements,
        label is the returned data. Index should not be in the oracle. Example and cost should
        accord with the initializing (If exists in initializing, then must be provided.)
        The data provided must have the same type with the initializing data. If different, a
        transform is attempted.
        Parameters
        ----------
        label:  array-like
            Label matrix.
        index: object
            Index of examples, should not in the oracle.
        example: array-like, optional (default=None)
            Array of examples, initialize with this parameter to turn
            "strategies by instance" on.
        cost: array_like, optional (default=None)
            Cost of each queried instance, should have the same length
            and is one-to-one correspondence of y, default is 1.
        """
        if isinstance(label, np.generic):
            # label = np.asscalar(label)    # deprecated in numpy v1.16
            label = label.item()
        if isinstance(label, list):
            label = np.array(label)
        if not isinstance(label, self._label_type):
            raise TypeError("Different types of _labels found when adding entries: %s is expected but received: %s" %
                            (str(self._label_type), str(type(label))))
        if self._instance_flag:
            if example is None:
                raise Exception("This oracle has the instance information,"
                                "must provide example parameter when adding entry")
            example = tuple(example)
            self._exa2ind[example] = index
        if self._cost_flag:
            if cost is None:
                raise Exception("This oracle has the cost information,"
                                "must provide cost parameter when adding entry")
            self._ind2all[index] = (label, cost)
        else:
            self._ind2all[index] = label

    def add_knowledge(self, labels, indexes, examples=None, cost=None):
        """Adding entries to the oracle.
        Add new entries to the oracle for future querying where indexes are the queried elements,
        labels are the returned data. Indexes should not be in the oracle. Examples and cost should
        accord with the initializing (If exists in initializing, then must be provided.)
        Parameters
        ----------
        labels: array-like or object
            Label matrix.
        indexes: array-like or object
            Index of examples, should not in the oracle.
            if update multiple entries to the oracle, a list or np.ndarray type is expected,
            otherwise, it will be cheated as only one entry.
        examples: array-like, optional (default=None)
            Array of examples.
        cost: array_like, optional (default=None)
            Cost of each queried instance, should have the same length
            and is one-to-one correspondence of y, default is 1.
        """
        labels, indexes, examples, cost = unpack(labels, indexes, examples, cost)
        if not isinstance(indexes, (list, np.ndarray, BaseCollection)):
            self._add_one_entry(labels, indexes, examples, cost)
        else:
            if not check_one_to_one_correspondence(labels, indexes, examples, cost):
                raise ValueError("Different length of parameters found.")
            for i in range(len(labels)):
                self._add_one_entry(labels[i], indexes[i], example=examples[i] if examples is not None else None,
                                    cost=cost[i] if cost is not None else None)

    def query_by_index(self, indexes):
        """Query function.
        Parameters
        ----------
        indexes: list or int
            Index to strategies, if only one index to strategies (batch_size = 1),
            an int is expected.
        Returns
        -------
        labels: list
            supervised information of queried index.
        cost: list
            corresponding cost produced by strategies.
        """
        if not isinstance(indexes, (list, np.ndarray, BaseCollection)):
            indexes = [indexes]
        sup_info = []
        cost = []
        for k in indexes:
            if k in self._ind2all.keys():
                if self._cost_flag:
                    sup_info.append(self._ind2all[k][0])
                    cost.append(self._ind2all[k][1])
                else:
                    sup_info.append(self._ind2all[k])
                    cost.append(1)
            else:
                self._do_missing(k)
        return sup_info, cost

    def query_by_example(self, queried_examples):
        """Query function, strategies information giving an instance.
        Note that, this function only available if initializes with
        data matrix.
        Parameters
        ----------
        queried_examples: array_like
            [n_samples, n_features]
        Returns
        -------
        sup_info: list
            supervised information of queried instance.
        costs: list
            corresponding costs produced by strategies.
        """
        if not self._instance_flag:
            raise Exception("This oracle do not have the instance information, query_by_instance is not supported")
        if not isinstance(queried_examples, (list, np.ndarray)):
            raise TypeError("An list or numpy.ndarray is expected, but received:%s" % str(type(queried_examples)))
        if len(np.shape(queried_examples)) == 1:
            queried_examples = [queried_examples]
        q_id = []
        for k in queried_examples:
            k = tuple(k)
            if k in self._exa2ind.keys():
                q_id.append(self._exa2ind[k])
            else:
                self._do_missing(k, 'instance pool')
        return self.query_by_index(q_id)

    def _do_missing(self, key, dict_name='index pool'):
        """
        Parameters
        ----------
        key
        Returns
        -------
        """
        raise KeyError("%s is not in the " + dict_name + " of this oracle" % str(key))

    def __repr__(self):
        return str(self._ind2all)

    def save_oracle(self, saving_path):
        """Save the oracle to file.
        Parameters
        ----------
        saving_path: str
            path to save the settings. If a dir is provided, it will generate a file called
            'al_settings.pkl' for saving.
        """
        if saving_path is None:
            return
        else:
            if not isinstance(saving_path, str):
                raise TypeError("A string is expected, but received: %s" % str(type(saving_path)))
        import pickle
        saving_path = os.path.abspath(saving_path)
        if os.path.isdir(saving_path):
            f = open(os.path.join(saving_path, 'oracle.pkl'), 'wb')
        else:
            f = open(os.path.abspath(saving_path), 'wb')
        pickle.dump(self, f)
        f.close()

    @classmethod
    def load_oracle(cls, path):
        """Loading ToolBox object from path.
        Parameters
        ----------
        path: str
            Path to a specific file, not a dir.
        Returns
        -------
        setting: ToolBox
            Object of ToolBox.
        """
        if not isinstance(path, str):
            raise TypeError("A string is expected, but received: %s" % str(type(path)))
        import pickle
        f = open(os.path.abspath(path), 'rb')
        setting_from_file = pickle.load(f)
        f.close()
        return setting_from_file


class SimulatedOracleQueryIndex(Oracle):
    """Oracle to label all _labels of an instance.
    """

    def __init__(self, labels, examples=None, indexes=None, cost=None):
        """

        :param labels:
        :param examples:
        :param indexes:
        :param cost:
        """
        super().__init__(labels, examples, indexes, cost)