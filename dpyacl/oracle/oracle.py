"""
Pre-defined simOracle class

Implemented Simulated Oracle
Implemented Console Human Oracle
"""
# Authors: Alfredo Lorie extended from Ying-Peng Tang version

from abc import ABCMeta, abstractmethod

import dask
import dask.array as da
import numpy as np
from sklearn.utils.validation import check_array

from dpyacl.core.collections import BaseCollection
from dpyacl.core.misc.misc import check_one_to_one_correspondence, unpack

__all__ = ['Oracle',
           'SimulatedOracle',
           'ConsoleHumanOracle']


class Oracle(metaclass=ABCMeta):
    """
    Oracle in active learning whose role is to label the given strategies.
    This class implements basic definition of simOracle used in experiment.
    Oracle can provide information given an index. The returned information will
    depend on the specific scenario.
    """

    @abstractmethod
    def query(self, **kwargs):
        pass


class SimulatedOracle(Oracle, metaclass=ABCMeta):
    """
    Class that represents an simOracle in a simulated way. The class of the selected
    instances are known previously to the AL process. The labels of the instances
    are hidden and the simOracle only reveals the label

    Parameters
    ----------
    labels:  array-like
        label matrix. shape like [n_samples, n_classes] or [n_samples]
    indexes: array-like, optional (default=None)
        index of _examples, if not provided, it will be generated
        automatically started from 0.
    cost: array_like, optional (default=None)
        costs of each queried instance, should have the same length
        and is one-to-one correspondence of y, default is 1.
    """

    def __init__(self, labels, indexes=None, cost=None):
        """
        :param labels:
        :param indexes:
        :param cost:
        """
        if not check_one_to_one_correspondence(labels, indexes, cost):
            raise ValueError("Different length of parameters found. "
                             "All parameters should be list type with the same length")

        labels = check_array(labels, ensure_2d=False, dtype=None)

        if isinstance(labels[0], np.generic):
            self._label_type = type(labels[0].item())
        else:
            self._label_type = type(labels[0])

        self._label_dim = labels.ndim
        self._label_unique = da.unique(labels)

        # check parameters
        self._cost_flag = True if cost is not None else False

        # several _indexes construct
        if self._cost_flag:
            self._ind2all = dict(zip(indexes if indexes is not None else [i for i in range(len(labels))], zip(labels, cost)))
        else:
            self._ind2all = dict(zip(indexes if indexes is not None else [i for i in range(len(labels))], labels))

    @property
    def index_keys(self):
        return self._ind2all.keys()

    def query(self, instances, indexes):
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

    def _add_one_entry(self, label, index, cost=None):
        """Adding entry to the simOracle.
        Add new entry to the simOracle for future querying where index is the queried elements,
        label is the returned data. Index should not be in the simOracle. Example and cost should
        accord with the initializing (If exists in initializing, then must be provided.)
        The data provided must have the same type with the initializing data. If different, a
        transform is attempted.
        Parameters
        ----------
        label:  array-like
            Label matrix.
        index: object
            Index of examples, should not in the simOracle.
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
        if self._cost_flag:
            if cost is None:
                raise Exception("This simOracle has the cost information,"
                                "must provide cost parameter when adding entry")
            self._ind2all[index] = (label, cost)
        else:
            self._ind2all[index] = label

    def add_knowledge(self, labels, indexes, cost=None):
        """Adding entries to the simOracle.
        Add new entries to the simOracle for future querying where indexes are the queried elements,
        labels are the returned data. Indexes should not be in the simOracle. Cost should
        accord with the initializing (If exists in initializing, then must be provided.)
        Parameters
        ----------
        labels: array-like or object
            Label matrix.
        indexes: array-like or object
            Index of examples, should not in the simOracle.
            if update multiple entries to the simOracle, a list or np.ndarray type is expected,
            otherwise, it will be cheated as only one entry.
        cost: array_like, optional (default=None)
            Cost of each queried instance, should have the same length
            and is one-to-one correspondence of y, default is 1.
        """
        labels, indexes, cost = unpack(labels, indexes, cost)
        if not isinstance(indexes, (list, np.ndarray, BaseCollection)):
            self._add_one_entry(labels, indexes, cost)
        else:
            if not check_one_to_one_correspondence(labels, indexes, cost):
                raise ValueError("Different length of parameters found.")
            for i in range(len(labels)):
                self._add_one_entry(labels[i], indexes[i], cost=cost[i] if cost is not None else None)

    def _do_missing(self, key, dict_name='index pool'):
        """
        Parameters
        ----------
        key
        Returns
        -------
        """
        raise KeyError("%s is not in the " + dict_name + " of this simOracle" % str(key))

    def __repr__(self):
        return str(self._ind2all)


class ConsoleHumanOracle(SimulatedOracle, metaclass=ABCMeta):
    """Oracle to label all _labels of an instance.
    """

    def __init__(self, labels, indexes=None, cost=None):
        """

        :param labels:
        :param indexes:
        :param cost:
        """
        super().__init__(labels, indexes, cost)


    def query(self, instances, indexes):
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

        def options(processed, total):
            print("Instance %s of %s" % (processed+1, total))
            print("  Show Classes (C)")
            print("  Assign Label (L)")
            return input().upper()

        def labelInstance(instance, index, label_type):
            if type(instance) is dask.array.core.Array:
                print("  >> What is the class of this instance: %s " % (instance.compute()))
            else:
                print("  >> What is the class of this instance: %s " % (instance))

            instClass = input()
            self.add_knowledge([label_type(instClass)], [index])
            return label_type(instClass)

        numInstances = len(instances)
        print("Welcome to the Human Oracle Interface")

        sup_info = []
        for i in range(numInstances):
            option = options(i, numInstances)

            while option != 'L':
                if option == 'C':
                    print(self._label_unique.compute())
                else:
                    print("invalid option skipping")
                option = options(i, numInstances)

            sup_info.append(labelInstance(instances[i], indexes[i], self._label_type))

        return sup_info, indexes

