"""
Classes for storing all information regarding the AL experiment

The information includes:
1. The performance after each query. (Added support for several performance metrics in a single query)
2. The selected index for each query

"""
# Authors: Ying-Peng Tang, Alfredo Lorie

import collections.abc
import copy
import functools
import os
import pickle
import sys
import warnings

import numpy as np
import prettytable as pt

from dpyacl.core.collections import BaseCollection, IndexCollection

__all__ = ['StateItem', 'State', 'StateContainer']


class StateItem:
    """
    A class to store information in one iteration of active learning for auditing and analysing.
    The information includes:

    1. The performance after each strategies
    2. The selected index for each strategies
    3. Additional user-defined entry

    Parameters
    ----------
    :param select_index: array-like or object
        If multiple select_index are provided, it should be a list, np.ndarray or BaseCollection type.
        otherwise, it will be treated as only one pair for adding.
    :param performance_metrics: array-like
        List of the me performance metrics
    :param performance: array-like
        List of dictionary elements with the values corresponding to performance_metrics list
    :param queried_label: array-like or object, optional
        The queried label.
    :param cost: array-like or object, optional
        Cost corresponds to the strategies.
    """

    def __init__(self, select_index, performance_metrics, performance, queried_label=None, cost=None):

        if not isinstance(select_index, (list, np.ndarray, BaseCollection)):
            select_index = [select_index]

        self._save_seq = {}
        self._save_seq['select_index'] = copy.deepcopy(select_index)
        self._save_seq['performance_metrics'] = performance_metrics

        param_pref_metrics = [metric['name'] for metric in performance]
        if functools.reduce(lambda i, j: i and j, map(lambda m, k: m == k, param_pref_metrics, performance_metrics), True):
            self._save_seq['performance'] = copy.copy(performance)
        else:
            raise ValueError("performance names should be the same as performance_metrics")

        if queried_label is not None:
            self._save_seq['queried_label'] = copy.deepcopy(queried_label)
        if cost is not None:
            self._save_seq['cost'] = copy.copy(cost)
        self.batch_size = len(select_index)

    def __getitem__(self, item):
        """Get a specific value given key."""
        return self.get_value(key=item)

    def __setitem__(self, key, value):
        return self.add_element(key=key, value=value)

    def keys(self):
        """Return the stored keys."""
        return self._save_seq.keys()

    def add_element(self, key, value):
        """add_element
        Parameters
        ----------
        key: object
            Key to be added, should not in the object.
        value: object
            The value corresponds to the key.
        """
        self._save_seq[key] = copy.deepcopy(value)

    def del_element(self, key):
        """Deleting an element in the object.
        Parameters
        ----------
        key: object
            Key for deleting. Should not be one of the critical information:
            ['select_index', 'queried_info', 'performance', 'cost']
        """
        if key in ['select_index', 'queried_info', 'performance', 'cost']:
            warnings.warn("Critical information %s can not be discarded." % str(key))
        elif key not in self._save_seq.keys():
            warnings.warn("Key %s to be discarded is not in the object, skip." % str(key))
        else:
            self._save_seq.pop(key)

    def get_value(self, key):
        """Get a specific value given key."""
        return self._save_seq[key]

    def set_value(self, key, value):
        """Modify the value of an existed item.
        Parameters
        ----------
        key: object
            Key in the StateItem, must a existed key
        value: object,
            Value to cover the original value
        """
        if key not in self._save_seq.keys():
            raise KeyError('key must be an existed one in StateItem')
        self._save_seq[key] = copy.deepcopy(value)

    def __repr__(self):
        return self._save_seq.__repr__()


class State:
    """
    A class to store states.

    Functions including:
    1. Gathering and checking the information stored in StateItem object.
    2. Print active learning progress: current_iteration, current_mean_performance, current_cost, etc.

    Parameters
    ----------
    :param round: int
        Number of k-fold experiments loop. 0 <= round < k
    :param train_idx: array_like
        Training index of one fold experiment.
    :param test_idx: array_like
        Testing index of one fold experiment.
    :param init_L: array_like
        Initial labeled index of one fold experiment.
    :param init_U: array_like
        Initial unlabeled index of one fold experiment.
    :param performance_metrics: array-like
        List of the me performance metrics
    :param initial_point: object, optional (default=None)
        The performance before any querying.
        If not specify, the initial point of different methods will be different.
    :param check_flag: bool, optional (default=True)
        Whether to check the validity of states.
    :param verbose: bool, optional (default=True)
        Whether to print strategies information during the AL process.
    :param print_interval: int optional (default=1)
        How many queries will trigger a print when verbose is True.
    """
    def __init__(self, round,
                 train_idx, test_idx, init_L, init_U,
                 performance_metrics=[],
                 initial_point=None,
                 check_flag=True,
                 verbose=True,
                 print_interval=1):

        self._performance_metrics = performance_metrics

        assert (isinstance(check_flag, bool))
        assert (isinstance(verbose, bool))
        self.__check_flag = check_flag
        self.__verbose = verbose
        self.__print_interval = print_interval

        if self.__check_flag:
            # check validity
            assert (isinstance(train_idx, collections.abc.Iterable))
            assert (isinstance(test_idx, collections.abc.Iterable))
            assert (isinstance(init_U, collections.abc.Iterable))
            assert (isinstance(init_L, collections.abc.Iterable))
            assert (isinstance(round, int) and round >= 0)

        self.round = round
        self.train_idx = copy.copy(train_idx)
        self.test_idx = copy.copy(test_idx)

        if isinstance(init_U, BaseCollection) and isinstance(init_L, BaseCollection):
            self.init_U = copy.deepcopy(init_U)
            self.init_L = copy.deepcopy(init_L)
        else:
            # try:
            self.init_U = copy.deepcopy(IndexCollection(init_U))
            self.init_L = copy.deepcopy(IndexCollection(init_L))

            #     check_index_multilabel(init_L)
            #     check_index_multilabel(init_U)
            #     self.init_U = copy.deepcopy(MultiLabelIndexCollection(init_U))
            #     self.init_L = copy.deepcopy(MultiLabelIndexCollection(init_L))
            # except TypeError:
            #     self.init_U = copy.deepcopy(IndexCollection(init_U))
            #     self.init_L = copy.deepcopy(IndexCollection(init_L))

        self.init_U = copy.deepcopy(IndexCollection(init_U) if not isinstance(init_U, BaseCollection) else init_U)
        self.init_L = copy.deepcopy(IndexCollection(init_L) if not isinstance(init_L, BaseCollection) else init_L)
        self.initial_point = initial_point
        self.batch_size = 0
        self.__state_list = []
        self._first_print = True
        self.cost_inall = 0
        self._numqdata = 0

        self._ml_technique = None

    def set_initial_point(self, perf):
        """The initial point of performance before querying.
        Parameters
        ----------
        perf: float
            The performance value.
        """
        self.initial_point = perf

    def add_state(self, state):
        """Add a StateItem object to the container.
        Parameters
        ----------
        state: {dict, StateItem}
            StateItem object to be added. Or a dictionary with
            the following keys: ['select_index', 'queried_info', 'performance']
        """
        if not isinstance(state, StateItem):
            assert isinstance(state, dict), "state must be dict or StateItem object."
            assert 'select_index' in state and 'queried_info' in state and 'performance' in state, "The dict must contain the following keys: ['select_index', 'queried_info', 'performance']"
        self.__state_list.append(copy.deepcopy(state))
        self.__update_info()

        if self.__verbose and len(self) % self.__print_interval == 0:
            if self._first_print:
                print('\n' + self.__repr__(), end='')
                self._first_print = False
            else:
                print('\r' + self._refresh_dataline(), end='')
                sys.stdout.flush()

    def get_state(self, index):
        """Get a State object in the container.
        Parameters
        ----------
        index: int
            The index of the State object. 0 <= index < len(self)
        Returns
        -------
        st: StateItem
            The State object in the previous iteration.
        """
        assert (0 <= index < len(self))
        return copy.deepcopy(self.__state_list[index])

    def check_batch_size(self):
        """Check if all queries have the same batch size.
        Returns
        -------
        result: bool
            Whether all the states have the same batch size.
        """
        ind_uni = np.unique(
            [self.__state_list[i].batch_size for i in range(len(self.__state_list) - 1)], axis=0)
        if len(ind_uni) == 1:
            self.batch_size = ind_uni[0]
            return True
        else:
            return False

    def pop(self, i=None):
        """remove and return item at index (default last)."""
        return self.__state_list.pop(i)

    def recover_workspace(self, iteration=None):
        """Recover workspace after $iteration$ querying.
        For example, if 0 is given, the initial workspace without any querying will be recovered.
        Note that, the object itself will be recovered, the information after the iteration will be discarded.
        Parameters
        ----------
        iteration: int, optional(default=None)
            Number of iteration to recover, start from 0.
            If nothing given, it will return the current workspace.
        Returns
        -------
        train_idx: list
            Index of training set, shape like [n_training_samples]
        test_idx: list
            Index of testing set, shape like [n_testing_samples]
        label_idx: list
            Index of labeling set, shape like [n_labeling_samples]
        unlabel_idx: list
            Index of unlabeling set, shape like [n_unlabeling_samples]
        """
        if iteration is None:
            iteration = len(self.__state_list)
        assert (0 <= iteration <= len(self))
        work_U = copy.deepcopy(self.init_U)
        work_L = copy.deepcopy(self.init_L)
        for i in range(iteration):
            state = self.__state_list[i]
            work_U.difference_update(state.get_value('select_index'))
            work_L.update(state.get_value('select_index'))
        self.__state_list = self.__state_list[0:iteration]
        return copy.copy(self.train_idx), copy.copy(self.test_idx), copy.deepcopy(work_L), copy.deepcopy(work_U)

    def get_workspace(self, iteration=None):
        """Get workspace after $iteration$ querying.
        For example, if 0 is given, the initial workspace without any querying will be recovered.
        Parameters
        ----------
        iteration: int, optional(default=None)
            Number of iteration, start from 0.
            If nothing given, it will get the current workspace.
        Returns
        -------
        train_idx: list
            Index of training set, shape like [n_training_samples]
        test_idx: list
            Index of testing set, shape like [n_testing_samples]
        label_idx: list
            Index of labeling set, shape like [n_labeling_samples]
        unlabel_idx: list
            Index of unlabeling set, shape like [n_unlabeling_samples]
        """
        if iteration is None:
            iteration = len(self.__state_list)
        assert (0 <= iteration <= len(self))
        work_U = copy.deepcopy(self.init_U)
        work_L = copy.deepcopy(self.init_L)
        for i in range(iteration):
            state = self.__state_list[i]
            work_U.difference_update(state.get_value('select_index'))
            work_L.update(state.get_value('select_index'))
        return copy.copy(self.train_idx), copy.copy(self.test_idx), copy.deepcopy(work_L), copy.deepcopy(work_U)

    def num_of_query(self):
        """Return the number of queries"""
        return len(self.__state_list)

    def get_current_performance(self, metric_index):
        """
        Return the mean ± std performance of all existed states.
        Only available when the performance of each state is a single_label float value.
        Returns
        -------
        mean: float
            Mean performance of the existing states.
        std: float
            Std performance of the existing states.
        """
        if len(self) == 0:
            return 0, 0
        else:
            tmp = [self[i].get_value('performance')[metric_index]["value"]for i in range(self.__len__())]
            if isinstance(tmp[0], collections.abc.Iterable):
                return np.NaN, np.nan
            else:
                return np.mean(tmp), np.std(tmp)

    def __len__(self):
        return len(self.__state_list)

    def __getitem__(self, item):
        return self.__state_list.__getitem__(item)

    def __contains__(self, other):
        return other in self.__state_list

    def __iter__(self):
        return iter(self.__state_list)

    def refresh_info(self):
        """re-calculate current active learning progress."""
        numqdata = 0
        cost = 0.0
        for state in self.__state_list:
            numqdata += len(state.get_value('select_index'))
            if 'cost' in state.keys():
                cost += np.sum(state.get_value('cost'))
        self.cost_inall = cost
        self._numqdata = numqdata
        return numqdata, cost

    def __update_info(self):
        """Update current active learning progress"""
        state = self.__state_list[len(self) - 1]
        if 'cost' in state.keys():
            self.cost_inall += np.sum(state.get_value('cost'))
        self._numqdata += len(state.get_value('select_index'))

    def __repr__(self):
        numqdata = self._numqdata
        cost = self.cost_inall
        tb = pt.PrettyTable()
        tb.set_style(pt.MSWORD_FRIENDLY)
        tb.add_column('round', [self.round])
        tb.add_column('initially labeled data',
                      [" %d (%.2f%% of all)" % (len(self.init_L), 100 * len(self.init_L) / (len(self.init_L) + len(self.init_U)))])
        tb.add_column('number of queries', [len(self.__state_list)])
        # tb.add_column('queried data', ["%d (%.2f%% of unlabeled data)" % (numqdata, self.queried_percentage)])
        tb.add_column('cost', [cost])
        for metric_index in range(len(self._performance_metrics)):
            tb.add_column('%s:' % self._performance_metrics[metric_index],
                          ["%.3f ± %.2f" % self.get_current_performance(metric_index=metric_index)])
        return str(tb)

    def _refresh_dataline(self):
        tb = self.__repr__()
        return tb.splitlines()[1]

    def set_ml_technique(self, ml_technique):
        if self._ml_technique is None:
            self._ml_technique = ml_technique
        else:
            raise ValueError("The attribute _ml_technique has already been set")

    @property
    def ml_technique(self):
        return copy.deepcopy(self._ml_technique)

    @property
    def performance_metrics(self):
        return copy.deepcopy(self._performance_metrics)

class StateContainer:
    """Class to process State object.
    If a list of State objects is given, the data stored
    in each State object can be extracted with this class.
    """

    def __init__(self, method_name, method_results):
        self.method_name = method_name
        self.__results = list()
        self.add_folds(method_results)

    def add_fold(self, src):
        """
        Add one fold of active learning experiment.
        Parameters
        ----------
        src: object or str
            State object or path to the intermediate results file.
        """
        if isinstance(src, State):
            self.__add_fold_by_object(src)
        elif isinstance(src, str):
            self.__add_fold_from_file(src)
        else:
            raise TypeError('State object or str is expected, but received:%s' % str(type(src)))

    def add_folds(self, folds):
        """Add multiple folds.
        Parameters
        ----------
        folds: list
            The list contains n State objects.
        """
        for item in folds:
            self.add_fold(item)

    def __add_fold_by_object(self, result):
        """
        Add one fold of active learning experiment
        Parameters
        ----------
        result: utils.State
            object stored a complete fold of active learning experiment
        """
        self.__results.append(copy.deepcopy(result))

    def __add_fold_from_file(self, path):
        """
        Add one fold of active learning experiment from file
        Parameters
        ----------
        path: str
            path of result file.
        """
        f = open(os.path.abspath(path), 'rb')
        result = pickle.load(f)
        f.close()
        assert (isinstance(result, State))
        if not result.check_batch_size():
            warnings.warn('Checking validity fails, different batch size is found.')
        self.__results.append(copy.deepcopy(result))

    def extract_matrix(self, extract_keys='performance'):
        """Extract the data stored in the State obejct.
        Parameters
        ----------
        extract_keys: str or list of str, optional (default='performance')
            Extract what value in the StateItem object.
            The extract_keys should be a subset of the keys of each StateItem object.
            Such as: 'select_index', 'performance', 'queried_label', 'cost', etc.
            Note that, the extracted matrix is associated with the extract_keys.
            If more than 1 key is given, each element in the matrix is a tuple.
            The values in tuple are one-to-one correspondence to the extract_keys.
        Returns
        -------
        extracted_matrix: list
            The extracted matrix.
        """
        extracted_matrix = []
        if isinstance(extract_keys, str):
            for stateio in self:
                stateio_line = []
                if stateio.initial_point is not None:
                    stateio_line.append(stateio.initial_point)
                for state in stateio:
                    if extract_keys not in state.keys():
                        raise ValueError('The extract_keys should be a subset of the keys of each StateItem object.\n'
                                         'But keys in the state are: %s' % str(state.keys()))
                    stateio_line.append(state.get_value(extract_keys))
                extracted_matrix.append(copy.copy(stateio_line))

        elif isinstance(extract_keys, list):
            for stateio in self:
                stateio_line = []
                for state in stateio:
                    state_line = []
                    for key in extract_keys:
                        if key not in state.keys():
                            raise ValueError('The extract_keys should be a subset of the keys of each StateItem object.\n'
                                             'But keys in the state are: %s' % str(state.keys()))
                        state_line.append(state.get_value(key))
                    stateio_line.append(tuple(state_line))
                extracted_matrix.append(copy.copy(stateio_line))

        else:
            raise TypeError("str or list of str is expected, but received: %s" % str(type(extract_keys)))

        return extracted_matrix

    def to_list(self):
        """
            Return all StateIOs as a list.
        """
        return copy.deepcopy(self.__results)

    def __len__(self):
        return len(self.__results)

    def __getitem__(self, item):
        return self.__results.__getitem__(item)

    def __iter__(self):
        return iter(self.__results)
