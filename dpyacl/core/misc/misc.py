"""
Misc functions to be settled
Added distributed processing capabilities with Dask
"""

import collections

import dask.array as da
import dask_ml.utils as dmlu
import numpy as np
from dask import compute
from sklearn.utils import validation

__all__ = ['check_2d_array',
           'randperm',
           'nlargestarg',
           'nsmallestarg',
           'check_one_to_one_correspondence',
           'unpack',
           'split',
           'split_features',
           'split_multi_label',
           'check_index_multilabel',
           'integrate_multilabel_index',
           'flattern_multilabel_index']


def check_2d_array(matrix):
    """check if the given arrays has 2 dimensions."""
    matrix = check_array(matrix, accept_sparse='csr', ensure_2d=False, order='C')
    if matrix.ndim == 1 and len(matrix) == 1:
        matrix = matrix.reshape(1, -1)

    return matrix


def randperm(n, k=None):
    """Generate a random array which contains k elements range from (n[0]:n[1])
    Parameters
    ----------
    n: int or tuple
        range from [n[0]:n[1]], include n[0] and n[1].
        if an int is given, then n[0] = 0
    k: int, optional (default=end - start + 1)
        how many numbers will be generated. should not larger than n[1]-n[0]+1,
        default=n[1] - n[0] + 1.
    Returns
    -------
    perm: list
        the generated array.
    """
    if isinstance(n, np.generic):
        # n = np.asscalar(n)  # deprecated in numpy v1.16
        n = n.item()
    if isinstance(n, tuple):
        if n[0] is not None:
            start = n[0]
        else:
            start = 0
        end = n[1]
    elif isinstance(n, int):
        start = 0
        end = n
    else:
        raise TypeError("n must be tuple or int.")

    if k is None:
        k = end - start + 1
    if not isinstance(k, int):
        raise TypeError("k must be an int.")
    if k > end - start + 1:
        raise ValueError("k should not larger than n[1]-n[0]+1")

    # randarr = np.arange(start, end + 1)
    # np.random.shuffle(randarr)
    # return randarr[0:k]

    randarr = da.random.choice(n, n, replace=False)
    return randarr[0:k]


def nlargestarg(a, n):
    """Return n largest values' indexes of the given array a.
    Parameters
    ----------
    a: {list, np.ndarray}
        Data array.
    n: int
        The number of returned args.
    Returns
    -------
    nlargestarg: list
        The n largest args in array a.
    """
    assert (validation._is_arraylike(a))
    assert (n > 0)
    if isinstance(a, (list, np.ndarray)):
        argret = da.argtopk(da.from_array(a), n)
    else:
        argret = da.argtopk(a, n)

    # ascent
    return argret[argret.size - n:]


def nsmallestarg(a, n):
    """Return n smallest values' indexes of the given array a.
    Parameters
    ----------
    a: {list, np.ndarray}
        Data array.
    n: int
        The number of returned args.
    Returns
    -------
    nlargestarg: list
        The n smallest args in array a.
    """
    assert (validation._is_arraylike(a))
    assert (n > 0)
    if isinstance(a, (list, np.ndarray)):
        argret = da.argtopk(da.from_array(a), n)
    else:
        argret = da.argtopk(a, n)

    # ascent
    return argret[0:n]


def check_one_to_one_correspondence(*args):
    """Check if the parameters are one-to-one correspondence.
    Parameters
    ----------
    args: object
        The parameters to test.
    Returns
    -------
    result: int
        Whether the parameters are one-to-one correspondence.
        1 : yes
        0 : no
        -1: some parameters have the length 1.
    """

    from ..collections import BaseCollection

    first_not_none = True
    result = True
    for item in args:
        # only check not none object
        if item is not None:
            if first_not_none:
                # record item type
                first_not_none = False
                if_array = isinstance(item, (list, np.ndarray, BaseCollection))
                if if_array:
                    itemlen = len(item)
                else:
                    itemlen = 1
            else:
                if isinstance(item, (list, np.ndarray, BaseCollection)):
                    if len(item) != itemlen:
                        return False
                else:
                    if itemlen != 1:
                        return False
    return True


def unpack(*args):
    """
    Unpack the list with only one element.
    """

    from ..collections import BaseCollection

    ret_args = []
    for arg in args:
        if isinstance(arg, (list, np.ndarray, BaseCollection)):
            if len(arg) == 1:
                ret_args.append(arg[0])
            else:
                ret_args.append(arg)
        else:
            ret_args.append(arg)
    return tuple(ret_args)


# TODO misc.split -> revisar
def split(X=None, y=None,
          instance_indexes=None,
          test_ratio=0.3,
          initial_label_rate=0.05,
          split_count=10,
          all_class=True):
    """Split given data.
    Provide one of X, y or instance_indexes to execute the split.
    Parameters
    ----------
    X: array-like, optional
        Data matrix with [n_samples, n_features]
    y: array-like, optional
        labels of given data [n_samples, n_labels] or [n_samples]
    instance_indexes: list, optional (default=None)
        List contains instances' names, used for image datasets,
        or provide index list instead of data matrix.
        Must provide one of [instance_names, X, y]
    test_ratio: float, optional (default=0.3)
        Ratio of test set
    initial_label_rate: float, optional (default=0.05)
        Ratio of initial label set
        e.g. Initial_labelset*(1-test_ratio)*n_samples
    split_count: int, optional (default=10)
        Random split data _split_count times
    all_class: bool, optional (default=True)
        Whether each split will contain at least one instance for each class.
        If False, a totally random split will be performed.
        Giving None to disable saving.

    Returns
    -------
    train_idx: list
        index of training set, shape like [n_split_count, n_training_indexes]
    test_idx: list
        index of testing set, shape like [n_split_count, n_testing_indexes]
    label_idx: list
        index of labeling set, shape like [n_split_count, n_labeling_indexes]
    unlabel_idx: list
        index of unlabeling set, shape like [n_split_count, n_unlabeling_indexes]
    """

    # check input parameters
    if X is None and y is None and instance_indexes is None:
        raise Exception("Must provide one of X, y or instance_indexes.")

    len_of_parameters = [len(X) if X is not None else None,
                         len(y) if y is not None else None,
                         len(instance_indexes) if instance_indexes is not None else None]
    number_of_instance = np.unique([i for i in len_of_parameters if i is not None])
    if len(number_of_instance) > 1:
        raise ValueError("Different length of instances and _labels found.")
    else:
        number_of_instance = number_of_instance[0]

    if instance_indexes is not None:
        instance_indexes = da.array(instance_indexes)
    else:
        instance_indexes = da.arange(number_of_instance)

    # split
    train_idx = []
    test_idx = []
    label_idx = []
    unlabel_idx = []

    for i in range(split_count):
        if (not all_class) or y is None:
            rp = randperm(number_of_instance)
            cutpoint = int(round((1 - test_ratio) * len(rp)))
            tp_train = instance_indexes[rp[0:cutpoint]]
            train_idx.append(tp_train)
            test_idx.append(instance_indexes[rp[cutpoint:]])
            cutpoint = int(round(initial_label_rate * len(tp_train)))
            if cutpoint <= 1:
                cutpoint = 1
            label_idx.append(tp_train[0:cutpoint])
            unlabel_idx.append(tp_train[cutpoint:])
        else:
            if y is None:
                raise Exception("y must be provided when all_class flag is True.")
            if isinstance(y , da.core.Array):
                check_array(y, ensure_2d=False, dtype=None, distributed=False)
            else:
                y = check_array(y, ensure_2d=False, dtype=None, distributed=True)

            if y.ndim == 1:
                label_num = len(da.unique(y).compute())
            else:
                label_num = y.shape[1]
            if round((1 - test_ratio) * initial_label_rate * number_of_instance) < label_num:
                raise ValueError(
                    "The initial rate is too small to guarantee that each "
                    "split will contain at least one instance for each class.")

            # check validaty
            while 1:
                rp = randperm(number_of_instance)
                cutpoint = int(round((1 - test_ratio) * len(rp)))
                tp_train = instance_indexes[rp[0:cutpoint]]
                cutpointlabel = int(round(initial_label_rate * len(tp_train)))
                if cutpointlabel <= 1:
                    cutpointlabel = 1
                label_id = tp_train[0:cutpointlabel]
                if y.ndim == 1:
                    if len(da.unique(y[label_id]).compute()) == label_num:
                        break
                else:
                    temp = da.sum(y[label_id], axis=0)
                    if not da.any(temp == 0):
                        break

            train_idx.append(tp_train)
            test_idx.append(instance_indexes[rp[cutpoint:]])
            label_idx.append(tp_train[0:cutpointlabel])
            unlabel_idx.append(tp_train[cutpointlabel:])

    return compute(train_idx, test_idx, label_idx, unlabel_idx)


# TODO misc.__split_data_matrix -> revisar
def __split_data_matrix(data_matrix=None,
                        matrix_shape=None,
                        test_ratio=0.3,
                        initial_label_rate=0.05,
                        split_count=10,
                        all_class=True,
                        partially_labeled=False):
    """Split given data matrix with shape like [n_samples, n_labels or n_features]
    Giving one of matrix or matrix_shape to split the data.

    Parameters
    ----------
    data_matrix: array-like, optional
        Labels of given data, shape like [n_samples, n_labels]

    matrix_shape: tuple, optional (default=None)
        The shape of data_matrix, should be a tuple with 2 elements.
        The first one is the number of instances, and the other is the
        number of _labels.

    test_ratio: float, optional (default=0.3)
        Ratio of test set

    initial_label_rate: float, optional (default=0.05)
        Ratio of initial label set
        e.g. Initial_labelset*(1-test_ratio)*n_samples

    split_count: int, optional (default=10)
        Random split data _split_count times

    all_class: bool, optional (default=True)
        Whether each split will contain at least one instance for each class.
        If False, a totally random split will be performed.

    partially_labeled: bool, optional (default=False)
        Whether split the data as partially labeled in the multi-label setting.
        If False, the labeled set is fully labeled, otherwise, only part of _labels of each
        instance will be labeled initialized.
        Only available in multi-label setting.

    Returns
    -------
    train_idx: list
        index of training set, shape like [n_split_count, n_training_indexes]

    test_idx: list
        index of testing set, shape like [n_split_count, n_testing_indexes]

    label_idx: list
        index of labeling set, shape like [n_split_count, n_labeling_indexes]

    unlabel_idx: list
        index of unlabeling set, shape like [n_split_count, n_unlabeling_indexes]

    """

    def check_matrix(matrix):
        """check if the given matrix is legal."""
        matrix = check_array(matrix, accept_sparse='csr', ensure_2d=False, order='C')
        if matrix.ndim != 2:
            if matrix.ndim == 1 and len(matrix) == 1:
                matrix = matrix.reshape(1, -1)
            else:
                raise TypeError("Matrix should be a 2D array with [n_samples, n_features] or [n_samples, n_classes].")
        return matrix

    # check parameters
    if data_matrix is None and matrix_shape is None:
        raise Exception("Must provide one of data matrix or matrix_shape.")
    data_shape = None
    if data_matrix is not None:
        data_matrix = check_matrix(data_matrix)
        data_shape = data_matrix.shape
    if matrix_shape is not None:
        if not isinstance(matrix_shape, tuple) and len(matrix_shape) == 2:
            raise TypeError("the shape of data matrix should be a tuple with 2 elements."
                            "The first one is the number of instances, and the other is the"
                            "number of _labels.")
        data_shape = matrix_shape
    instance_indexes = np.arange(data_shape[0])

    # split
    train_idx = []
    test_idx = []
    label_idx = []
    unlabel_idx = []
    for i in range(split_count):
        if partially_labeled:
            # split train test
            rp = randperm(data_shape[0] - 1)
            cutpoint = int(round((1 - test_ratio) * len(rp)))
            tp_train = instance_indexes[rp[0:cutpoint]]

            # split label & unlabel
            train_size = len(tp_train)
            lab_ind = randperm((0, train_size * data_shape[1] - 1),
                               int(round(initial_label_rate * train_size * data_shape[1])))
            if all_class:
                if round(initial_label_rate * train_size) < data_shape[1]:
                    raise ValueError("The initial rate is too small to guarantee that each "
                                     "split will contain at least one instance for each class.")
                while len(np.unique([item % data_shape[1] for item in lab_ind])) != data_shape[1]:
                    # split train test
                    rp = randperm(data_shape[0] - 1)
                    cutpoint = int(round((1 - test_ratio) * len(rp)))
                    tp_train = instance_indexes[rp[0:cutpoint]]
                    # split label & unlabel
                    train_size = len(tp_train)
                    lab_ind = randperm((0, train_size * data_shape[1] - 1), int(round(initial_label_rate * train_size)))
            train_idx.append(tp_train)
            test_idx.append(instance_indexes[rp[cutpoint:]])
            unlab_ind = set(np.arange(train_size * data_shape[1]))
            unlab_ind.difference_update(set(lab_ind))
            label_idx.append([(tp_train[item // data_shape[1]], item % data_shape[1]) for item in lab_ind])
            unlabel_idx.append([(tp_train[item // data_shape[1]], item % data_shape[1]) for item in unlab_ind])
        else:
            rp = randperm(data_shape[0] - 1)
            cutpoint = int(round((1 - test_ratio) * len(rp)))
            tp_train = instance_indexes[rp[0:cutpoint]]

            cutpoint_lab = int(round(initial_label_rate * len(tp_train)))
            if cutpoint_lab <= 1:
                cutpoint_lab = 1
            if all_class:
                if cutpoint_lab < data_shape[1]:
                    raise ValueError(
                        "The initial rate is too small to guarantee that each "
                        "split will contain at least one instance-feature pair for each class.")
                while 1:
                    label_id = tp_train[0:cutpoint_lab]
                    temp = np.sum(data_matrix[label_id], axis=0)
                    if not np.any(temp == 0):
                        break
                    rp = randperm(data_shape[0] - 1)
                    cutpoint = int(round((1 - test_ratio) * len(rp)))
                    tp_train = instance_indexes[rp[0:cutpoint]]

                    cutpoint_lab = int(round(initial_label_rate * len(tp_train)))
            train_idx.append(tp_train)
            test_idx.append(instance_indexes[rp[cutpoint:]])
            label_idx.append([(i,) for i in tp_train[0:cutpoint_lab]])
            unlabel_idx.append([(i,) for i in tp_train[cutpoint_lab:]])

    return train_idx, test_idx, label_idx, unlabel_idx


# TODO misc.split_features -> revisar
def split_features(feature_matrix=None,
                   feature_matrix_shape=None,
                   test_ratio=0.3,
                   missing_rate=0.5,
                   split_count=10,
                   all_features=True):
    """
    Split given feature matrix in feature querying setting.
    Giving one of feature_matrix or feature_matrix_shape to split the data.
    The matrix will be split randomly in _split_count times, the testing set
    is the set of instances with complete feature vectors. The training set
    has missing feature with the rate of missing_rate.
    Parameters
    ----------
    feature_matrix: array-like, optional
        Feature matrix, shape [n_samples, n_labels].
    feature_matrix_shape: tuple, optional (default=None)
        The shape of data_matrix, should be a tuple with 2 elements.
        The first one is the number of instances, and the other is the
        number of feature.
    test_ratio: float, optional (default=0.3)
        Ratio of test set.
    missing_rate: float, optional (default=0.5)
        Ratio of missing value.
    split_count: int, optional (default=10)
        Random split data split_count times
    all_features: bool, optional (default=True)
        Whether each split will contain at least one instance for each feature.
        If False, a totally random split will be performed.
    Returns
    -------
    train_idx: list
        index of training set, shape like [n_split_count, n_training_indexes]
    test_idx: list
        index of testing set, shape like [n_split_count, n_testing_indexes]
    label_idx: list
        index of labeling set, shape like [n_split_count, n_labeling_indexes]
    unlabel_idx: list
        index of unlabeling set, shape like [n_split_count, n_unlabeling_indexes]
    """
    return __split_data_matrix(data_matrix=feature_matrix, matrix_shape=feature_matrix_shape, test_ratio=test_ratio,
                               initial_label_rate=1 - missing_rate, split_count=split_count,
                               all_class=all_features, partially_labeled=True)


# TODO misc.split_multi_label -> revisar
def split_multi_label(y=None, label_shape=None, test_ratio=0.3, initial_label_rate=0.05,
                      split_count=10, all_class=True):
    """Split given data matrix with shape like [n_samples, n_labels]
    Giving one of matrix or matrix_shape to split the data.
    Parameters
    ----------
    y: array-like, optional
        Labels of given data, shape like [n_samples, n_labels]
    label_shape: tuple, optional (default=None)
        The shape of data_matrix, should be a tuple with 2 elements.
        The first one is the number of instances, and the other is the
        number of _labels.
    test_ratio: float, optional (default=0.3)
        Ratio of test set
    initial_label_rate: float, optional (default=0.05)
        Ratio of initial label set
        e.g. Initial_labelset*(1-test_ratio)*n_samples
    split_count: int, optional (default=10)
        Random split data _split_count times
    all_class: bool, optional (default=True)
        Whether each split will contain at least one instance for each class.
        If False, a totally random split will be performed.
    saving_path: str, optional (default='.')
        Giving None to disable saving.
    Returns
    -------
    train_idx: list
        index of training set, shape like [n_split_count, n_training_indexes]
    test_idx: list
        index of testing set, shape like [n_split_count, n_testing_indexes]
    label_idx: list
        index of labeling set, shape like [n_split_count, n_labeling_indexes]
    unlabel_idx: list
        index of unlabeling set, shape like [n_split_count, n_unlabeling_indexes]
    """
    return __split_data_matrix(data_matrix=y, matrix_shape=label_shape, test_ratio=test_ratio,
                               initial_label_rate=initial_label_rate,
                               split_count=split_count, all_class=all_class, partially_labeled=False)


def check_index_multilabel(index):
    """check if the given indexes are legal.

    Parameters
    ----------
    index: list or np.ndarray
        index of the data.
    """

    from ..collections import BaseCollection

    if isinstance(index, BaseCollection):
        return index
    if not isinstance(index, (list, np.ndarray)):
        index = [index]
    datatype = collections.Counter([type(i) for i in index])
    if len(datatype) != 1:
        raise TypeError("Different types found in the given indexes.")
    if not datatype.popitem()[0] == tuple:
        raise TypeError("Each index should be a tuple.")
    return index


def __infer_label_size_multilabel(index_arr, check_arr=True):
    """Infer the label size from a set of index arr.

    raise if all index are example index only.

    Parameters
    ----------
    index_arr: list or np.ndarray
        index array.

    Returns
    -------
    label_size: int
    the inferred label size.
    """
    if check_arr:
        index_arr = check_index_multilabel(index_arr)
    data_len = np.array([len(i) for i in index_arr])
    if np.any(data_len == 2):
        label_size = np.max([i[1] for i in index_arr if len(i) == 2]) + 1
    elif np.all(data_len == 1):
        raise Exception(
            "Label_size can not be induced from fully labeled set, label_size must be provided.")
    else:
        raise ValueError(
            "All elements in indexes should be a tuple, with length = 1 (example_index, ) "
            "to strategies all labels or length = 2 (example_index, [label_indexes]) to strategies specific labels.")
    return label_size


def integrate_multilabel_index(index_arr, label_size=None, check_arr=True):
    """ Integrated the indexes of multi-label.

    Parameters
    ----------
    index_arr: list or np.ndarray
        multi-label index array.

    label_size: int, optional (default = None)
        the size of label set. If not provided, an inference is attempted.
        raise if the inference is failed.

    check_arr: bool, optional (default = True)
        whether to check the validity of index array.

    Returns
    -------
    array: list
        the integrated array.
    """
    if check_arr:
        index_arr = check_index_multilabel(index_arr)
    if label_size is None:
        label_size = __infer_label_size_multilabel(index_arr)
    else:
        assert (label_size > 0)

    integrated_arr = []
    integrated_dict = {}
    for index in index_arr:
        example_ind = index[0]
        if len(index) == 1:
            integrated_dict[example_ind] = set(range(label_size))
        else:
            # length = 2
            if example_ind in integrated_dict.keys():
                integrated_dict[example_ind].update(
                    set(index[1] if isinstance(index[1], collections.Iterable) else [index[1]]))
            else:
                integrated_dict[example_ind] = set(
                    index[1] if isinstance(index[1], collections.Iterable) else [index[1]])

    for item in integrated_dict.items():
        if len(item[1]) == label_size:
            integrated_arr.append((item[0],))
        else:
            # -------------------------------------------------------------------------------------------
            # integrated_arr.append((item[0], tuple(item[0])))
            integrated_arr.append((item[0], tuple(item[1])))

    return integrated_arr


def flattern_multilabel_index(index_arr, label_size=None, check_arr=True):
    """
    Falt the multilabel_index to one-dimensional.

    Parameters
    ----------
    index_arr: list or np.ndarray
        index array.

    label_size: int
        the inferred label size.

    check_arr: bool
        if True,check the index_arr is a legal multilabel_index.

    Returns
    -------
    decomposed_data: list
        the decomposed data after falting.
    """
    if check_arr:
        index_arr = check_index_multilabel(index_arr)
    if label_size is None:
        label_size = __infer_label_size_multilabel(index_arr)
    else:
        assert (label_size > 0)
    decomposed_data = []
    for item in index_arr:
        if len(item) == 1:
            for i in range(label_size):
                decomposed_data.append((item[0], i))
        else:
            if isinstance(item[1], collections.Iterable):
                label_ind = [i for i in item[1] if 0 <= i < label_size]
            else:
                assert (0 <= item[1] < label_size)
                label_ind = [item[1]]
            for j in range(len(label_ind)):
                decomposed_data.append((item[0], label_ind[j]))
    return decomposed_data


def check_X_y(X, y, accept_sparse=False, *, accept_large_sparse=True,
              dtype="numeric", order=None, copy=False, force_all_finite=True,
              ensure_2d=True, allow_nd=False, multi_output=False,
              ensure_min_samples=1, ensure_min_features=1, y_numeric=False,
              estimator=None, distributed=False, chunks="16MB"):

    _X, _y = validation.check_X_y(
        X=X, y=y, accept_sparse=accept_sparse, accept_large_sparse=accept_large_sparse,
        dtype=dtype, order=order, copy=copy, force_all_finite=force_all_finite,
        ensure_2d=ensure_2d, allow_nd=allow_nd, multi_output=multi_output,
        ensure_min_samples=ensure_min_samples, ensure_min_features=ensure_min_features,
        y_numeric=y_numeric, estimator=estimator)

    if distributed:
        return da.from_array(_X, chunks=chunks), da.from_array(_y, chunks=chunks)
    else:
        return _X, _y


def check_array(array, accept_sparse=False, *, accept_large_sparse=True,
                dtype="numeric", order=None, copy=False, force_all_finite=True,
                ensure_2d=True, allow_nd=False, ensure_min_samples=1,
                ensure_min_features=1, estimator=None, distributed=False, chunks="16MB"):

    _X = dmlu.check_array(
        array, accept_sparse=accept_sparse, accept_large_sparse=accept_large_sparse,
        dtype=dtype, order=order, copy=copy, force_all_finite=force_all_finite,
        ensure_2d=ensure_2d, allow_nd=allow_nd, ensure_min_samples=ensure_min_samples,
        ensure_min_features=ensure_min_features, estimator=estimator
    )

    if distributed:
        return da.from_array(_X, chunks=chunks)
    else:
        return _X
