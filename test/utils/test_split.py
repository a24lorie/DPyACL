"""
Test the functions in al split modules

"""

from __future__ import division

import unittest
import dask.array as da

from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.multiclass import unique_labels
import numpy as np

from dpyacl.core.misc import split


class TestSplit(unittest.TestCase):
    """
    Test 3 types of split setting:
    1. common setting with int indexes.
    2. multi-label setting with the fully labeled warm start (index is tuple)
    3. split feature matrix to discard some values randomly (similar to multi-label).
    """

    # _IrisX, _IrisY = load_iris(return_X_y=True)
    # _X = da.from_array(_IrisX)
    # _y = da.from_array(_IrisY)

    _split_count = 10
    _feature_num = 1000
    _label_num = 3
    _instance_num = 100000

    _X, _y = make_classification(n_samples=_instance_num, n_features=_feature_num, n_informative=2*_label_num, n_redundant=_label_num,
                                 n_repeated=0, n_classes=_label_num, n_clusters_per_class=_label_num, weights=None,
                                 flip_y=0.01,
                                 class_sep=1.0,
                                 hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)

    _X = da.from_array(_X, chunks="16MB")
    _y = da.from_array(_y, chunks="16MB")

    _mult_y = LabelBinarizer().fit_transform(y=_y)


    def test_split(self):
        train_idx, test_idx, label_idx, unlabel_idx = split(X=self._X,
                                                            y=self._y,
                                                            all_class=False,
                                                            split_count=self._split_count,
                                                            test_ratio=0.3,
                                                            initial_label_rate=0.05)
        assert len(train_idx) == self._split_count
        assert len(test_idx) == self._split_count
        assert len(label_idx) == self._split_count
        assert len(unlabel_idx) == self._split_count

        for i in range(self._split_count):
            train = train_idx[i]
            test = test_idx[i]
            lab = label_idx[i]
            unl = unlabel_idx[i]

            assert len(test) == round(0.3 * self._instance_num)
            assert len(lab) == round(0.05 * len(train))

            # validity
            traintest = da.concatenate([train, test], axis=0)
            labun = da.concatenate([lab, unl], axis=0)

            assert (traintest.topk(-len(traintest)).compute() == list(range(self._instance_num))).all()
            assert (labun == train).all()

    def test_split_allclass(self):
        train_idx, test_idx, label_idx, unlabel_idx = split(X=self._X,
                                                            y=self._y,
                                                            all_class=True,
                                                            split_count=self._split_count,
                                                            test_ratio=0.3,
                                                            initial_label_rate=0.05)
        assert len(train_idx) == self._split_count
        assert len(test_idx) == self._split_count
        assert len(label_idx) == self._split_count
        assert len(unlabel_idx) == self._split_count

        for i in range(self._split_count):
            train = train_idx[i]
            test = test_idx[i]
            lab = label_idx[i]
            unl = unlabel_idx[i]

            assert len(test) == round(0.3 * self._instance_num)
            assert len(lab) == round(0.05 * len(train))

            # validity
            traintest = da.concatenate([train, test], axis=0)
            labun = da.concatenate([lab, unl], axis=0)

            assert (traintest.topk(-len(traintest)).compute() == list(range(self._instance_num))).all()
            assert (labun == train).all()

            # is all-class
            assert (len(unique_labels(self._y[label_idx[i]].compute())) == self._label_num)

    # def test_split_allclass_assert(self):
    #     self.assertRaises(ValueError, split, X=self._X,
    #                       y=self._y,
    #                       all_class=True, split_count=self._split_count,
    #                       test_ratio=0.3, initial_label_rate=0.01)

    # def test_split_multi_label(self):
    #     self.assertRaises(TypeError, split_multi_label,
    #                       y=self._y, label_shape=(self._instance_num, self._label_num),
    #                       all_class=False, split_count=self._split_count,
    #                       test_ratio=0.3, initial_label_rate=0.05)
    #
    #     train_idx, test_idx, label_idx, unlabel_idx = split_multi_label(
    #         y=self._mult_y, label_shape=(self._instance_num, self._label_num),
    #         all_class=False, split_count=self._split_count,
    #         test_ratio=0.3, initial_label_rate=0.05)
    #
    #     assert len(train_idx) == self._split_count
    #     assert len(test_idx) == self._split_count
    #     assert len(label_idx) == self._split_count
    #     assert len(unlabel_idx) == self._split_count
    #     for i in range(self._split_count):
    #         check_index_multilabel(label_idx[i])
    #         check_index_multilabel(unlabel_idx[i])
    #         train = set(train_idx[i])
    #         test = set(test_idx[i])
    #         assert len(test) == round(0.3 * self._instance_num)
    #
    #         len(label_idx[i]) == len(integrate_multilabel_index(label_idx[i], label_size=self._label_num))
    #         # validity
    #         lab = set([j[0] for j in label_idx[i]])
    #         assert len(lab) == round(0.05 * len(train))
    #
    #         unl = set([j[0] for j in unlabel_idx[i]])
    #         traintest = train.union(test)
    #         labun = lab.union(unl)
    #
    #         assert traintest == set(range(self._instance_num))
    #         assert labun == train
    #
    # def test_split_multi_label_allclass(self):
    #     train_idx, test_idx, label_idx, unlabel_idx = split_multi_label(
    #         y=self._mult_y, label_shape=(self._instance_num, self._label_num),
    #         all_class=True, split_count=self._split_count,
    #         test_ratio=0.3, initial_label_rate=0.05)
    #
    #     assert len(train_idx) == self._split_count
    #     assert len(test_idx) == self._split_count
    #     assert len(label_idx) == self._split_count
    #     assert len(unlabel_idx) == self._split_count
    #     for i in range(self._split_count):
    #         check_index_multilabel(label_idx[i])
    #         check_index_multilabel(unlabel_idx[i])
    #         train = set(train_idx[i])
    #         test = set(test_idx[i])
    #
    #         assert len(label_idx[i]) == len(integrate_multilabel_index(label_idx[i], label_size=self._label_num))
    #         # validity
    #         lab = set([j[0] for j in label_idx[i]])
    #         unl = set([j[0] for j in unlabel_idx[i]])
    #         traintest = train.union(test)
    #         labun = lab.union(unl)
    #
    #         assert len(test) == round(0.3 * self._instance_num)
    #         assert len(lab) == round(0.05 * len(train))
    #         assert traintest == set(range(self._instance_num))
    #         assert labun == train
    #
    # def test_split_features(self):
    #     train_idx, test_idx, label_idx, unlabel_idx = split_features(feature_matrix=self._X,
    #                                                                  feature_matrix_shape=self._X.shape,
    #                                                                  test_ratio=0.3, missing_rate=0.2,
    #                                                                  split_count=self._split_count,
    #                                                                  all_features=False)
    #
    #     assert len(train_idx) == self._split_count
    #     assert len(test_idx) == self._split_count
    #     assert len(label_idx) == self._split_count
    #     assert len(unlabel_idx) == self._split_count
    #     for i in range(self._split_count):
    #         train = set(train_idx[i])
    #         test = set(test_idx[i])
    #         traintest = train.union(test)
    #
    #         # validity
    #         assert len(flattern_multilabel_index(index_arr=unlabel_idx[i], label_size=self._feature_num)) == round(
    #             0.2 * len(train) * self._feature_num)
    #         assert len(test) == round(0.3 * self._instance_num)
    #
    #         assert traintest == set(range(self._instance_num))
    #         assert len(
    #             [j[0] for j in
    #              integrate_multilabel_index(label_idx[i] + unlabel_idx[i], label_size=self._feature_num)]) == len(
    #             train_idx[i])
    #
    # def test_split_all_features(self):
    #     train_idx, test_idx, label_idx, unlabel_idx = split_features(feature_matrix=self._X,
    #                                                                  feature_matrix_shape=self._X.shape,
    #                                                                  test_ratio=0.3, missing_rate=0.2,
    #                                                                  split_count=self._split_count,
    #                                                                  all_features=True)
    #     assert len(train_idx) == self._split_count
    #     assert len(test_idx) == self._split_count
    #     assert len(label_idx) == self._split_count
    #     assert len(unlabel_idx) == self._split_count
    #     for i in range(self._split_count):
    #         train = set(train_idx[i])
    #         test = set(test_idx[i])
    #         traintest = train.union(test)
    #
    #         # validity
    #         assert len(flattern_multilabel_index(index_arr=unlabel_idx[i], label_size=self._feature_num)) == round(
    #             0.2 * len(train) * self._X.shape[1])
    #         assert len(test) == round(0.3 * self._instance_num)
    #
    #         assert traintest == set(range(self._instance_num))
    #         assert len(
    #             [j[0] for j in
    #              integrate_multilabel_index(label_idx[i] + unlabel_idx[i], label_size=self._feature_num)]) == len(
    #             train_idx[i])

    def test_split9(self):

        self._instance_num = 100
        self._split_count = 3
        self._X, self._y = make_classification(n_samples=self._instance_num, n_features=4, n_informative=2, n_redundant=2,
                                               n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None,
                                               flip_y=0.01,
                                               class_sep=1.0,
                                               hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)

        train_idx, test_idx, label_idx, unlabel_idx = split(X=self._X,
                                                            y=self._y,
                                                            all_class=False,
                                                            split_count=self._split_count,
                                                            test_ratio=0.3,
                                                            initial_label_rate=0.05)

        print("shape X:  %s" % np.shape(self._X).__str__())
        print("shape y:  %s" % np.shape(self._y).__str__())
        print("shape train_idx: %s" % np.shape(train_idx).__str__())
        print("shape test_idx:  %s" % np.shape(test_idx).__str__())
        print("shape label_idx:  %s" % np.shape(label_idx).__str__())
        print("shape unlabel_idx:  %s" % np.shape(unlabel_idx).__str__())

        assert len(train_idx) == self._split_count
        assert len(test_idx) == self._split_count
        assert len(label_idx) == self._split_count
        assert len(unlabel_idx) == self._split_count

        for i in range(self._split_count):
            train = train_idx[i]
            test = test_idx[i]
            lab = label_idx[i]
            unl = unlabel_idx[i]

            assert len(test) == round(0.3 * self._instance_num)
            assert len(lab) == round(0.05 * len(train))

            # validity
            traintest = da.concatenate([train, test], axis=0)
            labun = da.concatenate([lab, unl], axis=0)

            assert (traintest.topk(-len(traintest)).compute() == list(range(self._instance_num))).all()
            assert (labun == train).all()