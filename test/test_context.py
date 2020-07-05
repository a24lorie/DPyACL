import unittest

from sklearn.datasets import load_iris, make_classification

from dpyacl.core.stop_criteria import UnlabelSetEmpty
from dpyacl.experiment.context import HoldOutExperiment, CrossValidationExperiment
from dpyacl.core.misc.misc import split


class TestEvaluation(unittest.TestCase):
    # Get the data
    __X, __y = load_iris(return_X_y=True)

    def test_hold_out_self_partitioning(self):

        split_count = 1
        instance_num = 100

        self.__X, self.__y = make_classification(n_samples=instance_num, n_features=4, n_informative=2, n_redundant=2,
                                                 n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None,
                                                 flip_y=0.01,
                                                 class_sep=1.0,
                                                 hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)

        # init the ALExperiment
        experiment = HoldOutExperiment(
            self.__X,
            self.__y,
            self_partition=True,
            stopping_criteria=UnlabelSetEmpty(),
            test_ratio=0.3,
            initial_label_rate=0.05,
            all_class=True
        )

        assert len(experiment._train_idx) == split_count
        assert len(experiment._test_idx) == split_count
        assert len(experiment._label_idx) == split_count
        assert len(experiment._unlabel_idx) == split_count

        for i in range(split_count):
            train = set(experiment._train_idx[i])
            test = set(experiment._test_idx[i])
            lab = set(experiment._label_idx[i])
            unl = set(experiment._unlabel_idx[i])

            assert len(test) == round(0.3 * instance_num)
            assert len(lab) == round(0.05 * len(train))

            # validity
            traintest = train.union(test)
            labun = lab.union(unl)
            assert traintest == set(range(instance_num))
            assert labun == train

    def test_cross_validation_self_partitioning(self):

        split_count = 5
        instance_num = 100

        self.__X, self.__y = make_classification(n_samples=instance_num, n_features=4, n_informative=2, n_redundant=2,
                                                 n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None,
                                                 flip_y=0.01,
                                                 class_sep=1.0,
                                                 hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)

        # init the AlExperiment
        experiment = CrossValidationExperiment(
            self.__X,
            self.__y,
            self_partition=True,
            kfolds=split_count,
            stopping_criteria=UnlabelSetEmpty(),
            test_ratio=0.3,
            initial_label_rate=0.05,
            all_class=True
        )

        assert len(experiment._train_idx) == split_count
        assert len(experiment._test_idx) == split_count
        assert len(experiment._label_idx) == split_count
        assert len(experiment._unlabel_idx) == split_count

        for i in range(split_count):
            train = set(experiment._train_idx[i])
            test = set(experiment._test_idx[i])
            lab = set(experiment._label_idx[i])
            unl = set(experiment._unlabel_idx[i])

            assert len(test) == round(0.3 * instance_num)
            assert len(lab) == round(0.05 * len(train))

            # validity
            traintest = train.union(test)
            labun = lab.union(unl)
            assert traintest == set(range(instance_num))
            assert labun == train

    def test_cross_validation_without_self_partitioning_ok(self):

        split_count = 5
        instance_num = 100

        self.__X, self.__y = make_classification(n_samples=instance_num, n_features=4, n_informative=2, n_redundant=2,
                                                 n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None,
                                                 flip_y=0.01,
                                                 class_sep=1.0,
                                                 hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)

        train_idx, test_idx, label_idx, unlabel_idx = split(
            X=self.__X,
            y=self.__y,
            test_ratio=0.3,
            initial_label_rate=0.05,
            split_count=split_count,
            all_class=True)

        # init the AlExperiment
        experiment = CrossValidationExperiment(
            self.__X,
            self.__y,
            self_partition=False,
            stopping_criteria=UnlabelSetEmpty(),
            train_idx=train_idx,
            test_idx=test_idx,
            label_idx=label_idx,
            unlabel_idx=unlabel_idx
        )

        assert len(experiment._train_idx) == split_count
        assert len(experiment._test_idx) == split_count
        assert len(experiment._label_idx) == split_count
        assert len(experiment._unlabel_idx) == split_count

        for i in range(split_count):
            train = set(experiment._train_idx[i])
            test = set(experiment._test_idx[i])
            lab = set(experiment._label_idx[i])
            unl = set(experiment._unlabel_idx[i])

            assert len(test) == round(0.3 * instance_num)
            assert len(lab) == round(0.05 * len(train))

            # validity
            traintest = train.union(test)
            labun = lab.union(unl)
            assert traintest == set(range(instance_num))
            assert labun == train

    def test_cross_validation_without_self_partitioning_wrong_index_dim(self):

        split_count = 5
        instance_num = 100

        self.__X, self.__y = make_classification(n_samples=instance_num, n_features=4, n_informative=2, n_redundant=2,
                                                 n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None,
                                                 flip_y=0.01,
                                                 class_sep=1.0,
                                                 hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)

        train_idx, test_idx, label_idx, unlabel_idx = split(
            X=self.__X,
            y=self.__y,
            test_ratio=0.3,
            initial_label_rate=0.05,
            split_count=split_count,
            all_class=True)

        test_idx.pop()

        # init the AlExperiment
        try:
            CrossValidationExperiment(
                X=self.__X,
                Y=self.__y,
                self_partition=False,
                kfolds=5,
                stopping_criteria=UnlabelSetEmpty(),
                train_idx=train_idx,
                test_idx=test_idx,
                label_idx=label_idx,
                unlabel_idx=unlabel_idx
            )
        except ValueError as valExc:
            assert ("Different numbers of folds for inputs" in "{0}".format(valExc))
        else:
            raise Exception("Expected ValueError exception")

    def test_cross_validation_without_self_partitioning_wrong_kfold_size(self):

        split_count = 5
        instance_num = 100

        self.__X, self.__y = make_classification(n_samples=instance_num, n_features=4, n_informative=2, n_redundant=2,
                                                 n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None,
                                                 flip_y=0.01,
                                                 class_sep=1.0,
                                                 hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)

        train_idx, test_idx, label_idx, unlabel_idx = split(
            X=self.__X,
            y=self.__y,
            test_ratio=0.3,
            initial_label_rate=0.05,
            split_count=split_count,
            all_class=True)

        train_idx.pop()
        test_idx.pop()
        label_idx.pop()
        unlabel_idx.pop()

        # init the AlExperiment
        try:
            CrossValidationExperiment(
                X=self.__X,
                Y=self.__y,
                self_partition=False,
                kfolds=5,
                stopping_criteria=UnlabelSetEmpty(),
                train_idx=train_idx,
                test_idx=test_idx,
                label_idx=label_idx,
                unlabel_idx=unlabel_idx
            )
        except ValueError as valExc:
            assert ("Number of folds for inputs" in "{0}".format(valExc))
        else:
            raise Exception("Expected ValueError exception")


if __name__ == '__main__':
    unittest.main()
