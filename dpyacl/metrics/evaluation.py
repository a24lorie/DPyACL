"""
Pre-defined Performance Metrics Implementations
Added new metrics and distributed processing capabilities with Dask
Modified implementation with an Object Oriented design
"""
# Authors: Guo-Xiang Li, Alfredo Lorie


from abc import ABCMeta, abstractmethod

import numpy as np
import sklearn
from dask import compute
from dask_ml import metrics
from scipy.sparse import csr_matrix

__all__ = [
    'BaseMetrics',
    'Accuracy',
    'Mse',
    'ZeroOneLoss',
    'F1',
    'HammingLoss',
    'RocAuc',
    'Precision',
    'Recall'
]


class BaseMetrics(metaclass=ABCMeta):

    @abstractmethod
    def compute(self, y_true, y_pred, sample_weight=None, **kwargs):
        """
        Abstract definition for computing the metric

        Parameters
        ----------
        :param y_true:  1d array-like, or label indicator array / sparse matrix
            Ground truth (correct) _labels.
        :param y_pred: 1d array-like, or label indicator array / sparse matrix
            Predicted _labels, as returned by a classifier.
        :param sample_weight : array-like of shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        :return: score : float
        """
        pass

    @abstractmethod
    def metric_name(self):
        """
        :return: metric name : str
        """
        pass

    def auc(self, x, y, reorder=True):
        """
        Compute Area Under the Curve (AUC) using the trapezoidal rule

        Parameters
        ----------
        :param x: array, shape = [n]
            x coordinates. These must be either monotonic increasing or monotonic decreasing.
        :param y: array, shape = [n]
            y coordinates.
        :param reorder: boolean optional (default=True)
            Whether to sort x before computing. If False, assume that x must be
            either monotonic increasing or monotonic decreasing. If True, y is
            used to break ties when sorting x. Make sure that y has a monotonic
            relation to x when setting reorder to True.

        Returns
        -------
        :return: area
        """
        self._check_consistent_length(x, y)

        if x.shape[0] < 2:
            raise ValueError('At least 2 points are needed to compute'
                             ' area under curve, but x.shape = %s' % x.shape)

        direction = 1
        if reorder is True:
            # reorder the data points according to the x axis and using y to
            # break ties
            order = np.lexsort((y, x))
            x, y = x[order], y[order]
        else:
            dx = np.diff(x)
            if np.any(dx < 0):
                if np.all(dx <= 0):
                    direction = -1
                else:
                    raise ValueError("x is neither increasing nor decreasing "
                                     ": {}.".format(x))

        area = direction * np.trapz(y, x)
        if isinstance(area, np.memmap):
            # Reductions such as .sum used internally in np.trapz do not return a
            # scalar by default for numpy.memmap instances contrary to
            # regular numpy.ndarray instances.
            area = area.dtype.type(area)
        return area

    def _num_samples(self, x):
        """
        Return number of samples in array-like x.

        Parameters
        ----------
        :param x

        Returns
        -------
        :return samples: int
        """
        if hasattr(x, 'fit') and callable(x.fit):
            # Don't get num_samples from an ensembles length!
            raise TypeError('Expected sequence or array-like, got '
                            'estimator %s' % x)
        if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
            if hasattr(x, '__array__'):
                x = np.asarray(x)
            else:
                raise TypeError("Expected sequence or array-like, got %s" %
                                type(x))
        if hasattr(x, 'shape'):
            if len(x.shape) == 0:
                raise TypeError("Singleton array %r cannot be considered"
                                " a valid collection." % x)
            return x.shape[0]
        else:
            return len(x)

    def _check_consistent_length(self, *arrays):
        """
        Check that all arrays have consistent first dimensions.

        Parameters
        ----------
        :param *array
        """
        lengths = [self._num_samples(X) for X in arrays if X is not None]
        uniques = np.unique(lengths)
        if len(uniques) > 1:
            raise ValueError("Found input variables with inconsistent numbers of"
                             " samples: %r" % [int(l) for l in lengths])

    def _type_of_target(self, y):
        """
        Determine the type of data indicated by the target.
        """
        y = np.asarray(y)
        if len(np.unique(y)) <= 2:
            if y.ndim >= 2 and len(y[0]) > 1:
                return 'multilabel'
            else:
                return 'binary'
        elif (len(np.unique(y)) > 2) or (y.ndim >= 2 and len(y[0]) > 1):
            return 'multiclass'
        return 'unknown'

    def _check_targets(self, y_true, y_pred):
        """Check that y_true and y_pred belong to the same classification task

        This converts multiclass or binary types to a common shape, and raises a
        ValueError for a mix of multilabel and multiclass targets, a mix of
        multilabel formats, for the presence of continuous-valued or multioutput
        targets, or for targets of different lengths.
        Column vectors are squeezed to 1d, while multilabel formats are returned
        as CSR sparse label indicators.

        Parameters
        ----------
        :param y_true : array-like
        :param y_pred : array-like

        Returns
        -------
        type_true : one of {'multilabel-indicator', 'multiclass', 'binary'}
        y_true : array or indicator matrix
        y_pred : array or indicator matrix
        """

        self._check_consistent_length(y_true, y_pred)
        type_true = self._type_of_target(y_true)
        type_pred = self._type_of_target(y_pred)

        y_type = set([type_true, type_pred])
        if y_type == set(["binary", "multiclass"]):
            y_type = set(["multiclass"])

        if len(y_type) > 1:
            raise ValueError("Classification metrics can't handle a mix of {0} "
                             "and {1} targets".format(type_true, type_pred))

        # We can't have more than one value on y_type => The set is no more needed
        y_type = y_type.pop()

        # No metrics support "multiclass-multioutput" format
        if y_type not in ["binary", "multiclass", "multilabel"]:
            raise ValueError("{0} is not supported".format(y_type))

        if y_type in ["binary", "multiclass"]:
            # Ravel column or 1d numpy array
            y_true = np.ravel(y_true)
            y_pred = np.ravel(y_pred)
            if y_type == "binary":
                unique_values = np.union1d(y_true, y_pred)
                if len(unique_values) > 2:
                    y_type = "multiclass"

        if y_type.startswith('multilabel'):
            y_true = csr_matrix(y_true)
            y_pred = csr_matrix(y_pred)
            y_type = 'multilabel'

        return y_type, y_true, y_pred


class Mse(BaseMetrics, metaclass=ABCMeta):
    """Accuracy classification score."""

    def __init__(self, squared=True):
        """
         :param squared : boolean value, optional (default = True)
            If True returns MSE value, if False returns RMSE value.
        """
        self._squared = squared

    @property
    def metric_name(self):
        """
        :return: metric name : str
        """
        if self._squared:
            return "rmse"
        else:
            return "mse"

    def compute(self, y_true, y_pred, sample_weight=None, **kwargs):
        """
        Parameters
        ----------
        :param y_true : 1d array-like, or label indicator array / sparse matrix
            Ground truth (correct) _labels.
        :param y_pred : 1d array-like, or label indicator array / sparse matrix
            Predicted _labels, as returned by a classifier.
        :param sample_weight : array-like of shape = [n_samples], optional
            Sample weights.
        :param normalize: bool, optional(default=True)
            If ``False``, return the number of misclassifications.
            Otherwise, return the fraction of misclassifications.

        Returns
        -------
        :return score : float
        """
        if self._squared:
            return metrics.mean_squared_error(y_true, compute(y_pred), sample_weight=None)
        else:
            return sklearn.metrics.mean_squared_error(y_true, y_pred, sample_weight=None, squared=self._squared)


class Accuracy(BaseMetrics, metaclass=ABCMeta):
    """Accuracy classification score."""

    @property
    def metric_name(self):
        """
        :return: metric name : str
        """
        return "accuracy_score"

    def compute(self, y_true, y_pred, sample_weight=None, **kwargs):
        """
        Parameters
        ----------
        :param y_true : 1d array-like, or label indicator array / sparse matrix
            Ground truth (correct) _labels.
        :param y_pred : 1d array-like, or label indicator array / sparse matrix
            Predicted _labels, as returned by a classifier.
        :param sample_weight : array-like of shape = [n_samples], optional
            Sample weights.
        :param normalize: bool, optional(default=True)
            If ``False``, return the number of misclassifications.
            Otherwise, return the fraction of misclassifications.

        Returns
        -------
        :return score : float
        """
        normalize = kwargs.pop("normalize", True)
        return metrics.accuracy_score(y_true, compute(y_pred), normalize=normalize, sample_weight=None)


class ZeroOneLoss(BaseMetrics, metaclass=ABCMeta):
    """
    Zero-one classification loss.
    """

    @property
    def metric_name(self):
        """
        :return: metric name : str
        """
        return "zero_one_loss"

    def compute(self, y_true, y_pred, sample_weight=None, **kwargs):
        """
        If normalize is ``True``, return the fraction of misclassifications
        (float), else it returns the number of misclassifications (int). The best
        performance is 0.

        Parameters
        ----------
        :param y_true : 1d array-like, or label indicator array / sparse matrix
            Ground truth (correct) _labels.
        :param y_pred : 1d array-like, or label indicator array / sparse matrix
            Predicted _labels, as returned by a classifier.
        :param sample_weight : array-like of shape = [n_samples], optional
            Sample weights.
        :param normalize: bool, optional(default=True)
            If ``False``, return the number of misclassifications.
            Otherwise, return the fraction of misclassifications.

        Returns
        -------
        :return score : float
            If ``normalize == True``, return the fraction of misclassifications
            (float), else it returns the number of misclassifications (int).
        """
        normalize = kwargs.pop("normalize", True)
        return sklearn.metrics.zero_one_loss(y_true, y_pred, normalize=normalize, sample_weight=None)


class F1(BaseMetrics, metaclass=ABCMeta):
    """
    F1 score, also known as balanced F-score or F-measure
    """

    def __init__(self, average='binary'):
        """
         :param average : string, [None, 'binary' (default), 'micro', 'macro', 'samples', 'weighted']
            This parameter is required for multiclass/multilabel targets.
            If ``None``, the scores for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:

            ``'binary'``:
                Only report results for the class specified by ``pos_label``.
                This is applicable only if targets (``y_{true,pred}``) are binary.
            ``'micro'``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``:
                Calculate metrics for each label, and find their average weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall.
            ``'samples'``:
                Calculate metrics for each instance, and find their average (only
                meaningful for multilabel classification where this differs from
                :func:`accuracy_score`).
        """

        if average not in [None, "binary", "micro", "macro", "weighted", "samples"]:
            raise ValueError("param average must be one of: [None 'binary', 'micro', 'macro', 'weighted', 'samples']")
        self._average = average

    @property
    def metric_name(self):
        """
        :return: metric name : str
        """
        return "f1_score"

    def compute(self, y_true, y_pred, sample_weight=None, **kwargs):
        """
        The F1 score can be interpreted as a weighted average of the precision and
        recall, where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are
        equal. The formula for the F1 score is::

            F1 = 2 * (precision * recall) / (precision + recall)

        Parameters
        ----------
        :param y_true : 1d array-like, or label indicator array / sparse matrix
            Ground truth (correct) target values.
        :param y_pred : 1d array-like, or label indicator array / sparse matrix
            Estimated targets as returned by a classifier.
        :param sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        :return f1_score : float or array of float, shape = [n_unique_labels]
            F1 score of the positive class in binary classification or weighted
            average of the F1 scores of each class for the multiclass task.
            """

        return sklearn.metrics.f1_score(y_true, y_pred,
                                        sample_weight=sample_weight,
                                        labels=None,
                                        pos_label=1,
                                        average=self._average,
                                        zero_division=0)


class HammingLoss(BaseMetrics, metaclass=ABCMeta):
    """
    The Hamming loss is the fraction of labels that are incorrectly predicted.
    """

    @property
    def metric_name(self):
        """
        :return: metric name : str
        """
        return "hamming_loss"

    def compute(self, y_true, y_pred, sample_weight=None, **kwargs):
        """
        The Hamming loss is the fraction of labels that are incorrectly predicted.

        Read more in the :ref:`User Guide <hamming_loss>`.

        Parameters
        ----------
        :param y_true : 1d array-like, or label indicator array / sparse matrix
            Ground truth (correct) labels.
        :param y_pred : 1d array-like, or label indicator array / sparse matrix
            Predicted labels, as returned by a classifier.
        :param sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        :return loss : float or int,
            Return the average Hamming loss between element of ``y_true`` and
            ``y_pred``.

        """

        return sklearn.metrics.hamming_loss(y_true, y_pred, sample_weight=None)


class RocAuc(BaseMetrics, metaclass=ABCMeta):
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)

    Note: this implementation is restricted to the binary classification task.
    """

    def __init__(self, average='macro', max_fpr=1, multi_class="raise"):
        """
         :param average : string, [None, 'binary' (default), 'micro', 'macro', 'samples', \
                           'weighted']
            This parameter is required for multiclass/multilabel targets.
            If ``None``, the scores for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:

            ``'binary'``:
                Only report results for the class specified by ``pos_label``.
                This is applicable only if targets (``y_{true,pred}``) are binary.
            ``'micro'``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``:
                Calculate metrics for each label, and find their average weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall.
            ``'samples'``:
                Calculate metrics for each instance, and find their average (only
                meaningful for multilabel classification where this differs from
                :func:`accuracy_score`).
        :param max_fpr : float > 0 and <= 1, default=None
            If not ``None``, the standardized partial AUC [2]_ over the range
            [0, max_fpr] is returned. For the multiclass case, ``max_fpr``,
            should be either equal to ``None`` or ``1.0`` as AUC ROC partial
            computation currently is not supported for multiclass.

        :param multi_class : {'raise', 'ovr', 'ovo'}, default='raise'
            Multiclass only. Determines the type of configuration to use. The
            default value raises an error, so either ``'ovr'`` or ``'ovo'`` must be
            passed explicitly.

            ``'ovr'``:
                Computes the AUC of each class against the rest [3]_ [4]_. This
                treats the multiclass case in the same way as the multilabel case.
                Sensitive to class imbalance even when ``average == 'macro'``,
                because class imbalance affects the composition of each of the
                'rest' groupings.
            ``'ovo'``:
                Computes the average AUC of all possible pairwise combinations of
                classes [5]_. Insensitive to class imbalance when
                ``average == 'macro'``.
        """

        if average not in [None, "binary", "micro", "macro", "weighted", "samples"]:
            raise ValueError("param average must be one of: [None, 'binary', 'micro', 'macro', 'weighted', 'samples']")
        self._average = average

        if max_fpr <= 0 or max_fpr > 1:
            raise ValueError("param max_fpr must be one between [0,1)")
        self._max_fpr = max_fpr

        if multi_class not in ["raise", "ovr", "ovo"]:
            raise ValueError("param average must be one of: ['raise', 'ovr', 'ovo']")
        self._multi_class = multi_class

    @property
    def metric_name(self):
        """
        :return: metric name : str
        """
        return "roc_auc_score"

    def compute(self, y_true, y_pred, sample_weight=None, **kwargs):
        """
        Compute Receiver operating characteristic (ROC)
        Note: this implementation can be used with binary, multiclass and
        multilabel classification, but some restrictions apply (see Parameters).

        Parameters
        ----------
        :param y_true : array-like of shape (n_samples,) or (n_samples, n_classes)
            True labels or binary label indicators. The binary and multiclass cases
            expect labels with shape (n_samples,) while the multilabel case expects
            binary label indicators with shape (n_samples, n_classes).
        :param y_score : array-like of shape (n_samples,) or (n_samples, n_classes)
            Target scores. In the binary and multilabel cases, these can be either
            probability estimates or non-thresholded decision values (as returned
            by `decision_function` on some classifiers). In the multiclass case,
            these must be probability estimates which sum to 1. The binary
            case expects a shape (n_samples,), and the scores must be the scores of
            the class with the greater label. The multiclass and multilabel
            cases expect a shape (n_samples, n_classes). In the multiclass case,
            the order of the class scores must correspond to the order of
            ``labels``, if provided, or else to the numerical or lexicographical
            order of the labels in ``y_true``.
        :param sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        auc : float
        """

        return sklearn.metrics.roc_auc_score(y_true, y_pred, average=self._average,
                                             sample_weight=sample_weight,
                                             max_fpr=self._max_fpr,
                                             multi_class=self._multi_class,
                                             labels=None)


class Precision(BaseMetrics, metaclass=ABCMeta):
    """
    Compute the precision

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.
    """

    def __init__(self, average='binary'):
        """
        Compute the precision

        :param average : string, [None, 'binary' (default), 'micro', 'macro', 'samples', 'weighted']
            This parameter is required for multiclass/multilabel targets.
            If ``None``, the scores for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:

            ``'binary'``:
                Only report results for the class specified by ``pos_label``.
                This is applicable only if targets (``y_{true,pred}``) are binary.
            ``'micro'``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``:
                Calculate metrics for each label, and find their average weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall.
            ``'samples'``:
                Calculate metrics for each instance, and find their average (only
                meaningful for multilabel classification where this differs from
                :func:`accuracy_score`).
        """

        if average not in [None, "binary", "micro", "macro", "weighted", "samples"]:
            raise ValueError("param average must be one of: [None 'binary', 'micro', 'macro', 'weighted', 'samples']")
        self._average = average

    @property
    def metric_name(self):
        """
        :return: metric name : str
        """
        return "precision_score"

    def compute(self, y_true, y_pred, sample_weight=None, **kwargs):
        """
        Compute the precision

        Parameters
        ----------
        :param y_true : 1d array-like, or label indicator array / sparse matrix
            Ground truth (correct) target values.
        :param y_pred : 1d array-like, or label indicator array / sparse matrix
            Estimated targets as returned by a classifier.
        :param sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        :return precision_score : float (if average is not None) or array of float,
            shape = [n_unique_labels]
            Precision of the positive class in binary classification or weighted
            average of the precision of each class for the multiclass task..
        """

        return sklearn.metrics.precision_score(y_true, y_pred,
                                               sample_weight=sample_weight,
                                               labels=None,
                                               pos_label=1,
                                               average=self._average,
                                               zero_division=0)


class Recall(BaseMetrics, metaclass=ABCMeta):
    """
        Compute the precision

        The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
        true positives and ``fp`` the number of false positives. The precision is
        intuitively the ability of the classifier not to label as positive a sample
        that is negative.
        """

    def __init__(self, average='binary'):
        """
        Compute the precision

        :param average : string, [None, 'binary' (default), 'micro', 'macro', 'samples', 'weighted']
            This parameter is required for multiclass/multilabel targets.
            If ``None``, the scores for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:

            ``'binary'``:
                Only report results for the class specified by ``pos_label``.
                This is applicable only if targets (``y_{true,pred}``) are binary.
            ``'micro'``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``:
                Calculate metrics for each label, and find their average weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall.
            ``'samples'``:
                Calculate metrics for each instance, and find their average (only
                meaningful for multilabel classification where this differs from
                :func:`accuracy_score`).
        """

        if average not in [None, "binary", "micro", "macro", "weighted", "samples"]:
            raise ValueError("param average must be one of: [None 'binary', 'micro', 'macro', 'weighted', 'samples']")
        self._average = average

    @property
    def metric_name(self):
        """
        :return: metric name : str
        """
        return "recall_score"

    def compute(self, y_true, y_pred, sample_weight=None, **kwargs):
        """
        Compute the precision

        Parameters
        ----------
        :param y_true : 1d array-like, or label indicator array / sparse matrix
            Ground truth (correct) target values.
        :param y_pred : 1d array-like, or label indicator array / sparse matrix
            Estimated targets as returned by a classifier.
        :param sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        :return recall_score : float (if average is not None) or array of float,
            shape = [n_unique_labels]
            Recall of the positive class in binary classification or weighted
            average of the recall of each class for the multiclass task.
        """

        return sklearn.metrics.recall_score(y_true, y_pred,
                                            sample_weight=sample_weight,
                                            labels=None,
                                            pos_label=1,
                                            average=self._average,
                                            zero_division=0)