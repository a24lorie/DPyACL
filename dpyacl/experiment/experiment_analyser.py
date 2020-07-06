"""
Class to gathering, process and visualize active learning experiment results.
Added support for several performance metrics in a single experiment
"""
# Authors: Ying-Peng Tang, Alfredo Lorie

import collections
import copy
import warnings
from abc import abstractmethod, ABCMeta

import matplotlib.pyplot as plt
import numpy as np
import prettytable as pt
import scipy
from scipy import interpolate

__all__ = ['BaseAnalyser',
           'ExperimentAnalyserFactory',
           ]

from dpyacl.core.state import State, StateContainer


class BaseAnalyser(metaclass=ABCMeta):
    """
    Base Analyser class for analysing experiment result.
    Functions include various validity checking and visualizing of the given data.
    Normally, the results should be a list which contains k elements. Each element represents
    one fold experiment result.
    Legal result object includes:
        - State object.
        - A list contains n performances for n queries.
        - A list contains n tuples with 2 elements, in which, the first
          element is the x_axis (e.g., iteration, cost),
          and the second element is the y_axis (e.g., the performance)
    """

    def __init__(self, performance_metrics):
        # The data extracted from the original data.
        self._data_extracted = dict()
        # The summary of the data (each entry is optional according to the type of data):
        # 1. length
        # 2. batch_size
        # 3. performance mean and std
        # 4. cost
        self._data_summary = dict()
        self._performance_metrics = performance_metrics

    def get_methods_names(self):
        return self.__data_raw.keys()

    def get_extracted_data(self, method_name):
        return self._data_extracted[method_name]

    @abstractmethod
    def add_method(self, method_results, method_name):
        """Add the results of a method."""
        pass

    @abstractmethod
    def plot_learning_curves(self, *args, **kwargs):
        """Plot the performance curves of different methods."""
        pass

    # some commonly used tool function for experiment analysing.
    @classmethod
    def paired_ttest(cls, a, b, alpha=0.05):
        """Performs a two-tailed paired t-test of the hypothesis that two
        matched samples, in the arrays a and b, come from distributions with
        equal means. The difference a-b is assumed to come from a normal
        distribution with unknown variance.  a and b must have the same length.
        Parameters
        ----------
        a: array-like
            array for paired t-test.
        b: array-like
            array for paired t-test.
        alpha: float, optional (default=0.05)
            A value alpha between 0 and 1 specifying the
            significance level as (100*alpha)%. Default is
            0.05 for 5% significance.
        Returns
        -------
        H: int
            the result of the test.
            H=0     -- indicates that the null hypothesis ("mean is zero")
                    cannot be rejected at the alpha% significance level
                    (No significant difference between a and b).
            H=1     -- indicates that the null hypothesis can be rejected at the alpha% level
                    (a and b have significant difference).
        Examples
        -------
        >>> a = [1.2, 2, 3]
        >>> b = [1.6, 2.5, 1.1]
        >>> print(ExperimentAnalyser.paired_ttest(a, b))
        1
        """
        rava = a
        ravb = b
        # check a,b
        sh = np.shape(a)
        if len(sh) == 1:
            pass
        elif sh[0] == 1 or sh[1] == 1:
            rava = np.ravel(a)
            ravb = np.ravel(b)
        else:
            raise Exception("a or b must be a 1-D array. but received: %s" % str(sh))
        assert (len(a) == len(b))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            statistic, pvalue = scipy.stats.ttest_rel(rava, ravb)
        return int(pvalue <= alpha)

    def _type_of_data(self, result):
        """Judge type of data is given by the user.
        Returns
        -------
        type: int
            0 - State object.
            1 - A list contains n performances for n queries.
            2 - A list contains n tuples with 2 elements, in which, the first
                element is the x_axis (e.g., iteration, cost),
                and the second element is the y_axis (e.g., the performance)
        """
        if isinstance(result[0], State):
            return 0
        elif isinstance(result[0], (list, np.ndarray)):
            if isinstance(result[0][0], collections.abc.Iterable):
                if len(result[0][0]) > 1:
                    return 2
            return 1
        else:
            raise ValueError("Illegal result data is given.\n"
                             "Legal result object includes:\n"
                             "\t- State object.\n"
                             "\t- A list contains n performances for n queries.\n"
                             "\t- A list contains n tuples with 2 elements, in which, "
                             "the first element is the x_axis (e.g., iteration, cost),"
                             "and the second element is the y_axis (e.g., the performance)")


class _CostEffectiveAnalyser(BaseAnalyser, metaclass=ABCMeta):
    """Class to process the cost sensitive experiment results.
    The validity checking will depend only on the cost.
    """

    def __init__(self, performance_metrics):
        super().__init__(performance_metrics)

    def add_method(self, method_name, method_results):
        """
        Add results of a method.
        Parameters
        ----------
        method_results: {list, np.ndarray, StateContainer}
            experiment results of a method. contains k stateIO object or
            a list contains n tuples with 2 elements, in which, the first
            element is the x_axis (e.g., iteration, cost),
            and the second element is the y_axis (e.g., the performance)
        method_name: str
            Name of the given method.
        """
        if isinstance(method_results, (list, np.ndarray)):
            self.__add_list_result(method_name, method_results)
        elif isinstance(method_results, StateContainer):
            self.__add_stateio_container(method_name, method_results)
        else:
            raise TypeError('method_results must be one of {list, numpy.ndarray, StateContainer}.')

    def __add_stateio_container(self, method_name, method_results):
        self._is_all_stateio = True
        self._data_extracted[method_name] = method_results.extract_matrix(extract_keys=['cost', 'performance'])
        self._data_summary[method_name] = _ContentSummary(method_results=method_results.to_list(), method_type=0)

    def __add_list_result(self, method_name, method_results):
        self._is_all_stateio = True
        result_type = self._type_of_data(method_results)
        if result_type == 0:
            method_container = StateContainer(method_name=method_name, method_results=method_results)
            self._data_extracted[method_name] = method_container.extract_matrix(extract_keys=['cost', 'performance'])
            # get method summary
            # The summary, however, can not be inferred from a list of performances.
            # So if any lists of extracted data are given, we assume all the results are legal,
            # and thus we will not do further checking on it.
            self._data_summary[method_name] = _ContentSummary(method_results=method_results, method_type=result_type)
        elif result_type == 2:
            self._data_extracted[method_name] = copy.copy(method_results)
            self._is_all_stateio = False
            self._data_summary[method_name] = _ContentSummary(method_results=method_results, method_type=result_type)
        else:
            raise ValueError("Illegal result data in cost sensitive setting is given.\n"
                             "Legal result object includes:\n"
                             "\t- State object.\n"
                             "\t- A list contains n tuples with 2 elements, in which, "
                             "the first element is the x_axis (e.g., iteration, cost),"
                             "and the second element is the y_axis (e.g., the performance)")

    def _check_and_get_total_cost(self):
        """Check if the total cost is the same for all folds.
        Returns
        -------
        same: bool
            If the total cost for all folds are the same.
        effective_cost: float
            If the total cost are the same, return the total cost.
            Otherwise, return the min value of total cost for all folds.
        method_cost: dict
            The effective cost for all methods.
        """
        # return value initialize
        effective_cost = set()
        method_cost = dict()

        # gathering information
        for method_name in self._data_extracted.keys():
            total_cost_folds = []
            for fold in self._data_extracted[method_name]:
                total_cost_fold = [np.sum(query_info[0]) for query_info in fold]
                total_cost_folds.append(np.sum(total_cost_fold))

            method_unique_cost = np.unique(total_cost_folds)
            effective_cost.update(set(method_unique_cost))
            method_cost[method_name] = method_unique_cost
        # return
        same = True if len(effective_cost) == 1 else False
        return same, min(effective_cost), method_cost

    def plot_learning_curves(self, title=None, x_shift=0, start_point=None, plot_interval=None,
                             std_area=False, std_alpha=0.3, show=True):
        """
        Plot the performance curves.

        Parameters
        ----------
        title: str, optional (default=None)
            The tile of the figure.
        x_shift: float, optional (default=0)
            The shift value of x_axis.
            For example, the original x_axis is np.arange(0,100,1), x_shift = 1,
            then the new x_axis will be np.arange(1,101,1)
        start_point: float, optional (default=None)
            The value of start point. This value will added before the first data
            point for all methods. If not provided, an infer is attempted.
        plot_interval: float, optional (default=None)
            The interpolate interval in plotting the cost sensitive curves.
            The interpolate is needed because the x_axis is not aligned due to the different cost of labels.
            If not provided, it will use cost_budget/100 as the default interval.
        std_area: bool, optional (default=False)
            Whether to show the std values of the performance after each strategies.
        std_alpha: float, optional (default=0.3)
            The alpha value of the std shaded area.
            The smaller the value, the lighter the color.
        show: bool, optional (default=True)
            Whether to show the figure.
            If False, it will return the matplotlib.pyplot object directly.

        Returns
        -------
        plt: {matplotlib.pyplot, None}
            If passing `show=False`, the matplot object will be returned.
            Else, None will be returned.
        """
        same, effective_cost, method_cost = self._check_and_get_total_cost()
        interplt_interval = plot_interval if plot_interval is not None else effective_cost / 100

        # plotting
        for i in self._data_extracted.keys():
            # get un-aligned row data
            data_mat = self._data_extracted[i]
            x_axis = [[np.sum(tup[0]) for tup in line] for line in data_mat]
            # calc accumulative cost in x_axis
            for fold_num in range(len(x_axis)):
                ori_data = x_axis[fold_num]
                acc_data = [np.sum(ori_data[0:list_ind + 1]) for list_ind in range(len(ori_data))]
                x_axis[fold_num] = acc_data

            y_axis = [[tup[1] for tup in line] for line in data_mat]

            if start_point is None:
                # attempt to infer the start point
                if not self._is_all_stateio or self._data_summary[i].ip is None:
                    pass
                else:
                    for fold_num in range(len(y_axis)):
                        x_axis[fold_num].insert(0, 0)
                        y_axis[fold_num].insert(0, self._data_summary[i].ip)
            else:
                # Use the specified start point
                for fold_num in range(len(y_axis)):
                    x_axis[fold_num].insert(0, 0)
                    y_axis[fold_num].insert(0, start_point)

            # interpolate
            intplt_arr = []
            for fold_num in range(len(x_axis)):  # len(x_axis) == len(y_axis)
                intplt_arr.append(
                    interpolate.interp1d(x=x_axis[fold_num], y=y_axis[fold_num], bounds_error=False, fill_value=0.1))

            new_x_axis = np.arange(max([x[0] for x in x_axis]), effective_cost, interplt_interval)
            new_y_axis = []
            for fold_num in range(len(y_axis)):
                new_y_axis.append(intplt_arr[fold_num](new_x_axis))

            # plot data
            points = np.mean(new_y_axis, axis=0)
            if std_area:
                std_points = np.std(new_y_axis, axis=0)
            plt.plot(new_x_axis + x_shift, points, label=i)
            if std_area:
                plt.fill_between(new_x_axis, points - std_points, points + std_points,
                                 interpolate=True, alpha=std_alpha)

        # axis & title
        plt.legend(fancybox=True, framealpha=0.5)
        plt.xlabel("Cost")
        plt.ylabel("Performance")
        if title is not None:
            plt.title(str(title))

        # saving
        # if saving_path is not None:
        #     saving_path = os.path.abspath(saving_path)
        #     if os.path.isdir(saving_path):
        #         try:
        #             plt.savefig(os.path.join(saving_path, 'alipy_plotting.jpg'))
        #         except:
        #             plt.savefig(os.path.join(saving_path, 'alipy_plotting.pdf'))
        #     else:
        #         plt.savefig(saving_path)

        if show:
            try:
                # show before draw will raise an error in some versions of matplotlib
                plt.show()
            except:
                plt.draw()
                plt.show()
        else:
            return plt

    def __repr__(self):
        """summary of current methods."""
        same, effective_cost, method_cost = self._check_and_get_total_cost()
        tb = pt.PrettyTable()
        tb.field_names = ['Methods', 'number_of_different_split', 'performance', 'cost_budget']
        for i in self._data_extracted.keys():
            summary = self._data_summary[i]
            tb.add_row([i, summary.folds,
                        "%.3f ± %.2f" % (summary.mean, summary.std),
                        method_cost[i] if len(method_cost[i]) == 1 else 'Not same budget'])
        return '\n' + str(tb)


class _NumOfQueryAnalyser(BaseAnalyser, metaclass=ABCMeta):
    """Class to process the data whose x_axis is the number of strategies.
    The validity checking will depend only on the number of strategies.
    """

    def __init__(self, performance_metrics):
        super().__init__(performance_metrics)

    def add_method(self, method_name, method_results):
        """
        Add results of a method.
        Parameters
        ----------
        method_results: {list, np.ndarray, StateContainer}
            experiment results of a method. contains k stateIO object or
            a list contains n tuples with 2 elements, in which, the first
            element is the x_axis (e.g., iteration, accumulative_cost),
            and the second element is the y_axis (e.g., the performance)
        method_name: str
            Name of the given method.
        """
        if isinstance(method_results, (list, np.ndarray)):
            self.__add_list_result(method_name, method_results)
        elif isinstance(method_results, StateContainer):
            self.__add_stateio_container(method_name, method_results)
        else:
            raise TypeError('method_results must be one of {list, numpy.ndarray, StateContainer}.')

    def __add_stateio_container(self, method_name, method_results):
        self._is_all_stateio = True
        self._data_extracted[method_name] = method_results.extract_matrix()
        self._data_summary[method_name] = _ContentSummary(method_results=method_results.to_list(), method_type=0)

    def __add_list_result(self, method_name, method_results):
        """
        Add results of a method.
        Parameters
        ----------
        method_results: {list, np.ndarray}
            experiment results of a method. contains k stateIO object with k-fold experiment results.
        method_name: str
            Name of the given method.
        """

        # The type must be one of [0,1,2], otherwise, it will raise in that function.
        self._is_all_stateio = True
        result_type = self._type_of_data(method_results)

        if result_type == 0:
            method_container = StateContainer(method_name=method_name, method_results=method_results)
            self._data_extracted[method_name] = method_container.extract_matrix()
            self._data_summary[method_name]=[]
            for i in range(0, len(self._performance_metrics)):
                item = {}
                item["name"] = self._performance_metrics[i]
                item["value"] = _ContentSummary(metric_index=i, method_results=method_results, method_type=result_type)
                self._data_summary[method_name].append(item)

        elif result_type == 1:
            self._data_extracted[method_name] = copy.copy(method_results)
            self._is_all_stateio = False
            self._data_summary[method_name] = _ContentSummary(method_results=method_results, method_type=result_type)
        else:
            raise ValueError("The element in each list should be a single_label performance value.")

    def _check_plotting(self, metric_index):
        """
        check:
        1.NaN, Inf etc.
        2.methods_continuity
        """
        if not self._check_methods_continuity:
            warnings.warn('Settings among all methods are not the same. The difference will be ignored.')
        for i in self._data_extracted.keys():
            if np.isnan(list(map(lambda x: x[metric_index]["value"], self._data_extracted[i][0]))).any() != 0:
                raise ValueError('NaN is found in methods %s in %s.' % (
                    i, str(np.argwhere(np.isnan(self._data_extracted[i]) == True))))
            if np.isinf(list(map(lambda x: x[metric_index]["value"], self._data_extracted[i][0]))).any() != 0:
                raise ValueError('Inf is found in methods %s in %s.' % (
                    i, str(np.argwhere(np.isinf(self._data_extracted[i]) == True))))
        return True

    def _check_methods_continuity(self):
        """
        check if all methods have the same batch size, length and folds
        Returns
        -------
        result: bool
            True if the same, False otherwise.
        """
        first_flag = True
        bs = 0
        el = 0
        folds = 0
        ip = None
        for i in self._data_extracted.keys():
            summary = self._data_summary[i]
            if first_flag:
                bs = summary.batch_size
                el = summary.effective_length
                folds = summary.folds
                ip = summary.ip
                first_flag = False
            else:
                if bs != summary.batch_size or el != summary.effective_length or folds != summary.folds or not isinstance(
                        ip, type(summary.ip)):
                    return False
        return True

    def plot_learning_curves(self, title=None, x_shift=None, start_point=None, plot_interval=1,
                             std_area=False, std_alpha=0.3, show=True):
        """
        plotting the performance curves.
        Parameters
        ----------
        title: str, optioanl (default=None)
            The tile of the figure.
        x_shift: float, optional (default=None)
            The shift value of x_axis.
            For example, the original x_axis is np.arange(0,100,1), x_shift = 1,
            then the new x_axis will be np.arange(1,101,1)
        start_point: float, optional (default=None)
            The value of start point. This value will added before the first data
            point for all methods. If not provided, an infer is attempted.
        plot_interval: int, optional (default=1)
            The interval (x_axis) of each two data point.
            Default is 1, which means plot each data passed to the analyser.
        std_area: bool, optional (default=False)
            Whether to show the std values of the performance after each strategies.
        std_alpha: float, optional (default=0.3)
            The alpha value of the std shaded area.
            The smaller the value, the lighter the color.
        show: bool, optional (default=True)
            Whether to show the figure.
            If False, it will return the matplotlib.pyplot object directly.
        saving_path: str, optional (default='.')
            The path to save the image.
            Passing None to disable the saving.
        Returns
        -------
        plt: {matplotlib.pyplot, None}
            If passing `show=False`, the matplot object will be returned.
            Else, None will be returned.
        """
        assert len(self._data_extracted) > 0
        if self._is_all_stateio:
            for i in range(0, len(self._performance_metrics)):
                self._check_plotting(metric_index=i)
        plot_interval = int(round(plot_interval))

        # plotting
        for i in self._data_extracted.keys():
            legend = []
            for metric_index in range(0, len(self._performance_metrics)):
                data = [list(map(lambda x: x[metric_index]["value"], self._data_extracted[i][0]))]

                points = np.mean(data, axis=0)
                ori_ponits_len = len(points)
                if std_area:
                    std_points = np.std(data, axis=0)
                if plot_interval != 1:
                    points = np.asarray(
                        [points[point_ind] for point_ind in range(ori_ponits_len) if point_ind % plot_interval == 0])
                    if std_area:
                        std_points = np.asarray([std_points[point_ind] for point_ind in range(len(std_points)) if
                                                 point_ind % plot_interval == 0])
                if x_shift is None:
                    if not self._is_all_stateio or self._data_summary[i][metric_index]["value"].ip is None:
                        x_shift = 1
                    else:
                        x_shift = 0
                if start_point is not None:
                    x_shift = 0
                    legend.append("%s - %s" % (i, self._performance_metrics[metric_index]))
                    plt.plot(np.arange(ori_ponits_len + 1, step=plot_interval) + x_shift, [start_point] + list(points))

                    if std_area:
                        plt.fill_between(np.arange(ori_ponits_len, step=plot_interval) + x_shift + 1, points - std_points,
                                         points + std_points,
                                         interpolate=True, alpha=std_alpha)
                else:
                    legend.append("%s - %s" % (i, self._performance_metrics[metric_index]))
                    plt.plot(np.arange(ori_ponits_len, step=plot_interval) + x_shift, points)
                    if std_area:
                        plt.fill_between(np.arange(ori_ponits_len, step=plot_interval) + x_shift, points - std_points,
                                         points + std_points,
                                         interpolate=True, alpha=std_alpha)

        # axis & title
        plt.legend(legend, fancybox=True, framealpha=0.5)
        plt.xlabel("Number of queries")
        plt.ylabel("Performance")
        if title is not None:
            plt.title(str(title))

        if show:
            try:
                # show before draw will raise an error in some versions of matplotlib
                plt.show()
            except:
                plt.draw()
                plt.show()
        else:
            return plt

    def __repr__(self):
        """summary of current methods."""
        tb = pt.PrettyTable()
        tb.field_names = ['Methods', 'number_of_queries', 'number_of_different_split', 'performance']
        for i in self._data_extracted.keys():
            summary = self._data_summary[i]
            tb.add_row([i, summary.effective_length, summary.folds,
                        "%.3f ± %.2f" % (summary.mean, summary.std)])
        if self._is_all_stateio:
            tb.add_column('batch_size', [
                self._data_summary[i].batch_size if self._data_summary[i].batch_flag else 'Not_same_batch_size' for i
                in self._data_extracted.keys()])
        return '\n' + str(tb)


class _ContentSummary:
    """
    store summary info of a given method experiment result
    """

    def __init__(self, metric_index, method_results, method_type):
        self.method_type = method_type
        # basic info
        self.mean = 0
        self.std = 0
        self.folds = len(method_results)

        # for stateio object only
        self.batch_flag = False
        self.ip = None
        self.batch_size = 0

        # Only for num of strategies
        self.effective_length = 0

        # Only for Cost
        self.cost_inall = []

        if self.method_type == 0:  # A list of State object.
            self.stateio_summary(metric_index, method_results)
        else:
            self.list_summary(method_results)

    def stateio_summary(self, metric_index, method_results):
        """Calculate summary of a method.
        Parameters
        ----------
        metric_index:
        method_results: list
            A list of State object that contains experiment results of a method.
        """
        # examine the AlExperiment object
        if not np.all([sio.check_batch_size() for sio in method_results]):
            # warnings.warn('Checking validity fails, different batch size is found.',
            #               category=ValidityWarning)
            self.batch_flag = False
        else:
            bs = np.unique([sio.batch_size for sio in method_results])
            if len(bs) == 1:
                self.batch_flag = True
                self.batch_size = bs[0]

        result_len = [len(sio) for sio in method_results]
        # if len(np.unique(result_len))!=1:
        #     warnings.warn('Checking validity fails, different length of folds is found.',
        #                   category=ValidityWarning)
        self.effective_length = np.min(result_len)

        # get matrix
        ex_data = []
        for result in method_results:
            self.ip = result.initial_point
            one_fold_perf = [result[i].get_value('performance')[metric_index]["value"] for i in range(self.effective_length)]
            one_fold_cost = [result[i].get_value('cost') if 'cost' in result[i].keys() else 0 for i in
                             range(self.effective_length)]
            self.cost_inall.append(one_fold_cost)
            if self.ip is not None:
                one_fold_perf.insert(0, self.ip[metric_index]["value"])
            ex_data.append(one_fold_perf)
        mean_ex = np.mean(ex_data, axis=1)
        self.mean = np.mean(mean_ex)
        self.std = np.std(mean_ex)

    def list_summary(self, method_results):
        # Only for num of strategies
        self.effective_length = np.min([len(i) for i in method_results])
        if self.method_type == 1:
            # basic info
            self.mean = np.mean(method_results)
            self.std = np.std(method_results)
        else:
            perf_mat = [[np.sum(tup[1]) for tup in line] for line in method_results]
            cost_mat = [[tup[0] for tup in line] for line in method_results]
            mean_perf_for_each_fold = [np.mean(perf) for perf in perf_mat]
            self.mean = np.mean(mean_perf_for_each_fold)
            self.std = np.std(mean_perf_for_each_fold)
            # Only for Cost
            self.cost_inall = [np.sum(cost_one_fold) for cost_one_fold in cost_mat]


class ExperimentAnalyserFactory(metaclass=ABCMeta):
    """
    Class to gathering, process and visualize active learning experiment results.
    Normally, the results should be a list which contains k elements. Each element represents
    one fold experiment result.

    Legal result object includes:
        - State object.
        - A list contains n performances for n queries.
        - A list contains n tuples with 2 elements, in which, the first
          element is the x_axis (e.g., iteration, accumulative_cost),
          and the second element is the y_axis (e.g., the performance)

    Functions include:
        - Line chart (different x,y,axis, mean±std bars)
        - Paired t-test
    """

    @staticmethod
    def experiment_analyser(performance_metrics,
                            method_name,
                            method_results,
                            type='queries') -> BaseAnalyser:
        """
        Parameters
        ----------
        :param method_name: str
            Name of the given method.
        :param method_results: {list, np.ndarray, StateContainer}
            experiment results of a method. contains k stateIO object or
            a list contains n tuples with 2 elements, in which, the first
            element is the x_axis (e.g., iteration, cost),
            and the second element is the y_axis (e.g., the performance)
        :param type: str, optional (default='queries')
            The x_axis when analysing the result.
            x_axis should be one of ['queries', 'cost'],
            if 'cost' is given, your experiment results must contains the
            cost value for each performance value.

        Returns
        -------
        :return analyser: BaseAnalyser
            The experiment analyser object
        """
        if type not in ['queries', 'cost']:
            raise ValueError("type should be one of ['queries', 'cost'].")
        if type == 'queries':
            analyser = _NumOfQueryAnalyser(performance_metrics)
        else:
            analyser = _CostEffectiveAnalyser(performance_metrics)

        analyser.add_method(method_name=method_name, method_results=method_results)

        return analyser