"""
Stopping Criteria Implementations with an Object Oriented design
"""
# Authors: Alfredo Lorie

import time
from abc import ABCMeta, abstractmethod

__all__ = ['IStopCriterion',
           'AbstractStopCriterion',
           'MaxIteration',
           'CostLimit',
           'PercentOfUnlabel',
           'UnlabelSetEmpty',
           'TimeLimit']


class IStopCriterion(metaclass=ABCMeta):
    """
    Class that represents a stop criterion
    Define additional criteria for stopping the active learning process
    """

    @abstractmethod
    def is_stop(self) -> bool:
        """
        Tells If the experiment must stop or continue.
        Parameters
        ----------
        kwargs: optional
        """
        pass

    @abstractmethod
    def update_information(self, saver):
        pass

    @abstractmethod
    def reset(self):
        pass


class AbstractStopCriterion(IStopCriterion, metaclass=ABCMeta):

    def __init__(self, value=50):
        self._value = value
        self._init_value = value

    @abstractmethod
    def is_stop(self):
        """
        Tells If the experiment must stop or continue.
        if so,return True.
        """
        pass

    @abstractmethod
    def update_information(self, saver):
        """
        update value according to the specific criterion.
        Parameters
        ----------
        saver: State
            State object
        """
        return self

    def reset(self):
        """
            Reset the current state to the initial.
        """
        self._value = self._init_value


class MaxIteration(AbstractStopCriterion, metaclass=ABCMeta):
    """
    Stop when preset number of queries is reached
    Parameters
    ----------
    value: int, default=50
        Max of iterations
    """

    def __init__(self, value=50):
        super().__init__(value)
        self._current_iter = 0

    def is_stop(self):
        if self._current_iter >= self._value:
            return True

        return False

    def update_information(self, saver):
        self._current_iter = len(saver)
        return self

    def reset(self):
        super().reset()
        self._value = self._init_value
        self._current_iter = 0


class CostLimit(AbstractStopCriterion, metaclass=ABCMeta):
    """
     Stop when cost reaches the limit.
    """

    def __init__(self, value=50):
        super().__init__(value)
        self._accum_cost = 0

    def is_stop(self):
        if self._accum_cost >= self._value:
            return True
        else:
            return False

    def update_information(self, saver):
        self._accum_cost = saver.cost_inall
        return self

    def reset(self):
        self._accum_cost = 0


class PercentOfUnlabel(AbstractStopCriterion, metaclass=ABCMeta):
    """
     Stop when specific percentage of unlabeled data pool is labeled.
    """

    def __init__(self, value=50):
        super().__init__(value)
        if self._value < 0 or self._value > 100:
            raise ValueError("Value of percent of unlabel should be  [0, 100].")
        self._percent = 0

    def is_stop(self):
        if self._percent >= self._value:
            return True
        else:
            return False

    def update_information(self, saver):
        super().update_information(saver)
        _, _, _, Uindex = saver.get_workspace()
        _, _, _, ini_Uindex = saver.get_workspace(iteration=0)
        self._percent = (len(ini_Uindex) - len(Uindex)) / len(ini_Uindex) * 100
        return self

    def reset(self):
        self._percent = 0


class UnlabelSetEmpty(PercentOfUnlabel, metaclass=ABCMeta):
    """
    Stop when no unlabeled samples available
    """

    def __init__(self):
        super().__init__(100)


class TimeLimit(AbstractStopCriterion, metaclass=ABCMeta):
    """
    Stop when CPU time reaches the limit
    """

    def __init__(self, value):
        super().__init__(value)
        self._start_time = time.process_time()

    def is_stop(self):
        if time.process_time() - self._start_time >= self._value:
            return True
        else:
            return False

    def update_information(self, saver):
        return self

    def reset(self):
        """
            Reset the current state to the initial.
        """
        self._start_time = time.process_time()