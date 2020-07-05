from __future__ import division

import copy
import unittest

from dpyacl.core.stop_criteria import MaxIteration, CostLimit, PercentOfUnlabel, UnlabelSetEmpty, TimeLimit
from dpyacl.core import State, StateItem


class TestStopCriteria(unittest.TestCase):
    __example_saver = State(round=0, train_idx=list(range(8)), test_idx=[8, 9, 10], init_L=[0, 1],
                            init_U=[2, 3, 4, 5, 6, 7])

    def test_max_iteration(self):
        max_iteration = MaxIteration(value=10)

        assert not max_iteration.is_stop()
        max_iteration.update_information(self.__example_saver)
        example_saver_local = copy.deepcopy(self.__example_saver)
        assert max_iteration._current_iter == 0

        example_saver_local.add_state(
            StateItem(select_index=[2],
                      performance=[{"name": "accuracy_score", "value": 0.89}],
                      performance_metrics=["accuracy_score"])
        )
        max_iteration.update_information(example_saver_local)

        assert max_iteration._current_iter == 1
        assert not max_iteration.is_stop()

        max_iteration._current_iter = 10
        assert max_iteration.is_stop()

    def test_cost_limit(self):
        cost_limit = CostLimit(value=10)

        assert not cost_limit.is_stop()
        cost_limit.update_information(self.__example_saver)
        example_saver_local = copy.deepcopy(self.__example_saver)

        assert cost_limit._accum_cost == 0
        example_saver_local.add_state(
            StateItem(select_index=[2],
                      performance=[{"name": "accuracy_score", "value":0.89}],
                      cost=[3],
                      performance_metrics=["accuracy_score"])
        )
        cost_limit.update_information(example_saver_local)

        assert cost_limit._accum_cost == 3
        assert not cost_limit.is_stop()

        example_saver_local.add_state(
            StateItem(select_index=[3],
                      performance=[{"name": "accuracy_score", "value":0.89}],
                      cost=[7],
                      performance_metrics=["accuracy_score"])
        )
        cost_limit.update_information(example_saver_local)

        assert cost_limit._accum_cost == 10
        assert cost_limit.is_stop()

    def test_percent_of_unlabel(self):
        percent_of_unlabel = PercentOfUnlabel(10)
        assert not percent_of_unlabel.is_stop()

        percent_of_unlabel.update_information(self.__example_saver)
        example_saver_local = copy.deepcopy(self.__example_saver)

        assert percent_of_unlabel._percent == 0
        example_saver_local.add_state(
            StateItem(select_index=[2],
                      performance=[{"name": "accuracy_score", "value": 0.89}],
                      cost=[3],
                      performance_metrics=["accuracy_score"])
        )

        percent_of_unlabel.update_information(example_saver_local)
        assert percent_of_unlabel._percent == 1 / 6 * 100
        assert percent_of_unlabel.is_stop()

    def test_unlabel_set_empty(self):
        unlabel_set_empty = UnlabelSetEmpty()
        assert not unlabel_set_empty.is_stop()

        unlabel_set_empty.update_information(self.__example_saver)
        example_saver_local = copy.deepcopy(self.__example_saver)

        assert unlabel_set_empty._percent == 0
        example_saver_local.add_state(
            StateItem(select_index=[2],
                      performance=[{"name": "accuracy_score", "value": 0.89}],
                      cost=[3],
                      performance_metrics=["accuracy_score"])
        )
        unlabel_set_empty.update_information(example_saver_local)

        assert unlabel_set_empty._percent == 1 / 6 * 100
        assert not unlabel_set_empty.is_stop()

        example_saver_local.add_state(
            StateItem(select_index=[3],
                      performance=[{"name": "accuracy_score", "value": 0.89}],
                      cost=[3],
                      performance_metrics=["accuracy_score"])
        )
        unlabel_set_empty.update_information(example_saver_local)

        assert unlabel_set_empty._percent == 2 / 6 * 100
        assert not unlabel_set_empty.is_stop()

        example_saver_local.add_state(
            StateItem(select_index=[4],
                      performance=[{"name": "accuracy_score", "value": 0.89}],
                      cost=[3],
                      performance_metrics=["accuracy_score"])
        )
        unlabel_set_empty.update_information(example_saver_local)

        assert unlabel_set_empty._percent == 3 / 6 * 100
        assert not unlabel_set_empty.is_stop()

        example_saver_local.add_state(
            StateItem(select_index=[5],
                      performance=[{"name": "accuracy_score", "value": 0.89}],
                      cost=[3],
                      performance_metrics=["accuracy_score"])
        )
        unlabel_set_empty.update_information(example_saver_local)

        assert unlabel_set_empty._percent == 4 / 6 * 100
        assert not unlabel_set_empty.is_stop()

        example_saver_local.add_state(
            StateItem(select_index=[6],
                      performance=[{"name": "accuracy_score", "value": 0.89}],
                      cost=[3],
                      performance_metrics=["accuracy_score"])
        )
        unlabel_set_empty.update_information(example_saver_local)

        assert unlabel_set_empty._percent == 5 / 6 * 100
        assert not unlabel_set_empty.is_stop()

        example_saver_local.add_state(
            StateItem(select_index=[7],
                      performance=[{"name": "accuracy_score", "value": 0.89}],
                      cost=[3],
                      performance_metrics=["accuracy_score"])
        )
        unlabel_set_empty.update_information(example_saver_local)

        assert unlabel_set_empty._percent == 100
        assert unlabel_set_empty.is_stop()

    def test_time_limit(self):
        time_limit = TimeLimit(0.2)
        assert not time_limit.is_stop()

        time_limit.update_information(self.__example_saver)
        example_saver_local = copy.deepcopy(self.__example_saver)

        for i in range(2000):
            example_saver_local.add_state(
                StateItem(select_index=[i],
                          performance=[{"name": "accuracy_score", "value": 0.89}],
                          cost=[3],
                          performance_metrics=["accuracy_score"])
            )
            time_limit.update_information(example_saver_local)

        assert time_limit.is_stop()


if __name__ == '__main__':
    unittest.main()
