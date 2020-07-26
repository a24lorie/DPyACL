
import unittest

import numpy as np

from dpyacl.core.collections import *


class TestIndexCollection(unittest.TestCase):

    def test_create_from_list(self):
        __index1 = IndexCollection([1, 2, 3])
        __index2 = IndexCollection([1, 2, 2, 3])

    def test_create_from_nparray(self):
        __index1 = IndexCollection(np.array([1, 2, 3]))
        __index1 = IndexCollection(np.array([1, 2, 2, 3]))

    def test_create_from_index_collection(self):
        __index1 = IndexCollection(IndexCollection([1, 2, 3]))

    def test_basic_index(self):
        __index1 = IndexCollection([1, 2, 3])
        __index2 = IndexCollection([1, 2, 2, 3])

        for item in __index1:
            assert item in __index2
        for item in __index2:
            assert item in __index1
        assert 1 in __index2
        assert len(__index1) == 3
        __index1.add(4)
        assert (len(__index1) == 4)
        __index1.discard(4)
        assert __index1.index == [1, 2, 3]
        __index1.update([4, 5])
        assert __index1.index == [1, 2, 3, 4, 5]
        __index1.difference_update([4, 5])
        assert __index1.index == [1, 2, 3]
        assert len(__index1.random_sampling(0.66)) == 2

    def test_warn_ind1(self):
        with pytest.warns(match=r'.*same elements in the given data'):
            a = IndexCollection([1, 2, 2, 3])
        with pytest.warns(match=r'.*has already in the collection.*'):
            a.add(3)
        a.add(4)
        with pytest.warns(match=r'.*to discard is not in the collection.*'):
            a.discard(6)
        assert a.pop() == 4
        with pytest.warns(match=r'.*has already in the collection.*'):
            a.update(IndexCollection([2, 9, 10]))
        with pytest.warns(match=r'.*to discard is not in the collection.*'):
            a.difference_update(IndexCollection([2, 100]))

    # def test_raise_ind1():
    #     # with pytest.raises(TypeError, match='Different types found in the given _indexes.'):
    #     #     a = IndexCollection([1, 0.5, ])
    #     b = IndexCollection([1, 2, 3, 4])
    #     with pytest.raises(TypeError, match=r'.*parameter is expected, but received.*'):
    #         b.update([0.2, 0.5])

    # def test_basic_multiind():
    #     assert len(multi_lab_ind1) == 6
    #     multi_lab_ind1.update((0, 0))
    #     assert len(multi_lab_ind1) == 7
    #     assert (0, 0) in multi_lab_ind1
    #     multi_lab_ind1.discard((0, 0))
    #     assert len(multi_lab_ind1) == 6
    #     assert (0, 0) not in multi_lab_ind1
    #     multi_lab_ind1.update([(1, 2), (1, (3, 4))])
    #     assert (1, 3) in multi_lab_ind1
    #     multi_lab_ind1.update([(2,)])
    #     assert (2, 0) in multi_lab_ind1
    #     with pytest.warns(InexistentElementWarning):
    #         multi_lab_ind1.difference_update([(0,)])
    #     assert (0, 1) not in multi_lab_ind1