import unittest


class TestMisc(unittest.TestCase):

    def test_check_2D_matrix(self):
        from dpyacl.core.misc.misc import check_2d_array
        "generate 4x8 matrix"
        matrix = [[0 for x in range(4)] for y in range(8)]
        check_2d_array(matrix)

    def test_check_1D_matrix(self):
        from dpyacl.core.misc.misc import check_2d_array
        "generate 4x8x12 matrix"
        matrix = [0 for x in range(4)]
        check_2d_array(matrix)


if __name__ == '__main__':
    unittest.main()
