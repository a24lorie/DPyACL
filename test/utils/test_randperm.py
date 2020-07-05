import unittest


class TestRandperm(unittest.TestCase):

    def test_randperm(self):
        from dpyacl.core.misc.misc import randperm
        randperm(10, 2)


if __name__ == '__main__':
    unittest.main()
