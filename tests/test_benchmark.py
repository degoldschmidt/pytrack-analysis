""" Tests for benchmark module. """
# -*- coding: utf-8 -*-
# author: degoldschmidt
# date: 25/8/2017
# -*- reviewed
import unittest
from pytrack_analysis import Benchmark, Multibench


""" Simple function for benchmarking """
def square_of_nine(): return 9*9

""" Simple function for exception handling """
def raise_KeyError():
    a = {"b": 3}
    return a["c"] # not defined in dictionary -> raises KeyError


class TestBenchmark(unittest.TestCase):
    """
    Basic test class
    """

    def test_bench(self):
        with Benchmark("") as result:
            res = square_of_nine()
        self.assertEqual(res, 81)

    def test_exceptions(self):
        with Benchmark("") as result:
            self.assertRaises(KeyError, raise_KeyError)

    def test_multi(self):
        """
        The actual test.
        Any method which starts with ``test_`` will considered as a test case.
        """
        test = Multibench("", times = 5)
        res = test(square_of_nine)
        del test
        self.assertEqual(res, 81)

if __name__ == '__main__':
    unittest.main()
