"""
    Unit tests for metrics.py module
"""
import unittest
import numpy as np
from numcompute_stream.benchmarking import benchit

from numcompute_stream.benchmarking import quantile_loop, min_loop, max_loop, stan_dev_loop

from numcompute_stream.stats import StreamQuantileP2, StreamStats

import numcompute_stream

import random
import statistics


import unittest

import numpy as np


class TestBenchmark(unittest.TestCase):

    def test_quantile_values_produce_equal_results(self):
        test_arr_verify = [random.randint(1, 50) for _ in range(100)]

        # Verification Step (e.g both vectorized and Python loop functions produce similar output)

        quantile_stream = StreamQuantileP2(0.3)
        quantile_stream.update(test_arr_verify)

        np.testing.assert_allclose(quantile_stream.result(), quantile_loop(test_arr_verify, 0.3), rtol=0.1)


    def test_minimum_values_produce_equal_results(self):
        # Test inputs
        test_arr_verify = [random.randint(1, 50) for _ in range(100)]

        # Verification Step (e.g both vectorized and Python loop functions produce same output)

        stream_min = StreamStats().min_stream(test_arr_verify)

        np.testing.assert_allclose(min_loop(test_arr_verify), stream_min)

    def test_maximum_values_produce_equal_results(self):
        # Test inputs
        test_arr_verify = [random.randint(1, 50) for _ in range(100)]

        # Verification Step (e.g both vectorized and Python loop functions produce same output)

        stream_max = StreamStats().max_stream(test_arr_verify)

        np.testing.assert_allclose(max_loop(test_arr_verify), stream_max)


    def test_stan_dev_values_produce_equal_results(self):
        # Test inputs
        test_arr_verify = [random.randint(1, 50) for _ in range(100)]

        # Verification Step (e.g both vectorized and Python loop functions produce same output)
        stats = StreamStats().update_chunks(test_arr_verify)
        stream_stan_dev = stats.stan_dev()

        np.testing.assert_allclose(stan_dev_loop(test_arr_verify), stream_stan_dev)

    def test_partial_fit_imputer_values_produce_equal_results(self):
        test_arr_verify = np.full((5, 5), np.nan)
        test_pyloop_verify = test_arr_verify

        for i in range(len(test_pyloop_verify)):
            for j in range(len(test_pyloop_verify[i])):
                test_pyloop_verify[i][j] = -1

        # Verification Step (e.g both vectorized and Python loop functions produce same output)
        np.testing.assert_array_equal(numcompute_stream.SimpleImputer(fill_value=-1).partial_fit(test_arr_verify).transform(test_arr_verify),
                                      test_pyloop_verify)


