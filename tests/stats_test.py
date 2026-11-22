"""
    Unit tests for stats module
"""
import unittest

import numpy as np

from numcompute_stream import StreamStats, StreamHistogram, median, StreamQuantileP2




class statsTests(unittest.TestCase):
    def test_mean(self):
        test_data = np.array([10, 5, 20, 15])
        actual_mean = 12.5

        stats = StreamStats()
        test_mean = stats.update_chunks(test_data).mean
        self.assertEqual(test_mean, actual_mean)

    def test_median(self):
        test_data = np.array([10 , 5, 20, 5, 7])
        actual_median = 7

        test_median = median(test_data)
        self.assertEqual(test_median, actual_median)

    def test_stan_dev(self):
        test_data = np.array([5, 10, 5 , 10, 5, 10])
        actual_sd = 2.5

        stats = StreamStats()
        test_sd = stats.update_chunks(test_data).stan_dev()
        self.assertEqual(test_sd, actual_sd)

    def test_min(self):
        test_data = np.array([5,10,20,30])
        actual_min = 5

        stats = StreamStats()
        test_min = stats.min_stream(test_data)

        self.assertEqual(test_min, actual_min)

    def test_max(self):
        test_data = np.array([5,10,20,30])
        actual_max = 30

        stats = StreamStats()
        test_max = stats.max_stream(test_data)
        self.assertEqual(test_max, actual_max)

    def test_hist(self):
        test_data = np.array([0,1,2,3])
        test_bins = 4
        actual_hist = (np.array([1,1,1,1]),np.array([0, 0.75, 1.5, 2.25, 3]))

        histogram = StreamHistogram(test_bins).histogram(test_data)

        test_hist = histogram
        np.testing.assert_array_equal(actual_hist[0],test_hist[0])
        np.testing.assert_array_equal(actual_hist[1], test_hist[1])

    def test_quan(self):
        test_data = np.array([0,1,2,3])
        test_prob = 0.5
        actual_div = 1.5

        quantile = StreamQuantileP2(test_prob)
        test_div = quantile.update(test_data).result()
        self.assertEqual(test_div, actual_div)

    def test_nan_max(self):
        test_data = np.array([5,10,20,np.nan,30])
        actual_max = 30

        stats = StreamStats()
        test_max = stats.max_stream(test_data)
        self.assertEqual(test_max, actual_max)

    def test_nan_quan(self):
        test_data = np.array([0,1,2,3, np.nan])
        test_prob = 0.5
        actual_div = 1.5

        quantile = StreamQuantileP2(test_prob)
        test_div = quantile.update(test_data).result()
        self.assertEqual(test_div, actual_div)

    def test_nan_hist(self):
        test_data = np.array([0,1,2,3, np.nan])
        test_bins = 4
        actual_hist = (np.array([1,1,1,1]),np.array([0, 0.75, 1.5, 2.25, 3]))

        histogram = StreamHistogram(test_bins)
        test_hist = histogram.histogram(test_data)
        np.testing.assert_array_equal(actual_hist[0],test_hist[0])
        np.testing.assert_array_equal(actual_hist[1], test_hist[1])

if __name__ == "__main__":
    unittest.main()