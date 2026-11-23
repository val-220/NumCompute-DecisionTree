"""
Unit tests for the preprocessing.py module
"""

import unittest

import numpy as np

from numcompute_stream.exceptions import NotFittedError
from numcompute_stream.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
)

from itertools import batched

class TestStandardScaler(unittest.TestCase):
    def test_output_has_zero_mean_unit_std(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

        result = StandardScaler().fit_transform(X)

        np.testing.assert_allclose(result.mean(axis=0), [0, 0], atol=1e-12)
        np.testing.assert_allclose(result.std(axis=0), [1, 1])

    def test_constant_column_becomes_zeros(self):
        X = np.full((3, 1), 0.1)

        result = StandardScaler().fit_transform(X)

        np.testing.assert_allclose(result, np.zeros((3, 1)), atol=1e-12)

    def test_transform_before_fit_raises(self):
        with self.assertRaises(NotFittedError):
            StandardScaler().transform([[1.0, 2.0]])

    def test_wrong_feature_count_raises(self):
        scaler = StandardScaler().fit([[1.0, 2.0], [3.0, 4.0]])

        with self.assertRaises(ValueError):
            scaler.transform([[1.0, 2.0, 3.0]])

    """Partial fit tests"""
    def test_partial_output_has_zero_mean_unit_std(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

        chunked = StandardScaler()

        for row_chunk in batched(X, 2):
             chunked.partial_fit(row_chunk)

        result = chunked.transform(X)

        np.testing.assert_allclose(result.mean(axis=0), [0, 0], atol=1e-12)
        np.testing.assert_allclose(result.std(axis=0), [1, 1])

    def test_partial_constant_column_becomes_zeros(self):
        X = np.full((3, 1), 0.1)

        chunked = StandardScaler()

        for row_chunk in batched(X, 2):
             chunked.partial_fit(row_chunk)

        result = chunked.transform(X)

        np.testing.assert_allclose(result, np.zeros((3, 1)), atol=1e-12)

    def test_partial_wrong_feature_count_raises(self):

        X = np.array([[1.0, 2.0], [3.0, 4.0]])

        chunked = StandardScaler()
        for row_chunk in batched(X, 2):
             chunked.partial_fit(row_chunk)

        with self.assertRaises(ValueError):
            chunked.transform([[1.0, 2.0, 3.0]])

    def test_chunk_invariance(self):
        x_whole = np.array([[1.0, 2.0], [3.0, 4.0]])

        scaler_whole = StandardScaler().partial_fit(x_whole).transform(x_whole)

        scaler_chunked = StandardScaler()
        for i in range(len(x_whole)):
            scaler_chunked.partial_fit(x_whole[i:i + 1])

        np.testing.assert_allclose(scaler_chunked.transform(x_whole), scaler_whole, atol=1e-12)




class TestMinMaxScaler(unittest.TestCase):
    def test_default_range(self):
        X = np.array([[1.0], [2.0], [3.0]])

        result = MinMaxScaler().fit_transform(X)

        np.testing.assert_allclose(result, [[0.0], [0.5], [1.0]])

    def test_custom_range(self):
        X = np.array([[1.0], [2.0], [3.0]])

        result = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)

        np.testing.assert_allclose(result, [[-1.0], [0.0], [1.0]])

    def test_constant_column_maps_to_range_min(self):
        X = np.full((3, 1), 5.0)

        result = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)

        np.testing.assert_allclose(result, np.full((3, 1), -1.0))

    def test_invalid_range_raises(self):
        with self.assertRaises(ValueError):
            MinMaxScaler(feature_range=(1, 0))


    def test_chunk_invariance(self):
        x_whole = np.array([[1.0, 2.0], [3.0, 4.0]])

        minmax_whole = MinMaxScaler().partial_fit(x_whole).transform(x_whole)

        minmax_chunked = MinMaxScaler()
        for i in range(len(x_whole)):
            minmax_chunked.partial_fit(x_whole[i:i + 1])

        np.testing.assert_allclose(minmax_chunked.transform(x_whole), minmax_whole, atol=1e-12)


class TestOneHotEncoder(unittest.TestCase):
    def setUp(self):
        self.X = [["red", "S"], ["blue", "L"], ["red", "L"]]

    def test_encoding(self):
        # Categories are sorted: ["blue", "red"] and ["L", "S"].
        expected = np.array(
            [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 1, 0],
            ]
        )

        result = OneHotEncoder().fit_transform(self.X)

        np.testing.assert_array_equal(result, expected)

    def test_unknown_category_raises(self):
        encoder = OneHotEncoder().fit(self.X)

        with self.assertRaises(ValueError):
            encoder.transform([["green", "S"]])

    def test_unknown_category_ignored(self):
        encoder = OneHotEncoder(handle_unknown="ignore").fit(self.X)

        result = encoder.transform([["green", "S"]])

        np.testing.assert_array_equal(result, [[0, 0, 0, 1]])

    """ Partial fit tests"""
    def test_partial_fit_encoding(self):
        expected = np.array(
            [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 1, 0],
            ]
        )

        encoder = OneHotEncoder().partial_fit(self.X)
        np.testing.assert_array_equal(encoder.transform(self.X), expected)

    def test_chunk_invariance(self):

        expected_outcome = OneHotEncoder().partial_fit(self.X).transform(self.X)

        chunked = OneHotEncoder()
        for row_chunk in batched(self.X,2):
            chunked.partial_fit(row_chunk)

        np.testing.assert_array_equal(chunked.transform(self.X), expected_outcome)


class TestSimpleImputer(unittest.TestCase):
    def test_nan_replaced_with_fill_value(self):
        X = np.array([[1.0, np.nan], [np.nan, 4.0]])

        result = SimpleImputer(fill_value=-1).fit_transform(X)

        np.testing.assert_array_equal(result, [[1.0, -1.0], [-1.0, 4.0]])

    def test_input_not_modified(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0]])

        SimpleImputer().fit_transform(X)

        self.assertTrue(np.isnan(X[0, 1]))

    def test_non_numeric_fill_value_raises(self):
        with self.assertRaises(TypeError):
            SimpleImputer(fill_value="missing")


    """ Partial fit tests"""
    def test_partial_fit_input_not_modified(self):

        ## Input not modified
        X = np.array([[1.0, np.nan], [3.0, 4.0]])

        for row_chunk in batched(X,2):
            not_modified = SimpleImputer().partial_fit(row_chunk).transform(row_chunk)

        self.assertTrue(np.isnan(X[0, 1]))

    def test_partial_fit_nan_replaced_with_fill_value(self):

        X = np.array([[1.0, np.nan], [3.0, 4.0]])

        for row_chunk in batched(X, 2):
            fill = SimpleImputer(fill_value=-1).partial_fit(row_chunk).transform(row_chunk)

        np.testing.assert_array_equal(fill, [[1.0, -1.0], [3.0, 4.0]])


    def test_chunk_invariance(self):

        X = np.array([[1.0, np.nan], [np.nan, 4.0]])

        whole = SimpleImputer(fill_value=-1).partial_fit(X).transform(X)

        for row_chunk in batched(X, 2):
            fill = SimpleImputer(fill_value=-1).partial_fit(row_chunk).transform(row_chunk)

        np.testing.assert_array_equal(fill, whole)






if __name__ == "__main__":
    unittest.main()
