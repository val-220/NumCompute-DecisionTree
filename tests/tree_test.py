"""
    Unit tests for tree module
"""
import unittest

import numpy as np

from numcompute_stream import DecisionTreeClassifier

import logging
import unittest

import numpy as np

import pytest

from numcompute_stream.stream import StreamTrainer

from unittest.mock import MagicMock

class TestDecisionTree(unittest.TestCase):

    """Flipped labels test"""
    def test_flipped_labels(self):
        tree = DecisionTreeClassifier()
        x = np.array([[0],[0],[1],[1]])
        y = np.array([1,1,0,0])

        tree.fit(x, y)
        assert list(tree.predict(np.array(x))) == [1,1,0,0]

    """Correct split with max_depth"""

    def test_split(self):
        tree = DecisionTreeClassifier(max_depth = 1)
        # Using XOR data, each side should have 1 zero and 1 one and accuracy becomes 0.5
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 0])
        tree.fit(x, y)

        assert np.mean(tree.predict(x) == y) == pytest.approx(0.5)


    """Stream/partial_fit Test with different buffer sizes"""
    def test_stream(self):

        # With Buffer size 4
        tree_1= DecisionTreeClassifier(buffer_size=4)

        x_chunk= np.array([[0],[0]])
        y_chunk= np.array([0,0])

        x_chunk2 = np.array([[1],[1]])
        y_chunk2 = np.array([1,1])

        tree_1.partial_fit(x_chunk, y_chunk)

        tree_1.partial_fit(x_chunk2, y_chunk2)

        assert list(tree_1.predict([[0], [1]])) == [0,1]

        # with buffer size 2
        tree_2= DecisionTreeClassifier(buffer_size=2)

        tree_2.partial_fit(x_chunk, y_chunk)

        tree_2.partial_fit(x_chunk2, y_chunk2)

        assert list(tree_2.predict([[0], [1]])) == [1,1]

    """Chunk-size invariance test"""
    def test_chunk_invariance(self):

        tree_whole = DecisionTreeClassifier(max_depth = 8)

        x_chunk = np.array([[0], [0], [1], [1], [0], [1], [0], [1]])
        y = np.array([0, 0, 1, 1, 0, 1, 0, 1])  # label equals the feature

        tree_whole.fit(x_chunk, y)

        tree_one_by_one = DecisionTreeClassifier(max_depth = 8)
        for chunk in range(8):
            tree_one_by_one.partial_fit(x_chunk[chunk:chunk+1], y[chunk:chunk+1])

        # The predictions should be the same no matter how the chunk is fed.
        assert list(tree_one_by_one.predict([[0], [1]])) == list(tree_whole.predict([[0], [1]]))
