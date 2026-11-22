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

    """ Correct gini entropy calculations"""


    """Correct split with max_depth"""

    def test_split(self):
        tree = DecisionTreeClassifier(max_depth = 1)
        # Using XOR data, each side should have 1 zero and 1 one and accuracy becomes 0.5
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 0])
        tree.fit(x, y)

        assert np.mean(tree.predict(x) == y) == pytest.approx(0.5)