"""
    Unit tests for stream/streamtrainer module
"""
import logging
import unittest

import numpy as np

import pytest

from numcompute_stream.stream import StreamTrainer

from unittest.mock import MagicMock

class TestStreamTrainer(unittest.TestCase):

    """fit_chunk output tests"""
    def test_fit_chunk(self):
        pipe, model = MagicMock(), MagicMock()
        pipe.partial_fit_transform.return_value =[[10, 20], [30, 40]]
        x= [[1, 2],[3, 4]]
        y= [0,1]

        trainer = StreamTrainer(model, pipe)

        trainer.fit_chunk(x, y)

        # pipeline received the raw X
        pipe.partial_fit_transform.assert_called_once_with([[1, 2], [3, 4]])

        # model received the transformed X and the original y
        model.partial_fit.assert_called_once_with([[10, 20], [30, 40]], [0, 1])

    """Score_chunk output tests"""

    """ Test logging output"""
    def test_logging(self):
            pipe, model = MagicMock(), MagicMock()
            model.predict.side_effect = [np.array([1, 0, 0, 0])]
            trainer = StreamTrainer(model, pipe)


            with self.assertLogs(level=logging.INFO) as cm:
                trainer.score_chunk([[0]] * 4, np.array([1, 0, 1, 1]))

            self.assertEqual(len(cm.records),1)
            msg = cm.records[0].msg
            assert msg["Accuracy"] == pytest.approx(0.5)
            assert msg["Precision"] == pytest.approx(1.0)
            assert msg["Recall"] == pytest.approx(1 / 3)
            assert msg["F1 Score"] == pytest.approx(0.5)
            assert msg["MSE"] == pytest.approx(0.5)
            assert tuple(msg["Confusion Matrix"]) == (1, 0, 1, 2)

    """Testing confusion matrix output"""
    def test_confusion_matrix_counts_each_sample_once(self):
        pipe, model = MagicMock(), MagicMock()
        trainer = StreamTrainer(model, pipe)

        x = np.zeros((4, 1))
        y = np.array([1, 1, 0, 0])

        model.predict.side_effect = [np.array([1, 0, 0, 0])]

        with self.assertLogs(level=logging.INFO) as cm:
            trainer.score_chunk(x, y)
        result = cm.records[0].msg
        assert sum(result["Confusion Matrix"]) == 4

    """All wrong predicted values produces a correct confusion matrix"""
    def test_confusion_matrix_all_wrong(self):
        pipe, model = MagicMock(), MagicMock()
        trainer = StreamTrainer(model, pipe)

        X = np.zeros((4, 1))
        y = np.array([1, 0, 1, 0])

        model.predict.side_effect = [np.array([0, 1, 0, 1])]

        with self.assertLogs(level=logging.INFO) as cm:
            trainer.score_chunk(X, y)
        msg = cm.records[0].msg
        assert tuple(msg["Confusion Matrix"]) == (0,2,0,2)


