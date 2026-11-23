import logging

import numpy as np

from numcompute_stream import StreamMetrics

from numcompute_stream import Pipeline


class StreamTrainer:
    """
        Trains and scores a model based on its metrics, via streaming.
    """

    def __init__(self, model, pipeline):
        self.model = model
        self.pipeline = pipeline  # Pipeline object
        self.metrics = StreamMetrics()
        self.logger = logging.getLogger(__name__)
        self.cumulative_accuracy = 0  # for cumulative accuracy

    def fit_chunk(self, X, y):
        """
        Transforms a chunk by feeding it into the pipeline,
        then, updates the model via a partial_fit.
        Arguments:
            X : The features that are inputed into the data
            y : Labels/ target variables of those features.

        Returns: Fitted model to the inputted chunk

        Raises:
            ValueError: If chunks are empty or of an incorrect value

        Time Complexity: 

        Space Complexity: 
        """

        if X is not None and y is not None:
            raise ValueError("Chunks are empty!")
        x, y = np.array(X), np.array(y)

        Xt = x if self.pipeline is None else self.pipeline.partial_fit_transform(X)
        self.model.partial_fit(Xt, y)

        return self.model

    def score_chunk(self, X, y):
        x, y = np.array(X), np.array(y)

        Xt = x if self.pipeline is None else self.pipeline.transform(X)
        y_pred = self.model.predict(Xt)

        self.metrics.update(y, y_pred)  # once
        result = self.metrics.result()  # once

        self.cumulative_accuracy = result["Accuracy"]
        self.logger.info(result)
        return result
