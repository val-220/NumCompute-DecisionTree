import logging

import numpy as np

from numcompute_stream import StreamMetrics


class StreamTrainer:
    def __init__(self, model, pipeline, metrics=None, logger=None, track_memory=True):
        self.model = model
        self.pipeline = pipeline          # list of (name, transformer) or None
        self.metrics = metrics or StreamMetrics()
        self.logger = logging.getLogger(__name__)
        self.history = []                 # list of per-chunk dicts
        self.n_chunks_ = 0
        self.n_seen_ = 0
        self.cumulative_accuracy = 0                 # for cumulative accuracy


def fit_chunk(self, X, y):
    Xt = self.pipeline.partial_fit_transform(X)

    return self.model.partial_fit(Xt, y)


def score_chunk(self, X,y):
    Xt = self.pipeline.transform(X)
    y_pred = self.model.predict(Xt)

    metrics = StreamMetrics()

    self.cumulative_accuracy += metrics.update(y, y_pred).accuracy(y, y_pred)
    self.logger.into(metrics.update(y, y_pred).result())
