import logging
import tracemalloc

from numcompute_stream import StreamMetrics


class StreamTrainer:
    """
        Trains and scores a model based on its metrics, via streaming.
    """
    def __init__(self, model, pipeline):
        self.model = model
        self.pipeline = pipeline          # list of (name, transformer) or None
        self.metrics =  StreamMetrics()
        self.logger = logging.getLogger(__name__)
        self.history = []                 # list of per-chunk dicts
        self.cumulative_accuracy = 0                 # for cumulative accuracy


    def fit_chunk(self, X, y):
        """
        Transforms a chunk by feeding it into the pipeline,
        then, updates the model via a partial_fit.
        Arguments:
            X : The features that are inputed into the data
            y : Labels/ target variables of those features.

        Returns: Nothing

        Raises: 

        Time Complexity: 

        Space Complexity: 
        """
        tracemalloc.start()
        Xt = self.pipeline.partial_fit_transform(X)
        self.model.partial_fit(Xt, y)
        tracemalloc.stop()
        self.history.append(tracemalloc.get_traced_memory)



    def score_chunk(self, X,y):
        Xt = self.pipeline.transform(X)
        y_pred = self.model.predict(Xt)

        metrics = StreamMetrics()

        self.cumulative_accuracy += metrics.update(y, y_pred).accuracy(y, y_pred)
        self.logger.info(metrics.update(y, y_pred).result())
