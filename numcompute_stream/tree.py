from collections import deque
import numpy as np


def gini(y):
    """
        The proportion of impurity of a node i.e how mixed responses are. 
        e.g a gini impurity of 0.5 would be "half and half".  The lower
        the impurity, the more consensus.

        Parameters: 
        y: The feature which gini impurity is calculated on.
        
    """
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=2) / len(y)
    return 1.0 - (p ** 2).sum()
    

class Node:

    def __init__(self, feature=None, left=None, right=None, label=None):
        self.feature = feature            # the question: X[feature] == 1 ?
        self.left = left                  # where the yes rows went
        self.right = right                # where the no rows went
        self.label = label 



class DecisionTree:
    """
        A decision tree classifier which uses gini for impurity. Based on in-class material.
    """
    def __init__(self, max_depth=2, buffer_size=5):
        """
        Initializing the values, buffer uses a "deque" from Python collections.
        """
        self.max_depth = max_depth
        self.buffer_size = buffer_size
        self.buffer_X = deque(maxlen=buffer_size)
        self.buffer_y = deque(maxlen=buffer_size)
    
    def _gain(self, y, mask):
        """Information gain: impurity before the question minus after it."""
        before = self.impurity(y)
        after = (mask.sum() * self.impurity(y[mask])
                 + (~mask).sum() * self.impurity(y[~mask])) / len(y)
        return before - after

    def _best_feature(self, X, y):
        """Ask every feature; keep the one whose gain is largest."""
        best_f, best_gain = None, 0.0
        for f in range(X.shape[1]):
            mask = X[:, f] == 1
            if mask.all() or not mask.any():    # everyone on one side
                continue
            g = self._gain(y, mask)
            if g > best_gain:
                best_f, best_gain = f, g
        return best_f

    def _build(self, X, y, depth):
        if (depth >= self.max_depth             # deep enough
                or len(np.unique(y)) == 1):     # everyone agrees
            return Node(label=np.bincount(y).argmax())
        f = self._best_feature(X, y)
        if f is None:                           # no question helps
            return Node(label=np.bincount(y).argmax())
        mask = X[:, f] == 1
        return Node(feature=f,
                    left=self._build(X[mask], y[mask], depth + 1),
                    right=self._build(X[~mask], y[~mask], depth + 1))

    def partial_fit(self, chunk_X, chunk_y):
        """
        Implements streaming support for the tree, via buffered retraining.
        - Each call appends the chunk to an internal data buffer
        - Once the buffer holds at least `buffer_size` samples, the tree is retrained on the buffered data and the buffer is cleared

        Parameters:
        chunk_X: Streamed features for the X features 
        chunk_y: Streamed features for the y target/labels
        """
        chunk_X = check_array(chunk_X, dtype=None)
        chunk_y = check_array(chunk_y, dtype=None)
        
        if not len(buffer_X) & len(buffer_y) == buffer.maxlen:
            buffer_X.append(chunk_X)
            buffer_y.append(chunk_y)

        else:
            fit_chunk_X = np.concatenate(buffer_X)
            fit_chunk_y = np.concatenate(buffer_y)
            self.fit(self, fit_chunk_X, fit_chunk_y)

    def reset(self):
        """ clears the buffer """
        buffer_X.clear()
        buffer_y.clear()
        return self


    def fit(self, X, y):
        self.root = self._build(np.asarray(X), np.asarray(y), 0)
        return self

    def predict(self, X):
        out = []
        for x in np.asarray(X):
            node = self.root
            while node.label is None:           # still a question
                node = node.left if x[node.feature] == 1 else node.right
            out.append(node.label)
        return np.array(out)   

    