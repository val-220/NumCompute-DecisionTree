from collections import deque

import numpy as np

from numcompute_stream.utils import check_array


def gini(y):
    """
        The proportion of impurity of a node i.e how mixed responses are. 
        e.g a gini impurity of 0.5 would be "half and half".  The lower
        the impurity, the more consensus.

        Parameters: 
        y: The feature which gini impurity is calculated on.

        Returns: a calculated gini impurity

        Time Complexity:
        O(n) where n is the number of samples

        Space Complexity:
        O(n) where n is the number of unique class count
    """
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=2) / len(y)
    return 1.0 - (p ** 2).sum()

class Node:
    """
        Data structure represent nodes in the Decision Tree

        Parameters:
            feature:  the question: X[feature] == 1 ?
            left:   where the "yes" rows go
            right: where the "no" rows go
            label: label of the node.

    """
    def __init__(self, feature=None, left=None, right=None, label=None):
        self.feature = feature            # the question: X[feature] == 1 ?
        self.left = left                  # where the yes rows went
        self.right = right                # where the no rows went
        self.label = label

    """
    """


class DecisionTreeClassifier:
    """
        A decision tree classifier which uses gini for impurity. Based on in-class material and sk-learn documentation.

        Parameters:
            max_depth: maximum depth of the decision tree
            buffer_size: buffer window for the decision tree
            min_samples_split: minimum number of samples required to split an internal node

    """
    def __init__(self, max_depth=2, buffer_size=5, min_samples_split = 2,
    max_features = None):
        """
        Initializing the values, buffer uses a list
        """
        self.root = None
        self.max_depth = max_depth
        self.buffer_size = buffer_size
        self.buffer_X = []
        self.buffer_y = []
        self.min_samples_split = min_samples_split
        self.max_features = max_features

    def _gain(self, y, mask):
        """
            Information gain: impurity before the question minus after it.

            Parameters: 
            y: The samples

            mask: used to calculate the gain given the gini impurity.

            Returns: The numerical gain

        """
        before = gini(y)
        after = (mask.sum() * gini(y[mask])
                 + (~mask).sum() * gini(y[~mask])) / len(y)
        return before - after

    def _best_feature(self, X, y):
        """Ask every feature; keep the one whose gain is largest.

        Parameters:
            X: Features 
            y: Samples

        Returns:
            best_f: The best feature calculated via gain from a maximum
            number of features

        Time Complexity:
            O(m * nlogn) where n is the number of samples and m is the features

        Space Complexity:
            O(n) where n is the n values of a single feature column
        """
        best_f, best_gain = None, 0.0
        for f in range(self.max_features is not None or X.shape[1]):
            mask = X[:, f] == 1
            if mask.all() or not mask.any():    # everyone on one side
                continue
            g = self._gain(y, mask)
            if g > best_gain:
                best_f, best_gain = f, g
        return best_f

    def _build(self, X, y, depth):
        if (len(y) < self.min_samples_split         # Check minimum samples split
                                                    # threshold has been reached or not.
            or depth >= self.max_depth   # deep enough 
            or len(np.unique(y)) == 1):         # everyone agrees
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

        self.buffer_X.append(chunk_X)
        self.buffer_y.append(chunk_y)

        if  np.size(self.buffer_y) >= self.buffer_size:
            fit_chunk_X = np.concatenate(self.buffer_X)
            fit_chunk_y = np.concatenate(self.buffer_y)
            self.fit(fit_chunk_X, fit_chunk_y)
            self.reset()
        return self

    def reset(self):
        """ clears the buffers """
        self.buffer_X.clear()
        self.buffer_y.clear()
        return self


    def fit(self, X, y):
        self.root = self._build(np.asarray(X), np.asarray(y), 0)
        return self

    def predict(self, X):
        out = []
        for x in np.asarray(X):
            node = self.root
            assert node is not None
            while node.label is None:           # still a question
                node = node.left if x[node.feature] == 1 else node.right
            out.append(node.label)
        return np.array(out)