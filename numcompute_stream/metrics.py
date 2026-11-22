"""
Evaluates the predicted values against known targets using classification and 
regression metrics.

For classification, this module will evaluate the predicted values using 
performance metrics like:

  - Accuracy, which calculates the proportion of predictions that are correct.

  - Precision, which measures the proportion of predicted positive cases that 
    were actually positive.

  - Recall, which measures the proportion of actual positive cases that the 
    model correctly identified.

  - F1 score, which combines precision and recall to evaluate the model on 
    a range of [0,1].

  - Confusion matrix, which summarises the number of correct and incorrect 
    predictions for each class. 

For regression, this module evaluates the numerical predictions using:
  - The Mean Squared Error (MSE), which evaluates the model by calculating
    the average squared difference between the predicted values and the target
    values. 

Metrics in this module are designed to operate on NumPy arrays and provide 
consistent behaviour during comparison of predicted values and their related
target values. 
"""
import numpy as np
from collections import deque

class StreamMetrics:
  def __init__(self, pred_class = 1):
    self.pred_class = pred_class
    self.y_true_chunks = []
    self.y_pred_chunks = []
    

  def update(self, y_true_chunk, y_pred_chunk, window_size = 200):
    """
      Updates the true and predicted values for metrics calculations in a chunked manner and stores them

      Parameters:
      y_true_chunk: chunked true data
      y_pred_chunk: chunked predicted data
      window_size: Rolling window size

      Returns:
      self, 2 ndarrays of predicted and true masked values as they come in to the function
      
      Time Complexitity:
        O(n) where n is the number of predictions in y_pred and y_true
      Space Complexitity:
        O(n) where n is the number of predictions in y_pred and y_true

    """
    y_true_chunk = np.asarray(y_true_chunk)
    y_pred_chunk = np.asarray(y_pred_chunk)

    # Masking out nans
    y_pred_mask_chunk = ~np.isnan(y_pred_chunk)
    y_pred_chunk = y_pred_chunk[y_pred_mask_chunk]
    y_true_chunk = y_true_chunk[y_pred_mask_chunk]

    y_true_mask_chunk = ~np.isnan(y_true_chunk)
    y_pred_chunk = y_pred_chunk[y_true_mask_chunk]
    y_true_chunk = y_true_chunk[y_true_mask_chunk]


    if y_true_chunk.size == 0:
        return self

    self.y_pred_chunks.append(y_pred_chunk)
    self.y_true_chunks.append(y_true_chunk)

    total = sum(c.size for c in self.y_true_chunks)

    excess = total - window_size
    if excess > 0:
        self.y_true_chunks[0] = self.y_true_chunks[0][excess:]
        self.y_pred_chunks[0] = self.y_pred_chunks[0][excess:]

    return self

  def result(self):
    """
      Concatenates the true and predicted values for metrics calculations, and calls each metric function

      Parameters:
      self

      Returns:
      A set containing the results of every metric function called.

      Time Complexitity:
        O(n) where n is the number of predictions in y_pred and y_true
      Space Complexitity:
        O(n) where n is the number of predictions in y_pred and y_true

    """
    y_true_chunk = np.concatenate(self.y_true_chunks)
    y_pred_chunk = np.concatenate(self.y_pred_chunks)

    return {
    "Accuracy": self.accuracy(y_pred_chunk, y_true_chunk),
    "Precision": self.precision(y_pred_chunk, y_true_chunk, self.pred_class),
    "Recall": self.recall(y_pred_chunk, y_true_chunk, self.pred_class),
    "F1 Score": self.f1score(y_pred_chunk, y_true_chunk, self.pred_class),
    "Confusion Matrix": self.confusion_matrix(y_pred_chunk, y_true_chunk, self.pred_class) ,
    "MSE": self.MSE(y_pred_chunk, y_true_chunk)
    }

  def reset(self):
    """
      Resets and preserves hyperparameters, needed if working on a new dataset

      Time Complexity:
         O(1) for deleting everything

      Space Complexity:
         O(1).
    """
    self.y_true_chunks = []
    self.y_pred_chunks = []
  
  def accuracy(self, y_pred, y_true):
    """
      Determines the accuracy, percentage of correctly predicted labels
      Parameters:
        y_pred: the predicted labels in the form of a 1D numpy array
        y_true: the true labels in the form of a 1D numpy array
      Returns:
        ndarray or scalar. If inputs are correct it will be a scalar
      Time Complexitity:
        O(n) where n is the number of predictions in y_pred and y_true
      Space Complexitity:
        O(n) where n is the number of predictions in y_pred and y_true

    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    return np.divide(np.sum(np.where(y_pred ==y_true,1,0)),y_true.size)
    

  def precision(self, y_pred, y_true, pred_class=1):
    """
      Determines the precision, the number of correctly predicted true labels divied by total number of predicted true labels
      Parameters:
        y_pred: the predicted labels in the form of a 1D numpy array
        y_true: the true labels in the form of a 1D numpy array
        pred_class: The class to determine to precision of for multiclass classification
      Returns:
        ndarray or scalar. If inputs are correct it will be a scalar
      Time Complexitity:
        O(n) where n is the number of predictions in y_pred and y_true
      Space Complexitity:
        O(n) where n is the number of predictions in y_pred and y_true
    """
    #Masking out nans
    y_pred_mask= ~np.isnan(y_pred)
    y_pred= y_pred[y_pred_mask]
    y_true = y_true[y_pred_mask]

    y_true_mask = ~np.isnan(y_true)
    y_pred = y_pred[y_true_mask]
    y_true= y_true[y_true_mask]

    pos_index = y_pred == pred_class
    return np.divide(np.sum(y_pred[pos_index] == y_true[pos_index]), y_pred[pos_index].size)


  def recall(self, y_pred, y_true, pred_class=1):
    """
      Determines to recall, the number of correctly predicted positive labels divied by the number of true positive labels
      Parameters:
        y_pred: the predicted labels in the form of a 1D numpy array
        y_true: the true labels in the form of a 1D numpy array
        pred_class: The class to determine to precision of for multiclass classification
      Returns:
        ndarray or scalar. If inputs are correct it will be a scalar
      Raises:

      Time Complexitity:
        O(n) where n is the number of predictions in y_pred and y_true
      Space Complexitity:
        O(n) where n is the number of predictions in y_pred and y_true
    """
    #Masking out nans
    y_pred_mask= ~np.isnan(y_pred)
    y_pred= y_pred[y_pred_mask]
    y_true = y_true[y_pred_mask]

    y_true_mask = ~np.isnan(y_true)
    y_pred = y_pred[y_true_mask]
    y_true= y_true[y_true_mask]

    pos_index = y_true == pred_class
    return np.divide(np.sum(y_pred[pos_index] == y_true[pos_index]), y_true[pos_index].size)  


  def f1score(self, y_pred, y_true, pred_class=1):
    """
    F1 score, which combines precision and recall to evaluate the model on 
      a range of [0,1].
      Parameters:
        y_pred: the predicted labels in the form of a 1D numpy array
        y_true: the true labels in the form of a 1D numpy array
        pred_class: The class to determine to precision of for multiclass classification
      Returns:
        ndarray or scalar. If inputs are correct it will be a scalar
      Raises:

      Time Complexitity:
        O(n) where n is the number of predictions in y_pred and y_true
      Space Complexitity:
        O(n) where n is the number of predictions in y_pred and y_true
    """

    #Masking out nans
    y_pred_mask= ~np.isnan(y_pred)
    y_pred= y_pred[y_pred_mask]
    y_true = y_true[y_pred_mask]

    y_true_mask = ~np.isnan(y_true)
    y_pred = y_pred[y_true_mask]
    y_true= y_true[y_true_mask]

    prec = self.precision(y_pred,y_true,pred_class)
    reca = self.recall(y_pred,y_true,pred_class)

    return 2* ((prec*reca)/(prec+reca)) if prec + reca > 0 else 0



  def confusion_matrix(self, y_pred, y_true, pred_class=1):
    """
      Confusion matrix, determines the confusion matrix which is the number of true/false pos/neg results
      Parameters:
        y_pred: the predicted labels in the form of a 1D numpy array
        y_true: the true labels in the form of a 1D numpy array
        pred_class: The class to determine to precision of for multiclass classification
      Returns:
        truple of sclars in the order true positive, false positive, true negative, false negative

      Raises:

      Time Complexity:
        O(n) where n is the number of predictions in y_pred and y_true
      Space Complexity:
        O(n) where n is the number of predictions in y_pred and y_true
    """

    #Masking out nans
    y_pred_mask= ~np.isnan(y_pred)
    y_pred= y_pred[y_pred_mask]
    y_true = y_true[y_pred_mask]

    y_true_mask = ~np.isnan(y_true)
    y_pred = y_pred[y_true_mask]
    y_true= y_true[y_true_mask]

    pred_pos = y_pred == pred_class
    pred_neg = y_pred != pred_class

    true_pos = np.count_nonzero(y_pred[pred_pos] == y_true[pred_pos])
    true_neg = np.count_nonzero(y_pred[pred_neg] == y_true[pred_neg])
    false_pos = np.count_nonzero(y_pred[pred_pos] != y_true[pred_pos])
    false_neg = np.count_nonzero(y_pred[pred_neg] != y_true[pred_neg])
    return (true_pos,false_pos,true_neg,false_neg)

  def MSE(self, y_pred_chunk, y_true_chunk):
    """
    The Mean Squared Error (MSE), which evaluates the model by calculating
      the average squared difference between the predicted values and the target
      values. 
      Parameters:
        y_pred: the predicted labels in the form of a 1D numpy array
        y_true: the true labels in the form of a 1D numpy array
      Returns:
        ndarray or scalar. If inputs are correct it will be a scalar
      Raises:

      Time Complexitity:
        O(n) where n is the number of predictions in y_pred and y_true
      Space Complexitity:
        O(n) where n is the number of predictions in y_pred and y_true
    """

    # Masking out nans
    y_pred_mask_chunk = ~np.isnan(y_pred_chunk)
    y_pred_chunk = y_pred_chunk[y_pred_mask_chunk]
    y_true_chunk = y_true_chunk[y_pred_mask_chunk]

    y_true_mask_chunk = ~np.isnan(y_true_chunk)
    y_pred_chunk = y_pred_chunk[y_true_mask_chunk]
    y_true_chunk = y_true_chunk[y_true_mask_chunk]

    return np.divide(np.sum(np.power(y_pred_chunk-y_true_chunk,2)),y_true_chunk.size)


