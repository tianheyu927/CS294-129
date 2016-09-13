import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train, dim = X.shape
  num_class = W.shape[1]
  for i in xrange(num_train):
    scores = X[i].dot(W)
    max_score = np.amax(scores)
    scores -= max_score
    scores = np.exp(scores)
    loss += -np.log(scores[y[i]] / np.sum(scores))
    for j in xrange(num_class):
      if j != y[i]:
        dW[:, j] += X[i] * scores[j] / np.sum(scores)
      else:
        dW[:, j] -= X[i] * (np.sum(scores) - scores[y[i]]) / np.sum(scores)
  loss /= num_train
  dW /= num_train
  loss += 0.5*reg*np.sum(W*W)
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  max_scores = np.amax(scores, axis=1)
  scores -= max_scores.reshape(-1, 1)
  scores = np.exp(scores)
  prob = -np.log(scores[np.arange(num_train), y] / np.sum(scores, axis=1))
  loss = np.sum(prob) / num_train + 0.5*reg*np.sum(W*W)
  dscore = np.zeros_like(scores)
  dscore = scores / np.sum(scores, axis=1).reshape(-1, 1)
  dscore[np.arange(num_train), y] -= 1
  dW = X.T.dot(dscore) / num_train + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

