import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
	dW[:,j] += X[i,:].T
	dW[:,y[i]] -= X[i,:].T

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.
  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
  num_classes = W.shape[1]
  num_train = X.shape[0]

  
    #print X.shape      <- (500, 3073)
    #print X[i].shape   <- (3073,)
    #print W.shape      <- (3073,10)
    #print scores.shape <- (10,)

  scores = X.dot(W)    # (500, 10) matrix of scores
  y = y[:,np.newaxis]  # (500,1) treat y like a col vector

  # Create a col vector of the correct score values
  rows = np.arange(num_train)
  rows = rows[:, np.newaxis]
  
  # Index the scores to pull out the correct values
  correctScores = scores[rows, y]
#  print 'correctScores.shape {} '.format(correctScores.shape)
#
#  print scores[:2,]
#  print y[:2]
#  print correctScores[:2,]

  scoresDiff = scores - correctScores
  #print scoresDiff[:2,]
  
  margins = np.maximum(0, scoresDiff + 1)
  margins[rows, y] = 0
  #print margins[:2,]
  #print y[:2,]
  
  # Final loss calculation - add up all margins and normalise by # training
  loss = np.sum(margins) 
  loss /= num_train
  # Don't forget to regularize
  loss += 0.5 * reg * np.sum(W * W)
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
    #print X.shape      <- (500, 3073)
    #print X[i].shape   <- (3073,)
    #print W.shape      <- (3073,10)
    #print scores.shape <- (10,)

  # margins for the correct class have already been set to 0
  margins[margins > 0] = 1
  marginsSum = np.sum(margins, axis=-1) # Correct class already set to 0
  
  # Replace the correct class with the sum of incorrect margins
  margins[rows, y] =  -1 * marginsSum[:,np.newaxis]
  
  #print '--- Vector section ---'
  #print margins[:2,]
  #print marginsSum[:2,]
  #print scores[:2,]
  #print y[:2,]
  
# Now the margins matrix is ready, take the matrix multiply and normalize
  dW = X.T.dot(margins)
  dW /= num_train
  # Now add in regularization
  dW += reg * W
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
