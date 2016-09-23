import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    
    self.params = {}
    self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b2'] = np.zeros(num_classes)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################

    # Unpack the dictionary of weights and bias
    W1 = self.params['W1']
    b1 = self.params['b1']
    W2 = self.params['W2']
    b2 = self.params['b2']

    # Fancy layer_utils functions combining affine and relu
    (a1, cache1) = affine_relu_forward(X, W1, b1)
    (z2, cache2) = affine_forward(a1, W2, b2)
    scores = z2

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    
    # Compute the loss and output gradients in softmax
    data_loss, dz2 = softmax_loss(scores, y)

    # New fancy layer_utils.py combined functions
    (da1, dW2, db2) = affine_backward(dz2, cache2)
    (dx, dW1, db1) = affine_relu_backward(da1, cache1)

    # Combine data and reg losses ready to return    
    reg_loss = 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
    loss = reg_loss + data_loss

    # Regularize the weights
    dW2 += self.reg * W2
    dW1 += self.reg * W1
    
    # Pack the grads values and return
    grads['W1'] = dW1
    grads['b1'] = db1
    grads['W2'] = dW2
    grads['b2'] = db2    
                
    #print 'data loss = {}'.format(data_loss)
    #print 'grads = {}'.format(grads)        
        
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    
    # Weight dimensions have to be cols of previous stage x rows of next stage

    assert hidden_dims > 0, "Error - expected hidden dimensions, got 0"
    
    dimensions = []
    hidden_idx = 0
    
    while (hidden_idx < len(hidden_dims)):
        # First dimension has to be from input stage to first hidden dim
        if (hidden_idx == 0):
            dimensions.append((input_dim, hidden_dims[0]))
        # Bridge from one hidden dim to another hiden dim size
        else:
            dimensions.append((hidden_dims[hidden_idx-1], hidden_dims[hidden_idx]))

        hidden_idx += 1

    # Last dimension has to be last hidden dim to output classes
    dimensions.append((hidden_dims[len(hidden_dims)-1], num_classes))
    #print 'Dimensions of network are: {}'.format(dimensions)
    
    # Populate the weights and bias values for the stages
    for idx, dim in enumerate(dimensions):
        #print idx, dim
        stage_idx = idx + 1
        weights = 'W' + str(stage_idx)
        bias = 'b' + str(stage_idx)
        if ((idx < self.num_layers-1) and self.use_batchnorm):
            gamma = 'gamma' + str(stage_idx)
            beta = 'beta' + str(stage_idx)
            self.params[gamma] = np.ones(dim[1]) # Need gamma and beta for each col
            self.params[beta] = np.zeros(dim[1])

        
        self.params[weights] = weight_scale * np.random.randn(dim[0], dim[1])
        self.params[bias] = np.zeros(dim[1])
        
                
    #print self.params.keys()
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    
    outputs = []
    inputs = []
    dropout_params = []
    
    #print 'Forward pass'
    
    for layer in xrange(self.num_layers):
        stage = str(layer + 1)
        #print 'Stage is ' + stage

        W = self.params['W' + stage]
        b = self.params['b' + stage]
        if (self.use_batchnorm) and (layer < self.num_layers-1):
            gamma = self.params['gamma' + stage]
            beta = self.params['beta' + stage]
            bn_param = self.bn_params[layer]


        # First stage is a special case, use X as input
        if (layer == 0):
            if (self.use_batchnorm):
                output, input = affine_batchnorm_relu_forward(X, W, b, gamma, beta, bn_param)
            else:
                output, input = affine_relu_forward(X, W, b)
                if self.use_dropout:
                    dropout_output, dropout_param = dropout_forward(output, self.dropout_param)
                    output = dropout_output
                                        
        # Last stage is a special case - no RELU and no batchnorm
        elif (layer == self.num_layers - 1):
            output, input = affine_forward(outputs[layer-1], W, b)
            
        # Hidden layer-to-hidden layer case uses RELU (and batchnorm)
        else:
            if (self.use_batchnorm):
                output, input = affine_batchnorm_relu_forward(outputs[layer-1], W, b, gamma, beta, bn_param)
            else:
                output, input = affine_relu_forward(outputs[layer-1], W, b)
                if self.use_dropout:
                    dropout_output, dropout_param = dropout_forward(output, self.dropout_param)
                    output = dropout_output
               
        outputs.append(output)
        inputs.append(input)
        if self.use_dropout:
            dropout_params.append(dropout_param)
    
    scores = outputs[-1]


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    
    # Compute the loss (combined data and regularization)
    data_loss, grad_out = softmax_loss(scores, y)

    reg_loss = 0
    
    layer_grad = grad_out

    #print 'Backward pass'

    for layer in xrange(self.num_layers, 0, -1):
        stage = str(layer)
        
        w_key = 'W' + stage
        b_key = 'b' + stage
        
        batchnorm_active = (self.use_batchnorm) and (layer < self.num_layers)
                
        W = self.params[w_key]    
        b = self.params[b_key]
        if batchnorm_active:
            gamma_key = 'gamma' + stage
            beta_key = 'beta' + stage            
            gamma = self.params[gamma_key]
            beta = self.params[beta_key]

        reg_loss += 0.5 * self.reg * np.sum(W * W)
        
        # Output stage is just an affine transform, no RELU or batchnorm
        if (layer == self.num_layers):     
            dx, dw, db = affine_backward(layer_grad, inputs[layer-1])
        # Rest of the network is an affine + (batchnorm) + RELU
        else:
            if batchnorm_active:
                dx, dw, db, dgamma, dbeta = affine_batchnorm_relu_backward(layer_grad, inputs[layer-1])
            else:
                if (self.use_dropout):
                    layer_grad = dropout_backward(layer_grad, dropout_params[layer-1])
                dx, dw, db = affine_relu_backward(layer_grad, inputs[layer-1])
                
        
        layer_grad = dx
    
        grads[w_key] = dw
        grads[b_key] = db
        
        if batchnorm_active:
            grads[gamma_key] = dgamma
            grads[beta_key] = dbeta

        
                        
        # Regularize the weights only (not bias)
        grads[w_key] += self.reg * W

    
    loss = data_loss + reg_loss

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


