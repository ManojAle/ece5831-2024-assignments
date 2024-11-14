import numpy as np
from activations import Activations
from errors import Errors
from collections import OrderedDict

class Relu:
    """
    Implements the ReLU (Rectified Linear Unit) activation function.
    """
    def __init__(self):
        self.mask = None

    def forward(self, x):
        """
        Forward pass for the ReLU function.
        
        Parameters:
        x (numpy.ndarray): Input data.

        Returns:
        numpy.ndarray: Output with ReLU applied, where negative values are set to zero.
        """
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        """
        Backward pass for the ReLU function.
        
        Parameters:
        dout (numpy.ndarray): Gradient of the loss with respect to the output of this layer.

        Returns:
        numpy.ndarray: Gradient of the loss with respect to the input of this layer.
        """
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid:
    """
    Implements the Sigmoid activation function.
    """
    def __init__(self):
        self.out = None
        self.activations = Activations()

    def forward(self, x):
        """
        Forward pass for the Sigmoid function.
        
        Parameters:
        x (numpy.ndarray): Input data.

        Returns:
        numpy.ndarray: Output after applying Sigmoid function.
        """
        out = self.activations.sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        """
        Backward pass for the Sigmoid function.
        
        Parameters:
        dout (numpy.ndarray): Gradient of the loss with respect to the output of this layer.

        Returns:
        numpy.ndarray: Gradient of the loss with respect to the input of this layer.
        """
        dx = dout * (1.0 - self.out) * self.out
        return dx


class Affine:
    """
    Implements the Affine (fully connected) layer.
    """
    def __init__(self, w, b):
        """
        Parameters:
        w (numpy.ndarray): Weight matrix.
        b (numpy.ndarray): Bias vector.
        """
        self.w = w
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        Forward pass for the Affine layer.
        
        Parameters:
        x (numpy.ndarray): Input data.

        Returns:
        numpy.ndarray: Output after applying affine transformation.
        """
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)  # Flatten input if needed
        self.x = x
        out = np.dot(self.x, self.w) + self.b
        return out

    def backward(self, dout):
        """
        Backward pass for the Affine layer.
        
        Parameters:
        dout (numpy.ndarray): Gradient of the loss with respect to the output of this layer.

        Returns:
        numpy.ndarray: Gradient of the loss with respect to the input of this layer.
        """
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)  # Reshape to original input shape
        return dx


class SoftmaxWithLoss:
    """
    Combines Softmax activation with Cross-Entropy loss.
    """
    def __init__(self):
        self.loss = None
        self.y_hat = None    # Output of softmax
        self.y = None        # True labels
        self.activations = Activations()
        self.errors = Errors()

    def forward(self, x, y):
        """
        Forward pass to compute the loss using Softmax and Cross-Entropy.
        
        Parameters:
        x (numpy.ndarray): Input data (logits).
        y (numpy.ndarray): True labels (one-hot encoded or label indices).

        Returns:
        float: Cross-Entropy loss.
        """
        self.y = y
        self.y_hat = self.activations.softmax(x)
        self.loss = self.errors.cross_entropy_error(self.y_hat, self.y)
        return self.loss

    def backward(self, dout=1):
        """
        Backward pass to compute the gradient of the loss with respect to the input.
        
        Parameters:
        dout (float, optional): Upstream gradient. Defaults to 1.

        Returns:
        numpy.ndarray: Gradient of the loss with respect to the input `x`.
        """
        batch_size = self.y.shape[0]
        dx = (self.y_hat - self.y) / batch_size
        return dx
