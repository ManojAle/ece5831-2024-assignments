import numpy as np

class Activations:
    def sigmoid(self, x):
        """
        Compute the sigmoid activation function for a given input.

        Parameters:
        x (numpy.ndarray): Input array or scalar for which to compute the sigmoid.

        Returns:
        numpy.ndarray: Sigmoid of the input, with the same shape as `x`.
        """
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """
        Compute the softmax activation function for a given input.

        Parameters:
        x (numpy.ndarray): Input array, can be 1-dimensional or 2-dimensional.
                           For 2D arrays, the softmax is computed along the columns.

        Returns:
        numpy.ndarray: Softmax of the input, with the same shape as `x`.
        For 2D input, each column sums to 1.
        """
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)  # For numerical stability
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T

        x = x - np.max(x)  # For numerical stability
        return np.exp(x) / np.sum(np.exp(x))
