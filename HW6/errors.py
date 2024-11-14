import numpy as np

class Errors:
    def cross_entropy_error(self, y, t):
        """
        Compute the cross-entropy error (loss) between the predicted output and the target values.

        Parameters:
        y (numpy.ndarray): Predicted output, with shape (batch_size, num_classes) for multiple samples
                           or (num_classes,) for a single sample.
        t (numpy.ndarray): Target (true) values, typically a one-hot encoded vector of the same shape as `y`.
                           Can also be a class index if `y` is 1D.

        Returns:
        float: The cross-entropy error averaged over the batch.

        Notes:
        A small constant `delta` is added to `y` to prevent log(0) and ensure numerical stability.
        """
        delta = 1e-7  # Small value to prevent log(0)
        batch_size = 1 if y.ndim == 1 else y.shape[0]

        return -np.sum(t * np.log(y + delta)) / batch_size
