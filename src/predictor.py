import numpy as np

def predictor(x, w):
    """Perceptron with simple step activation function and vectorized operations.

    Args:
        x (numpy.ndarray): Input data (features) with shape (m, n), where m is the number of samples and n is the number of features.
        w (numpy.ndarray): Weights vector with shape (n,), where n is the number of features.

    Returns:
        numpy.ndarray: Predicted labels (y) with shape (m,), where m is the number of samples.
    """

    # Create a 'y' array with the same number of rows as 'x'
    y = np.zeros_like(x[:, 0])  # Use x[:, 0] to get the first column of 'x' for array size

    # Calculate weighted sums for all samples using vectorization
    z = np.dot(x, w)  # np.dot performs matrix multiplication

    # Apply step activation function to all samples
    y = np.sign(z)  # np.sign returns 1 for positive, -1 for negative, and 0 for zero

    return y
