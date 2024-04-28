import numpy as np

def compute_error_step(x, y, w):
    """Calculates the mean squared error (MSE) using the step activation function.
    Args:
        x (numpy.ndarray): Input data (features) with shape (m, n), where m is the number of samples and n is the number of features.
        y (numpy.ndarray): Target labels (ground truth) with shape (m,), where m is the number of samples.
        w (numpy.ndarray): Weights vector with shape (n,), where n is the number of features.
    """

    # Check input data shapes and types
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(w, np.ndarray):
        raise TypeError("Input arrays must be NumPy arrays.")
    if x.shape[1] != w.shape[0]:
        raise ValueError("Number of features in x must match the dimension of w.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Number of samples in x must match the number of labels in y.")

    error = 0
    for u in range(len(x)):
        # Calculate weighted sums for all samples using vectorization
        h = np.dot(x[u], w)
        o = 1 if h >= 0 else -1
        error += (y[u] - o)**2
        print(f'error: {error}')
    print(f'error medio: {error/len(x)}')
    return error / len(x)
