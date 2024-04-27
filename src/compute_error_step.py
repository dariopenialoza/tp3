import numpy as np

"""def compute_error_step(x, y, w):
    #print('Nuevo cálculo de error')
    error = 0
    for u in range(x.shape[0]):
        z = 0
        for i in range(x.shape[1]):
            z = z + w[i] * x[u, i] 
            #print(f'z=w*x {z} = {w[i]} * {x[u, i]}')

        y_pred = 1 if z >= 0 else -1      # Activador escalon
        error += (y_pred - y[u])**2
        #print(f'Diff= y_pred - y: {y_pred}-{y[u]} error: {error}')
    #print(f'Error final {error/ len(x)}')
    return error / len(x)
"""


def compute_error_step(x, y, w):
    """Calculates the mean squared error (MSE) using the step activation function.

    Args:
        x (numpy.ndarray): Input data (features) with shape (m, n), where m is the number of samples and n is the number of features.
        y (numpy.ndarray): Target labels (ground truth) with shape (m,), where m is the number of samples.
        w (numpy.ndarray): Weights vector with shape (n,), where n is the number of features.

    Returns:
        float: Mean squared error (MSE).
    """

    # Check input data shapes and types
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(w, np.ndarray):
        raise TypeError("Input arrays must be NumPy arrays.")
    if x.shape[1] != w.shape[0]:
        raise ValueError("Number of features in x must match the dimension of w.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Number of samples in x must match the number of labels in y.")

    # Calculate weighted sums for all samples using vectorization
    z = np.dot(x, w)  # np.dot performs matrix multiplication

    # Apply step activation function to all samples
    y_pred = np.sign(z)  # np.sign returns 1 for positive, -1 for negative, and 0 for zero

    # Calculate squared errors for all samples
    squared_errors = (y_pred - y)**2

    # Mean squared error
    mse = np.mean(squared_errors)

    return mse
