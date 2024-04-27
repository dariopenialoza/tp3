import numpy as np

def compute_error_step(x, y, w):

    error = 0
    for u in range(x.shape[0]):
        z = 0
        for i in range(x.shape[1]):
            z = z + w[i] * x[u, i] 
        y_pred = 1 if z >= 0 else 0      # Activador escalon
        error += (y_pred - y[u])**2

    return error / len(x)