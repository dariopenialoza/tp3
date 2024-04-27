import numpy as np

def compute_error2v(x, y, w1, w2):
    
    error = 0
    y_pred = np.array([0, 0, 0, 0])
    for i in range(len(x)):
        z = w1 * x[i, 0] + w2 * x[i, 1]
        y_pred = 1 if z >= 0 else 0
        error += (y_pred - y[i])**2
        #print(f'w1e={w1} w2e={w2} y_pred: {y_pred}-{y[i]} error: {error}')
    return error / len(x)