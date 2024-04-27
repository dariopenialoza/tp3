import numpy as np

def compute_error2v(x, y, w1, w2):
    #print('Nuevo cálculo de error')
    error = 0
    y_pred = np.array([0, 0, 0, 0])
    for i in range(len(x)):
        z = w1 * x[i, 0] + w2 * x[i, 1]
        #print(f'z=w*x {z} = {w1} * {x[i,0]} + {w2} * {x[i,1]}')
        y_pred = 1 if z >= 0 else -1
        error += (y_pred - y[i])**2
        #print(f'Diff= y_pred - y: {y_pred}-{y[i]} error: {error}')
    #print(f'Error final {error/ len(x)}')
    return error / len(x)