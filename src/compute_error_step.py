import numpy as np

def compute_error_step(x, y, w):
    print('Nuevo cálculo de error')
    error = 0
    for u in range(x.shape[0]):
        z = 0
        for i in range(x.shape[1]):
            z = z + w[i] * x[u, i] 
            print(f'z=w*x {z} = {w[i]} * {x[u, i]}')

        y_pred = 1 if z >= 0 else -1      # Activador escalon
        error += (y_pred - y[u])**2
        print(f'Diff= y_pred - y: {y_pred}-{y[u]} error: {error}')
    print(f'Error final {error/ len(x)}')
    return error / len(x)