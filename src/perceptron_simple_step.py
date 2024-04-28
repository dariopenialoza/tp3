import numpy as np
from compute_error_step import compute_error_step

def perceptron_simple_step(x, y, eta, epoch):
    # Check input data shapes and types
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Input arrays must be NumPy arrays.")

    if len(x) != len(y):
        raise ValueError("Number of features in x must match the number of labels in y.")

    # Initialize weights randomly
    w = np.random.rand(x.shape[1])
    dw = np.zeros_like(w, dtype=np.float32)
      
   # Initialize variables for tracking minimum error and best weights
    error = 0 
    min_error = np.inf  
    w_min = np.copy(w)
    c = 0
    while (min_error > 0) and (c < epoch):
        u = np.random.randint(0, len(x))
        print(f'u={u}')
        # Calcular la salida bruta
        h = np.dot(x[u], w)
        # Aplicar la función de activación escalón
        o = 1 if h >= 0 else -1
        # Actualizar los pesos
        dw = eta * (y[u] - o) * x[u]
        w = w + dw

        # Calculate error using mean squared error (MSE)
        error = compute_error_step(x,y,w)

        # Update minimum error and best weights if current error is lower
        if error < min_error:
            min_error = error
            w_min = np.copy(w)
            print(f'En la corrida {c} del la fila {u} con error={error}')
            print(f'Guarde estos valores: {w_min}')

        c +=1

    return w_min

