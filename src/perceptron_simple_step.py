import numpy as np

def compute_error_step(x, y, w):
    # Check input data shapes and types
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(w, np.ndarray):
        raise TypeError("Input arrays must be NumPy arrays.")
    if x.shape[1] != w.shape[0]:
        raise ValueError("Number of features in x must match the dimension of w.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Number of samples in x must match the number of labels in y.")
    """
    error = 0
    for u in range(len(x)):
        # Calculate weighted sums for all samples using vectorization
        h = np.dot(x[u], w)
        o = 1 if h >= 0 else -1
        diff = abs(y[u] - o)
        #print(f'diff={diff}')
        error += diff
    #print(f'error: {error}')
    """
    h = np.dot(x, w)  # Calculate all weighted sums at once
    o = np.where(h >= 0, 1, -1)  # Apply activation function vectorized
    error = np.mean((y - o) ** 2)  # Calculate MSE using vectorized operations
    return error

def perceptron_simple_step_predictor(x1,w):
    # Create a column of 1s
    ones_col = np.ones((x1.shape[0], 1), dtype=x1.dtype)
    # Add the column of 1s to the beginning of the array
    x = np.hstack((ones_col, x1))

    o = np.zeros_like(x[:, 0])
    for u in range(len(x)):
        # Calcular la salida bruta
        h = np.dot(w, x[u])

        # Aplicar la función de activación escalón
        o[u] = 1 if h >= 0 else -1
 
    return o

def perceptron_simple_step(x1, y, eta, epoch):
    # Check input data shapes and types
    if not isinstance(x1, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Input arrays must be NumPy arrays.")

    if len(x1) != len(y):
        raise ValueError("Number of features in x must match the number of labels in y.")

    # Create a column of 1s
    ones_col = np.ones((x1.shape[0], 1), dtype=x1.dtype)
    # Add the column of 1s to the beginning of the array
    x = np.hstack((ones_col, x1))

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
        #print(f'u={u}')
        
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
            print(f'En la corrida {c} del la fila {u} con error={error} garde los pesos')
            print(f'Guarde estos valores: {w_min}')

        c +=1

    return w_min

