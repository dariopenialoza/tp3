import numpy as np

def compute_error_lineal(x, y, w):
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
        o = h
        error += (y[u] - o)**2
        #print(f'error: {error}')
    #print(f'error medio: {error/len(x)}')
    return error / len(x)

def perceptron_simple_lineal_predictor(x,w):
    o = np.zeros_like(x[:, 0])
    for u in range(len(x)):
        # Calcular la salida bruta
        h = np.dot(w, x[u])

        # Aplicar la función de activación escalón
        o[u] = h
 
    return o

def perceptron_simple_lineal(x, y, eta, epsilon, epoch):
    # Check input data shapes and types
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Input arrays must be NumPy arrays.")

    if len(x) != len(y):
        raise ValueError("Number of features in x must match the number of labels in y.")

    # Initialize weights randomly
    w = np.random.rand(x.shape[1])
    dw = np.zeros_like(w, dtype=np.float32)
    
    #print(f'cantidad de filas={len(x)} ')  
    # Initialize variables for tracking minimum error and best weights
    error = 0 
    min_error = np.inf  
    w_min = np.copy(w)
    c = 0
    while (min_error > epsilon) and (c < epoch):
        u = np.random.randint(0, len(x))
        #print(f'iteracion={c} u={u}')
        
        # Calcular la salida bruta
        h = np.dot(x[u], w)
        # Aplicar la función de activación escalón
        o = h
        # Actualizar los pesos
        dw = eta * (y[u] - o) * h * x[u]
        w = w + dw

        # Calculate error using mean squared error (MSE)
        error = compute_error_lineal(x,y,w)

        # Update minimum error and best weights if current error is lower
        if error < min_error:
            min_error = error
            w_min = np.copy(w)
            #print(f'En la corrida {c} del la fila {u} con error={error}')
            #print(f'Guarde estos valores: {w_min}')

        c +=1

    return w_min

