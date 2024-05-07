import random
import numpy as np

def perceptron_simple_nolineal_predictor(x1, w):
    # Create a column of 1s
    ones_col = np.ones((x1.shape[0], 1), dtype=x1.dtype)
    # Add the column of 1s to the beginning of the array
    x = np.hstack((ones_col, x1))

    o = np.zeros_like(x[:, 0])
    for u in range(len(x)):
        # Calcular la salida bruta
        h = np.dot(w, x[u])
        # Aplicar la función de activación lineal (identity function)
        o[u] = 1 / (1 + np.exp(-2 * h))
    return o

def compute_error_nolineal(x, y, w):
    # Check input data shapes and types
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(w, np.ndarray):
        raise TypeError("Input arrays must be NumPy arrays.")
    if x.shape[1] != w.shape[0]:
        raise ValueError("Number of features in x must match the dimension of w.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Number of samples in x must match the number of labels in y.")
    """error = 0.0
    for u in range(len(x)):
        # Calculate weighted sums for all samples using vectorization
        h = np.dot(x[u], w)
        o = h
        diff = abs(y[u] - o)  # Calculate absolute difference
        error += diff"""
    h = np.dot(x, w)  # Calculate all weighted sums at once
    o = 1 / (1 + np.exp(-2 * h))
    error = np.mean((y - o) ** 2)  # Calculate MSE using vectorized operations
    return error

def perceptron_simple_nolineal(x1, y, eta, epsilon, epoch):
    # Check input data shapes and types
    if not isinstance(x1, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Input arrays must be NumPy arrays.")
    if len(x1) != len(y):
        raise ValueError("Number of features in x must match the number of labels in y.")
    
    # Create a column of 1s
    ones_col = np.ones((x1.shape[0], 1), dtype=x1.dtype)
    # Add the column of 1s to the beginning of the array
    x = np.hstack((ones_col, x1))
    #print(x[:5, :])

    # Initialize weights randomly using higher precision float type (float64)
    w = np.random.rand(x.shape[1]).astype(np.float64)
    dw = np.zeros_like(w, dtype=np.float64)
    # Initialize variables for tracking minimum error and best weights
    error = 0.0
    min_error = np.inf
    w_min = np.copy(w)
    c = 0
    while (min_error > epsilon) and (c < epoch):
        u = np.random.randint(0, len(x))
        # Calcular la salida bruta
        h = np.dot(x[u], w)
        # Aplicar la función de activación lineal (identity function)
        o = 1 / (1 + np.exp(-2 * h))
        # Actualizar los pesos (using higher precision float type in calculations)
        diff = abs(y[u] - o)
        dw = eta * diff * (2 * np.exp(-2 * h)) / ((1 + np.exp(-2 * h)) **2) * x[u]     # dw = eta * (y[u] - o) * o' *x[u]  # 0'=1
        w = w + dw
        # Calculate error using mean squared error (MSE)
        error = compute_error_nolineal(x,y,w)
        # Update minimum error and best weights if current error is lower
        if error < min_error:
            min_error = error
            w_min = np.copy(w)
            #print(f'En la corrida {c} del la fila {u} con error={error}')
            #print(f'Guarde estos valores: {w_min}')
        c +=1
    return w_min

def getTrainingSet(x,y,k):
    # Crear una lista de índices aleatorios
    indices_aleatorios = random.sample(range(len(x)), len(x))

    # Reordenar los vectores utilizando los índices aleatorios
    x_reordenado = [x[i] for i in indices_aleatorios]
    y_reordenado = [y[i] for i in indices_aleatorios]
    #print("vector 1:", x_reordenado)
    #print("ector 2:", y_reordenado)
    
    # Dividir el conjunto ordenado en 4 subconjuntos
    subconjuntos = np.array_split(x_reordenado, 2)
    subconjuntos_salida = np.array_split(y_reordenado, 2)
    #print(subconjuntos)
    #print(subconjuntos_salida)
    return subconjuntos, subconjuntos_salida

def perceptron_simple_nolineal_k(x1, y, eta, epsilon, epoch):
    # Check input data shapes and types
    if not isinstance(x1, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Input arrays must be NumPy arrays.")
    if len(x1) != len(y):
        raise ValueError("Number of features in x must match the number of labels in y.")
    
    # Create a column of 1s
    ones_col = np.ones((x1.shape[0], 1), dtype=x1.dtype)
    # Add the column of 1s to the beginning of the array
    x = np.hstack((ones_col, x1))
    #print(x[:5, :])

    # Initialize weights randomly using higher precision float type (float64)
    w = np.random.rand(x.shape[1]).astype(np.float64)
    dw = np.zeros_like(w, dtype=np.float64)
    # Initialize variables for tracking minimum error and best weights
    error = 0.0
    min_error = np.inf
    w_min = np.copy(w)
    c = 0

    sub_x, sub_y = getTrainingSet(x, y, 2)

    while (min_error > epsilon) and (c < epoch):
        for u in range(len(sub_x)):
            # Calcular la salida bruta
            h = np.dot(sub_x[0][u], w)
            # Aplicar la función de activación lineal (identity function)
            o = 1 / (1 + np.exp(-2 * h))
            # Actualizar los pesos (using higher precision float type in calculations)
            diff = abs(y[u] - o)
            dw = eta * diff * (2 * np.exp(-2 * h)) / ((1 + np.exp(-2 * h)) **2) * x[u]     # dw = eta * (y[u] - o) * o' *x[u]  # 0'=1
            w = w + dw
            # Calculate error using mean squared error (MSE)
            error = compute_error_nolineal(sub_x[0],sub_y[0],w)
            # Update minimum error and best weights if current error is lower
            if error < min_error:
                min_error = error
                w_min = np.copy(w)
                #print(f'En la corrida {c} del la fila {u} con error={error}')
                #print(f'Guarde estos valores: {w_min}')
            c +=1
    return w_min
