import csv
import random
import numpy as np

def norm_to_im(x,y,a,b):
    y_min = np.min(y)
    y_max = np.max(y)
    y = ((y -y_min)/(y_max - y_min)) * (b-a)+a
    x = ((x -y_min)/(y_max - y_min)) * (b-a)+a
    return x, y

def perceptron_simple_nolineal_predictor(x1, w, beta):
    # Create a column of 1s
    ones_col = np.ones((x1.shape[0], 1), dtype=x1.dtype)
    # Add the column of 1s to the beginning of the array
    x = np.hstack((ones_col, x1))

    o = np.zeros_like(x[:, 0])
    for u in range(len(x)):
        # Calcular la salida bruta
        h = np.dot(w, x[u])
        # Aplicar la función de activación lineal (identity function)
        o[u] = np.tanh(beta * h)
    return o

def compute_error_nolineal(x, y, w, beta):
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
    o = np.tanh(beta * h)
    error = np.mean((y - o) ** 2)  # Calculate MSE using vectorized operations
    return error

def perceptron_simple_nolineal(x0, y0, beta, learning_rate, epsilon, epoch):
    # Check input data shapes and types
    if not isinstance(x0, np.ndarray) or not isinstance(y0, np.ndarray):
        raise TypeError("Input arrays must be NumPy arrays.")
    if len(x0) != len(y0):
        raise ValueError("Number of features in x must match the number of labels in y.")
    
    x1, y = norm_to_im(x0, y0, -1, 1)

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
    w_min = np.copy(dw)
    c = 0
    error_por_c = []
    while (min_error > epsilon) and (c < epoch):
        for u in range(len(x)):
            # Calcular la salida bruta
            h = np.dot(x[u], w)
            # Aplicar la función de activación lineal (identity function)
            o = np.tanh(beta * h)
            # Actualizar los pesos (using higher precision float type in calculations)
            dw =learning_rate * (y[u] - o) * beta * (1 - ((np.tanh(beta*h)) ** 2) ) * x[u]     # dw =learning_rate * (y[u] - o) * o' *x[u]  # 0'=1
            w = w + dw
            # Calculate error using mean squared error (MSE)
            error = compute_error_nolineal(x,y,w,beta)
            # Update minimum error and best weights if current error is lower
            if error < min_error:
                min_error = error
                w_min = np.copy(w)
                #print(f'En la corrida {c} del la fila {u} con error={error}')
                #print(f'Guarde estos valores: {w_min}')
        error_por_c.append([c,error]) 
        c +=1
    with open(f'perceptron_simple_NOlineal_error_c-{epoch}-{learning_rate}-{beta}.csv', 'w', newline='') as archivo:
        escritor_csv = csv.writer(archivo)
        escritor_csv.writerows(error_por_c)
    return w_min, min_error

def perceptron_simple_nolineal_u(x0, y0, beta, learning_rate, epsilon, epoch):
    # Check input data shapes and types
    if not isinstance(x0, np.ndarray) or not isinstance(y0, np.ndarray):
        raise TypeError("Input arrays must be NumPy arrays.")
    if len(x0) != len(y0):
        raise ValueError("Number of features in x must match the number of labels in y.")
    
    x1, y = norm_to_im(x0, y0, -1, 1)

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
    w_min = np.copy(dw)
    c = 0
    while (min_error > epsilon) and (c < epoch):
        u = np.random.randint(0, len(x))
        # Calcular la salida bruta
        h = np.dot(x[u], w)
        # Aplicar la función de activación lineal (identity function)
        o = np.tanh(beta * h)
        # Actualizar los pesos (using higher precision float type in calculations)
        dw =learning_rate * (y[u] - o) * beta * (1 - ((np.tanh(beta*h)) ** 2) ) * x[u]     # dw =learning_rate * (y[u] - o) * o' *x[u]  # 0'=1
        w = w + dw
        # Calculate error using mean squared error (MSE)
        error = compute_error_nolineal(x,y,w,beta)
        # Update minimum error and best weights if current error is lower
        if error < min_error:
            min_error = error
            w_min = np.copy(w)
            #print(f'En la corrida {c} del la fila {u} con error={error}')
            #print(f'Guarde estos valores: {w_min}')
        c +=1
    return w_min, min_error

def getTrainingSet(x,y,k_perc):
    # Crear una lista de índices aleatorios
    indices_aleatorios = random.sample(range(len(x)), len(x))

    # Reordenar los vectores utilizando los índices aleatorios
    x_reordenado = [x[i] for i in indices_aleatorios]
    y_reordenado = [y[i] for i in indices_aleatorios]

    k = int(len(x) * k_perc)
    # Dividir los arrays desordenados
    x_training = x_reordenado[:k]
    x_test = x_reordenado[k:]
    y_training = y_reordenado[:k]
    y_test = y_reordenado[k:]

    return x_training, x_test, y_training, y_test

def perceptron_simple_nolineal_k(x0, y0, beta, learning_rate, epsilon, epoch, k_perc):
    # Check input data shapes and types
    if not isinstance(x0, np.ndarray) or not isinstance(y0, np.ndarray):
        raise TypeError("Input arrays must be NumPy arrays.")
    if len(x0) != len(y0):
        raise ValueError("Number of features in x must match the number of labels in y.")
    
    x1, y = norm_to_im(x0, y0, -1, 1)

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
    w_min = np.copy(dw)
    c = 0

    sub_x, x_test, sub_y, y_test= getTrainingSet(x, y, k_perc)

    while (min_error > epsilon) and (c < epoch):
        for u in range(len(sub_x[0])):
            # Calcular la salida bruta
            h = np.dot(sub_x[0][u], w)
            # Aplicar la función de activación lineal (identity function)
            o = np.tanh(beta * h)
            # Actualizar los pesos (using higher precision float type in calculations)
            dw = learning_rate * (sub_y[0][u] - o) * beta * (1 - ((np.tanh(beta*h)) ** 2) ) * sub_x[0][u]     # dw =learning_rate * (y[u] - o) * o' *x[u]  # 0'=1
            w = w + dw
            # Calculate error using mean squared error (MSE)
            error = compute_error_nolineal(sub_x[0],sub_y[0],w,beta)
            # Update minimum error and best weights if current error is lower
            if error < min_error:
                min_error = error
                w_min = np.copy(w)
                #print(f'En la corrida {c} del la fila {u} con error={error}')
                #print(f'Guarde estos valores: {w_min}')
            c +=1
    return w_min, min_error
