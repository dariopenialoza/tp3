import sys
import numpy as np

from compute_error2v import compute_error2v

def perceptron_simple2v(dato_x,dato_y):
    # Definir los datos de entrenamiento
    #x = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    #y = np.array([-1, -1, -1, 1])
    x = dato_x
    y = dato_y

    error = 0       
    min_error = sys.maxsize 
    w_min = np.array([0, 0])    
    epoch = 1000    
    c = 0           # corrida
    # Inicializar los pesos aleatoriamente
    w1 = np.random.rand()
    w2 = np.random.rand()

    # Definir el factor de aprendizaje
    alpha = 0.1

    # Ciclo de entrenamiento
    for i in range(len(x)):
        while (min_error > 0) and (c < epoch):
            # Calcular la salida bruta
            z = w1 * x[i, 0] + w2 * x[i, 1]

            # Aplicar la función de activación escalón
            y_pred = 1 if z >= 0 else 0

            # Actualizar los pesos
            w1_new = w1 + alpha * (y[i] - y_pred) * x[i, 0]
            w2_new = w2 + alpha * (y[i] - y_pred) * x[i, 1]

            # Actualizar los pesos
            w1 = w1_new
            w2 = w2_new

            error = compute_error2v(x, y, w1, w2)
            if error < min_error:
                min_error = error
                w_min = [w1, w2]

            c += 1

    # Imprimir los pesos finales
    print(f'Pesos: w1={w_min[0]} w2={w_min[1]}')

    return w_min