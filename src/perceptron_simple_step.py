import sys
import numpy as np
import random
from compute_error_step import compute_error_step

def perceptron_simple_step(x, y, eta, epoch):
    # Inicializar los pesos aleatoriamente
    w = np.array([random.random() for i in range(x.shape[1])]) 
    w_new = np.zeros_like(w, dtype=np.float32)

    error = 0       
    min_error = sys.maxsize 
    w_min = np.zeros_like(w, dtype=np.float32)   

    for u in range(len(x)):
        c = 0           # corrida
        while (min_error > 0) and (c < epoch):
            # Calcular la salida bruta
            z=0
            for i in range(x.shape[1]):
                z = z + w[i] * x[u, i] 

            # Aplicar la función de activación escalón
            y_pred = 1 if z >= 0 else -1

            # Actualizar los pesos
            for j in range(x.shape[1]):
                w_new[j] = w[j] + eta * (y[u] - y_pred) * x[u, j]
                #print('Hubo diferencia') if (y[u] - y_pred) != 0 else print('No hubo diferencia')
                #Actualizar los pesos
                w[j] = w_new[j]

            error = compute_error_step(x, y, w)

            if error < min_error:
                min_error = error
                w_min = w.copy()
                print(f'En la corrida {c} del la fila {u} con error={error}')
                print(f'Guarde estos valores: {w_min}')


            c += 1
    
    # Imprimir los pesos finales
    print(f'************ Pesos finales: {w_min} ************')

    return w_min
