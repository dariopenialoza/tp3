import sys
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

def perceptron_simple2v(dato_x,dato_y,eta,epoch):
    # Definir los datos de entrenamiento
    #x = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    #y = np.array([-1, -1, -1, 1])
    x = dato_x
    y = dato_y

    error = 0       
    min_error = sys.maxsize 
    w_min = np.array([0, 0])            
    # Inicializar los pesos aleatoriamente
    w1 = np.random.rand()
    w2 = np.random.rand()

    # Ciclo de entrenamiento
    for i in range(len(x)):
        c = 0 
        while (min_error > 0) and (c < epoch):
            # Calcular la salida bruta
            z = w1 * x[i, 0] + w2 * x[i, 1]

            #print(f'z={z} : {w1} * {x[i, 0]} + {w2} * {x[i, 1]}')
            # Aplicar la función de activación escalón
            y_pred = 1 if z >= 0 else -1

            # Actualizar los pesos
            w1_new = w1 + eta * (y[i] - y_pred) * x[i, 0]
            w2_new = w2 + eta * (y[i] - y_pred) * x[i, 1]

            # Actualizar los pesos
            w1 = w1_new
            w2 = w2_new
            
            error = compute_error2v(x, y, w1, w2)
            #print(f'error: {error}')
            if error < min_error:
                min_error = error
                w_min = [w1, w2]
                #print(f'En la corrida {c} del la fila {i} con error={error}')
                #print(f'guarde estos Pesos: w1={w_min[0]} w2={w_min[1]} ')

            c += 1

    # Imprimir los pesos finales
    #print(f'************ Pesos finales: w1={w_min[0]} w2={w_min[1]} ************')

    return w_min