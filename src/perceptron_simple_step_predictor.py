import numpy as np

def perceptron_simple_step_predictor(x,w):

    o = np.zeros_like(x[:, 0])
    for u in range(len(x)):
        # Calcular la salida bruta
        h = np.dot(w, x[u])

        # Aplicar la función de activación escalón
        o[u] = 1 if h >= 0 else -1
 
    return o
