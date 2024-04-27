import numpy as np

def perceptron_simple_step_predictor(x,w):

    y = np.zeros_like(x[:, 0])
    for u in range(len(x)):
        # Calcular la salida bruta
        z=0.0
        for i in range(x.shape[1]):
            z = z + w[i] * x[u, i] 

        # Aplicar la función de activación escalón
        
        y[u] = 1 if z >= 0 else -1 
        #print(f'z={z} u={u} y={y[u]}')
    
    return y
