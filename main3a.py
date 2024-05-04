import json
import numpy as np
import pandas as pd
from src.perceptron_multilayer import MultiLayerPerceptron

def main3a():
    print('EJERCICIO 3 A')
    
    with open('./config3a.json', 'r') as f:
        configData = json.load(f)
        f.close()
        
    epoch = configData["epoch"]
    eta = configData["eta"]
    epsilon = configData["epsilon"]
    f1 = configData["f1"]
    
    hidden_size = 2
    
    # numero de neuronas en capa oculta.
    hiddenNeuronas = [1]
    # numero de neuronas en capa salida
    outputNeuronas =1
    
    print("Función lógica XOR")
    x2 = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y2 = np.array([1, 1, -1, -1])
    
    inputNeuronas = x2.shape[1]
    mlp = MultiLayerPerceptron(x2, y2, eta, inputNeuronas, hiddenNeuronas, outputNeuronas, epoch, epsilon, f1)
    
    # Entreno el perceptron
    errores = mlp.train()
    
    
    i = 0
    while i < len(y2):
        print("input: ", x2[i])
        print("expected: ", y2[i])
        print("prediction: ", mlp.predict(x2[i]))
        i += 1


if __name__ == "__main__":
    main3a()