import json
import numpy as np
import pandas as pd
from src.perceptron_multilayer import MultiLayerPerceptron, perceptron_multilayer_predictor

def main3a():
    print('EJERCICIO 3 A')
    
    eta = 0.01
    epoch = 1000
    hidden_size = 2
    
    inputNeuronas = 2
    hiddenNeuronas = 2
    outputNeuronas =1
    
    with open('./config3.json', 'r') as f:
        configData = json.load(f)
        f.close()
    
    print("Función lógica XOR")
    x2 = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y2 = np.array([1, 1, -1, -1])
    
    #asignar despues las funciones en el config
    f1 = '' 
    der_f1 = ''
    epsilon = 0.00001
    mlp = MultiLayerPerceptron(inputNeuronas, hiddenNeuronas, outputNeuronas, epoch, epsilon, f1, der_f1)
    
    # Entreno el perceptron
    mlp.train(x2, y2, eta)
    

if __name__ == "__main__":
    main3a()