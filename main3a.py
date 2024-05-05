import json
import numpy as np
import pandas as pd
from src.perceptron_multilayer import MultiLayerPerceptron

def main3a():
    print('EJERCICIO 3 A')
    
    with open('./config3a.json', 'r') as f:
        configData = json.load(f)
        f.close()
        
    epochs = configData["epochs"]
    epsilon = configData["epsilon"]
    f1 = configData["f1"]
    learningRate = configData["learningRate"]
    hiddenLayers = configData["hiddenLayers"]
    nodesInHiddenLayers = configData["nodesInHiddenLayers"]
    
    print("Función lógica XOR")
    input = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    expectedOutput = np.array([1, 1, -1, -1])
    
    print(input)
    print(expectedOutput)
    mlp = MultiLayerPerceptron(input, expectedOutput, epochs, epsilon, learningRate, hiddenLayers, nodesInHiddenLayers)
    
    # Entreno el perceptron
    mse = mlp.train()
    
    """ i = 0
    while i < len(y2):
        print("input: ", x2[i])
        print("expected: ", y2[i])
        print("prediction: ", mlp.predict(x2[i]))
        i += 1 """
    print(f'mse: {mse}')
    #TODO accuracy
if __name__ == "__main__":
    main3a()