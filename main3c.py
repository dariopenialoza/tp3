import json
import numpy as np
import pandas as pd
from src.perceptron_multilayer import MultiLayerPerceptron

def main3c():
    print('EJERCICIO 3 C')
    with open('./config3b.json', 'r') as f:
        configData = json.load(f)
        f.close()
    
    epochs = configData["epochs"]
    epsilon = configData["epsilon"]
    f1 = configData["f1"]
    learningRate = configData["learningRate"]
    hiddenLayers = configData["hiddenLayers"]
    nodesInHiddenLayers = configData["nodesInHiddenLayers"]
    k_index = configData["nodesInHiddenLayers"]
    
    input = parse_file('./TP3-ej3-digitos.txt')
    expectedOutput = np.array([
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 
           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 
           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    
    mlp = MultiLayerPerceptron(input, expectedOutput, epochs, epsilon, learningRate, hiddenLayers, nodesInHiddenLayers)
    
    # Entreno el perceptron
    mse = mlp.train(k_index, 3)
    
    #print(f'mse: {mse}')
    #TODO accuracy 
    
def parse_file(srcFile):
  with open(srcFile, 'r') as file:
    lines = file.readlines()
    data = [[int(value) for value in line.strip().split()] for line in lines]

    # Convierte la lista de listas a una matriz numpy
    matrix = np.array(data)

    # Número de columnas en cada fila
    num_columns = 7

    # Inicializa una lista para almacenar las filas agrupadas
    grouped_rows = []

    # Divide la matriz en grupos de filas y apila las filas agrupadas
    for i in range(0, len(matrix), num_columns):
        grouped_rows.append(matrix[i:i+num_columns].flatten())

    # Convierte la lista de filas agrupadas en una matriz numpy
    grouped_matrix = np.array(grouped_rows)
    
    return grouped_matrix

if __name__ == "__main__":
    main3c()