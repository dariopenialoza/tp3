# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import sys
sys.path.append("src")
from src.perceptron_simple_nolineal import perceptron_simple_nolineal
from src.perceptron_simple_lineal import perceptron_simple_lineal, getTrainingSet

def main2():

    beta = 1.0
    learning_rate = 0.1
    epoch = 1000

    K_perc = 0.8

    print('EJERCIO 2')
    epsilon = 0.01
    # Ruta del archivo CSV
    archivo_csv = "TP3-ej2-conjunto.csv"
    # Leer datos CSV en un DataFrame de pandas
    datos_df = pd.read_csv(archivo_csv)
    # Convertir DataFrame a un array de NumPy
    datos_array = datos_df.to_numpy()
    # Extract feature columns (x1, x2, x3) into 'x' array
    x3 = datos_array[:, :-1]  # Assuming 'y' is the last column

    # Extract target column (y) into 'y' array
    y3 = datos_array[:, -1]

    x_training, x_test, y_training, y_test = getTrainingSet(x3, y3, K_perc)

    print("Entrenando con Perceptron Simple Lineal")
    print(f'learning_rate={learning_rate}, epochs={epoch}')
    w3, error3 = perceptron_simple_lineal(x_training, y_training, learning_rate, epsilon, epoch)
    print(f"Pesos finales: {w3}, error: {error3}")

 

    print("Entrenando con Perceptron Simple No Lineal")
    print(f'learning_rate={learning_rate}, epochs={epoch} beta={beta}')
    w5, error5 = perceptron_simple_nolineal(x_training, y_training, beta, learning_rate, epsilon, epoch)
    print(f"Pesos finales: {w5}, error: {error5}")




if __name__ == "__main__":
    main2()