# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import sys
sys.path.append("src")
from src.perceptron_simple_nolineal import perceptron_simple_nolineal, perceptron_simple_nolineal_k
from src.perceptron_simple_lineal import perceptron_simple_lineal, perceptron_simple_lineal_k

def main2():

    beta = 1.0
    learning_rate = 0.1
    epoch = 100

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

    print("Entrenando con Perceptron Simple Lineal")
    w3, error3 = perceptron_simple_lineal(x3, y3, learning_rate, epsilon, epoch)
    print(f"Pesos finales: {w3}, error: {error3}")

    print("Entrenando con Perceptron Simple Lineal K")
    w4, error4 = perceptron_simple_lineal_k(x3, y3, learning_rate, epsilon, epoch)
    print(f"Pesos finales: {w4}, error: {error4}")

    print("Entrenando con Perceptron Simple No Lineal")
    w5, error5 = perceptron_simple_nolineal(x3, y3, beta, learning_rate, epsilon, epoch)
    print(f"Pesos finales: {w5}, error: {error5}")

    print("Entrenando con Perceptron Simple No Lineal K")
    w6, error6 = perceptron_simple_nolineal_k(x3, y3, beta, learning_rate, epsilon, epoch)
    print(f"Pesos finales: {w6}, error: {error6}")


if __name__ == "__main__":
    main2()