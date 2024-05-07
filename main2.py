# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import sys
sys.path.append("src")
from src.perceptron_simple_nolineal import perceptron_simple_nolineal, perceptron_simple_nolineal_k
#from scratch.perceptron_simple2v import perceptron_simple2v
from src.perceptron_simple_lineal import perceptron_simple_lineal, perceptron_simple_lineal_k

def main2():
    print('TP 3: PERCEPTRON SIMPLE')
    eta = 0.01
    epoch = 1000

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
    w3 = perceptron_simple_lineal(x3, y3, eta, epsilon, epoch)
    print(f"Pesos finales: {w3}")

    print("Entrenando con Perceptron Simple Lineal K")
    w4 = perceptron_simple_lineal_k(x3, y3, eta, epsilon, epoch)
    print(f"Pesos finales: {w4}")

    print("Entrenando con Perceptron Simple No Lineal")
    w5 = perceptron_simple_nolineal(x3, y3, eta, epsilon, epoch)
    print(f"Pesos finales: {w5}")

    print("Entrenando con Perceptron Simple No Lineal K")
    w6 = perceptron_simple_nolineal_k(x3, y3, eta, epsilon, epoch)
    print(f"Pesos finales: {w6}")


if __name__ == "__main__":
    main2()