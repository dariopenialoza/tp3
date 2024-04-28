# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import sys
sys.path.append("src")

#from scratch.perceptron_simple2v import perceptron_simple2v
from src.perceptron_simple_lineal import perceptron_simple_lineal
from src.perceptron_simple_step import perceptron_simple_step, perceptron_simple_step_predictor
#from src.perceptron_simple_step_predictor import perceptron_simple_step_predictor



def main():
    print('TP 3: PERCEPTRON SIMPLE')
    eta = 0.01
    epoch = 1000

    print('EJERCIO 1')

    print("Función lógica AND")
    x1 = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y1 = np.array([-1, -1, -1, 1])
    w1 = np.zeros_like(x1)
    #w1v2 = np.zeros_like(x1)
    y_res1 = np.zeros_like(y1)

    #print("Con Perceptron con 2 variables")
    #w1v2 = perceptron_simple2v(x1,y1,eta,epoch)
    #print()
    print("Entrenando con Perceptron Simple Escalon")
    w1 = perceptron_simple_step(x1, y1, eta, epoch)
    print(f"Pesos finales: {w1}")

    #print("Generalizando con Perceptron Simple 2V Escalon")
    #y_res1 = perceptron_simple_step_predictor(x1, w1v2)
    #print(f'Entrada: {x1}')
    #print(f"Resultado: {y_res1}")

    print("Generalizando con Perceptron Simple Escalon")
    y_res1 = perceptron_simple_step_predictor(x1, w1)
    print(f"Entrada: ")
    print(x1)
    print(f"Resultado: {y_res1}")

    print()
    print("Función lógica XOR")
    x2 = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y2 = np.array([1, 1, -1, -1])
    w2 = np.zeros_like(x2)
    y_res2 = np.zeros_like(y2)

    #print("Con Perceptron con 2 variables")
    #perceptron_simple2v(x2,y2,eta,epoch)
    #print()
    print("Entrenando con Perceptron Simple Escalon")
    w2 = perceptron_simple_step(x2, y2, eta, epoch)
    print(f"Pesos finales: {w2}")

    print("Generalizando con Perceptron Simple Escalon")
    y_res2 = perceptron_simple_step_predictor(x2, w2)
    print(f"Entrada: ")
    print(x2)
    print(f"Resultado: {y_res2}")

    print()
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


if __name__ == "__main__":
    main()