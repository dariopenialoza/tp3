import numpy as np
import pandas as pd
import csv

import sys
sys.path.append("src")

from src.perceptron_simple2v import perceptron_simple2v
from src.perceptron_simple_step import perceptron_simple_step
from src.perceptron_simple_step_predictor import perceptron_simple_step_predictor

def main():

    eta = 0.1
    epoch = 10000 

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
    print(f'Pesos finales: {w1} ')

    #print("Generalizando con Perceptron Simple 2V Escalon")
    #y_res1 = perceptron_simple_step_predictor(x1, w1v2)
    #print(f'Entrada: {x1}')
    #print(f"Resultado: {y_res1}")

    print("Generalizando con Perceptron Simple Escalon")
    y_res1 = perceptron_simple_step_predictor(x1, w1)
    print(f'Entrada: {x1}')
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
    print(f'Pesos finales: {w2} ')

    print("Generalizando con Perceptron Simple Escalon")
    y_res2 = perceptron_simple_step_predictor(x2, w2)
    print(f'Entrada: {x2}')
    print(f"Resultado: {y_res2}")


    """
    # Ruta del archivo CSV
    archivo_csv = "TP3-ej2-conjunto.csv"
    # Leer datos CSV en un DataFrame de pandas
    datos_df = pd.read_csv(archivo_csv)
    # Convertir DataFrame a un array de NumPy
    datos_array = datos_df.to_numpy()
    # Extract feature columns (x1, x2, x3) into 'x' array
    x = datos_array[:, :-1]  # Assuming 'y' is the last column

    # Extract target column (y) into 'y' array
    y = datos_array[:, -1]

    eta = 0.1
    epoch = 100  

    print("Pesos para el conjunto de datos:")
    perceptron_simple_step(x, y, eta, epoch)
    """

if __name__ == "__main__":
    main()