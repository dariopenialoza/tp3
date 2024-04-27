import numpy as np
import pandas as pd
import csv

import sys
sys.path.append("src")

from src.perceptron_simple2v import perceptron_simple2v
from src.perceptron_simple_step import perceptron_simple_step

def main():
    print("Función lógica AND")
    x = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y = np.array([-1, -1, -1, 1])

    perceptron_simple2v(x,y)
    
    print("Función lógica XOR")
    x = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y = np.array([1, 1, -1, -1])

    perceptron_simple2v(x,y)


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

if __name__ == "__main__":
    main()