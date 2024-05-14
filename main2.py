# -*- coding: utf-8 -*-

import csv
import numpy as np
import pandas as pd

import sys
sys.path.append("src")
from src.perceptron_simple_nolineal import crossvalidation_nolineal_error_estimacion, perceptron_simple_nolineal, perceptron_simple_nolineal_predictor, perceptron_simple_nolineal_predictor_error
from src.perceptron_simple_lineal import  crossvalidation_lineal_error_estimacion, perceptron_simple_lineal, getTrainingSet, perceptron_simple_lineal_predictor, perceptron_simple_lineal_predictor_error

def main2():

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

   
    beta = 1.0
    epoch = 2500
    K_perc = 0.8


    x_training, x_test, y_training, y_test = getTrainingSet(x3, y3, K_perc)

    # VARIANDO LEARNING-RATE
    learning_rate = 0.01

    #TRAINING
    print("Entrenando con Perceptron Simple Lineal")
    print(f'learning_rate={learning_rate}, epochs={epoch}')
    w3, error3 = perceptron_simple_lineal(x_training, y_training, learning_rate, epsilon, epoch,11)
    print(f"error(MSE): {error3}")

    #TESTING
    y3l_result = perceptron_simple_lineal_predictor(x_test, w3)
    error3_predictor = perceptron_simple_lineal_predictor_error(y_test,y3l_result)
    print(f"error(MSE) de estimación: {error3_predictor}")

    print()
    #TRAINING
    print("Entrenando con Perceptron Simple No Lineal")
    print(f'learning_rate={learning_rate}, epochs={epoch} beta={beta}')
    w5, error5 = perceptron_simple_nolineal(x_training, y_training, beta, learning_rate, epsilon, epoch,12)
    print(f"error(MSE): {error5}")

    #TESTING
    y3nol_result = perceptron_simple_nolineal_predictor(x_test, y_test, w5, beta)
    error5_predictor = perceptron_simple_nolineal_predictor_error(x_test, y_test,y3nol_result)
    print(f"error(MSE) de estimación: {error5_predictor}")

    print()

    # CON LEARNING RATE 0.1 Y EPOCHS 2500
    learning_rate = 0.001

    #TRAINING
    print("Entrenando con Perceptron Simple Lineal")
    print(f'learning_rate={learning_rate}, epochs={epoch}')
    w3, error3 = perceptron_simple_lineal(x_training, y_training, learning_rate, epsilon, epoch,21)
    print(f"error (MSE): {error3}")

    #TESTING
    y3l_result = perceptron_simple_lineal_predictor(x_test, w3)
    error3_predictor = perceptron_simple_lineal_predictor_error(y_test,y3l_result)
    print(f"error(MSE) de estimación: {error3_predictor}")

    print()    
    #TRAINING
    print("Entrenando con Perceptron Simple No Lineal")
    print(f'learning_rate={learning_rate}, epochs={epoch} beta={beta}')
    w5, error5 = perceptron_simple_nolineal(x_training, y_training, beta, learning_rate, epsilon, epoch,22)
    print(f"error(MSE): {error5}")

    #TESTING
    y3nol_result = perceptron_simple_nolineal_predictor(x_test, y_test, w5, beta)
    error5_predictor = perceptron_simple_nolineal_predictor_error(x_test, y_test,y3nol_result)
    print(f"error(MSE) de estimación: {error5_predictor}")

    print()
    # CON LEARNING RATE 0.1 Y EPOCHS 2500
    learning_rate = 0.0001

    #TRAINING
    print("Entrenando con Perceptron Simple Lineal")
    print(f'learning_rate={learning_rate}, epochs={epoch}')
    w3, error3 = perceptron_simple_lineal(x_training, y_training, learning_rate, epsilon, epoch,31)
    print(f"error(MSE): {error3}")

    #TESTING
    y3l_result = perceptron_simple_lineal_predictor(x_test, w3)
    error3_predictor = perceptron_simple_lineal_predictor_error(y_test,y3l_result)
    print(f"error(MSE) de estimación: {error3_predictor}")

    print()
    #TRAINING
    print("Entrenando con Perceptron Simple No Lineal")
    print(f'learning_rate={learning_rate}, epochs={epoch} beta={beta}')
    w5, error5 = perceptron_simple_nolineal(x_training, y_training, beta, learning_rate, epsilon, epoch,32)
    print(f"error(MSE): {error5}")

    #TESTING
    y3nol_result = perceptron_simple_nolineal_predictor(x_test, y_test, w5, beta)
    error5_predictor = perceptron_simple_nolineal_predictor_error(x_test, y_test,y3nol_result)
    print(f"error(MSE) de estimación: {error5_predictor}")
    

    learning_rate = 0.01
    # VARIANDO BETA
    beta=0.001
    print()
    #TRAINING
    print("Entrenando con Perceptron Simple No Lineal")
    print(f'learning_rate={learning_rate}, epochs={epoch} beta={beta}')
    w5, error5 = perceptron_simple_nolineal(x_training, y_training, beta, learning_rate, epsilon, epoch,0)
    print(f"error(MSE): {error5}")

    #TESTING
    y3nol_result = perceptron_simple_nolineal_predictor(x_test, y_test, w5, beta)
    error5_predictor = perceptron_simple_nolineal_predictor_error(x_test,y_test,y3nol_result)
    print(f"error(MSE) de estimación: {error5_predictor}")

    print()

    beta=0.01
    print()
    #TRAINING
    print("Entrenando con Perceptron Simple No Lineal")
    print(f'learning_rate={learning_rate}, epochs={epoch} beta={beta}')
    w5, error5 = perceptron_simple_nolineal(x_training, y_training, beta, learning_rate, epsilon, epoch,1)
    print(f"error(MSE): {error5}")

    #TESTING
    y3nol_result = perceptron_simple_nolineal_predictor(x_test, y_test, w5, beta)
    error5_predictor = perceptron_simple_nolineal_predictor_error(x_test,y_test,y3nol_result)
    print(f"error(MSE) de estimación: {error5_predictor}")

    print()

    beta=0.1   

    print()    
    #TRAINING
    print("Entrenando con Perceptron Simple No Lineal")
    print(f'learning_rate={learning_rate}, epochs={epoch} beta={beta}')
    w5, error5 = perceptron_simple_nolineal(x_training, y_training, beta, learning_rate, epsilon, epoch,2)
    print(f"error(MSE): {error5}")

    #TESTING
    y3nol_result = perceptron_simple_nolineal_predictor(x_test, y_test, w5, beta)
    error5_predictor = perceptron_simple_nolineal_predictor_error(x_test,y_test,y3nol_result)
    print(f"error(MSE) de estimación: {error5_predictor}")

    print()
    beta=1.0

    print()
    #TRAINING
    print("Entrenando con Perceptron Simple No Lineal")
    print(f'learning_rate={learning_rate}, epochs={epoch} beta={beta}')
    w5, error5 = perceptron_simple_nolineal(x_training, y_training, beta, learning_rate, epsilon, epoch,3)
    print(f"error(MSE): {error5}")

    #TESTING
    y3nol_result = perceptron_simple_nolineal_predictor(x_test, y_test, w5, beta)
    error5_predictor = perceptron_simple_nolineal_predictor_error(x_test,y_test,y3nol_result)
    print(f"error(MSE) de estimación: {error5_predictor}")

    print()
    beta=1.1

    print()
    #TRAINING
    print("Entrenando con Perceptron Simple No Lineal")
    print(f'learning_rate={learning_rate}, epochs={epoch} beta={beta}')
    w5, error5 = perceptron_simple_nolineal(x_training, y_training, beta, learning_rate, epsilon, epoch,4)
    print(f"error(MSE): {error5}")

    #TESTING
    y3nol_result = perceptron_simple_nolineal_predictor(x_test, y_test, w5, beta)
    error5_predictor = perceptron_simple_nolineal_predictor_error(x_test,y_test,y3nol_result)
    print(f"error(MSE) de estimación: {error5_predictor}")

    
    
    print()
    print('CROSS VALIDATION')

    
    learning_rate = 0.01
    beta = 1.0
    epsilon=0.00001
    epoch=2500

    error_predic_lineal = []
    error_predic_nolineal = []
    for k in range(2,11):
        wl,el,elp = crossvalidation_lineal_error_estimacion(x3, y3, k, learning_rate, epsilon, epoch)
        error_predic_lineal.append([k,el,elp])
        print()
        wnl,enl,enlp = crossvalidation_nolineal_error_estimacion(x3, y3, k, beta,learning_rate, epsilon, epoch)
        error_predic_nolineal.append([k,enl,enlp])
    with open(f'perceptron_simple_lineal_error_predic_Cross-{epoch}-{learning_rate}.csv', 'w', newline='') as archivo:
        escritor_csv = csv.writer(archivo)
        escritor_csv.writerows(error_predic_lineal)
    with open(f'perceptron_simple_Nolineal_error_predic_Cross-{epoch}-{learning_rate}.csv', 'w', newline='') as archivo2:
        escritor_csv2 = csv.writer(archivo2)
        escritor_csv2.writerows(error_predic_nolineal)  
        
    
    learning_rate = 0.01
    epsilon=0.00001
    beta = 1.0
    
    for k_perc in [0.5, 0.6, 0.7, 0.8, 0.9]:
        x_training, x_test, y_training, y_test = getTrainingSet(x3, y3, k_perc)
        print()
        print(f'K={k_perc}')
        print('PERCEPTRON SIMPLE LINEAL')
        w1, e = perceptron_simple_lineal(x_training, y_training, learning_rate, epsilon, epoch, k_perc)
        print(f"error(MSE): {e}")
        y3l_result = perceptron_simple_lineal_predictor(x_test, w1)
        error3_predictor = perceptron_simple_lineal_predictor_error(y_test,y3l_result)
        print(f"error(MSE) de estimación: {error3_predictor}")

        print()
        print('PERCEPTRON SIMPLE NO LINEAL')
        w2, e = perceptron_simple_nolineal(x_training, y_training, beta, learning_rate, epsilon, epoch, K_perc)
        print(f"error(MSE): {e}")
        y3nol_result = perceptron_simple_nolineal_predictor(x_test, y_test, w2, beta)
        error5_predictor = perceptron_simple_nolineal_predictor_error(y_test,y3nol_result)
        print(f"error(MSE) de estimación: {error5_predictor}")
    
if __name__ == "__main__":
    main2()