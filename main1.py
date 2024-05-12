# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import sys
sys.path.append("src")

#from scratch.perceptron_simple2v import perceptron_simple2v
from src.perceptron_simple_lineal import perceptron_simple_lineal
from src.perceptron_simple_step import perceptron_simple_step, perceptron_simple_step_predictor
#from src.perceptron_simple_step_predictor import perceptron_simple_step_predictor



def main1():
    
    eta = 0.001
    epoch = 500

    print('EJERCIO 1')

    print()
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
    print(f'learning_rate={eta}, epochs={epoch}')
    w1, error = perceptron_simple_step(x1, y1, eta, epoch)
    print(f"Pesos finales: {w1}, error: {error}")

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
    print(f'learning_rate={eta}, epochs={epoch}')
    w2, error2 = perceptron_simple_step(x2, y2, eta, epoch)
    print(f"Pesos finales: {w2}, error: {error2}")

    print("Generalizando con Perceptron Simple Escalon")
    y_res2 = perceptron_simple_step_predictor(x2, w2)
    print(f"Entrada: ")
    print(x2)
    print(f"Resultado: {y_res2}")
    

if __name__ == "__main__":
    main1()