import numpy as np

def sign(x):
    if x >= 0:
        return 1
    return -1

def der_sign(x):
    return 1

def sigmoid(x):
    return 1 / (1 + np.exp(-2 * x))

def der_sigmoid(x):
    return (2 * np.exp(-2 * x)) / ((1 + np.exp(-2 * x)) **2)

def lineal(x):
    return x

def der_lineal(x):
    return 1

def tanh(x):
    return np.tanh(1 * x)

def der_tanh(x):
    return 1 / ((np.cosh(x)) ** 2)

def softmax(x):
    aux = np.exp(x - np.max(x))
    return aux / np.sum(aux)