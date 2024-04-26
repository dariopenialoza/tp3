import numpy as np
import pandas as pd
import csv

import sys
sys.path.append("src")

from src.psAND import perceptron_simple2v

def main():
    print("Función lógica AND")
    x = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y = np.array([-1, -1, -1, 1])

    perceptron_simple2v(x,y)
    
    print("Función lógica XOR")
    x = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y = np.array([1, 1, -1, -1])

    perceptron_simple2v(x,y)
    

if __name__ == "__main__":
    main()