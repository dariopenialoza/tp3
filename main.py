import numpy as np
import pandas as pd
import csv

import sys
sys.path.append("src")

from src.psAND import psAnd

def main():
    # Definir los datos de entrenamiento
    x = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y = np.array([-1, -1, -1, 1])

    psAnd(x,y)

#

if __name__ == "__main__":
    main()