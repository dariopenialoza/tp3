import json
import numpy as np
import pandas as pd

def main3a():
    print('EJERCICIO 3 A')
    with open('./config3.json', 'r') as f:
        configData = json.load(f)
        f.close()
    
    print("Función lógica XOR")
    x2 = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y2 = np.array([1, 1, -1, -1])
    
        
    
if __name__ == "__main__":
    main3a()