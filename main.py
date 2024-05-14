# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys

from main3a import main3a
from main3b import main3b
from main3c import main3c
sys.path.append("src")
from main1 import main1
from main2 import main2
def main():
    print('TP 3: PERCEPTRON SIMPLE')
    main1()
    print()
    main2()
    main3a()
    main3b()
    main3c()

if __name__ == "__main__":
    main()