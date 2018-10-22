import numpy as np
import matplotlib.pyplot as plt

datos = np.genfromtxt("WDBC.dat", delimiter = ",")
datos_str = np.genfromtxt("WDBC.dat", delimiter = "," , dtype = 'str', usecols = (1))

for i in range(datos_str.size):
    if datos_str[i] == "B":
        datos[i, 1] = 1
    else: datos[i, 1] = 0
