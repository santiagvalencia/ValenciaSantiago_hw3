import numpy as np
import matplotlib.pyplot as plt

datos = np.genfromtxt("WDBC.dat", delimiter = ",")
diagnosticos_str = np.genfromtxt("WDBC.dat", delimiter = "," , dtype = 'str', usecols = (1))
diagnosticos = np.zeros(diagnosticos_str.size)
for i in range(diagnosticos_str.size):
    if diagnosticos_str[i] == "B":
        datos[i, 1] = 1
        diagnosticos[i] = 1
    else:
        datos[i, 1] = 0

parametros = datos[:,2:]

N = parametros[0].size #numero de parametros (sin numero de indentificacion ni diagnostico)

Mcov = np.zeros((N, N))

def cov(x, y):
    return np.sum((x-np.mean(x))*(y-np.mean(y)))/(x.size-1)

for i in range(N):
    for j in range(N):
        Mcov[i, j] = cov(parametros[:, i], parametros[:, j])
print(Mcov)

eigValues, eigVectors = np.linalg.eig(Mcov)

for i in range(eigValues.size):
    print("valor propio",i+1, ":", eigValues[i])
    print("vector propio",i+1,":", eigVectors[i])
    print("\n")
