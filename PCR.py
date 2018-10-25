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

parametros = datos[:,1:]

for i in range(parametros[0].size):
    parametros[:, i] -= np.mean(parametros[:, i])

N = parametros[0].size #numero de parametros (sin numero de indentificacion ni diagnostico)

Mcov = np.zeros((N, N))

def cov(x, y):
    return np.sum((x-np.mean(x))*(y-np.mean(y)))/(x.size-1)

for i in range(N):
    for j in range(N):
        Mcov[i, j] = cov(parametros[:, i], parametros[:, j])
print(Mcov)

numeros = np.arange(1, N+1)
eigValues, eigVectors = np.linalg.eig(Mcov)

print("\n")
for i in range(eigValues.size):
    print("valor propio",i+1, ":", eigValues[i])
    print("vector propio",i+1,":", eigVectors[i])
    print("\n")

#ordena valores propios y sus correspondientes vectores propios
for i in range(eigValues.size):
    for j in range(eigValues.size):
        if(eigValues[j]<eigValues[i]):
            temp = eigValues[i]
            temp_vect = eigVectors[i]
            temp_num = numeros[i]
            eigValues[i] = eigValues[j]
            eigVectors[i] = eigVectors[j]
            numeros[i] = numeros[j]
            eigValues[j] = temp
            eigVectors[j] = temp_vect
            numeros[j] = temp_num

print("Componente principal 1: vector", numeros[0], "= ", eigVectors[0], " correspondiente al valor propio", eigValues[0])
print("Componente principal 2: vector", numeros[1], "= ", eigVectors[1], " correspondiente al valor propio", eigValues[1])
