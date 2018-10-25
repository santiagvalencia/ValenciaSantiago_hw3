import numpy as np
import matplotlib.pyplot as plt

datos = np.genfromtxt("WDBC.dat", delimiter = ",")
diagnosticos_str = np.genfromtxt("WDBC.dat", delimiter = "," , dtype = 'str', usecols = (1))

parametros = datos[:,2:]

for i in range(parametros[0].size):
    parametros[:, i] -= np.mean(parametros[:, i])
    parametros[:, i] /= np.sqrt(np.var(parametros[:, i]))

N = parametros[0].size #numero de parametros (sin numero de indentificacion ni diagnostico)

Mcov = np.zeros((N, N))

def cov(x, y):
    return np.sum((x-np.mean(x))*(y-np.mean(y)))/(x.size-1)

for i in range(N):
    for j in range(N):
        Mcov[i, j] = cov(parametros[:, i], parametros[:, j])
print("\nMatriz de covarianza: ")
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
            temp_vect = np.copy(eigVectors[i])
            temp_num = numeros[i]
            eigValues[i] = eigValues[j]
            eigVectors[i] = np.copy(eigVectors[j])
            numeros[i] = numeros[j]
            eigValues[j] = temp
            eigVectors[j] = temp_vect
            numeros[j] = temp_num
print("\n")
print("Componente principal 1: vector", numeros[0], "= ", eigVectors[0], " correspondiente al valor propio", eigValues[0])
print("Componente principal 2: vector", numeros[1], "= ", eigVectors[1], " correspondiente al valor propio", eigValues[1])
print("\n")

PC1 = eigVectors[0]
PC2 = eigVectors[1]

Mproy = np.matmul(np.hstack((PC1.reshape(N, 1), PC2.reshape(N, 1))) , np.vstack((PC1, PC2)))
#Mproy2 = np.vstack((PC1, PC2))

print(np.allclose(Mproy, np.matmul(Mproy, Mproy)))
print("\n")

datos_proy = np.matmul(parametros, Mproy)

cM = 0
cB = 0
for i in range(datos_proy[:, 0].size):
    if diagnosticos_str[i] == "B":
        if cM == 0:
            plt.scatter(datos_proy[i, 0], datos_proy[i, 1], color = "green", alpha = 0.3, label = "Benigno", marker = "+")
            cM+=1
        else: plt.scatter(datos_proy[i, 0], datos_proy[i, 1], color = "green", alpha = 0.3, marker = "+")
    else:
        if cB == 0:
            plt.scatter(datos_proy[i, 0], datos_proy[i, 1], color = "red", alpha = 0.3, label = "Maligno", marker = "+")
            cB+=1
        else: plt.scatter(datos_proy[i, 0], datos_proy[i, 1], color = "red", alpha = 0.3, marker = "+")
plt.legend()
plt.show()
