import numpy as np
import matplotlib.pyplot as plt
#importa los datos del archivo
datos = np.genfromtxt("WDBC.dat", delimiter = ",")
#incluye los diagnosticos como tipo string en un arreglo aparte
diagnosticos_str = np.genfromtxt("WDBC.dat", delimiter = "," , dtype = 'str', usecols = (1))

parametros = datos[:,2:]#guarda los parametros de diagnostico

for i in range(parametros[0].size): #normaliza los datos para poder aplicar PCA
    parametros[:, i] -= np.mean(parametros[:, i])
    parametros[:, i] /= np.sqrt(np.var(parametros[:, i]))

N = parametros[0].size #numero de parametros (sin numero de indentificacion ni diagnostico)

Mcov = np.zeros((N, N))#crea la matriz de covarianza, tamano NxN

def cov(x, y):#funcion que retorna la covarianza entre dos arreglos
    return np.sum((x-np.mean(x))*(y-np.mean(y)))/(x.size-1)
#doble ciclo for recorre la matriz de covarianza
for i in range(N):
    for j in range(N):
        Mcov[i, j] = cov(parametros[:, i], parametros[:, j])#llena la matriz de covarianza entre cada par de parametros
print("\nMatriz de covarianza: ")
print(Mcov)#imprime la matriz de covarianza

numeros = np.arange(1, N+1)#crea un arreglo en el que se guardan los indices iniciales de cada vector y valor propio
eigValues, eigVectors = np.linalg.eig(Mcov)#obtiene los vectores y valores propios de la matriz de covarianza

print("\n")
for i in range(eigValues.size):#imprime los valores propios y sus correspondientes vectores propios
    print("valor propio",i+1, ":", eigValues[i])
    print("vector propio",i+1,":", eigVectors[i])
    print("\n")

#ordena valores propios y sus correspondientes vectores propios
numeros = np.arange(1, N+1)
for i in range(eigValues.size):
    for j in range(eigValues.size):
        if(eigValues[j]<eigValues[i]):
            temp = eigValues[i]
            temp_vect = np.copy(eigVectors[:, i])
            temp_num = numeros[i]
            eigValues[i] = eigValues[j]
            eigVectors[:, i] = np.copy(eigVectors[:, j])
            numeros[i] = numeros[j]
            eigValues[j] = temp
            eigVectors[:, j] = temp_vect
            numeros[j] = temp_num
print("\n")#imprime los dos vectores propios correspondientes a los dos mayores valores propios
print("Componente principal 1: vector", numeros[0], "= ", eigVectors[:, 0].reshape(N, 1), " correspondiente al valor propio", eigValues[0])
print("Componente principal 2: vector", numeros[1], "= ", eigVectors[:, 1].reshape(N, 1), " correspondiente al valor propio", eigValues[1])
print("\n")
#guarda los componentes principales en sus propias variables
PC1 = eigVectors[:, 0].reshape(N, 1)
PC2 = eigVectors[:, 1].reshape(N, 1)
#crea la matriz de proyeccion sobre el plano PC1, PC2
A = np.hstack((PC1, PC2))
#Mproy = np.matmul(A, np.matmul(np.linalg.inv(np.matmul(np.transpose(A), A)), np.transpose(A)))

datos_proy = np.matmul(parametros, A)
print("\n",datos_proy,"\n")
cM = 0#contador de diagnostico maligno
cB = 0#contador de diagnostico benigno
for i in range(datos_proy[:, 0].size):#recorre los datos proyectados
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


"""from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(parametros)

traces = []

for name in ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'):

    trace = Scatter(
        x=Y_sklearn[y==name,0],
        y=Y_sklearn[y==name,1],
        mode='markers',
        name=name,
        marker=Marker(
            size=12,
            line=Line(
                color='rgba(217, 217, 217, 0.14)',
                width=0.5),
            opacity=0.8))
    traces.append(trace)


data = Data(traces)
layout = Layout(xaxis=XAxis(title='PC1', showline=False),
                yaxis=YAxis(title='PC2', showline=False))
fig = Figure(data=data, layout=layout)
py.iplot(fig)"""
