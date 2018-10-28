import numpy as np
import matplotlib.pyplot as plt
#importa los datos del archivo
datos = np.genfromtxt("WDBC.dat", delimiter = ",")
#incluye los diagnosticos como tipo string en un arreglo aparte
diagnosticos_str = np.genfromtxt("WDBC.dat", delimiter = "," , dtype = 'str', usecols = (1))

parametros = datos[:,2:]#guarda los parametros de diagnostico
nombres = ["radius_mean", "radius_std_error", "radius_worst", "texture_mean", "texture_std_error", "texture_worst", "perimeter_mean", "perimeter_std_error", "perimeter_worst" ,"area_mean", "area_std_error", "area_worst", "smoothness_mean", "smoothness_std_error", "smoothness_worst","compactness_mean", "compactness_std_error", "compactness_worst", "concavity_mean", "concavity_std_error", "concavity_worst", "concave_points_mean", "concave_points_std_error", "concave_points_worst", "symmetry_mean", "symmetry_std_error", "symmetry_worst", "fractal_dimension_mean", "fractal_dimension_std_error", "fractal_dimension_worst"]

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
    print("valor propio "+ str(i+1) + ": " + str(eigValues[i]))
    print("vector propio "+str(i+1)+":\n"+ str(eigVectors[:, i].reshape(N, 1)))
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
print("Componente principal 1: vector " + str(numeros[0]) + " = \n" + str(eigVectors[:, 0].reshape(N, 1)) + " correspondiente al valor propio " + str(eigValues[0]))
print("\n")
print("Componente principal 2: vector " +str(numeros[1])+ " = \n" + str(eigVectors[:, 1].reshape(N, 1)) + " correspondiente al valor propio " +str(eigValues[1]))
print("\n")
#guarda los componentes principales en sus propias variables
PC1 = np.copy(eigVectors[:, 0].reshape(N, 1))
PC2 = np.copy(eigVectors[:, 1].reshape(N, 1))
#crea la matriz de proyeccion sobre el sistema de coordenadas PC1, PC2
A = np.hstack((np.copy(PC1), np.copy(PC2)))
#crea dos arreglos de 3 strings para encontrar los parametros mas importantes en cada componente principal
comp1 = ["", "", ""]
comp2 = ["", "", ""]
n1 = 0
n2 = 0
#encuentra los tres parametros mas importantes en PC1 y PC2
while n1 < 3 and n2 < 3:
    for i in range(PC1.size):
        if PC1[i] == np.amax(PC1):
            comp1[n1] = nombres[i]
            PC1[i] = 0
            n1+=1
            break
    for j in range(PC2.size):
        if PC2[j] == np.amax(PC2):
            comp2[n2] = nombres[j]
            PC2[j] = 0
            n2+=1
            break

PC1 = eigVectors[:, 0].reshape(N, 1)
PC2 = eigVectors[:, 1].reshape(N, 1)

print("\nParametros mas importantes en PC1: " + str(comp1[0]) +", "+ str(comp1[1])+", "+str(comp1[2]))
print("Parametros mas importantes en PC2: "+str(comp2[0])+", "+str(comp2[1])+", "+str(comp2[2]))
print("\n")

#proyecta los datos al sistema de coordenadas PC1, PC2
datos_proy = np.matmul(parametros, A)
cM = 0#contador de diagnostico maligno
cB = 0#contador de diagnostico benigno
for i in range(datos_proy[:, 0].size):#recorre los datos proyectados
    if diagnosticos_str[i] == "B":#si el dato es benigno lo colorea de verde
        if cB == 0:#si no se ha encontrado un dato benigno crea el label correspondiente
            plt.scatter(datos_proy[i, 0], datos_proy[i, 1], color = "green", alpha = 0.3, label = "Benigno", marker = "+")
            cB+=1
        else: plt.scatter(datos_proy[i, 0], datos_proy[i, 1], color = "green", alpha = 0.3, marker = "+"); cB+=1
    else:#si el dato es maligno lo colorea de rojo
        if cM == 0:#si no se ha encontrado un dato maligno crea el label correspondiente
            plt.scatter(datos_proy[i, 0], datos_proy[i, 1], color = "red", alpha = 0.3, label = "Maligno", marker = "+")
            cM+=1
        else: plt.scatter(datos_proy[i, 0], datos_proy[i, 1], color = "red", alpha = 0.3, marker = "+"); cM+=1
x = np.linspace(-5, 10, 100)
plt.plot(x, 1.5*x-0.5, linestyle = "--", color = "grey")
plt.legend()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA aplicado a los datos")
plt.savefig("ValenciaSantiago_PCA.pdf")
plt.close("all")

print("El metodo PCA es util para clasificar los diagnosticos, ya que, como se ve en la grafica, los datos proyectados muestran un patron claro que separa los diagnosticos malignos y benignos.")
print("Se puede trazar la linea PC2 = 1.5*PC1-0.5 para separar la mayoria de los diagnosticos")
print("Sin embargo, algunos de los datos no siguen este patron, por lo que el metodo PCA puede ser util para una primera aproximacion al diagnostico pero no como una herramienta definitiva.")
