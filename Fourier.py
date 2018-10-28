import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp

#extrae los datos de los archivos de texto
signal = np.genfromtxt("signal.dat", delimiter = " , ")
incompletos = np.genfromtxt("incompletos.dat", delimiter = " , ")
#crea arreglos de amplitud y tiempo de signal.dat
signal_Amplitud = signal[:, 1]
signal_t = signal[:, 0]
#grafica la senal de signal.dat
plt.plot(signal_t, signal_Amplitud)
plt.title("Datos de signal.dat")
plt.ylabel("Amplitud")
plt.xlabel("Tiempo (s)")
plt.savefig("ValenciaSantiago_signal.pdf")
plt.close("all")
#determina el intervalo de tiempo de los datos
dt = np.mean(signal_t[1:]-signal_t[:-1])

def FT(g):#funcion que implementa la TDF de una funcion g
    N = g.size #toma el tamano de g
    n = N #primer indice de la suma
    k = N #segundo indice de la suma
    G = np.zeros(N) + 0j #crea el arreglo complejo donde se almacenara la transformada
    for ni in range(n): #recorre el primer indice
        sum = 0 #inicializa la suma en 0
        for ki in range(k): #recorre el segundo indice
            sum += g[ki]*np.exp(-1j*2.0*np.pi*ki*ni/N) #suma segun la formula de TDF
        G[ni] = sum #incorpora el resultado al arreglo de la transformada
    return G #retorna el arreglo de la transformada

def frecuencias(n, dt): #funcion que encuentra las frecuencias correspondientes segun el numero de datos y el intervalo de tiempo
    n = (int)(n) #convierte el numero de datos a int para poder utilizar como indice
    freq = np.zeros(n) #crea el arreglo donde se almacenaran las frecuencias
    if n%2 == 0: #caso de n par segun la documentacion de ifft
        c = -n/2 #crea un contador para la segunda mitad de los datos
        for i in range(n):
            if i <= n/2 - 1:
                freq[i] = i #antes de la mitad la frecuencia es simplemente la posicion
            else:
                freq[i] = c #despues de la mitad la frecuencia es -n/2 y aumenta 1 hasta llegar a -1
                c += 1
    else: #caso de n impar segun la documentacion de ifft
        freq = np.zeros((int)((n-1)/2+1)) #crea un arreglo de tamano (n-1)/2 +1
        for i in range((int)((n-1)/2+1)): #llena el arreglo con su posicion correspondiente para la primera mitad
                freq[i] = i
        freq = np.append(freq, -np.flip(freq[1:]))#la segunda mitad del arreglo es el "reflejo negativo" de la primera mitad

    freq/=(dt*n) #divide entre dt*n segun la documentacion de ifft

    return freq #retorna las frecuencias

def IFT(G): #funcion que encuentra la transformada inversa de Fourier de una funcion G
    N = G.size #toma el tamano de G
    n = N #primer indice de la suma
    k = N #segundo indice de la suma
    g = np.zeros(N) + 0j #arreglo complejo que almacenara los valores de la funcion transformada inversamente
    for ni in range(n): #recorre el primer indice de la suma
        sum = 0 #inicializa el sumador en 0
        for ki in range(k): #recorre el segundo indice de la suma
            sum += G[ki]*np.exp(1j*2.0*np.pi*ki*ni/N) #suma segun la formula de la transformada inversa de Fourier
        g[ni] = sum #incorpora el resultado al arreglo de la transformada inversa
    g/=N #normaliza la transformada inversa segun el numero de datos
    return g #retorna la transformada inversa
#transforada y frecuencias correspondientes de signal.dat:
G = FT(signal_Amplitud)
freq = frecuencias(signal_Amplitud.size, dt)
#grafica de la transformada de Fourier
print("\nNo se usa el paquete fftfreq (bono)\n")
plt.plot(freq, np.real(G))
plt.title("Transformada de Fourier de signal.dat")
plt.xlabel("Frecuencia (Hz)")
plt.xlim(-2500, 2500)#limita el rango de frecuencias
plt.ylabel("Transformada de Fourier")
plt.savefig("ValenciaSantiago_TF.pdf")
plt.close("all")

print("Frecuencias principales:")
G_real = np.real(G)
for i in range(G_real.size):
    if np.fabs(G_real[i]) >= 200 and freq[i]>=0: #se tomo el rango de frecuencias principales como aquellas en las que la transformada tuviera un valor mayor o igual a 200
        print (freq[i], " Hz")

def pasabajas(G, freq, fc): #funcion que toma un arreglo en dominio de Fourier, sus frecuencias correspondientes y una frecuencia de corte y retorna un arreglo en dominio de tiempo filtrado (con pasabajas)
    G2 = np.copy(G) #copia el arreglo original
    for i in range(freq.size):#recorre las frecuencias
        if np.fabs(freq[i]) >= fc: #si la frecuencia es mayor a la frecuencia de corte, se anula el dato correspondiente en el arreglo en dominio de Fourier
            G2[i] = 0
    g2 = IFT(G2) #se invierte la senal filtrada para obtener la senal en dominio de tiempo

    return g2 #retorna la senal filtrada en dominio de tiempo

signal_filtrado = pasabajas(G, freq, 1000) #filtra la senal de signal.dat como pide el enunciado
#grafica y guarda la senal filtrada
plt.plot(signal_t, np.real(signal_filtrado))
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.title("Datos filtrados de signal.dat")
plt.savefig("ValenciaSantiago_filtrada.pdf")
plt.close("all")
#crea arreglos de amplitud y tiempo de incompletos.dat
incompletos_Amplitud = incompletos[:, 1]
incompletos_t = incompletos[:, 0]

print("\nNo se puede hacer una buena TDF con incompletos.dat porque el espaciamiento temporal entre los datos es muy grande para el rango de tiempo que cubre la senal.\n")
#crea el arreglo de 512 datos de tiempo con el que se hara la interpolacion
incompletos_interpolacion_t = np.linspace(np.amin(incompletos_t), np.amax(incompletos_t), 512)
#crea las funciones de interpolacion cuadratica y cubica para los datos de incompletos.dat
interp_cuadratica = interp.interp1d(incompletos_t, incompletos_Amplitud, kind='quadratic')
interp_cubica = interp.interp1d(incompletos_t, incompletos_Amplitud, kind='cubic')
#crea arreglos interpolados de 512 datos
incompletos_Amplitud_cuadratica = interp_cuadratica(incompletos_interpolacion_t)
incompletos_Amplitud_cubica = interp_cubica(incompletos_interpolacion_t)
#transforma los arreglos interpolados
transf_cuad = FT(incompletos_Amplitud_cuadratica)
transf_cub = FT(incompletos_Amplitud_cubica)
frec_interp = frecuencias(incompletos_interpolacion_t.size, np.mean(incompletos_interpolacion_t[1:]-incompletos_interpolacion_t[:-1]))#obtiene las frecuencias correspondientes a los arreglos interpolados
#grafica las transformadas de los arreglos interpolados y de los datos de signal.dat
fig, plots = plt.subplots(3, sharex=True)
fig.suptitle("TDF para signal.dat e interpolaciones de incompletos.dat")
plots[0].plot(freq, np.real(G))
plots[0].set_title("TDF de signal.dat")
plots[1].plot(frec_interp, np.real(transf_cuad), color = "black")
plots[1].set_title("TDF de interp. cuadratica")
plots[2].plot(frec_interp, np.real(transf_cub), color = "red")
plots[2].set_title("TDF de interp. cubica")
fig.subplots_adjust(hspace=0.5)
plt.xlim(-2500, 2500)#limita el rango de frecuencias
plt.xlabel("Frecuencia (Hz)")
for p in plots:
    p.set(ylabel = "TDF")
plt.savefig("ValenciaSantiago_TF_Interpola.pdf")
plt.close("all")

print("\nLas transformadas de los datos interpolados tienen un mayor ruido en las zonas de alta frecuencia. Ademas, los picos presentes en la frecuencia de 385 Hz en la transformada de signal.dat se encuentran suavizados en las interpolaciones.\n")
#filtra las interpolaciones con pasabajas de 1000 Hz (los datos de signal.dat ya se habian filtrado en el arreglo signal_filtrado)
cuad_pasabajas_1000 = pasabajas(transf_cuad, frec_interp, 1000)
cub_pasabajas_1000 = pasabajas(transf_cub, frec_interp, 1000)
#filtra las interpolaciones y los datos de signal.dat con pasabajas de 500 Hz
pasabajas_500 = pasabajas(G, freq, 500)
cuad_pasabajas_500 = pasabajas(transf_cuad, frec_interp, 500)
cub_pasabajas_500 = pasabajas(transf_cub, frec_interp, 500)
#grafica los resultados
fig, plots = plt.subplots(2, sharex = True)
plots[0].plot(incompletos_interpolacion_t, np.real(cuad_pasabajas_1000), linestyle = "dashed", label = "interp. cuad.")
plots[0].plot(incompletos_interpolacion_t, np.real(cub_pasabajas_1000), linestyle = "dotted", label = "interp.cub")
plots[0].plot(signal_t, np.real(signal_filtrado), linestyle = "-.", label = "signal.dat")
plots[0].legend()
plots[0].set_title("Pasabajas 1000 Hz")
plots[1].plot(incompletos_interpolacion_t, np.real(cuad_pasabajas_500), linestyle = "dashed", label = "interp. cuad.")
plots[1].plot(incompletos_interpolacion_t, np.real(cub_pasabajas_500), linestyle = "dotted", label = "interp.cub")
plots[1].plot(signal_t, np.real(pasabajas_500), linestyle = "-.", label = "signal.dat")
plots[1].legend()
plots[1].set_title("Pasabajas 500 Hz")
for p in plots:
    p.set(ylabel = "Amplitud")
plt.xlabel("Tiempo (s)")
plt.savefig("ValenciaSantiago_2Filtros.pdf")
