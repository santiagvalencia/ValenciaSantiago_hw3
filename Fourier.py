import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp

signal = np.genfromtxt("signal.dat", delimiter = " , ")
incompletos = np.genfromtxt("incompletos.dat", delimiter = " , ")

signal_Amplitud = signal[:, 1]
signal_t = signal[:, 0]

plt.plot(signal_t, signal_Amplitud)
plt.title("Datos de signal.dat")
plt.ylabel("Amplitud")
plt.xlabel("Tiempo (s)")
plt.savefig("ValenciaSantiago_signal.pdf")
plt.close("all")

dt = np.mean(signal_t[1:]-signal_t[:-1])

def FT(g):
    N = g.size
    n = N
    k = N
    G = np.zeros(N) + 0j
    for ni in range(n):
        sum = 0
        for ki in range(k):
            sum += g[ki]*np.exp(1j*2.0*np.pi*ki*ni/N)
        G[ni] = sum
    return G

def frecuencias(n, dt):
    n = (int)(n)
    freq = np.zeros(n)
    if n%2 == 0:
        c = -n/2
        for i in range(n):
            if i <= n/2 - 1:
                freq[i] = i
            else:
                freq[i] = c
                c += 1
    else:
        freq = np.zeros((int)((n-1)/2+1))
        for i in range((int)((n-1)/2+1)):
                freq[i] = i
        freq = np.append(freq, -np.flip(freq[1:]))

    freq/=(dt*n)

    return freq

def IFT(G):
    N = G.size
    n = N
    k = N
    g = np.zeros(N) + 0j
    for ni in range(n):
        sum = 0
        for ki in range(k):
            sum += G[ki]*np.exp(-1j*2.0*np.pi*ki*ni/N)
        g[ni] = sum
    g/=n
    return g

G = FT(signal_Amplitud)
freq = frecuencias(signal_Amplitud.size, dt)

print("\nNo se usa el paquete fftfreq (bono)\n")
plt.plot(freq, np.real(G))
plt.title("Transformada de Fourier de signal.dat")
plt.xlabel("Frecuencia (Hz)")
plt.xlim(-2500, 2500)
plt.ylabel("Transformada de Fourier")
plt.savefig("ValenciaSantiago_TF.pdf")
plt.close("all")

print("Frecuencias principales:")
G_real = np.real(G)
for i in range(G_real.size):
    if np.fabs(G_real[i]) >= 200 and freq[i]>=0:
        print (freq[i], " Hz")

def pasabajas(G, freq, fc):
    G2 = np.copy(G)
    for i in range(freq.size):
        if np.fabs(freq[i]) >= fc:
            G2[i] = 0
    return G2

G2 = pasabajas(G, freq, 1000)

g2 = IFT(G2)

plt.plot(signal_t, np.real(g2))
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.title("Datos filtrados de signal.dat")
plt.savefig("ValenciaSantiago_filtrada.pdf")
plt.close("all")

incompletos_Amplitud = incompletos[:, 1]
incompletos_t = incompletos[:, 0]

print("\nNo se puede hacer una buena TDF con incompletos.dat porque el espaciamiento temporal entre los datos es muy grande para el rango de tiempo que cubre la senal.\n")

incompletos_interpolacion_t = np.linspace(np.amin(incompletos_t), np.amax(incompletos_t), 512)

interp_cuadratica = interp.interp1d(incompletos_t, incompletos_Amplitud, kind='quadratic')
interp_cubica = interp.interp1d(incompletos_t, incompletos_Amplitud, kind='cubic')

incompletos_Amplitud_cuadratica = interp_cuadratica(incompletos_interpolacion_t)
incompletos_Amplitud_cubica = interp_cubica(incompletos_interpolacion_t)

transf_cuad = FT(incompletos_Amplitud_cuadratica)
transf_cub = FT(incompletos_Amplitud_cubica)
frec_interp = frecuencias(incompletos_interpolacion_t.size, np.mean(incompletos_interpolacion_t[1:]-incompletos_interpolacion_t[:-1]))


fig, plots = plt.subplots(3, sharex=True)
fig.suptitle("TDF para signal.dat e interpolaciones de incompletos.dat")
plots[0].plot(freq, np.real(G))
plots[0].set_title("TDF de signal.dat")
plots[1].plot(frec_interp, np.real(transf_cuad), color = "black")
plots[1].set_title("TDF de interp. cuadratica")
plots[2].plot(frec_interp, np.real(transf_cub), color = "red")
plots[2].set_title("TDF de interp. cubica")
fig.subplots_adjust(hspace=0.5)
plt.xlim(-2500, 2500)
plt.xlabel("Frecuencia (Hz)")
for p in plots:
    p.set(ylabel = "TDF")
plt.savefig("ValenciaSantiago_TF_Interpola.pdf")
plt.close("all")
