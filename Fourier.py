import numpy as np
import matplotlib.pyplot as plt

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
        freq = np.zeros((int)((n-1)/2))
        for i in range((int)((n-1)/2)):
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
plt.ylabel("Transformada de Fourier")
plt.savefig("ValenciaSantiago_TF.pdf")
plt.close("all")

print("Frecuencias principales:")
G_real = np.real(G)
for i in range(G_real.size):
    if np.fabs(G_real[i]) >= 200 and freq[i]>=0:
        print (freq[i], " Hz")

G2 = np.copy(G)
for i in range(freq.size):
    if np.fabs(freq[i]) >= 1000:
        G2[i] = 0

g2 = IFT(G2)

plt.plot(signal_t, np.real(g2))
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.title("Datos filtrados de signal.dat")
plt.savefig("ValenciaSantiago_filtrada.pdf")
plt.close("all")
