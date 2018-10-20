import numpy as np
import matplotlib.pyplot as plt

signal = np.genfromtxt("signal.dat", delimiter = " , ")
incompletos = np.genfromtxt("incompletos.dat", delimiter = " , ")

signal_Amplitud = signal[:, 0]
signal_t = signal[:, 1]

plt.plot(signal_Amplitud, signal_t)
plt.title("Datos de signal.dat")
plt.ylabel("Amplitud")
plt.xlabel("Tiempo")
plt.show()





































#
