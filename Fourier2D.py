import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from numpy.fft import fft2, ifft2

img = plt.imread("arbol.png")

img_FT = np.abs(np.real(fft2(img)))
plt.imshow(img_FT, norm = LogNorm(vmin = 1), cmap = "hot")
plt.colorbar(label = "Transformada 2D de Fourier")
plt.title("Transformada de Fourier de la imagen")
plt.show()
plt.close("all")

#el ruido se ve como unos cuantos puntos con valores altos (>10**2.5)
#por eso el filtro debe ser pasabajas
img_FT_filtro = np.copy(img_FT)
for i in range(img_FT_filtro[:, 0].size):
    for j in range(img_FT_filtro[0].size):
        if img_FT_filtro[i, j] >= 10.0**3.5:
            img_FT_filtro[i, j] = 10.0**1.5
            print("o")

plt.imshow(img_FT_filtro, norm = LogNorm(vmin = 1), cmap = "hot")
plt.colorbar(label = "Transformada 2D de Fourier")
plt.title("Transformada de Fourier filtrada")
plt.show()
plt.close("all")

img_filtrada = np.real(ifft2(img_FT_filtro))
plt.imshow(img_filtrada)
plt.show()



#plt.imshow(img, cmap = "gray")
#plt.show()
