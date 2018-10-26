import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from numpy.fft import fft2, ifft2

img = plt.imread("arbol.png")

img_FT = fft2(img)
plt.imshow(np.abs(np.real(img_FT)), norm = LogNorm(vmin = 5), cmap = "hot")
plt.colorbar(label = "Transformada 2D de Fourier")
plt.title("Transformada de Fourier de la imagen")
plt.show()
plt.close("all")

#el ruido se ve como unos cuantos puntos con valores altos (>10**3.3)
#por eso el filtro debe ser pasabajas
img_FT_filtro = np.copy(img_FT)
for i in range(img_FT_filtro[:, 0].size):
    for j in range(img_FT_filtro[0].size):
        if np.abs(np.real(img_FT[i, j])) >= 10.0**3.3:
            img_FT_filtro[i, j] = (img_FT_filtro[i+1, j] + img_FT_filtro[i-1, j] + img_FT_filtro[i, j+1] + img_FT_filtro[i, j-1])/4

plt.imshow(np.abs(np.real(img_FT_filtro)), norm = LogNorm(vmin = 5), cmap = "hot")
plt.colorbar(label = "Transformada 2D de Fourier")
plt.title("Transformada de Fourier filtrada")
plt.show()
plt.close("all")

img_filtrada = ifft2(img_FT_filtro)
plt.imshow(np.real(img_filtrada), cmap = "gray")
plt.show()
