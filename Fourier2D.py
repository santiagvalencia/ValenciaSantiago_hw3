import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from numpy.fft import fft2, ifft2

img = plt.imread("arbol.png")

img_FT = np.abs(np.real(fft2(img)))
plt.imshow(img_FT, norm = LogNorm(vmin = 1), cmap = "hot")
plt.colorbar(label = "Transformada 2D de Fourier")
plt.show()
plt.close("all")

#el ruido se ve como unos cuantos puntos con valores altos (>10**2.5)
#por eso el filtro debe ser pasabajas
for i in range(img_FT[:, 0].size):
    for j in range(img_FT[0].size):
        if img_FT[i, j] >= 10.0**2.5:
            img_FT[i, j] = 0

img_filtrada = np.abs(np.real(ifft2(img_FT)))
plt.imshow(img_filtrada, cmap = "gray")
plt.show()
plt.close("all")


#plt.imshow(img, cmap = "gray")
#plt.show()
