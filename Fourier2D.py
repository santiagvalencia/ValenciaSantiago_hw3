import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from numpy.fft import fft2, ifft2, fftshift
#Lee la imagen
img = plt.imread("arbol.png")
#Aplica la transformada de Fourier de la imagen y muestra en un plot su parte real en valor absoluto
img_FT = fft2(img)
plt.imshow(np.abs(np.real(fftshift(img_FT))), norm = LogNorm(vmin = 3), cmap = "hot")
plt.colorbar(label = "Transformada 2D de Fourier")
plt.title("Transformada de Fourier de la imagen (shifted)")
plt.xlabel("u")
plt.ylabel("v")
plt.savefig("ValenciaSantiago_FT2D.pdf")
plt.close("all")

#el ruido se ve como unos cuantos puntos con valores altos (>10**3.3)
#por eso el filtro debe ser pasabandas y que elimine las frecuencias correspondientes a esos picos
img_FT_filtro = np.copy(img_FT)#copia la transformada y la recorre
for i in range(img_FT_filtro[:, 0].size):
    for j in range(img_FT_filtro[0].size):
        if np.abs(np.real(img_FT[i, j])) >= 10.0**3.3: #si el valor de la transformada es mayor a 10**3.3 este valor pasa a 0, lo que efectivamente elimina las frecuencias correspondientes a los picos de ruido
            img_FT_filtro[i, j] = 0.01
plt.imshow(np.abs(np.real(fftshift(img_FT_filtro))), norm = LogNorm(vmin = 3), cmap = "hot")
plt.colorbar(label = "Transformada 2D de Fourier")
plt.title("Transformada de Fourier filtrada (shifted)")
plt.xlabel("u")
plt.ylabel("v")
plt.savefig("ValenciaSantiago_FT2D_filtrada.pdf")
plt.close("all")
#invierte la transformada de la imagen y guarda el resultado
img_filtrada = ifft2(img_FT_filtro)
plt.imsave("ValenciaSantiago_Imagen_filtrada.pdf", np.real(img_filtrada), format = "pdf", cmap = "gray")
plt.close("all")
