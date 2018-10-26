import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from numpy.fft import fft2

img = plt.imread("arbol.png")

img_FT = np.abs(np.real(fft2(img)))
plt.imshow(img_FT, norm = LogNorm(vmin = 1), cmap = "jet")
plt.colorbar()
plt.show()



#plt.imshow(img, cmap = "gray")
#plt.show()
