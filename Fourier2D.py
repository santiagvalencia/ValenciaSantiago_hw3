import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftn, ifftn

img = plt.imread("arbol.png")
plt.imshow(img, cmap = "gray")
plt.show()
