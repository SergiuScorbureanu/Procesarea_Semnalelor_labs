from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt

# Definirea funcțiilor
def func1(n1, n2):
    return np.sin(2 * np.pi * n1 + 3 * np.pi * n2)

def func2(n1, n2):
    return np.sin(4 * np.pi * n1) + np.cos(6 * np.pi * n2)

def func3(m1, m2):
    if (m1 == 0 and m2 == 5) or (m1 == 0 and m2 == -5):
        return 1
    else:
        return 0

def func4(m1, m2):
    if (m1 == 5 and m2 == 0) or (m1 == -5 and m2 == 0):
        return 1
    else:
        return 0

def func5(m1, m2):
    if (m1 == 5 and m2 == 5) or (m1 == -5 and m2 == -5):
        return 1
    else:
        return 0

# Definirea domeniului funcțiilor
n1 = np.arange(-10, 11)
n2 = np.arange(-10, 11)

m1 = np.arange(-10, 11)
m2 = np.arange(-10, 11)

# Calcularea valorilor functiilor pentru fiecare pereche de indici
xn1_n2_1 = func1(n1[:, np.newaxis], n2[np.newaxis, :])
xn1_n2_2 = func2(n1[:, np.newaxis], n2[np.newaxis, :])
Ym1_m2_3 = np.fromfunction(np.vectorize(func3), (21, 21), dtype=int)
Ym1_m2_4 = np.fromfunction(np.vectorize(func4), (21, 21), dtype=int)
Ym1_m2_5 = np.fromfunction(np.vectorize(func5), (21, 21), dtype=int)

# Afisarea imaginilor
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(xn1_n2_1, cmap='viridis', extent=(-10, 10, -10, 10))
plt.title('Img: sin(2πn1 + 3πn2)')

plt.subplot(2, 3, 2)
plt.imshow(xn1_n2_2, cmap='viridis', extent=(-10, 10, -10, 10))
plt.title('Img: sin(4πn1) + cos(6πn2)')

plt.subplot(2, 3, 3)
plt.imshow(Ym1_m2_3, cmap='viridis', extent=(-10, 10, -10, 10))
plt.title('Y0,5 = Y0,-5 = 1, altfel 0')

plt.subplot(2, 3, 4)
plt.imshow(Ym1_m2_4, cmap='viridis', extent=(-10, 10, -10, 10))
plt.title('Y5,0 = Y-5,0 = 1, altfel 0')

plt.subplot(2, 3, 5)
plt.imshow(Ym1_m2_5, cmap='viridis', extent=(-10, 10, -10, 10))
plt.title('Y5,5 = Y-5,-5 = 1, altfel 0')

# Calcularea si afișarea spectrului
plt.subplot(2, 3, 6)
plt.specgram(xn1_n2_1, cmap='viridis', NFFT=256, Fs=1, noverlap=128)
plt.title('Spectrul pentru sin(2πn1 + 3πn2)')

plt.tight_layout()
plt.show()



