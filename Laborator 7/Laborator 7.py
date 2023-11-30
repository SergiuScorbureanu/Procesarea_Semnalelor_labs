from scipy import misc, ndimage
from scipy import misc, fftpack
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import ifft, fft, fftshift, fft2, ifft2, ifftshift

### Exercitiul 1
# Creez 2 matrice n1, n2 care contin coord punctelor unui grid
N = 64
n1, n2 = np.meshgrid(np.arange(N), np.arange(N))

x1 = np.sin(2 * np.pi * n1 + 3 * np.pi * n2)
x2 = np.sin(4 * np.pi * n1) + np.cos(6 * np.pi * n2)

# Declar o matrice goala cu dimensiunea NxN
Y = np.zeros((N, N), dtype=complex)

# Setez elementele impuse in cerinta pentru matricea Y
Y[0, 5] = Y[0, N - 5] = 1
Y[5, 0] = Y[N - 5, 0] = 1
Y[5, 5] = Y[N - 5, N - 5] = 1

y = ifft2(ifftshift(Y))

# Aplic tranformatele Fourier pt a obtine spectrul in domeniul frecventei pentru x1 si x2
Y1 = fftshift(fft2(x1))
Y2 = fftshift(fft2(x2))

# Creez subragicele
fig, axs = plt.subplots(2, 3, figsize=(12, 8))

axs[0, 0].imshow(x1, cmap='viridis', extent=(0, N, 0, N))
axs[0, 0].set_title('Poza x1(n1, n2)')

axs[0, 1].imshow(np.abs(Y1), cmap='viridis', extent=(0, N, 0, N))
axs[0, 1].set_title('Spectru x1')

axs[0, 2].imshow(x2, cmap='viridis', extent=(0, N, 0, N))
axs[0, 2].set_title('Poza x2(n1, n2)')

axs[1, 0].imshow(np.abs(Y2), cmap='viridis', extent=(0, N, 0, N))
axs[1, 0].set_title('Spectru x2')

axs[1, 1].imshow(np.abs(y), cmap='viridis', extent=(0, N, 0, N))
axs[1, 1].set_title('Poza y')

axs[1, 2].imshow(np.abs(Y), cmap='viridis', extent=(0, N, 0, N))
axs[1, 2].set_title('Spectru Y')

axs[1, 2].axis('off')

plt.tight_layout()
plt.show()


### Exercitiul 2

# X = misc.face(gray=True)
# plt.imshow(X, cmap=plt.cm.gray)
# plt.show()

def compress_image(image, snr_threshold):
    # Aplicăm transformata Fourier bidimensională
    image_fft = fftpack.fft2(image)

    # Calculăm spectrul imaginii (magnitudinea coeficienților Fourier)
    magnitude_spectrum = np.abs(image_fft)

    # Calculăm SNR pentru fiecare frecvență
    snr = magnitude_spectrum / np.std(magnitude_spectrum)

    # Atenuăm coeficienții de frecvențe înalte bazat pe pragul SNR
    compressed_spectrum = np.where(snr > snr_threshold, magnitude_spectrum, 0)

    # Reconstruim imaginea prin aplicarea inversei transformatei Fourier bidimensionale
    compressed_image = fftpack.ifft2(compressed_spectrum).real

    return compressed_image

# Încărcați imaginea
X = misc.face(gray=True)

# Specificați pragul SNR pentru atenuarea frecvențelor înalte
snr_threshold = 20.0

# Comprimăm imaginea
compressed_X = compress_image(X, snr_threshold)

# Afișăm imaginea originală și comprimată în două subgrafice
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Subgrafic pentru imaginea originală
axs[0].imshow(X, cmap=plt.cm.gray)
axs[0].set_title('Imaginea originală')

# Subgrafic pentru imaginea comprimată
axs[1].imshow(compressed_X, cmap=plt.cm.gray)
axs[1].set_title(f'Imaginea comprimată (SNR Threshold = {snr_threshold})')

plt.show()


