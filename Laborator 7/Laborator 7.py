from scipy import misc, ndimage
from scipy import misc, fftpack
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import ifft, fft, fftshift, fft2, ifft2, ifftshift
from skimage.restoration import estimate_sigma, denoise_nl_means

# ### Exercitiul 1
# # Creez 2 matrice n1, n2 care contin coord punctelor unui grid
# N = 64
# n1, n2 = np.meshgrid(np.arange(N), np.arange(N))
#
# x1 = np.sin(2 * np.pi * n1 + 3 * np.pi * n2)
# x2 = np.sin(4 * np.pi * n1) + np.cos(6 * np.pi * n2)
#
# # Declar o matrice goala cu dimensiunea NxN
# Y = np.zeros((N, N), dtype=complex)
#
# # Setez elementele impuse in cerinta pentru matricea Y
# Y[0, 5] = Y[0, N - 5] = 1
# Y[5, 0] = Y[N - 5, 0] = 1
# Y[5, 5] = Y[N - 5, N - 5] = 1
#
# y = ifft2(ifftshift(Y))
#
# # Aplic tranformatele Fourier pt a obtine spectrul in domeniul frecventei pentru x1 si x2
# Y1 = fftshift(fft2(x1))
# Y2 = fftshift(fft2(x2))
#
# # Creez subragicele
# fig, axs = plt.subplots(2, 3, figsize=(12, 8))
#
# axs[0, 0].imshow(x1, cmap='viridis', extent=(0, N, 0, N))
# axs[0, 0].set_title('Poza x1(n1, n2)')
#
# axs[0, 1].imshow(np.abs(Y1), cmap='viridis', extent=(0, N, 0, N))
# axs[0, 1].set_title('Spectru x1')
#
# axs[0, 2].imshow(x2, cmap='viridis', extent=(0, N, 0, N))
# axs[0, 2].set_title('Poza x2(n1, n2)')
#
# axs[1, 0].imshow(np.abs(Y2), cmap='viridis', extent=(0, N, 0, N))
# axs[1, 0].set_title('Spectru x2')
#
# axs[1, 1].imshow(np.abs(y), cmap='viridis', extent=(0, N, 0, N))
# axs[1, 1].set_title('Poza y')
#
# axs[1, 2].imshow(np.abs(Y), cmap='viridis', extent=(0, N, 0, N))
# axs[1, 2].set_title('Spectru Y')
#
# axs[1, 2].axis('off')
#
# plt.tight_layout()
# plt.show()


### Exercitiul 2

def compresare_imagine(img, snr_threshold):
    gray = img

    f = fft2(gray)
    fshift = fftshift(f)

    rows, cols = gray.shape

    # centru
    crow, ccol = rows // 2, cols // 2

    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - int(snr_threshold):crow + int(snr_threshold), ccol - int(snr_threshold):ccol + int(snr_threshold)] = 1

    fshift = fshift * mask
    f_ishift = ifftshift(fshift)

    img_back = ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back


X = misc.face(gray=True)

snr = 50

imagine_compresata = compresare_imagine(X, snr)

plt.subplot(1, 2, 1)
plt.imshow(X, cmap=plt.cm.gray)

plt.subplot(1, 2, 2)
plt.imshow(imagine_compresata, cmap=plt.cm.gray)
plt.show()

def calcul_snr(signal, noise):
    mean_signal = np.mean(signal)
    mean_noise = np.mean(noise)
    std_noise = np.std(noise)
    return 20 * np.log10(mean_signal / std_noise)

### Exercitiul 3
pixel_noise = 200

X = misc.face(gray=True)

noise = np.random.randint(-pixel_noise, high=pixel_noise + 1, size=X.shape)
X_noisy = X + noise

# Calculul SNR inainte de eliminarea zgomotului
snr_inainte = calcul_snr(X, X_noisy - X)

# Estimarea sigma zgomotului pentru Non-Local Means Denoising
sigma_est = np.mean(estimate_sigma(X_noisy))

# Aplicarea Non-Local Means Denoising
X_denoised_nl_means = denoise_nl_means(X_noisy, h=1.15 * sigma_est, fast_mode=True,
                                        patch_size=5, patch_distance=6)

# Calculul SNR dupa eliminarea zgomotului cu Non-Local Means Denoising
snr_after_nl_means = calcul_snr(X, X_denoised_nl_means - X)

# Afișarea imaginilor
plt.figure(figsize=(20, 7))
plt.subplot(1, 3, 1)
plt.imshow(X, cmap='gray')
plt.title('Imagine originala')
plt.subplot(1, 3, 2)
plt.imshow(X_noisy, cmap='gray')
plt.title('Imagine cu zgomot')
plt.subplot(1, 3, 3)
plt.imshow(X_denoised_nl_means, cmap='gray')
plt.title('NL Means Denoised Image')
plt.show()

print("SNR inainte de comprimarea zgomotului:", snr_inainte)
print("SNR dupa comprimare:", snr_after_nl_means)
