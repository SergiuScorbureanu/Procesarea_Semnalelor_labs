import numpy as np
import matplotlib.pyplot as plt

N = 8
F = np.zeros((N, N), dtype=np.complex128)
for n in range(N):
    for k in range(N):
        F[n, k] = np.exp(-2j * np.pi * n * k / N) / np.sqrt(N)

# Dupa ce am creat matricea Fourier, verificam daca este ortogonala, complexa sau unitara
matrice_ortogonala = np.allclose(np.eye(N), F @ F.conj().T)
matrice_complexa = np.allclose(F, F.conj().T)

if matrice_ortogonala and matrice_complexa:
    print("Matricea Fourier este unitara")
else:
    if matrice_ortogonala:
        print("Matricea Fourier este ortogonala")
    elif matrice_complexa:
        print("Matricea Fourier este complexa")
    else:
        print("Matricea Fourier nu este ortogonala, complexa sau unitara")


plt.figure(figsize=(12, 8))

for x in range(N):
    plt.subplot(N, 2, 2 * x + 1)
    plt.plot(np.real(F[:, x]))
    plt.title(f"Partea reala {x + 1}")

    plt.subplot(N, 2, 2 * x + 2)
    plt.plot(np.imag(F[:, x]))
    plt.title(f"Partea imaginara {x + 1}")

plt.tight_layout()
plt.show()