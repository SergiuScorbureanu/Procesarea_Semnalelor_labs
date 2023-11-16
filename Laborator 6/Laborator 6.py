import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

### Exercitiul 1
# Generare vector aleator x
N = 100
x = np.random.rand(N)

# # Afișare semnal inițial
# plt.subplot(221)
# plt.plot(x)
# plt.title('Semnal Initial')
#
# # Iteratia 1: x ← x * x
# x = x * x
# plt.subplot(222)
# plt.plot(x)
# plt.title('Iteratia 1: x ← x * x')
#
# # Iteratia 2: x ← x * x
# x = x * x
# plt.subplot(223)
# plt.plot(x)
# plt.title('Iteratia 2: x ← x * x')
#
# # Iteratia 3: x ← x * x
# x = x * x
# plt.subplot(224)
# plt.plot(x)
# plt.title('Iteratia 3: x ← x * x')
#
# plt.tight_layout()
# plt.show()

# Observatie: semnalul se modifica in timpul fiecarei operatii. Acesta se inmulteste cu el insusi si astfel creste exponential


### Exercitiul 2

# Metoda 1: Inmultirea directa a polinoamelor
# Generare aleatoare a coeficienților pentru p(x) si q(x)
N = 5 # Gradul maxim al polinomului
# Luam coeficientii dintr-un polinom de la -10 la 10, de grad N
coeficienti_p = np.random.randint(-10, 11, size=N+1)  # +1 pentru a include coeficientul constant
coeficienti_q = np.random.randint(-10, 11, size=N+1)

# Calcularea produsului direct prin convolutie (coeficientii se inmultesc fiecare cu fiecare, iar la final se aduna)
rez_convolutie = np.convolve(coeficienti_p, coeficienti_q)

# Afisarea rezultatului
print("Coeficientii lui p(x):", coeficienti_p)
print("Coeficientii lui q(x):", coeficienti_q)
print("Produsul direct p(x) * q(x):", rez_convolutie)


# Metoda 2: Inmultirea cu FFT
# Zero-padding pentru a evita pierderea informației în FFT
p_padded = np.pad(coeficienti_p, (0, N), mode='constant')
q_padded = np.pad(coeficienti_q, (0, N), mode='constant')

# Aplicarea FFT pe p(x) și q(x)
p_fft = fft(p_padded)
q_fft = fft(q_padded)

# Calcularea produsului în domeniul frecvenței
rez_fft = p_fft * q_fft

# Transformarea inversa FFT pentru a obtine rezultatul în domeniul timpului
r_fft = ifft(rez_fft).real

# Afișarea rezultatului
print("Produsul prin FFT:", r_fft)
