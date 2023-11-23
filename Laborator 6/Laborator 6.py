import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import pandas as pd

### Exercitiul 1
# Generare vector aleator x
N = 100
x = np.random.rand(N)

# Afișare semnal inițial
plt.subplot(221)
plt.plot(x)
plt.title('Semnal Initial')

# Iteratia 1
x = x * x
plt.subplot(222)
plt.plot(x)
plt.title('Iteratia 1')

# Iteratia 2
x = x * x
plt.subplot(223)
plt.plot(x)
plt.title('Iteratia 2')

# Iteratia 3
x = x * x
plt.subplot(224)
plt.plot(x)
plt.title('Iteratia 3')

plt.tight_layout()
plt.show()

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
p_pad = np.pad(coeficienti_p, (0, N), mode='constant')
q_pad = np.pad(coeficienti_q, (0, N), mode='constant')

# Aplicarea FFT pe p(x) și q(x)
p_fft = fft(p_pad)
q_fft = fft(q_pad)

# Calcularea produsului în domeniul frecvenței
rez_fft = p_fft * q_fft

# Transformarea inversa FFT pentru a obtine rezultatul în domeniul timpului
r_fft = ifft(rez_fft).real

# Afișarea rezultatului
print("Produsul prin FFT:", r_fft)


### Exercitiul 4
# (a) si (b)

data = pd.read_csv('TrainL6.csv')

# Convertirea coloanei Datetime la tipul de date datetime
data['Datetime'] = pd.to_datetime(data['Datetime'], format='%d-%m-%Y %H:%M')

# Selectarea unei portiuni de 3 zile
data_start = '25-08-2012 00:00'
data_sfarsit = '27-08-2012 23:00'
date_selectate = data[(data['Datetime'] >= data_start) & (data['Datetime'] <= data_sfarsit)]

# Definirea semnalului x (nr de vehicule)
x = date_selectate['Count'].values

# Definirea dimensiunilor diferite ale ferestrei
dimensiuni_fereastra = [5, 9, 13, 17]

# Filtrarea cu media alunecatoare pentru diferite dimensiuni ale ferestrei
for dim in dimensiuni_fereastra:
    semnal_netezit = np.convolve(x, np.ones(dim) / dim, 'valid')

    # Afisarea rezultatelor
    plt.figure()
    plt.plot(date_selectate['Datetime'][:-dim+1], semnal_netezit)
    plt.title(f'Semnal netezit cu fereastra {dim}')
    plt.xlabel('Datetime')
    plt.ylabel('Semnal')
    plt.show()

