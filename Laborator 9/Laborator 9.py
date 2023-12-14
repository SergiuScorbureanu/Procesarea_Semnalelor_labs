import numpy as np
import matplotlib.pyplot as plt

# Definirea seriei de timp
N = 1000
t = np.arange(N)
trend = 0.005 * t**2 + 0.05 * t + 10
sezon = 5 * np.sin(2 * np.pi * t / 50) + 3 * np.cos(2 * np.pi * t / 30)
zgomot = np.random.normal(0, 2, N)
seria_de_timp = trend + sezon + zgomot

# alpha poate lua valori intre 0 si 1
alpha_manual = 0.02

# Functia de mediere exponențiala
def mediere_exponentiala(alpha, data):
    result = np.zeros_like(data, dtype=float)
    result[0] = data[0]
    for t in range(1, len(data)):
        result[t] = alpha * data[t] + (1 - alpha) * result[t-1]
    return result

# Calcularea si afisarea noii serii rezultate din medierea exponențiala
noua_serie = mediere_exponentiala(alpha_manual, seria_de_timp)

# Afisarea graficelor
plt.figure(figsize=(12, 6))
plt.plot(t, seria_de_timp, label='Seria de timp originala', color='blue')
plt.plot(t, noua_serie, label=f'Seria mediata cu alpha={alpha_manual}', color='red')
plt.title('Compararea seriilor de timp')
plt.xlabel('Timp')
plt.ylabel('Valoare')
plt.legend()
plt.show()
