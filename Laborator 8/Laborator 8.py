import numpy as np
import matplotlib.pyplot as plt

### Exercitiul 1
### A

# Setam dimensiunea seriei de timp
N = 1000

# Generam timpul de la 0 la N-1
t = np.arange(N)

# Definim componenta de trend cu o ecuație de grad 2
trend = 0.0005 * t**2 + 0.005 * t + 10

# Definim componenta sezonala prin suma a 2 frecvente
sezon = 5 * np.sin(2 * np.pi * t / 50) + 3 * np.cos(2 * np.pi * t / 30)

# Generam componente de zgomot alb gaussian pentru variabilitate mica
zgomot = np.random.normal(0, 2, N)

# Apelam toate componentele pentru a obține seria de timp finala
seria_de_timp = trend + sezon + zgomot

plt.figure(figsize=(12, 6))

plt.subplot(4, 1, 1)
plt.plot(t, trend, label='Trend')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(t, sezon, label='Sezon')
plt.legend()


plt.subplot(4, 1, 3)
plt.plot(t, zgomot, label='Zgomot alb')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(t, seria_de_timp, label='Seria de timp', color='red')
plt.legend()

plt.tight_layout()
plt.show()

