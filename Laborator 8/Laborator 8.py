import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg

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

# Generam componente de zgomot alb gaussian
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

### B

# Calculul autocorelației folosind np.correlate
autocorelatie = np.correlate(seria_de_timp, seria_de_timp, mode='full')

# Normalizarea autocorelației la intervalul [-1, 1]
autocorelatie /= np.max(np.abs(autocorelatie))

# Desenarea vectorului de autocorelație
plt.figure(figsize=(12, 4))
plt.plot(np.arange(-N+1, N), autocorelatie, label='Autocorelatie')
plt.title('Vector autocorelatie')
plt.xlabel('Lag')
plt.ylabel('Autocorelatie')
plt.legend()
plt.grid(True)
plt.show()

### C

# Dimensiunea modelului AR
p = 7

# Crearea si antrenarea modelului AR
model_ar = AutoReg(seria_de_timp, lags=p)
model_ar_fit = model_ar.fit()

# Obtinerea predictiilor
predictions = model_ar_fit.predict(start=p, end=N - 1, dynamic=False)

# Desenam seria de timp originala si predicțiile
plt.figure(figsize=(14, 6))
plt.plot(seria_de_timp, label='Serie de Timp Originala', color='blue')
plt.plot(range(p, N), predictions, label='Predicții AR', color='red', linestyle='dashed')
plt.title('Serie de Timp si Predictiile Modelului AR')
plt.legend()
plt.show()
