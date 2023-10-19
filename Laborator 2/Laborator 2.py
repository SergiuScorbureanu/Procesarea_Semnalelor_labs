import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice

### Exercitiul 1

amplitudine = 1.5
frecventa = 5
faza = np.pi/4

timp = np.linspace(0, 1, 500)
semnal_sinus = amplitudine * np.sin(2 * np.pi * frecventa * timp + faza)
semnal_cosinus = amplitudine * np.cos(2 * np.pi * frecventa * timp + faza)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(timp, semnal_sinus)
plt.title('Semnal sinusoidal')

plt.subplot(2, 1, 2)
plt.plot(timp, semnal_cosinus)
plt.title('Semnal cosinusoidal')

plt.show()


### Exercitiul 2

amplitudine = 1
frecventa = 5
faze = [0, np.pi/4, np.pi/2, 3*np.pi/4]

timp = np.linspace(0, 1, 500)

semnale = []
for faza in faze:
    semnal = amplitudine * np.sin(2 * np.pi * frecventa * timp + faza)
    semnale.append(semnal)

plt.figure(figsize=(10, 6))
for i, semnal in enumerate(semnale):
    plt.plot(timp, semnal)

n = np.random.normal(0, 1, 500)
valori_snr = [0.1, 1, 10, 100]
for snr in valori_snr:
    zgomot = np.random.normal(0, 1 / snr, 500)
    semnal_zgomot = semnal + zgomot
    plt.plot(timp, semnal_zgomot)

plt.title('Semnale sinusoidale')
plt.show()