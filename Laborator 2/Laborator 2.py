import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd

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


### Exercitiul 4
frecv_esant = 44100

timp = np.linspace(0, 1, frecv_esant)

amp = 2
frecv = 5
phi = np.pi/4
semnal_sinus = amp * np.sin(2 * np.pi * frecv * timp + phi)

semnal_sawtooth = np.linspace(-amp, amp, frecv_esant)

suma_semnale = semnal_sinus + semnal_sawtooth
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(timp, semnal_sinus)
plt.title('Semnal Sinusoidal')
plt.xlabel('Timp')
plt.ylabel('Amplitudine')

plt.subplot(3, 1, 2)
plt.plot(timp, semnal_sawtooth)
plt.title('Semnal Sawtooth')
plt.xlabel('Timp')
plt.ylabel('Amplitudine')

plt.subplot(3, 1, 3)
plt.plot(timp, suma_semnale)
plt.title('Suma semnalelor')
plt.xlabel('Timp')
plt.ylabel('Amplitudine')

plt.tight_layout()
plt.show()

wavfile.write('semnal_combinat.wav', frecv_esant, suma_semnale)

sd.play(suma_semnale, frecv_esant)
sd.wait()
