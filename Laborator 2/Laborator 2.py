import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice

### EX 1
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
