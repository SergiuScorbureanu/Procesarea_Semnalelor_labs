import numpy as np
import time
import matplotlib.pyplot as plt

time_manual = []
time_numpy = []
Ns = [128, 256, 512, 1024, 2048, 4096, 8192]


for N in Ns:
    x = np.random.random(N)

    start_time = time.time()
    X_manual = np.zeros((N, N), dtype=np.complex128)
    for n in range(N):
        X_manual[n, :] = np.exp(-2j * np.pi * np.arange(N) * n / N) / np.sqrt(N)
    end_time = time.time()
    time_manual.append(end_time - start_time)

    start_time = time.time()
    X_numpy = np.fft.fft(x)
    end_time = time.time()
    time_numpy.append(end_time - start_time)


plt.figure(figsize=(10, 6))
plt.plot(Ns, time_manual, label="Implementarea manuala")
plt.plot(Ns, time_numpy, label="numpy.fft")
plt.yscale('log')
plt.xlabel('Dimensiunea vectorului N')
plt.ylabel('Timp de executie (s)')
plt.legend()
plt.title('Comparatia Transformata Fourier')
plt.show()