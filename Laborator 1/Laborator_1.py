import numpy as np
import matplotlib.pyplot as plt

### 1A
t = np.arange(0, 0.03, 0.0005)


def fx(t):
    return np.cos(520 * np.pi * t + np.pi / 3)


def fy(t):
    return np.cos(280 * np.pi * t - np.pi / 3)


def fz(t):
    return np.cos(120 * np.pi * t + np.pi / 3)


print(t)


### 1B
figura, axs = plt.subplots(3)
figura.suptitle('1 (b)')

axs[0].plot(t, fx(t))
axs[1].plot(t, fy(t))
axs[2].plot(t, fz(t))

for ax in axs:
    ax.set_xlim([0, 0.03])

plt.show()


### 1C
t2 = np.arange(0, 0.03, 0.03/6)

figura2, axs2 = plt.subplots(3)
figura2.suptitle('1 (c)')

axs2[0].plot(t2, fx(t2))
axs2[1].plot(t2, fy(t2))
axs2[2].plot(t2, fz(t2))

for ax in axs2:
    ax.set_xlim([0, 0.03])

plt.show()
