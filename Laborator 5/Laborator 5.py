import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Citirea datelor din fisierul CSV
data = pd.read_csv('Train.csv')

# Extragerea datelor
datetime_values = pd.to_datetime(data['Datetime'], format='%d-%m-%Y %H:%M')
count_values = data['Count']

# Crearea graficului
plt.figure(figsize=(12, 6))
plt.plot(datetime_values, count_values)
plt.title('Graficul din Train.csv', fontsize=16)
plt.xlabel('Timp', fontsize=12)
plt.ylabel('Nr masini', fontsize=12)
plt.grid(True)
plt.show()


### A
# Pentru calcularea frecventei de esantionare a semnalului, ne uitam la diferenta dintre doua esantioane diferite din setul de date.
# Semnalul este esantionat cu frecventa de 1 esantion pe ora, ceea ce inseamna ca frecventa de esantionare este de 1/3600,
# deoarece frecventa se calculeaza in secunde.


### B

#Definim formatul pentru timp
format = '%d-%m-%Y %H:%M'

primul_esantion = '25-08-2012 00:00'
ultimul_esantion = '25-09-2014 23:00'

# Convertim string-urile în obiecte datetime
primul_esantion_dt = datetime.strptime(primul_esantion, format)
ultimul_esantion_dt = datetime.strptime(ultimul_esantion, format)

interval_timp = ultimul_esantion_dt - primul_esantion_dt
print("B) Intervalul de timp acoperit de eșantioane este de:", interval_timp)

#Intervalul de timp de esantionare este de 761 de zile


### C

# Citirea datelor din fisierul CSV
data = pd.read_csv('Train.csv')

# Convertirea coloanei 'Datetime' în format de data/timp
data['Datetime'] = pd.to_datetime(data['Datetime'], format='%d-%m-%Y %H:%M')

# Sortarea valorilor în funcție de 'Datetime'
data.sort_values('Datetime', inplace=True)

# Calculul timpului total si a frecventei maxime
timp_total = (data['Datetime'].iloc[-1] - data['Datetime'].iloc[0]).total_seconds() / 3600
frecventa_max = 1 / (timp_total / (len(data) - 1))

print(f"C) Frecventa maxima prezenta in semnal: {frecventa_max:.5f} Hz")


### D

frecv_esantioane = len(data) / timp_total

# Transformata Fourier
tr_Fourier = np.fft.fft(data['Count'])

# Frecvente la fiecare punct din transformata
frecvente = np.fft.fftfreq(len(data), d=1/frecv_esantioane)

# Generare grafic
plt.plot(frecvente, np.abs(tr_Fourier))
plt.title('Modulul tranformatei Fourier')
plt.xlabel('Frecventa')
plt.ylabel('Amplitudine')
plt.show()
