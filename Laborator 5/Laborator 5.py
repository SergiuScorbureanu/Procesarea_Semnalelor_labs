import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Citirea datelor din fișierul CSV
data = pd.read_csv('Train.csv')

# Extragerea datelor
datetime_values = pd.to_datetime(data['Datetime'], format='%d-%m-%Y %H:%M')
count_values = data['Count']

# Crearea graficului
plt.figure(figsize=(12, 6))
plt.plot(datetime_values, count_values)
plt.title('Graficul din Train.csv', fontsize=16)
plt.xlabel('Datetime', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(True)
plt.show()


### A
# Pentru calcularea frecventei de esantionare a semnalului, ne uitam la diferenta dintre doua esantioane diferite din setul de date.
# Semnalul este esantionat cu frecventa de 1 esantion pe ora, ceea ce inseamna ca frecventa de esantionare este de 1 ora.


### B

#Definim formatul pentru timp
format = '%d-%m-%Y %H:%M'

primul_esantion = '25-08-2012 00:00'
ultimul_esantion = '25-09-2014 23:00'

# Convertim string-urile în obiecte datetime
primul_esantion_dt = datetime.strptime(primul_esantion, format)
ultimul_esantion_dt = datetime.strptime(ultimul_esantion, format)

interval_timp = ultimul_esantion_dt - primul_esantion_dt
print("Intervalul de timp acoperit de eșantioane este de:", interval_timp)

#Intervalul de timp de esantionare este de 762 de zile


