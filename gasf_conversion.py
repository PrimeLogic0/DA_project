import pandas as pd
import numpy as np
import os
from pyts.image import GramianAngularField
import matplotlib.pyplot as plt

# Path al file e alle directory
data_file = "Mirage-VIDEO.parquet"  # Sostituisci con il percorso corretto
save_dir = "Output_images_10"  # Directory principale per salvare le immagini
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Parametri
num_packets = 10  # Numero massimo di pacchetti per campione
output_size = 10  # Dimensione dell'immagine finale

# Caricamento del dataset
df = pd.read_parquet(data_file)

all_pl_adjusted = []
for _, row in df.iterrows():
    pl = row['PL']  # Lista delle dimensioni dei pacchetti
    dir_ = row['DIR']  # Lista delle direzioni dei pacchetti (1 o 0)

    # Creazione dei valori positivi/negativi per PL
    pl_adjusted = [p if d == 1 else -p for p, d in zip(pl, dir_)]
    all_pl_adjusted.extend(pl_adjusted)

# Calcolo del minimo e massimo globali
min_global = min(all_pl_adjusted)
max_global = max(all_pl_adjusted)

print(f"Global Min: {min_global}, Global Max: {max_global}")

# Iterazione sui campioni
for idx, row in df.iterrows():
    pl = row['PL']  # Lista delle dimensioni dei pacchetti
    dir_ = row['DIR']  # Lista delle direzioni dei pacchetti (1 o 0)

    # Creazione dei valori positivi/negativi per PL
    pl_adjusted = [p if d == 1 else -p for p, d in zip(pl, dir_)]

    if not pl_adjusted:  # Verifica se la lista è vuota
        continue
    # Riempimento o troncamento a 10 valori
    pl_adjusted = (pl_adjusted + [0] * num_packets)[:num_packets]

    # Normalizzazione Min-Max globale tra -1 e 1
    if max_global - min_global != 0:  # Evita divisioni per zero
        pl_normalized = [(p - min_global) / (max_global - min_global) for p in pl_adjusted]
    else:
        pl_normalized = [0] * num_packets


    if idx == 0:
        print(pl_adjusted)


    # Conversione a numpy array
    X = np.array([pl_adjusted])

# Conversione in immagine GASF
    gasf = GramianAngularField(sample_range=(0, 1), method='summation')
    X_gasf = gasf.transform(X)

    # Gamma correction
    gasf_img = X_gasf[0]

    # Ottieni l'etichetta e crea la directory corrispondente
    label = row['LABEL']  # Nome della classe del campione
    label_dir = os.path.join(save_dir, str(label))  # Directory specifica per l'etichetta
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)  # Crea la directory se non esiste

    # Salvataggio dell'immagine
    filename = f"sample_{idx+2}.png"
    filepath = os.path.join(label_dir, filename)
    plt.imsave(filepath, gasf_img, cmap='viridis')
