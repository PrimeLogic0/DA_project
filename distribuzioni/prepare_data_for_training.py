import os
import numpy as np
import pandas as pd
from pathlib import Path

# Configurazione
cartella_input = "npz_files"
cartella_output = "npz_image_folder"  # Cambia con il percorso della tua cartella output
file_csv = "label_map.csv"  # Cambia con il percorso del tuo file CSV

# Carica il file CSV con la mappatura tra indice e nome della classe
mapping = pd.read_csv(file_csv)
mapping_dict = dict(zip(mapping['index'], mapping['label']))

# Crea la cartella output se non esiste
Path(cartella_output).mkdir(parents=True, exist_ok=True)

# Funzione per salvare una singola immagine in un file .npz
def salva_immagine(immagine, percorso_file):
    np.savez_compressed(percorso_file, immagine=immagine)

# Itera su tutti i file .npz nella cartella input
for file_npz in Path(cartella_input).glob("*.npz"):
    print(f"Processando {file_npz}...")

    # Carica il file .npz
    with np.load(file_npz) as data:
        immagini = data["arr_0"]  # Matrice 2000*10*10
        classi = data["arr_1"]  # Lista delle classi corrispondenti

    # Itera su ogni immagine e classe
    for idx, (immagine, classe) in enumerate(zip(immagini, classi)):
        # Ottieni il nome della classe dalla mappatura
        nome_classe = mapping_dict.get(classe, f"unknown_{classe}")

        # Crea la cartella della classe se non esiste
        cartella_classe = Path(cartella_output) / nome_classe
        cartella_classe.mkdir(parents=True, exist_ok=True)

        # Crea il nome del file per l'immagine
        nome_file = f"{file_npz.stem}_{idx}.npz"
        percorso_file = cartella_classe / nome_file

        # Salva l'immagine in un file .npz
        salva_immagine(immagine, percorso_file)

print("Elaborazione completata!")

