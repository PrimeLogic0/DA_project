import os
import numpy as np
import matplotlib.pyplot as plt

# Parametri di normalizzazione
min_global = -1460
max_global = 1460

# Percorso dell'immagine
gasf_img_path = "/home/rodolfo/Desktop/unina/DA/NetDiffus/test_/sample_2_2.png"

# Verifica se il file esiste
if not os.path.exists(gasf_img_path):
    raise FileNotFoundError(f"File non trovato: {gasf_img_path}")

# Carica l'immagine
gasf_img = plt.imread(gasf_img_path)

# Se ha 4 canali (RGBA), usa solo il primo canale
if gasf_img.ndim == 3 and gasf_img.shape[2] == 4:
    gasf_img = gasf_img[:, :, 0]


# Estrai la diagonale
diagonale = np.diagonal(gasf_img)

valori_calcolati = np.cos(np.arccos(diagonale) / 2)

# Denormalizza i valori al range originale [-1460, 1460]
valori_originali = [v * (max_global - min_global) + min_global for v in valori_calcolati]

# Risultati finali
print("Valori originali:", valori_originali)
