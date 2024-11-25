import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Specifica il percorso della cartella contenente le immagini
cartella_immagini = "/Users/matteospavone/Desktop/Netflix copia"

# Lista per contenere tutti i pixel di tutte le immagini
tutti_i_pixel = []

# Leggi tutte le immagini nella cartella
for file in os.listdir(cartella_immagini):
    if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # Filtra i formati di immagine
        percorso_file = os.path.join(cartella_immagini, file)
        # Apri l'immagine e converti in scala di grigi
        immagine = Image.open(percorso_file).convert('L')  # 'L' = scala di grigi
        # Aggiungi i pixel dell'immagine alla lista
        tutti_i_pixel.extend(np.array(immagine).flatten())

# Calcola la distribuzione media
if len(tutti_i_pixel) > 0:
    # Creare un istogramma della distribuzione dei pixel
    plt.figure(figsize=(10, 6))
    plt.hist(tutti_i_pixel, bins=256, range=(0, 255), color='blue', alpha=0.7, edgecolor='black')
    plt.title("Distribuzione media dei pixel nelle immagini")
    plt.xlabel("Valore dei pixel (0-255)")
    plt.ylabel("Frequenza")
    plt.grid(axis='y', alpha=0.75)
    plt.show()
else:
    print("Nessuna immagine trovata nella cartella!")
