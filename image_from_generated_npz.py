import os
import numpy as np
import pandas as pd
from PIL import Image

max_global=1460
min_global=-1460

def save_images_from_npz_with_labels(npz_path, output_dir, label_map_path):
    # Carica il file npz
    data = np.load(npz_path)

    # Estrai immagini e indici delle etichette
    images = data['arr_0']
    labels = data['arr_1']

    print("Etichette:", labels)

    if images.shape[0] != labels.shape[0]:
        raise ValueError("Il numero di immagini non corrisponde al numero di etichette!")

    # Carica il file CSV con la mappatura delle etichette
    label_map = pd.read_csv(label_map_path)
    label_dict = dict(zip(label_map['index'], label_map['label']))  # Mappa indice -> nome etichetta

    # Assicurati che la directory di output esista
    os.makedirs(output_dir, exist_ok=True)

    # Per ogni immagine, salva nella sottodirectory corretta
    for idx, (image, label_idx) in enumerate(zip(images, labels)):
        # Ottieni il nome dell'etichetta corrispondente
        label_name = label_dict.get(label_idx, f"unknown_{label_idx}")  # Usa un nome predefinito se l'indice non esiste

        # Crea la sottodirectory per la classe
        class_dir = os.path.join(output_dir, label_name)
        os.makedirs(class_dir, exist_ok=True)

        # Rimuovi dimensione extra per immagini in scala di grigi
        if image.ndim == 3 and image.shape[-1] == 1:
            image = np.squeeze(image, axis=-1)

        diagonale = np.diagonal(image)

        diagonale_normalizzata = diagonale / 255.0

        valori_calcolati = np.sqrt((diagonale_normalizzata + 1) / 2)

        # Denormalizza al range originale [-1460, 1460]
        valori_originali = valori_calcolati * (max_global - min_global) + min_global

        # Stampa i valori della diagonale
        print(f"Immagine {idx}: Diagonale = {valori_originali}")

        # Converti l'immagine in formato PIL
        img = Image.fromarray(image.astype(np.uint8))

        # Salva l'immagine in formato PNG
        img_path = os.path.join(class_dir, f"image_{idx}.png")
        img.save(img_path, format='PNG')

        print(f"Immagine salvata: {img_path}")

    print(f"Tutte le immagini sono state salvate in {output_dir}")

# Esempio di utilizzo
#cambiare il percorso del file
npz_path = "128/iterate/df/synth_models/samples_10x10x10x1.npz"  # Percorso al file .npz
label_map_path = "scripts/128/iterate/df/synth_models/label_map.csv"  # Percorso al file CSV
output_dir = "model_npz_image"      # Directory di destinazione
save_images_from_npz_with_labels(npz_path, output_dir, label_map_path)
