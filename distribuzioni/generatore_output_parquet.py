import os
import numpy as np
import pandas as pd

# Percorsi dei file
npz_dir = "npz_files"  # Directory contenente i file NPZ
label_map_path = "label_map.csv"  # File con la mappatura delle etichette
output_parquet_path = "output_data.parquet"

# Parametri globali per la denormalizzazione
min_global = -1460
max_global = 1460

# Funzione per denormalizzare i dati
def recover_original_data(gasf_img, min_global, max_global):
    """
    Recupera i dati originali dalla matrice GASF.

    Args:
        gasf_img (np.ndarray): Immagine GASF (matrice).
        min_global (float): Valore minimo del range originale.
        max_global (float): Valore massimo del range originale.

    Returns:
        np.ndarray: Dati originali denormalizzati.
    """
    # Estrai la diagonale della matrice
    decoded_diagonale = np.diag(gasf_img)

    normalized_diagonal = ((decoded_diagonale / 255.0) * 2) - 1

    # Calcola i valori originali dal GASF
    valori_calcolati = np.sqrt((normalized_diagonal + 1) / 2)

    # Denormalizza al range originale
    valori_originali = valori_calcolati * (max_global - min_global) + min_global

    return valori_originali

# Carica il file label_map.csv
label_map_df = pd.read_csv(label_map_path)

# Crea un dizionario per la mappatura
label_map = dict(zip(label_map_df['index'], label_map_df['label']))

# Ricerca dei file NPZ nella directory
npz_files = [os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith('.npz')]

if not npz_files:
    print("Nessun file NPZ trovato nella directory specificata.")
    exit(1)

# Inizializza liste per i dati
all_PL = []
all_DIR = []
all_LABEL = []

# Elaborazione di ogni file NPZ
for npz_path in npz_files:
    print(f"Elaborazione del file: {npz_path}")

    # Carica il file .npz
    npz_data = np.load(npz_path)

    # Assumendo che `arr_0` sia la matrice GASF e `arr_1` contenga le classi
    arr_0 = npz_data['arr_0']
    arr_1 = npz_data['arr_1']

    # Denormalizza i dati dalla diagonale di ogni immagine GASF
    original_values = [
        recover_original_data(gasf_img.squeeze(), min_global, max_global)
        for gasf_img in arr_0
    ]

    # Convertire in valori assoluti per PL
    PL = np.abs(np.array(original_values))

    # Creare la colonna DIR basandosi sui valori originali
    DIR = (np.array(original_values) >= 0).astype(int)  # 1 se positivo, 0 altrimenti

    # Mappare le classi in base al file di etichette
    LABEL = [label_map.get(class_id, "Unknown") for class_id in arr_1]

    # Aggiungi i dati alle liste globali
    all_PL.extend(PL)
    all_DIR.extend(DIR)
    all_LABEL.extend(LABEL)

# Creare un DataFrame
df = pd.DataFrame({
    "PL": list(all_PL),  # Lista di array per ogni riga
    "DIR": list(all_DIR),  # Lista di array per ogni riga
    "LABEL": all_LABEL
})

# Salva il DataFrame in formato Parquet
df.to_parquet(output_parquet_path, index=False)

print(f"File Parquet generato con successo: {output_parquet_path}")
