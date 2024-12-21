import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Configurazioni ----
npz_dir = "npz_files"  # Percorso al file NPZ
parquet_path = "Mirage-VIDEO.parquet"  # Percorso al file PARQUET
label_map_path = "label_map.csv"       # Percorso al file CSV con la mappatura delle classi
output_dir = "class_distributions"    # Directory di output

# Assicurati che la directory di output esista
os.makedirs(output_dir, exist_ok=True)

# Valori globali per la normalizzazione/denormalizzazione
min_global = -1460
max_global = 1460

# ---- Funzione per caricare la mappatura delle classi ----
def load_label_map(label_map_file):
    """
    Carica la mappatura delle classi da un file CSV.

    Args:
        label_map_file (str): Percorso al file CSV.

    Returns:
        dict: Dizionario con indice -> nome classe.
    """
    label_map_df = pd.read_csv(label_map_file)
    return dict(zip(label_map_df['index'], label_map_df['label']))

# ---- Funzione per estrarre valori dalla diagonale del file NPZ ----
def extract_values_from_npz(npz_files, label_map):
    """
    Estrai e denormalizza i valori sulla diagonale delle immagini GASF in una lista di file .npz.

    Args:
        npz_files (list): Lista di percorsi ai file .npz.
        label_map (dict): Mappatura delle classi.

    Returns:
        dict: Dizionario con nomi classi e valori estratti.
    """
    class_values = {}

    for npz_file in npz_files:
        print(f"Elaborazione del file: {npz_file}")
        data = np.load(npz_file)
        images = data['arr_0']
        labels = data['arr_1']

        for image, label in zip(images, labels):
            # Estrai la diagonale
            diagonal = np.diagonal(image)

            normalized_diagonal = ((diagonal / 255.0) * 2) - 1
            # Calcola i valori originali dal GASF
            calculated_values = np.sqrt((normalized_diagonal + 1) / 2)

            # Denormalizza al range originale [-1460, 1460]
            original_values = calculated_values * (max_global - min_global) + min_global

            # Ottieni il nome della classe
            class_name = label_map.get(label, f"unknown_{label}")

            # Aggiungi ai valori per classe
            if class_name not in class_values:
                class_values[class_name] = []
            class_values[class_name].extend(original_values)

    return class_values

# ---- Funzione per calcolare valori dal file PARQUET ----
def compute_values_from_parquet(parquet_file):
    """
    Calcola i valori basati su PL e DIR dal file PARQUET.

    Args:
        parquet_file (str): Percorso al file PARQUET.

    Returns:
        dict: Dizionario con etichette e valori calcolati per classe.
    """
    df = pd.read_parquet(parquet_file)

    class_values = {}
    for _, row in df.iterrows():
        pl = row['PL']
        dir_ = row['DIR']

        # Verifica che PL e DIR non siano vuoti
        if not isinstance(pl, (list, np.ndarray)) or not isinstance(dir_, (list, np.ndarray)):
            continue

        # Crea valori positivi/negativi
        pl_adjusted = [p if d == 1 else -p for p, d in zip(pl, dir_)]

        # Aggiungi ai valori per classe
        label = row['LABEL']
        if label not in class_values:
            class_values[label] = []
        class_values[label].extend(pl_adjusted)

    return class_values

# ---- Funzione per generare un grafico di confronto per ogni classe ----
def generate_comparison_plots(npz_values, parquet_values, output_dir):
    """
    Genera grafici di confronto per ogni classe.

    Args:
        npz_values (dict): Valori estratti dal file NPZ per classe.
        parquet_values (dict): Valori calcolati dal file PARQUET per classe.
        output_dir (str): Directory di output per i grafici.
    """
    classes = set(npz_values.keys()).union(set(parquet_values.keys()))

    for cls in sorted(classes):
        npz_cls_values = npz_values.get(cls, [])
        parquet_cls_values = parquet_values.get(cls, [])

        # Assicurati che i valori siano array 1D
        npz_cls_values = np.array(npz_cls_values).flatten() if npz_cls_values else np.array([])
        parquet_cls_values = np.array(parquet_cls_values).flatten() if parquet_cls_values else np.array([])

        # Controlla se i valori sono vuoti
        if npz_cls_values.size == 0 and parquet_cls_values.size == 0:
            print(f"Classe {cls} senza valori disponibili, saltata.")
            continue

        plt.figure(figsize=(12, 6))

        # Istogramma dei valori del file NPZ
        if npz_cls_values.size > 0:
            plt.subplot(1, 2, 1)
            plt.hist(npz_cls_values, bins=50, alpha=0.6, label=f'NPZ - Classe {cls}', color='blue')
            plt.title(f'Distribuzione NPZ - Classe {cls}')
            plt.xlabel('Valori')
            plt.ylabel('Frequenza')
            plt.legend()
        else:
            print(f"Classe {cls} - Nessun dato NPZ disponibile.")

        # Istogramma dei valori del file PARQUET
        if parquet_cls_values.size > 0:
            plt.subplot(1, 2, 2)
            plt.hist(parquet_cls_values, bins=50, alpha=0.6, label=f'PARQUET - Classe {cls}', color='orange')
            plt.title(f'Distribuzione PARQUET - Classe {cls}')
            plt.xlabel('Valori')
            plt.ylabel('Frequenza')
            plt.legend()
        else:
            print(f"Classe {cls} - Nessun dato PARQUET disponibile.")

        # Salva il grafico
        output_path = os.path.join(output_dir, f'class_{cls}_comparison.png')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        print(f"Grafico salvato per la classe {cls}: {output_path}")


# ---- Esecuzione dello script ----
if __name__ == "__main__":
    print("Caricamento della mappatura delle classi...")
    label_map = load_label_map(label_map_path)

    print("Ricerca dei file NPZ nella directory...")
    npz_files = [os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith('.npz')]

    if not npz_files:
        print("Nessun file NPZ trovato nella directory specificata.")
        exit(1)

    print("Estrazione dei valori dai file NPZ...")
    npz_values = extract_values_from_npz(npz_files, label_map)

    print("Calcolo dei valori dal file PARQUET...")
    parquet_values = compute_values_from_parquet(parquet_path)

    print("Generazione dei grafici di confronto...")
    generate_comparison_plots(npz_values, parquet_values, output_dir)

    print("Completato!")
