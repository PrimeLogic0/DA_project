import numpy as np

# Valori globali min e max utilizzati per la normalizzazione originale
min_global = -1460  # Sostituisci con il valore reale
max_global = 1460   # Sostituisci con il valore reale

def recover_original_data(npz_file, min_global, max_global):
    """
    Recupera i dati originali da un'immagine GASF salvata come .npz.

    Args:
        npz_file (str): Percorso al file .npz.
        min_global (float): Valore minimo del range originale.
        max_global (float): Valore massimo del range originale.

    Returns:
        np.ndarray: Dati originali denormalizzati.
    """
    # Carica l'immagine GASF dal file .npz
    gasf_img = np.load(npz_file)['gasf_img']

    # Estrai la diagonale della matrice
    decoded_diagonale = np.diag(gasf_img)

    print("Diagonale:", decoded_diagonale)

    # Calcola i valori originali dal GASF
    valori_calcolati = np.sqrt((decoded_diagonale + 1) / 2)

    # Denormalizza al range originale [-1460, 1460]
    valori_originali = valori_calcolati * (max_global - min_global) + min_global

    return valori_originali

# Esempio di utilizzo
npz_file = "./test_/sample_2.npz"  # Sostituisci con il percorso corretto
original_data = recover_original_data(npz_file, min_global, max_global)

print("Dati originali recuperati:", original_data)
